from __future__ import annotations

import argparse, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .absorb import AbsorptionError, TiltedNull, _feature_matrix as absorption_feature_matrix
from .core import Window
from .dsl import parse_program
from .search import evolutionary_search, _fit_eval
from .zeros import GUESpacingNull, PoissonSpacingNull, load_zeros, unfold


@dataclass
class ZeroSliceDataset:
    zeros_all: np.ndarray
    lo: int = 0
    hi: int | None = None
    window_len: int = 64
    name: str = "zeta_zero_spacings"
    domain: str = "zeros"

    @property
    def zeros(self):
        return self.zeros_all[self.lo : self.hi]

    @property
    def spacings(self):
        return unfold(self.zeros)

    def range_dataset(self, n_min: int, n_max: int):
        lo = self.lo + max(0, int(n_min))
        hi = self.lo + min(int(n_max), len(self.zeros))
        return ZeroSliceDataset(self.zeros_all, lo, hi, self.window_len, self.name, self.domain)

    def sample(self, n_windows: int, rng: np.random.Generator):
        s = self.spacings
        max_start = max(1, s.size - self.window_len)
        out = []
        for _ in range(int(n_windows)):
            idx = int(rng.integers(0, max_start))
            out.append(Window(s[idx : idx + self.window_len].astype(np.float32), self.lo + idx, {"domain": self.domain, "gamma_start": float(self.zeros[idx])}))
        return out

    def scale_of(self, w: Window): return int(np.log10(max(float(w.meta.get("gamma_start", 1.0)), 1.0)))


class MeanSpacing:
    name = "spacing_mean"
    def __call__(self, w): return np.asarray([float(np.mean(w.values))], dtype=np.float32)
    def complexity(self): return 1.0
    def describe(self): return self.name


class VarSpacing:
    name = "spacing_variance"
    def __call__(self, w): return np.asarray([float(np.var(w.values))], dtype=np.float32)
    def complexity(self): return 1.0
    def describe(self): return self.name


def empirical_stats():
    return [MeanSpacing(), VarSpacing(), parse_program("fhist(w,8)"), parse_program("fhist(ratios(w),8)")]


def _constraint_labels(stats):
    labels = []
    for stat in stats:
        width = int(np.asarray(stat(Window(np.ones(64, dtype=np.float32), 0, {"domain": "zeros"}))).reshape(-1).size)
        labels.extend([stat.describe() if width == 1 else f"{stat.describe()}[{i}]" for i in range(width)])
    return labels


def support_precheck(real, base, stats, n_windows: int, rng: np.random.Generator) -> dict[str, Any]:
    real_windows = real.sample(n_windows, rng)
    candidates = base.sample_like(real, max(4000, n_windows), rng)
    target = absorption_feature_matrix(stats, real_windows).mean(axis=0)
    phi = absorption_feature_matrix(stats, candidates)
    lo, hi = phi.min(axis=0), phi.max(axis=0)
    failures = []
    for label, t, a, b in zip(_constraint_labels(stats), target, lo, hi):
        if float(t) < float(a) or float(t) > float(b):
            gap = float(a - t) if float(t) < float(a) else float(t - b)
            failures.append({"constraint": label, "target": float(t), "support_min": float(a), "support_max": float(b), "outside_by": gap})
    return {"n_real": len(real_windows), "n_candidates": len(candidates), "failures": failures}


def sequential_tilt(real, matrix_size: int, cache_windows: int, probe_n: int, seed: int) -> tuple[Any, list[dict[str, Any]]]:
    base = GUESpacingNull(seed=seed, matrix_size=matrix_size)
    absorbed: list[Any] = []
    steps: list[dict[str, Any]] = []
    schedule = [[MeanSpacing(), VarSpacing()], [parse_program("fhist(w,8)")], [parse_program("fhist(ratios(w),8)")]]
    oversample = max(10, int(np.ceil(float(cache_windows) / max(float(probe_n), 1.0))))
    rng = np.random.default_rng(880_000 + int(seed))
    current = base
    for group in schedule:
        names = [s.describe() for s in group]
        check = support_precheck(real, base, group, probe_n, rng)
        step: dict[str, Any] = {"stats": names, "support_precheck": check, "absorbed": False}
        if check["failures"]:
            step["status"] = "support_failed"
            steps.append(step)
            continue
        candidate = TiltedNull(base=base, stats=[*absorbed, *group], oversample_factor=oversample, match_second_moments=False)
        try:
            candidate.sample_like(real, probe_n, rng)
            step["ess"] = candidate.ess
            step["status"] = "absorbed"
            step["absorbed"] = True
            absorbed.extend(group)
            current = candidate
        except AbsorptionError as exc:
            step["status"] = "absorption_failed"
            step["error"] = str(exc)
        steps.append(step)
    return current, steps


def split_ranges(n_zeros: int):
    n_sp = n_zeros - 1
    a, b = n_sp // 3, (2 * n_sp) // 3
    return (0, a), (a, b), (b, n_sp)


def rung(name: str, real, null, budget: int, seeds: list[int], far=None, bonferroni_controls: int | None = None) -> dict[str, Any]:
    train, val, ood = split_ranges(len(real.zeros_all))
    out: dict[str, Any] = {"null": name, "seeds": {}, "shuffle_controls": {}, "total_programs_evaluated": 0, "bonferroni_controls": int(bonferroni_controls or max(1, len(seeds)))}
    for seed in seeds:
        try:
            res = evolutionary_search(real, null, budget, seed, train_range=train, val_range=val, ood_range=ood)
        except Exception as exc:
            out["seeds"][str(seed)] = {"error": type(exc).__name__, "message": str(exc)}
            continue
        out["total_programs_evaluated"] += int(res.n_evaluated)
        best = None if res.best is None else {"program": res.best.describe(), "ood_auc": res.best.ood_auc, "p": res.best.meta.get("permutation_p"), "promoted": res.best.promoted, "n_evaluated": res.n_evaluated, "search_log": res.log.__dict__}
        if best and far is not None:
            try:
                auc, _, _ = _fit_eval(res.best, real.range_dataset(*train), far, null, 512, 768, seed=777_000 + seed)
                best["far_ood_auc"] = auc
            except Exception as exc:
                best["far_ood_error"] = str(exc)
        out["seeds"][str(seed)] = best
        try:
            sh = evolutionary_search(real, null, max(2000, budget // 10), seed + 10_000, shuffle_labels=True, train_range=train, val_range=val, ood_range=ood)
        except Exception as exc:
            out["shuffle_controls"][str(seed)] = {"error": type(exc).__name__, "message": str(exc)}
            continue
        out["total_programs_evaluated"] += int(sh.n_evaluated)
        if sh.best is None:
            out["shuffle_controls"][str(seed)] = None
        else:
            p = float(sh.best.meta.get("permutation_p", 1.0))
            out["shuffle_controls"][str(seed)] = {"program": sh.best.describe(), "ood_auc": sh.best.ood_auc, "p": p, "p_bonferroni": min(1.0, p * out["bonferroni_controls"]), "search_log": sh.log.__dict__}
    vals = [v["ood_auc"] for v in out["seeds"].values() if v and "ood_auc" in v]
    out["mean_ood_auc"] = float(np.mean(vals)) if vals else None
    out["promoted_count"] = sum(1 for v in out["seeds"].values() if v and v.get("promoted"))
    return out


def run(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zeros_results.json"
    if out_path.exists(): raise SystemExit(f"{out_path} exists; refusing to overwrite")
    zeros = load_zeros(args.zeros)
    real = ZeroSliceDataset(zeros, window_len=args.window_len)
    far = ZeroSliceDataset(load_zeros(args.far_zeros), window_len=args.window_len) if args.far_zeros else None
    seeds = [int(s) for s in args.seeds]
    gue = GUESpacingNull(seed=0, matrix_size=args.gue_matrix_size)
    tilted, tilt_steps = sequential_tilt(real, args.gue_matrix_size, args.gue_cache_windows, args.support_probe_n, seed=0)
    payload = {
        "cmd": " ".join([sys.executable, "-m", "leadengine.run_zeros", *sys.argv[1:]]),
        "zeros": args.zeros, "far_zeros": args.far_zeros, "seeds": seeds, "budget": args.budget,
        "gue_matrix_size": args.gue_matrix_size, "gue_cache_windows": args.gue_cache_windows,
        "rungs": [
            rung("poisson", real, PoissonSpacingNull(), args.budget, seeds, far, bonferroni_controls=3 * len(seeds)),
            rung("gue", real, gue, args.budget, seeds, far, bonferroni_controls=3 * len(seeds)),
            rung("gue_tilted_empirical_low_order", real, tilted, args.budget, seeds, far, bonferroni_controls=3 * len(seeds)),
        ],
        "absorbed_in_tilted": [s.describe() for s in empirical_stats()],
        "tilted_sequential_steps": tilt_steps,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeros", default="runs/science_001/c_zeros/odlyzko/zeros6")
    p.add_argument("--far-zeros", default="runs/science_001/c_zeros/odlyzko/zeros3_absolute_decimal.txt")
    p.add_argument("--budget", type=int, default=20_000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--out", default="runs/science_001/c_zeros")
    p.add_argument("--gue-matrix-size", type=int, default=1024)
    p.add_argument("--gue-cache-windows", type=int, default=4000)
    p.add_argument("--support-probe-n", type=int, default=512)
    print(json.dumps(run(p.parse_args()), sort_keys=True)[:1000])


if __name__ == "__main__":
    main()
