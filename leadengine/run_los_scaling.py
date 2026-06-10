from __future__ import annotations

import argparse, json, math, statistics, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core import Window
from .dsl import parse_program
from .nulls import WheelNull
from .prereg import register, register_prediction_interval, score_extrapolation
from .scale import fit_templates, effect_profile
from .sieve import primes_in
from .stats import ResiduePairCount


@dataclass
class SpanDataset:
    decade: int
    window_len: int
    seed: int
    spans: int
    span_len: int
    name: str = "batched_prime_windows"
    domain: str = "primes"
    _cache: list[tuple[int, np.ndarray]] = field(default_factory=list, init=False)

    def _build(self) -> None:
        if self._cache:
            return
        lo, hi = 10 ** self.decade, 10 ** (self.decade + 1)
        usable = max(self.window_len + 1, min(self.span_len + self.window_len, 10_000_000, hi - lo))
        rng = np.random.default_rng(self.seed + 1009 * self.decade)
        for _ in range(max(1, self.spans)):
            start = int(rng.integers(lo, max(lo + 1, hi - usable)))
            indicator = np.zeros(usable, dtype=np.int8)
            ps = primes_in(start, start + usable)
            if ps.size:
                indicator[(ps - start).astype(np.int64)] = 1
            self._cache.append((start, indicator))

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        self._build()
        out: list[Window] = []
        for _ in range(int(n_windows)):
            base, ind = self._cache[int(rng.integers(0, len(self._cache)))]
            off = int(rng.integers(0, max(1, ind.size - self.window_len)))
            start = base + off
            vals = np.arange(start, start + self.window_len, dtype=np.int64)
            out.append(Window(ind[off : off + self.window_len].copy(), start, {"domain": self.domain, "integer_values": vals}))
        return out

    def scale_of(self, w: Window) -> int:
        return int(math.log10(max(int(w.start), 1)))


class Factory:
    def __init__(self, seed: int, window_len: int, spans: int, span_len: int):
        self.seed, self.window_len, self.spans, self.span_len = seed, window_len, spans, span_len
        self.cache: dict[int, SpanDataset] = {}

    def __call__(self, decade: int) -> SpanDataset:
        d = int(decade)
        if d not in self.cache:
            self.cache[d] = SpanDataset(d, self.window_len, self.seed, self.spans, self.span_len)
        return self.cache[d]


def _mean_sd(vals: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(vals)), "sd": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0}


def _summarize(seed_profiles: dict[int, dict[int, tuple[float, float]]]) -> dict[str, Any]:
    decades = sorted(next(iter(seed_profiles.values())).keys())
    out = {}
    for d in decades:
        effects = [seed_profiles[s][d][0] for s in sorted(seed_profiles)]
        ses = [seed_profiles[s][d][1] for s in sorted(seed_profiles)]
        out[str(d)] = {**_mean_sd(effects), "mean_bootstrap_se": float(statistics.mean(ses)), "seed_values": effects, "seed_se": ses}
    return out


def _predict(profile: dict[int, tuple[float, float]], template: str, target: int) -> float:
    fit = fit_templates(profile)[template]
    c = float(fit["coefficients"][0])
    n = 10.0 ** (target + 0.5)
    logn = math.log(n)
    x = {"1": 1.0, "1/log n": 1.0 / logn, "1/log^2 n": 1.0 / (logn * logn), "loglog n / log n": math.log(logn) / logn}[template]
    return c * x


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prereg_dir = out / "prereg"
    null = WheelNull(30)
    stats = {
        "residue_pair_q10": ResiduePairCount(10),
        "phase3_pair_hist_q10": parse_program(args.phase3_program),
    }
    payload: dict[str, Any] = {"cmd": " ".join(args.cmd), "seeds": args.seeds, "n_per_decade": args.n_per_decade, "stats": {}, "total_statistics_evaluated": len(stats)}
    for name, stat in stats.items():
        seed_profiles: dict[int, dict[int, tuple[float, float]]] = {}
        for seed in args.seeds:
            factory = Factory(seed, args.window_len, args.spans, args.span_len)
            seed_profiles[int(seed)] = effect_profile(stat, factory, null, args.fit_decades, args.n_per_decade, np.random.default_rng(seed))
        mean_profile = {d: (float(statistics.mean([seed_profiles[s][d][0] for s in seed_profiles])), float(statistics.mean([seed_profiles[s][d][1] for s in seed_profiles]))) for d in args.fit_decades}
        fits = fit_templates(mean_profile)
        best = max(fits, key=lambda k: fits[k]["r2"])
        seed_preds = [_predict(seed_profiles[s], best, args.target_decade) for s in sorted(seed_profiles)]
        pred, pred_sd = _mean_sd(seed_preds)["mean"], _mean_sd(seed_preds)["sd"]
        ci_low, ci_high = pred - 2.0 * max(pred_sd, 1e-6), pred + 2.0 * max(pred_sd, 1e-6)
        lead_id = f"{name}_decade_{args.target_decade}"
        prereg_path = prereg_dir / f"{lead_id}.json"
        if prereg_path.exists():
            raise SystemExit(f"{prereg_path} exists; refusing to overwrite")
        if str(args.interval_type).upper() == "PI":
            largest = max(int(d) for d in args.fit_decades)
            sampling_se = float(mean_profile[largest][1])
            register_prediction_interval(
                lead_id,
                args.fit_decades,
                pred,
                fit_se=max(pred_sd, 1e-6),
                sampling_se=sampling_se,
                out_dir=prereg_dir,
                z=float(args.z),
            )
        else:
            register(lead_id, args.fit_decades, pred, ci_low, ci_high, prereg_dir)

        def scorer(decade: int, seed: int) -> float:
            prof = effect_profile(stat, Factory(seed, args.window_len, args.spans, args.span_len), null, [decade], args.n_per_decade, np.random.default_rng(seed))
            return prof[int(decade)][0]

        extrap = score_extrapolation(lead_id, args.target_decade, out_dir=prereg_dir, scorer=scorer)
        payload["stats"][name] = {
            "formula": stat.describe(), "seed_profiles": {str(k): {str(d): list(v) for d, v in p.items()} for k, p in seed_profiles.items()},
            "summary": _summarize(seed_profiles), "mean_profile": {str(d): list(v) for d, v in mean_profile.items()},
            "template_fits": fits, "best_template": best, "target_seed": extrap["seed"], "target_seeds": extrap["target_seeds"], "extrapolation": extrap,
        }
    path = out / "los_scaling_results.json"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/science_001/a_los_scaling")
    p.add_argument("--n-per-decade", type=int, default=4000)
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--fit-decades", nargs="+", type=int, default=[5, 6, 7, 8, 9, 10])
    p.add_argument("--target-decade", type=int, default=11)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--spans", type=int, default=4)
    p.add_argument("--span-len", type=int, default=9_999_488)
    p.add_argument("--phase3-program", default="pair_hist(pairs(mod(positions(w),10),1),10)")
    p.add_argument("--interval-type", choices=["CI", "PI"], default="CI")
    p.add_argument("--z", type=float, default=2.0)
    args = p.parse_args()
    args.cmd = [sys.executable, "-m", "leadengine.run_los_scaling", *sys.argv[1:]]
    print(json.dumps(run(args), sort_keys=True)[:1000])


if __name__ == "__main__":
    main()
