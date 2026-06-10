from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np

from .absorb import AbsorptionError
from .core import Window
from .datasets import PrimeWindowDataset
from .dsl import parse_program
from .nulls import WheelNull
from .scoring import auc_real_vs_null
from .search import evolutionary_search
from .stats import ResiduePairCount


class GapHist:
    name = "gap_histogram_even_2_60"
    def __call__(self, w: Window):
        pos = np.flatnonzero(np.asarray(w.values) > 0) + int(w.start)
        gaps = np.diff(pos) if pos.size > 1 else np.asarray([], dtype=np.int64)
        bins = np.arange(2, 62, 2)
        c = np.zeros(len(bins), dtype=np.float32)
        for g in gaps:
            if 2 <= int(g) <= 60 and int(g) % 2 == 0:
                c[(int(g) - 2) // 2] += 1
        return c / max(float(c.sum()), 1.0)
    def complexity(self): return 4.0
    def describe(self): return self.name


class Density:
    name = "window_density"
    def __call__(self, w: Window): return np.asarray([float(np.mean(w.values))], dtype=np.float32)
    def complexity(self): return 1.0
    def describe(self): return self.name


def knowns() -> list[Any]:
    out: list[Any] = []
    for q in (3, 4, 6, 10, 12, 30):
        out.append(parse_program(f"hist(mod(positions(w),{q}),{q})"))
    for q in (3, 4, 6, 10, 12):
        out.append(ResiduePairCount(q))
    out.append(GapHist())
    for k in range(1, 17):
        out.append(parse_program(f"scalar_vec(fourier_power(w,{k}))"))
    out.append(Density())
    return out


def find_known(best, current, real, seed: int, n: int) -> tuple[str | None, Any | None, float | None]:
    for stat in knowns():
        try:
            test_null = current.absorb(stat, real)
            auc = auc_real_vs_null(best, real, test_null, n=n, rng=np.random.default_rng(seed), seed_split=seed)
        except AbsorptionError:
            continue
        if auc <= 0.58:
            return stat.describe(), stat, float(auc)
    return None, None, None


def same_structure(best_i, best_j, current, real, seed: int, n: int) -> float:
    absorbed = current.absorb(best_i, real)
    return auc_real_vs_null(best_j, real, absorbed, n=n, rng=np.random.default_rng(seed), seed_split=seed)


def replay_null(absorbed: list[str], real):
    current = WheelNull(30)
    by_name = {s.describe(): s for s in knowns()}
    for name in absorbed:
        current = current.absorb(by_name[name], real)
    return current


def write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ledgers = sorted(out.glob("absorption_ledger_*.json"))
    real = PrimeWindowDataset(args.train_min, args.ood_max - 1, args.window_len)
    ledger: dict[str, Any]
    absorbed: list[str]
    start_round = 0
    if args.resume and ledgers:
        ledger = json.loads(ledgers[-1].read_text())
        absorbed = list(ledger.get("absorbed", []))
        start_round = len(ledger.get("rounds", []))
    else:
        if ledgers:
            raise SystemExit(f"{ledgers[-1]} exists; use --resume or a new out dir")
        ledger = {"cmd": " ".join([sys.executable, "-m", "leadengine.run_absorption", *sys.argv[1:]]), "rounds": [], "absorbed": [], "total_programs_evaluated": 0, "scope_note": "prime-domain evolutionary_search eligible set from validated search.py"}
        absorbed = []
    current = replay_null(absorbed, real)
    for r in range(start_round, args.max_rounds):
        round_data: dict[str, Any] = {"round": r, "null": current.name, "seeds": {}, "shuffle_controls": {}, "promoted": [], "known_labels": []}
        promoted = []
        for seed in args.seeds:
            res = evolutionary_search(real, current, args.budget, seed, train_range=(args.train_min,args.train_max), val_range=(args.val_min,args.val_max), ood_range=(args.ood_min,args.ood_max))
            ledger["total_programs_evaluated"] += int(res.n_evaluated)
            best = None if res.best is None else {"program": res.best.describe(), "ood_auc": res.best.ood_auc, "p": res.best.meta.get("permutation_p"), "promoted": res.best.promoted, "n_evaluated": res.n_evaluated}
            round_data["seeds"][str(seed)] = best
            if res.best is not None and res.best.promoted:
                promoted.append(res.best)
            sh = evolutionary_search(real, current, max(2000, args.budget // 10), seed + 10_000, shuffle_labels=True, train_range=(args.train_min,args.train_max), val_range=(args.val_min,args.val_max), ood_range=(args.ood_min,args.ood_max))
            ledger["total_programs_evaluated"] += int(sh.n_evaluated)
            round_data["shuffle_controls"][str(seed)] = None if sh.best is None else {"program": sh.best.describe(), "ood_auc": sh.best.ood_auc, "p": sh.best.meta.get("permutation_p")}
            if sh.best is not None and sh.best.ood_auc > 0.55 and float(sh.best.meta.get("permutation_p", 1.0)) < 0.01:
                round_data["tripwire"] = "T2_ENGINE_LEAKAGE"
                ledger["rounds"].append(round_data); write_new(out / f"absorption_ledger_round_{r:02d}.json", ledger); return ledger
        if len(promoted) < 2:
            round_data["stop"] = "fixed_point_no_promotion_in_at_least_2_of_3_seeds"
            ledger["rounds"].append(round_data); write_new(out / f"absorption_ledger_round_{r:02d}.json", ledger); return ledger
        pairs = [same_structure(promoted[0], p, current, real, 31_000 + r * 100 + i, args.comparator_n) for i, p in enumerate(promoted[1:], 1)]
        round_data["same_structure_auc"] = pairs
        if not all(x <= 0.58 for x in pairs):
            round_data["stop"] = "promotions_not_same_structure"
            ledger["rounds"].append(round_data); write_new(out / f"absorption_ledger_round_{r:02d}.json", ledger); return ledger
        label, stat, auc = find_known(promoted[0], current, real, 41_000 + r, args.comparator_n)
        if label and stat is not None:
            round_data["known_labels"].append({"label": f"REDISCOVERED({label})", "kill_auc": auc})
            absorbed.append(label); ledger["absorbed"] = absorbed
            current = current.absorb(stat, real)
        else:
            round_data["promoted"] = [p.describe() for p in promoted]
            round_data["stop"] = "survivor_requires_gauntlet"
            ledger["rounds"].append(round_data); write_new(out / f"absorption_ledger_round_{r:02d}.json", ledger); return ledger
        ledger["rounds"].append(round_data); write_new(out / f"absorption_ledger_round_{r:02d}.json", ledger)
    return ledger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/science_001/b_absorption")
    p.add_argument("--budget", type=int, default=20_000)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--train-min", type=int, default=100_000); p.add_argument("--train-max", type=int, default=1_000_000)
    p.add_argument("--val-min", type=int, default=1_000_000); p.add_argument("--val-max", type=int, default=10_000_000)
    p.add_argument("--ood-min", type=int, default=10_000_000); p.add_argument("--ood-max", type=int, default=100_000_000)
    p.add_argument("--comparator-n", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    print(json.dumps(run(p.parse_args()), sort_keys=True)[:1000])


if __name__ == "__main__":
    main()
