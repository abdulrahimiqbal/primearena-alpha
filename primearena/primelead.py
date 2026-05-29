from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lead_cards import make_lead_card, write_cards
from .lead_finder import run_null_experiment
from .measurements import default_measurements, measurement_summary
from .null_refiner import suggest_refined_null


DECISION_PROMOTED = "PrimeLead found promoted candidate leads."
DECISION_NONE = "PrimeLead found no promoted leads beyond the current null ladder."
DECISION_BUG_PREFIX = "PrimeLead found a likely bug or leakage:"


DEFAULTS: Dict[str, Any] = {
    "n_min": 100_000,
    "n_max": 100_000_000,
    "window_size": 512,
    "nulls": ["cramer", "wheel", "gap_hist", "residue_pair", "block_bootstrap"],
    "q_values": [6, 10, 30, 210, 2310],
    "sieve_bounds": [30, 210, 1000],
    "measurement_budget": 10.0,
    "complexity_penalty": True,
    "max_complexity": 80.0,
    "ood_multiplier": 10,
    "train_steps": 250,
}


def _load_primelead_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PrimeLead config not found: {p}")
    data = json.loads(p.read_text())
    section = data.get("primelead", data)
    cfg.update(section)
    return cfg


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for null_name in sorted({str(r["null_model"]) for r in rows}):
        subset = [r for r in rows if r["null_model"] == null_name]
        stats: Dict[str, float] = {}
        for key in sorted({k for r in subset for k, v in r.items() if isinstance(v, (int, float)) and k != "seed"}):
            vals = np.asarray([float(r[key]) for r in subset if isinstance(r.get(key), (int, float))], dtype=np.float64)
            if vals.size:
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[null_name] = stats
    return out


def _promotion_failures(row_or_stats: Dict[str, float], seeds: int, pass_seed_count: int) -> List[str]:
    failures: List[str] = []
    auc = float(row_or_stats.get("real_vs_fake_AUC", row_or_stats.get("real_vs_fake_AUC_mean", 0.0)))
    ood_auc = float(row_or_stats.get("OOD_AUC", row_or_stats.get("OOD_AUC_mean", 0.0)))
    bits = float(row_or_stats.get("heldout_bits_saved", row_or_stats.get("heldout_bits_saved_mean", 0.0)))
    perm_delta = float(row_or_stats.get("permutation_auc_delta", row_or_stats.get("permutation_auc_delta_mean", 0.0)))
    complexity = float(row_or_stats.get("feature_complexity", row_or_stats.get("feature_complexity_mean", 0.0)))
    if auc < 0.55:
        failures.append("AUC vs current null < 0.55")
    if ood_auc < 0.53:
        failures.append("OOD AUC < 0.53")
    if bits <= 0:
        failures.append("heldout bits_saved <= 0")
    if perm_delta <= 0:
        failures.append("permutation test did not beat shuffled-label baseline")
    if pass_seed_count < 3:
        failures.append("effect did not survive at least 3 seeds")
    if complexity > float(DEFAULTS["max_complexity"]):
        failures.append("complexity score above threshold")
    if seeds < 3:
        failures.append("fewer than 3 seeds requested, so promotion is disabled")
    return failures


def _seed_pass(row: Dict[str, Any], max_complexity: float) -> bool:
    return bool(
        float(row.get("real_vs_fake_AUC", 0.0)) >= 0.55
        and float(row.get("OOD_AUC", 0.0)) >= 0.53
        and float(row.get("heldout_bits_saved", 0.0)) > 0.0
        and float(row.get("permutation_auc_delta", 0.0)) > 0.0
        and float(row.get("feature_complexity", 0.0)) <= float(max_complexity)
    )


def _bug_or_leakage(rows: List[Dict[str, Any]]) -> Optional[str]:
    for row in rows:
        if float(row.get("permutation_test_AUC", 0.5)) >= 0.60:
            return f"permutation_test_AUC={row['permutation_test_AUC']:.4f} for null {row['null_model']}"
    return None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# PrimeLead Null-Ladder Report",
        "",
        "PrimeLead searches for statistics that distinguish real prime windows from synthetic null worlds. This is a lead-generation report, not a discovery claim.",
        "",
        f"Conclusion: **{summary['decision']}**",
        "",
        "## Measurement Budget",
        "",
        f"- Budget: `{summary['measurement_budget']}`",
        f"- Selected measurement cost: `{summary['measurement_summary']['total_cost']:.4f}`",
        f"- Selected complexity: `{summary['measurement_summary']['total_complexity']:.4f}`",
        "",
        "Selected measurements:",
    ]
    for name in summary["measurement_summary"]["names"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Null Results",
            "",
            "| null | AUC | acc | OOD AUC | bits saved | permutation delta | promoted |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for null_name, stats in summary["aggregates"].items():
        promoted = any(card["tested_null_model"] == null_name and card["promoted"] for card in summary["lead_cards"])
        lines.append(
            "| "
            + " | ".join(
                [
                    null_name,
                    _fmt(stats.get("real_vs_fake_AUC_mean", 0.0)),
                    _fmt(stats.get("real_vs_fake_accuracy_mean", 0.0)),
                    _fmt(stats.get("OOD_AUC_mean", 0.0)),
                    _fmt(stats.get("heldout_bits_saved_mean", 0.0)),
                    _fmt(stats.get("permutation_auc_delta_mean", 0.0)),
                    str(promoted),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Lead Cards", ""])
    for card in summary["lead_cards"]:
        lines.append(
            f"- `{card['lead_id']}` promoted={card['promoted']} AUC={card['effect_size']['AUC']:.4f} "
            f"OOD={card['effect_size']['OOD_AUC']:.4f} bits={card['bits_saved']:.6f}"
        )
    lines.extend(["", "No discovery claim is made.", "", summary["decision"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_primelead(
    config: str,
    out_dir: str,
    samples: int,
    nulls: Sequence[str],
    measurement_budget: float,
    seeds: int,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = _load_primelead_config(config)
    if n_min is not None:
        cfg["n_min"] = int(n_min)
    if n_max is not None:
        cfg["n_max"] = int(n_max)
    cfg["nulls"] = [str(x) for x in nulls] if nulls else list(cfg["nulls"])
    cfg["measurement_budget"] = float(measurement_budget)
    q_values = [int(q) for q in cfg.get("q_values", DEFAULTS["q_values"])]
    sieve_bounds = [int(x) for x in cfg.get("sieve_bounds", DEFAULTS["sieve_bounds"])]
    measurements = default_measurements(q_values, sieve_bounds, cfg["measurement_budget"])
    ms = measurement_summary(measurements)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    for null_name in cfg["nulls"]:
        for seed_index in range(1, int(seeds) + 1):
            seed = 810_000 + seed_index * 1009 + sum(ord(c) for c in null_name)
            print(f"[primelead] null={null_name} seed={seed_index}/{seeds}", flush=True)
            result = run_null_experiment(
                null_name=null_name,
                measurements=measurements,
                samples=int(samples),
                n_min=int(cfg["n_min"]),
                n_max=int(cfg["n_max"]),
                window_size=int(cfg["window_size"]),
                q_values=q_values,
                sieve_bounds=sieve_bounds,
                seed=seed,
                ood_multiplier=int(cfg.get("ood_multiplier", 10)),
                train_steps=int(cfg.get("train_steps", 250)),
            )
            raw_results.append(result)
            row = {
                "null_model": null_name,
                "seed": seed,
                "seed_index": seed_index,
                **{k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))},
                "top_feature": result["top_features"][0]["feature"] if result["top_features"] else "",
            }
            rows.append(row)

    aggregates = _aggregate(rows)
    lead_cards: List[Dict[str, Any]] = []
    max_complexity = float(cfg.get("max_complexity", DEFAULTS["max_complexity"]))
    for result in raw_results:
        null_name = str(result["null_model"])
        pass_count = sum(1 for r in rows if r["null_model"] == null_name and _seed_pass(r, max_complexity))
        failures = _promotion_failures(result["metrics"], int(seeds), pass_count)
        promoted = len(failures) == 0
        lead_stub = {**result, "top_features": result.get("top_features", [])}
        next_null = suggest_refined_null(lead_stub)
        lead_cards.append(make_lead_card(result, promoted=promoted, failure_reasons=failures, next_null=next_null))

    bug = _bug_or_leakage(rows)
    if bug:
        decision = f"{DECISION_BUG_PREFIX} {bug}."
    elif any(card["promoted"] for card in lead_cards):
        decision = DECISION_PROMOTED
    else:
        decision = DECISION_NONE

    summary = {
        "config": config,
        "samples": int(samples),
        "n_min": int(cfg["n_min"]),
        "n_max": int(cfg["n_max"]),
        "window_size": int(cfg["window_size"]),
        "nulls": list(cfg["nulls"]),
        "seeds": int(seeds),
        "measurement_budget": float(cfg["measurement_budget"]),
        "measurement_summary": ms,
        "rows": rows,
        "aggregates": aggregates,
        "lead_cards": lead_cards,
        "decision": decision,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "summary.csv", rows)
    write_cards(lead_cards, out / "lead_cards")
    _write_report(out / "PRIMELEAD_REPORT.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/primelead_null_ladder.json")
    parser.add_argument("--out-dir", default="runs/primelead_null_ladder_001")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--nulls", nargs="+", default=None)
    parser.add_argument("--measurement-budget", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    cfg = _load_primelead_config(args.config)
    nulls = args.nulls if args.nulls is not None else cfg.get("nulls", DEFAULTS["nulls"])
    summary = run_primelead(
        config=args.config,
        out_dir=args.out_dir,
        samples=args.samples,
        nulls=nulls,
        measurement_budget=args.measurement_budget,
        seeds=args.seeds,
        n_min=args.n_min,
        n_max=args.n_max,
    )
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
