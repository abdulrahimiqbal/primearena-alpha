from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .lead_cards import make_lead_card, write_cards
from .lead_finder import run_null_experiment
from .measurements import (
    Measurement,
    consecutive_residue_pair_counts,
    consecutive_residue_pair_transition_matrix,
    diagonal_vs_offdiagonal_pair_rate,
    local_density_measurement,
    measurement_summary,
    pair_bias_spectrum,
    residue_hist_measurement,
    same_residue_repeat_rate,
)
from .null_refiner import suggest_refined_null


DECISION_PASSED = "PrimeLead calibration passed: known residue-pair bias rediscovered."
DECISION_FAILED = "PrimeLead calibration failed: known signal was not recovered."
DECISION_BUG_PREFIX = "PrimeLead calibration found likely leakage/bug:"


DEFAULTS: Dict[str, Any] = {
    "n_min": 100_000,
    "n_max": 100_000_000,
    "window_size": 1024,
    "q_values": [10, 30],
    "samples": 20_000,
    "seeds": 3,
    "measurement_budget": 10.0,
    "ood_multiplier": 10,
    "train_steps": 250,
}


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PrimeLead calibration config not found: {p}")
    data = json.loads(p.read_text())
    cfg.update(data.get("primelead_calibration", data))
    return cfg


def calibration_measurements(q: int, budget: float) -> List[Measurement]:
    q = int(q)
    candidates = [
        same_residue_repeat_rate(q),
        diagonal_vs_offdiagonal_pair_rate(q),
        local_density_measurement(64),
        local_density_measurement(128),
        consecutive_residue_pair_counts(q),
        consecutive_residue_pair_transition_matrix(q),
        pair_bias_spectrum(q),
        residue_hist_measurement(q),
    ]
    selected: List[Measurement] = []
    spent = 0.0
    for m in candidates:
        if spent + m.cost <= float(budget) or not selected:
            selected.append(m)
            spent += m.cost
    return selected


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
    keys = sorted({(int(r["q"]), str(r["null_model"])) for r in rows})
    for q, null_name in keys:
        subset = [r for r in rows if int(r["q"]) == q and str(r["null_model"]) == null_name]
        stats: Dict[str, float] = {}
        for key in sorted({k for r in subset for k, v in r.items() if isinstance(v, (int, float)) and k not in {"seed", "q"}}):
            vals = np.asarray([float(r[key]) for r in subset if isinstance(r.get(key), (int, float))], dtype=np.float64)
            if vals.size:
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[f"q{q}:{null_name}"] = stats
    return out


def _is_pair_lead(row: Dict[str, Any]) -> bool:
    feature = str(row.get("top_feature", "")).lower()
    return any(token in feature for token in ["pair", "residue_repeat", "diagonal_vs_offdiagonal"])


def _seed_pass(row: Dict[str, Any]) -> bool:
    return bool(
        float(row.get("real_vs_fake_AUC", 0.0)) >= 0.55
        and float(row.get("OOD_AUC", 0.0)) >= 0.53
        and float(row.get("heldout_bits_saved", 0.0)) > 0.0
        and float(row.get("permutation_auc_delta", 0.0)) > 0.0
        and _is_pair_lead(row)
    )


def _bug_or_leakage(rows: List[Dict[str, Any]]) -> Optional[str]:
    for row in rows:
        if float(row.get("permutation_test_AUC", 0.5)) >= 0.60:
            return f"permutation baseline AUC={row['permutation_test_AUC']:.4f} for q={row['q']} null={row['null_model']}"
    return None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# PrimeLead Calibration Report",
        "",
        "Known-positive target: consecutive-prime residue-pair bias. This is a calibration report, not a discovery claim.",
        "",
        f"Conclusion: **{summary['decision']}**",
        "",
        "## Metrics By q And Null",
        "",
        "| q | null | AUC | OOD AUC | bits saved | permutation AUC | promoted leads | top feature |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["q"]),
                    str(row["null_model"]),
                    _fmt(row.get("real_vs_fake_AUC", 0.0)),
                    _fmt(row.get("OOD_AUC", 0.0)),
                    _fmt(row.get("heldout_bits_saved", 0.0)),
                    _fmt(row.get("permutation_test_AUC", 0.0)),
                    str(row.get("promoted", False)),
                    f"`{row.get('top_feature', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Calibration Checks", ""])
    for q, check in summary["checks_by_q"].items():
        lines.append(
            f"- q={q}: iid_auc={check['iid_auc']:.4f}, matched_auc={check['matched_auc']:.4f}, "
            f"auc_drop={check['auc_drop']:.4f}, iid_promoted={check['iid_promoted_count']}, "
            f"matched_promoted={check['matched_promoted_count']}, passes={check['passes']}"
        )
    lines.extend(["", "## Lead Cards", ""])
    for card in summary["lead_cards"]:
        lines.append(
            f"- `{card['lead_id']}` q={card.get('q')} null={card['tested_null_model']} promoted={card['promoted']} "
            f"AUC={card['effect_size']['AUC']:.4f} OOD={card['effect_size']['OOD_AUC']:.4f}"
        )
    lines.extend(["", "No discovery claim is made.", "", summary["decision"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def _promotion_failures(row: Dict[str, Any], group_pass_count: int, seeds: int) -> List[str]:
    failures: List[str] = []
    if float(row.get("real_vs_fake_AUC", 0.0)) < 0.55:
        failures.append("AUC < 0.55")
    if float(row.get("OOD_AUC", 0.0)) < 0.53:
        failures.append("OOD AUC < 0.53")
    if float(row.get("heldout_bits_saved", 0.0)) <= 0.0:
        failures.append("heldout bits_saved <= 0")
    if float(row.get("permutation_auc_delta", 0.0)) <= 0.0:
        failures.append("permutation baseline was not beaten")
    if not _is_pair_lead(row):
        failures.append("top lead is not residue-pair based")
    if group_pass_count < 3:
        failures.append("residue-pair effect did not repeat across 3 seeds")
    if int(seeds) < 3:
        failures.append("fewer than 3 seeds requested, so calibration promotion is disabled")
    return failures


def run_calibration(
    config: str,
    out_dir: str,
    samples: int,
    q_values: Sequence[int],
    seeds: int,
) -> Dict[str, Any]:
    cfg = _load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q_values = [int(q) for q in q_values] if q_values else [int(q) for q in cfg.get("q_values", [10, 30])]
    rows: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    nulls = ["wheel_iid_no_pair_bias", "residue_pair_matched_null"]
    for q in q_values:
        measurements = calibration_measurements(q, float(cfg.get("measurement_budget", 10.0)))
        for null_name in nulls:
            for seed_index in range(1, int(seeds) + 1):
                seed = 910_000 + q * 100 + seed_index * 1009 + sum(ord(c) for c in null_name)
                print(f"[primelead-calibration] q={q} null={null_name} seed={seed_index}/{seeds}", flush=True)
                result = run_null_experiment(
                    null_name=null_name,
                    measurements=measurements,
                    samples=int(samples),
                    n_min=int(cfg["n_min"]),
                    n_max=int(cfg["n_max"]),
                    window_size=int(cfg["window_size"]),
                    q_values=[q],
                    sieve_bounds=[q],
                    seed=seed,
                    ood_multiplier=int(cfg.get("ood_multiplier", 10)),
                    train_steps=int(cfg.get("train_steps", 250)),
                )
                result["q"] = q
                raw_results.append(result)
                row = {
                    "q": q,
                    "null_model": null_name,
                    "seed": seed,
                    "seed_index": seed_index,
                    **{k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))},
                    "top_feature": result["top_features"][0]["feature"] if result["top_features"] else "",
                    "best_lead_expression": result["top_features"][0]["feature"] if result["top_features"] else "",
                    "best_lead_complexity": float(result["metrics"].get("feature_complexity", 0.0)),
                    "effect_size": float(result["metrics"].get("real_vs_fake_AUC", 0.0) - 0.5),
                    "auc_ci_low": float(result["metrics"].get("auc_ci_low", 0.0)),
                    "auc_ci_high": float(result["metrics"].get("auc_ci_high", 0.0)),
                }
                rows.append(row)

    group_pass_counts: Dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["q"]), str(row["null_model"]))
        group_pass_counts[key] = group_pass_counts.get(key, 0) + int(_seed_pass(row))

    lead_cards: List[Dict[str, Any]] = []
    for result, row in zip(raw_results, rows):
        key = (int(row["q"]), str(row["null_model"]))
        failures = _promotion_failures(row, group_pass_counts.get(key, 0), int(seeds))
        promoted = len(failures) == 0
        next_null = suggest_refined_null(result)
        card = make_lead_card(result, promoted=promoted, failure_reasons=failures, next_null=next_null)
        card["q"] = int(row["q"])
        card["lead_id"] = f"{card['lead_id']}_q{int(row['q'])}"
        card["lead_name"] = f"q={int(row['q'])} {card['lead_name']}"
        lead_cards.append(card)
        row["promoted"] = bool(promoted)
        row["promotion_failure_reasons"] = "; ".join(failures)

    aggregates = _aggregate(rows)
    checks_by_q: Dict[str, Dict[str, Any]] = {}
    passing_qs: List[int] = []
    for q in q_values:
        iid = aggregates.get(f"q{q}:wheel_iid_no_pair_bias", {})
        matched = aggregates.get(f"q{q}:residue_pair_matched_null", {})
        iid_auc = float(iid.get("real_vs_fake_AUC_mean", 0.0))
        matched_auc = float(matched.get("real_vs_fake_AUC_mean", 0.0))
        iid_promoted = sum(1 for c in lead_cards if int(c.get("q", -1)) == q and c["tested_null_model"] == "wheel_iid_no_pair_bias" and c["promoted"])
        matched_promoted = sum(1 for c in lead_cards if int(c.get("q", -1)) == q and c["tested_null_model"] == "residue_pair_matched_null" and c["promoted"])
        iid_basic = bool(
            iid_auc >= 0.55
            and float(iid.get("OOD_AUC_mean", 0.0)) >= 0.53
            and float(iid.get("heldout_bits_saved_mean", 0.0)) > 0.0
            and iid_promoted >= 1
        )
        weakened = bool((iid_auc - matched_auc) >= 0.03 or matched_promoted == 0)
        passes = bool(iid_basic and weakened)
        if passes:
            passing_qs.append(q)
        checks_by_q[str(q)] = {
            "iid_auc": iid_auc,
            "matched_auc": matched_auc,
            "auc_drop": float(iid_auc - matched_auc),
            "iid_ood_auc": float(iid.get("OOD_AUC_mean", 0.0)),
            "matched_ood_auc": float(matched.get("OOD_AUC_mean", 0.0)),
            "iid_bits_saved": float(iid.get("heldout_bits_saved_mean", 0.0)),
            "matched_bits_saved": float(matched.get("heldout_bits_saved_mean", 0.0)),
            "iid_promoted_count": int(iid_promoted),
            "matched_promoted_count": int(matched_promoted),
            "iid_basic_gates_pass": iid_basic,
            "matched_weakened": weakened,
            "passes": passes,
        }

    bug = _bug_or_leakage(rows)
    if bug:
        decision = f"{DECISION_BUG_PREFIX} {bug}."
    elif passing_qs:
        decision = DECISION_PASSED
    else:
        decision = DECISION_FAILED

    summary = {
        "config": config,
        "samples": int(samples),
        "q_values": q_values,
        "seeds": int(seeds),
        "n_min": int(cfg["n_min"]),
        "n_max": int(cfg["n_max"]),
        "window_size": int(cfg["window_size"]),
        "rows": rows,
        "aggregates": aggregates,
        "checks_by_q": checks_by_q,
        "passing_q_values": passing_qs,
        "lead_cards": lead_cards,
        "decision": decision,
        "measurement_summaries": {
            str(q): measurement_summary(calibration_measurements(q, float(cfg.get("measurement_budget", 10.0)))) for q in q_values
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "summary.csv", rows)
    write_cards(lead_cards, out / "lead_cards")
    _write_report(out / "PRIMELEAD_CALIBRATION_REPORT.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/primelead_calibration.json")
    parser.add_argument("--out-dir", default="runs/primelead_calibration_001")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--q-values", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    cfg = _load_config(args.config)
    summary = run_calibration(
        config=args.config,
        out_dir=args.out_dir,
        samples=int(args.samples if args.samples is not None else cfg.get("samples", 20_000)),
        q_values=args.q_values if args.q_values is not None else cfg.get("q_values", [10, 30]),
        seeds=int(args.seeds if args.seeds is not None else cfg.get("seeds", 3)),
    )
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
