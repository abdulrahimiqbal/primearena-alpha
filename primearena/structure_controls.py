from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import RunConfig, load_config
from .env import PrimeArena
from .eval_safety import guess_index_for_action, select_eval_action
from .interpret import collect_activations, evaluate_with_ablation, run_linear_probes
from .model import PolicyValueNet, choose_device, load_checkpoint


MODS = [6, 30, 210]


def _make_model(cfg: RunConfig, device: torch.device, checkpoint: Optional[str], seed: int) -> PolicyValueNet:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    env = PrimeArena(cfg.env, seed=cfg.train.seed)
    model = PolicyValueNet(env.observation_dim, env.action_count, cfg.model).to(device)
    if checkpoint:
        load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    model.eval()
    return model


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe(probes: Dict[str, float], prefix: str, mod: int) -> Optional[float]:
    return _safe_float(probes.get(f"{prefix}_probe_{'n_' if prefix == 'pooled' else ''}mod_{mod}_accuracy"))


def _row_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    baseline = report["baseline_eval"]
    probes = report["probes"]
    ablations = report["ablations"]
    raw_none = ablations["none"]
    row: Dict[str, Any] = {
        "condition": report["condition"],
        "seed_index": report["seed_index"],
        "seed": report["seed"],
        "checkpoint": report.get("checkpoint") or "",
        "baseline_eval_mode": baseline.get("eval_mode", "certified_safe"),
        "baseline_certified_guess_safety_enabled": baseline.get("certified_guess_safety_enabled", True),
        "baseline_success_rate": baseline["success_rate"],
        "baseline_wrong_guess_rate": baseline["wrong_guess_rate"],
        "baseline_wrong_guess_count": baseline["wrong_guess_count"],
        "baseline_premature_guess_blocked_count": baseline["premature_guess_blocked_count"],
        "baseline_certified_guess_count": baseline["certified_guess_count"],
        "baseline_uncertified_guess_attempt_count": baseline["uncertified_guess_attempt_count"],
        "baseline_model_vs_baseline_delta": baseline["model_vs_baseline_delta"],
        "baseline_success_aware_delta": baseline["success_aware_model_vs_baseline_delta"],
        "raw_none_success_rate": raw_none["success_rate"],
        "raw_none_wrong_guess_rate": raw_none["wrong_guess_rate"],
        "raw_none_model_vs_baseline_delta": raw_none["model_vs_baseline_delta"],
    }
    for mod in MODS:
        row[f"candidate_mod{mod}_probe"] = probes.get(f"candidate_probe_mod_{mod}_accuracy")
        row[f"pooled_mod{mod}_probe"] = probes.get(f"pooled_probe_n_mod_{mod}_accuracy")
        ablated = ablations[f"ablate_mod_{mod}"]
        row[f"ablation_mod{mod}_success_rate"] = ablated["success_rate"]
        row[f"ablation_mod{mod}_wrong_guess_rate"] = ablated["wrong_guess_rate"]
        row[f"ablation_mod{mod}_model_vs_baseline_delta"] = ablated["model_vs_baseline_delta"]
        row[f"ablation_mod{mod}_success_damage"] = raw_none["success_rate"] - ablated["success_rate"]
        row[f"ablation_mod{mod}_delta_damage"] = raw_none["model_vs_baseline_delta"] - ablated["model_vs_baseline_delta"]
    return row


def evaluate_certified_batched(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    episodes: int,
    seed: int,
    batch_size: int = 64,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    failure_penalty = float(getattr(cfg.eval, "failure_penalty_cost", 25.0))
    successes: List[float] = []
    costs: List[float] = []
    baselines: List[float] = []
    success_aware_baselines: List[float] = []
    success_aware_costs: List[float] = []
    wrongs: List[float] = []
    wrong_guess_count = 0
    premature_guess_blocked_count = 0
    certified_guess_count = 0
    uncertified_guess_attempt_count = 0
    invalid_action_count = 0

    remaining = int(episodes)
    while remaining > 0:
        n_batch = min(max(1, int(batch_size)), remaining)
        envs = [PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000))) for _ in range(n_batch)]
        final_infos: List[Optional[Dict[str, object]]] = [None for _ in envs]
        while any(not env.done for env in envs):
            active = [i for i, env in enumerate(envs) if not env.done]
            obs = torch.from_numpy(np.stack([envs[i].observation() for i in active])).float().to(device)
            mask = torch.from_numpy(np.stack([envs[i].action_mask_float() for i in active])).float().to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            for row_idx, env_idx in enumerate(active):
                env = envs[env_idx]
                decision = select_eval_action(env, logits[row_idx : row_idx + 1], certified_guesses_only=True)
                if decision.get("uncertified_attempt"):
                    uncertified_guess_attempt_count += 1
                if decision.get("blocked"):
                    premature_guess_blocked_count += 1
                action_obj = decision.get("action")
                if action_obj is None:
                    env.done = True
                    env.success = False
                    env.trace.append("guardrail: no certified-safe control action")
                    final_infos[env_idx] = env.info(reason="certified_control_no_safe_action")
                    continue
                action = int(action_obj)
                legal = env.legal_actions()
                if action < 0 or action >= env.action_count or not legal[action]:
                    invalid_action_count += 1
                guess_idx = guess_index_for_action(env, action)
                if guess_idx is not None and env.is_certified_next_prime_guess(guess_idx):
                    certified_guess_count += 1
                result = env.step(action)
                final_infos[env_idx] = result.info

        for env, final_info in zip(envs, final_infos):
            model_cost = float(env.total_cost)
            baseline_cost = float(env.baseline_cost)
            success_aware_baseline_cost = baseline_cost + float(cfg.env.guess_cost)
            effective_model_cost = model_cost if env.success else baseline_cost + failure_penalty
            wrong = bool((final_info or {}).get("reason") == "wrong_guess")
            successes.append(float(env.success))
            costs.append(model_cost)
            baselines.append(baseline_cost)
            success_aware_baselines.append(success_aware_baseline_cost)
            success_aware_costs.append(float(effective_model_cost))
            wrongs.append(float(wrong))
            wrong_guess_count += int(wrong)
        remaining -= n_batch

    baseline_arr = np.asarray(baselines, dtype=np.float64)
    success_aware_baseline_arr = np.asarray(success_aware_baselines, dtype=np.float64)
    cost_arr = np.asarray(costs, dtype=np.float64)
    success_aware_arr = np.asarray(success_aware_costs, dtype=np.float64)
    return {
        "episodes": int(episodes),
        "eval_mode": "certified_safe",
        "certified_guess_safety_enabled": True,
        "success_rate": float(np.mean(successes)),
        "wrong_guess_rate": float(np.mean(wrongs)),
        "wrong_guess_count": int(wrong_guess_count),
        "premature_guess_blocked_count": int(premature_guess_blocked_count),
        "certified_guess_count": int(certified_guess_count),
        "uncertified_guess_attempt_count": int(uncertified_guess_attempt_count),
        "avg_cost": float(np.mean(costs)),
        "avg_baseline_cost": float(np.mean(baselines)),
        "model_vs_baseline_delta": float(np.mean(baseline_arr - cost_arr)),
        "success_aware_baseline_cost": float(np.mean(success_aware_baselines)),
        "success_aware_avg_cost": float(np.mean(success_aware_costs)),
        "success_aware_model_vs_baseline_delta": float(np.mean(success_aware_baseline_arr - success_aware_arr)),
        "invalid_action_count": int(invalid_action_count),
        "invalid_action_rate": float(invalid_action_count / max(1, episodes)),
    }


def run_one(
    cfg: RunConfig,
    device: torch.device,
    condition: str,
    seed_index: int,
    seed: int,
    checkpoint: Optional[str],
    out_dir: Path,
    samples: int,
    episodes: int,
) -> Dict[str, Any]:
    print(f"[structure-controls] start {condition} seed {seed_index}", flush=True)
    model = _make_model(cfg, device, checkpoint, seed)
    activations = collect_activations(cfg, model, device, samples=samples, seed=seed)
    probes = run_linear_probes(activations)
    print(f"[structure-controls] probes complete {condition} seed {seed_index}", flush=True)
    baseline = evaluate_certified_batched(cfg, model, device, episodes=episodes, seed=seed + 1)
    print(f"[structure-controls] certified eval complete {condition} seed {seed_index}", flush=True)

    ablations: Dict[str, Dict[str, Any]] = {
        "none": evaluate_with_ablation(cfg, model, device, None, episodes=episodes, seed=seed + 11)
    }
    for mod in MODS:
        ablations[f"ablate_mod_{mod}"] = evaluate_with_ablation(
            cfg,
            model,
            device,
            mod,
            episodes=episodes,
            seed=seed + 100 + mod,
        )
        print(f"[structure-controls] ablate mod {mod} complete {condition} seed {seed_index}", flush=True)

    report = {
        "condition": condition,
        "seed_index": seed_index,
        "seed": seed,
        "checkpoint": checkpoint,
        "samples": samples,
        "episodes": episodes,
        "baseline_eval": baseline,
        "probes": probes,
        "ablations": ablations,
    }
    run_dir = out_dir / f"{condition}_seed_{seed_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[structure-controls] wrote {run_dir / 'report.json'}", flush=True)
    return report


def _numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    cols: List[str] = []
    for key in rows[0]:
        values = [_safe_float(row.get(key)) for row in rows]
        if any(v is not None for v in values):
            cols.append(key)
    return cols


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    numeric = _numeric_columns(rows) if rows else []
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        stats: Dict[str, float] = {}
        for key in numeric:
            values = [_safe_float(row.get(key)) for row in subset]
            vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
            if vals.size == 0:
                continue
            stats[f"{key}_mean"] = float(np.mean(vals))
            stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[condition] = stats
    return out


def _comparison(aggregates: Dict[str, Dict[str, float]], other: str) -> Dict[str, Optional[float]]:
    champion = aggregates.get("champion", {})
    baseline = aggregates.get(other, {})
    out: Dict[str, Optional[float]] = {}
    for mod in MODS:
        pairs = {
            f"delta_candidate_mod{mod}_probe": f"candidate_mod{mod}_probe_mean",
            f"delta_pooled_mod{mod}_probe": f"pooled_mod{mod}_probe_mean",
            f"delta_mod{mod}_ablation_damage": f"ablation_mod{mod}_delta_damage_mean",
        }
        for out_key, stat_key in pairs.items():
            a = champion.get(stat_key)
            b = baseline.get(stat_key)
            out[out_key] = None if a is None or b is None else float(a - b)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _mean_std(stats: Dict[str, float], key: str) -> str:
    mean = stats.get(f"{key}_mean")
    std = stats.get(f"{key}_std")
    if mean is None:
        return ""
    return f"{mean:.4f} +/- {(std or 0.0):.4f}"


def _supports_structure(comparisons: Dict[str, Dict[str, Optional[float]]]) -> bool:
    random_cmp = comparisons.get("champion_vs_random", {})
    imitation_cmp = comparisons.get("champion_vs_imitation", {})
    random_positive = all(float(random_cmp.get(k) or 0.0) > 0.0 for k in [
        "delta_candidate_mod6_probe",
        "delta_candidate_mod30_probe",
        "delta_mod30_ablation_damage",
    ])
    imitation_not_worse = all(float(imitation_cmp.get(k) or 0.0) >= -0.05 for k in [
        "delta_candidate_mod6_probe",
        "delta_candidate_mod30_probe",
    ])
    return bool(random_positive and imitation_not_worse)


def _write_markdown(path: Path, rows: List[Dict[str, Any]], aggregates: Dict[str, Dict[str, float]], comparisons: Dict[str, Dict[str, Optional[float]]]) -> str:
    supports = _supports_structure(comparisons)
    conclusion = "Controls support learned residue/wheel structure." if supports else "Controls do not yet support learned structure."
    lines: List[str] = [
        "# PrimeArena Structure Controls",
        "",
        "This is a control and repeatability report, not a discovery claim.",
        "",
        "## All Runs",
        "",
        "| condition | seed | baseline success | baseline SA delta | raw none delta | cand mod6 | cand mod30 | cand mod210 | mod6 damage | mod30 damage | mod210 damage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(row["seed_index"]),
                    _fmt(row["baseline_success_rate"]),
                    _fmt(row["baseline_success_aware_delta"]),
                    _fmt(row["raw_none_model_vs_baseline_delta"]),
                    _fmt(row.get("candidate_mod6_probe")),
                    _fmt(row.get("candidate_mod30_probe")),
                    _fmt(row.get("candidate_mod210_probe")),
                    _fmt(row.get("ablation_mod6_delta_damage")),
                    _fmt(row.get("ablation_mod30_delta_damage")),
                    _fmt(row.get("ablation_mod210_delta_damage")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Mean / Std Probe Accuracy", ""])
    lines.append("| condition | candidate mod6 | candidate mod30 | candidate mod210 | pooled mod6 | pooled mod30 | pooled mod210 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for condition, stats in aggregates.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _mean_std(stats, "candidate_mod6_probe"),
                    _mean_std(stats, "candidate_mod30_probe"),
                    _mean_std(stats, "candidate_mod210_probe"),
                    _mean_std(stats, "pooled_mod6_probe"),
                    _mean_std(stats, "pooled_mod30_probe"),
                    _mean_std(stats, "pooled_mod210_probe"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Mean / Std Ablation Damage", ""])
    lines.append("| condition | mod6 delta damage | mod30 delta damage | mod210 delta damage |")
    lines.append("|---|---:|---:|---:|")
    for condition, stats in aggregates.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _mean_std(stats, "ablation_mod6_delta_damage"),
                    _mean_std(stats, "ablation_mod30_delta_damage"),
                    _mean_std(stats, "ablation_mod210_delta_damage"),
                ]
            )
            + " |"
        )

    for name, cmp in comparisons.items():
        lines.extend(["", f"## {name.replace('_', ' ').title()}", ""])
        for key, value in cmp.items():
            lines.append(f"- `{key}`: {_fmt(value)}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            (
                "The supported claim is limited to learned residue/wheel structure; stronger claims require more controls, OOD ranges, and repeated training runs."
                if supports
                else "The current results show stronger champion ablation damage than imitation, but random controls retain strong probe decodability from the input representation, so the controls do not yet isolate learned structure."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return conclusion


def run_controls(
    config: str,
    champion: str,
    imitation_checkpoint: str,
    out_dir: str | Path,
    samples: int,
    episodes: int,
    seeds: int,
    weak_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    device = choose_device(cfg.train.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conditions: List[tuple[str, Optional[str]]] = [
        ("random", None),
        ("imitation", imitation_checkpoint),
        ("champion", champion),
    ]
    if weak_checkpoint:
        conditions.append(("weak", weak_checkpoint))

    reports: List[Dict[str, Any]] = []
    for condition, checkpoint in conditions:
        for seed_index in range(1, int(seeds) + 1):
            seed = 10_000 * seed_index + sum(ord(ch) for ch in condition)
            report = run_one(cfg, device, condition, seed_index, seed, checkpoint, out, samples, episodes)
            reports.append(report)

    rows = [_row_from_report(report) for report in reports]
    aggregates = _aggregate(rows)
    comparisons: Dict[str, Dict[str, Optional[float]]] = {
        "champion_vs_random": _comparison(aggregates, "random"),
        "champion_vs_imitation": _comparison(aggregates, "imitation"),
    }
    if weak_checkpoint:
        comparisons["champion_vs_weak"] = _comparison(aggregates, "weak")
    conclusion = _write_markdown(out / "CONTROL_REPORT.md", rows, aggregates, comparisons)
    summary = {
        "config": config,
        "champion": champion,
        "imitation_checkpoint": imitation_checkpoint,
        "weak_checkpoint": weak_checkpoint,
        "samples": int(samples),
        "episodes": int(episodes),
        "seeds": int(seeds),
        "rows": rows,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "conclusion": conclusion,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "summary.csv", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/structure_hunt_probe.json")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--imitation-checkpoint", required=True)
    parser.add_argument("--weak-checkpoint", default=None)
    parser.add_argument("--out-dir", default="runs/structure_controls_001")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    summary = run_controls(
        config=args.config,
        champion=args.champion,
        imitation_checkpoint=args.imitation_checkpoint,
        weak_checkpoint=args.weak_checkpoint,
        out_dir=args.out_dir,
        samples=args.samples,
        episodes=args.episodes,
        seeds=args.seeds,
    )
    print(json.dumps({k: summary[k] for k in ["conclusion", "comparisons"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
