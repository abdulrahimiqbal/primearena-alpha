from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import load_config
from .env import PrimeArena
from .evaluate import evaluate
from .expert import rollout_expert
from .oracle import is_prime, next_prime


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def read_latest_metrics(run_dir: Optional[str | Path]) -> Dict[str, Any]:
    rows = read_metric_rows(run_dir)
    latest: Dict[str, Any] = {}
    for row in rows:
        latest.update(row)
    return latest


def read_metric_rows(run_dir: Optional[str | Path]) -> List[Dict[str, Any]]:
    if not run_dir:
        return []
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    lines = [line for line in p.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def check_oracle() -> Dict[str, Any]:
    known = {
        0: 2,
        1: 2,
        2: 3,
        3: 5,
        10: 11,
        11: 13,
        100: 101,
        1_000: 1009,
        10_000: 10007,
    }
    failures = []
    for n, p in known.items():
        got = next_prime(n)
        if got != p or not is_prime(got):
            failures.append({"n": n, "expected": p, "got": got})
    return {"passed": len(failures) == 0, "failures": failures}


def check_environment(cfg, episodes: int = 50) -> Dict[str, Any]:
    rng = np.random.default_rng(123)
    failures = 0
    success = []
    for _ in range(episodes):
        env = PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        rollout_expert(env)
        failures += int(not is_prime(env.true_next_prime))
        success.append(float(env.success))
    return {"passed": failures == 0 and float(np.mean(success)) >= 0.99, "oracle_failures": failures, "expert_success_rate": float(np.mean(success))}


def threshold_check(name: str, value: float, op: str, threshold: float) -> Dict[str, Any]:
    if op == ">=":
        passed = value >= threshold
    elif op == ">":
        passed = value > threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == "<":
        passed = value < threshold
    else:
        raise ValueError(f"Unknown threshold op: {op}")
    return {"name": name, "value": value, "op": op, "threshold": threshold, "passed": bool(passed)}


def boolean_check(name: str, value: object, expected: bool = True) -> Dict[str, Any]:
    passed = bool(value) is bool(expected)
    return {"name": name, "value": bool(value), "expected": bool(expected), "passed": bool(passed)}


def file_check(name: str, path: Path, should_exist: bool = True) -> Dict[str, Any]:
    exists = path.exists()
    return {"name": name, "path": str(path), "exists": bool(exists), "expected_exists": bool(should_exist), "passed": bool(exists is should_exist)}


def any_metric_true(rows: List[Dict[str, Any]], key: str) -> bool:
    return any(bool(row.get(key)) for row in rows)


def run_success_benchmark(config_path: str, benchmark_path: str, run_dir: Optional[str] = None, episodes: int = 100) -> Dict[str, Any]:
    cfg = load_config(config_path)
    bench = load_yaml(benchmark_path)
    thresholds = bench.get("thresholds", {})
    report: Dict[str, Any] = {
        "config": config_path,
        "benchmark": benchmark_path,
        "run_dir": run_dir,
        "checks": [],
    }

    oracle = check_oracle()
    report["checks"].append({"name": "oracle_known_values", **oracle})

    env_check = check_environment(cfg, episodes=min(episodes, 50))
    report["checks"].append({"name": "environment_expert_rollout", **env_check})

    eval_result = evaluate(cfg, episodes=episodes, seed=777)
    report["eval"] = eval_result

    if "wheel_vs_odd_delta_min" in thresholds:
        report["checks"].append(threshold_check("wheel_vs_odd_delta", eval_result["wheel_vs_odd_delta"], ">=", float(thresholds["wheel_vs_odd_delta_min"])))
    if "expert_success_rate_min" in thresholds:
        report["checks"].append(threshold_check("expert_success_rate", eval_result["expert_success_rate"], ">=", float(thresholds["expert_success_rate_min"])))

    metric_rows = read_metric_rows(run_dir)
    latest = {}
    for row in metric_rows:
        latest.update(row)
    report["latest_run_metrics"] = latest
    if latest:
        if "train_success_rate_min" in thresholds and "success_rate" in latest:
            report["checks"].append(threshold_check("latest_train_success_rate", float(latest["success_rate"]), ">=", float(thresholds["train_success_rate_min"])))
        if "eval_success_rate_min" in thresholds and "eval_success_rate" in latest:
            report["checks"].append(threshold_check("latest_eval_success_rate", float(latest["eval_success_rate"]), ">=", float(thresholds["eval_success_rate_min"])))
        if "model_vs_baseline_delta_min" in thresholds:
            if "eval_success_aware_model_vs_baseline_delta" in latest:
                report["checks"].append(
                    threshold_check(
                        "latest_eval_success_aware_model_vs_baseline_delta",
                        float(latest["eval_success_aware_model_vs_baseline_delta"]),
                        ">=",
                        float(thresholds["model_vs_baseline_delta_min"]),
                    )
                )
            elif "model_vs_baseline_delta" in latest:
                report["checks"].append(
                    threshold_check(
                        "latest_model_vs_baseline_delta",
                        float(latest["model_vs_baseline_delta"]),
                        ">=",
                        float(thresholds["model_vs_baseline_delta_min"]),
                    )
                )
        if "success_aware_model_vs_baseline_delta_min" in thresholds and "eval_success_aware_model_vs_baseline_delta" in latest:
            report["checks"].append(
                threshold_check(
                    "latest_eval_success_aware_model_vs_baseline_delta",
                    float(latest["eval_success_aware_model_vs_baseline_delta"]),
                    ">=",
                    float(thresholds["success_aware_model_vs_baseline_delta_min"]),
                )
            )
        if "eval_success_aware_model_vs_baseline_delta_min" in thresholds and "eval_success_aware_model_vs_baseline_delta" in latest:
            report["checks"].append(
                threshold_check(
                    "latest_eval_success_aware_model_vs_baseline_delta",
                    float(latest["eval_success_aware_model_vs_baseline_delta"]),
                    ">=",
                    float(thresholds["eval_success_aware_model_vs_baseline_delta_min"]),
                )
            )
        if "replay_size_min" in thresholds and "replay_size" in latest:
            report["checks"].append(threshold_check("latest_replay_size", float(latest["replay_size"]), ">", float(thresholds["replay_size_min"])))
        if "eval_wrong_guess_rate_max" in thresholds and "eval_wrong_guess_rate" in latest:
            report["checks"].append(threshold_check("latest_eval_wrong_guess_rate", float(latest["eval_wrong_guess_rate"]), "<=", float(thresholds["eval_wrong_guess_rate_max"])))

        requirements = bench.get("requirements", {})
        if requirements.get("eval_checkpoint_eligible") and "eval_checkpoint_eligible" in latest:
            report["checks"].append(boolean_check("latest_eval_checkpoint_eligible", latest["eval_checkpoint_eligible"], True))
        if requirements.get("checkpoint_promoted_success"):
            promoted = any_metric_true(metric_rows, "checkpoint_promoted_success") or float(latest.get("best_promoted_eval_success_rate", -1e18)) > -1e17
            report["checks"].append(boolean_check("ever_checkpoint_promoted_success", promoted, True))
        if requirements.get("checkpoint_promoted_delta"):
            promoted = any_metric_true(metric_rows, "checkpoint_promoted_delta") or float(latest.get("best_promoted_eval_success_aware_model_vs_baseline_delta", -1e18)) > -1e17
            report["checks"].append(boolean_check("ever_checkpoint_promoted_delta", promoted, True))
        if requirements.get("batched_mcts_inference_when_mcts") and str(latest.get("mode")) == "mcts":
            report["checks"].append(boolean_check("latest_batched_mcts_inference", latest.get("batched_mcts_inference"), True))
        if run_dir:
            run_path = Path(run_dir)
            if requirements.get("best_by_success_exists"):
                report["checks"].append(file_check("best_by_success_exists", run_path / "checkpoints" / "best_by_success.pt", True))
            if requirements.get("best_by_delta_exists"):
                report["checks"].append(file_check("best_by_delta_exists", run_path / "checkpoints" / "best_by_delta.pt", True))
            if requirements.get("best_by_delta_only_if_gate_passes"):
                delta_path = run_path / "checkpoints" / "best_by_delta.pt"
                delta_ok = not delta_path.exists() or any_metric_true(metric_rows, "checkpoint_promoted_delta")
                report["checks"].append(boolean_check("best_by_delta_only_if_gate_passes", delta_ok, True))

    report["passed"] = all(bool(c.get("passed")) for c in report["checks"])
    return report


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# PrimeArena Success Benchmark Report", "", f"Passed: **{report['passed']}**", "", "## Checks", ""]
    for c in report["checks"]:
        status = "PASS" if c.get("passed") else "FAIL"
        detail = ", ".join(f"{k}={v}" for k, v in c.items() if k not in {"name", "passed"})
        lines.append(f"- **{status}** `{c['name']}` {detail}")
    lines.append("\n## Eval Metrics\n")
    for k, v in sorted(report.get("eval", {}).items()):
        lines.append(f"- `{k}`: {v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.json")
    parser.add_argument("--benchmark", type=str, default="benchmarks/success_benchmark.yaml")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs/success_benchmark_report.json")
    args = parser.parse_args()
    report = run_success_benchmark(args.config, args.benchmark, args.run_dir, args.episodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, out.with_suffix(".md"))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
