from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _metric_rows(run_dir: Path) -> List[Dict[str, Any]]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _latest_metrics(run_dir: Path) -> Dict[str, Any]:
    latest: Dict[str, Any] = {}
    for row in _metric_rows(run_dir):
        latest.update(row)
    return latest


def _valid_run(run_dir: Path) -> Dict[str, Any]:
    rows = _metric_rows(run_dir)
    latest = _latest_metrics(run_dir)
    success_path = run_dir / "checkpoints" / "best_by_success.pt"
    delta_path = run_dir / "checkpoints" / "best_by_delta.pt"
    reasons: List[str] = []
    best_success = float(latest.get("best_promoted_eval_success_rate", -1e18))
    best_delta = float(latest.get("best_promoted_eval_success_aware_model_vs_baseline_delta", -1e18))
    delta_promoted = any(bool(row.get("checkpoint_promoted_delta", False)) for row in rows) or best_delta >= 0.0
    success_promoted = any(bool(row.get("checkpoint_promoted_success", False)) for row in rows) or best_success >= 0.80
    if not latest:
        reasons.append("metrics.jsonl missing or empty")
    if best_success < 0.80:
        reasons.append("best_promoted_eval_success_rate < 0.80")
    if best_delta < 0.0:
        reasons.append("best_promoted_eval_success_aware_model_vs_baseline_delta < 0.0")
    if not delta_promoted:
        reasons.append("no success-aware delta checkpoint promotion")
    if not success_promoted:
        reasons.append("no success checkpoint promotion")
    if not success_path.exists():
        reasons.append("best_by_success.pt missing")
    if not delta_path.exists():
        reasons.append("best_by_delta.pt missing")
    return {
        "run_dir": str(run_dir),
        "valid": not reasons,
        "failure_reasons": reasons,
        "latest_metrics": latest,
        "best_promoted_eval_success_rate": best_success,
        "best_promoted_eval_success_aware_model_vs_baseline_delta": best_delta,
        "best_by_success": str(success_path) if success_path.exists() else None,
        "best_by_delta": str(delta_path) if delta_path.exists() else None,
    }


def select_champion(runs_root: str | Path, imitation_run: str, mcts_run: Optional[str], out: str | Path) -> Dict[str, Any]:
    root = Path(runs_root)
    evaluated: List[Dict[str, Any]] = []
    if mcts_run:
        evaluated.append({"kind": "mcts", **_valid_run(root / mcts_run)})
    evaluated.append({"kind": "imitation", **_valid_run(root / imitation_run)})

    selected: Optional[Dict[str, Any]] = None
    for candidate in evaluated:
        if not candidate["valid"]:
            continue
        checkpoint = candidate.get("best_by_delta") or candidate.get("best_by_success")
        if checkpoint:
            selected = {**candidate, "checkpoint": checkpoint}
            break

    report: Dict[str, Any] = {
        "selected": selected is not None,
        "checkpoint": selected.get("checkpoint") if selected else None,
        "source_run": selected.get("run_dir") if selected else None,
        "source_kind": selected.get("kind") if selected else None,
        "evaluated_runs": evaluated,
        "failure_reasons": [] if selected else [f"{r['kind']}: {', '.join(r['failure_reasons'])}" for r in evaluated],
    }
    if selected:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--imitation-run", required=True)
    parser.add_argument("--mcts-run", default=None)
    parser.add_argument("--out", default="runs/champion.json")
    args = parser.parse_args()
    report = select_champion(args.runs_root, args.imitation_run, args.mcts_run, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["selected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
