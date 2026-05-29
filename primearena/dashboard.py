from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


DEFAULT_METRICS = [
    "episode_reward",
    "success_rate",
    "avg_cost",
    "avg_baseline_cost",
    "model_vs_baseline_delta",
    "eval_success_rate",
    "eval_model_vs_baseline_delta",
    "eval_success_aware_model_vs_baseline_delta",
    "eval_success_aware_baseline_cost",
    "eval_success_aware_avg_cost",
    "eval_wrong_guess_count",
    "eval_premature_guess_blocked_count",
    "eval_certified_guess_count",
    "eval_uncertified_guess_attempt_count",
    "eval_certified_guesses_only",
    "best_eval_model_vs_baseline_delta",
    "best_eval_success_aware_model_vs_baseline_delta",
    "best_eval_success_rate",
    "best_promoted_eval_success_aware_model_vs_baseline_delta",
    "best_promoted_eval_success_rate",
    "eval_success_checkpoint_eligible",
    "eval_checkpoint_eligible",
    "checkpoint_promotion_success_gate",
    "checkpoint_promotion_delta_gate",
    "checkpoint_promotion_wrong_guess_gate",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "lr",
    "replay_size",
    "generated_items",
    "curriculum_hard_cases",
    "batched_mcts_inference",
    "model_architecture",
]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "metrics.jsonl"
    rows: List[Dict[str, object]] = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_worker_progress(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    progress_dir = run_dir / "worker_progress"
    if not progress_dir.exists():
        return pd.DataFrame(rows)
    for p in sorted(progress_dir.glob("worker_*.latest.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_structure_readiness(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "structure_readiness.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_residual_structure(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "residual_structure.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def find_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        [
            p
            for p in runs_dir.iterdir()
            if p.is_dir()
            and (
                (p / "metrics.jsonl").exists()
                or (p / "structure_readiness.json").exists()
                or (p / "residual_structure.json").exists()
            )
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def show_readiness(readiness: Dict[str, object]) -> None:
    st.subheader("Readiness")
    rcols = st.columns(5)
    readiness_keys = [
        "ready_for_structure_hunt",
        "in_domain_eval_success_rate",
        "ood_eval_success_rate",
        "in_domain_success_aware_delta",
        "ood_success_aware_delta",
    ]
    for col, key in zip(rcols, readiness_keys):
        val = readiness.get(key, "—")
        if isinstance(val, float):
            col.metric(key, f"{val:.4f}")
        else:
            col.metric(key, val)
    safety_cols = st.columns(4)
    for col, key in zip(
        safety_cols,
        [
            "wrong_guess_count",
            "premature_guess_blocked_count",
            "certified_guess_count",
            "uncertified_guess_attempt_count",
        ],
    ):
        col.metric(key, readiness.get(key, "—"))
    probe_cols = st.columns(3)
    for col, key in zip(probe_cols, ["probe_mod6_accuracy", "probe_mod30_accuracy", "probe_mod210_accuracy"]):
        val = readiness.get(key, "—")
        col.metric(key, f"{val:.4f}" if isinstance(val, float) else val)
    ablation_cols = st.columns(3)
    for col, key in zip(ablation_cols, ["ablation_mod6_delta", "ablation_mod30_delta", "ablation_mod210_delta"]):
        val = readiness.get(key, "—")
        col.metric(key, f"{val:.4f}" if isinstance(val, float) else val)
    reasons = readiness.get("failure_reasons", [])
    if reasons:
        st.error("\n".join(map(str, reasons)))


def show_residual_structure(report: Dict[str, object]) -> None:
    st.subheader("Residual Structure")
    cols = st.columns(4)
    for col, key in zip(cols, ["ready_for_residual_structure_hunt", "samples", "episodes", "seeds"]):
        col.metric(key, report.get(key, "—"))
    rows = []
    for bound, result in dict(report.get("results_by_sieve_bound", {})).items():
        if isinstance(result, dict):
            rows.append(
                {
                    "sieve_bound": bound,
                    "success_rate": result.get("success_rate"),
                    "delta_vs_nearest": result.get("success_aware_delta_vs_nearest"),
                    "delta_vs_random": result.get("success_aware_delta_vs_random"),
                    "avg_survivor_tests": result.get("avg_survivor_tests"),
                    "matched_counterfactual_accuracy": result.get("matched_counterfactual_accuracy"),
                    "hidden_patching_effect_size": result.get("hidden_patching_effect_size"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    reasons = report.get("failure_reasons", [])
    if reasons:
        st.error("\n".join(map(str, reasons)))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="PrimeArena Runs", layout="wide")
    st.title("PrimeArena Run Monitor")
    st.caption("Reads metrics.jsonl from each run directory, including distributed-shard, curriculum, batched-MCTS, and checkpoint-promotion signals.")

    runs_dir = Path(args.runs_dir)
    runs = find_runs(runs_dir)
    if not runs:
        st.warning(f"No runs found in `{runs_dir}` yet. Start one with `python -m primearena.train --config configs/smoke.json --run-name local_smoke`.")
        return

    run_labels = [p.name for p in runs]
    selected_label = st.sidebar.selectbox("Run", run_labels, index=0)
    run_dir = runs[run_labels.index(selected_label)]
    df = load_metrics(run_dir)
    readiness = load_structure_readiness(run_dir)
    residual_structure = load_residual_structure(run_dir)
    if df.empty:
        if readiness:
            st.sidebar.markdown(f"**Run directory**: `{run_dir}`")
            show_readiness(readiness)
            st.subheader("Readiness report")
            st.json(readiness)
            return
        if residual_structure:
            st.sidebar.markdown(f"**Run directory**: `{run_dir}`")
            show_residual_structure(residual_structure)
            st.subheader("Residual structure report")
            st.json(residual_structure)
            return
        st.warning("Selected run has no metrics yet.")
        return

    st.sidebar.markdown(f"**Run directory**: `{run_dir}`")
    latest = df.iloc[-1].to_dict()

    cols = st.columns(6)
    for col, key in zip(cols, ["step", "success_rate", "avg_cost", "eval_success_rate", "eval_success_aware_model_vs_baseline_delta", "replay_size"]):
        val = latest.get(key, "—")
        if isinstance(val, float):
            col.metric(key, f"{val:.4f}")
        else:
            col.metric(key, val)

    metric_options = [m for m in DEFAULT_METRICS if m in df.columns]
    extra = [c for c in df.columns if c not in metric_options and pd.api.types.is_numeric_dtype(df[c])]
    selected_metrics = st.multiselect("Metrics", metric_options + extra, default=metric_options[:5])

    for metric in selected_metrics:
        if metric in df.columns:
            st.subheader(metric)
            plot_df = df[["step", metric]].dropna() if "step" in df.columns else df[[metric]].dropna()
            st.line_chart(plot_df, x="step" if "step" in plot_df.columns else None, y=metric)

    st.subheader("Latest metrics")
    display = {k: v for k, v in latest.items() if k not in {"last_trace", "eval_failed_traces"}}
    st.json(display)

    if readiness:
        show_readiness(readiness)
    if residual_structure:
        show_residual_structure(residual_structure)

    worker_df = load_worker_progress(run_dir)
    if not worker_df.empty:
        st.subheader("Distributed worker progress")
        preferred = [
            "worker_id",
            "event",
            "episodes_done",
            "episodes_total",
            "success_rate",
            "items",
            "items_per_second",
            "elapsed_sec",
            "device",
            "error_type",
            "error",
        ]
        cols = [c for c in preferred if c in worker_df.columns]
        rest = [c for c in worker_df.columns if c not in cols]
        st.dataframe(worker_df[cols + rest].sort_values("worker_id"), use_container_width=True)

    st.subheader("Recent episode trace")
    trace = latest.get("last_trace", [])
    if isinstance(trace, str):
        try:
            trace = json.loads(trace)
        except Exception:
            trace = [trace]
    if trace:
        st.code("\n".join(map(str, trace)))
    else:
        st.info("No trace logged yet.")

    st.subheader("Raw metrics table")
    st.dataframe(df.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
