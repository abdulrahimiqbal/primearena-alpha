from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


PHASES = [
    {
        "phase": "1. Infrastructure",
        "status": "Validated",
        "result": "Local environment, oracle, episode traces, replay, checkpoints, dashboard, and Modal stack were exercised.",
    },
    {
        "phase": "2. Safe Learning",
        "status": "Validated",
        "result": "Success-aware promotion and certified-guess evaluation prevent cheap failed policies from becoming champions.",
    },
    {
        "phase": "3. Structure Readiness",
        "status": "Validated",
        "result": "A champion checkpoint reached safe eval/readiness gates and enabled first interpretability runs.",
    },
    {
        "phase": "4. Residual Rank",
        "status": "Negative Control",
        "result": "Hard residual ranking did not beat trained input-only / nearest controls; signal audit indicated current visible features were information-limited.",
    },
    {
        "phase": "5. PrimeLead Calibration",
        "status": "Validated",
        "result": "Known q=10 consecutive-prime residue-pair bias was recovered and weakened after the null absorbed pair frequencies.",
    },
    {
        "phase": "6. PrimeLead Null Ladder",
        "status": "Blocked",
        "result": "The full null ladder completed but flagged likely leakage/bug in gap_hist permutation control, so promoted leads are not scientifically valid yet.",
    },
]


NEXT_ACTIONS = [
    {
        "priority": "P0",
        "task": "Audit PrimeLead gap_hist leakage",
        "detail": "Permutation baseline AUC reached 0.6867 for gap_hist. Inspect train/eval split construction, label permutation, and null metadata leakage.",
    },
    {
        "priority": "P0",
        "task": "Fix strong-null generators",
        "detail": "gap_hist, residue_pair, and block_bootstrap should not leave trivial residue/density channels that produce near-perfect AUC.",
    },
    {
        "priority": "P1",
        "task": "Rerun targeted null checks",
        "detail": "Run small gap_hist/residue_pair/block_bootstrap diagnostics before rerunning the full 50k ladder.",
    },
    {
        "priority": "P1",
        "task": "Rerun full ladder",
        "detail": "Only after permutation controls stay near chance should full-ladder promoted leads or no-lead results be trusted.",
    },
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(rows)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return pd.DataFrame(rows)


def repo_remote() -> str:
    try:
        out = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
        return out
    except Exception:
        return "Not configured"


def status_class(status: str) -> str:
    key = status.lower()
    if "validated" in key:
        return "ok"
    if "blocked" in key or "bug" in key:
        return "bad"
    if "negative" in key:
        return "warn"
    return "neutral"


def metric_card(label: str, value: Any, note: str = "", state: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card {state}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, body: str, state: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="section-card {state}">
          <div class="section-title">{title}</div>
          <div class="section-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_page() -> None:
    st.set_page_config(page_title="PrimeArena Project Status", layout="wide")
    st.markdown(
        """
        <style>
        :root {
          --bg: #f7f8fb;
          --ink: #151923;
          --muted: #5b6475;
          --line: #dce1ea;
          --ok: #0f766e;
          --warn: #a16207;
          --bad: #b42318;
          --panel: #ffffff;
        }
        .block-container {
          padding-top: 1.4rem;
          max-width: 1420px;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        .topline {
          color: var(--muted);
          font-size: 0.95rem;
          margin-top: -0.4rem;
          margin-bottom: 1.2rem;
        }
        .metric-card, .section-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-left-width: 5px;
          border-radius: 8px;
          padding: 14px 15px;
          min-height: 112px;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .metric-card.ok, .section-card.ok { border-left-color: var(--ok); }
        .metric-card.warn, .section-card.warn { border-left-color: var(--warn); }
        .metric-card.bad, .section-card.bad { border-left-color: var(--bad); }
        .metric-card.neutral, .section-card.neutral { border-left-color: #506690; }
        .metric-label, .section-title {
          color: var(--muted);
          font-size: 0.78rem;
          text-transform: uppercase;
          font-weight: 700;
          letter-spacing: 0.04em;
        }
        .metric-value {
          color: var(--ink);
          font-size: 1.55rem;
          font-weight: 750;
          line-height: 1.15;
          margin-top: 8px;
        }
        .metric-note, .section-body {
          color: var(--muted);
          font-size: 0.9rem;
          line-height: 1.35;
          margin-top: 8px;
        }
        .phase-row {
          display: grid;
          grid-template-columns: 210px 130px 1fr;
          gap: 12px;
          align-items: stretch;
          margin-bottom: 10px;
        }
        .phase-cell {
          background: white;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px;
          min-height: 74px;
        }
        .phase-status {
          font-weight: 750;
        }
        .phase-status.ok { color: var(--ok); }
        .phase-status.warn { color: var(--warn); }
        .phase-status.bad { color: var(--bad); }
        .code-link {
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          color: #334155;
          font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def aggregate_calibration(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for q, data in dict(summary.get("checks_by_q", {})).items():
        rows.append(
            {
                "q": q,
                "weak_iid_auc": data.get("iid_auc"),
                "weak_iid_ood_auc": data.get("iid_ood_auc"),
                "weak_iid_bits_saved": data.get("iid_bits_saved"),
                "weak_iid_promoted": data.get("iid_promoted_count"),
                "pair_matched_auc": data.get("matched_auc"),
                "auc_drop": data.get("auc_drop"),
                "pair_matched_promoted": data.get("matched_promoted_count"),
                "passes": data.get("passes"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_ladder(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for null, stats in dict(summary.get("aggregates", {})).items():
        rows.append(
            {
                "null_model": null,
                "auc": stats.get("real_vs_fake_AUC_mean"),
                "ood_auc": stats.get("OOD_AUC_mean"),
                "bits_saved": stats.get("heldout_bits_saved_mean"),
                "permutation_auc": stats.get("permutation_test_AUC_mean"),
                "permutation_delta": stats.get("permutation_auc_delta_mean"),
                "accuracy": stats.get("real_vs_fake_accuracy_mean"),
            }
        )
    order = ["cramer", "wheel", "gap_hist", "residue_pair", "block_bootstrap"]
    if rows:
        df = pd.DataFrame(rows)
        df["order"] = df["null_model"].apply(lambda x: order.index(x) if x in order else 99)
        return df.sort_values("order").drop(columns=["order"])
    return pd.DataFrame(rows)


def promoted_cards(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for card in summary.get("lead_cards", []):
        if not card.get("promoted"):
            continue
        effect = card.get("effect_size", {})
        rows.append(
            {
                "lead_id": card.get("lead_id"),
                "null_model": card.get("tested_null_model"),
                "feature": card.get("feature_statistic"),
                "auc": effect.get("AUC"),
                "ood_auc": effect.get("OOD_AUC"),
                "bits_saved": card.get("bits_saved"),
                "complexity": card.get("complexity_score"),
            }
        )
    return pd.DataFrame(rows)


def show_overview(runs_dir: Path, calibration: Dict[str, Any], ladder: Dict[str, Any]) -> None:
    st.title("PrimeArena Research Status")
    st.markdown(
        "<div class='topline'>Project-level dashboard for the PrimeArena, ResidualRank, and PrimeLead research stack.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("Current scientific status", "Blocked", "Full PrimeLead ladder flagged likely leakage in gap_hist permutation control.", "bad")
    with cols[1]:
        metric_card("Detector calibration", "Passed", "q=10 residue-pair bias recovered across 3 seeds and weakened under matched null.", "ok")
    with cols[2]:
        metric_card("Champion safety", "Validated", "Success-aware eval and certified guesses prevent cheap-failure promotion.", "ok")
    with cols[3]:
        metric_card("GitHub remote", "Private", repo_remote(), "neutral")

    st.subheader("Phase Timeline")
    for phase in PHASES:
        cls = status_class(phase["status"])
        st.markdown(
            f"""
            <div class="phase-row">
              <div class="phase-cell"><strong>{phase['phase']}</strong></div>
              <div class="phase-cell phase-status {cls}">{phase['status']}</div>
              <div class="phase-cell">{phase['result']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Evidence Snapshot")
    c1, c2, c3 = st.columns(3)
    cal_decision = calibration.get("decision", "Missing")
    ladder_decision = ladder.get("decision", "Missing")
    with c1:
        metric_card("Calibration decision", "Passed" if "passed" in cal_decision else "Missing", cal_decision, "ok" if "passed" in cal_decision else "warn")
    with c2:
        metric_card("Null ladder decision", "Leakage flagged" if "bug" in ladder_decision or "leakage" in ladder_decision else "Unknown", ladder_decision, "bad" if ladder_decision else "warn")
    with c3:
        run_count = len([p for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.exists() else 0
        metric_card("Local run artifacts", run_count, f"Loaded from {runs_dir}", "neutral")


def show_calibration(calibration: Dict[str, Any]) -> None:
    st.header("PrimeLead Calibration")
    if not calibration:
        st.warning("No calibration summary found.")
        return
    st.write(calibration.get("decision", "No decision recorded."))
    df = aggregate_calibration(calibration)
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        chart = df.set_index("q")[["weak_iid_auc", "pair_matched_auc", "weak_iid_ood_auc"]]
        st.bar_chart(chart)
    cards = promoted_cards(calibration)
    if not cards.empty:
        st.subheader("Promoted Calibration Leads")
        st.dataframe(cards, width="stretch", hide_index=True)
    section_card(
        "Interpretation",
        "The detector recovered the known q=10 consecutive residue-pair signal. This validates the real-vs-fake machinery before using it for exploratory null-ladder runs.",
        "ok",
    )


def show_null_ladder(ladder: Dict[str, Any]) -> None:
    st.header("PrimeLead Null Ladder")
    if not ladder:
        st.warning("No null-ladder summary found.")
        return
    decision = str(ladder.get("decision", "No decision recorded."))
    state = "bad" if "bug" in decision or "leakage" in decision else "ok"
    section_card("Decision", decision, state)
    df = aggregate_ladder(ladder)
    if not df.empty:
        st.subheader("Null Model Metrics")
        st.dataframe(df, width="stretch", hide_index=True)
        chart = df.set_index("null_model")[["auc", "ood_auc", "permutation_auc"]]
        st.line_chart(chart)

    rows_df = pd.DataFrame(ladder.get("rows", []))
    if not rows_df.empty:
        st.subheader("Seed-Level Diagnostics")
        cols = [
            "null_model",
            "seed_index",
            "real_vs_fake_AUC",
            "OOD_AUC",
            "heldout_bits_saved",
            "permutation_test_AUC",
            "permutation_auc_delta",
            "top_feature",
        ]
        st.dataframe(rows_df[[c for c in cols if c in rows_df.columns]], width="stretch", hide_index=True)

    bad_rows = pd.DataFrame()
    if not rows_df.empty and "permutation_test_AUC" in rows_df.columns:
        bad_rows = rows_df[rows_df["permutation_test_AUC"] >= 0.60]
    if not bad_rows.empty:
        st.subheader("Leakage Trigger")
        st.dataframe(bad_rows, width="stretch", hide_index=True)
        section_card(
            "Why this blocks science claims",
            "A shuffled-label permutation control should be near chance. AUC above 0.60 means the dataset/control path contains a non-scientific separability channel or evaluation bug.",
            "bad",
        )

    cards = promoted_cards(ladder)
    if not cards.empty:
        st.subheader("Promoted Cards")
        st.caption("These are shown for debugging only while the leakage warning is active.")
        st.dataframe(cards, width="stretch", hide_index=True)


def show_learning_stack(runs_dir: Path) -> None:
    st.header("Learning Stack Evidence")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()] if runs_dir.exists() else [], key=lambda p: p.name)
    metric_runs = []
    for p in run_dirs:
        df = load_jsonl(p / "metrics.jsonl")
        if df.empty:
            continue
        latest = df.iloc[-1].to_dict()
        metric_runs.append(
            {
                "run": p.name,
                "steps": latest.get("step"),
                "eval_success_rate": latest.get("eval_success_rate"),
                "success_aware_delta": latest.get("eval_success_aware_model_vs_baseline_delta"),
                "wrong_guess_count": latest.get("eval_wrong_guess_count"),
                "replay_size": latest.get("replay_size"),
                "checkpoint_eligible": latest.get("eval_checkpoint_eligible"),
            }
        )
    if metric_runs:
        st.dataframe(pd.DataFrame(metric_runs), width="stretch", hide_index=True)
    else:
        st.info("No metrics.jsonl runs found.")

    st.subheader("What Was Proven")
    for title, body, state in [
        ("Environment correctness", "Prime oracle, candidate windows, actions, reward/cost accounting, and episode traces were validated by smoke tests.", "ok"),
        ("Distributed execution", "Modal workers, replay shards, learner checkpoints, AMP/CUDA training, bounded self-play, and worker progress all ran end to end.", "ok"),
        ("Evaluation safety", "Success-aware deltas and certified guess masks prevent wrong early guesses from being rewarded.", "ok"),
        ("Residual controls", "Residual hard ranking did not demonstrate signal beyond trained input-only and nearest baselines.", "warn"),
    ]:
        section_card(title, body, state)


def show_next_actions() -> None:
    st.header("Next Actions")
    st.write("The current blocker is methodological, not infrastructure. Fix the leakage/control path before trusting any PrimeLead lead card.")
    for item in NEXT_ACTIONS:
        section_card(f"{item['priority']} - {item['task']}", item["detail"], "bad" if item["priority"] == "P0" else "warn")
    st.subheader("Next Command After Fix")
    st.code(
        ".venv/bin/python -m primearena.primelead "
        "--config configs/primelead_null_ladder.json "
        "--out-dir runs/primelead_null_ladder_002 "
        "--samples 50000 "
        "--n-min 100000 "
        "--n-max 100000000 "
        "--nulls cramer wheel gap_hist residue_pair block_bootstrap "
        "--measurement-budget 10 "
        "--seeds 3",
        language="bash",
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args, _ = parser.parse_known_args()
    runs_dir = Path(args.runs_dir)

    setup_page()
    calibration = load_json(runs_dir / "primelead_calibration_001" / "summary.json")
    ladder = load_json(runs_dir / "primelead_null_ladder_001" / "summary.json")

    tabs = st.tabs(["Overview", "Calibration", "Null Ladder", "Learning Stack", "Next Actions"])
    with tabs[0]:
        show_overview(runs_dir, calibration, ladder)
    with tabs[1]:
        show_calibration(calibration)
    with tabs[2]:
        show_null_ladder(ladder)
    with tabs[3]:
        show_learning_stack(runs_dir)
    with tabs[4]:
        show_next_actions()


if __name__ == "__main__":
    main()
