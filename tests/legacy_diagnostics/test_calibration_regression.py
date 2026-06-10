import json, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
import pytest

OUT = Path("runs/_calib_regression")

CAL_ARGS = [
    sys.executable, "-m", "primearena.primelead",
    "--config", "configs/primelead_null_ladder.json",
    "--out-dir", str(OUT),
    "--samples", "5000",
    "--n-min", "100000",
    "--n-max", "10000000",
    "--nulls", "wheel", "residue_pair",
    "--measurement-budget", "10",
    "--seeds", "2",
]


@dataclass(frozen=True)
class CalibrationResults:
    weak_null_auc: float
    pair_matched_auc: float
    promoted_pair_leads_weak: int
    promoted_pair_leads_matched: int


def _is_q10_pair_feature(value) -> bool:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    pair_markers = (
        "pair_residue_mod10",
        "consecutive_residue_pair_counts_mod10",
        "consecutive_residue_pair_transition_mod10",
        "diagonal_vs_offdiagonal_pair_rate_mod10",
        "same_residue_repeat_rate_mod10",
        "pair_bias_spectrum_mod10",
    )
    return any(marker in text for marker in pair_markers)


def _mean_auc(rows, null_model: str) -> float:
    aucs = [float(row["real_vs_fake_AUC"]) for row in rows if row.get("null_model") == null_model]
    if not aucs:
        raise AssertionError(f"No calibration rows found for null model {null_model!r}")
    return sum(aucs) / len(aucs)


def _promoted_q10_pair_cards(cards, null_model: str) -> int:
    return sum(
        1
        for card in cards
        if card.get("tested_null_model") == null_model
        and bool(card.get("promoted"))
        and _is_q10_pair_feature(
            {
                "feature_statistic": card.get("feature_statistic"),
                "top_features": card.get("top_features", []),
                "next_null_model_to_test": card.get("next_null_model_to_test", {}),
            }
        )
    )


def load_calibration_results(out_dir: Path):
    """Parse PrimeLead summary.json into the calibration contract fields."""
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise AssertionError(f"Missing PrimeLead summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    rows = summary.get("rows", [])
    cards = summary.get("lead_cards", [])
    return CalibrationResults(
        weak_null_auc=_mean_auc(rows, "wheel"),
        pair_matched_auc=_mean_auc(rows, "residue_pair"),
        promoted_pair_leads_weak=_promoted_q10_pair_cards(cards, "wheel"),
        promoted_pair_leads_matched=_promoted_q10_pair_cards(cards, "residue_pair"),
    )


@pytest.mark.slow
def test_los_calibration_q10():
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run(CAL_ARGS, check=True, timeout=1800)
    r = load_calibration_results(OUT)
    assert r.weak_null_auc >= 0.62
    assert r.pair_matched_auc <= 0.62
    assert r.weak_null_auc - r.pair_matched_auc >= 0.04
    assert r.promoted_pair_leads_weak >= 1
    assert r.promoted_pair_leads_matched == 0
