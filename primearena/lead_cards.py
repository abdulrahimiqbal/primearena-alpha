from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def make_lead_card(result: Dict[str, Any], promoted: bool, failure_reasons: Iterable[str], next_null: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    null_model = str(result["null_model"])
    seed = int(result["seed"])
    top = result.get("top_features", [])
    top_expr = top[0]["feature"] if top else "no_selected_feature"
    return {
        "lead_id": f"primelead_{null_model}_seed_{seed}",
        "lead_name": f"{null_model} discriminator statistic",
        "feature_statistic": top_expr,
        "feature_formula": top_expr,
        "null_models_beaten": [null_model] if promoted else [],
        "tested_null_model": null_model,
        "effect_size": {
            "AUC": float(metrics.get("real_vs_fake_AUC", 0.5)),
            "accuracy": float(metrics.get("real_vs_fake_accuracy", 0.5)),
            "heldout_bits_saved": float(metrics.get("heldout_bits_saved", 0.0)),
            "OOD_AUC": float(metrics.get("OOD_AUC", 0.5)),
            "permutation_auc_delta": float(metrics.get("permutation_auc_delta", 0.0)),
        },
        "confidence_interval": {
            "AUC_low": float(metrics.get("auc_ci_low", 0.5)),
            "AUC_high": float(metrics.get("auc_ci_high", 0.5)),
        },
        "scale_range_tested": result.get("n_range"),
        "OOD_result": {
            "n_range": result.get("ood_n_range"),
            "AUC": float(metrics.get("OOD_AUC", 0.5)),
        },
        "bits_saved": float(metrics.get("heldout_bits_saved", 0.0)),
        "complexity_score": float(metrics.get("feature_complexity", 0.0)),
        "measurement_cost": float(metrics.get("measurement_cost", 0.0)),
        "top_features": top,
        "failure_cases": list(failure_reasons),
        "promoted": bool(promoted),
        "interpretation": "Candidate real-vs-fake statistic. This is a lead card, not a theorem or discovery claim.",
        "next_null_model_to_test": next_null,
    }


def write_lead_card(card: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lead_id = str(card["lead_id"])
    (out_dir / f"{lead_id}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {card['lead_name']}",
        "",
        f"Lead ID: `{lead_id}`",
        f"Promoted: **{card['promoted']}**",
        "",
        "## Statistic",
        "",
        f"- Feature/statistic: `{card['feature_statistic']}`",
        f"- Formula: `{card['feature_formula']}`",
        f"- Complexity score: `{card['complexity_score']:.4f}`",
        f"- Measurement cost: `{card['measurement_cost']:.4f}`",
        "",
        "## Evidence",
        "",
        f"- Null model tested: `{card['tested_null_model']}`",
        f"- AUC: `{card['effect_size']['AUC']:.4f}`",
        f"- OOD AUC: `{card['effect_size']['OOD_AUC']:.4f}`",
        f"- Held-out bits saved: `{card['bits_saved']:.6f}`",
        f"- Permutation AUC delta: `{card['effect_size']['permutation_auc_delta']:.4f}`",
        f"- AUC 95% CI: `[{card['confidence_interval']['AUC_low']:.4f}, {card['confidence_interval']['AUC_high']:.4f}]`",
        "",
        "## Top Features",
        "",
    ]
    for feat in card.get("top_features", [])[:8]:
        lines.append(f"- `{feat['feature']}` weight `{feat['weight']:.4f}`")
    lines.extend(["", "## Failure Cases", ""])
    failures = card.get("failure_cases", [])
    if failures:
        for reason in failures:
            lines.append(f"- {reason}")
    else:
        lines.append("- None under current gates.")
    lines.extend(
        [
            "",
            "## Next Null",
            "",
            f"- Model: `{card['next_null_model_to_test'].get('next_null_model')}`",
            f"- Refinement: {card['next_null_model_to_test'].get('refinement')}",
            "",
            "No discovery claim is made.",
        ]
    )
    (out_dir / f"{lead_id}.md").write_text("\n".join(lines), encoding="utf-8")


def write_cards(cards: List[Dict[str, Any]], out_dir: Path) -> None:
    for card in cards:
        write_lead_card(card, out_dir)
