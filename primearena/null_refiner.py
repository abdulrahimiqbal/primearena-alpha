from __future__ import annotations

from typing import Any, Dict


def suggest_refined_null(lead: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest the next fake-prime model that should absorb a lead.

    This is intentionally conservative: it does not mutate generators in place.
    It records the next null-world pressure that would make the lead less cheap.
    """

    features = " ".join(str(x.get("feature", "")) for x in lead.get("top_features", []))
    current = str(lead.get("null_model", "unknown"))
    if "pair_residue" in features or "residue_pair" in features:
        return {
            "next_null_model": "residue_pair",
            "refinement": "match consecutive marked residue-pair frequencies for the strongest q values",
            "absorbs": "residue-pair bias",
        }
    if "gap" in features:
        return {
            "next_null_model": "gap_hist",
            "refinement": "match empirical gap histograms within n-scale buckets",
            "absorbs": "gap distribution and first-gap effects",
        }
    if "density" in features:
        return {
            "next_null_model": "ktuple_local",
            "refinement": "match local density blocks and simple admissible tuple counts",
            "absorbs": "short-interval density drift",
        }
    if "wheel" in features or "extra_sieve" in features:
        return {
            "next_null_model": "wheel",
            "refinement": "increase sieve bound and forbid small-prime divisibility leakage",
            "absorbs": "small-prime modular obstruction",
        }
    return {
        "next_null_model": "block_bootstrap" if current != "block_bootstrap" else current,
        "refinement": "bootstrap local indicator blocks and re-test held-out compression",
        "absorbs": "unclassified low-complexity statistic",
    }
