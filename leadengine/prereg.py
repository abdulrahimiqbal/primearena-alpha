from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


class PreregError(RuntimeError):
    pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _path(lead_id: str, out_dir: str | Path = "prereg") -> Path:
    return Path(out_dir) / f"{lead_id}.json"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(lead_id, fit_decades, predicted_effect, ci_low, ci_high, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "lead_id": str(lead_id),
        "fit_decades": [int(d) for d in fit_decades],
        "predicted_effect": float(predicted_effect),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "git_sha": _git_sha(),
        "timestamp": "1970-01-01T00:00:00Z",
    }
    path = _path(str(lead_id), out)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256_path(path)


def _derive_seed(lead_id: str, out_dir: str | Path = "prereg") -> int:
    path = _path(str(lead_id), out_dir)
    if not path.exists():
        raise PreregError(f"missing preregistration for lead {lead_id!r}")
    return int(_sha256_path(path)[:8], 16)


def score_extrapolation(
    lead_id,
    target_decade,
    observed_effect: float | None = None,
    out_dir: str | Path = "prereg",
    scorer: Callable[[int, int], float] | None = None,
    **kwargs: Any,
):
    path = _path(str(lead_id), out_dir)
    if not path.exists():
        raise PreregError(f"missing preregistration for lead {lead_id!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    seed = _derive_seed(str(lead_id), out_dir)
    if scorer is not None:
        observed = float(scorer(int(target_decade), seed))
    elif observed_effect is not None:
        observed = float(observed_effect)
    else:
        observed = None
    predicted = float(payload["predicted_effect"])
    ci = (float(payload["ci_low"]), float(payload["ci_high"]))
    passed = None if observed is None else bool(ci[0] <= observed <= ci[1])
    return {
        "lead_id": str(lead_id),
        "target_decade": int(target_decade),
        "seed": int(seed),
        "predicted": predicted,
        "ci": ci,
        "observed": observed,
        "passed": passed,
    }


score_extrapolation.derive_seed = _derive_seed
