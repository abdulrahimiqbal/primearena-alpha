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


def register(lead_id, fit_decades, predicted_effect, ci_low, ci_high, out_dir, interval_type="CI", **extra: Any):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "lead_id": str(lead_id),
        "fit_decades": [int(d) for d in fit_decades],
        "predicted_effect": float(predicted_effect),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "interval_type": str(interval_type),
        "git_sha": _git_sha(),
        "timestamp": "1970-01-01T00:00:00Z",
    }
    payload.update(extra)
    path = _path(str(lead_id), out)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256_path(path)


def register_prediction_interval(lead_id, fit_decades, predicted_effect, fit_se, sampling_se, out_dir, z=2.0):
    half_width = float(z) * ((float(fit_se) ** 2 + float(sampling_se) ** 2) ** 0.5)
    return register(
        lead_id,
        fit_decades,
        predicted_effect,
        float(predicted_effect) - half_width,
        float(predicted_effect) + half_width,
        out_dir,
        interval_type="PI",
        fit_se=float(fit_se),
        sampling_se=float(sampling_se),
        z=float(z),
        half_width=float(half_width),
    )


def _derive_seed(lead_id: str, out_dir: str | Path = "prereg") -> int:
    path = _path(str(lead_id), out_dir)
    if not path.exists():
        raise PreregError(f"missing preregistration for lead {lead_id!r}")
    return int(_sha256_path(path)[:8], 16)


def _derive_target_seeds(lead_id: str, out_dir: str | Path = "prereg", k: int = 3) -> list[int]:
    path = _path(str(lead_id), out_dir)
    if not path.exists():
        raise PreregError(f"missing preregistration for lead {lead_id!r}")
    prereg_hash = _sha256_path(path)
    return [int(hashlib.sha256(f"{prereg_hash}||{i}".encode("utf-8")).hexdigest()[:8], 16) for i in range(int(k))]


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
    interval_type = str(payload.get("interval_type", "CI"))
    target_seeds = [seed]
    if scorer is not None:
        if interval_type == "PI":
            target_seeds = _derive_target_seeds(str(lead_id), out_dir, k=3)
            observed_values = [float(scorer(int(target_decade), s)) for s in target_seeds]
            observed = float(sum(observed_values) / len(observed_values))
        else:
            observed_values = [float(scorer(int(target_decade), seed))]
            observed = observed_values[0]
    elif observed_effect is not None:
        observed = float(observed_effect)
        observed_values = [observed]
    else:
        observed = None
        observed_values = None
    predicted = float(payload["predicted_effect"])
    ci = (float(payload["ci_low"]), float(payload["ci_high"]))
    passed = None if observed is None else bool(ci[0] <= observed <= ci[1])
    return {
        "lead_id": str(lead_id),
        "target_decade": int(target_decade),
        "seed": int(seed),
        "target_seeds": [int(s) for s in target_seeds],
        "interval_type": interval_type,
        "predicted": predicted,
        "ci": ci,
        "observed": observed,
        "observed_values": observed_values,
        "passed": passed,
    }


score_extrapolation.derive_seed = _derive_seed
score_extrapolation.derive_target_seeds = _derive_target_seeds
