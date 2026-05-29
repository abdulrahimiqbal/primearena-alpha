from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import RunConfig, load_config
from .curriculum import load_hard_cases
from .env import PrimeArena
from .eval_safety import guess_index_for_action, select_eval_action
from .interpret import ablate_mod_features, collect_activations, make_model, run_linear_probes
from .model import PolicyValueNet, choose_device


def _raw_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(p.read_text()) or {}
    return json.loads(p.read_text())


def _structure_settings(config_path: str | Path) -> Dict[str, Any]:
    return _raw_config(config_path).get("structure_readiness", {})


def _cfg_for_range(cfg: RunConfig, n_min: int, n_max: int) -> RunConfig:
    out = copy.deepcopy(cfg)
    out.env.n_min = int(n_min)
    out.env.n_max = int(n_max)
    return out


def evaluate_policy(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    episodes: int,
    seed: int,
    ablate_mod: Optional[int] = None,
    hard_ns: Optional[List[int]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    successes: List[float] = []
    costs: List[float] = []
    baselines: List[float] = []
    success_aware_baselines: List[float] = []
    success_aware_costs: List[float] = []
    wrongs: List[float] = []
    invalid_actions = 0
    failed_traces: List[Dict[str, Any]] = []
    cost_overrun_traces: List[Dict[str, Any]] = []
    failure_penalty = float(getattr(cfg.eval, "failure_penalty_cost", 25.0))
    certified_guesses_only = bool(getattr(cfg.eval, "readiness_certified_guesses_only", True))
    wrong_guess_count = 0
    premature_guess_blocked_count = 0
    certified_guess_count = 0
    uncertified_guess_attempt_count = 0

    for i in range(max(1, episodes)):
        env = PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        if hard_ns:
            env.reset(int(hard_ns[i % len(hard_ns)]))
        final_info: Optional[Dict[str, Any]] = None
        selected_actions: List[Dict[str, Any]] = []
        last_decision: Dict[str, Any] = {}
        final_guess_status: Optional[Dict[str, Any]] = None
        while not env.done:
            obs_np = env.observation()
            if ablate_mod is not None:
                obs_np = ablate_mod_features(obs_np, cfg, ablate_mod)
            obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask_float()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            decision = select_eval_action(env, logits, certified_guesses_only=certified_guesses_only)
            last_decision = decision
            if decision.get("uncertified_attempt"):
                uncertified_guess_attempt_count += 1
            if decision.get("blocked"):
                premature_guess_blocked_count += 1
            action_obj = decision.get("action")
            if action_obj is None:
                env.done = True
                env.success = False
                env.trace.append("guardrail: no certified-safe readiness action")
                final_info = env.info(reason="certified_readiness_no_safe_action")
                selected_actions.append(
                    {
                        "action": None,
                        "action_text": "no certified-safe readiness action",
                        "blocked": bool(decision.get("blocked", False)),
                        "blocked_action": decision.get("blocked_action"),
                        "blocked_action_text": decision.get("blocked_action_text"),
                        "blocked_candidate_status": decision.get("blocked_candidate_status"),
                        "certification_action": decision.get("certification_action"),
                        "certification_action_text": decision.get("certification_action_text"),
                    }
                )
                break
            action = int(action_obj)
            legal = env.legal_actions()
            if action < 0 or action >= env.action_count or not legal[action]:
                invalid_actions += 1
            guess_idx = guess_index_for_action(env, action)
            if guess_idx is not None:
                final_guess_status = env.candidate_status(guess_idx)
                if env.is_certified_next_prime_guess(guess_idx):
                    certified_guess_count += 1
            selected_actions.append(
                {
                    "action": action,
                    "action_text": env.action_to_str(action),
                    "blocked": bool(decision.get("blocked", False)),
                    "blocked_action": decision.get("blocked_action"),
                    "blocked_action_text": decision.get("blocked_action_text"),
                    "blocked_candidate_status": decision.get("blocked_candidate_status"),
                    "certification_action": decision.get("certification_action"),
                    "certification_action_text": decision.get("certification_action_text"),
                    "selected_candidate_status": final_guess_status if guess_idx is not None else None,
                }
            )
            result = env.step(action)
            final_info = result.info

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
        if not env.success and len(failed_traces) < 5:
            failed_traces.append(
                {
                    "n": int(env.n),
                    "true_next_prime": int(env.true_next_prime),
                    "reason": (final_info or {}).get("reason"),
                    "total_cost": model_cost,
                    "baseline_cost": baseline_cost,
                    "trace": list(env.trace[-20:]),
                    "selected_actions": selected_actions[-30:],
                    "blocked_guess_count": int(sum(1 for row in selected_actions if row.get("blocked"))),
                    "candidate_status_before_final_guess": final_guess_status or last_decision.get("blocked_candidate_status"),
                    "top5_actions_at_failure": last_decision.get("top5_actions", []),
                    "safe_top5_actions_at_failure": last_decision.get("safe_top5_actions", []),
                }
            )
        if env.success and model_cost > success_aware_baseline_cost + 1e-9 and len(cost_overrun_traces) < 5:
            cost_overrun_traces.append(
                {
                    "n": int(env.n),
                    "true_next_prime": int(env.true_next_prime),
                    "total_cost": model_cost,
                    "baseline_cost": baseline_cost,
                    "success_aware_baseline_cost": success_aware_baseline_cost,
                    "overrun": float(model_cost - success_aware_baseline_cost),
                    "trace": list(env.trace[-30:]),
                    "selected_actions": selected_actions[-40:],
                }
            )

    baseline_arr = np.asarray(baselines, dtype=np.float64)
    success_aware_baseline_arr = np.asarray(success_aware_baselines, dtype=np.float64)
    cost_arr = np.asarray(costs, dtype=np.float64)
    success_aware_arr = np.asarray(success_aware_costs, dtype=np.float64)
    return {
        "episodes": int(max(1, episodes)),
        "success_rate": float(np.mean(successes)),
        "wrong_guess_rate": float(np.mean(wrongs)),
        "avg_cost": float(np.mean(costs)),
        "avg_baseline_cost": float(np.mean(baselines)),
        "model_vs_baseline_delta": float(np.mean(baseline_arr - cost_arr)),
        "success_aware_baseline_cost": float(np.mean(success_aware_baselines)),
        "success_aware_avg_cost": float(np.mean(success_aware_costs)),
        "success_aware_model_vs_baseline_delta": float(np.mean(success_aware_baseline_arr - success_aware_arr)),
        "wrong_guess_count": int(wrong_guess_count),
        "premature_guess_blocked_count": int(premature_guess_blocked_count),
        "certified_guess_count": int(certified_guess_count),
        "uncertified_guess_attempt_count": int(uncertified_guess_attempt_count),
        "certified_guesses_only": certified_guesses_only,
        "invalid_action_count": int(invalid_actions),
        "invalid_action_rate": float(invalid_actions / max(1, episodes)),
        "failed_traces": failed_traces,
        "cost_overrun_traces": cost_overrun_traces,
    }


def _probe_accuracy(probes: Dict[str, float], mod: int) -> Optional[float]:
    candidate_key = f"candidate_probe_mod_{mod}_accuracy"
    pooled_key = f"pooled_probe_n_mod_{mod}_accuracy"
    if candidate_key in probes:
        return float(probes[candidate_key])
    if pooled_key in probes:
        return float(probes[pooled_key])
    return None


def _hard_case_ns(cfg: RunConfig, limit: int) -> List[int]:
    if not cfg.curriculum.hard_cases_path:
        return []
    cases = load_hard_cases(cfg.curriculum.hard_cases_path)
    return [int(c.n) for c in cases[:limit]]


def run_structure_readiness(config_path: str, checkpoint: str, run_dir: str | Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    settings = _structure_settings(config_path)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))

    device = choose_device(cfg.train.device)
    model = make_model(cfg, checkpoint, device)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_domain_cfg = _cfg_for_range(
        cfg,
        int(settings.get("in_domain_n_min", cfg.env.n_min)),
        int(settings.get("in_domain_n_max", cfg.env.n_max)),
    )
    ood_cfg = _cfg_for_range(
        cfg,
        int(settings.get("ood_n_min", cfg.env.n_max + 1)),
        int(settings.get("ood_n_max", max(cfg.env.n_max + 1, cfg.env.n_max * 10))),
    )

    in_domain = evaluate_policy(
        in_domain_cfg,
        model,
        device,
        episodes=int(settings.get("in_domain_episodes", cfg.eval.episodes)),
        seed=int(settings.get("seed", 9001)),
    )
    ood = evaluate_policy(
        ood_cfg,
        model,
        device,
        episodes=int(settings.get("ood_episodes", cfg.eval.episodes)),
        seed=int(settings.get("seed", 9001)) + 101,
    )

    hard_case_success_rate = None
    hard_case_delta = None
    hard_ns = _hard_case_ns(cfg, int(settings.get("hard_case_episodes", 50)))
    if hard_ns:
        hard = evaluate_policy(
            cfg,
            model,
            device,
            episodes=min(len(hard_ns), int(settings.get("hard_case_episodes", 50))),
            seed=int(settings.get("seed", 9001)) + 202,
            hard_ns=hard_ns,
        )
        hard_case_success_rate = float(hard["success_rate"])
        hard_case_delta = float(hard["success_aware_model_vs_baseline_delta"])

    probes_run = False
    probes: Dict[str, float] = {}
    probe_failure: Optional[str] = None
    try:
        activations = collect_activations(
            cfg,
            model,
            device,
            samples=int(settings.get("probe_samples", 256)),
            seed=int(settings.get("seed", 9001)) + 303,
        )
        probes = run_linear_probes(activations)
        probes_run = True
    except Exception as exc:  # pragma: no cover - report path
        probe_failure = f"{type(exc).__name__}: {exc}"

    ablations_run = False
    ablation_failure: Optional[str] = None
    ablation_deltas: Dict[int, Optional[float]] = {6: None, 30: None, 210: None}
    try:
        base_delta = float(in_domain["success_aware_model_vs_baseline_delta"])
        for mod in [6, 30, 210]:
            ablated = evaluate_policy(
                in_domain_cfg,
                model,
                device,
                episodes=int(settings.get("ablation_episodes", cfg.eval.episodes)),
                seed=int(settings.get("seed", 9001)) + 400 + mod,
                ablate_mod=mod,
            )
            ablation_deltas[mod] = float(base_delta - float(ablated["success_aware_model_vs_baseline_delta"]))
        ablations_run = True
    except Exception as exc:  # pragma: no cover - report path
        ablation_failure = f"{type(exc).__name__}: {exc}"

    min_in_success = float(settings.get("min_in_domain_success_rate", 0.80))
    min_ood_success = float(settings.get("min_ood_success_rate", 0.60))
    min_in_delta = float(settings.get("min_in_domain_success_aware_delta", 0.0))
    min_ood_delta = float(settings.get("min_ood_success_aware_delta", 0.0))
    failure_reasons: List[str] = []
    if float(in_domain["success_rate"]) < min_in_success:
        failure_reasons.append(f"in_domain_eval_success_rate < {min_in_success}")
    if float(ood["success_rate"]) < min_ood_success:
        failure_reasons.append(f"ood_eval_success_rate < {min_ood_success}")
    if float(in_domain["success_aware_model_vs_baseline_delta"]) < min_in_delta:
        failure_reasons.append(f"in_domain_success_aware_delta < {min_in_delta}")
    if float(ood["success_aware_model_vs_baseline_delta"]) < min_ood_delta:
        failure_reasons.append(f"ood_success_aware_delta < {min_ood_delta}")
    if int(in_domain.get("wrong_guess_count", 0)) > 0:
        failure_reasons.append("in_domain_wrong_guess_count > 0")
    if int(ood.get("wrong_guess_count", 0)) > 0:
        failure_reasons.append("ood_wrong_guess_count > 0")
    if not probes_run:
        failure_reasons.append(f"probes_failed: {probe_failure}")
    if not ablations_run:
        failure_reasons.append(f"ablations_failed: {ablation_failure}")

    report: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "in_domain_eval_success_rate": float(in_domain["success_rate"]),
        "in_domain_success_aware_delta": float(in_domain["success_aware_model_vs_baseline_delta"]),
        "in_domain_wrong_guess_count": int(in_domain.get("wrong_guess_count", 0)),
        "in_domain_premature_guess_blocked_count": int(in_domain.get("premature_guess_blocked_count", 0)),
        "in_domain_certified_guess_count": int(in_domain.get("certified_guess_count", 0)),
        "in_domain_uncertified_guess_attempt_count": int(in_domain.get("uncertified_guess_attempt_count", 0)),
        "ood_eval_success_rate": float(ood["success_rate"]),
        "ood_success_aware_delta": float(ood["success_aware_model_vs_baseline_delta"]),
        "ood_wrong_guess_count": int(ood.get("wrong_guess_count", 0)),
        "ood_premature_guess_blocked_count": int(ood.get("premature_guess_blocked_count", 0)),
        "ood_certified_guess_count": int(ood.get("certified_guess_count", 0)),
        "ood_uncertified_guess_attempt_count": int(ood.get("uncertified_guess_attempt_count", 0)),
        "wrong_guess_count": int(in_domain.get("wrong_guess_count", 0)) + int(ood.get("wrong_guess_count", 0)),
        "premature_guess_blocked_count": int(in_domain.get("premature_guess_blocked_count", 0)) + int(ood.get("premature_guess_blocked_count", 0)),
        "certified_guess_count": int(in_domain.get("certified_guess_count", 0)) + int(ood.get("certified_guess_count", 0)),
        "uncertified_guess_attempt_count": int(in_domain.get("uncertified_guess_attempt_count", 0)) + int(ood.get("uncertified_guess_attempt_count", 0)),
        "hard_case_success_rate": hard_case_success_rate,
        "hard_case_success_aware_delta": hard_case_delta,
        "probe_mod6_accuracy": _probe_accuracy(probes, 6),
        "probe_mod30_accuracy": _probe_accuracy(probes, 30),
        "probe_mod210_accuracy": _probe_accuracy(probes, 210),
        "ablation_mod6_delta": ablation_deltas[6],
        "ablation_mod30_delta": ablation_deltas[30],
        "ablation_mod210_delta": ablation_deltas[210],
        "ready_for_structure_hunt": not failure_reasons,
        "failure_reasons": failure_reasons,
        "details": {
            "in_domain": in_domain,
            "ood": ood,
            "probes": probes,
            "probes_run": probes_run,
            "probe_failure": probe_failure,
            "ablations_run": ablations_run,
            "ablation_failure": ablation_failure,
            "readiness_thresholds": {
                "min_in_domain_success_rate": min_in_success,
                "min_ood_success_rate": min_ood_success,
                "min_in_domain_success_aware_delta": min_in_delta,
                "min_ood_success_aware_delta": min_ood_delta,
            },
        },
    }
    out_path = out_dir / "structure_readiness.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/structure_hunt_probe.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default="runs/structure_readiness_001")
    args = parser.parse_args()
    report = run_structure_readiness(args.config, args.checkpoint, args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
