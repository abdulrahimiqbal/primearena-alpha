from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .baselines import all_residual_baselines
from .config import RunConfig, load_config
from .counterfactuals import build_counterfactual_pairs
from .env_factory import make_arena_env
from .eval_safety import guess_index_for_action, select_eval_action
from .interpret import _ridge_probe_accuracy, run_linear_probes
from .model import PolicyValueNet, choose_device, load_checkpoint


MODS = [6, 30, 210]


def _cfg_for_bound(cfg: RunConfig, bound: int, ood: bool = False) -> RunConfig:
    out = copy.deepcopy(cfg)
    out.env.residual_mode = True
    out.env.residual_sieve_bound = int(bound)
    if ood:
        old_max = int(out.env.n_max)
        out.env.n_min = old_max + 1
        out.env.n_max = max(old_max + 10, old_max * 10)
    return out


def _make_model(cfg: RunConfig, checkpoint: Optional[str], device: torch.device) -> Tuple[PolicyValueNet, bool, Optional[str]]:
    env = make_arena_env(cfg.env, seed=cfg.train.seed)
    model = PolicyValueNet(env.observation_dim, env.action_count, cfg.model).to(device)
    if not checkpoint:
        model.eval()
        return model, False, "no checkpoint provided"
    try:
        load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
        model.eval()
        return model, True, None
    except Exception as exc:
        model.eval()
        return model, False, f"{type(exc).__name__}: {exc}"


def _ablate_obs(obs: np.ndarray, cfg: RunConfig, kind: Optional[str]) -> np.ndarray:
    if kind is None:
        return obs
    arr = obs.copy()
    w = int(cfg.env.residual_window_size)
    feat_dim = 6
    cand = arr[: w * feat_dim].reshape(w, feat_dim)
    if kind == "distance":
        cand[:, 1:3] = 0.0
    elif kind == "density":
        cand[:, 5] = 0.0
        arr[w * feat_dim + 5] = 0.0
    elif kind == "state":
        cand[:, 3:5] = 0.0
    arr[: w * feat_dim] = cand.reshape(-1)
    return arr.astype(np.float32)


def evaluate_residual_policy(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    episodes: int,
    seed: int,
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    failure_penalty = float(getattr(cfg.eval, "failure_penalty_cost", 25.0))
    successes: List[float] = []
    wrongs: List[float] = []
    costs: List[float] = []
    tests: List[float] = []
    nearest_costs: List[float] = []
    random_costs: List[float] = []
    logn_costs: List[float] = []
    effective_costs: List[float] = []
    wrong_guess_count = 0
    premature_guess_blocked_count = 0
    certified_guess_count = 0
    uncertified_guess_attempt_count = 0
    failed_traces: List[Dict[str, Any]] = []

    for ep in range(max(1, int(episodes))):
        env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        baselines = all_residual_baselines(env.n, cfg.env, bound=cfg.env.residual_sieve_bound, seed=seed + ep)
        final_info: Optional[Dict[str, Any]] = None
        selected: List[str] = []
        while not env.done:
            obs_np = _ablate_obs(env.observation(), cfg, ablation)
            obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask_float()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            decision = select_eval_action(
                env,
                logits,
                certified_guesses_only=bool(getattr(cfg.env, "residual_certified_guesses_only", True)),
            )
            if decision.get("uncertified_attempt"):
                uncertified_guess_attempt_count += 1
            if decision.get("blocked"):
                premature_guess_blocked_count += 1
            action_obj = decision.get("action")
            if action_obj is None:
                env.done = True
                env.success = False
                env.trace.append("guardrail: no certified residual action")
                final_info = env.info(reason="certified_residual_no_safe_action")
                break
            action = int(action_obj)
            guess_idx = guess_index_for_action(env, action)
            if guess_idx is not None and env.is_certified_next_prime_guess(guess_idx):
                certified_guess_count += 1
            selected.append(env.action_to_str(action))
            result = env.step(action)
            final_info = result.info

        wrong = bool((final_info or {}).get("reason") == "wrong_guess")
        wrong_guess_count += int(wrong)
        successes.append(float(env.success))
        wrongs.append(float(wrong))
        costs.append(float(env.total_cost))
        tests.append(float(getattr(env, "total_tests", 0)))
        nearest_costs.append(float(baselines["nearest"].cost))
        random_costs.append(float(baselines["random"].cost))
        logn_costs.append(float(baselines["one_over_logn"].cost))
        effective_costs.append(float(env.total_cost if env.success else baselines["nearest"].cost + failure_penalty))
        if not env.success and len(failed_traces) < 5:
            failed_traces.append(
                {
                    "n": int(env.n),
                    "true_next_prime": int(env.true_next_prime),
                    "reason": (final_info or {}).get("reason"),
                    "trace": list(env.trace[-20:]),
                    "selected_actions": selected[-30:],
                }
            )

    eff = np.asarray(effective_costs, dtype=np.float64)
    return {
        "success_rate": float(np.mean(successes)),
        "wrong_guess_rate": float(np.mean(wrongs)),
        "wrong_guess_count": int(wrong_guess_count),
        "premature_guess_blocked_count": int(premature_guess_blocked_count),
        "certified_guess_count": int(certified_guess_count),
        "uncertified_guess_attempt_count": int(uncertified_guess_attempt_count),
        "avg_cost": float(np.mean(costs)),
        "avg_survivor_tests": float(np.mean(tests)),
        "avg_nearest_survivor_cost": float(np.mean(nearest_costs)),
        "avg_random_survivor_cost": float(np.mean(random_costs)),
        "avg_one_over_logn_cost": float(np.mean(logn_costs)),
        "success_aware_delta_vs_nearest": float(np.mean(np.asarray(nearest_costs) - eff)),
        "success_aware_delta_vs_random": float(np.mean(np.asarray(random_costs) - eff)),
        "success_aware_delta_vs_one_over_logn": float(np.mean(np.asarray(logn_costs) - eff)),
        "failed_traces": failed_traces,
        "eval_mode": "certified_safe_residual",
        "certified_guess_safety_enabled": bool(getattr(cfg.env, "residual_certified_guesses_only", True)),
    }


def collect_residual_activations(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pooled: List[np.ndarray] = []
    candidate_tokens: List[np.ndarray] = []
    n_residues: Dict[int, List[int]] = {m: [] for m in MODS}
    candidate_residues: Dict[int, List[np.ndarray]] = {m: [] for m in MODS}
    batch_size = 128
    remaining = int(samples)
    while remaining > 0:
        n_batch = min(batch_size, remaining)
        envs = [make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000))) for _ in range(n_batch)]
        obs = torch.from_numpy(np.stack([env.observation() for env in envs])).float().to(device)
        mask = torch.from_numpy(np.stack([env.action_mask_float() for env in envs])).float().to(device)
        with torch.no_grad():
            _, _, acts = model(obs, mask, return_activations=True)
        pooled.extend(list(acts["pooled"].detach().cpu().numpy()))
        if "candidate_tokens" in acts:
            candidate_tokens.extend(list(acts["candidate_tokens"].detach().cpu().numpy()))
        for env in envs:
            cands = env.candidates()
            for mod in MODS:
                n_residues[mod].append(int(env.n % mod))
                candidate_residues[mod].append((cands % mod).astype(np.int64))
        remaining -= n_batch
    out: Dict[str, np.ndarray] = {"pooled": np.stack(pooled)}
    if candidate_tokens:
        out["candidate_tokens"] = np.stack(candidate_tokens)
    for mod in MODS:
        out[f"n_mod_{mod}"] = np.asarray(n_residues[mod], dtype=np.int64)
        out[f"candidate_mod_{mod}"] = np.stack(candidate_residues[mod])
    return out


def residual_input_probes(cfg: RunConfig, samples: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    obs_rows: List[np.ndarray] = []
    cand_rows: List[np.ndarray] = []
    n_residues: Dict[int, List[int]] = {m: [] for m in MODS}
    candidate_residues: Dict[int, List[np.ndarray]] = {m: [] for m in MODS}
    for _ in range(samples):
        env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        obs = env.observation()
        obs_rows.append(obs)
        cand_rows.append(obs[: cfg.env.residual_window_size * 6].reshape(cfg.env.residual_window_size, 6))
        cands = env.candidates()
        for mod in MODS:
            n_residues[mod].append(int(env.n % mod))
            candidate_residues[mod].append((cands % mod).astype(np.int64))
    pooled = np.stack(obs_rows)
    cand = np.stack(cand_rows)
    b, w, h = cand.shape
    flat = cand.reshape(b * w, h)
    out: Dict[str, float] = {}
    for mod in MODS:
        out[f"input_pooled_probe_n_mod_{mod}_accuracy"] = _ridge_probe_accuracy(pooled, np.asarray(n_residues[mod]), mod)
        out[f"input_candidate_probe_mod_{mod}_accuracy"] = _ridge_probe_accuracy(
            flat,
            np.stack(candidate_residues[mod]).reshape(b * w),
            mod,
        )
    return out


def evaluate_counterfactuals(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    pairs = build_counterfactual_pairs(cfg.env, samples=max(4, samples // 8), seed=seed, bound=cfg.env.residual_sieve_bound)
    correct = 0
    total = 0
    margins: List[float] = []
    for pair in pairs:
        for n, true_idx in [(pair.n_a, pair.true_index_a), (pair.n_b, pair.true_index_b)]:
            env = make_arena_env(cfg.env, seed=seed + total)
            env.reset(n)
            obs = torch.from_numpy(env.observation()).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask_float()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            row = logits.detach().float().cpu().numpy().reshape(-1)
            w = cfg.env.residual_window_size
            scores = np.maximum(row[:w], row[w : 2 * w])
            pred = int(np.argmax(scores))
            if 0 <= true_idx < len(scores):
                correct += int(pred == true_idx)
                margins.append(float(scores[true_idx] - np.max(np.delete(scores, true_idx))))
                total += 1
    return {
        "matched_counterfactual_pairs": len(pairs),
        "matched_counterfactual_examples": total,
        "matched_counterfactual_accuracy": float(correct / max(total, 1)),
        "matched_counterfactual_true_index_margin": float(np.mean(margins)) if margins else 0.0,
        "pairs": [p.to_dict() for p in pairs[:20]],
    }


def hidden_patch_effect(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    envs = [make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000))) for _ in range(samples)]
    obs = torch.from_numpy(np.stack([env.observation() for env in envs])).float().to(device)
    mask = torch.from_numpy(np.stack([env.action_mask_float() for env in envs])).float().to(device)
    with torch.no_grad():
        logits, _, acts = model(obs, mask, return_activations=True)
    if "candidate_tokens" not in acts:
        return {"hidden_patching_supported": 0.0, "hidden_patching_effect_size": 0.0, "hidden_patching_top_action_changed_rate": 0.0}
    cand = acts["candidate_tokens"].detach().clone()
    pooled = acts["pooled"].detach()
    patched = cand.roll(shifts=1, dims=0)
    with torch.no_grad():
        filter_logits = model.filter_head(pooled)
        test_logits = model.test_head(patched).squeeze(-1)
        guess_logits = model.guess_head(patched).squeeze(-1)
        expand_logit = model.expand_head(pooled)
        patched_logits = torch.cat([filter_logits, test_logits, guess_logits, expand_logit], dim=-1)
        patched_logits = patched_logits.masked_fill(mask <= 0, torch.finfo(patched_logits.dtype).min)
    original = logits.detach().float().cpu().numpy()
    patched_np = patched_logits.detach().float().cpu().numpy()
    finite = np.isfinite(original) & np.isfinite(patched_np)
    return {
        "hidden_patching_supported": 1.0,
        "hidden_patching_effect_size": float(np.mean(np.abs(original[finite] - patched_np[finite]))),
        "hidden_patching_top_action_changed_rate": float(np.mean(np.argmax(original, axis=1) != np.argmax(patched_np, axis=1))),
    }


def run_residual_structure(
    config: str,
    checkpoint: str,
    out_dir: str | Path,
    samples: int,
    episodes: int,
    sieve_bounds: List[int],
    seeds: int,
) -> Dict[str, Any]:
    base_cfg = load_config(config)
    base_cfg.env.residual_mode = True
    if base_cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(base_cfg.train.torch_num_threads))
    device = choose_device(base_cfg.train.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_bound: Dict[str, Any] = {}
    failure_reasons: List[str] = []
    checkpoint_loaded_any = False

    for bound in sieve_bounds:
        cfg = _cfg_for_bound(base_cfg, int(bound), ood=False)
        seed_reports: List[Dict[str, Any]] = []
        for seed_idx in range(1, seeds + 1):
            seed = 70_000 + 997 * seed_idx + int(bound)
            model, loaded, load_error = _make_model(cfg, checkpoint, device)
            checkpoint_loaded_any = checkpoint_loaded_any or loaded
            eval_metrics = evaluate_residual_policy(cfg, model, device, episodes=episodes, seed=seed)
            ood_cfg = _cfg_for_bound(base_cfg, int(bound), ood=True)
            ood_model, _, _ = _make_model(ood_cfg, checkpoint if loaded else None, device)
            if loaded:
                try:
                    load_checkpoint(checkpoint, ood_model, optimizer=None, map_location=device)
                except Exception:
                    pass
            ood_metrics = evaluate_residual_policy(ood_cfg, ood_model, device, episodes=max(10, episodes // 5), seed=seed + 101)
            activations = collect_residual_activations(cfg, model, device, samples=samples, seed=seed + 202)
            probes = run_linear_probes(activations)
            input_probes = residual_input_probes(cfg, samples=samples, seed=seed + 303)
            ablations = {
                "distance": evaluate_residual_policy(cfg, model, device, episodes=max(10, episodes // 2), seed=seed + 401, ablation="distance"),
                "density": evaluate_residual_policy(cfg, model, device, episodes=max(10, episodes // 2), seed=seed + 402, ablation="density"),
                "state": evaluate_residual_policy(cfg, model, device, episodes=max(10, episodes // 2), seed=seed + 403, ablation="state"),
            }
            counterfactuals = evaluate_counterfactuals(cfg, model, device, samples=samples, seed=seed + 504)
            patching = hidden_patch_effect(cfg, model, device, samples=min(samples, 512), seed=seed + 605)
            seed_reports.append(
                {
                    "seed_index": seed_idx,
                    "seed": seed,
                    "checkpoint_loaded": loaded,
                    "checkpoint_load_error": load_error,
                    "eval": eval_metrics,
                    "ood_eval": ood_metrics,
                    "probes": probes,
                    "input_probes": input_probes,
                    "ablations": ablations,
                    "counterfactuals": counterfactuals,
                    "hidden_patching": patching,
                }
            )

        def mean(path: Tuple[str, ...]) -> float:
            vals: List[float] = []
            for report in seed_reports:
                cur: Any = report
                for key in path:
                    cur = cur[key]
                vals.append(float(cur))
            return float(np.mean(vals)) if vals else 0.0

        bound_summary = {
            "sieve_bound": int(bound),
            "checkpoint_loaded": bool(any(r["checkpoint_loaded"] for r in seed_reports)),
            "success_rate": mean(("eval", "success_rate")),
            "wrong_guess_count": int(sum(int(r["eval"]["wrong_guess_count"]) for r in seed_reports)),
            "avg_survivor_tests": mean(("eval", "avg_survivor_tests")),
            "success_aware_delta_vs_nearest": mean(("eval", "success_aware_delta_vs_nearest")),
            "success_aware_delta_vs_random": mean(("eval", "success_aware_delta_vs_random")),
            "success_aware_delta_vs_one_over_logn": mean(("eval", "success_aware_delta_vs_one_over_logn")),
            "ood_success_rate": mean(("ood_eval", "success_rate")),
            "ood_delta_vs_nearest": mean(("ood_eval", "success_aware_delta_vs_nearest")),
            "matched_counterfactual_accuracy": mean(("counterfactuals", "matched_counterfactual_accuracy")),
            "hidden_patching_effect_size": mean(("hidden_patching", "hidden_patching_effect_size")),
            "hidden_patching_top_action_changed_rate": mean(("hidden_patching", "hidden_patching_top_action_changed_rate")),
            "seed_reports": seed_reports,
        }
        by_bound[str(bound)] = bound_summary

    readiness_bound = str(max(sieve_bounds))
    main = by_bound[readiness_bound]
    if not checkpoint_loaded_any:
        failure_reasons.append("checkpoint was not compatible with residual model shape")
    if max(sieve_bounds) < 211:
        failure_reasons.append("residual_sieve_bound < 211")
    if float(main["success_rate"]) < 0.99:
        failure_reasons.append("residual_eval_success_rate < 0.99")
    if float(main["success_aware_delta_vs_nearest"]) < 0.0:
        failure_reasons.append("residual_success_aware_delta_vs_nearest < 0.0")
    if float(main["success_aware_delta_vs_random"]) < 0.0:
        failure_reasons.append("residual_success_aware_delta_vs_random < 0.0")
    if int(main["wrong_guess_count"]) != 0:
        failure_reasons.append("residual_wrong_guess_count != 0")

    report = {
        "checkpoint": checkpoint,
        "config": config,
        "samples": int(samples),
        "episodes": int(episodes),
        "seeds": int(seeds),
        "sieve_bounds": [int(x) for x in sieve_bounds],
        "results_by_sieve_bound": by_bound,
        "ready_for_residual_structure_hunt": not failure_reasons,
        "failure_reasons": failure_reasons,
    }
    (out / "residual_structure.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "RESIDUAL_STRUCTURE_REPORT.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# PrimeArena Residual Structure Report",
        "",
        "This benchmark removes known wheel/sieve structure before the model acts. It is not a discovery claim.",
        "",
        f"Ready: `{report['ready_for_residual_structure_hunt']}`",
        "",
        "| sieve bound | success | wrong guesses | avg tests | delta nearest | delta random | OOD success | counterfactual | hidden patch effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bound, row in report["results_by_sieve_bound"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(bound),
                    f"{row['success_rate']:.4f}",
                    str(row["wrong_guess_count"]),
                    f"{row['avg_survivor_tests']:.4f}",
                    f"{row['success_aware_delta_vs_nearest']:.4f}",
                    f"{row['success_aware_delta_vs_random']:.4f}",
                    f"{row['ood_success_rate']:.4f}",
                    f"{row['matched_counterfactual_accuracy']:.4f}",
                    f"{row['hidden_patching_effect_size']:.6f}",
                ]
            )
            + " |"
        )
    if report["failure_reasons"]:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend([f"- {reason}" for reason in report["failure_reasons"]])
    lines.extend(["", "No discovery claim is made.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/residual_structure_probe.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/residual_structure_001")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--sieve-bounds", nargs="+", type=int, default=[30, 210, 211, 1000])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    report = run_residual_structure(
        config=args.config,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        samples=args.samples,
        episodes=args.episodes,
        sieve_bounds=args.sieve_bounds,
        seeds=args.seeds,
    )
    print(json.dumps({"ready_for_residual_structure_hunt": report["ready_for_residual_structure_hunt"], "failure_reasons": report["failure_reasons"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
