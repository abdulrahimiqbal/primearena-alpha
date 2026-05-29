from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .config import RunConfig, load_config
from .env import PrimeArena
from .model import PolicyValueNet, choose_device, infer_primearena_shape, load_checkpoint
from .train import evaluate_greedy


def make_model(cfg: RunConfig, checkpoint: str, device: torch.device) -> PolicyValueNet:
    env = PrimeArena(cfg.env, seed=cfg.train.seed)
    model = PolicyValueNet(env.observation_dim, env.action_count, cfg.model).to(device)
    load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    model.eval()
    return model


def collect_activations(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    samples: int = 256,
    seed: int = 0,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pooled: List[np.ndarray] = []
    candidate_tokens: List[np.ndarray] = []
    layer_arrays: Dict[str, List[np.ndarray]] = {}
    candidate_residues: Dict[int, List[np.ndarray]] = {6: [], 30: [], 210: []}
    n_residues: Dict[int, List[int]] = {6: [], 30: [], 210: []}

    remaining = int(samples)
    while remaining > 0:
        n_batch = min(max(1, int(batch_size)), remaining)
        envs = [PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000))) for _ in range(n_batch)]
        obs = torch.from_numpy(np.stack([env.observation() for env in envs])).float().to(device)
        mask = torch.from_numpy(np.stack([env.action_mask_float() for env in envs])).float().to(device)
        with torch.no_grad():
            _, _, acts = model(obs, mask, return_activations=True)
        pooled.extend(list(acts["pooled"].detach().cpu().numpy()))
        if "candidate_tokens" in acts:
            candidate_tokens.extend(list(acts["candidate_tokens"].detach().cpu().numpy()))
        for key, value in acts.items():
            if key.startswith("layer_"):
                layer_arrays.setdefault(key, []).extend(list(value.detach().cpu().numpy()))
        for env in envs:
            candidates = env.candidates()
            for m in [6, 30, 210]:
                n_residues[m].append(int(env.n % m))
                candidate_residues[m].append((candidates % m).astype(np.int64))
        remaining -= n_batch

    out: Dict[str, np.ndarray] = {"pooled": np.stack(pooled)}
    for m in [6, 30, 210]:
        out[f"n_mod_{m}"] = np.asarray(n_residues[m], dtype=np.int64)
        out[f"candidate_mod_{m}"] = np.stack(candidate_residues[m])
    if candidate_tokens:
        out["candidate_tokens"] = np.stack(candidate_tokens)
    for key, values in layer_arrays.items():
        if values:
            out[key] = np.stack(values)
    return out


def _ridge_probe_accuracy(x: np.ndarray, y: np.ndarray, classes: int, ridge: float = 1e-3) -> float:
    # Closed-form linear probe: X -> one-hot(y). No sklearn dependency.
    x = x.astype(np.float64)
    y = y.astype(np.int64) % classes
    n = x.shape[0]
    if n == 0:
        return 0.0
    x_aug = np.concatenate([x, np.ones((n, 1))], axis=1)
    y_oh = np.zeros((n, classes), dtype=np.float64)
    y_oh[np.arange(n), y] = 1.0
    xtx = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1])
    w = np.linalg.solve(xtx, x_aug.T @ y_oh)
    pred = np.argmax(x_aug @ w, axis=1)
    return float(np.mean(pred == y))


def run_linear_probes(activations: Dict[str, np.ndarray]) -> Dict[str, float]:
    report: Dict[str, float] = {}
    pooled = activations["pooled"]
    for m in [6, 30, 210]:
        report[f"pooled_probe_n_mod_{m}_accuracy"] = _ridge_probe_accuracy(pooled, activations[f"n_mod_{m}"], m)
    if "candidate_tokens" in activations:
        cand = activations["candidate_tokens"]
        b, w, h = cand.shape
        flat = cand.reshape(b * w, h)
        for m in [6, 30, 210]:
            y = activations[f"candidate_mod_{m}"].reshape(b * w)
            report[f"candidate_probe_mod_{m}_accuracy"] = _ridge_probe_accuracy(flat, y, m)
    for key, value in activations.items():
        if key.startswith("layer_") and key.endswith("_pooled"):
            layer = key[: -len("_pooled")]
            for m in [6, 30, 210]:
                report[f"{layer}_pooled_probe_n_mod_{m}_accuracy"] = _ridge_probe_accuracy(
                    value,
                    activations[f"n_mod_{m}"],
                    m,
                )
        if key.startswith("layer_") and key.endswith("_candidate_tokens"):
            layer = key[: -len("_candidate_tokens")]
            b, w, h = value.shape
            flat = value.reshape(b * w, h)
            for m in [6, 30, 210]:
                y = activations[f"candidate_mod_{m}"].reshape(b * w)
                report[f"{layer}_candidate_probe_mod_{m}_accuracy"] = _ridge_probe_accuracy(flat, y, m)
    return report


def ablate_mod_features(obs: np.ndarray, cfg: RunConfig, mod: int) -> np.ndarray:
    """Zero input features that directly reveal mod-6/mod-30/mod-210 wheel info.

    mod 6: parity + residue/divisible features for 3
    mod 30: mod 6 features + residue/divisible features for 5
    mod 210: mod 30 features + residue/divisible features for 7
    """
    num_filters, window, feat_dim, global_dim = infer_primearena_shape(len(obs), len(cfg.env.small_primes) + 2 * cfg.env.window_size + 1)
    arr = obs.copy().reshape(-1)
    cand = arr[: window * feat_dim].reshape(window, feat_dim)
    # base candidate features: offset, log distance, parity, eliminated, tested composite, tested prime
    if mod in {6, 30, 210}:
        cand[:, 2] = 0.0  # parity
    primes = []
    if mod >= 6:
        primes.append(3)
    if mod >= 30:
        primes.append(5)
    if mod >= 210:
        primes.append(7)
    for q in primes:
        if q in cfg.env.small_primes:
            j = cfg.env.small_primes.index(q)
            base = 6 + 2 * j
            cand[:, base : base + 2] = 0.0
    arr[: window * feat_dim] = cand.reshape(-1)
    return arr.astype(np.float32)


def evaluate_with_ablation(
    cfg: RunConfig,
    model: PolicyValueNet,
    device: torch.device,
    mod: Optional[int],
    episodes: int = 100,
    seed: int = 0,
    batch_size: int = 64,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    successes: List[float] = []
    costs: List[float] = []
    baselines: List[float] = []
    wrongs: List[float] = []
    wrong_guess_count = 0
    remaining = int(episodes)
    while remaining > 0:
        n_batch = min(max(1, int(batch_size)), remaining)
        envs = [PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000))) for _ in range(n_batch)]
        final_infos: List[Optional[Dict[str, object]]] = [None for _ in envs]
        while any(not env.done for env in envs):
            active = [i for i, env in enumerate(envs) if not env.done]
            obs_rows = []
            mask_rows = []
            for i in active:
                obs_np = envs[i].observation()
                if mod is not None:
                    obs_np = ablate_mod_features(obs_np, cfg, mod)
                obs_rows.append(obs_np)
                mask_rows.append(envs[i].action_mask_float())
            obs = torch.from_numpy(np.stack(obs_rows)).float().to(device)
            mask = torch.from_numpy(np.stack(mask_rows)).float().to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
                actions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            for i, action in zip(active, actions):
                result = envs[i].step(int(action))
                final_infos[i] = result.info
        for env, final_info in zip(envs, final_infos):
            successes.append(float(env.success))
            costs.append(float(env.total_cost))
            baselines.append(float(env.baseline_cost))
            wrong = bool((final_info or {}).get("reason") == "wrong_guess")
            wrongs.append(float(wrong))
            wrong_guess_count += int(wrong)
        remaining -= n_batch
    return {
        "eval_mode": "raw_policy_ablation" if mod is not None else "raw_policy",
        "certified_guess_safety_enabled": False,
        "success_rate": float(np.mean(successes)),
        "wrong_guess_rate": float(np.mean(wrongs)),
        "wrong_guess_count": int(wrong_guess_count),
        "premature_guess_blocked_count": 0,
        "certified_guess_count": 0,
        "uncertified_guess_attempt_count": int(wrong_guess_count),
        "avg_cost": float(np.mean(costs)),
        "avg_baseline_cost": float(np.mean(baselines)),
        "model_vs_baseline_delta": float(np.mean(np.asarray(baselines) - np.asarray(costs))),
    }


def interpret_checkpoint(
    cfg: RunConfig,
    checkpoint: str,
    samples: int = 256,
    episodes: int = 100,
    seed: int = 0,
) -> Dict[str, object]:
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    device = choose_device(cfg.train.device)
    model = make_model(cfg, checkpoint, device)
    activations = collect_activations(cfg, model, device, samples=samples, seed=seed)
    probes = run_linear_probes(activations)
    ablations = {"none": evaluate_with_ablation(cfg, model, device, None, episodes=episodes, seed=seed + 1)}
    for m in [6, 30, 210]:
        ablations[f"ablate_mod_{m}"] = evaluate_with_ablation(cfg, model, device, m, episodes=episodes, seed=seed + 10 + m)
    return {"checkpoint": checkpoint, "samples": samples, "episodes": episodes, "probes": probes, "ablations": ablations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/interpretability_report.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = interpret_checkpoint(cfg, args.checkpoint, samples=args.samples, episodes=args.episodes, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
