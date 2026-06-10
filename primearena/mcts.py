"""FROZEN — legacy RL stack, do not extend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .env import PrimeArena
from .model import PolicyValueNet


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


class BatchedPolicyValueEvaluator:
    """Batches neural policy/value calls for many independent MCTS trees.

    AlphaGo-style search becomes much more GPU-efficient when leaf evaluations are
    grouped. This class is deliberately small: MCTS owns the tree mechanics, while
    this evaluator owns `torch.no_grad()` batching, masking, and normalization.
    """

    def __init__(self, model: PolicyValueNet, device: torch.device):
        self.model = model
        self.device = device

    def predict_batch(self, envs: List[PrimeArena]) -> Tuple[List[np.ndarray], List[float]]:
        if not envs:
            return [], []
        obs = torch.from_numpy(np.stack([e.observation() for e in envs])).float().to(self.device)
        masks_np = np.stack([e.action_mask_float() for e in envs])
        mask = torch.from_numpy(masks_np).float().to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs, mask)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            values = value.detach().cpu().numpy().astype(np.float32)
        out_probs: List[np.ndarray] = []
        for p, m in zip(probs, masks_np):
            p = p * m
            s = float(p.sum())
            if s <= 0 or not np.isfinite(s):
                legal_sum = float(m.sum())
                p = m.astype(np.float32) / max(legal_sum, 1.0)
            else:
                p = (p / s).astype(np.float32)
            out_probs.append(p)
        return out_probs, [float(v) for v in values]


def _policy_value(env: PrimeArena, model: PolicyValueNet, device: torch.device) -> Tuple[np.ndarray, float]:
    evaluator = BatchedPolicyValueEvaluator(model, device)
    priors, values = evaluator.predict_batch([env])
    return priors[0], values[0]


def expand_with_priors(node: Node, env: PrimeArena, priors: np.ndarray) -> None:
    legal = env.legal_actions()
    node.children.clear()
    for action in np.nonzero(legal)[0]:
        node.children[int(action)] = Node(prior=float(priors[action]))


def expand(node: Node, env: PrimeArena, model: PolicyValueNet, device: torch.device) -> float:
    priors, value = _policy_value(env, model, device)
    expand_with_priors(node, env, priors)
    return value


def add_dirichlet_noise(node: Node, alpha: float, fraction: float, rng: np.random.Generator) -> None:
    if not node.children or fraction <= 0 or alpha <= 0:
        return
    actions = list(node.children.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        child = node.children[a]
        child.prior = (1.0 - fraction) * child.prior + fraction * float(n)


def select_child(node: Node, c_puct: float) -> Tuple[int, Node]:
    best_score = -1e18
    best_action = -1
    best_child: Optional[Node] = None
    parent_sqrt = np.sqrt(max(1, node.visit_count))
    for action, child in node.children.items():
        q = child.value
        u = c_puct * child.prior * parent_sqrt / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def _backpropagate(path: List[Node], rewards: List[float], leaf_value: float, gamma: float) -> None:
    path[-1].visit_count += 1
    path[-1].value_sum += leaf_value
    g = leaf_value
    for idx in reversed(range(len(rewards))):
        g = rewards[idx] + gamma * g
        path[idx].visit_count += 1
        path[idx].value_sum += g


def _visits_to_policy(root: Node, action_count: int, legal_actions: np.ndarray) -> np.ndarray:
    visits = np.zeros(action_count, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    if visits.sum() == 0:
        legal = legal_actions.astype(np.float32)
        visits = legal / max(legal.sum(), 1.0)
    else:
        visits /= visits.sum()
    return visits


def run_mcts(
    env: PrimeArena,
    model: PolicyValueNet,
    simulations: int,
    c_puct: float,
    gamma: float,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
    add_root_noise_flag: bool = True,
    root_dirichlet_alpha: float = 0.30,
    root_exploration_fraction: float = 0.25,
) -> np.ndarray:
    return run_mcts_batch(
        [env],
        model,
        simulations,
        c_puct,
        gamma,
        device,
        rng=rng,
        add_root_noise_flag=add_root_noise_flag,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )[0]


def run_mcts_batch(
    envs: List[PrimeArena],
    model: PolicyValueNet,
    simulations: int,
    c_puct: float,
    gamma: float,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
    add_root_noise_flag: bool = True,
    root_dirichlet_alpha: float = 0.30,
    root_exploration_fraction: float = 0.25,
) -> List[np.ndarray]:
    """Run many independent MCTS roots in lockstep with batched leaf inference."""
    if rng is None:
        rng = np.random.default_rng()
    evaluator = BatchedPolicyValueEvaluator(model, device)
    roots = [Node(prior=1.0) for _ in envs]

    root_priors, _ = evaluator.predict_batch(envs)
    for root, env, priors in zip(roots, envs, root_priors):
        expand_with_priors(root, env, priors)
        if add_root_noise_flag:
            add_dirichlet_noise(root, root_dirichlet_alpha, root_exploration_fraction, rng)

    for _ in range(simulations):
        leaf_envs: List[PrimeArena] = []
        leaf_nodes: List[Node] = []
        leaf_paths: List[List[Node]] = []
        leaf_rewards: List[List[float]] = []
        terminal_jobs: List[Tuple[List[Node], List[float], float]] = []

        for root, env in zip(roots, envs):
            sim_env = env.clone()
            node = root
            path = [node]
            rewards: List[float] = []

            while node.expanded and not sim_env.done:
                action, child = select_child(node, c_puct)
                result = sim_env.step(action)
                rewards.append(float(result.reward))
                node = child
                path.append(node)
                if result.done:
                    break

            if sim_env.done:
                terminal_jobs.append((path, rewards, 0.0))
            else:
                leaf_envs.append(sim_env)
                leaf_nodes.append(node)
                leaf_paths.append(path)
                leaf_rewards.append(rewards)

        if leaf_envs:
            priors_batch, values = evaluator.predict_batch(leaf_envs)
            for node, sim_env, priors, value, path, rewards in zip(
                leaf_nodes, leaf_envs, priors_batch, values, leaf_paths, leaf_rewards
            ):
                expand_with_priors(node, sim_env, priors)
                _backpropagate(path, rewards, float(value), gamma)

        for path, rewards, value in terminal_jobs:
            _backpropagate(path, rewards, value, gamma)

    return [_visits_to_policy(root, env.action_count, env.legal_actions()) for root, env in zip(roots, envs)]


def sample_from_policy(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    legal = policy.astype(np.float64).copy()
    if legal.sum() <= 0:
        return int(rng.integers(0, len(policy)))
    if temperature <= 1e-6:
        return int(np.argmax(legal))
    adjusted = np.power(legal, 1.0 / temperature)
    adjusted_sum = adjusted.sum()
    if adjusted_sum <= 0 or not np.isfinite(adjusted_sum):
        return int(np.argmax(legal))
    adjusted /= adjusted_sum
    return int(rng.choice(len(policy), p=adjusted))
