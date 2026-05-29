from __future__ import annotations

from typing import List, Optional

from .env import PrimeArena


def expert_action(env: PrimeArena, max_filters: Optional[int] = None) -> int:
    """Simple symbolic expert: apply wheel filters, test survivors in order, then guess.

    This is intentionally not a magical expert that directly guesses the true prime from
    hidden information. It follows a transparent wheel-scan strategy.
    """
    legal = env.legal_actions()
    p = env.num_filter_actions
    w = env.window_size
    max_f = p if max_filters is None else min(max_filters, p)

    # If a prime has already been tested and every earlier candidate in this window is
    # eliminated or known composite, guess that tested prime.
    candidates = env.candidates()
    for i in range(w):
        if candidates[i] < 2:
            continue
        if env.eliminated[i] or env.tested[i] == -1:
            continue
        if env.tested[i] == 1:
            action = p + w + i
            if legal[action]:
                return int(action)
        # First unresolved survivor blocks any later guess.
        break

    # Apply unused filters first.
    for i in range(max_f):
        if legal[i]:
            return int(i)

    # Test first unresolved survivor.
    for i in range(w):
        if candidates[i] >= 2 and not env.eliminated[i] and env.tested[i] == 0:
            action = p + i
            if legal[action]:
                return int(action)

    # If the true next prime is not in the current window or no survivors remain, expand.
    if legal[env.expand_action]:
        return int(env.expand_action)

    # Fallback: any legal action.
    idx = legal.nonzero()[0]
    if len(idx) == 0:
        return 0
    return int(idx[0])


def rollout_expert(env: PrimeArena, max_steps: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    limit = max_steps or env.cfg.max_steps
    for _ in range(limit):
        if env.done:
            break
        obs = env.observation().copy()
        mask = env.action_mask_float().copy()
        action = expert_action(env)
        result = env.step(action)
        rows.append(
            {
                "observation": obs,
                "mask": mask,
                "action": action,
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
            }
        )
        if result.done:
            break
    final_reward = rows[-1]["reward"] if rows else 0.0
    for row in rows:
        row["final_reward"] = final_reward
    return rows
