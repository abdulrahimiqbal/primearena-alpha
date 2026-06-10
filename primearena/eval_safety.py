"""FROZEN — legacy RL stack, do not extend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .env import PrimeArena


def guess_index_for_action(env: PrimeArena, action: int) -> Optional[int]:
    offset = env.guess_action_offset
    idx = int(action) - offset
    if 0 <= idx < env.window_size:
        return idx
    return None


def test_index_for_action(env: PrimeArena, action: int) -> Optional[int]:
    offset = env.num_filter_actions
    idx = int(action) - offset
    if 0 <= idx < env.window_size:
        return idx
    return None


def certified_eval_mask(env: PrimeArena, legal: Optional[np.ndarray] = None) -> np.ndarray:
    mask = np.array(env.legal_actions() if legal is None else legal, dtype=np.bool_, copy=True)
    offset = env.guess_action_offset
    for idx in range(env.window_size):
        action = offset + idx
        if mask[action] and not env.is_certified_next_prime_guess(idx):
            mask[action] = False
    return mask


def top_actions_from_logits(env: PrimeArena, logits: torch.Tensor, mask: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
    row = logits.detach().float().cpu().numpy().reshape(-1)
    masked = np.where(mask.astype(bool), row, -np.inf)
    if not np.isfinite(masked).any():
        return []
    order = np.argsort(masked)[::-1][:limit]
    out: List[Dict[str, Any]] = []
    for action in order:
        if not np.isfinite(masked[action]):
            continue
        idx = guess_index_for_action(env, int(action))
        item: Dict[str, Any] = {
            "action": int(action),
            "action_text": env.action_to_str(int(action)),
            "logit": float(row[action]),
        }
        if idx is not None:
            item["candidate_status"] = env.candidate_status(idx)
        out.append(item)
    return out


def _best_filter_for_indices(env: PrimeArena, legal: np.ndarray, indices: List[int]) -> Optional[int]:
    if not indices:
        return None
    candidates = env.candidates()
    best_action: Optional[int] = None
    best_eliminated = 0
    for action, q in enumerate(env.cfg.small_primes[: env.num_filter_actions]):
        if not legal[action]:
            continue
        eliminated = 0
        for idx in indices:
            c = int(candidates[idx])
            if c >= 2 and not env.eliminated[idx] and env.tested[idx] == 0 and c % q == 0 and c != q:
                eliminated += 1
        if eliminated > best_eliminated:
            best_eliminated = eliminated
            best_action = action
    return best_action


def _certification_action_for_blocked_guess(env: PrimeArena, guess_idx: int, legal: np.ndarray) -> Optional[int]:
    """Choose a visible, non-oracle action that moves a blocked guess toward certification."""
    if guess_idx < 0 or guess_idx >= env.window_size:
        return None

    unresolved_to_target = [
        i
        for i in range(guess_idx + 1)
        if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
    ]
    filter_action = _best_filter_for_indices(env, legal, unresolved_to_target)
    if filter_action is not None:
        return filter_action

    test_offset = env.num_filter_actions
    if env.tested[guess_idx] == 0 and legal[test_offset + guess_idx]:
        return test_offset + guess_idx

    if env.tested[guess_idx] == 1:
        unresolved_priors = [
            i
            for i in range(guess_idx)
            if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
        ]
        filter_action = _best_filter_for_indices(env, legal, unresolved_priors)
        if filter_action is not None:
            return filter_action
        for idx in unresolved_priors:
            action = test_offset + idx
            if legal[action]:
                return action

    return None


def _certification_action_for_tested_prime(env: PrimeArena, legal: np.ndarray) -> Optional[int]:
    for prime_idx in range(env.window_size):
        if not env.is_candidate_tested_prime(prime_idx):
            continue
        if env.is_certified_next_prime_guess(prime_idx):
            return None
        unresolved_priors = [
            i
            for i in range(prime_idx)
            if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
        ]
        filter_action = _best_filter_for_indices(env, legal, unresolved_priors)
        if filter_action is not None:
            return filter_action
        test_offset = env.num_filter_actions
        for idx in unresolved_priors:
            action = test_offset + idx
            if legal[action]:
                return action
    return None


def _filter_before_obvious_composite_test(env: PrimeArena, test_idx: int, legal: np.ndarray) -> Optional[int]:
    if test_idx < 0 or test_idx >= env.window_size or env.tested[test_idx] != 0 or env.eliminated[test_idx]:
        return None
    direct = _best_filter_for_indices(env, legal, [test_idx])
    if direct is not None:
        return direct
    unresolved = [
        i
        for i in range(env.window_size)
        if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
    ]
    return _best_filter_for_indices(env, legal, unresolved)


def select_eval_action(
    env: PrimeArena,
    logits: torch.Tensor,
    certified_guesses_only: bool,
) -> Dict[str, Any]:
    legal = env.legal_actions()
    normal_top5 = top_actions_from_logits(env, logits, legal, limit=5)
    if not legal.any():
        return {
            "action": None,
            "blocked": False,
            "uncertified_attempt": False,
            "top5_actions": normal_top5,
            "safe_top5_actions": [],
        }

    row = logits.detach().float().cpu().numpy().reshape(-1)
    legal_logits = np.where(legal, row, -np.inf)
    first_action = int(np.argmax(legal_logits))
    first_guess_idx = guess_index_for_action(env, first_action)
    uncertified_attempt = bool(
        certified_guesses_only
        and first_guess_idx is not None
        and not env.is_certified_next_prime_guess(first_guess_idx)
    )
    certified_guess_mask = env.certified_guess_action_mask() if certified_guesses_only else np.zeros_like(legal)

    if certified_guesses_only and not uncertified_attempt and first_guess_idx is None and certified_guess_mask.any():
        action = int(np.argmax(np.where(certified_guess_mask, row, -np.inf)))
        return {
            "action": action,
            "blocked": False,
            "uncertified_attempt": False,
            "certified_guess_available": True,
            "top5_actions": normal_top5,
            "safe_top5_actions": top_actions_from_logits(env, logits, certified_guess_mask, limit=5),
        }

    if certified_guesses_only and first_guess_idx is not None and not uncertified_attempt:
        return {
            "action": first_action,
            "blocked": False,
            "uncertified_attempt": False,
            "top5_actions": normal_top5,
            "safe_top5_actions": normal_top5,
        }

    tested_prime_certification_action = (
        _certification_action_for_tested_prime(env, legal) if certified_guesses_only else None
    )
    if tested_prime_certification_action is not None and not uncertified_attempt:
        return {
            "action": int(tested_prime_certification_action),
            "blocked": False,
            "uncertified_attempt": False,
            "certification_action": int(tested_prime_certification_action),
            "certification_action_text": env.action_to_str(int(tested_prime_certification_action)),
            "certification_action_source": "tested_prime",
            "top5_actions": normal_top5,
            "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
        }

    if certified_guesses_only and not uncertified_attempt:
        unresolved = [
            i
            for i in range(env.window_size)
            if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
        ]
        global_filter = _best_filter_for_indices(env, legal, unresolved)
        if global_filter is not None and first_action != global_filter:
            return {
                "action": int(global_filter),
                "blocked": False,
                "uncertified_attempt": False,
                "certification_action": int(global_filter),
                "certification_action_text": env.action_to_str(int(global_filter)),
                "certification_action_source": "useful_filter",
                "top5_actions": normal_top5,
                "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
            }

    if certified_guesses_only and not uncertified_attempt and first_action == env.expand_action:
        unresolved = [
            i
            for i in range(env.window_size)
            if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
        ]
        if unresolved:
            test_offset = env.num_filter_actions
            for idx in unresolved:
                action = test_offset + idx
                if legal[action]:
                    return {
                        "action": int(action),
                        "blocked": False,
                        "uncertified_attempt": False,
                        "certification_action": int(action),
                        "certification_action_text": env.action_to_str(int(action)),
                        "certification_action_source": "avoid_premature_expand",
                        "top5_actions": normal_top5,
                        "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
                    }

    first_test_idx = test_index_for_action(env, first_action)
    if certified_guesses_only and first_test_idx is not None:
        unresolved_priors = [
            i
            for i in range(first_test_idx)
            if int(env.candidates()[i]) >= 2 and not env.eliminated[i] and env.tested[i] == 0
        ]
        filter_action = _best_filter_for_indices(env, legal, unresolved_priors)
        if filter_action is not None:
            return {
                "action": int(filter_action),
                "blocked": False,
                "uncertified_attempt": False,
                "certification_action": int(filter_action),
                "certification_action_text": env.action_to_str(int(filter_action)),
                "certification_action_source": "resolve_prior_candidates",
                "top5_actions": normal_top5,
                "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
            }
        test_offset = env.num_filter_actions
        for idx in unresolved_priors:
            action = test_offset + idx
            if legal[action]:
                return {
                    "action": int(action),
                    "blocked": False,
                    "uncertified_attempt": False,
                    "certification_action": int(action),
                    "certification_action_text": env.action_to_str(int(action)),
                    "certification_action_source": "resolve_prior_candidates",
                    "top5_actions": normal_top5,
                    "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
                }

    if certified_guesses_only and first_test_idx is not None:
        filter_action = _filter_before_obvious_composite_test(env, first_test_idx, legal)
        if filter_action is not None:
            return {
                "action": int(filter_action),
                "blocked": False,
                "uncertified_attempt": False,
                "filter_redirected_test_action": first_action,
                "filter_redirected_test_action_text": env.action_to_str(first_action),
                "certification_action": int(filter_action),
                "certification_action_text": env.action_to_str(int(filter_action)),
                "top5_actions": normal_top5,
                "safe_top5_actions": top_actions_from_logits(env, logits, legal, limit=5),
            }

    if not uncertified_attempt:
        return {
            "action": first_action,
            "blocked": False,
            "uncertified_attempt": False,
            "top5_actions": normal_top5,
            "safe_top5_actions": normal_top5,
        }

    safe = certified_eval_mask(env, legal)
    safe_top5 = top_actions_from_logits(env, logits, safe, limit=5)
    if certified_guess_mask.any():
        action = int(np.argmax(np.where(certified_guess_mask, row, -np.inf)))
        return {
            "action": action,
            "blocked": True,
            "uncertified_attempt": True,
            "blocked_action": first_action,
            "blocked_action_text": env.action_to_str(first_action),
            "blocked_candidate_status": env.candidate_status(first_guess_idx),
            "certified_guess_available": True,
            "top5_actions": normal_top5,
            "safe_top5_actions": safe_top5,
        }
    if tested_prime_certification_action is not None:
        return {
            "action": int(tested_prime_certification_action),
            "blocked": True,
            "uncertified_attempt": True,
            "blocked_action": first_action,
            "blocked_action_text": env.action_to_str(first_action),
            "blocked_candidate_status": env.candidate_status(first_guess_idx),
            "certification_action": int(tested_prime_certification_action),
            "certification_action_text": env.action_to_str(int(tested_prime_certification_action)),
            "certification_action_source": "tested_prime",
            "top5_actions": normal_top5,
            "safe_top5_actions": safe_top5,
        }
    certification_action = _certification_action_for_blocked_guess(env, first_guess_idx, legal)
    if certification_action is not None:
        return {
            "action": int(certification_action),
            "blocked": True,
            "uncertified_attempt": True,
            "blocked_action": first_action,
            "blocked_action_text": env.action_to_str(first_action),
            "blocked_candidate_status": env.candidate_status(first_guess_idx),
            "certification_action": int(certification_action),
            "certification_action_text": env.action_to_str(int(certification_action)),
            "top5_actions": normal_top5,
            "safe_top5_actions": safe_top5,
        }
    if not safe.any():
        return {
            "action": None,
            "blocked": True,
            "uncertified_attempt": True,
            "blocked_action": first_action,
            "blocked_action_text": env.action_to_str(first_action),
            "blocked_candidate_status": env.candidate_status(first_guess_idx),
            "top5_actions": normal_top5,
            "safe_top5_actions": safe_top5,
        }

    action = int(np.argmax(np.where(safe, row, -np.inf)))
    return {
        "action": action,
        "blocked": True,
        "uncertified_attempt": True,
        "blocked_action": first_action,
        "blocked_action_text": env.action_to_str(first_action),
        "blocked_candidate_status": env.candidate_status(first_guess_idx),
        "top5_actions": normal_top5,
        "safe_top5_actions": safe_top5,
    }
