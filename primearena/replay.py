"""FROZEN — legacy RL stack, do not extend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional
from collections import deque
import random

import numpy as np


@dataclass
class ReplayItem:
    observation: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    value_target: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.data: Deque[ReplayItem] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: ReplayItem) -> None:
        self.data.append(item)

    def extend(self, items: Iterable[ReplayItem]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[ReplayItem]:
        if len(self.data) == 0:
            raise ValueError("ReplayBuffer is empty")
        n = min(batch_size, len(self.data))
        return self.rng.sample(list(self.data), n)

    @staticmethod
    def to_arrays(batch: List[ReplayItem]) -> Dict[str, np.ndarray]:
        return {
            "observations": np.stack([x.observation for x in batch]).astype(np.float32),
            "action_masks": np.stack([x.action_mask for x in batch]).astype(np.float32),
            "policy_targets": np.stack([x.policy_target for x in batch]).astype(np.float32),
            "value_targets": np.asarray([x.value_target for x in batch], dtype=np.float32),
        }


def save_replay_items(path: str | Path, items: List[ReplayItem], metadata: Optional[dict] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = ReplayBuffer.to_arrays(items) if items else {
        "observations": np.empty((0, 0), dtype=np.float32),
        "action_masks": np.empty((0, 0), dtype=np.float32),
        "policy_targets": np.empty((0, 0), dtype=np.float32),
        "value_targets": np.empty((0,), dtype=np.float32),
    }
    meta = np.asarray([metadata or {}], dtype=object)
    np.savez_compressed(p, **arrays, metadata=meta)
    return p


def load_replay_items(path: str | Path) -> List[ReplayItem]:
    with np.load(Path(path), allow_pickle=True) as data:
        obs = data["observations"]
        if obs.size == 0:
            return []
        masks = data["action_masks"]
        policies = data["policy_targets"]
        values = data["value_targets"]
        return [
            ReplayItem(obs[i].astype(np.float32), masks[i].astype(np.float32), policies[i].astype(np.float32), float(values[i]))
            for i in range(obs.shape[0])
        ]


def load_replay_shards(paths: Iterable[str | Path], limit: Optional[int] = None) -> List[ReplayItem]:
    items: List[ReplayItem] = []
    for path in paths:
        items.extend(load_replay_items(path))
        if limit is not None and len(items) >= limit:
            return items[:limit]
    return items
