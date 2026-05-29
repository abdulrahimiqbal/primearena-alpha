from __future__ import annotations

from typing import Optional, Union

from .config import EnvConfig, RunConfig
from .env import PrimeArena
from .residual_env import PrimeArenaResidual


ArenaEnv = Union[PrimeArena, PrimeArenaResidual]


def make_arena_env(cfg: EnvConfig, seed: Optional[int] = None) -> ArenaEnv:
    if bool(getattr(cfg, "residual_mode", False)):
        return PrimeArenaResidual(cfg, seed=seed)
    return PrimeArena(cfg, seed=seed)


def make_probe_env(cfg: RunConfig) -> ArenaEnv:
    return make_arena_env(cfg.env, seed=cfg.train.seed)
