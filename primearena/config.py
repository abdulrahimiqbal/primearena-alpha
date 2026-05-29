from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


DEFAULT_SMALL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


@dataclass
class EnvConfig:
    n_min: int = 10_000
    n_max: int = 1_000_000
    window_size: int = 128
    max_steps: int = 80
    budget: float = 80.0
    small_primes: List[int] = field(default_factory=lambda: DEFAULT_SMALL_PRIMES.copy())
    ensure_prime_in_window: bool = True
    max_resample_attempts: int = 1_000
    filter_cost: float = 0.10
    test_cost: float = 1.00
    guess_cost: float = 0.15
    expand_cost: float = 0.50
    invalid_action_penalty: float = 0.25
    step_cost_scale: float = 0.02
    correct_reward: float = 1.0
    wrong_guess_reward: float = -1.0
    timeout_reward: float = -1.0
    residual_mode: bool = False
    residual_sieve_bound: int = 211
    residual_window_size: int = 512
    residual_hide_small_mod_features: bool = True
    residual_include_distance_features: bool = True
    residual_include_density_features: bool = True
    residual_certified_guesses_only: bool = True
    residual_rank_mode: bool = False
    residual_rank_sieve_bound: int = 211
    residual_rank_window_size: int = 512
    residual_rank_target: str = "next_prime_index"  # next_prime_index | survivor_primality
    residual_rank_hide_small_mod_features: bool = True
    residual_rank_eval_bounds: List[int] = field(default_factory=lambda: [30, 210, 211, 1000])
    residual_rank_hard_mode: bool = False
    residual_rank_hard_fraction: float = 0.75
    residual_rank_hard_pool_multiplier: int = 12
    residual_rank_hard_min_true_index: int = 1
    residual_rank_hard_include_input_only: bool = True
    residual_rank_hard_solver_uncertainty: bool = True
    residual_rank_hard_balance_indices: bool = True
    residual_rank_hard_match_mod: int = 210
    residual_rank_hard_signature_len: int = 32


@dataclass
class ModelConfig:
    # architecture options:
    #   residual_mlp          flat AlphaZero-style policy/value residual MLP
    #   candidate_transformer candidate-token transformer over the search window
    #   candidate_conv        candidate-token 1D convolutional residual network
    architecture: str = "residual_mlp"
    hidden_dim: int = 256
    layers: int = 3
    dropout: float = 0.05
    residual: bool = True
    layer_norm: bool = True
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 5
    use_positional_embeddings: bool = True


@dataclass
class MCTSConfig:
    simulations: int = 32
    c_puct: float = 1.5
    gamma: float = 1.0
    temperature: float = 1.0
    root_dirichlet_alpha: float = 0.30
    root_exploration_fraction: float = 0.25
    add_root_noise: bool = True
    # When train.mode == "mcts" and episodes_per_step > 1, run independent
    # MCTS trees in lockstep and batch all neural leaf expansions.
    batched_inference: bool = True


@dataclass
class CurriculumConfig:
    enabled: bool = False
    hard_cases_path: Optional[str] = None
    hard_case_mix: float = 0.35
    mine_every: int = 0
    mine_candidates: int = 512
    top_k: int = 128
    min_gap: int = 0
    score: str = "gap_wheel"  # gap_wheel | wheel_cost


@dataclass
class DistributedConfig:
    # Used by primearena.distributed and modal_distributed.py.
    enabled: bool = False
    shard_dir: str = "runs/replay_shards"
    checkpoint_dir: str = "runs/checkpoints"
    selfplay_workers: int = 4
    episodes_per_worker: int = 32
    max_shards_per_train: int = 128
    worker_mode: str = "mcts"  # mcts | imitation
    # Guardrails for distributed self-play. These prevent one unlucky worker
    # from producing a huge MCTS shard and blocking the whole Modal round.
    selfplay_batch_size: int = 2
    max_worker_seconds: int = 1800
    max_replay_items_per_shard: int = 4096
    max_selfplay_episode_steps: int = 48
    round_count: int = 1
    league_dir: str = "runs/league"
    promotion_min_success_delta: float = 0.00
    promotion_min_score_delta: float = 0.00


@dataclass
class TrainConfig:
    seed: int = 7
    mode: str = "imitation"  # imitation | mcts
    steps: int = 1_000
    batch_size: int = 64
    lr: float = 3e-4
    replay_capacity: int = 50_000
    episodes_per_step: int = 1
    eval_every: int = 50
    checkpoint_every: int = 250
    device: str = "auto"
    run_dir: str = "runs"
    run_name: Optional[str] = None
    max_episode_steps: Optional[int] = None
    use_amp: bool = True
    cosine_lr: bool = True
    min_lr_ratio: float = 0.10
    resume_checkpoint: Optional[str] = None
    torch_num_threads: Optional[int] = None
    prefill_replay_shards: List[str] = field(default_factory=list)
    save_replay_shards: bool = False


@dataclass
class EvalConfig:
    episodes: int = 200
    n_buckets: List[List[int]] = field(default_factory=lambda: [[10_4, 10_5], [10_5, 10_6], [10_6, 10_7]])
    # Success-aware evaluation prevents cheap failures from looking better
    # than baseline scans just because they spend less cost.
    failure_penalty_cost: float = 25.0
    min_promotion_success_rate: float = 0.80
    min_success_aware_delta: float = 0.0
    max_promotion_wrong_guess_rate: float = 0.20
    eval_certified_guesses_only: bool = True
    readiness_certified_guesses_only: bool = True


@dataclass
class RunConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _from_dict(data: Dict[str, Any]) -> RunConfig:
    cfg = RunConfig()
    merged = _deep_update(cfg.to_dict(), data)
    return RunConfig(
        env=EnvConfig(**merged.get("env", {})),
        model=ModelConfig(**merged.get("model", {})),
        mcts=MCTSConfig(**merged.get("mcts", {})),
        train=TrainConfig(**merged.get("train", {})),
        eval=EvalConfig(**merged.get("eval", {})),
        curriculum=CurriculumConfig(**merged.get("curriculum", {})),
        distributed=DistributedConfig(**merged.get("distributed", {})),
    )


def load_config(path: str | Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML config files.") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _from_dict(data)


def save_config(cfg: RunConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
