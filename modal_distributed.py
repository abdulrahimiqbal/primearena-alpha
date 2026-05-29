from __future__ import annotations

import glob
import os
from pathlib import Path

import modal

GPU_LEARNER = os.environ.get("PRIMEARENA_LEARNER_GPU", "A10")
GPU_SELFPLAY = os.environ.get("PRIMEARENA_SELFPLAY_GPU", "T4")
APP_NAME = "primearena-distributed"
VOLUME_NAME = "primearena-runs"
REMOTE_ROOT = "/root/primearena_alpha"
REMOTE_RUNS = "/root/primearena_runs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "pyyaml>=6.0",
        "tensorboard>=2.16",
        "torch>=2.3",
    )
    .add_local_dir(".", remote_path=REMOTE_ROOT, ignore=[".venv", "runs", "__pycache__", "*.pyc", ".git"])
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _bootstrap():
    import os
    import sys

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)


@app.function(gpu=GPU_SELFPLAY, timeout=60 * 60 * 6, volumes={REMOTE_RUNS: volume})
def selfplay_worker(
    config: str,
    checkpoint: str | None,
    worker_id: int,
    episodes: int,
    mode: str = "mcts",
    run_name: str = "distributed_round_001",
) -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.config import load_config
    from primearena.distributed import generate_selfplay_shard

    volume.reload()
    cfg = load_config(config)
    cfg.train.run_dir = REMOTE_RUNS
    cfg.train.device = "auto"
    run_root = Path(REMOTE_RUNS) / run_name
    cfg.distributed.shard_dir = str(run_root / "replay_shards")
    progress_dir = run_root / "worker_progress"
    ckpt_path = checkpoint
    if ckpt_path and not ckpt_path.startswith("/"):
        ckpt_path = str(Path(REMOTE_RUNS) / ckpt_path)

    def commit_progress(_payload: dict) -> None:
        volume.commit()

    out = generate_selfplay_shard(
        cfg,
        checkpoint=ckpt_path,
        out_dir=cfg.distributed.shard_dir,
        worker_id=worker_id,
        episodes=episodes,
        mode=mode,
        progress_dir=progress_dir,
        progress_every_episodes=max(1, episodes // 16),
        progress_callback=commit_progress,
    )
    volume.commit()
    return str(out)


@app.function(gpu=GPU_LEARNER, timeout=60 * 60 * 10, volumes={REMOTE_RUNS: volume})
def learner_from_shards(
    config: str,
    run_name: str,
    shard_glob: str = "replay_shards/*.npz",
    resume_checkpoint: str | None = None,
    max_shards: int | None = None,
) -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.distributed import train_from_replay_shards

    volume.reload()
    full_glob = shard_glob if shard_glob.startswith("/") else str(Path(REMOTE_RUNS) / shard_glob)
    resume = resume_checkpoint
    if resume and not resume.startswith("/"):
        resume = str(Path(REMOTE_RUNS) / resume)
    out = train_from_replay_shards(
        config_path=config,
        shard_glob=full_glob,
        run_name=run_name,
        resume_checkpoint=resume,
        max_shards=max_shards,
        run_dir=REMOTE_RUNS,
    )
    volume.commit()
    return str(out)


@app.function(gpu=GPU_LEARNER, timeout=60 * 60 * 4, volumes={REMOTE_RUNS: volume})
def league_eval(config: str, checkpoint_glob: str, episodes: int = 100, out_dir: str = "league") -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.arena import run_league
    from primearena.config import load_config

    volume.reload()
    cfg = load_config(config)
    cfg.train.run_dir = REMOTE_RUNS
    glob_path = checkpoint_glob if checkpoint_glob.startswith("/") else str(Path(REMOTE_RUNS) / checkpoint_glob)
    checkpoints = sorted(glob.glob(glob_path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {glob_path}")
    result = run_league(cfg, checkpoints, episodes=episodes, out_dir=Path(REMOTE_RUNS) / out_dir)
    volume.commit()
    return str(result["out_dir"])


@app.function(timeout=60 * 30, volumes={REMOTE_RUNS: volume})
def select_champion_remote(imitation_run: str, mcts_run: str | None = None, out: str = "champion.json") -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.champion import select_champion

    volume.reload()
    out_path = Path(REMOTE_RUNS) / out
    result = select_champion(REMOTE_RUNS, imitation_run=imitation_run, mcts_run=mcts_run, out=out_path)
    volume.commit()
    if not result["selected"]:
        raise RuntimeError(f"No valid champion: {result['failure_reasons']}")
    return str(out_path)


@app.function(gpu=GPU_LEARNER, timeout=60 * 60 * 4, volumes={REMOTE_RUNS: volume})
def structure_readiness_remote(config: str, checkpoint: str, run_dir: str = "structure_readiness_001") -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.structure_readiness import run_structure_readiness

    volume.reload()
    ckpt = checkpoint if checkpoint.startswith("/") else str(Path(REMOTE_RUNS) / checkpoint)
    out_dir = Path(run_dir) if run_dir.startswith("/") else Path(REMOTE_RUNS) / run_dir
    run_structure_readiness(config, ckpt, out_dir)
    volume.commit()
    return str(out_dir / "structure_readiness.json")


@app.function(gpu=GPU_LEARNER, timeout=60 * 60 * 12, volumes={REMOTE_RUNS: volume})
def residual_rank_train_remote(
    config: str,
    run_name: str,
    steps: int = 5000,
    batch_size: int = 256,
    hard_solver_uncertainty: bool | None = None,
    resume_checkpoint: str | None = None,
) -> str:
    _bootstrap()
    from primearena.config import load_config
    from primearena.train_residual_rank import train_residual_rank

    volume.reload()
    cfg = load_config(config)
    cfg.train.run_dir = REMOTE_RUNS
    cfg.train.device = "cuda"
    if resume_checkpoint:
        cfg.train.resume_checkpoint = resume_checkpoint if resume_checkpoint.startswith("/") else str(Path(REMOTE_RUNS) / resume_checkpoint)
    if hard_solver_uncertainty is not None:
        cfg.env.residual_rank_hard_solver_uncertainty = bool(hard_solver_uncertainty)
    out = train_residual_rank(cfg, run_name=run_name, steps=steps, batch_size=batch_size)
    volume.commit()
    return str(out)


@app.function(gpu=GPU_LEARNER, timeout=60 * 60 * 12, volumes={REMOTE_RUNS: volume})
def residual_rank_controls_remote(
    config: str,
    out_dir: str,
    rank_checkpoint: str,
    samples: int = 20000,
    seeds: int = 3,
    sieve_bounds: str = "30,210,211,1000",
    trained_input_steps: int = 1000,
    bootstrap_rounds: int = 2000,
) -> str:
    _bootstrap()
    from pathlib import Path
    from primearena.residual_rank_controls import run_controls

    volume.reload()
    bounds = [int(x.strip()) for x in str(sieve_bounds).replace(" ", ",").split(",") if x.strip()]
    ckpt = rank_checkpoint if rank_checkpoint.startswith("/") else str(Path(REMOTE_RUNS) / rank_checkpoint)
    out_path = out_dir if out_dir.startswith("/") else str(Path(REMOTE_RUNS) / out_dir)
    run_controls(
        config=config,
        out_dir=out_path,
        sieve_bounds=bounds,
        samples=samples,
        seeds=seeds,
        rank_checkpoint=ckpt,
        hard_mode=True,
        model_chunk_size=512,
        trained_input_steps=trained_input_steps,
        trained_input_batch_size=512,
        trained_input_hidden_dim=64,
        bootstrap_rounds=bootstrap_rounds,
    )
    volume.commit()
    return str(Path(out_path) / "RESIDUAL_RANK_HARD_DECISION_REPORT.md")


@app.local_entrypoint()
def main(
    config: str = "configs/distributed_modal.json",
    run_name: str = "distributed_round_001",
    checkpoint: str | None = None,
    workers: int = 4,
    episodes_per_worker: int = 16,
    mode: str = "mcts",
    train_after: bool = True,
    task: str = "round",
    imitation_run: str = "modal_imitation_gate_001",
    mcts_run: str | None = None,
    structure_run_dir: str = "structure_readiness_001",
    out_dir: str = "residual_rank_hard_controls_001",
    steps: int = 5000,
    batch_size: int = 256,
    samples: int = 20000,
    seeds: int = 3,
    trained_input_steps: int = 1000,
    bootstrap_rounds: int = 2000,
    hard_solver_uncertainty: bool = False,
):
    """One distributed round: fan out self-play shards, then optionally train on them."""
    if task == "select-champion":
        out = select_champion_remote.remote(imitation_run=imitation_run, mcts_run=mcts_run)
        print(f"Champion selected: {out}")
        return
    if task == "structure-readiness":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --task structure-readiness")
        out = structure_readiness_remote.remote(config=config, checkpoint=checkpoint, run_dir=structure_run_dir)
        print(f"Structure readiness report: {out}")
        return
    if task == "residual-rank-train":
        call = residual_rank_train_remote.spawn(
            config=config,
            run_name=run_name,
            steps=steps,
            batch_size=batch_size,
            hard_solver_uncertainty=hard_solver_uncertainty,
            resume_checkpoint=checkpoint,
        )
        print(f"Residual rank train spawned: {call.object_id}")
        return
    if task == "residual-rank-controls":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --task residual-rank-controls")
        call = residual_rank_controls_remote.spawn(
            config=config,
            out_dir=out_dir,
            rank_checkpoint=checkpoint,
            samples=samples,
            seeds=seeds,
            trained_input_steps=trained_input_steps,
            bootstrap_rounds=bootstrap_rounds,
        )
        print(f"Residual rank controls spawned: {call.object_id}")
        return
    if task != "round":
        raise ValueError(f"Unknown task: {task}")

    shard_paths = list(
        selfplay_worker.map(
            [config] * workers,
            [checkpoint] * workers,
            list(range(workers)),
            [episodes_per_worker] * workers,
            [mode] * workers,
            [run_name] * workers,
        )
    )
    print("Generated replay shards:")
    for p in shard_paths:
        print(" ", p)
    if train_after:
        run_dir = learner_from_shards.remote(
            config=config,
            run_name=run_name,
            shard_glob=f"{run_name}/replay_shards/*.npz",
            resume_checkpoint=checkpoint,
        )
        print(f"Learner run complete: {run_dir}")
