from __future__ import annotations

import os
from pathlib import Path

import modal

GPU = os.environ.get("PRIMEARENA_MODAL_GPU", "A10")
APP_NAME = "primearena-alpha"
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


@app.function(gpu=GPU, timeout=60 * 60 * 8, volumes={REMOTE_RUNS: volume})
def train_on_modal(config: str = "configs/modal_a10.json", run_name: str = "modal_primearena") -> str:
    import os
    import sys
    from pathlib import Path

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)

    from primearena.config import load_config
    from primearena.train import train

    cfg = load_config(config)
    cfg.train.run_dir = REMOTE_RUNS
    print(f"Starting PrimeArena Modal training on GPU={GPU}")
    print(f"Config: {config}; run_name: {run_name}; run_dir: {cfg.train.run_dir}")
    run_dir = train(cfg, run_name=run_name)
    volume.commit()
    return str(run_dir)


@app.local_entrypoint()
def main(config: str = "configs/modal_a10.json", run_name: str = "modal_primearena"):
    path = train_on_modal.remote(config=config, run_name=run_name)
    print(f"Remote run complete: {path}")
