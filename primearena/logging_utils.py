from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json
import time


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, object]) -> None:
        payload = {"time": time.time(), **row}
        with self.path.open("a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def make_summary_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(log_dir))
    except Exception:
        return None


def tb_add(writer, tag: str, value: object, step: int) -> None:
    if writer is None:
        return
    try:
        if isinstance(value, (int, float, bool)):
            writer.add_scalar(tag, float(value), step)
    except Exception:
        pass
