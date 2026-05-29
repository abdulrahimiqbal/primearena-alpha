#!/usr/bin/env bash
set -euo pipefail
python scripts/smoke_test.py
python -m primearena.train --config configs/smoke.json --mode imitation --steps 200 --run-name local_smoke
python -m primearena.success_benchmark --config configs/smoke.json --benchmark benchmarks/success_benchmark.yaml --run-dir runs/local_smoke --episodes 50
