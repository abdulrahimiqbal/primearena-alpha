# PrimeArena Alpha v0.3

PrimeArena is an AlphaGo-Zero-inspired research harness for a prime-search game.

The goal is **not** to beat serious number-theory libraries on day one. The goal is to create a controlled game where an agent learns to find the next prime after `n` using a limited action budget, while we monitor whether it is merely rediscovering known sieve/wheel structure or learning something more interesting.

## What is included

- `PrimeArena` environment: cost-limited search for the next prime.
- Deterministic 64-bit primality oracle and next-prime oracle.
- Baselines: odd scan, wheel scan, segmented sieve approximation.
- PyTorch policy/value model with three backbones:
  - `residual_mlp`
  - `candidate_transformer`
  - `candidate_conv`
- AlphaZero-style MCTS with PUCT selection and optional root Dirichlet noise.
- **Batched neural-network inference inside MCTS** via `run_mcts_batch`.
- Imitation warm-start from a symbolic expert.
- Replay buffer storing `(state, search_policy, value_target)`.
- Distributed replay shards for self-play/search workers.
- Modal distributed orchestration in `modal_distributed.py`.
- Checkpoint league / arena evaluation in `primearena.arena`.
- Adversarial curriculum miner in `primearena.curriculum`.
- Interpretability probes and mod-6/mod-30/mod-210 ablations in `primearena.interpret`.
- Policy cross-entropy + value MSE loss, AdamW, gradient clipping, AMP, TF32, and cosine LR schedule.
- Best-checkpoint saving by evaluation success and by model-vs-baseline delta.
- JSONL run logging + TensorBoard logging.
- Streamlit dashboard for run monitoring.
- Success benchmark thresholds and runnable benchmark checker.
- `docs/ALPHAGO_ZERO_PRACTICES.md` and `docs/DISTRIBUTED_STACK.md`.

## Project layout

```text
primearena_alpha/
  primearena/
    env.py                 # PrimeArena game environment
    oracle.py              # primality / next-prime oracle
    baselines.py           # odd scan, wheel scan, sieve baselines
    expert.py              # symbolic expert policy for imitation warm-start
    model.py               # residual MLP + candidate transformer/conv policy-value nets
    mcts.py                # AlphaZero-style MCTS + batched leaf inference
    replay.py              # replay buffer + compressed shard IO
    train.py               # imitation + MCTS training loop
    distributed.py         # local/Modal worker shard generation helpers
    curriculum.py          # hard-case mining and adversarial curriculum
    arena.py               # checkpoint league / promotion evaluation
    interpret.py           # probes + mod ablations
    evaluate.py            # benchmark model/baselines
    dashboard.py           # Streamlit run monitor
    success_benchmark.py   # runnable benchmark gate
  configs/
    smoke.json             # tiny local candidate-transformer config
    smoke_conv.json        # tiny local candidate-conv config
    modal_a10.json         # GPU imitation warm-start config for Modal
    modal_mcts_a10.json    # GPU MCTS continuation config for Modal
    distributed_modal.json # distributed self-play + learner config
  benchmarks/
    success_benchmark.yaml
    success_benchmark_v1.yaml
  docs/
    ALPHAGO_ZERO_PRACTICES.md
    DISTRIBUTED_STACK.md
  modal_train.py           # single-GPU Modal entrypoint
  modal_distributed.py     # distributed Modal self-play/search + learner entrypoint
  scripts/
    smoke_test.py
```

## Local quickstart

```bash
cd primearena_alpha
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke_test.py
python -m primearena.evaluate --config configs/smoke.json --episodes 100
python -m primearena.train --config configs/smoke.json --mode imitation --steps 50 --run-name local_smoke
python -m primearena.success_benchmark --config configs/smoke.json --benchmark benchmarks/success_benchmark.yaml --run-dir runs/local_smoke
```

## Run batched MCTS locally

```bash
python -m primearena.train --config configs/smoke.json --mode mcts --steps 5 --run-name local_mcts_batched
```

When `train.mode == "mcts"`, `mcts.batched_inference == true`, and `train.episodes_per_step > 1`, training runs independent MCTS trees in lockstep and batches neural leaf evaluations.

## Distributed replay-shard workflow, locally

```bash
python -m primearena.distributed generate-shard \
  --config configs/smoke.json \
  --out-dir runs/replay_shards \
  --worker-id 0 \
  --episodes 4 \
  --mode imitation

python -m primearena.distributed train-from-shards \
  --config configs/smoke.json \
  --shard-glob 'runs/replay_shards/*.npz' \
  --run-name local_from_shards
```

## Mine hard `n` values for adversarial curriculum

```bash
python -m primearena.curriculum \
  --config configs/distributed_modal.json \
  --out runs/hard_cases_modal.jsonl \
  --candidates 2048 \
  --top-k 256
```

## Run checkpoint league / arena evaluation

```bash
python -m primearena.arena \
  --config configs/smoke.json \
  --checkpoint-glob 'runs/local_from_shards/checkpoints/*.pt' \
  --episodes 25 \
  --out-dir runs/league/local_from_shards
```

## Run interpretability probes and ablations

```bash
python -m primearena.interpret \
  --config configs/smoke.json \
  --checkpoint runs/local_from_shards/checkpoints/best_by_success.pt \
  --samples 64 \
  --episodes 25 \
  --out runs/interpretability_report.json
```

The report includes:

- linear probe accuracy for pooled activations predicting `n mod 6`, `n mod 30`, `n mod 210`;
- candidate-token probe accuracy for candidate residues when using `candidate_transformer` or `candidate_conv`;
- greedy evaluation after ablating direct mod-6/mod-30/mod-210 input features.

## Run the project status UI

```bash
streamlit run primearena/project_ui.py -- --runs-dir runs
```

The project UI summarizes the research arc: infrastructure validation, safe learning, structure readiness, residual controls, PrimeLead calibration, the full null-ladder result, and the current blocker/next actions.

## Run the training dashboard

```bash
streamlit run primearena/dashboard.py -- --runs-dir runs
```

The training dashboard reads `runs/*/metrics.jsonl` and shows reward, success rate, cost, losses, learning rate, replay size, checkpoint promotions, batched-MCTS flags, curriculum hard-case counts, model-vs-baseline deltas, worker progress, readiness metrics, and recent episode traces.

TensorBoard is also logged under each run:

```bash
tensorboard --logdir runs
```

## Single-GPU Modal run

```bash
pip install modal
modal setup
cd primearena_alpha
modal run modal_train.py --config configs/modal_a10.json --run-name modal_primearena_imitation_v0
```

Run MCTS training after you have a stable checkpoint. Edit `configs/modal_mcts_a10.json` and set `train.resume_checkpoint`, then run:

```bash
modal run modal_train.py --config configs/modal_mcts_a10.json --run-name modal_primearena_mcts_v0
```

By default the single-GPU Modal function requests `A10`:

```bash
PRIMEARENA_MODAL_GPU=A100 modal run modal_train.py --config configs/modal_a10.json --run-name modal_a100_test
```

## Distributed Modal run

One round of distributed search workers producing replay shards, followed by learner training:

```bash
PRIMEARENA_SELFPLAY_GPU=T4 PRIMEARENA_LEARNER_GPU=A10 modal run modal_distributed.py \
  --config configs/distributed_modal.json \
  --run-name modal_distributed_round_001 \
  --workers 8 \
  --episodes-per-worker 64 \
  --mode mcts
```

Outputs are written to a Modal volume named `primearena-runs`.

Run league evaluation on Modal after checkpoints exist:

```bash
modal run modal_distributed.py::league_eval \
  --config configs/distributed_modal.json \
  --checkpoint-glob 'modal_distributed_round_001/checkpoints/*.pt' \
  --episodes 100 \
  --out-dir league/modal_distributed_round_001
```

## Success benchmark philosophy

A run is only interesting if it clears progressively harder gates:

1. Environment and oracle correctness.
2. Wheel baseline beats naive odd scan.
3. Imitation-trained policy successfully follows an efficient symbolic search policy.
4. MCTS policy improves or at least matches imitation on cost-normalized reward.
5. Out-of-distribution evaluation by number magnitude does not collapse.
6. Interpretability probes identify known wheel/gap features before claiming novelty.

Use the permissive smoke benchmark first:

```bash
python -m primearena.success_benchmark --config configs/smoke.json --benchmark benchmarks/success_benchmark.yaml --run-dir runs/local_smoke
```

After Modal warm-start, use the stricter benchmark:

```bash
python -m primearena.success_benchmark --config configs/modal_a10.json --benchmark benchmarks/success_benchmark_v1.yaml --run-dir runs/modal_primearena_imitation_v0
```

## Honest status

v0.3 has the full **distributed self-play/search + batched MCTS + checkpoint league + hard-case curriculum + candidate-wise model + interpretability probe** stack in code. The local path is smoke-tested. The Modal path is packaged against current Modal primitives but still needs to be exercised in your Modal account because this environment cannot launch remote GPUs.
