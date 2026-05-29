# PrimeArena v0.3 Distributed Stack

This version adds the pieces that were intentionally deferred from v0.2.

## Added systems

### 1. Distributed self-play/search workers

- `primearena.distributed.generate_selfplay_shard(...)` generates compressed replay shards.
- `modal_distributed.py` fans out many Modal self-play workers and writes shards to the shared Modal volume.
- Workers can use imitation mode for cheap warm-start shards or MCTS mode for AlphaZero-style search targets.

### 2. Batched neural-network inference inside MCTS

- `primearena.mcts.BatchedPolicyValueEvaluator` batches observation/mask tensors.
- `primearena.mcts.run_mcts_batch(...)` runs independent MCTS roots in lockstep and evaluates leaf states in batches.
- `primearena.train` uses this automatically when `train.mode == "mcts"`, `mcts.batched_inference == true`, and `train.episodes_per_step > 1`.

### 3. Checkpoint league / arena evaluation

- `primearena.arena.run_league(...)` ranks checkpoints on identical seeded episodes.
- `primearena.arena.compare_and_promote(...)` supports champion-vs-challenger promotion gates.
- The output files are `league.json` and `leaderboard.csv`.

### 4. Adversarial curriculum

- `primearena.curriculum.mine_hard_cases(...)` scans for hard `n` values based on next-prime gap and wheel baseline cost.
- `AdversarialCurriculum` mixes hard cases into training episodes.
- `python -m primearena.curriculum --config configs/distributed_modal.json --out runs/hard_cases_modal.jsonl` creates an initial hard-case pool.

### 5. Candidate-wise architectures

- `model.architecture = "candidate_transformer"` preserves the candidate-window structure and supports candidate-token interpretability.
- `model.architecture = "candidate_conv"` gives a cheaper local-pattern alternative.
- `model.architecture = "residual_mlp"` remains available for backwards compatibility.

### 6. Interpretability probes and ablations

- `primearena.interpret` runs linear probes for `n mod 6/30/210` and candidate `mod 6/30/210`.
- It also evaluates ablations that zero direct parity/residue input features for mod-6, mod-30, and mod-210 information.

## Local commands

```bash
python scripts/smoke_test.py
python -m primearena.curriculum --config configs/smoke.json --out runs/hard_cases_smoke.jsonl --candidates 128 --top-k 16
python -m primearena.train --config configs/smoke.json --mode mcts --steps 5 --run-name local_mcts_batched
python -m primearena.distributed generate-shard --config configs/smoke.json --out-dir runs/replay_shards --worker-id 0 --episodes 2 --mode imitation
python -m primearena.distributed train-from-shards --config configs/smoke.json --shard-glob 'runs/replay_shards/*.npz' --run-name local_from_shards
python -m primearena.arena --config configs/smoke.json --checkpoint-glob 'runs/local_from_shards/checkpoints/*.pt' --episodes 10
python -m primearena.interpret --config configs/smoke.json --checkpoint runs/local_from_shards/checkpoints/best_by_success.pt --samples 32 --episodes 10
```

## Modal distributed run

```bash
modal run modal_distributed.py \
  --config configs/distributed_modal.json \
  --run-name modal_distributed_round_001 \
  --workers 8 \
  --episodes-per-worker 64 \
  --mode mcts
```

Use these environment variables to choose GPUs:

```bash
PRIMEARENA_SELFPLAY_GPU=T4 PRIMEARENA_LEARNER_GPU=A10 modal run modal_distributed.py ...
```

## Honest limitations

- The Modal distributed path is packaged and written against current Modal primitives, but must be validated in your Modal account because this environment cannot launch remote GPUs.
- The league is greedy-policy based by default for cost. You can extend it to MCTS evaluation, but that is much more expensive.
- The curriculum is adversarial over sampled `n` values, not a learned adversary yet. This is the right first step because it is stable and debuggable.
- The interpretability probes are linear and causal-input ablations. They are useful first-pass diagnostics, not a full mechanistic interpretability stack.
