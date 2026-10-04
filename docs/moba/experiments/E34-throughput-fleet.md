# E34: training throughput (11× per pod) and a 20-pod fleet

**Status:** fleet running (started 2026-10-04 12:29 local), 20,000 updates per pod.

## 1. The bottleneck was one einsum backward
- **Measurement:** `scripts/moba/profile_update.py`, held-out games, A40 and A100.
- **Findings:**
  - Time per update grew linearly with batch size, and an A100 was no faster than an A40.
  - The backward pass was 90% of each update.
  - **88% of GPU time was one `aten::bmm`:** the weight gradient of the ally/enemy/neutral attention bias,
    `einsum("bnmr,hr->bhnm", rel, team_bias)`. It is 3 × (B·207·207) × heads and ran about 40 ms per call
    through a slow `cutlass_75_wmma` kernel.
- **Fix:** three masked adds (`model.py`). The values are identical (verified).
- **`--compile`** (`torch.compile` of the model step) halves the time again. Its error against fp32 is the
  same as eager bf16 (mean 1.27e-3 vs 1.30e-3).

| pod, batch 384 | before | after fix | + compile |
|---|---|---|---|
| A40 ($0.49/hr) | 11.0 s (35 lanes/s) | 1.82 s | **0.94 s (409 lanes/s)** |
| A100 80GB ($1.59/hr) | 11.7 s | 0.90 s | **0.51 s (752 lanes/s; 1,037 at batch 768)** |

The old single-H200 runs did about 160 lanes/s (batch 768, 4.8 s per update).

## 2. Fleet training (zelda-RL pattern)
- **Transport:** `mobawm/fleet_link.py`, ported from zelda-RL. Each pod trains on its own shard. Every
  `--fleet-every` updates the weights are averaged through Upstash Redis (FedAvg): rank 0 aggregates, and
  stragglers catch up. RunPod pods cannot reach each other.
- **Within a pod:** multi-GPU gradient all-reduce (torchrun) is also supported, but unused so far.
- **Shards:** each 10-game training file goes to rank i mod N, so every corpus is in every shard. A shard
  (about 7.8 GB) fits on the GPU, which removes the CPU gather bottleneck. A40 pods have only about
  7.6 vCPU and 50 GB RAM (cgroup).
- **Launch:** `tools/fleet.py` creates the pods, starts the rank launchers and clears the sync keys.
  `tools/fleet_rank.py` sets up and launches one pod. `SKIP_SETUP=1` relaunches on existing pods in seconds.
  `FLEET_SHARDS` lets a smoke test use the full-fleet shard size.
- **Run `fleet20`:**
  - **Pods:** 19× A40 plus an A100 for rank 0 (rank 0 validates, checkpoints, and runs the 100-game
    watcher).
  - **Training:** batch 384 per pod, `--compile`, lr 1e-4, init P1 @1000, average every 50 updates.
  - **Speed:** pod setup about 2–2.5 min. Steady state is **0.91 updates/s per pod with syncs, ≈6,950
    lanes/s for the fleet (≈43× the old single-H200 run)**. All 20 pods share one weight hash. A sync costs
    about 12.5 s per 50 updates (about 19%).

**Artifacts:** S3 `moba/runs/fleet20_r0` (rank 0 checkpoints, tower scores). Smoke `fsmoke_r*`.
Code: `mobawm/fleet_link.py`, `tools/fleet.py`, `tools/fleet_rank.py`, train.py `--fleet-*`, `--compile`, torchrun support.
