# E20 — Scenario mining: 2M tower-death slices from the real simulator

**The one-line result.** From the 12,794 tower/ancient deaths in the 2,560 training games,
the real C game generated **2,014,178 128-tick slices**:
- **1,011,007 where the tower still dies;**
- **1,003,171 near-misses where it survives.**

Each slice is a variation of a real event: a new random stream, small state nudges, and
random-action stretches for some heroes. All outcomes are simulated exactly, not predicted.

- **Where:** in S3 at `moba/runs/scenarios_tower_v1/` (520 files, 16.25 GB, every file checked
  identical).
- **Cost:** built in about 3 minutes of mining on 4 × 32-vCPU CPU pods.
- **Purpose:** training data targeted at the dream's tower errors (E15: 35% precision; damage
  lags reality). The near-misses target false tower deaths.

- **Status:** corpus complete, 2026-09-29. Training on it (a slice loader) is the next step.
- **Code** (uncommitted, experimental; snapshot in the manifest):
  - `moba/moba_fork.c`: `ms_run` (mine) and `ms_replay` (regenerate);
  - `scripts/moba/mine_scenarios.py`.
- **Author decisions** (2026-09-29): variation = actions + random seed + **state changes**;
  window 96 ticks before + 32 after; 1M deaths + 1M survivals; corpus stored in S3.

## How a slice is made

1. **Replay:** the game is replayed from its log (only the logs are needed). Every tower or
   ancient that falls at tick te ≥ 96 is an event.
2. **Restore:** the real state at t0 = te − 96 is restored into a scratch game (`fork_restore`,
   byte-exact since E6).
3. **Vary**, scaled by a strength s ∈ [0, 1] that adapts per event (up when survivals are
   missing, down when deaths are):
   - **random stream:** always new;
   - **state:** each hero with probability 0.5·s moves up to ±2 cells (`move_to`, keeping the
     map consistent); hero health × (1 ± 0.1·s); the target tower's health × (1 ± 0.05);
   - **actions:** the recorded actions, except that each hero with probability 0.6·s plays
     uniform-random actions over a random stretch of the window.
4. **Simulate** 128 ticks with the real game (a slice ends early only if an ancient falls,
   since that's game over) and label it: the target tower died (and at which step) or survived.
5. **Quota:** per event, keep variants until it has its quota of each outcome. Pass 1: 72 + 72,
   `--var-seed 0`. Pass 2: 8 deaths + 17 survivals, `--var-seed 1`. At most 600 attempts per
   event.
6. **Store:** the perturbed start state (MobaState, zlib), the 128 × 10 action codes, and the
   length, event tick, tower, outcome, death step, strength and game.
   `ms_replay(state, codes)` regenerates every state of the slice exactly. Each shard's
   spot-check (20 slices) reproduced the stored outcome with **0 mismatches** (8 shards).

## Corpus

| part | source games (training only) | events | slices | died | survived |
|---|---|---|---|---|---|
| A (+ A_s1) | mix1k 0–449 | 3,625 | 501,389 + 89,436 | 256,533 + 28,942 | 244,856 + 60,494 |
| B (+ B_s1) | mix1k 450–899 | 3,665 | 508,234 + 90,459 | 260,082 + 29,283 | 248,152 + 61,176 |
| C (+ C_s1) | mixrand1k (880 train games) | 3,901 | 516,175 + 92,142 | 277,791 + 31,165 | 238,384 + 60,977 |
| D (+ D_s1) | dyna2 (780 train games) | 1,603 | 184,336 + 32,007 | 114,398 + 12,813 | 69,938 + 19,194 |
| **total** | 2,560 | **12,794** | **2,014,178** | **1,011,007** | **1,003,171** |

- **Events vs the full count:** 12,794 is lower than the 15,625 tower deaths in the full
  dataset, because held-out games are excluded and deaths before tick 96 can't be sliced.
- **Shortfalls:** some events (for example a tower already at ~1% health) can't produce
  survivals even at full strength, and some can't produce deaths.
- **Summaries:** `results/moba/scenarios/summary_*.json`.
- **Speed:** pod B mined its 450 games in 61 s (~8,300 slices/s on 30 processes). The pods ran
  as 4 × 32 vCPU at $0.96/hr each, a few minutes in total.
- **A bug found and fixed:** `tools/pod.py s3_sync_to_pod` used fixed local temp-file names,
  so parallel pods overwrote each other's credential files. It now uses per-pod names.

## Held out and next

- **Held out:** no held-out game (mix1k 900–999, the 120 random held-out games, dyna2's 30
  reserved) contributed events. A held-out scenario set can be mined the same way from them.
- **Next:** a slice loader for world-model training that expands slices with `ms_replay` into
  states for rollout lanes, then train on all scenarios, judged by the tower audit.
