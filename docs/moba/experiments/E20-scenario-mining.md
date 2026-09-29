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

## E21: training with the scenario slices (stopped: not working at this share)

- **Recipe:** the regular recipe (E12 @4000 + real attack graph, full data, batch 768, lr 1e-3)
  **plus scenario lanes in every update.**
- **Author's spec:** "the regular training with the tower connection in the state; 100% of
  regular training has the tower scenarios alongside; no crazy weighting."
- **How the lanes work** (`train.py --scenarios`, `ScenarioFeeder` / `ScenarioLanes`):
  - 128 scenario lanes next to the 768 normal ones, each regenerating a slice with `ms_replay`;
  - memory warmed on 4 real ticks, then 124 ticks open loop with the same step loss and decay;
  - weighted per lane (loss × 128/768);
  - sampled 50/50 died / survived.
- **New validation** (`--scen-heldout`): 128 fixed slices (64 / 64) from 12,609 dives mined from
  **held-out** games 900–999 (795 events). It reports outcome accuracy (does the dream's target
  tower die when it really does?), survivors the dream kills, and the tower loss.
- **Logging:** every log line shows the regular-data epoch and the scenario epoch.
- **Runs:** W&B `2t1xkbtz`; S3 `moba/runs/e21_scenarios/` (checkpoints every 250 updates to 3,250).

E21 (W&B 2t1xkbtz): E12 @4000 + real attack graph + scenario lanes (128 of 896, per-lane weight), full data, batch 768, lr 1e-3. Dives: 128 fixed held-out slices (64 died / 64 survived) from held-out games 900-999; towers: 64 fixed held-out windows (23 real falls).

| update | dives: outcome acc | dives: survivors killed | dives: 'dies' precision / recall | dives: tower loss | towers OL128 precision / recall (pred/real) | expert h1 / h16 / h256 / h1024 |
|---|---|---|---|---|---|---|
| 0 | 52% | 62% | 52% / 67% | 0.286 | 41% / 61% (34/23) | 2.90 / 8.11 / 31.62 / 37.42 |
| 250 | 48% | 69% | 48% / 64% | 0.292 | 50% / 70% (32/23) | 1.50 / 7.20 / 29.50 / 37.99 |
| 500 | 51% | 67% | 51% / 69% | 0.331 | 34% / 65% (44/23) | 1.88 / 7.35 / 31.18 / 43.52 |
| 750 | 51% | 75% | 51% / 77% | 0.372 | 12% / 74% (142/23) | 1.89 / 8.11 / 33.39 / 49.88 |
| 1000 | 52% | 67% | 51% / 70% | 0.288 | 12% / 65% (122/23) | 1.42 / 7.97 / 44.54 / 41.52 |
| 1250 | 53% | 58% | 53% / 64% | 0.285 | 13% / 83% (149/23) | 2.77 / 7.40 / 43.99 / 48.68 |
| 1500 | 51% | 72% | 51% / 73% | 0.276 | 12% / 78% (149/23) | 1.47 / 7.79 / 34.63 / 34.05 |
| 1750 | 55% | 56% | 54% / 67% | 0.308 | 12% / 74% (142/23) | 1.77 / 7.41 / 34.64 / 36.78 |
| 2000 | 53% | 67% | 52% / 73% | 0.287 | 7% / 74% (230/23) | 1.86 / 7.33 / 39.50 / 42.45 |
| 2250 | 52% | 66% | 52% / 70% | 0.317 | 13% / 74% (126/23) | 1.71 / 7.53 / 45.06 / 36.35 |
| 2500 | 52% | 78% | 51% / 83% | 0.305 | 11% / 87% (176/23) | 1.54 / 7.26 / 43.92 / 42.84 |
| 2750 | 57% | 44% | 57% / 58% | 0.289 | 9% / 57% (149/23) | 1.45 / 7.57 / 47.21 / 49.68 |
| 3000 | 46% | 44% | 45% / 36% | 0.285 | 19% / 52% (63/23) | 1.59 / 7.82 / 38.05 / 37.97 |
| 3250 | 57% | 45% | 57% / 59% | 0.271 | 7% / 61% (202/23) | 1.37 / 7.06 / 67.31 / 54.85 |

(Every 250 updates; the rows from 750 on are in `results/moba/tables/e21_scenario_training.md`.)

- **Tower dives are not learned:** outcome accuracy stays at 46–57%, a coin flip; survivors
  killed swing between 44% and 78%.
- **Normal tower prediction degrades exactly as in E18:** best at update 250 (50% precision),
  then 7–19% from update ~750, with 120–230 predicted falls for 23 real.
- **One-step improves** (h1 2.90 → 1.37), **long horizons worsen and swing.**
- **Scenario coverage is tiny:** 2.6% of the 2M slices seen by update 3,250 (the scenario epoch
  was 0.026).
- **Stopped** at about 3,300 updates. The suspected cause of the degradation is the learning
  rate: every continuation of E12 at lr 1e-3 with a fresh optimizer degrades towers.
- **E22 tests that:** the same recipe at lr 3e-4.
- **E22's first launch (`09pk4awu`)** died when **RunPod stopped the pod at 08:26 UTC because the
  account balance ran out** (−$0.29). It had produced no results. Relaunched after funds were
  added.
