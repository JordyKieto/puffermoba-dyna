# E7 — PPO inside the world model's dream: the dream reward rose, the real game did not (NEGATIVE)

**The one-line result.** A policy trained only inside E5's dream more than doubles its
dream reward in 2.4M agent steps. In the real C game it wins 0 of 120 games and earns
less than an untrained policy (M: `results/moba/81jwi4lw/`,
`results/moba/dream/arena_real_game_lr3e-4_mb2.jsonl`).

- **Status:** concluded, 2026-09-25/26, as a first attempt. Nothing is running.
- **Code:** `mobawm/rl/dream_ppo.py`, `mobawm/rl/arena.py`, `moba/moba_arena.c`.
  **Uncommitted**; the exact version is in the code snapshot
  (`../PROVENANCE.md` §1).
- **W&B:** `drift-ai-models/moba-dream-rl`, backfilled from local logs, groups
  `P0-first-5min`, `P1-sweep-mac`, `P1-sweep-h200`, `P2-arena-real`. Exported to
  `results/moba/<run_id>/`. The table of run ids is in `../PROVENANCE.md` §4.

## Setup

- **The dream:** E5 @20,500 (`a61d64aed9c2`), frozen.
  - Every episode starts from a real state of a training game (mix1k 0–19), with 8
    real ticks to warm the model's memory.
  - All of the world model's rules R1–R10 are active (spawns, level, etc.).
- **Players:** all 10 heroes are played by **one shared policy**. Radiant and dire are
  the same network, and each hero maximises its own reward.
- **Observation:** each hero's 510-byte ego observation, rebuilt byte-exact from the
  dreamed state (claim T2).
- **Reward:** the game's own per-hero reward, computed from dreamed state changes:
  - −0.164 on death (a >12-cell jump with health reset);
  - +0.0067 per XP;
  - +0.642 for a tower kill, credited to the nearest enemy hero within 6 cells (the
    dream has no kill credit).
- **Policy:** an MLP over the observation (tile embedding plus scalars), the game's 6
  action heads, PPO with GAE (γ 0.99, λ 0.95), clip 0.2, entropy 0.01.
- **State guard:** any non-finite value falls back to the previous state, and
  categorical fields are clamped. 0 repairs were needed in P1-h200.

## P0 — the first 5 minutes (Mac MPS)

| | first 4 updates | last 4 updates |
|---|---|---|
| deaths / 1k hero-steps | 2.00 | 1.00 |
| critic explained variance | 0.02 | 0.80 |
| entropy | 7.06 | 6.23 |
| dream reward / 1k | 4.02 | 3.48 |

- **Throughput:** about 1,040 agent steps/s; the PPO update took 70% of the time.
- **Why reward looks flat:** episodes were 512 ticks, 16 updates long, so the
  per-update reward mostly measured the **episode phase**, not policy quality.
- **Fix, used from P1 on:** one fresh 128-tick episode per update, with the same
  seeded start states in every run.

## P1 — LR × minibatch sweep

- **On the Mac, inconclusive:** 6 configs in parallel on one Mac GPU got only 2–4
  updates each; entropy never moved.
  - It is recorded because it happened, and because it produced a crash that exposed
    the need for the state guard. The garbage team index was likely an MPS glitch:
    0 repairs later on the H200.
- **On the H200:** 6 configs in parallel, 4.5 min each, about 2.4M agent steps and
  ~30 updates per config, about 9,000 agent steps/s per config.

Dream reward per 1k hero-steps (single seed per config); the table is generated in
`results/moba/tables/dream_ppo_sweep.md`:

| config (lr, minibatches) | reward/1k first | reward/1k last 3 | deaths/1k last 3 |
|---|---|---|---|
| lr 3e-4, 2 (large minibatches) | 4.47 | 13.91 | 0.28 |
| lr 3e-3, 10 | 4.47 | 13.30 | 1.02 |
| lr 1e-3, 10 | 4.47 | 11.33 | 0.89 |
| lr 3e-4, 10 | 4.47 | 10.88 | 0.91 |
| lr 1e-3, 2 | 4.47 | 10.98 | 0.93 |
| lr 3e-3, 2 | 4.47 | 9.20 | 1.14 |

Figure: `results/moba/figures/dream_ppo_sweep.png`. The lr 3e-4 / 2-minibatch config
went to P2.

## P2 — the real C game

The best P1 policy played one team of the real game (`moba_arena.c`). The other team
was the shipped PufferNet or the scripted bots.
- **Matchups:** 10 games per side × 2 sides per matchup, seeds 50,000,000+, max
  6,000 ticks.
- **Baselines on the same seeds:** the untrained network (uniform actions) and the
  shipped net playing our side.

| our team | vs | win % | loss % | return/hero | deaths/hero | level |
|---|---|---|---|---|---|---|
| **dream PPO** | net | 0 | 50 | 1.20 | 0.19 | 1.26 |
| **dream PPO** | script | 0 | 100 | 1.79 | 0.37 | 1.42 |
| untrained | net | 0 | 50 | 1.82 | 0.54 | 1.54 |
| untrained | script | 0 | 100 | 6.38 | 1.02 | 3.61 |
| shipped net | net | 50 | 50 | 11.21 | 7.12 | 5.27 |
| shipped net | script | 45 | 55 | 18.89 | 3.73 | 7.10 |

Generated: `results/moba/tables/arena_real_game.md`. Figure:
`results/moba/figures/arena_real_game.png`.

**Conclusions:**
1. **Dream reward is not a proxy for real reward here.** Dream XP rose about 3×, while
   real XP fell below the untrained policy's.
2. **The only transferred behaviour is not dying:** the fewest deaths of any team,
   mostly by being passive (level ~1.3).
3. **The harness is sound:** the shipped net scores sensibly (50% in the mirror
   match), so the negative result is about the policy, not the arena.

## P3 — training against the shipped net instead of self-play (2026-09-27): still negative

**Question:** does a policy trained against the shipped PufferNet inside the dream
transfer better than self-play?

**Setup:**
- **Arms:** two, in parallel on one H100 NVL, 4.5 min each, the P1 best recipe (lr 3e-4,
  2 minibatches, 64 games × 128 ticks, E5 @20,500 dream, games 0–19, seed 0).
  - **self-play rerun:** as in P1/P2.
  - **vs net:** our policy plays one randomly drawn side per game. The **shipped
    PufferNet** plays the other side inside the dream, via `ma_pool_*` in
    `moba/moba_arena.c`. Its logits were checked **bit-identical** to the collector's
    `net_act` over 50 real ticks (max difference 0). Only our 5 heroes are trained and
    scored.
- **W&B:** `moba-dream-rl`, group `P3-vs-net-ab`: `1q729jk7` (self-play), `iluni10r`
  (vs net), both logged live. Exported to `results/moba/<id>/`.
- **Checkpoints:** uploaded to S3 automatically as they were written.

**Compute was not matched.** The shipped net runs in C on the CPU (single-threaded, 640
heroes per tick), and only 5 of 10 heroes train per game. So in the same 4.5 min:

| arm | agent steps |
|---|---|
| vs net | 1.10M |
| self-play rerun | 9.74M |
| P1 original | 2.46M |

**Real game, same seeds and harness as P2.** Rows `results/moba/dream/arena_real_game_p3_*.jsonl`. The random and
shipped-net rows reproduce P2 exactly, so the harness is deterministic.

| our team | vs | win % | return/hero | deaths/hero | level |
|---|---|---|---|---|---|
| P3 vs-net PPO | net | 0 | 0.02 | 0.01 | 0.08 |
| P3 vs-net PPO | script | 0 | 0.86 | 0.35 | 0.78 |
| P3 self-play rerun | net | 0 | 0.01 | 0.00 | 0.02 |
| P3 self-play rerun | script | 0 | 3.75 | 0.47 | 2.35 |
| P2 self-play (reference) | net | 0 | 1.20 | 0.19 | 1.26 |
| untrained | net | 0 | 1.82 | 0.54 | 1.54 |
| untrained | script | 0 | 6.38 | 1.02 | 3.61 |

**Dream reward at the end:** self-play 23.98 per 1k hero-steps, vs net 22.05 (dream XP
~3,300–3,600 per 1k).

**Conclusions:**
1. **The opponent is not the problem.** Training against the shipped net transfers no
   better than self-play. Both are below the untrained policy against the net.
2. **More dream optimisation makes the real game worse.** The self-play rerun had 4×
   P1's steps and twice its dream reward, yet scores lower against the net in the real
   game (0.01 vs 1.20) and barely levels (0.02).
3. **The strongest evidence yet for model exploitation (claim R4):** the policies
   collect XP in the dream while doing almost nothing in the real game (near-zero
   deaths, near-zero levels). The fix belongs on the world-model side: train it on the
   policy's own real games (the Dyna loop) before more dream RL.

**Kept:**
- `runs/dream_ppo/p3/{vs_net,selfplay_rerun}/policy.pt` (md5 `a4566ca5e6b7` /
  `8c2887538c66`, byte-identical in `s3://puffertank/moba/runs/dream_ppo/p3/`);
- logs in `results/moba/dream/p3_*.log.jsonl`.

## What this does not show

- **It does not show that dream training cannot work.** It is one world model, one
  policy architecture and one short run, and every sweep result is a single seed.
- **The cause is not established.** Model exploitation (the policy finding states
  where the dream pays out wrongly) is the leading hypothesis (claim R4, **U**). The
  distribution shift from self-play to fixed opponents is the other.

## What to run next, in order

1. **Diagnose:** replay the P2 policy's real-game action sequences through the world
   model from the same starting states, and compare dreamed vs real XP, deaths and
   positions.
2. **Close the loop (Dyna-style):** add the policy's real games to world-model
   training, retrain PPO in the improved dream, and repeat.
3. **Guard the reward against model uncertainty,** for example where E5 and E6b
   disagree.
4. **Evaluate every sweep config in the real game,** not only the dream-best one.

## Kept

- **Policies** (md5 in `artifacts/moba/MANIFEST.json`, byte-identical in
  `s3://puffertank/moba/runs/dream_ppo/`):
  - `runs/dream_ppo/e5_selfplay_5min/policy.pt`;
  - `runs/dream_ppo/sweep1_gpu/<cfg>/policy.pt` for all 6 configs.
- **Logs:** `results/moba/dream/*.log.jsonl`, and the per-game arena records in
  `results/moba/dream/arena_real_game_lr3e-4_mb2.jsonl`.
