# E13 — Dream PPO in E11: first dream-trained policies above random in the real game

**The one-line result.** In E11's dream, PPO with a 50/50 self-play / vs-net curriculum
earns 2.2× the untrained policy's real return against the shipped net (P4). Starting
episodes from both expert and random-play states doubles that again, to 4.2× random and
57% of the net's return, with creep kills level with the net (P4b). Still 0 wins. Every
E5-dream policy (E7) was below random.

- **Status:**
  - P4 and P4b concluded (4.5 min each, single seed).
  - **P5, the P4b recipe at 100× the steps (169,543,500):** stopped by the author at
    ~88M steps (2026-09-28). W&B `moba-dream-rl/jfw1p3b5`. All 57 files (the latest
    policy plus numbered checkpoints every ~1.6M steps) are in
    `s3://puffertank/moba/runs/p5_mix_e11_mixstarts_100x/` and in
    `runs/token_wm/p5_mix_e11_mixstarts_100x/`.
- **Code:** `mobawm/rl/dream_ppo.py --opponent mix --net-frac 0.5` (uncommitted; code snapshot
  `code_snapshot_20260928_0215`).
- **Evaluation:** `mobawm/rl/arena.py`, the P2/P3 harness (same seeds).

## Arms

| arm | W&B | world model | opponent | start states | agent steps |
|---|---|---|---|---|---|
| P4 | `msjwyq7r` | E11 @1500 | 50% self-play / 50% shipped net (per game) | any tick of mix1k 0–19 (expert) | 2.12M |
| P4b | `ahoexos3` (`gqwfjvhm` = crashed first launch) | E11 @1500 | same | any tick of mix1k 0–99 + 100 mixrand1k training games (20 per matchup) | 1.70M |
| P5 | `jfw1p3b5` | E11 @1500 | same | as P4b | 83.8M of 169.5M (stopped) |

Everything else is P3's recipe: 64 envs, episode 128, rollout 128, 2 epochs, 2
minibatches, lr 3e-4, H200. Start states were already uniform over all ticks, never only
tick 0. P4b adds random-play states to the pool.

## Real game vs the shipped net

Real game, seeds 50000000+, 10 games per side, max 6,000 ticks (mobawm/rl/arena.py, the P2/P3 harness).

| our team (vs shipped net) | games | win % | return/hero | deaths | level | creeps | towers |
|---|---|---|---|---|---|---|---|
| P3 self-play (E5 dream, 4.5 min) | 20 | 0 | 0.01 | 0.00 | 0.02 | 0.02 | 0.00 |
| P3 vs net (E5 dream, 4.5 min) | 20 | 0 | 0.02 | 0.01 | 0.08 | 0.03 | 0.00 |
| P4 mix, expert starts (E11 @1500, 4.5 min) | 20 | 0 | 3.38 | 0.33 | 1.96 | 7.15 | 0.04 |
| P4b mix, expert + random starts (E11 @1500, 4.5 min) | 20 | 0 | 6.40 | 2.05 | 2.83 | 11.04 | 0.11 |
| random actions | 20 | 0 | 1.53 | 0.56 | 1.56 | 2.38 | 0.03 |
| shipped net | 20 | 50 | 11.21 | 7.12 | 5.27 | 11.35 | 0.49 |

## P5: more dream training made real play worse

- **Dream reward** (10M-step means): +3.1 (0–10M), +3.6, +6.6, +4.8, +5.9, +2.5, +2.9, +6.5
  (70–80M). Flat since ~20M steps (slope +0.013 per M overall). The swings track dream XP
  (384–988 per 1k).
- **Real game at 81.3M steps** (`policy_081342665`, table above): return 2.34 vs P4b's 6.40
  at 1.70M; creeps 3.6 vs 11.0. That is the E7 pattern again (R7): once the real behaviour is
  learned, longer optimisation mostly finds the dream's errors.
- **Not measured:** where real play peaks between 1.7M and 81M. The numbered checkpoints
  allow that sweep.

## Dream reward over training (P4)

- **Dream reward per 1k hero-steps:** +0.65 at 0.06M steps → +3.10 over the last 5 updates
  (2.12M).
- **Slope:** +1.32 per M steps overall, +2.52 over the second half. Still rising when the
  4.5-minute budget ended.
- **Dream XP:** 109 → 459 per 1k.
- **Source:** `results/moba/dream/p4_mix_e11.log.jsonl`; Runs `msjwyq7r`.

## Dream vs real for the P4 policy (why the dream reward overstates progress)

scripts/moba/dream_vs_real_reward.py: 8 real games x 900 ticks vs the shipped net, seeds 70000000+; reward = death + XP + tower terms per 1k hero-steps of our team.

| our team | world model | real reward/1k | dream, teacher-forced | dream, open loop 128 | real XP/1k | dream XP TF | dream XP OL |
|---|---|---|---|---|---|---|---|
| P4 policy | E5 @20,500 | 1.21 | 86.31 | 14.28 (real 1.20) | 188 | 12969 | 2146 (real 186) |
| P4 policy | E11 @1500 | 1.45 | 19.13 | 5.64 (real 1.43) | 226 | 2874 | 848 (real 225) |
| random | E5 @20,500 | 0.09 | 55.24 | 6.77 (real 0.08) | 15 | 8298 | 1031 (real 13) |
| random | E11 @1500 | 0.09 | 0.36 | 0.00 (real 0.08) | 15 | 54 | 0 (real 13) |
| net | E5 @20,500 | 3.79 | 5.96 | 2.34 (real 3.68) | 624 | 741 | 385 (real 612) |
| net | E11 @1500 | 3.79 | 10.84 | 1.70 (real 3.68) | 624 | 1362 | 268 (real 612) |

## Conclusions

1. **The E11 dream teaches something real:** above random for the first time (P4, P4b). The
   world model changed at the same time as the curriculum, so this is not an ablation.
2. **Mixed start states help (P4 → P4b ~2×),** with fewer steps. Single seed each.
3. **The dream still over-pays the policy (~3.9×) and under-pays the net (~0.5×) over 128
   ticks,** and P5 confirms it: at 81M steps real return fell to 2.34, from P4b's 6.40.
