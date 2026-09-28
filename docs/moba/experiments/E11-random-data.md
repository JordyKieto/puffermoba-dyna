# E11 — E5 resumed on random-play data: the dream stops hallucinating for random play

**The one-line result.** Adding 1,000 games with a uniform-random controller (`mixrand1k`)
to E5's data cuts held-out open-loop loss **on random-play games** 4–8× at horizons
16–256 within 500 updates. It removes the dream's XP hallucination for random actions
(teacher-forced 8,298 → 54 XP per 1k hero-steps; real 15). On the original expert
held-out set it stays within ~0–25% of E5 at 16–1,024 ticks, but one-step and h4096 are
worse.

- **Status:** stopped at update ~2,300 (2026-09-28 01:51 UTC) to start E12 from @2000.
- **W&B:** `moba-wm/mlbh5d3l` (group `E11-random-data`), exported to `results/moba/mlbh5d3l/`.
- **Checkpoints:** @1500 and @2000 in the manifest (`wm/e11_at1500`, `wm/e11_at2000`).
  @500 and @1000 were overwritten: `model.pt` is replaced at every eval, and numbered
  checkpoints were every 2,000. From E12 on, every eval checkpoint is kept.

## Setup

- **Init:** E5 @20,500 (`runs/token_wm/e5_rollout`, md5 `a61d64ae`), with E5's normalisation stats.
- **Recipe:** exactly E5's: `--baseline --rollout --bf16`, batch 768, lr 1e-3 (fresh optimizer),
  chunk 16, burn 4, log-normal spans μ6 σ3 up to 4,096, step decay half-life 64 / floor 0.02,
  `--store-cpu`, H100 NVL, 0.43–0.45 updates/s.
- **Data:** mix1k 0–899 + mixrand1k minus the held-out files: 1,780 games, 4,691,545 transitions.
- **Held-out sets** (neither trained on):
  - **random held-out (main metric):** the last whole files of each mixrand1k matchup block,
    100 games: 1140–1159 random/net, 1310–1329 net/random, 1620–1659 random/random,
    1810–1829 random/script, 1980–1999 script/random. `train.py --held2-games`, logged as
    `random/...`.
  - **mix1k 900–999,** E5's own held-out set, for comparability.
- **mixrand1k:** 1,000 games (DATA.md), replayed on the pod with 0 hash and 0 observation
  mismatches over 2,608,592 steps (`runs/pod/data_mixrand1k.log`).

## Result: held-out loss by open-loop horizon

E11 (W&B mlbh5d3l): update 0 = E5 @20,500. 32 fixed windows per horizon; x = loss relative to update 0.

| held-out set | update | h1 | h4 | h16 | h64 | h256 | h1024 | h4096 | hero cells @1024 |
|---|---|---|---|---|---|---|---|---|---|
| random held-out | 0 | 4.05 | 8.33 | 45.83 | 132.97 | 165.42 | 182.34 | - | 72.3 |
| random held-out | 500 | 3.60 (0.89x) | 4.50 (0.54x) | 10.72 (0.23x) | 23.33 (0.18x) | 48.73 (0.29x) | 82.44 (0.45x) | - | 27.2 |
| random held-out | 1000 | 3.92 (0.97x) | 6.89 (0.83x) | 11.02 (0.24x) | 22.15 (0.17x) | 45.33 (0.27x) | 98.18 (0.54x) | - | 35.2 |
| random held-out | 1500 | 4.94 (1.22x) | 7.72 (0.93x) | 9.78 (0.21x) | 18.86 (0.14x) | 32.23 (0.19x) | 48.28 (0.26x) | - | 16.5 |
| random held-out | 2000 | 3.64 (0.90x) | 4.49 (0.54x) | 8.23 (0.18x) | 17.17 (0.13x) | 35.31 (0.21x) | 60.45 (0.33x) | - | 21.5 |
| mix1k held-out 900-999 | 0 | 1.70 | 5.14 | 7.70 | 18.99 | 24.64 | 34.84 | 36.84 | 5.5 |
| mix1k held-out 900-999 | 500 | 2.11 (1.24x) | 5.30 (1.03x) | 8.24 (1.07x) | 20.65 (1.09x) | 32.20 (1.31x) | 51.76 (1.49x) | 90.35 (2.45x) | 6.2 |
| mix1k held-out 900-999 | 1000 | 2.50 (1.46x) | 6.00 (1.17x) | 8.07 (1.05x) | 21.76 (1.15x) | 26.11 (1.06x) | 36.30 (1.04x) | 68.31 (1.85x) | 5.7 |
| mix1k held-out 900-999 | 1500 | 3.79 (2.23x) | 8.19 (1.60x) | 8.89 (1.15x) | 20.29 (1.07x) | 26.98 (1.09x) | 33.70 (0.97x) | 77.18 (2.09x) | 5.5 |
| mix1k held-out 900-999 | 2000 | 2.33 (1.36x) | 5.89 (1.15x) | 7.91 (1.03x) | 19.26 (1.01x) | 30.41 (1.23x) | 39.58 (1.14x) | 50.30 (1.37x) | 7.6 |

**How to read it:**
- Update 0 is E5 itself.
- "x" is the loss relative to E5 on the same 32 fixed windows (below 1 is better).
- The random games are capped at 3,000 ticks, so they have no h4096.

## The dream-vs-real audit (E11 @1500 vs E5)

scripts/moba/dream_vs_real_reward.py: 8 real games x 900 ticks vs the shipped net, seeds 70000000+; reward = death + XP + tower terms per 1k hero-steps of our team.

| our team | world model | real reward/1k | dream, teacher-forced | dream, open loop 128 | real XP/1k | dream XP TF | dream XP OL |
|---|---|---|---|---|---|---|---|
| P4 policy | E5 @20,500 | 1.21 | 86.31 | 14.28 (real 1.20) | 188 | 12969 | 2146 (real 186) |
| P4 policy | E11 @1500 | 1.45 | 19.13 | 5.64 (real 1.43) | 226 | 2874 | 848 (real 225) |
| random | E5 @20,500 | 0.09 | 55.24 | 6.77 (real 0.08) | 15 | 8298 | 1031 (real 13) |
| random | E11 @1500 | 0.09 | 0.36 | 0.00 (real 0.08) | 15 | 54 | 0 (real 13) |
| net | E5 @20,500 | 3.79 | 5.96 | 2.34 (real 3.68) | 624 | 741 | 385 (real 612) |
| net | E11 @1500 | 3.79 | 10.84 | 1.70 (real 3.68) | 624 | 1362 | 268 (real 612) |

- **Random actions:** E11 pays essentially no fake XP (54 vs 8,298 teacher-forced; 0 vs
  1,031 open loop). This is the targeted failure from E8, fixed by data.
- **Dream-PPO actions:** still over-paid, but less: open-loop reward is 5.64 against a real
  1.43, where E5 gave 14.3.
- **The shipped net is under-paid:** open-loop 1.70 against a real 3.68. The dream still
  **ranks the PPO policy above the net**, which is wrong in the real game (E13).

## Conclusions

1. **The E8 hallucination was a data-coverage problem**, at least for random play. Adding
   that play fixes it without any rule or loss change.
2. **The cost is the expert-play horizons:** h1 is 1.4–2.2× worse and h4096 1.4–2.5× worse.
   One seed, and a fresh optimizer; E12 continues training.
3. **Not tested:** a different mix ratio, or training from scratch.
