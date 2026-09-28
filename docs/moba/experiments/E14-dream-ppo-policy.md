# E14 — Dream PPO: the policy architecture matters, the step budget matters most, the reward pays nothing for winning

**The one-line result.** Swapping our feed-forward MLP policy for PufferLib's recurrent
policy (Encoder 510→64 + 6-layer MinGRU, the old ocean-wm repo's and the shipped net's
family), with our PPO settings, gives the best dream policy so far (**arm C**). Over 120
real games against the shipped net:
- **As radiant:** return 18.1 per hero vs the net's own 14.3, with 2× the creep kills. But it
  never ends the game (0% wins, all timeouts).
- **As dire:** it loses 100% of games.

More dream optimisation makes it worse in every form we tried: 2× steps, PufferLib's
aggressive PPO settings, a lower entropy bonus. And neither the game nor the dream pays
anything for winning.

- **Status:** concluded 2026-09-28 (single seed per arm; arm C also over 120 games).
- **Code (uncommitted, experimental):** `mobawm/rl/dream_ppo.py`:
  - `--arch puffer` (`PufferPolicy`, PufferLib's `models.py` loaded from `third_party/pufferlib`);
  - the recurrent PPO update (whole 128-tick hero sequences, masked);
  - `--role-heads`;
  - `--r-tower`, `--tower-team`, `--r-win`;
  - Adam and clip flags.

  `mobawm/rl/arena.py` carries recurrent memory per hero.
- **Common recipe (P4b):**
  - world model E11 @1500;
  - 50/50 self-play / vs-net curriculum;
  - start states from expert and random-play games;
  - 64 envs, 128-tick episodes;
  - step-matched at 1,695,435 steps;
  - H200.
- **Evaluation:** `mobawm/rl/arena.py` vs the shipped net, seeds 50000000+, 10 games per side,
  unless stated.

## Arms (W&B `drift-ai-models/moba-dream-rl`)

| arm | group / run | change vs P4b | dream reward/1k (end) | real return/hero (20 games) |
|---|---|---|---|---|
| P6 base | `P6-roleheads` / `zdq4qs2a` | none (reruns P4b: **identical**, deterministic) | +1.81 | 6.40 |
| P6 role heads | `74m80azw` | role embedding + per-role action/value heads | +1.50 | 4.60 |
| P7 B | `P7-arch` / `1zl7yomf` | PufferLib MinGRU + the old repo's PPO (lr 3.4e-3, clip 0.57, vf 3.38, 1 epoch) | +7.02 (entropy 2.5) | 3.24 |
| **P7 C** | `s2aec2an` | **PufferLib MinGRU + our PPO** | +2.77 | **9.30** |
| P8 | `P8-ent-sweep` / `5z05e3kt gqha5cv0 znomih0f ps9yzz2j` | C with entropy 0.003 / 0.03 / 0.05 / 0.1 | +4.40 / +1.13 / +0.83 / +0.69 | 4.12 / 3.11 / 4.08 / 4.94 |
| P9 | `P9-puffer-2x` / `wcqdkrnr` | C at 2× steps (3.42M; first 1.70M identical to C) | +5.60 | 3.52 |
| P10 | `P10-e12-dream` / `jjdb2ezj` | C in the **E12 @4000** dream | +4.78 | 6.21 |
| P11 | `P11-reward-sweep` / `4gytv3k0 z04it194 7il3e2up dr5qofw7` | C + dream-only shaping: tower 3 to team / win ±10 / both / tower 10 team + win ±30 | +5.81 / +1.32 / +5.61 / +13.40 | 3.99 / 3.21 / 3.02 / 2.87 |

- **Arm C's policy is kept:**
  - `s3://puffertank/moba/runs/keep/p7_armC_puffer_ours_1.70M/` (policy, log.jsonl, train.log;
    md5 `3874281e`);
  - locally `runs/dream_ppo/p7/puffer_ours/`;
  - manifest `rl/P7_C_puffer_ours_1.70M`.

## All arms, real game by side

E14 arms, 1.70M dream steps each unless stated; 10 games per side, seeds 50000000+ (mobawm/rl/arena.py).

| our team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| P6 base (MLP, = P4b) | radiant | 10 | 0 | 0 | 12.80 ± 0.58 | 22.1 | 5.66 | 0.22 |
| P6 base (MLP, = P4b) | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P6 MLP + role heads | radiant | 10 | 0 | 0 | 9.19 ± 0.72 | 17.6 | 4.64 | 0.06 |
| P6 MLP + role heads | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P7 B: PufferLib MinGRU + old-repo PPO | radiant | 10 | 0 | 0 | 6.47 ± 0.52 | 9.7 | 2.96 | 0.14 |
| P7 B: PufferLib MinGRU + old-repo PPO | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P7 C: PufferLib MinGRU + our PPO | radiant | 10 | 0 | 0 | 17.98 ± 0.58 | 28.4 | 6.66 | 0.32 |
| P7 C: PufferLib MinGRU + our PPO | dire | 10 | 0 | 100 | 0.62 ± 0.21 | 0.5 | 0.72 | 0.00 |
| P8 C, entropy 0.003 | radiant | 10 | 0 | 0 | 8.15 ± 0.29 | 15.6 | 4.52 | 0.08 |
| P8 C, entropy 0.003 | dire | 10 | 0 | 100 | 0.09 ± 0.05 | 0.2 | 0.28 | 0.00 |
| P8 C, entropy 0.03 | radiant | 10 | 0 | 0 | 5.53 ± 0.42 | 12.2 | 3.74 | 0.10 |
| P8 C, entropy 0.03 | dire | 10 | 0 | 100 | 0.70 ± 0.22 | 0.7 | 1.12 | 0.00 |
| P8 C, entropy 0.05 | radiant | 10 | 0 | 0 | 7.40 ± 0.61 | 14.7 | 4.12 | 0.14 |
| P8 C, entropy 0.05 | dire | 10 | 0 | 100 | 0.75 ± 0.17 | 0.6 | 1.12 | 0.00 |
| P8 C, entropy 0.1 | radiant | 10 | 0 | 0 | 8.89 ± 0.54 | 15.2 | 4.60 | 0.20 |
| P8 C, entropy 0.1 | dire | 10 | 0 | 100 | 0.99 ± 0.33 | 1.8 | 1.46 | 0.00 |
| P9 C at 2x steps (3.42M) | radiant | 10 | 0 | 0 | 3.03 ± 0.33 | 1.9 | 2.22 | 0.02 |
| P9 C at 2x steps (3.42M) | dire | 10 | 0 | 100 | 4.01 ± 0.73 | 2.0 | 3.26 | 0.00 |
| P10 C in the E12 @4000 dream | radiant | 10 | 0 | 0 | 12.42 ± 0.54 | 27.2 | 5.62 | 0.16 |
| P10 C in the E12 @4000 dream | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| random actions | radiant | 10 | 0 | 0 | 2.82 ± 0.59 | 4.0 | 2.26 | 0.06 |
| random actions | dire | 10 | 0 | 100 | 0.24 ± 0.06 | 0.8 | 0.86 | 0.00 |
| shipped net (mirror) | radiant | 10 | 100 | 0 | 14.56 ± 1.19 | 15.0 | 5.94 | 0.98 |
| shipped net (mirror) | dire | 10 | 0 | 100 | 7.86 ± 1.02 | 7.7 | 4.60 | 0.00 |

## Arm C over 120 games, by side (`results/moba/dream/arena_real_game_p7_armC_120.jsonl`)

| team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| arm C | radiant | 60 | 0 | 0 | 18.11 ± 0.25 | 29.2 | 6.74 | 0.31 |
| arm C | dire | 60 | 0 | 100 | 0.94 ± 0.15 | 0.6 | 0.94 | 0.00 |
| shipped net (mirror) | radiant | 60 | 100 | 0 | 14.31 ± 0.41 | 14.6 | 5.85 | 0.89 |
| shipped net (mirror) | dire | 60 | 0 | 100 | 6.61 ± 0.45 | 7.1 | 4.23 | 0.00 |
| random | radiant | 60 | 0 | 0 | 2.63 ± 0.23 | 2.7 | 2.03 | 0.03 |
| random | dire | 60 | 0 | 100 | 0.31 ± 0.05 | 0.8 | 0.94 | 0.00 |

Both-sides averages: arm C 9.52, net 10.46, random 1.47.
- **As radiant:** arm C out-farms the net's own radiant play, but takes a third of the towers
  and never wins.
- **As dire:** it is crushed, much worse than the net as dire.

P10 by side: radiant 12.42 (towers 0.16), dire 0.01.

## The reward, literally

- **What the real game pays** (`third_party/pufferlib/ocean/moba/moba.h`, `moba.ini`):
  - death −0.164 to the hero that died;
  - XP ×0.0067, shared among allies in range;
  - distance, weight 0;
  - **+0.642 to the one hero landing the killing hit on a tower.**
- **Ancients are towers** (entities 204 dire, 205 radiant). Destroying the enemy ancient, which
  ends the game, pays the same +0.642 to one hero. `radiant_victory` and `dire_victory` only
  trigger the reset.
- **The dream copies this** (tower credit to the nearest enemy hero within 6 cells).
- **So nothing rewards winning,** and one tower is worth about one creep wave of XP. A policy
  maximising this reward has no reason to push, which is what arm C does.
- **P11** tests dream-only shaping (bigger team tower credit, a win/loss reward). The
  observation still shows the game's own reward components, so it keeps matching the real game.

## P11: reward shaping (negative)

Dream at 1.70M steps (reward/1k, XP/1k, towers/1k, entropy, critic loss):

| arm | reward | XP | towers | entropy | critic loss |
|---|---|---|---|---|---|
| arm C, game reward | +2.77 | 418 | 0.036 | 5.82 | 0.017 |
| tower 3 to team | +5.81 | 132 | 0.091 | 6.93 | 0.158 |
| win ±10 | +1.32 | 205 | 0.036 | 6.31 | 0.004 |
| tower 3 team + win ±10 | +5.61 | 106 | 0.055 | 6.97 | 0.155 |
| tower 10 team + win ±30 | +13.40 | 100 | 0.073 | 6.83 | 1.25 |

P11: arm C + dream-only reward shaping, 1.70M steps each; 10 games per side vs the shipped net.

| our team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| arm C: game reward | radiant | 10 | 0 | 0 | 17.98 ± 0.58 | 28.4 | 6.66 | 0.32 |
| arm C: game reward | dire | 10 | 0 | 100 | 0.62 ± 0.21 | 0.5 | 0.72 | 0.00 |
| P11 tower 3 to team | radiant | 10 | 0 | 0 | 6.66 ± 0.84 | 8.6 | 3.90 | 0.10 |
| P11 tower 3 to team | dire | 10 | 0 | 100 | 1.47 ± 0.36 | 2.0 | 1.86 | 0.00 |
| P11 win +-10 | radiant | 10 | 0 | 0 | 2.99 ± 0.36 | 6.8 | 2.66 | 0.02 |
| P11 win +-10 | dire | 10 | 0 | 100 | 3.43 ± 0.58 | 3.2 | 2.94 | 0.02 |
| P11 tower 3 team + win +-10 | radiant | 10 | 0 | 0 | 5.06 ± 0.47 | 5.3 | 3.26 | 0.08 |
| P11 tower 3 team + win +-10 | dire | 10 | 0 | 100 | 0.97 ± 0.37 | 1.1 | 1.26 | 0.00 |
| P11 tower 10 team + win +-30 | radiant | 10 | 0 | 0 | 5.72 ± 0.91 | 10.3 | 3.58 | 0.08 |
| P11 tower 10 team + win +-30 | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| shipped net (mirror) | radiant | 10 | 100 | 0 | 14.56 ± 1.19 | 15.0 | 5.94 | 0.98 |
| shipped net (mirror) | dire | 10 | 0 | 100 | 7.86 ± 1.02 | 7.7 | 4.60 | 0.00 |

- **No shaped arm wins or takes more towers.** Radiant towers fall from 0.32 to 0.02–0.10.
- **Every shaped arm learned less:** entropy stays near uniform, and the critic loss is 10–70×.
  Large, rare bonuses dominate the normalised advantages and drown the XP and creep signal.
- **Win ±10 is the only arm that improved dire** (3.43 vs 0.62), at a large radiant cost.
- **Likely cause:**
  - in 128-tick dream episodes towers and ancients fall rarely;
  - tower health is a known world-model weak spot (W14), so rewarded tower events may be
    invented by the dream.

## P12/P13: tower-damage and falling-tower shaping

- **What was shaped:**
  - **tower damage:** each tick a tower loses health, every enemy hero within 8 cells gets
    k × (lost/max) × (1 + boost × (1 − health left)); ancients count ×2;
  - **falling tower:** each enemy hero within radius R gets bonus × (1 − distance/R).
- **Offline check first,** on 88 real tower falls:
  - summed damage per fallen tower is exactly 100% of its health;
  - no revivals or health regen;
  - 71 of 88 falls had an enemy hero within 8 cells;
  - 87 of 88 paid only the attacking team.
- **Runs:** W&B `P12-tower-shaping`: `lz7opfch r1titdgl jd1xie1g n9o3hwu0`; `P13-fall-sweep`:
  `6l122crp ojvhv6dm iuwujj5g j75qsm75`.

Arm C + dream-only tower shaping, 1.70M steps each, single seed; 10 games per side vs the shipped net.

| our team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| arm C: game reward | radiant | 10 | 0 | 0 | 17.98 ± 0.58 | 28.4 | 6.66 | 0.32 |
| arm C: game reward | dire | 10 | 0 | 100 | 0.62 ± 0.21 | 0.5 | 0.72 | 0.00 |
| P12 tower damage (k2, boost 2) | radiant | 10 | 0 | 0 | 14.00 ± 0.46 | 22.3 | 5.96 | 0.36 |
| P12 tower damage (k2, boost 2) | dire | 10 | 0 | 100 | 0.03 ± 0.02 | 0.1 | 0.08 | 0.00 |
| P12 falling-tower bonus 5 | radiant | 10 | 0 | 0 | 15.74 ± 0.36 | 22.6 | 6.18 | 0.56 |
| P12 falling-tower bonus 5 | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P12 damage + fall | radiant | 10 | 0 | 0 | 7.62 ± 0.31 | 13.4 | 4.44 | 0.06 |
| P12 damage + fall | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P12 damage + fall, big | radiant | 10 | 0 | 0 | 8.60 ± 0.42 | 14.3 | 4.62 | 0.04 |
| P12 damage + fall, big | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P13 fall bonus 15 | radiant | 10 | 0 | 0 | 11.05 ± 0.34 | 18.4 | 5.34 | 0.18 |
| P13 fall bonus 15 | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P13 fall bonus 30 | radiant | 10 | 0 | 0 | 16.63 ± 0.72 | 27.1 | 6.60 | 0.38 |
| P13 fall bonus 30 | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P13 fall bonus 60 | radiant | 10 | 0 | 0 | 11.75 ± 0.77 | 18.0 | 5.50 | 0.28 |
| P13 fall bonus 60 | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P13 fall bonus 120 | radiant | 10 | 0 | 0 | 12.39 ± 0.66 | 22.5 | 5.52 | 0.24 |
| P13 fall bonus 120 | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| shipped net (mirror) | radiant | 10 | 100 | 0 | 14.56 ± 1.19 | 15.0 | 5.94 | 0.98 |
| shipped net (mirror) | dire | 10 | 0 | 100 | 7.86 ± 1.02 | 7.7 | 4.60 | 0.00 |

- **The falling-tower bonus at 5** gives the most real towers of any dream policy (0.56 vs
  arm C's 0.32; the net 0.98), but no wins.
- **Damage shaping** mostly paid for dream-only tower damage (dream towers 6×, real ≈ same).
- **Bigger bonuses (15–120) don't help:** single seed, non-monotonic.
- **Every shaped arm is ≈ 0 as dire.**

## Dream episode length (local smoke, 3 min each, E11 @1500)

- **Setup:** 128-tick vs 512-tick dream episodes.
- **Result:** at 512 ticks the dream produces 4.6× less XP and 3.6× fewer tower falls per step
  (and PPO gets 4× fewer updates).
- **Stability:** no state repairs.
- **Reading:** far from a real start state the dream goes quiet, so longer episodes don't give
  PPO more pushing to learn from. The tower audit (E15: 512-tick precision 13%) agrees.

## Conclusions

1. **Architecture:** the recurrent PufferLib policy beats our MLP (9.30 vs 6.40, same PPO,
   single seed). Per-role heads on the MLP did not help (4.60).
2. **Over-optimisation is the dominant failure:** 2× steps (3.52), aggressive PPO (3.24) and
   low entropy (4.12) all raise dream reward and lower real return. The entropy sweep is
   non-monotonic around C, so seed variance is large and unmeasured.
3. **The E12 dream (+ dyna2) did not help at this budget** (P10: 6.21). Its dream reward rose
   faster.
4. **Side asymmetry:** strong as radiant, helpless as dire. The shipped net was trained only as
   radiant. Our curriculum plays against it on both sides, but dire starts face a net radiant
   that pushes.
5. **Rewarding towers and wins in the dream did not create pushing** (P11). The candidate levers:
   - longer dream episodes, or start states near enemy towers, so pushing happens in the dream;
   - better tower prediction in the world model;
   - a dire-specific curriculum.
