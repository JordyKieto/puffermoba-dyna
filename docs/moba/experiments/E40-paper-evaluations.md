# E40: paper evaluations: holdout win rates, baselines, the contribution ladder, the frozen-world-model control, Dyna world-model metrics, and how the policy plays

**Status:** concluded 2026-10-05 (the world-model mechanics evaluation is in `results/moba/paper/wm_mechanics.csv`).

**Verdict:**
- **Headline.** The best Dyna-loop policy (ppol4 u69) wins **421/600 = 70.2% [66.4, 73.7]** of real radiant games against
  the shipped net, on two seed sets never used for any decision. It wins 0/600 as dire.
- **The shipped net against itself** wins **300/300 as radiant** (in ~2,270 ticks) and 0/300 as dire. Under this opponent
  the game is completely radiant-sided, so the dire 0% is the base rate. Our policy is still the weaker radiant player:
  70% vs 100%, at 2.5× the game length.
- **The random policy** wins 0/300 as radiant (0.17 towers per game). The radiant edge alone wins nothing.
- **The ladder, one ingredient per rung** (radiant, holdout):

  | rung | radiant wins |
  |---|---|
  | random | 0% |
  | E14 dream PPO | 0% |
  | + long episodes + win reward | 1.3% |
  | + PPO fleet | 5.7% |
  | + shaped reward, XP ×0.5 | 33.7% |
  | + continuous Dyna loop | 53.0% |
  | + death 0 + KL anchor | 70.2% |

- **World-model quality vs PPO, the matched control.** Same start (u71), same recipe (death 0 + KL 0.3 to u71), 40 updates,
  2 seeds each. Only the world model differs, frozen at either:
  - the pre-Dyna E38 dream: 49.0% [45.0, 53.0];
  - the Dyna-trained model: 58.2% [54.2, 62.0].

  **+9.2 points [3.5, 14.7] from the world model alone.**
- **What Dyna did to the world model.** On the policy's own real games, none of which any evaluated model trained on:
  - tower-fall precision 28.6% → 51.8%, predicted/real falls 1.79 → 1.16;
  - hero position error at 200 ticks of open loop 22.8 → 9.7 cells (persistence 28.5);
  - the pre-Dyna dream is **worse than holding the state still** for hero HP and tower HP at 200 ticks.
- **How the policy plays.** It does not win by teamwork. Compared with the shipped net's radiant play:
  - grouped (3+ heroes within 8 cells) 9% of ticks vs 54%;
  - 0.65 of our heroes at an enemy tower's fall vs 2.05;
  - 9.4 kills / 30.7 deaths per game vs 35.7 / 30.0;
  - support heals reach 0.13 allies vs 0.83;
  - a stun before 7% of kills vs 26%.

  It spreads across every lane and the jungle, keeps pressure with the creep waves, trades lives, and grinds ~10 towers
  before the ancient falls. The net marches its five heroes down the middle lane together and wins in less than half the
  time.

## 1. Seed-set evaluations

All rows are in `results/moba/paper/seed_evals.jsonl` (`scripts/moba/seed_eval.py`, CPU pod evc5, 60 per side per
chunk); per-game rows are in `results/moba/paper/games/`.

| policy | set | radiant | dire | towers / game (radiant) | game length radiant / dire (ticks) |
|---|---|---|---|---|---|
| random (untrained PufferPolicy) | holdout | 0/300 | 0/300 | 0.17 | 6,000 / 1,087 |
| shipped net (vs itself) | holdout | 300/300 | 0/300 | 4.39 | 2,269 / 2,269 |
| E14 arm C | holdout | 0/300 | 0/300 | 1.65 | 6,000 / 1,083 |
| E36 best (`J_envs128_4x_s2`) | holdout | 4/300 | 0/300 | 2.88 | 5,993 / 1,078 |
| E37 pf1 u7 | holdout | 17/300 | 0/300 | 3.03 | — |
| E38 u0017 | holdout | 101/300 | 0/300 | 3.37 | 5,744 / 1,230 |
| ppol u71 | holdout | 159/300 | 0/300 | 3.77 | 5,652 / 1,220 |
| ppol4 u10 | holdout | 167/300 | 0/300 | 3.86 | 5,630 / 1,263 |
| ppol4 u49 | holdout | 203/300 | 0/300 | 3.90 | 5,486 / 1,270 |
| **ppol4 u69** | holdout | **213/300** | 0/300 | 3.97 | 5,542 / 1,290 |
| **ppol4 u69** | holdout2 | **208/300** | 0/300 | — | 5,537 / 1,287 |
| E40 frozen E38 dream, seed 0 / 1 | holdout | 153 / 141 of 300 | 0 | 3.15 / 3.31 | ~5,630 |
| E40 frozen Dyna WM, seed 0 / 1 | holdout | 168 / 181 of 300 | 0 | 3.53 / 3.73 | ~5,590 |

- **Determinism check:** u0017, u71, u10 and u49 were first evaluated on the holdout with an earlier script (E39) and
  re-run here. Every count is identical.
- **Selection inflation:** u69 won 223/300 on the fresh seeds, which picked it, and 213 and 208 on the two untouched
  sets: about 4 points. The same effect for E38's u0017: 45% on the fixed seeds, 34% on holdout.
- **Dire game length:** as dire, our policies lose in ~1,250 ticks, the random policy in ~1,090 and the shipped net in
  ~2,270. Our dire play is barely better than random.

## 2. The frozen-world-model control

- **Launcher:** `scripts/moba/sweeps/fz_arm.py`. 2-pod FedAvg fleets, 128 games per pod, 1,400-tick episodes, lr 5e-5,
  death 0, KL 0.3 toward u71 (also the start; the anchor never moves), no resets, 40 updates.
- **Start states:** mix1k games 0–99.
- **World models** (frozen for the whole run):
  - `runs/token_wm/fleet20e_at500` (md5 `d843b9ff`);
  - `runs/token_wm/wml2_at3400` (md5 `19f8df99`).
- **Fixed-seed curves:** `results/moba/dream/fz_*_eval.jsonl` (Fig. 8 right).
- **Endpoints:** the holdout rows above.
- **Not separated by this control:** the remaining gap from 58% (frozen Dyna WM, fixed anchor, 2 pods, 40 updates) to 70%
  (continuous loop: refreshed WM, moving anchor, 8 pods, 69 updates).
- **Incident:** one pod failed its bootstrap twice (raylib download) and was replaced (`fz_e38r0b`), so `fz_e38` started
  ~20 minutes late. Results are compared by update.

## 3. Dyna world-model metrics (the policy's own real games)

**Data:** 10 collector files (100 games of ppol4, updates ~83–106), uploaded after `wml2`'s checkpoint 3,400.
`runs/pod/dyna_eval_set.json` lists them.

**Tower evaluation** (`scripts/moba/tower_eval_big.py`, 5,265 windows of 128 ticks, on dl0):

| world model | precision | recall | predicted / real tower falls |
|---|---|---|---|
| E38 dream (`fleet20e_at500`) | 28.6% | 51.0% | 2,502 / 1,401 = 1.79 |
| Dyna round 1 (`wm_r1`) | 39.8% | 50.7% | 1.27 |
| `wml` @3600 (one pod for most of its life) | 28.5% | 60.9% | 2.13 |
| **`wml2` @3400** | **51.8%** | **60.1%** | **1.16** |

**Debugger divergence** (`scripts/moba/debugger_eval.py`, 300 windows per model: 30 games × start ticks 500, 1,500, 3,000
and 4,500 where the game is long enough; open loop on the real recorded actions). Mean error at 200 ticks:

| world model | hero position (cells) | hero HP | tower HP | creep alive agreement |
|---|---|---|---|---|
| E38 dream | 22.8 | 127.5 | 87.0 | 82.4% |
| `wm_r1` | 13.6 | 79.6 | 64.0 | 82.4% |
| `wml` @3600 | 9.6 | 72.7 | 46.8 | 81.6% |
| `wml2` @3400 | 9.7 | 69.4 | 42.7 | 82.3% |
| persistence | 28.5 | 104.9 | 48.8 | — |

Creep alive agreement is propped up by the spawn rules (R1) and is the same for every model.

**Mechanics** (`scripts/moba/wm_mechanics_eval.py`; per-hero event matching within ±4 ticks; 64-tick windows every
200 ticks; batched rollouts, whose float reduction order differs slightly from the debugger's single rollout, so an
occasional discrete outcome flips: unbiased, and the same for every model). Pre-Dyna → Dyna `wml2`, each as
precision / recall / predicted÷real:

| mechanic | pre-Dyna | Dyna `wml2` |
|---|---|---|
| casts | 19% / 50% / 2.59 | 24% / 48% / 2.02 |
| tower falls | 14% / 23% / 1.68 | 33% / 35% / 1.06 |
| hero deaths | 48% / 15% / 0.32 | 47% / 20% / 0.42 |
| heals | 6% / 6% / 0.96 | 3% / 3% / 0.91 |
| stuns | 2% / 2% / 0.67 | 2% / 0% / 0.17 |
| slows | 3% / 2% / 0.52 | 3% / 1% / 0.34 |
| displacements | 4% / 23% / 6.16 | 5% / 16% / 3.55 |

- **The dream learns rates and the aggregate flow of fights,** not the timing of individual abilities.
- **Crowd control is essentially absent.**
- **Deaths are under-predicted 2–3×.**

## 5. Whole-project compute, cost and data

`scripts/moba/compute_accounting.py` → `results/moba/paper/compute.json`. From RunPod billing records per pod per day,
2026-09-22 → 2026-10-05, attributed by pod id:

| | cost | pod-hours |
|---|---|---|
| world-model training and ablations | $217 | 103 |
| dream-PPO sweeps and fleets | $88 | 89 |
| Dyna loop | $53 | 80 |
| E40 control | $1.4 | 1.3 |
| other | $22 | 7 |
| **attributed total** | **$382** | **279 (272 GPU-hours)** |
| unattributed on the account | $214 | |
| network volumes | $18 | |
| before the billing records begin (estimated) | ~$100 | |

**Data:** 10,700 real games in total. 7,490 of them are the loop's policy games, from 205 checkpoints. Plus 2.01M tower
scenario slices; 234 GB in S3.

## 4. How the policy plays (`scripts/moba/behaviour_analysis.py`)

20 holdout games each, as radiant; `results/moba/paper/behaviour.jsonl`, `behaviour_maps.npz`.

| | ppol4 u69 | E38 u0017 | shipped net |
|---|---|---|---|
| won (these 20 games) | 80% | 35% | 100% |
| game length | 5,634 | 5,778 | 2,200 |
| mean distance between our heroes (cells) | 39.7 | 42.5 | 31.7 |
| ticks with ≥ 3 heroes within 8 cells | 9% | 9% | 54% |
| our heroes near an enemy tower as it falls | 0.65 | 0.63 | 2.05 |
| enemy towers destroyed | 9.95 | 9.30 | 6.2 |
| kills / deaths | 9.4 / 30.7 | 6.9 / 29.1 | 35.7 / 30.0 |
| support heals per minute / allies reached per heal | 4.4 / 0.13 | 4.4 / 0.19 | 4.7 / 0.83 |
| enemy deaths with a stun in the previous 20 ticks | 7% | 12% | 26% |
| focus fire (2+ attackers on the same hero, when 2+ attack heroes) | 85% | 72% | 82% |

- **The Dyna loop sharpened the strategy E38 already had; it did not change it.** u0017 and u69 group, push and spread
  the same way. u69 converts more: more kills, more focus fire, more towers.
- **Ability casts that land, per minute,** are at the shipped net's level for every class (support heal 4.4 vs 4.7,
  assassin blink strike 1.6 vs 2.5, carry knockback 2.7 vs 2.3, tank burn 34 vs 42). What differs is the use:
  - the support's heal reaches allies 0.13 times per cast (vs 0.83): it heals itself;
  - stuns rarely set up kills.

  Per-ability rates are in the `cast_*_per_min` columns.

## Artifacts

- **Data and tables:** `results/moba/paper/` (generated by `scripts/moba/paper_data.py`).
- **Figures:** `docs/moba/paper/figures/` (from `figures.ipynb`).
- **Policies:**
  - `s3://puffertank/moba/dream_policies/dyna_live/`;
  - the E40 finals in `results/moba/dream/fz_*_policies/final.pt` (git-ignored).
- **Videos:** `s3://puffertank/moba/media/clips/2026-10-05_ppol4_u49/`, `s3://puffertank/moba/media/debugger_dyna/`.
