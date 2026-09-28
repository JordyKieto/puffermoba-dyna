# E15 — Towers in the dream: measured, diagnosed, and the real attack graph fixes half of it

**The one-line result.** In a 128-tick dream (the PPO episode length), E11 predicts tower
deaths with only **32–35% precision**: two thirds of dream tower falls never happen. The
cause is structural. The real game records who attacks whom every tick (`target_pid`), and
the model throws it away. Feeding it the real attack graph as an input (no new heads) and
fine-tuning for 5 minutes gives **60% precision at 73% recall**, the best of 6 arms.

- **Status:** concluded 2026-09-28 (5-minute arms, single seed, 10 held-out games). The
  combined recipe is running as **E16** (below).
- **Code** (uncommitted, experimental; snapshot in the manifest):
  - `scripts/moba/tower_event_audit.py`;
  - `mobawm/wm/data.py` (`derived_targets`, `attack_inputs`, `ATTACK_MODE`);
  - `mobawm/wm/model.py` (`attack_mode`, zero-initialised `atk_in`);
  - `mobawm/wm/train.py`: `--attack-mode`, `--tower-consist`, `--tower-health-weight`,
    `--tower-event-p`, and `tower_eval` in validation.

## 1. How well does E11 predict tower deaths? (held-out 900–919, 177 real falls)

Held-out games 900-919. Open loop: 128/512-tick windows from real states, real actions; a hit = the same tower falls within +-32 ticks. One step: alarms = runs of consecutive predicted falls.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 | 20 | 177 | 88% | 38% | 51% | 197 | 153 | 302 | 107 | 35% | 70% |
| E11 @1500 (open loop 512) | 20 |  |  |  |  |  | 134 | 368 | 48 | 13% | 36% |

- **One step:** the model sees 88% of real falls coming, but raises about 2 false alarms per real fall.
- **128-tick open loop (the dream PPO trains in):** 35% precision, 70% recall.
- **512 ticks:** 13% precision, 36% recall, and 2.7× too many falls. Longer dream episodes would
  make tower signals worse, not better.

## 2. Why: death is not tied to health, and nobody knows who is hitting whom

- **Death isn't tied to health.** `apply_prediction` produces a tower's death from a separate
  learned yes/no output (`alive_flip`), independent of its predicted health.
- **Nobody knows who is hitting whom.** In the real game, every combat unit records the unit it
  attacked this tick. From 5 held-out games:

| unit type | ticks with `target_pid` set | `is_hit` ticks | hits traceable to an attacker's `target_pid` |
|---|---|---|---|
| heroes | 6,459 (5.1% of alive ticks) | 6,591 | 86% |
| creeps | 32,733 (8.3%) | 29,923 | 95% |
| towers | 1,949 (0.8%) | 1,928 | 93% |
| neutrals | 407 | 919 | 48% (mostly ability/area damage, which sets no target) |

Attacker → target: hero→hero 3,471, hero→creep 1,995, hero→neutral 444, hero→tower 549,
creep→hero 3,618, creep→creep 27,585, creep→tower 1,530, neutral→hero 407, tower→hero 675,
tower→creep 1,274.

- **The model never sees any of this.** `target_pid` is not a model input (the `Featurizer`
  drops it), and rule R9 blanks it in the dream. The only combat input is `is_hit` (one bit, no
  attacker).
- **Where targets come from:**
  - for heroes, the action chooses a target *mode* (3 options), which the model gets as input;
    the game then resolves the actual unit;
  - creeps, towers and neutrals have built-in targeting and no action at all.
- **Deriving targets with the game's own rule** (`moba.h`: nearest enemy in each type's box;
  creeps and neutrals hit at L1 < 2) matches the recorded target for 84% of real attacks:
  neutrals 99.8%, creeps 91.9%, towers 63.9% (towers rescan only every 3 ticks), heroes 49.6%
  (the target mode isn't used).

## 3. Ablation (6 arms, E11 @1500 fine-tuned 5 minutes each, in parallel on 2 H200 pods)

W&B group `E15-tower-dynamics`:

| arm | run |
|---|---|
| control (fine-tune only) | `jox03h6v` |
| derived attack graph (the game's targeting rule on positions; same in training and the dream) | `1xflezs2` |
| **real attack graph** (recorded `target_pid` on real states; derived on model-generated states) | `wt4kffxm` |
| consistency loss (a tower's P(dies) must agree with its own predicted next health) | `zroavbb9` |
| tower health ×10 (and tower deaths ×10) | `0ab2b4gu` |
| tower-event sampling (at each span restart, p = 0.5 jump to ≤64 ticks before a tower loses health or falls) | `eaqg898j` |

Every arm uses the same data: mix1k 0–199 + 240 random-play games, batch 384, lr 1e-3, a fresh
optimizer and a 5-minute budget, reaching about 60–100 updates.

E15 arms: E11 @1500 fine-tuned 5 minutes each (~60-100 updates, batch 384); held-out games 900-909.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 (no fine-tune) | 10 | 87 | 87% | 38% | 53% | 94 | 75 | 165 | 53 | 32% | 71% |
| control: fine-tune only | 10 | 87 | 52% | 43% | 72% | 29 | 75 | 272 | 61 | 22% | 81% |
| real attack graph | 10 | 87 | 24% | 60% | 94% | 2 | 75 | 92 | 55 | 60% | 73% |
| derived attack graph | 10 | 87 | 54% | 51% | 65% | 33 | 75 | 92 | 50 | 54% | 67% |
| consistency loss | 10 | 87 | 48% | 61% | 84% | 11 | 75 | 83 | 45 | 54% | 60% |
| tower-event sampling | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 118 | 53 | 45% | 71% |
| tower health x10 | 10 | 87 | 39% | 42% | 65% | 28 | 75 | 627 | 43 | 7% | 57% |

- **The real attack graph is best:** in the dream it nearly doubles precision (32% → 60%) and
  keeps recall (71% → 73%). Predicted falls drop from 165 to 92.
- **Derived attack graph and the consistency loss** reach 54% precision with some recall loss;
  event sampling reaches 45% / 71%.
- **Tower health ×10 breaks tower prediction** (627 predicted falls, 7% precision).
- **The fine-tune alone makes precision worse** (22%), so the gains belong to the changes.
- **Caveats:**
  - single seed, 5 minutes, 10 games / 75 falls;
  - event sampling predicts no falls one step at a time but still does over 128 ticks (not
    investigated);
  - the "state: failed" in W&B is the end-of-run report crashing after the model was saved (the
    `--store-cpu` report used a GPU generator; fixed);
  - the train logs of consist, tower ×10 and events were lost with pod 2 (never uploaded,
    because 5-minute runs had no evaluation). Their models and W&B metrics survive.
- **Aborted launches** in the same W&B group (1,000-update runs stopped when the requested scale
  was 5 minutes): `j6c6g2ox skk7a1wh grd66rl7 9ttsnv28 3ifyreey ti1m431j r8r9tnzz b3cl5bq3 6krtubiy`.

## 4. E16: the combined recipe, trained on everything (stopped)

- **Recipe:** real attack graph + consistency loss + tower-event sampling.
- **Start:** **E12 @4000** (already trained on expert + random + dyna2).
- **Data:** the full dataset, 2,560 games, 6.77M transitions.
- **Settings:** E5's recipe (batch 768), open-ended.
- **Validation:** now reports tower-fall precision and recall every 500 updates (`tower/*` in W&B:
  64 fixed held-out windows, open loop 128 and one step).
- **Runs:** W&B `vele1qb8`; S3 `moba/runs/e16_towers/`.
- **Stopped at ~150 updates (no checkpoint)** by the author: it trained on a recipe that had
  not been validated, and it competed for the pod with the A/B that was meant to validate it.
- **Smoke first** (`uj692w3j`, 3 minutes): tower open-loop precision 41% → 62% at update 50, and
  one-step false predictions per real fall 10.8 → 2.8.
- **An earlier smoke** (`qsh9x6mg`) ran out of GPU memory: inserting `tower_eval` had moved
  `@torch.no_grad` off `eval_horizons`. Fixed before the launch.

## 5. E17: is the combined recipe worth it, or is the attack graph enough? (update-matched A/B)

- **Setup:** four arms, the E15 setup exactly: E11 @1500, the same data subset, batch 384,
  **exactly 60 updates each** (update-matched, so shared-pod speed doesn't matter). One H200,
  run alone after stopping E16.
- **Test:** the tower audit on held-out games 900–909 (75 falls in 128-tick windows).
- **Runs:** W&B group `E17-attack-ab`:

| arm | run |
|---|---|
| real only | `02klseww` |
| + consistency | `08wrb6xw` |
| + event sampling | `sdmvluu5` |
| combined | `h5jt4gw8` |

E17 update-matched A/B: E11 @1500 fine-tuned exactly 60 updates each (batch 384, same data), held-out games 900-909.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 (no fine-tune) | 10 | 87 | 87% | 38% | 53% | 94 | 75 | 165 | 53 | 32% | 71% |
| real attack graph only | 10 | 87 | 15% | 50% | 81% | 5 | 75 | 85 | 49 | 58% | 65% |
| real + consistency | 10 | 87 | 36% | 63% | 86% | 7 | 75 | 60 | 37 | 62% | 49% |
| real + event sampling | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 109 | 52 | 48% | 69% |
| real + consistency + events (combined) | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 96 | 50 | 52% | 67% |

In-training metric (64 fixed windows, 24 real falls), open-loop precision / recall:

| arm | precision / recall |
|---|---|
| E11 update 0 | 37% / 62% |
| real only | 49% / 71% |
| + consistency | 52% / 46% |
| + events | 36% / 75% |
| combined | 41% / 71% |

**Verdict: the real attack graph alone is the recipe.**
- **It reproduces E15:** 58% / 65% here vs 60% / 73% there, within noise.
- **Consistency** buys a little precision (62%) at a large recall cost (49%).
- **Event sampling** gives back precision (48%).
- **The combined recipe** (52% / 67%) is worse than the attack graph alone on precision.
- **Caveats:** single seed, 60 updates.

## 6. E18: E12 @4000 + real attack graph only, full data (running)

- **Recipe:** the recipe E17 selected, from E12 @4000 (expert + random + dyna2), all 2,560
  training games, E5's settings (batch 768, lr 1e-3), open-ended.
- **Validation:** tower metrics every 500 updates, next to the horizon losses.
- **Runs:** W&B `tz9mc91w`; S3 `moba/runs/e18_attack/`.
- **Launched 2026-09-28 23:16 UTC,** 0.6 updates/s.
- **Update 0 (= E12 @4000):** tower open-loop precision 41%, recall 61%; the number to beat.
