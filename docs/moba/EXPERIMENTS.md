# MOBA token world model: experiment log

## 🎯 TARGET: make every row of this table green

This is the goal of the MOBA world model work. Every experiment below is
judged by whether it moves rows of this scoreboard to green. It is not
judged by aggregate loss, which a model can improve while still ignoring
the abilities.

The question behind each row: **does the model understand what each action
does?** Two measurements per effect, on held-out games, one step ahead:

- **Better than copying?** How much lower the model's error is than just
  copying the last frame. wandb `skill/hero/<row>/better_than_copy_pct`, with
  `<row>` one of walking, attack, q, w, e, mana, health: one curve per row of
  this table, logged at every validation.
- **Depends on the actions?** Give the model the same state twice: once with
  the actions the heroes really took, once with actions from another moment
  of another game. If the prediction is driven by the actions, its error goes
  up a lot with the wrong ones: Q cast → cooldown reset, no Q → no reset. If
  the error stays the same, the model ignores the actions for that effect and
  predicts from the state alone. Reported as **"+X% error with wrong
  actions"**; 0 means the actions are ignored. wandb
  `skill/hero/<row>/action_dependence_pct`.

A row is **green** when it is **≥ 80% better than copying AND has ≥ +50%
error with wrong actions**. Most of these effects are deterministic in the
simulator: casting sets a fixed cooldown, a move applies a fixed step. So
near-exact is achievable, and 80% is a floor, not a stretch goal.

| effect | better than copying? | depends on the actions? (error with wrong actions) | status | best run |
|---|---|---|---|---|
| walking (hero position) | +43% | +110% ✅ | 🟡 driven by the actions, not accurate enough | E2 |
| basic attacks | +79% | +59% ✅ | 🟡 almost green | E2 |
| Q ability | **−38%** ❌ worse than copying | **−9%** ❌ ignores the Q action | 🔴 doesn't understand Q | E2 |
| E ability | +19% | **+3%** ❌ ignores the E action | 🔴 predicts E cooldowns, not from the E action | E2 |
| W ability | +23% | **+39%** | 🔴 weak | E2 |
| damage / health | +13% | **−2%** ❌ wrong actions do as well | 🔴 doesn't link fights to actions | E2 |

- **Update this table after every run** and record which run holds each row's
  best. The numbers come from the run's `final/onestep_heldout` table (field
  model error vs copy error vs wrong-actions error), or from the `skill/`
  curves.
- **Walking** combines hero x and y; **basic attacks** is the hero attack
  timer; **Q/W/E** are the hero Q/W/E cooldowns; **damage/health** is hero
  health.

Then, and only then, the same table for **open-loop rollouts** (64 ticks),
which is what a dream actually needs.

One entry per run, newest last. Every entry has:

- the wandb run;
- the exact command (the pod writes it to `runs/token_wm/<name>/launch.sh`);
- the data split;
- quantitative results against the persistence baseline;
- a qualitative note on what we learned.

Runs that were stopped or invalidated stay in the log with the reason.

- **wandb project:** https://wandb.ai/drift-ai-models/moba-wm
- **Design:** [TOKEN_WM_DESIGN.md](TOKEN_WM_DESIGN.md)
- **Infrastructure:** [RUNPOD.md](RUNPOD.md)
- **Data:** [DATA.md](DATA.md), corpus `mix100`: 100 games, 256k steps,
  `s3://puffertank/moba/mix100/`.
- **Qualitative:** `tools/moba_viz.py` shows ground truth and the model's
  open-loop rollout from the same start state side by side, in pygame.
  - It precomputes the rollout, so you can step forwards and backwards one
    frame at a time.
  - `G` overlays the true positions on the model panel.
  - A live divergence table compares the model with persistence.
  - Setup: `.venv/bin/pip install pygame-ce` (plain pygame's font module
    fails on Python 3.14).
  - Run: `.venv/bin/python tools/moba_viz.py --run runs/token_wm/<name> --dense data/moba/mix100/dense/games.0000091.mbs --game 0 --start 500 --horizon 300`
  - Runs from E2 on save `stats.pt`, so the viewer uses the exact training
    statistics. For E1 it refits them locally and prints a warning.

**What wandb shows** (runs after E2; E2 was back-filled with its final tables):

- **`val/<unit>/<effect>` and `train/<unit>/<effect>`: loss per effect.**
  Units: hero / creep / neutral / tower. Effects: movement, abilities
  (Q/W/E cooldowns + mana), attack, health, status (stun/slow), xp, deaths,
  got_hit, creep lane/team.
- **`skill/<unit>/<effect>/better_than_copy_pct`, every validation:** how
  much better than copying the last frame. 0 = no better; positive is good.
- **`skill/<unit>/<effect>/action_dependence_pct`, every validation:** how
  much the prediction depends on the actions actually taken, as the extra
  error when they are swapped for another moment's actions. 0 = ignored.
- **`skill/deaths/recall_pct`, `skill/deaths/false_pos_per_true`:** deaths
  caught, and false alarms per real death.
- **`final/onestep_heldout` table:** every field, with model error, copy
  error, error with wrong actions, `better_than_copy_pct` and
  `action_dependence_pct`.
- **`final/rollout_heldout` / `final/rollout_train` tables:** open-loop
  error at 1, 4, 16 and 64 ticks, model vs copy.

**How to read the numbers.**

- **The step-0 validation loss is the persistence baseline.** The output
  heads are zero-initialized, so an untrained model predicts "nothing
  changes", which is exactly persistence. Every later validation point is
  directly "how much better than copying the state".
- **Validation always uses the same fixed held-out windows** (games 91–99,
  seed 123, 4 × 64 windows of 16 ticks, 4 burn-in), so the curve compares like
  with like.
- **Losses are per unit type** (hero / creep / neutral / tower). Each is the
  sum of its heads: delta regression (Huber on RMS-scaled deltas), alive-flip
  BCE, is_hit BCE, and creep lane/team CE.

---

## E0 `smoke` — stopped, results invalid

- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/vuzjgdh8
- **Setup:** H200; train on games 0–90 (234k steps, 5.0 GB in VRAM), held out
  91–99; batch 64 × 16-tick windows; d=128, 2 layers, 4 heads, 0.66M params;
  `--minutes 10`.
- **Stopped at about 850 iterations.** Two problems:
  1. **No validation curve.** Held-out evaluation only ran at the end.
  2. **A loss-scaling bug dominated the loss.** Creep `waypoint` only ever
     changes by exactly +1, so the *std* of its non-zero deltas is 0. The
     scale was clamped to 0.001, so each waypoint advance became a target of
     1000. That one field was about 30 of the ~33 total loss.

  The run's "50% loss drop" was mostly that term, so it says nothing about the
  dynamics.
- **Fixes:**
  - scale deltas by the RMS of non-zero deltas;
  - live validation every 200 iterations;
  - per-unit-type losses for train and validation;
  - creep and neutral wave respawns (in-place teleports to spawn, fixed by
    `tick` and `creep_idx`) applied by rule and excluded from regression.

## E1 `smoke2` — complete

- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/ox4cx47i
- **Command:** `tools/pod.py train smoke2 -- --train-games 0-90 --held-games 91-99 --minutes 10 --batch 64 --log-every 50 --val-every 200 --val-batches 4 --wandb`
- **Setup:** as E0, with the fixes above.
  - Load 69 s: decompression dominates; `fit_stats` takes 0.5 s on the GPU
    (it was ~5 min single-threaded in numpy).
  - 2,357 iterations in 10 min at ~4.6 it/s with batch 64, about 3,500
    frames per second. 0 NaNs.

**Validation loss** (fixed held-out windows; step 0 = persistence):

| iteration | train loss | **val loss** | vs persistence | val hero | val creep | val neutral | val tower |
|---|---|---|---|---|---|---|---|
| 0 | — | **3.911** | — | 0.989 | 1.646 | 0.725 | 0.552 |
| 200 | 1.537 | **1.471** | −62% | 0.496 | 0.904 | 0.020 | 0.051 |
| 400 | 1.315 | **1.270** | −68% | 0.475 | 0.739 | 0.020 | 0.036 |
| 1000 | 1.082 | **1.067** | −73% | 0.405 | 0.618 | 0.020 | 0.024 |
| 1600 | 0.997 | **0.991** | −75% | 0.391 | 0.564 | 0.017 | 0.018 |
| 2357 (end) | 0.951 | **0.927** | −76% | 0.373 | 0.524 | 0.015 | 0.015 |

Log-loss slope over the last quarter: **−0.074 per 1k iterations**, still
falling. This model is under-trained, not converged.

**One-step, held-out** (MAE in game units; model / persistence. "Changed"
counts only entries whose true value changed):

| field | all entries | changed entries |
|---|---|---|
| hero x | **0.34** / 0.52 | **0.47** / 1.01 |
| hero y | **0.30** / 0.52 | **0.39** / 0.89 |
| creep x | **0.30** / 0.55 | **0.38** / 0.74 |
| hero health | 6.02 / **5.51** | **18.5** / 18.9 |
| hero mana | **1.89** / 2.18 | **3.9** / 4.9 |
| hero xp | 1.47 / **0.72** | **72.5** / 80.7 |
| creep health | 3.14 / **2.59** | **18.9** / 32.4 |
| tower health | 0.64 / **0.46** | **34.5** / 53.7 |
| attack timers (hero, creep, tower) | **0.20 / 0.27 / 0.03** vs 0.57 / 0.80 / 0.10 | |
| alive flips | recall **14%** (73 / 529), 83 false positives | |

Train vs held-out one-step: hero x 0.37 vs 0.34, creep x 0.31 vs 0.30. **No
generalization gap.**

**Shuffled-action control (held-out).** Hero x MAE rises from **0.34 → 0.70**
(y 0.30 → 0.67) with another window's actions. That is worse than
persistence (0.52): the model relies on the actions for hero movement, as it
should.

**Open-loop rollouts** (8 burn-in ticks, then predictions fed back; 16
windows; model / persistence; lower is better):

| horizon | hero pos L1 | hero hp | creep pos L1 | creep alive acc | tower hp |
|---|---|---|---|---|---|
| held-out 1 | **0.65** / 0.97 | **6.2** / 6.2 | **0.53** / 1.08 | 1.00 / 1.00 | 0.24 / **0.11** |
| held-out 4 | **2.33** / 3.43 | **16.3** / 18.3 | **1.11** / 3.41 | 0.99 / 0.99 | 0.90 / **0.40** |
| held-out 16 | **9.05** / 12.94 | 65.8 / **60.3** | **3.35** / 11.35 | 0.93 / 0.93 | 5.3 / **4.3** |
| held-out 64 | **31.6** / 34.0 | 177.6 / **133.3** | **16.3** / 37.9 | **0.81** / 0.78 | 30.3 / **15.3** |
| train 64 | 33.7 / **33.2** | 163.5 / **117.1** | **17.6** / 30.8 | **0.78** / 0.71 | 29.3 / **13.9** |

**What we learned:**
- **Movement is learned and action-driven.** Hero and creep positions beat
  persistence by 35–45% one-step. The shuffled-action control confirms heroes
  follow their actions. Creep positions stay far better than persistence even
  at 64 ticks open loop (16 vs 38 cells).
- **Health, XP and tower health: better on changes, worse overall.** On
  entries that changed, the model beats persistence. Over all entries it is
  worse, because it predicts small changes where nothing happens. Open loop
  these false changes accumulate: at 64 ticks, hero and tower health drift to
  ~1.3–2× the persistence error. This is the copy-by-default problem the prior
  study documented. The candidate fix is an explicit "does this field change"
  gate on the sparse fields.
- **Deaths are mostly missed:** alive-flip recall is 14%.
- **Long-horizon hero position** ends roughly at persistence (31.6 vs 34.0)
  by 64 ticks.
- **No overfitting:** held-out tracks train throughout.
- **Still improving:** the loss slope is negative at the end, so more data and
  more steps are the obvious next lever. That is E2.

## E2 `e2_mix1k` — complete: 10× data, same model

- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/asdg1vl1
- **Command:** `tools/pod.py train e2_mix1k -- --data data/moba/mix1k/dense --train-games 0-899 --held-games 900-999 --minutes 30 --batch 64 --log-every 100 --val-every 500 --val-batches 4 --wandb`
- **Data:** `mix1k`, 1,000 games collected on the pod (same controller mix),
  `s3://puffertank/moba/mix1k/` (30 MB logs + 15.3 GB full state, verified).
  - Train: games 0–899, 2.40M steps, **51.5 GB resident on the H200**.
  - Held out: games 900–999, 268k steps.
- **Setup:** E1's model and hyperparameters, unchanged. Only the data
  (×10) and the time budget (30 min) differ.
  - Load 156 s: parallel decompression into shared memory; `fit_stats` 1 s.
  - 8,823 iterations at ~4.9 it/s. 0 NaNs.
- **Comparing with E1:** the held-out games differ (E1 used 91–99, E2
  900–999), so compare each run *against its own persistence baseline*.
  E1's baseline was 3.911, E2's is 3.889.

**Validation loss:**

| iteration | **val loss** | vs persistence | hero | creep | neutral | tower |
|---|---|---|---|---|---|---|
| 0 | **3.889** | — | 0.975 | 1.640 | 0.724 | 0.550 |
| 1000 | **1.023** | −74% | 0.392 | 0.589 | 0.022 | 0.021 |
| 2500 | **0.879** | −77% | 0.358 | 0.493 | 0.014 | 0.013 |
| 4000 | **0.793** | −80% | 0.334 | 0.436 | 0.012 | 0.010 |
| 7000 | **0.716** | −82% | 0.304 | 0.395 | 0.010 | 0.007 |
| 8823 (end) | **0.690** | **−82%** | 0.290 | 0.384 | 0.010 | 0.006 |

E1 ended at −76%. The log-loss slope over the last quarter is −0.019 per 1k
iterations (E1: −0.074). Still improving, but flattening for this model size.

**One-step, held-out** (model / persistence; E1 in brackets):

| field | all entries | changed entries |
|---|---|---|
| hero x | **0.32** / 0.56 (0.34 / 0.52) | **0.46** / 1.02 |
| hero health | **5.63** / 6.44 — *now beats persistence* (E1 6.02 / 5.51) | **16.2** / 20.5 |
| creep x | **0.26** / 0.55 (0.30 / 0.55) | **0.33** / 0.75 |
| creep health | **2.28** / 2.51 — *now beats persistence* (E1 3.14 / 2.59) | **16.5** / 34.6 |
| tower health | 0.51 / **0.36** (0.64 / 0.46) | **18.8** / 46.3 |
| hero xp | 1.76 / **0.75** | **85.0** / 95.7 |
| alive flips | recall **43%** (262 / 615), 125 false positives (E1: 14%) | |

**Shuffled-action control:** hero x MAE 0.32 → 0.67 with wrong actions, worse
than persistence. Still action-driven.

**Open-loop rollouts, held-out** (model / persistence):

| horizon | hero pos L1 | hero hp | creep pos L1 | creep alive acc | tower hp |
|---|---|---|---|---|---|
| 1 | **1.76** / 2.33 | **8.0** / 9.5 | **0.55** / 1.11 | 1.00 / 1.00 | **0.31** / 0.46 |
| 4 | **2.83** / 4.41 | **15.1** / 20.2 | **1.09** / 3.32 | **1.00** / 0.99 | 0.96 / **0.52** |
| 16 | **6.25** / 11.19 | **36.4** / 44.3 | **2.73** / 10.92 | **0.99** / 0.97 | 4.89 / **2.55** |
| 64 | **19.6** / 28.6 | 136.2 / **97.5** | **13.3** / 38.2 | **0.90** / 0.75 | 28.2 / **19.9** |

**What we learned** (10× data, same model):
- **Everything that was working improved.** Validation −82% vs persistence
  (E1 −76%).
- **Hero and creep health now beat persistence over all entries,** not only
  on changes. The false-change problem shrank with more data.
- **Deaths are learned far better:** alive-flip recall rose from 14% to 43%.
- **Long-horizon rollouts improved most.** At 64 ticks, hero position error
  is 19.6 vs 28.6 for persistence; in E1 it was roughly equal. Creep alive
  agreement is 0.90 vs 0.75.
- **Still open:**
  - Tower health and XP remain worse than persistence over all entries.
  - Open-loop hero health drifts past persistence by 64 ticks.

  These are rare, sparse changes, so the candidate fix is still a change
  gate on those fields.
- **The model is now capacity-bound more than data-bound:** the loss slope
  flattened to −0.019 per 1k iterations. That motivates E3 (a bigger model),
  plus the batch/bf16 sweep for throughput.

### Is action conditioning solved? Not yet: movement yes, abilities and combat no

The per-field shuffled-action control is what decides it. If the model used
an action head, feeding it another window's actions would make the field it
drives worse.

Held-out, one step, E2 (model / persistence / model with shuffled actions):

| field | model | persistence | shuffled | reading |
|---|---|---|---|---|
| hero x / y | **0.32 / 0.31** | 0.56 / 0.54 | 0.67 / 0.66 | conditioned on the move heads |
| hero attack timer | **0.13** | 0.63 | 0.21 | partly conditioned (target mode) |
| hero Q cooldown | 0.216 | **0.156** | 0.196 | Q ignored: worse than persistence, no better than shuffled |
| hero E cooldown | 0.472 | 0.580 | 0.485 | E ignored |
| hero W cooldown | 0.545 | 0.704 | 0.758 | weakly conditioned |
| hero health | 5.63 | 6.44 | **5.54** | combat not conditioned on actions |

Many of these effects are deterministic in the simulator:
- casting sets a fixed cooldown, and otherwise it counts down by 1;
- movement is the action vector applied to the position.

A solved model would be near-exact on them. The model has learned where
heroes go, not what their abilities and attacks do. A rollout that looks
right in the viewer (heroes walking the right way) can still be wrong
exactly where the fights happen.

**Criteria for "action conditioning solved":**
1. **Deterministic action effects are near-exact:** cooldowns and mana
   correct on >99% of casts; hero position error well under 0.1 cell.
2. **Every action head passes the shuffled-action control,** not only
   movement.
3. **Counterfactual accuracy.** Restore one state in the simulator
   (`moba_restore`), apply two different action sets, and check the model
   predicts both outcomes. It is the direct test of action conditioning,
   and the exact simulator makes it free to generate.

## E3b `e3b_gate_events` — complete (15 min): framings 1 + 3, big batch

- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/65j98z3b
- **Command:** `tools/pod.py train e3b_gate_events -- --data data/moba/mix1k/dense --train-games 0-899 --held-games 900-999 --minutes 15 --batch 768 --lr 1e-3 --bf16 --gate --effect-weights abilities=3,health=3,deaths=3,attack=2 --event-frac 0.5 --log-every 20 --val-every 100 --val-batches 4 --wandb`
- **What it tests:** the two framings carried over from the prior study
  ([PRIOR_WORK.md](PRIOR_WORK.md)):
  - **Framing 1, semantic loss × event-balanced sampling.** Loss weights per
    effect (abilities ×3, health ×3, deaths ×3, attack ×2). Half of every
    batch comes from windows containing a hero cast, hero/tower damage, or a
    death.
  - **Framing 3, copy by default.** Per delta field, a "did this field change"
    gate, BCE on every supervised entry. The value is predicted and trained
    only where the field changed. Gates start at each field's observed change
    rate, so the untrained model is exactly persistence.
- **Scale:** batch 768, bf16, lr 1e-3. 923 iterations in 15 min at ~1 it/s,
  about 709k training windows (E2: ~565k). GPU 92% utilized, 115 GB.

**Scoreboard** (hero; better than copying / error with wrong actions):

| effect | E2 | **E3b** |
|---|---|---|
| walking | +43% / +110% | +26% / +80% |
| basic attacks | +79% / +59% | +67% / +10% |
| Q | −38% / −9% | **+14%** / −24% |
| W | +23% / +39% | +23% / +10% |
| E | +19% / +3% | +14% / **+16%** |
| mana | +11% / +29% | +16% / +7% |
| health | +13% / −2% | −20% / +7% |
| deaths caught (recall) | 43% | 19% (139 false positives) |

Open loop, held-out, 64 ticks (model / copy): hero position 24.4 / 28.6,
creep position 11.3 / 38.2 (E2 13.3), creep alive 0.86 / 0.75.

**Outcome: no row turned green.**
- Q stopped being worse than copying (−38% → +14%).
- E depends on the actions more (+3% → +16%).
- Health depends on the actions a little (−2% → +7%).
- Most other rows are weaker than E2.

**Limitations. This is not a clean test of the framings:**
1. **Undertrained.** 923 updates vs E2's 8,823. The log-loss slope over the
   last quarter was **−0.55 per 1k iterations** (E2: −0.019): still falling
   steeply when it stopped.
2. **Confounded.** Batch size (64 → 768) and learning rate (3e-4 → 1e-3)
   changed together with the framings, with no same-batch baseline.
3. **Every update was clipped.** Gradient norm 9–25 against `clip 1.0`. The
   effect weights inflate the loss, so the clip discarded most of each
   gradient's magnitude. zelda-RL's handoff names this exact pattern as the
   untested suspect for models that start at persistence and then degrade.
4. **Ability metrics were logged as one "abilities" number during training**
   (Q/W/E/mana combined). The per-ability split existed only in the final
   table. Fixed for the next run: per-ability curves every validation.
5. **Loss values aren't comparable across recipes.** The gated,
   effect-weighted loss has a different scale; compare scoreboard metrics
   only.

## Ablations: 5-minute A/B tests

**Setup, identical for every arm:**
- `mobawm/wm/ablate.py`: arms train side by side in one process on
  identical batches with the same init seed.
- Data: 300 training games (0–299), scored on 30 held-out games (900–929).
- Batch 256, bf16, lr 6e-4, **162 updates per arm**.
- Numbers are hero **better than copying / error with wrong actions**, in %.
- Early-training trends only; arms within a table are directly comparable.
- wandb groups `ablate1`–`ablate5`.

| test | arm | walking | attack | Q | W | E | mana | health |
|---|---|---|---|---|---|---|---|---|
| framings | base (E2 recipe) | **+26 / +67** | +20 / +6 | −27 / +4 | +11 / +17 | +3 / +12 | +9 / +8 | −19 / 0 |
| | gate (framing 3) | +16 / +40 | +22 / +18 | 0 / 0 | **+29 / +32** | **+28 / +17** | +19 / +9 | −23 / +8 |
| | weights + event batches (framing 1) | +25 / +63 | +18 / +2 | −28 / +4 | +14 / +18 | +10 / +16 | +11 / +8 | −17 / 0 |
| | both | +6 / +23 | +21 / +24 | +1 / 0 | +32 / +29 | +29 / +18 | +22 / +11 | −22 / +15 |
| gate only rare fields | gate, rare fields | −4 / +8 | +22 / +19 | 0 / 0 | +27 / +22 | +27 / +12 | +17 / +8 | −22 / +7 |
| same, clip 10 | base | +16 / +42 | +13 / +10 | −27 / +5 | +9 / +16 | +1 / +12 | +9 / +8 | −20 / 0 |
| | gate, rare fields | −8 / +6 | +28 / +13 | 0 / 0 | +31 / +29 | +28 / +15 | +20 / +11 | −27 / +7 |
| action tokens | action tokens, attend to all | −3 / +2 | +26 / +1 | 0 / 0 | +31 / +3 | +30 / 0 | +22 / +1 | −29 / 0 |
| | action tokens, own hero only + own-action bias | +5 / +4 | +29 / +1 | 0 / 0 | +35 / +12 | +30 / +2 | +18 / +2 | −33 / 0 |

**Findings:**
- **The change gate (framing 3) is what moves the ability rows.** W, E and
  mana go from near 0 to +19 to +29% better than copying, and Q stops being
  worse than copying. The cost is walking (+26 → +16).
- **Effect weights + event-oversampled batches (framing 1) do ~nothing alone**
  at this scale, and combined with the gate they hurt walking further. This
  does *not* reproduce the prior study's interaction result.
- **Gating only the rare fields broke walking** (−4 / −8), and loosening the
  clip didn't fix it. So clipping isn't the cause. Suspected instead: the
  gate's yes/no losses dominate the loss and starve walking. Untested.
- **Action tokens failed; dropped, code not kept in the branch.** 40 tokens
  (movement + Q/W/E per hero) carrying unified + slot + ability ID + ability
  numbers + a cast-rule context. The cast rule matched real casts at 77–91%
  recall and 4–7% false alarms.
  - Open to all tokens, the model stopped using actions: error with wrong
    actions ~0 on every row, walking −3%.
  - Restricted to the own hero, with a learned own-action bias starting at
    +3: a little better (walking +5 / +4, W +35 / +12). Still far below adding
    the action to the hero token (walking +16 / **+40**).
  - Why: the action signal has to travel through attention instead of
    arriving directly in the hero's token, and 162 updates isn't enough to
    learn that.
  - The diff is kept outside git: `runs/experiments/action_tokens.diff`.
- **Decision at that point:** the change gate on all fields, actions added to
  the hero token. Superseded below.

### Round 2: geometry, navgrid, and fixing walking under the gate

Same setup; wandb groups `ablate6`–`ablate10`.

| test | arm | walking | attack | Q | W | E | mana | health |
|---|---|---|---|---|---|---|---|---|
| reference | gate, all fields | +16 / +40 | +22 / +17 | 0 / 0 | +29 / +28 | +28 / +14 | +19 / +8 | −23 / +8 |
| geometry | + walking-distance code, 9×9 occupancy patch, 64 region tokens | +14 / +40 | +24 / +14 | 0 / 0 | +30 / +29 | +29 / +13 | +19 / +8 | −24 / +6 |
| navgrid | + 32×32 cell tokens, predicting next-tick contents; anchor ±5 | +14 / +35 | +29 / +12 | 0 / 0 | +33 / +26 | +29 / +12 | +21 / +8 | −31 / +7 |
| | anchor ±2 | +13 / +33 | +28 / +13 | +1 / +1 | +32 / +21 | +28 / +10 | +18 / +5 | −25 / +6 |
| loss balance | rare-fields gate, yes/no loss ×0.2 | +6 / +24 | +26 / +16 | 0 / 0 | +28 / +33 | +26 / +12 | +19 / +11 | −19 / **+20** |
| | **+ value loss scaled by change rate = BASELINE** | +15 / **+43** | +21 / +17 | 0 / 0 | **+31** / +26 | +28 / **+19** | **+21 / +13** | −24 / +9 |

**Geometry bundle: neutral.** Every row is within a couple of points, at 13%
more time per update.
- Inputs: walking distance to 42 landmarks, occupancy patch, region tokens.
- Not kept; the diff is in `runs/experiments/geometry_bundle.diff`.

**Navgrid: neutral on the scoreboard, 2× slower per update.**
- What it is: the map as 32×32 cell tokens, each holding the unit types
  present plus its wall fraction, beside the unit tokens, with units↔cells
  cross-attention; each cell predicts its next-tick contents.
- Its own prediction barely learned: 0% of changed cells right with the ±5
  "same as now" anchor, 5% with ±2.
- Not kept; the diff is in `runs/experiments/navgrid_and_baseline.diff`.
- Untested: whether a grid built from the baseline's predicted positions
  already beats 5%.

**Walking under the gate: it was loss balance, in two parts.**
1. The gate's yes/no losses crowd out movement. Down-weighting them ×0.2
   took walking from −4 to +6.
2. A gated field's value loss, averaged only over ticks where it changed, is
   much larger than the same field without a gate, averaged over all ticks
   (mostly zeros). Averaging it over all ticks, i.e. scaling it by the field's
   change rate, took walking to +15, on par with gating everything, while
   keeping the best ability numbers yet.

**What was given up:** health's dependence on the actions. The change-only
value loss had given health +20% error with wrong actions, the best of any
arm; the scaled version gives +9%.

Walking is still below the ungated model (+26 / +67).

## ✅ Current baseline (adopted 2026-09-25)

Change gate on every delta field **except movement** (x, y, waypoint). Gate
yes/no loss ×0.2. Gated fields' value loss averaged over all supervised
entries (scaled by change rate). Actions added to the hero token.

Where it stands, after 162 updates (5-minute test; hero better than copying /
error with wrong actions, %):

| effect | **baseline** | no gate | gate on all fields |
|---|---|---|---|
| walking | +15 / +43 | **+26 / +67** | +16 / +40 |
| attack | **+21 / +17** | +20 / +6 | +22 / +17 |
| Q | 0 / 0 | −27 / +4 | 0 / 0 |
| W | **+31 / +26** | +11 / +17 | +29 / +28 |
| E | **+28 / +19** | +3 / +12 | +28 / +14 |
| mana | **+21 / +13** | +9 / +8 | +19 / +8 |
| health | −24 / +9 | −19 / 0 | −23 / +8 |

- **Full training:** `python -m mobawm.wm.train --baseline ...`
  (sets `--gate --gate-fields <all but movement> --gate-loss-weight 0.2
  --gate-value-norm all`)
- **5-minute tests:** `python -m mobawm.wm.ablate --arms baseline,<new arm> ...`

**Open problems:**
- Q is stuck at 0 in every arm.
- Health is worse than copying.
- Walking is +15 vs +26 for the ungated model.
- Deaths are 0% at 162 updates.

**Candidate next steps:**
- a "can this press fire" input (the validated cast rule), for Q;
- keeping health's change-only value loss, for health.

---

## Proposed next: bigger model (E3) and a throughput sweep (not started)

*E2 (10× data) is done, above. The capacity numbers below still apply.*

**Capacity check, H200 143 GB.**
- Full-state data costs 21.4 KB per step (206 × 26 fp32). `mix100` is 5.0 GB;
  10× (~1,000 games, ~2.5M steps) would be **~55 GB**.
- The E1 model (0.66M params, batch 64) uses ~9 GB beyond its data.
- At 10× data, **~80 GB remains** for model and activations. That is enough
  for d = 384–512, 4–6 layers and batch 64–128 (~20–40M params).
- Storing non-position fields as fp16/int16 halves the data. That puts the
  ceiling at ~40–50× before the corpus needs sharding.

**Data.**
- Collect `mix1k` (1,000 games, same controller mix) on the pod: 160 CPUs,
  one game per process at ~1,700 steps/s, about a minute plus replay.
- Push the logs to `s3://puffertank/moba/mix1k/logs`.
- Hold out a fixed game range, as in E1.

**Before scaling:**
1. **Loading.** Decompression is ~69 s for 5 GB now, ~11 min at 10×.
   Decompress once into raw arrays on `/workspace` and memory-map them.
2. **bf16 autocast.** E1 runs the GPU at ~60%, bound by the per-tick Python
   loop. A bigger model plus bf16 is where the headroom is.

**Run plan.**
- E2: E1's architecture on 10× data. Isolates the data effect.
- E3: a scaled model on the same data.
- Both are compared on validation loss vs persistence, per unit type, and
  on open-loop rollout error at horizons 1, 4, 16 and 64.
