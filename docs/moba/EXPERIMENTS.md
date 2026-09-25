# MOBA token world model: experiment log

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

---

## Proposed next: 10× data, bigger model (not started)

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
