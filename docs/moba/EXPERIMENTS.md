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

## E1 `smoke2` — in progress

- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/ox4cx47i
- **Command:** `tools/pod.py train smoke2 -- --train-games 0-90 --held-games 91-99 --minutes 10 --batch 64 --log-every 50 --val-every 200 --val-batches 4 --wandb`
- **Setup:** as E0, with the fixes above. Load 69 s: decompression dominates;
  `fit_stats` takes 0.5 s on the GPU (it was ~5 min single-threaded in numpy).
  5.1 it/s at batch 64, about 4,000 frames per second.

| iteration | train loss | **val loss** | val hero | val creep | val neutral | val tower |
|---|---|---|---|---|---|---|
| 0 (persistence) | — | **3.911** | 0.989 | 1.646 | 0.725 | 0.552 |
| 200 | 1.537 | **1.471** | 0.496 | 0.904 | 0.020 | 0.051 |
| 400 | 1.315 | **1.270** | 0.475 | 0.739 | 0.020 | 0.036 |

Qualitative, so far:
- **Validation is below persistence by 62% after 200 iterations,** and tracks
  just under the training loss. No memorization at this point.
- **Neutral and tower losses collapse almost immediately.** They are mostly
  static; what's learned there is the is_hit / alive heads.
- **Hero and creep losses carry the remaining signal.** Hero loss flattens
  near 0.50 early. That is where the action-conditioned movement lives, and
  the end-of-run shuffled-action control will show whether it's being used.

Pending, at the end of the run:
- one-step per-field error vs persistence (train and held-out);
- open-loop rollouts at horizons 1, 4, 16 and 64;
- the shuffled-action control;
- the loss slope over the last quarter.
