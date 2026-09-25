# Replay viewer: real game vs world model, side by side

`tools/moba_viz.py` opens a pygame window with two panels:

- **Left: the real game**, replayed from a recorded game file.
- **Right: the world model.** It starts from the same state and is fed the
  **same recorded actions** every tick. After a short warm-up on real states,
  it runs **open loop**: each tick it steps from its own previous prediction.

If the model got every transition right, the two panels would stay identical.

Everything is precomputed before the window opens, so stepping forwards and
backwards is instant.

## Run it

From the repo root:

```sh
.venv/bin/python tools/moba_viz.py \
    --run runs/token_wm/e4_baseline_v2_1h \
    --dense data/moba/mix1k/dense/games.0000900.mbs \
    --game 0 --start 500 --horizon 300
```

| flag | meaning |
|---|---|
| `--run` | a training run folder containing `model.pt` and `stats.pt` |
| `--dense` | a full-state game file (`.mbs`) |
| `--game` | which game inside that file (0 = first; a file holds 9–13 games) |
| `--start` | the tick to start from |
| `--horizon` | how many ticks the model runs on its own (clamped to the game's end) |
| `--burn` | warm-up ticks on real states before the model runs alone (default 8) |
| `--ghost` | start with the ghost overlay on |
| `--device` | `cpu` (default) |

## Controls

| key | action |
|---|---|
| SPACE | play / pause (at the end, replays from the start) |
| → / ← | one tick forward / back |
| SHIFT + → / ← | 10 ticks forward / back |
| HOME / END | jump to start / end |
| ↑ / ↓ | play faster / slower |
| G | ghost overlay: true positions as hollow shapes on the model panel, with a line from each predicted hero to where it really is |
| Q / ESC | quit |

## What's on screen

- **Walls:** dark blocks.
- **Heroes:** circles in team colour (Radiant cyan, Dire red), with the class
  initial (S support, A assassin, B burst, T tank, C carry), level, and a
  health bar.
- **Creeps:** small squares. **Neutrals:** grey diamonds.
- **Towers:** larger squares with health bars; the two ancients are outlined.
- **Divergence table below the panels,** at the current tick, model vs "copy
  the start state":
  - hero position error (cells);
  - hero health error;
  - creep alive agreement;
  - tower health error.

  Green means the model beats copying, red that it doesn't.

## Getting a checkpoint and game files

**Checkpoints are written on the RunPod pod.** Pull the latest one:

```sh
.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import pod
for f in ('model.pt', 'stats.pt'):
    pod.rsync_from('/workspace/puffertank/runs/token_wm/e4_baseline_v2_1h/' + f, 'runs/token_wm/e4_baseline_v2_1h/')"
```

Numbered checkpoints (`model_000500.pt`, …) are kept every 500 updates. To
view one, copy it over `model.pt` in a local run folder.

**Game files:**

| file | games | use |
|---|---|---|
| `data/moba/mix1k/dense/games.0000900.mbs` | 900–909 | **held out** from every mix1k run (training used games 0–899) |
| `data/moba/mix100/dense/*.mbs` | 0–99 | same games as mix1k 0–99, so **in the training set** of mix1k runs |

More held-out files, from the pod (~150 MB each):

```sh
.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import pod
pod.rsync_from('/workspace/puffertank/data/moba/mix1k/dense/games.0000910.mbs', 'data/moba/mix1k/dense/')"
```

They are also in S3 at `s3://puffertank/moba/mix1k/dense/`.

## Examples

```sh
# the random-init model: identical to "copy the start state", a sanity check
.venv/bin/python tools/moba_viz.py --run runs/token_wm/random_init \
    --dense data/moba/mix1k/dense/games.0000900.mbs --start 500

# a later part of the same game, longer horizon, ghost overlay on
.venv/bin/python tools/moba_viz.py --run runs/token_wm/e4_baseline_v2_1h \
    --dense data/moba/mix1k/dense/games.0000900.mbs --game 0 --start 1500 --horizon 600 --ghost

# another held-out game in the same file
.venv/bin/python tools/moba_viz.py --run runs/token_wm/e4_baseline_v2_1h \
    --dense data/moba/mix1k/dense/games.0000900.mbs --game 3 --start 800

# save one frame as a PNG instead of opening a window
.venv/bin/python tools/moba_viz.py --run runs/token_wm/e4_baseline_v2_1h \
    --dense data/moba/mix1k/dense/games.0000900.mbs --start 500 --screenshot frame.png --frame 64
```

## Setup (once)

```sh
.venv/bin/pip install pygame-ce
```

Use `pygame-ce`: plain pygame's font module fails on this Python 3.14.
