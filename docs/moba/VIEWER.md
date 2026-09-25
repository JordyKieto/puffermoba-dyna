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
| `--frame` | the step to open at, render or query (0 = the start state) |

`--run` and `--dense` default to the E4 run and held-out file 900, so a plain
`.venv/bin/python tools/moba_viz.py` works too.

## Controls

| key | action |
|---|---|
| SPACE | play / pause (at the end, replays from the start) |
| → / ← | one tick forward / back |
| SHIFT + → / ← | 10 ticks forward / back |
| R or HOME / END | reset to the first frame (and pause) / jump to the last |
| N / P (or ] / [) | next / previous game in the same file. The model is re-run for that game, which takes a few seconds; the window shows "loading" meanwhile |
| ↑ / ↓ | play faster / slower |
| G | ghost overlay: true positions as hollow shapes on the model panel, with a line from each predicted hero to where it really is |
| Q / ESC | quit |

**Mouse:**
- **Buttons** below the panels: reset, one step back, play/pause, one step forward,
  previous game, next game.
- **Timeline** under the buttons: the whole game, with the model's window in blue
  and the current tick in yellow. Click inside the window to jump there.
- **Hover an entity while paused:** a tooltip shows every state field for that
  entity, real vs model and the difference. Examples: position, health,
  mana, the Q/W/E cooldowns, stun, target. Big differences show in red, small
  ones in orange. The entity is ringed on both panels.

**Global clock**, top right: the game number (and its position in the file),
the current game tick, and the game's final tick.

## What's on screen

- **Walls:** dark blocks.
- **Heroes:** circles in team colour (Radiant cyan, Dire red), with the class
  initial (S support, A assassin, B burst, T tank, C carry), level, and a
  health bar.
- **Ability casts:** a ring flashes around a hero when a Q, W or E actually
  fires: Q yellow, W green, E magenta. It is labelled with the key and fades
  over 3 ticks. A cast is detected when that ability's cooldown resets. The
  tank's Q has no cooldown, so it is detected by its mana cost instead.
  - The left panel shows real casts. The right panel shows casts the model
    predicts, meaning its predicted cooldown reset.
  - "casts so far: truth N model M" counts them up to the current tick.
- **Deaths:** a flashing red/white X labelled DEATH where a hero or tower
  died, for 12 ticks. "deaths so far: truth N model M" counts them.
  - Heroes never have `alive` = 0. The game respawns a hero in the same tick
    it reaches 0 health (`spawn_player` inside the damage code in `moba.h`).
  - So a hero death is detected as a jump of more than 12 cells with health
    reset to full. The assassin's blink also jumps, but keeps its health.
  - A tower death is `alive` going 1 → 0.
- **Key presses:** small coloured Q/W/E letters under a hero show the keys
  it pressed that tick. They are the same on both panels. A press without a
  ring means the ability didn't fire, usually because it was still on
  cooldown.
- **Creeps:** small squares. **Neutrals:** grey diamonds.
- **Towers:** larger squares with health bars; the two ancients are outlined.
- **Divergence table below the panels,** at the current tick, model vs "copy
  the start state":
  - hero position error (cells);
  - hero health error;
  - creep alive agreement;
  - tower health error.

  Green means the model beats copying, red that it doesn't.

## Headless: query a replay without the window

The window only draws. Everything it shows comes from a `Replay` object, which
you can use directly: from the command line, or from Python in a script, a
notebook, or an agent session.

**Steps vs ticks:** step k is frame k of the window. Step 0 is the real state
at `--start`. At step k the model shows its k-th open-loop prediction, next to
the real record `start + k`. `tick` is the game's own clock.

### Command line

With any of these flags the window doesn't open. They can be combined.

| flag | prints / writes |
|---|---|
| `--entity I` | every field of entity `I` (0–9 heroes, 10–109 creeps, 110–181 neutrals, 182–205 towers) at `--frame`: real, model, difference. The keys it pressed that step. Its position error. The same data as the hover tooltip. |
| `--casts` | every Q/W/E cast in the window: step, tick, hero, ability, and whether it happened in the real game, the model, or both |
| `--deaths` | every hero and tower death: step, tick, who, and whether it happened in the real game, the model, or both |
| `--json` | the three above as JSON |
| `--screenshot out.png` | the window at `--frame` as a PNG |
| `--mouse X Y` | with `--screenshot`: the mouse at window pixel (X, Y), so the tooltip shows |

It always prints the divergence table at steps 1, 4, 16, 64 and the horizon.

```sh
# hero 1 at step 41: real vs model, field by field
.venv/bin/python tools/moba_viz.py --frame 41 --entity 1

# which casts did the model reproduce?
.venv/bin/python tools/moba_viz.py --casts
#   122 casts: real 118, model 9, both 5
#     step   41  tick   541  hero 1  E  real only ...

# a picture of the same moment, tooltip on hero 1
.venv/bin/python tools/moba_viz.py --frame 41 --screenshot frame.png --mouse 354 331
```

### Python

```python
import sys; sys.path.insert(0, "tools")
from moba_viz import Replay, Viewer

rp = Replay("runs/token_wm/e4_baseline_v2_1h", "data/moba/mix1k/dense/games.0000900.mbs",
            game=0, start=500, horizon=300)          # loads the model and rolls it out
rp.entity(1, 41)       # {"name", "tick", "fields": {field: (real, model, diff)}, "pos_err", "pressed"}
rp.entity_table(1, 41) # the same as text
rp.cast_events()       # [{"step", "tick", "hero", "ability", "real", "model"}, ...]
rp.death_events()      # [{"step", "tick", "entity", "name", "real", "model"}, ...]
rp.divergence(64)      # {metric: (model, persistence)}, the HUD table
rp.entity_at(0, 67.9, 60.8, 41)   # entity nearest map cell (x, y) at step 41; panel 0 real, 1 model
rp.load(3)             # another game in the same file, same model

# raw arrays for anything else
rp.truth, rp.pred      # (horizon+1, 206, 26) states; fields by name via moba_viz.F["mana"]
rp.acts                # (horizon+1, 10, 6) the action each hero took from step k
rp.cast_truth, rp.cast_pred   # (horizon+1, 10, 3) bool: hero cast Q/W/E at step k
rp.death_truth, rp.death_pred # (horizon+1, 206) bool: entity died at step k (heroes, towers)

Viewer(rp, headless=True).screenshot("frame.png", frame=41, mouse=(354, 331))
```

**How casts are detected:** a cast resets that ability's cooldown. The
detector counts a cooldown rise of more than 5 ticks as a cast. Real resets
are 15 or more; the model's predicted timers wobble by a few ticks. The tank's
Q has no cooldown, so it is detected by its mana cost. A hero that died that
tick is ignored.

### Rewinding and the model's memory

The model is rolled out **once**, forward, when a game loads. Its per-entity
GRU memory is carried from step to step during that pass. Every predicted
state is stored.

Stepping back, rewinding and scrubbing only show stored frames. Nothing is
recomputed, and the GRU memory isn't kept or touched. Step k always shows
the same prediction, however you got there. Switching game re-runs the
rollout.

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
