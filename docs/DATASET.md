# The Race corpus

Built by `tools/make_dataset.sh`, which writes a `manifest.json` recording
exactly what was generated. Regeneration is deterministic: same seed, same
bytes.

```bash
tools/make_dataset.sh corpus 20     # name, episodes per track
python3 -m racewm.dataset data/corpus
```

## Splits are separate files over disjoint tracks

| split | env_idx | tracks | used for |
|---|---|---|---|
| `train` | 0–95 | 96 | fitting the world model / BC policy |
| `val` | 96–111 | 16 | model selection, early stopping, hyperparameters |
| `holdout` | 112–127 | 16 | **the final number only** — offline rollout *and* online eval |

**Why split by track.** Track identity determines the dynamics: the geometry
the whisker raycast intersects and the poses episodes start from. A model that
has seen a track has seen its physics. Splitting by row would put step *t* in
train and step *t+1* in validation, which is not a generalization measurement
at all — consecutive rows are near-duplicates.

**Why separate files rather than a runtime filter.** A filter is one refactor
from leaking, and a leaked holdout is invisible in every downstream metric —
the numbers just look better. `race_eval --track-start` cuts disjoint ranges of
track space, so the splits cannot overlap by construction, and every shard's
own `env_idx` values prove which split it belongs to.

`racewm.dataset.assert_disjoint()` re-derives the track sets from the shards
and raises on any overlap. `train_wm.py` calls it before training and refuses
to run on a leaky corpus.

## Four slices per split

| slice | policy | epsilon | used by |
|---|---|---|---|
| `expert` | `centerline` | 0.0 | BC + world model |
| `whisker` | `whisker` | 0.0 | BC + world model |
| `noisy` | `centerline` | 0.15 | world model only |
| `random` | uniform | 0.0 | world model only |

**Why mix expert with whisker.** `centerline` is a privileged pure-pursuit
controller (~5.6 laps, 51% crash) — clean trajectories that stay far from
walls. `whisker` only sees the two rangefinders the agent actually has (~2.6
laps, 88% crash). Its failures are the point: a pure-expert corpus contains no
approach-to-crash and no recovery, so a policy cloned from it has never seen
the states it will visit the moment it errs.

**Why noise is quarantined.** The world model must learn transitions for
actions a good policy never takes, so off-distribution actions are needed. But
at frameskip 4 a single random action swings the heading 18°, which is usually
fatal. Measured:

| policy | clean | with epsilon |
|---|---|---|
| `centerline` | 5.6 laps | **1.9 laps** at eps 0.02 |
| `whisker` | 2.6 laps | **0.55 laps** at eps 0.08 |

Folding that into the BC slices would mean cloning a policy far worse than the
one we already have. So noise lives in its own slice, which BC ignores
(`dataset.BC_SLICES`) and the world model uses (`dataset.WM_SLICES`).

## Loading

```python
from racewm import dataset

dataset.assert_disjoint("data/corpus")             # raises on leakage
train = dataset.load_split("data/corpus", "train", slices=dataset.WM_SLICES)
val   = dataset.load_split("data/corpus", "val")
hold  = dataset.load_split("data/corpus", "holdout")

x = dataset.wm_inputs(train)      # 30 state + 3 action + 44 geometry = 77
y = dataset.wm_targets(train)     # 30-dim delta, not absolute next state

xb = dataset.bc_inputs(train.select(dataset.BC_SLICES))              # 30
xp = dataset.bc_inputs(train, privileged=True)                       # 74
```

`bc_inputs(privileged=True)` adds the track geometry the `centerline` bot
actually used. Comparing the two measures how much of the expert's skill is
simply unavailable to a policy restricted to whiskers — an information ceiling,
not a training failure.

## Scale

At 20 episodes per track: **~1.16M transitions, 161 MB**.

| split | transitions |
|---|---|
| train | 817,669 |
| val | 147,656 |
| holdout | 190,528 |

Generation takes under a second — `race_eval` is headless C with no Python in
the loop. Scale with the episodes argument rather than by keeping old corpora
around; regeneration is deterministic and cheaper than storage.

## Known property: action skew

Actions are not balanced — LEFT outnumbers RIGHT roughly 1.8:1. Track control
points are laid out at increasing angle, so loops share a handedness. It is a
property of the environment, not of the corpus construction. BC should account
for it (class weighting or a balanced sampler) rather than treating a
LEFT-biased policy as having learned something.
