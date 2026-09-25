# Race state contract v1

Pinned in two places that are checked against each other:
`race/race.h` (`RACE_STATE_VERSION`, `RACE_OBS_SIZE`) and `racewm/contract.py`.
`tests/test_contract.py` fails if they drift.

**Schema hash: `74d4ee48e481be34`** — embedded in every dataset shard and every
checkpoint. Loading a checkpoint under a different hash is refused, not warned
about: the observation columns would no longer mean the same thing.

```bash
python3 -m racewm.contract      # print the layout and hash
```

## The observation is the state

Upstream observed 3 floats while the transition depended on heading, position,
the nearest-segment index, and all sector bookkeeping (`docs/AUDIT.md` §A). Race
observes the full state instead. 30 dimensions:

| idx | name | kind | inverse | note |
|-----|------|------|---------|------|
| 0 | `flw_length` | unit | `x` | whisker 0 — points geometric **right**, see below |
| 1 | `frw_length` | unit | `x` | whisker 1 — geometric **left** |
| 2–3 | `sin_ang`, `cos_ang` | signed | `atan2(sin, cos)` | heading; a pair because `ang` wraps at 2π |
| 4 | `px_norm` | unit | `× width` | |
| 5 | `py_norm` | unit | `× height` | |
| 6 | `v_norm` | unit | `× maxv` | constant; there is no throttle |
| 7–8 | `sin_sector`, `cos_sector` | signed | cyclic(S) | `current_sector` |
| 9–10 | `sin_near_point`, `cos_near_point` | signed | cyclic(`total_points`) | `near_point_idx` — load-bearing, see below |
| 11 | `sectors_crossed_norm` | count | `× S` | distinct sectors visited |
| 12 | `tick_norm` | unit | `× max_episode_steps` | the episode timer |
| 13 | `score_norm` | count | `× 100` | upstream's `obs[2]` scale, kept |
| 14–29 | `sector_completed_0..15` | binary | `x` | gates `total_sectors_crossed` only |

Three things worth knowing before modelling this:

**Angles come in `sin`/`cos` pairs.** Regressing them independently is wrong —
renormalize the pair after prediction. `contract.ANGLE_PAIRS` lists them.

**`near_point_idx` is dynamics, not bookkeeping.** `compute_whiskers` raycasts
only segments within ±5 of it, and `update_nearest_point` searches forward
`0..+3` from its own previous value. It carries history and the next whisker
reading depends on it.

**The whisker names are backwards.** `flw_ang = -w_ang`, `frw_ang = +w_ang`,
and the physics integrates in a standard math frame (`py += v·sin(ang)`, +ang
counter-clockwise; the *renderer* flips y, the physics does not). So "front
left" points geometric right. The `LEFT` action does `ang += …`, rotating
toward whisker 1. A near obstacle on `flw` calls for `LEFT`. This cost a
debugging cycle — see the comment in `race/race_bots.h`.

## Track geometry: episode-constant state

The transition function reads the track every step — `compute_whiskers`
intersects the edge polylines, `update_nearest_point` walks the centerline — so
the track is part of the Markov state. It is the part that does not change
within an episode, so it lives beside the shard rather than inside every record.

`race_eval --dump PREFIX` writes `PREFIX.tracks.bin`: per `env_idx`, the
`centerline`, `inner_edge`, and `outer_edge` polylines (64 points each).

For the world model, `shards.Tracks.local_window()` gathers the ±5 window
around `near_point_idx` from both edges — the exact 22 points the raycast can
see — and expresses them in the car's frame:

```
dx = point.x - px;   dy = point.y - py          # translate to the car
ex = dx·cos(-ang) - dy·sin(-ang)                # rotate to the heading
ey = dx·sin(-ang) + dy·cos(-ang)
```

22 points × (ex, ey) = **44 features**, appended to the 30-dim state → 74-dim
world-model input. Ego-framing is what lets a model transfer to a track it has
not seen; absolute pixel coordinates would only memorize these tracks.

This is **not** added to the agent's observation. Two consumers, two inputs:

| consumer | input | why |
|---|---|---|
| world model | 30 state + 44 geometry | must predict `s_{t+1}` exactly, so it needs everything the transition reads |
| RL agent | 30 state | whiskers *are* its sensing of the walls; partial observability is the task |

## Episode boundaries

`c_step` owns them. Upstream reset in the middle of `calc_whisker_lengths`,
inside the frameskip loop (`docs/AUDIT.md` §C).

- crash → `terminated = 1`
- `tick >= max_episode_steps` → `truncated = 1`
- either → `terminals[0] = 1`, `add_log`, `reset_episode`, then observe

PufferLib 4.0 has no truncation channel, so both map onto `terminals`. The
distinction survives in `env->truncated`, in the log, and in every shard
record, which is what matters for value bootstrapping and for world-model
training data.

## Determinism

Each env owns two PCG32 streams seeded from `(seed, env_idx)`:
`rng_track` for geometry (drawn once in `init`) and `rng_episode` for start
poses. They are independent, so changing the policy cannot perturb the track,
and thread count cannot affect results.

```bash
tests/test_determinism.sh
```

asserts: thread count is irrelevant (1/4/16 identical), reruns are
bit-identical, a different seed diverges, and N distinct `env_idx` values give
N distinct trajectories — the direct regression test for upstream's
all-envs-one-track bug.

## Replay

One format, both sides. `race_eval --record` writes it, `race_play` reads it,
neither has its own copy of the logic.

A replay stores only `(config, seed, env_idx, action bytes)` — sufficient
precisely because the env is deterministic. **1 byte per step**; a 1000-step
replay is ~1.1 KB.

**Temporal alignment.** One recorded action drives exactly one `c_step` and
exactly one rendered frame. `c_step` applies frameskip internally, so stepping
the env extra times per action — which the upstream demo did, for smoother
video — would desync client from server by a factor of `frameskip`.
`race_play` does not do that. Frame *t* is step *t*, always.

Every replay carries the FNV-1a trajectory hash from record time and
recomputes it on playback. `race_hash.h` defines the hashed byte sequence once,
so server and client cannot disagree by construction.

```bash
build/race_play replays/r.whisker.001.rr --verify     # headless, checks the hash
build/race_play replays/r.whisker.001.rr --mp4 out.mp4
build/race_play replays/r.whisker.001.rr --rgb-out out/run1
```

## RGB capture

`--rgb-out DIR` writes `frame_%06d.png`, a `states.bin` in the *same* format
`race_eval --dump` produces, and a `capture.json`. Record *t* in `states.bin`
is `frame_%06d.png` at *t* is step *t* in the env — one screen capture feeds
both the PNG sequence and the encoder, so they cannot disagree.

**Capture resolution is pinned to the replay's logical size**, never the
platform framebuffer. macOS HiDPI returns a 2× buffer — a 1080×720 env captures
at 2160×1440 — so capturing raw would make a dataset's resolution depend on
which machine rendered it. `capture.json` records both:

```json
{ "width": 1080, "height": 720,
  "native_width": 2160, "native_height": 1440,
  "hash_match": true }
```

Override with `--capture-size WxH`.

## Deferred

**Headless rendering on a GPU box.** raylib is a desktop GL app and needs a
display; a remote trainer has none. The fix is Xvfb plus Mesa software
rasterization in a container. Deferred out of phase 0 because local capture
already produces correct, resolution-stable output, and because raylib ships no
arm64 Linux binary (`docs/AUDIT.md` E4) — the image needs raylib built from
source, which is phase-2 work when we actually need remote RGB at scale.

**`puffer train race`.** Needs `tools/build.sh module`, which delegates to
PufferLib's `build.sh` and requires OpenMP. Apple clang has none, so this is
Linux/container only.
