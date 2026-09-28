# MOBA: game state, ego observations, and the game corpus

Data for training a **transition model** on MOBA (PufferLib Ocean `moba`).
Given the full game state and the actions of **all 10 heroes**, the model
predicts the next state. It is then meant to run as a dream: every hero acts
inside it and sees exactly the observation it would get in the real game.

What this provides, each part tested:

1. **A complete game state (`MobaState`).** Restored into a freshly built
   env, it reproduces the original game byte-for-byte (the Markov test
   below). The randomness is part of it, so the next state is an exact
   function of (state, 10 actions).
2. **10 recorded actions per tick.** Every hero on both teams acts through
   the same action interface.
3. **The ego observation as a pure function of the state,** byte-exact with
   the game, in C and in numpy.
4. **A corpus stored as seed + actions.** The full per-tick state sequence
   is regenerated bit-exactly, in order, including each game's terminal state.

Upstream is used unmodified (`third_party/pufferlib/ocean/moba`). Everything
here lives in `moba/`, `mobawm/` and `tools/`. What was carried over from the
earlier `pufferlib-ocean-wm` study, and what was not, is in
[PRIOR_WORK.md](PRIOR_WORK.md).

## Quick reference

```bash
tools/build.sh moba-demo                                   # upstream visual demo
(cd third_party/pufferlib && ../../build/moba)

DENSE=1 tools/moba_collect.sh 100 13 8 data/moba/mix100    # 100 games, 8 procs, resumable

build/moba_collect replay data/moba/mix100/logs/games.0000000.mbl   # verify a log
build/moba_collect markov data/moba/mix100/logs/games.0000091.mbl   # Markov test (~1 min / 9 games)
python3 -m mobawm.obs data/moba/mix100/dense/games.0000000.mbs      # verify numpy obs
```

```python
from mobawm import shards, obs

d = shards.load_dense("data/moba/mix100/dense/games.0000000.mbs")
recs = d.load_game(0)                        # one game: n_ticks + 1 records, in order
s_t, a_t, s_next = shards.transitions(recs)  # every step, terminal step included
o_t = obs.ego_obs_from_states(recs["state"], d.statics)   # (T, 10, 510) == recs["obs"]
```

## The corpus: `data/moba/mix100`

100 games, **256,049 steps** (+100 terminal states). Game *i* is config *i* mod 3:

| radiant | dire | games | steps | mean length | radiant wins | dire wins |
|---|---|---|---|---|---|---|
| net | net | 34 | 76,343 | 2,245 | 34 | 0 |
| net | script | 33 | 54,727 | 1,658 | 31 | 2 |
| script | net | 33 | 124,979 | 3,787 | 33 | 0 |

- **Size.** Logs 2.9 MB (~11 bytes/step). Full state 1.4 GB (71 KB/record
  raw, zlib per game, 12.7×).
- **Time.** 63 s wall on 8 processes, collection and full replay included.
- **Truncation.** No game hit the 20,000-tick cap.
- **Checks.** Every game is replay-verified: state hash, and the C obs
  function against the game's observation on every record.

**Outcome skew.** Radiant wins 98 of 100. Dire wins only when it is scripted
and Radiant is the net. The data covers Dire pushing and taking towers, but
rarely Dire *winning*.

## The game

A 128×128 grid. **206 entities**, always at the same indices:

| indices | what |
|---|---|
| 0–4 | Radiant heroes: support, assassin, burst, tank, carry |
| 5–9 | Dire heroes, same classes |
| 10–109 | creeps; 6 lanes × 5 spawn every 150 ticks, slots reused round-robin **across lanes** |
| 110–181 | neutrals; 18 camps × 4, respawn every 600 ticks |
| 182–205 | towers; 204 is the Dire ancient, 205 the Radiant ancient |

A game ends when an ancient dies. Upstream has no time limit; the collector
truncates at 20,000 ticks.

**All 10 heroes are agents** (`script_opponents = 0`). By default upstream
drives Dire with `creep_ai` directly, which records no Dire actions and turns
Dire's behaviour into physics for a transition model. Here every hero picks
the same 6 action heads every tick: move y (7), move x (7), target mode (3),
use Q, W, E (2 each). That's 1,176 combinations, stored as one `uint16` code
per hero.

## Controllers

- **`random`** (controller id 3, added 2026-09-27): every tick each hero samples every
  action head uniformly (`splitmix64(prng) % MOBA_ACTION_SIZES[k]`) from the game's
  seeded PRNG stream. It is replayable like the others.
- **`dream_ppo`** (id 2): a dream-trained policy, recorded by
  `scripts/moba/collect_policy_games.py` (the Dyna corpus `dyna1`).

**Corpora and their controller mix** (from the log headers;
`results/moba/tables/data_controller_census.md`):

| corpus | games (indices) | hero-ticks | notes |
|---|---|---|---|
| `mix1k` | 1,000 (0–999) | net 64.2%, script 35.8% | ε = 0: **no random actions**. The data behind E1–E10 |
| `mixrand1k` | 1,000 (1000–1999) | random 69.0%, net 12.7%, script 18.3% | blocks: random/net 1000–1159, net/random 1160–1329, random/random 1330–1659, random/script 1660–1829, script/random 1830–1999; 3,000-tick cap |
| `dyna1` | 100 (5000–5099) | dream_ppo 50%, net 50% | the two P3 policies vs the net, 0 replay mismatches |
| `dyna2` | 810 (6000–6809) | dream_ppo 66.7%, random 16.7%, net 16.7% | the two P3 policies: vs random 6000–6269, vs net 6270–6539, vs each other 6540–6809 (sides and policies balanced); 3,000-tick cap; 2,154,814 ticks; 0 replay mismatches; game list `results/moba/dyna/dyna2_games.json` |

All three are in `s3://puffertank/moba/<corpus>/{logs,dense}/`.

- **`net`** is the upstream pretrained policy (`resources/moba/moba_weights.bin`,
  encoder → 5×MinGRU → decoder). It runs on each hero's own observation and
  samples from its own RNG. It was trained only as Radiant against scripted
  Dire. As Dire it fights, farms and levels, but never pushes towers.
- **`script`** is upstream `creep_ai`'s hero behaviour, re-expressed as
  actions:
  - it chases the nearest non-neutral enemy within 7 cells, otherwise walks
    its lane's waypoints;
  - it steps along the BFS path `creep_ai` would take;
  - target mode 1, no skills (`creep_ai` never casts).

Two details of `script` were needed to stop games stalling:

1. **It picks the move the env will actually execute.** Heroes move by a
   continuous velocity from a fractional position, so a naive diagonal can
   land in a blocked side cell and repeat forever. It evaluates all 49
   move-head pairs with `step_players`' own math.
2. **It remembers lane progress.** Like `creep_ai`, it keeps a forward-only
   waypoint index, reset on respawn; some lanes double back. This is
   controller memory, like the net's recurrent state, and not game state.

## The full game state (`MobaState`, `moba/moba_state.h`)

| part | size | content |
|---|---|---|
| `entity[206][26]` f32 | 21.4 KB | alive, x, y, health/max, mana/max, damage, level, xp, move modifier, stun/move/Q/W/E/attack timers, is_hit, waypoint, target, **team, tile id, lane, entity type, attack cooldown, xp on kill** |
| `grid[128*128]` u8 | 16 KB | tile id per cell. **Not derivable from entity positions**: the env leaves stale tiles behind. |
| `pids[128*128]` u8 | 16 KB | entity index on each cell (255 = empty) |
| `reward[10][4]` f32 | 160 B | last step's death/xp/distance/tower components |
| `scan[206][64]` u8 | 13 KB | each entity's target list, 255-terminated. Creeps and neutrals rescan every 5 ticks and towers every 3, and act on the stale list in between. |
| `tick`, `creep_idx`, `rng_idx` | 12 B | spawn timing and scan phase; the next creep slot; position in the jitter table |
| `rng_state` u64 | 8 B | **the env's random stream** |

Per-hero constants (hero class, move speed) are stored once per dense file.
Everything else in upstream's `Entity` is either constant for the game
(spawn point, base stats, per-level gains, tier) or rewritten at the start of
every step before it is read (`last_x`/`last_y`, `attack_aoe`). The Markov
test checks both claims.

**Why team, tile, lane, type, cooldown and XP-on-kill are per tick.**
`spawn_creep` reuses creep slots round-robin across all 6 lanes and rewrites
these fields on every wave. An earlier version stored them once at game
start, before any creep existed. Every creep's team and lane were then wrong,
and a model could not tell which way a creep would walk. The observation
checks passed anyway, because observations read tiles rather than a creep's
team. Only the Markov test would have caught it.

### Randomness is state (`moba/moba_game.h`)

Upstream draws all randomness from the process-global `rand()`, whose state
can't be read or set. `moba_game.h` redirects `rand()` at compile time
(`#define rand() moba_rand()`, then `#include "moba.h"`) to a SplitMix64
stream. Its 64-bit state is `rng_state`.

- **Each `MobaGame` owns its stream** and swaps it in around every call into
  the env. So several games live in one process independently, and a
  restored state continues exactly.
- **The jitter table is a constant.** Upstream fills a cached jitter table
  from `rand()` when the env is built. It is now generated from a fixed seed
  in every game, so it's part of the simulator, and only its index
  (`rng_idx`) is state.

### Terminal state

`moba_step_env()` is upstream `c_step()` without the automatic reset. When an
ancient falls, it returns the result and leaves the terminal state in place.
`moba_game_end()` then does what upstream would have: `add_log` + `c_reset`.
The Markov test checks this path against upstream `c_step`, including 9
windows that run through the terminal step.

## The Markov test (`moba_collect markov`)

The original game A runs on **upstream** `c_step`. At checkpoints (every 250
ticks, plus one window through the final step), A's state S is captured and
A's next 64 steps are recorded. Probes are then built from S in a separate
env B. B is always a freshly started game with a different seed, so anything
not restored from S is wrong. Each probe runs in a forked child with a
timeout, because impossible states can drive upstream into `exit(0)` or an
endless retry loop.

Result on 9 games (91–99), **99 checkpoints**:

| check | result |
|---|---|
| restore → capture reproduces S | **0 / 99 differ** |
| full restore reproduces A for 64 steps: state, all 10 observations, rewards (and the terminal step against upstream `c_step`) | **0 / 99 diverge** |
| scramble each thing *left out* (`last_x/y`, `attack_aoe`, `entity.reward`, logs, target-list entries past the terminator, output buffers, pathing cache) | **0 / 99 diverge, each** |
| leave out each *saved* part (fresh game's value instead) | diverges at 90–99 / 99 for almost every part |

Necessity detail. A part is "detected" if the observation at the restore tick
or anything in the next 64 steps differs.

- **x, y, grid, pids, rng_state: 99/99.** Most other parts: 90/99. The 9
  misses are the tick-0 checkpoints, where a fresh game has the same value.
- **Status timers:** stun 66, move 69, Q 32, W 87. They matter only when
  someone is stunned, slowed or casting in the window.
- **creep_idx 81, reward 11, is_hit 34.** Reward components and `is_hit`
  only feed the *current* observation (they are reset each step), so they
  are needed for observations, not for dynamics.
- **target_pid 0/99:** reset at the start of every step and never read
  before being rewritten. It is kept as an event label (who attacked whom),
  not because the dynamics need it.

## Ego observations as a function of state

Each hero sees a 510-byte `uint8` observation, built by upstream
`compute_observations()` from four inputs: the hero's entity row, `grid`,
`pids` (with the rows they point to), and `reward`. Two copies:

- **C:** `moba_ego_obs()` in `moba/moba_state.h`
- **numpy, batched:** `mobawm.obs.ego_obs()` / `ego_obs_from_states()`,
  which return `(B, 10, 510)`

Both are **byte-exact** against the game for all 10 heroes, upstream quirks
included, so a policy fed from a predicted state sees what it would in the
game.

| bytes | content |
|---|---|
| 0–120 | 11×11 tile ids around the hero, row-major |
| 121–123 | leftover health/mana/level of the last window cells |
| 124–483 | **always 0** |
| 484–485 | hero's absolute x, y |
| 486–500 | level, health×10/max, mana×10/max, damage/50, speed, move modifier, stun/move/Q/W/E/attack timers, attack cooldown, is_hit, team |
| 501–505 | hero class one-hot |
| 506–509 | which reward components fired last step |

**Upstream observation bug, kept on purpose.** The window reserves 4 bytes
per cell but writes channel *c* of cell *i* at byte `i + c` instead of
`4i + c`, so the next cell's tile overwrites it. Agents see *which* unit is
on a cell, never its health, mana or level. The shipped policy was trained
on this. Also reproduced: windows past the left or right edge wrap into the
next row, and floats are truncated to `uint8`.

## Storage and sequence

A game is stored as its seed, which controller drove each team, the 10
action codes per tick (zlib) and an FNV-1a hash of the full `MobaState` of
every tick, terminal included. That is about 11 bytes/step.
`moba_collect replay` rebuilds every state from that, with no controller in
the loop, checks the hash, and with `--dense` writes the states.

**Sequence.**

- A game of *n* steps has *n* + 1 dense records, in order.
- Record *t* < *n* is (s_t, a_t, r_t), captured before the step, with `done`
  on the step that ends the game.
- Record *n* is the terminal state (`final = 1`).
- Within a game, s_{t+1} is always record *t*+1. `shards.transitions()`
  returns every step, including the terminal one.

**Determinism.** Each game depends only on its seed and its actions:
randomness is per-game state, and every game rebuilds the env from scratch.
Game *i* always gets the same seed, so the corpus is identical however it is
split or resumed.

## File formats

**`.mbl` game log** (v3): `MobaLogHeader` (env config, base seed, first game
index, weights hash). Then per game a `MobaGameHeader` (seed, index, steps,
winner, truncated, radiant/dire controller, zlib length, state hash),
followed by `zlib(uint16[n_ticks][10])`. Read with `mobawm.shards.read_log()`.

**`.mbs` dense file** (v3): `MobaDenseHeader` (record size, state size, scan
cap, field count, per-hero constants). Then per game a `MobaDenseGame`
(index, records, zlib length) followed by `zlib(MobaDenseRecord[n + 1])`.
Each record holds:

- game index, tick;
- done / truncated / winner / final;
- radiant / dire controller;
- actions `uint8[10][6]`, rewards `f32[10]`;
- the game's own observation `uint8[10][510]`, kept as a check;
- the `MobaState`.

Read with `mobawm.shards.load_dense()`, then `load_game(k)` or
`iter_games()`. The reader refuses a file whose sizes disagree with it.

## Throughput

| | |
|---|---|
| collection, net on 10 heroes, one process alone | ~1,700–1,900 steps/s |
| collection + full replay, 8 processes (M-series, 4P+4E) | 100 games in 63 s |
| Markov test | ~1 min per 9 games, one process |

## Other upstream findings

- **Death penalty mostly lost.** `step_players` clears the reward buffer
  *after* towers and creeps have acted, so a hero killed by a tower or creep
  gets no death reward. Measured with scripted Dire: 34 of 119 Radiant deaths
  were rewarded (29%). Rewards in the corpus are the env's, unchanged.
- **Stale grid tiles.** `spawn_at` (teleport skills) paints the new cell
  without clearing the old one; one source of tiles with no unit on them.
- **Target lists can overflow.** Creeps scan radius 7 (up to 225 cells) into
  121-entry rows. In practice lists are short; capture fails loudly past 64.
- **Impossible states hang or exit.** `spawn_player` retries a random
  position until one is free (can loop forever), and several checks call
  `exit(0)`.
- **Bloom shader on macOS.** `bloom_shader_330.fs` is GLSL 1.00 declared
  `#version 330`, which macOS rejects; the demo logs a warning.
