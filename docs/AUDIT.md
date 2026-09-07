# Audit: `ocean/whisker_racer` ("Race")

Upstream: [PufferLib](https://github.com/pufferai/pufferlib) branch `4.0`, pinned at
`42f70d6932c30ac977736f861006809c50168ba9` (2026-08-20), vendored at
`third_party/pufferlib` and never edited.

"Race" on [puffer.ai/ocean.html](https://puffer.ai/ocean.html) is `whisker_racer`.
`puffer.ai/docs/games.js:227` keys the entry `whisker_racer` with
`title: "Race"` — "Race" is the display name only.

The environment: single agent, constant speed, three discrete steering actions,
two whisker rangefinders, a procedurally generated Bézier loop.
**The observation is 3 floats**: `[flw_length, frw_length, score/100]`.

Every finding below is marked **verified** (reproduced by running code) or
**read** (established from source). Reproduce the verified ones with:

```bash
tools/build.sh eval && tests/test_determinism.sh    # B1-B3
clang -O1 -w -I... tests/upstream_probe.c -o probe && ./probe   # B1, B2 on upstream
```

---

## A. Markov violations

The transition function reads state that the observation does not expose.

| # | Gap | Consequence | Status |
|---|---|---|---|
| A1 | **Heading `ang` absent** | The only genuinely dynamic variable: `px += v·cos(ang)`, `py += v·sin(ang)`. Without it the transition is unidentifiable. | read |
| A2 | **Position `px, py` absent** | Reward and progress are defined by `atan2(py-cy, px-cx)`. | read |
| A3 | **`near_point_idx` absent** | Not a convenience field. `calc_whisker_lengths` only raycasts segments within ±5 of it, and `update_nearest_point` searches forward `0..+3` from its previous value, so it carries history. Next whiskers depend on it. | read |
| A4 | **Sector state absent** (`current_sector`, `sectors_completed[16]`, `total_sectors_crossed`) | Every unit of reward is a function of these. A policy cannot predict its own reward. | read |
| A5 | **No timer, and no truncation at all** | `tick` is not observable and there is no time limit — episodes end *only* on crash. | read |
| A6 | **Track geometry never observed** | The dynamics intersect `inner_edge`/`outer_edge` every step. Two envs give identical observations under different dynamics. | **verified** |
| A7 | `score` in obs is unbounded | `÷100` is arbitrary; grows without limit and aliases across laps. | read |
| A8 | 3 of 5 whiskers are commented out | `llw_length`, `ffw_length`, `rrw_length` still declared and still drawn by the renderer. | **verified** |

**A6 is the subtle one.** Measured with `racewm/markov.py` on 231k transitions
across 32 tracks — held-out NMSE, same MLP, same budget, varying only the input:

| state given to the model | NMSE | gain from adding history | verdict |
|---|---|---|---|
| 30-dim contract, no track context | 0.160 | **+19.3%** | not Markov |
| 30-dim contract + `track_id` one-hot | 0.167 | **+16.6%** | still not Markov |
| 30-dim contract + local ego geometry | **0.089** | **+4.0%** | **Markov** |
| upstream 3-float obs (control) | 0.543 | +22.3% | control confirms the test has power |

A one-hot track ID does **not** repair it — it asks one MLP to memorize 32
separate dynamics functions. The 44 ego-frame edge points that
`compute_whiskers` actually reads do, and they nearly halve base error.

---

## B. Determinism blockers

| # | Finding | Status |
|---|---|---|
| B1 | `init()` calls `srand(env->rng + env->i)` — the **process-global** libc stream. Every env's init reseeds a stream every other env shares. | **verified** |
| B2 | `get_random_start()` runs on every reset and draws from that same global `rand()`. Under the threaded vec backend, episode starts depend on reset ordering. | **verified** |
| B3 | `src/vecenv.h:363` sets `envs[i].rng = i`, then `my_init` **overwrites** it with the config constant `rng = 6`. Combined with a fixed `i`, **every env generates the identical track**. | **verified** |

`tests/upstream_probe.c` demonstrates B1–B3 against unmodified upstream:

```
env 0  track_hash=41c4d70ae0a1f77f
env 1  track_hash=41c4d70ae0a1f77f      <- all four envs, one track
env 2  track_hash=41c4d70ae0a1f77f
env 3  track_hash=41c4d70ae0a1f77f

order A,B ->  A start=(852.68, 302.15)  B start=(254.33, 454.87)
order B,A ->  A start=(254.33, 454.87)  B start=(852.68, 302.15)
```

Same seed, same envs, different reset order, different states.

---

## C. Reset-contract bugs

| # | Finding | Status |
|---|---|---|
| C1 | `c_reset()` calls `compute_observations()` **before** `score = 0` and **before** `reset_round()`. Every episode opens on stale data from the previous episode. | read |
| C2 | Auto-reset fires inside `calc_whisker_lengths` → `step_frame` → which runs `frameskip` (4) times per `c_step`. A crash on sub-step 0 leaves the env stepping the *new* episode for 3 more sub-steps; `tick` is zeroed mid-step then re-incremented, so `episode_length` is wrong. | read |
| C3 | That crash check sits inside the `for w in 0..2` whisker loop. If `w=0` crashes and resets, `w=1` continues against the already-reset car and can crash and `add_log` **again** — double-counted episodes. | read |
| C4 | `c_step` zeroes `terminals[0]` once at the top, so two terminals inside one frameskip collapse to one. | read |

---

## D. Config that does not do what it says

| # | Finding | Status |
|---|---|---|
| D1 | **`puffer train whisker_racer` cannot start.** `my_init` calls `dict_get` for `llw_ang`, `rrw_ang`, `flw_ang`, `frw_ang`, `render`, `i` — none exist in `config/whisker_racer.ini` or in the empty `[env]` block of `config/default.ini`. `dict_get` prints `dict_get failed to find key:` then asserts (`src/vecenv.h:47`). The whisker angles are derived from `w_ang` inside `init()` anyway. | read |
| D2 | `sectors_completed[]` is sized 16 and the math hardcodes `PI2/16.0f`; the `% num_radial_sectors` is a no-op for any S ≥ 16. **`num_radial_sectors = 180` is dead**, and the sweep over 8..100 does nothing. | read |
| D3 | `reward_green = -0.001` is configured and swept but **never read**. | **verified** |
| D4 | `max_score` is declared and never assigned → `log.perf = score / 0`. The headline metric is inf/nan. | **verified** |
| D5 | `continuous` computes `act = action`, then `act` is never used. Continuous steering is dead, contradicting the published description "Actions: Discrete and continuous steering supported". | **verified** |
| D6 | `gamma`, `ftmp1..ftmp4`, `half_max_score` declared, never read. (`corner_thresh` **is** used — in track generation.) | **verified** |
| D7 | "Full lap" reward fires on every re-cross of an already-completed sector. There is no lap concept. | read |
| D8 | `inv_pi2 = 1.0f / PI2` is computed and never read — and `#define PI2 PI * 2` is unparenthesized, so it evaluated as `(1/π)·2`, not `1/2π`. | **verified** |

---

## E. Tooling gaps

| # | Finding | Status |
|---|---|---|
| E1 | `VecEnv` exposes only `obs_ptr / rewards_ptr / terminals_ptr / reset / cpu_step / render / log / close` (`src/bindings_cpu.cpp:179`). No state export, no per-env seeding, no `set_state`. | read |
| E2 | PufferLib 4.0 has **no truncation channel** anywhere — every ending is a terminal. | read |
| E3 | `build.sh` hardcodes `-mavx2 -mfma` (x86-only; `drive.h` and `bf16.h` use AVX2 intrinsics) and `-fopenmp`, which Apple clang rejects. macOS builds fail outright. | **verified** |
| E4 | raylib publishes no arm64 Linux binary, so an arm64 container cannot link `raylib-5.5_linux_amd64`. Deferred; see `docs/CONTRACT.md`. | **verified** |

---

## What we changed

Scope was: **fix determinism and the Markov gaps; do not preserve upstream
behavior for its own sake.** Reward semantics are unchanged — `reward_yellow`
on any forward sector crossing, first time or not, exactly as upstream — so
returns remain comparable.

| Area | Change |
|---|---|
| B1–B3 | Two private PCG32 streams per env, seeded from `(seed, env_idx)`: one for track geometry, one for episode starts. No `srand`/`rand` anywhere. |
| A1–A5, A7 | Observation is now the full 30-dim Markov state. See `docs/CONTRACT.md`. |
| A6 | Track geometry dumped per `env_idx`; the world model consumes it as 44 ego-frame edge points. Not added to the agent's observation. |
| A5 | `max_episode_steps` with real truncation. `terminated` and `truncated` are tracked separately even though PufferLib collapses both onto `terminals` (E2). |
| C1–C4 | `c_step` owns the episode boundary. `compute_whiskers` is pure; crash detection raises a flag; reset happens once, and the observation is computed after it. |
| D1 | Missing config keys added; vestigial per-whisker angle inputs dropped. |
| D2 | `num_radial_sectors` is honored at runtime. Default set to 16 — the value the env actually ran — so the config stops lying. |
| D4 | `perf` is now fractional laps completed. `lap_count` and `sector_crossings` added as telemetry. |
| D3, D5, D6 | Left in place and inert, for config compatibility. Flagged here. |
| E1 | Avoided rather than patched: the observation *is* the state, so `obs_ptr` is sufficient and upstream needs no changes. |
| E3 | `tools/build.sh` compiles the standalone targets directly, skipping the x86/OpenMP assumptions. |

Upstream's pretrained `resources/whisker_racer/puffer_whisker_racer_weights.bin`
expects `obs_size = 3` and no longer loads. That is expected.
