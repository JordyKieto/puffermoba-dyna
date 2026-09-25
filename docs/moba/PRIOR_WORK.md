# Prior work: what was carried over from `pufferlib-ocean-wm`

An earlier study (the `pufferlib-ocean-wm` fork of PufferLib 4.0, phases 0–3)
built symbolic world models on the same Ocean MOBA. Its phase 0/1 ledgers
contain most of what matters for making the game state Markov. This page
records what was taken from it, what was not, and why, so the lessons stay
fixed while the implementation here moves on.

Source: `pufferlib-ocean-wm-4.0/docs/phases/00-environment.md`,
`01-dynamics.md`, `ocean/moba/moba.h`, `tools/moba_oracle_v3.py`,
`tests/dynamics/moba_oracle_v3_sufficiency.c`.

## Taken

### 1. Test state sufficiency by restore + intervention

The old study's most valuable method. Its MOBASD2 state was falsified this
way: restoring states that encoded identically but differed in omitted
simulator state gave **351 of 391 divergent futures**. No architecture can
learn a target that the input does not determine.

Implemented here as `moba_collect markov` (see [DATA.md](DATA.md)). It keeps
the old test's three parts:

- a full restore must reproduce the future;
- omitting each saved part must be detectable;
- scrambling each deliberately omitted thing must change nothing.

The positive control is `rng_state`: omitting it is detected 99/99.

**One gap in the old test is closed.** The old `moba_restore_oracle_state`
**clears every target list**, and every branch in the old sufficiency test
went through that restore. Both sides of each comparison were wiped equally,
so the test could not see that target lists are state. They are: creeps and
neutrals rescan every 5 ticks and towers every 3, and in between they act on
the stored list. The test here compares every restore against the original,
never-restored game, and stores the lists. Omitting them is detected at
90/99 checkpoints.

### 2. Randomness is part of the state

The old fork replaced the process-global `rand()` with a per-env SplitMix64
whose state was exported. That makes "Markov" a byte-exact, testable claim at
any tick. Taken, but without forking upstream: `moba/moba_game.h` redirects
`rand()` at compile time and gives each game its own stream.

### 3. The terminal state must be observable

The old `eval_once` flag stopped the env at the end of a game instead of
resetting. Taken as `moba_step_env()`, a copy of upstream `c_step()` without
the reset. It is verified against upstream `c_step` in the Markov test, and
needs no patch.

### 4. Record the joint action

The old MOBASD1 contract failed partly because each row held one hero's
action against a global transition. The same transition then appeared five
times with different "actions", which contradict each other. The old
checklist's rule: *record the joint action unless the omitted actors are
deterministic functions of the exported state.* Here all 10 heroes act
through the action interface, and every tick records all 10 actions.

### 5. Lessons for the transition model (not yet applied)

Phase 1 findings to carry into the model work:

- **Copying the state forward is a very strong baseline.** Only about 1–3%
  of values change per tick. Report sparse-change metrics (changed fields,
  events, false changes), not aggregate error that copying dominates.
- **Passive whole-world prediction at 4–8 steps did not beat copying**, even
  with a sufficient state. It is a learning and objective problem, not a
  missing variable. What worked was factoring effects by mechanic, predicting
  values only where something changes, and copying everything else exactly.
- **Hold entity identities out** between train and validation. Fixed slots
  (e.g. "tower 191 changes") are otherwise memorised.
- **Things that failed there,** from the old ledger: monolithic GRU,
  Transformer, RSSM and JEPA simulators; dense losses without change
  weighting; and teacher forcing that hid compounding error. Some of these
  failures predate a sufficient state, so they are not verdicts on the
  architectures.
- **Offline fit and closed-loop behaviour answer different questions.**
  Never substitute one for the other.

### 6. Coverage and splits

- **Measure event coverage per configuration:** kills, deaths, towers, both
  outcomes. Here Radiant wins 98/100, so Dire wins are nearly absent.
- **Split by game seed, never by row.**
- **Keep one controller configuration held out** as a distribution shift.

## Not taken

- **The 615-line env fork.** Scenario configs (focal hero, spawn, level),
  head-to-head mode, script variants and the one-controlled-hero framing are
  superseded by 10 heroes acting through the action interface. Upstream
  stays untouched here.
- **The observation fix.** The fork corrected the overlapping 11×11×4
  channels (`4i + c`). The upstream pretrained policy was trained on the
  buggy layout, so the observation here stays bug-compatible. A corrected
  variant is worth adding only if new policies are trained.
- **The reward-clearing fix.** The fork clears reward components at the start
  of `c_step`, which fixes lost death rewards but changes upstream's
  behaviour. The rewards here are the env's, unchanged; the loss is
  documented.
- **The 42-field full-struct dump.** It copied every `Entity` field,
  constants included. Here only fields that change during a game are stored
  (26), and the Markov test checks that the rest is constant or overwritten
  before it is read.
- **Occupancy reconstructed from entities.** MOBASD3 stored entities and
  rebuilt occupancy. The grid is not derivable from entities: the env leaves
  stale tiles, and in one game 7,488 creep tiles in agents' views had no live
  creep. Grid and pids are stored.
- **W&B/S3 evidence rules, the planner and event-pair generators.** Not
  needed for data collection. Revisit with the model work.
