# E30: stored target lists as state (A0 vs A1b)

**Status:** concluded 2026-10-03. **Verdict: no tower gain on the 100-game evaluation.**

## Question
Towers and creeps attack the nearest unit in a **stored target list** (`env->scanned_targets`). Towers rescan
every 3 ticks and creeps every 5, so the list is stale in between. The dense files record it (`state['scan']`),
but the 26 entity fields don't. Does carrying it as state fix tower damage?

## Evidence that it matters
- Nearest-of-stored-list explains 97.5% of tower shots, against 63% for the nearest-enemy rule.
- From these lists, `incoming_damage` (the sum of ready attackers' damage) predicts tower hits at
  91–93% precision and 85–86% recall, exact on 76–80% of hits (games 900, 903).
- Keeping the list rather than "nearest": with copy plus rescan on the unit's tick, aggro is right on 60% of
  tower changes and 90% of creep changes. A pointer predicting "nearest" directly got 10% and 3%.

## Setup
- **State:** 6 entity fields `lst0..5` carry 24 membership bits each, for creep and tower owners (124) over
  heroes, creeps and towers (134).
- **Inputs:** 13 aggro inputs (`data.aggro_inputs`) through a zero-initialised projection.
- **Output:** a list-membership head predicts next-tick lists. It is a dot product plus a learned table,
  initialised to "keep".
- **Dream:** an empty list for a unit predicted dead. That is an untested assumption added without approval;
  it is flagged here.
- **Run:** A1b = E12 @4000 + `--list-state`, E12 recipe (batch 768, lr 1e-3), 500 updates; A0 = control.

## Results
- **100-game tower evaluation** (`results/moba/tower_big/TABLE.md`): A1b @375 scored 32% / 61% / 0.65×,
  against A0 @375 at 28% / 76% / 0.86×. That is no gain.
- **In-dream check** (A1b @375):
  - the dream's own `incoming_damage` is 41% of the real value during sieges, because tower lists are exact
    only 30% of the time;
  - **90% of dream siege damage lands on ticks where its own incoming_damage is 0**, so the health output
    ignores the input.
  - A direct-damage path (`--direct-damage`) was implemented, but only smoke-tested.

**Artifacts:** S3 `moba/runs/a0_e12_control`, `a1b_e12_lists`; W&B `pvgc94tc` (A0), `6jfqu69i` (A1, superseded by A1b).
Code: `data.LIST_STATE`, `pack_lists_np`, `aggro_inputs`, `moba_fork.c:ms_replay_scan`.
