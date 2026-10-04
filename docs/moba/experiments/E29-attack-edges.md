# E29: predicted attack edges (E28c follow-up, K0–K3)

**Status:** concluded 2026-10-03. **Verdict: the edge fixes improve edge accuracy, not towers.**

## Question
E28c (from scratch, predicted attack graph, ×10 attacker-weighted edge loss, tower loss ×5, 384 scenario lanes)
reached only 2–3% attacker accuracy after 3,675 updates. Why, and do the fixes help towers?

## Diagnosis (E28c @3750, held-out game 900, real state each tick)
- `target_pid` is per-tick, not persistent: `c_step` clears it every tick (moba.h:1890). It is set only
  when damage lands *without killing* (moba.h:704–710). **The killing blow is never recorded:** 4 of the 6
  tower deaths in that game had no attacker edge on the death tick.
- Only 2.6% of living units have an edge on a given tick. Firing is almost exactly "attack timer == 1":
  63% of creeps fire at timer 1 and 0% at timers 2–5. The model learned a smooth fall-off instead.
- The model put "nobody" first for 92% of attacking creeps and towers. When forced to name a unit, it was
  right 45–53% of the time, against 83% for creeps and 63% for towers from the nearest-enemy rule.
- **Cause found in the code:** the edge loss had no rollout-depth decay, so deep dream ticks, where targets
  are unknowable, counted fully.
- A frozen-trunk probe (3 heads, 3,000 steps) separated the halves. A distance bias raised "right unit"
  from 58% to 82%. A linear gate learned "fires?" too slowly.

## Arms (all continue E28c @3750 at its full config, batch 768 + 384 scenario lanes, 500 updates, H200 each)
| arm | change | @250/500 edges exact | towers 128-tick (precision / recall) |
|---|---|---|---|
| K0 | control | 31% / 34% | 5% / 57% → 9% / 61% |
| K1 | + edge-loss decay | 40% / 45% | 10% / 9% → 5% / 35% |
| K2 | + decay + distance bias | 49% (@375) / 38% | 6% / 22% → 3% / 17% |
| K3 | + decay + distance + rebuilt kill edges | 45% (@250, own labels) | 5% / 22% |

These are the 64-window in-training tower numbers, which E33/EVIDENCE_RULES §7 later showed to be too noisy.

## Lessons
- The distance bias needed an embedding lookup. With advanced indexing its backward was 883 ms per step,
  against 41 ms (H200).
- Kill-edge rebuild: a unit whose attack timer resets to its full cooldown with no recorded target hit the
  nearest enemy that died within 12 cells. This matches 82% of such resets, with about 4% ambiguous.

**Artifacts:** S3 `moba/runs/k0_e28c_control`, `k1_edge_decay`, `k2_decay_dist`, `k3_decay_dist_kill`; W&B `uoztox05` (K0),
`kh3gzhtl` (K2, first launch), `oluiny09`/`o9vqrjij` (K3). Code: `--target-decay`, `--target-dist`, `--kill-edges` (train.py),
`data.rebuild_kill_edges`.
