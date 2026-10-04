# E31: global map + CNN encoder (G1, G2)

**Status:** concluded 2026-10-03. **Verdict: no stable tower gain. The gains came from learning rate (E32/E33).**

## Setup
- **Model:** TokenWM + `--grid-cnn`.
  - A 64×64 map (2 cells per pixel) is drawn from the state each step, with 10 channels: walls; heroes,
    creeps, attack-ready damage and tower health per team; neutrals.
  - A dilated 3-layer CNN reads it. Its features at each unit's pixel enter the token through a
    zero-initialised projection.
  - In the dream the map is redrawn from the predicted state.
- **Runs:** G1 = E12 @4000 + grid, E12 recipe (lr 1e-3), 500 updates. G2 continued G1 @500 for 1,500 more,
  with every checkpoint scored by the 100-game evaluation.

## Results (`results/moba/tower_big/TABLE.md`)
- **G1 @375: 45% / 67% / 0.75×.** It is the only checkpoint above E12 @4000 (40% / 65% / 0.65×) on all three.
- **At lr 1e-3 it is unstable.**
  - G2 precision went 21% → 33% → **54%** (@375) → 26% → 21% → 22% → **16%** (@875, 7,019 predicted falls
    for 1,561 real).
  - Averaging finished checkpoints did not help: 22–28%.
- **A local readout probe** trained a CNN on real states (not inside the WM). It found tower hits at 81%
  recall on real states, but did worse than the WM's own health output when run on dream states.

**Artifacts:** S3 `moba/runs/g1_e12_grid` (W&B `6odugwtq`), `g2_grid_long`.
