# E32: stability sweep: learning rate × EMA (S1–S4)

**Status:** concluded 2026-10-04. **Verdict: lr 1e-4 is the stable regime. EMA does not rescue lr 1e-3.**

## Setup
- **Start:** every arm continues G1 @500 (grid CNN) at the E12 recipe for 1,000 updates.
- **Scoring:** every checkpoint (every 125 updates) is scored by the 100-game tower evaluation next to training
  (`scripts/moba/eval_watch.py`).

## Results (`results/moba/tower_big/TABLE.md`, per-checkpoint sections)
| arm | setting | tower precision over the run | recall | siege |
|---|---|---|---|---|
| G2 | lr 1e-3 | 54% → 26% → 21% → 16% | 62–75% | 0.68–0.79× |
| S3 | lr 1e-3 + EMA 0.999 | 27% → 21% | 70–74% | 0.72–0.75× |
| S1 | lr 3e-4 | 29–37% | 66–77% | 0.70–0.87× |
| S4 | lr 3e-4 + EMA | 33–36% | 72–74% | 0.72–0.77× |
| **S2** | **lr 1e-4** | **32–36%** | 70–73% | 0.73–0.78× |

## Caveat found later
Training speed varied 2.5× by host, 0.16–0.43 updates/s. Most of that was not the host itself but the
attention-bias einsum backward (E34).

**Artifacts:** S3 `moba/runs/s1_grid_lr3e-4`, `s2_grid_lr1e-4`, `s3_grid_ema`, `s4_grid_lr3e-4_ema`; G3 (S2 continued at lr 1e-4)
`g3_grid_lr1e-4_long` (W&B `o5v97p56`). Code: `--ema` (train.py).
