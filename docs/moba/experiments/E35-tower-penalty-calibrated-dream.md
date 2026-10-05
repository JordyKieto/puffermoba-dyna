# E35: false-tower-death penalty, a calibrated dream, and dream PPO in it (P16, P17)

**Status:** concluded 2026-10-05. **Verdict:**
- the penalty moved tower precision for the first time (40% → 71–77%);
- at ×1.5 the dream predicts tower falls at the real rate;
- dream PPO in these dreams did not learn to take towers (P16, P17).

## The penalty
- `--tower-fp-weight W` (train.py): in the tower death loss, every "this tower does not die this tick" example
  weighs ×W, so a false tower death costs W× more. It is learned, not a rule.
- **Sweep** (`results/moba/tower_big/TABLE.md`, 100-game evaluation, 1,561 real falls):
  - setup: 20 fleet pods plus one A100, 300 updates each from P1 @1000;
  - precision rises and recall falls smoothly with W;
  - **best F1 at lr 3e-4, ×2.5–×4** (68% / 64%, 1,460 predicted falls).
- **Long run:** fleet20c (×2.5, lr 3e-4, 20 pods) held 74–77% precision and F1 0.65 over 1,500+ updates, but drifted
  cautious (1,309 → 1,131 predicted falls).

## Calibration
| world model | predicted falls | precision | recall | siege | F1 |
|---|---|---|---|---|---|
| E12 @4000 | 2,526 | 40% | 65% | 0.65× | 0.50 |
| fleet20c ×2.5 @1000 (P16's dream) | 1,162 | 76% | 57% | 0.73× | 0.65 |
| fleet20d ×2 @500 (P17's dream) | 1,294 | 76% | 63% | 0.76× | 0.69 |
| **fleet20e ×1.5 @500** (`runs/token_wm/fleet20e_at500`, md5 `d843b9ff`) | **1,513** | 71% | 68% | 0.77× | 0.69 |

## Dream PPO (P16 in fleet20c, P17 in fleet20d)
- **Setup:** arm C fine-tuned against the shipped net (ours on a random side), moderate or strong tower/win shaping,
  1.7M steps, then 60 real games per side.
- **Real games:**
  - towers taken as radiant fell from 1.6 (arm C) to 1.1–1.2 (moderate) and 0.3 (strong);
  - 0 wins;
  - dire lost all games.
- **In the dream,** the net out-took ours about 10× (0.016 vs 0.14–0.19 towers per episode).
- **Calibrating the dream did not change this** (P16 at 74% of real falls, P17 at 83%). That led to the reward sweep (E36).
