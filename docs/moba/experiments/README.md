# MOBA world model: experiments index

## Headline (2026-09-25)

- **Current baseline: v2.** A change gate on every field except movement,
  balanced gate losses, and the hero walking loss weighted ×10. Defined in
  [../EXPERIMENTS.md § Current baseline](../EXPERIMENTS.md).
- **Promising: [E5 — rollout training](E5-rollout-training.md)**, still
  running (6M updates). The model is trained on its own predictions, over
  spans from 1 tick to a whole game, continuing from E4. After 500 updates:
  - Held-out loss at 64–4,096 ticks is 2–6× lower.
  - Heroes stay within about 13 cells of the real ones over 4,096 ticks,
    against 123 before.
  - In the viewer the rollouts stay stable, with no degeneration over time.
  - The cost is one-step precision: h1 loss 0.78 → 4.4, and heroes jitter at
    2× the real speed.
- **How to judge a model:** watch it next to the real game in the
  [replay viewer](../VIEWER.md), fed the same actions.
- **Open problem being reframed:** the old per-ability "better than copying"
  numbers mostly measured cooldown countdowns. Effects are now judged per
  state variable, starting with hero position.

## Runs

| id | what | status | record |
|---|---|---|---|
| E4 | baseline v2, full mix1k, 1 h, batch 768 bf16, checkpoints every 500 | done | [E4-baseline-v2-1h.md](E4-baseline-v2-1h.md) |
| E5 | autoregressive (rollout) training, continuing from E4, 6M updates | running | [E5-rollout-training.md](E5-rollout-training.md) |
| E0–E3c, 5-minute ablations | earlier runs and ablations | done | legacy log: [../EXPERIMENTS.md](../EXPERIMENTS.md) |

The earlier work is still in the single legacy log. New runs get one file
each here.
