# MOBA world model: experiments index

## Headline (2026-09-25)

- **Current baseline: v2.** A change gate on every field except movement,
  balanced gate losses, and the hero walking loss weighted ×10. Defined in
  [../EXPERIMENTS.md § Current baseline](../EXPERIMENTS.md).
- **Now running:** [E4 — baseline v2, 1 hour](E4-baseline-v2-1h.md) on the
  H200, with numbered checkpoints.
- **How to judge a model:** watch it next to the real game in the
  [replay viewer](../VIEWER.md), fed the same actions.
- **Open problem being reframed:** the old per-ability "better than copying"
  numbers mostly measured cooldown countdowns. Effects are now judged per
  state variable, starting with hero position.

## Runs

| id | what | status | record |
|---|---|---|---|
| E4 | baseline v2, full mix1k, 1 h, batch 768 bf16, checkpoints every 500 | running | [E4-baseline-v2-1h.md](E4-baseline-v2-1h.md) |
| E0–E3c, 5-minute ablations | earlier runs and ablations | done | legacy log: [../EXPERIMENTS.md](../EXPERIMENTS.md) |

The earlier work is still in the single legacy log. New runs get one file
each here.
