# MOBA world model: experiments index

## Headline (2026-09-25)

- **⚠ The world model is not fully learned.** Spawns, creep constants, hero
  level and level-derived stats, counters, targets and clamps are filled in by
  **hand-coded rules** (R1–R10), and every rollout and viewer result depends
  on them. Everything dynamic (movement, damage, cooldowns, deaths, towers) is
  learned. Full table:
  [WM_ARCHITECTURE.md § Rules vs learned](../WM_ARCHITECTURE.md#rules-vs-learned-what-the-model-does-not-predict).
  Not signed off; decision pending.

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
- **Also running: [E6b — counterfactual siblings + effect losses](E6-counterfactual-siblings.md).**
  The real sim forks every recorded state into edited-action branches, giving
  about 100× the corpus in counterfactual content
  ([how](../COUNTERFACTUALS.md)).
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
| E6 / E6b | E5 + counterfactual siblings from the real sim (32k per update), from E5 @3,500; E6b adds the effect auxiliary losses | E6b running | [E6-counterfactual-siblings.md](E6-counterfactual-siblings.md) |
| E0–E3c, 5-minute ablations | earlier runs and ablations | done | legacy log: [../EXPERIMENTS.md](../EXPERIMENTS.md) |

The earlier work is still in the single legacy log. New runs get one file
each here.
