# E9 — Dyna smoke: the policy's own real games cut XP hallucination 3–7×

**The one-line result.** Adding 100 real games played by the dream-PPO policies (~8% of
the data) to world-model training cuts the dream's XP hallucination on those policies'
states 3–7× within ~125 updates, compared with the same fine-tune on mix1k alone.

- **Status:** concluded as a smoke test, 2026-09-27. Single seed, ~125 updates.
- **Data (`dyna1`):** real games, game ids 5000–5099, seeds 80,000,000+.
  - Both P3 policies (50 games each, alternating sides) vs the shipped net, capped at
    3,000 ticks; 198,131 ticks.
  - Recorded as standard `.mbl` logs (controller id 2 = dream_ppo) by
    `scripts/moba/collect_policy_games.py`, with the collector's state hash.
  - `build/moba_collect replay` regenerated the dense states with **0 hash and 0
    observation mismatches**.
  - In `s3://puffertank/moba/dyna1/`; per-game list in `results/moba/dyna/dyna1_games.json`.
- **Runs** (W&B group `E9-dyna-smoke`, exported to `results/moba/usrc41qc/` and
  `results/moba/tnnlmx2e/`): both from E5 @20,500 with `--rollout --store-cpu`, batch
  384, 10 min in parallel on the H100 NVL.
  - Control: mix1k 0–899.
  - Dyna: mix1k 0–899 + 5000–5099.
  - Checkpoints are in the manifest.

## Result (audit seeds 70,000,000+, not the Dyna games)

Both from E5 @20,500, ~125 updates, batch 384, H100 NVL, single seed; audit seeds 70000000+ (the Dyna games used 80000000+).

| world model | our team's actions | measure | deaths/1k | XP/1k | tower kills/1k |
|---|---|---|---|---|---|
| control: mix1k only | ppo | real game | -0.00 | 3 | 0.000 |
| control: mix1k only | ppo | dream, teacher-forced | -0.00 | 44999 | 0.000 |
| control: mix1k only | ppo | dream, open loop 128 | 0.03 | 5178 | 0.130 |
| control: mix1k only | net | real game | 2.94 | 624 | 0.194 |
| control: mix1k only | net | dream, teacher-forced | 0.25 | 3055 | 2.861 |
| control: mix1k only | net | dream, open loop 128 | 1.11 | 890 | 0.098 |
| Dyna: mix1k + 100 policy games | ppo | real game | -0.00 | 0 | 0.000 |
| Dyna: mix1k + 100 policy games | ppo | dream, teacher-forced | -0.00 | 6283 | 0.000 |
| Dyna: mix1k + 100 policy games | ppo | dream, open loop 128 | -0.00 | 1643 | 0.000 |
| Dyna: mix1k + 100 policy games | net | real game | 2.94 | 624 | 0.194 |
| Dyna: mix1k + 100 policy games | net | dream, teacher-forced | 0.28 | 2366 | 2.944 |
| Dyna: mix1k + 100 policy games | net | dream, open loop 128 | 1.53 | 955 | 0.293 |

- **Dyna vs control on the policy's own actions:** teacher-forced 44,999 → 6,283,
  open loop 5,178 → 1,643.
- **Both short fine-tunes degraded the model compared with E5 itself:** they restarted
  with a fresh optimizer and a warm-up at lr 1e-3, and only ~125 updates ran at
  0.2 updates/s, because the two runs shared 15 CPUs for data gathering. So compare Dyna
  with control, not with E5.
- **Not yet shown:** that a policy trained in the Dyna-improved dream does better in the
  real game. That's the loop's next step.
