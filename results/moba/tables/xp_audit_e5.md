<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Our team (radiant) vs the shipped net, 8 real games x 900 ticks, seeds 70000000+; identical actions fed to the WM. scripts/moba/dream_vs_real_reward.py

| world model | our team's actions | measure | deaths/1k | XP/1k | tower kills/1k |
|---|---|---|---|---|---|
| E5 @20,500 | ppo | real game | -0.00 | 0 | 0.000 |
| E5 @20,500 | ppo | dream, teacher-forced | -0.00 | 16992 | 0.000 |
| E5 @20,500 | ppo | dream, open loop 128 | 0.16 | 4074 | 0.065 |
| E5 @20,500 | net | real game | 2.94 | 624 | 0.194 |
| E5 @20,500 | net | dream, teacher-forced | 0.14 | 741 | 1.639 |
| E5 @20,500 | net | dream, open loop 128 | 1.60 | 385 | 0.130 |
| E5 @20,500 | random | real game | 0.03 | 15 | 0.000 |
| E5 @20,500 | random | dream, teacher-forced | -0.00 | 8298 | 0.000 |
| E5 @20,500 | random | dream, open loop 128 | 0.49 | 1051 | 0.000 |
