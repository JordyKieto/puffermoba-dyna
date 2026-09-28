<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

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
