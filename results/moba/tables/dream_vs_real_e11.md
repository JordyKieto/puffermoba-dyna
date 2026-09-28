<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

scripts/moba/dream_vs_real_reward.py: 8 real games x 900 ticks vs the shipped net, seeds 70000000+; reward = death + XP + tower terms per 1k hero-steps of our team.

| our team | world model | real reward/1k | dream, teacher-forced | dream, open loop 128 | real XP/1k | dream XP TF | dream XP OL |
|---|---|---|---|---|---|---|---|
| P4 policy | E5 @20,500 | 1.21 | 86.31 | 14.28 (real 1.20) | 188 | 12969 | 2146 (real 186) |
| P4 policy | E11 @1500 | 1.45 | 19.13 | 5.64 (real 1.43) | 226 | 2874 | 848 (real 225) |
| random | E5 @20,500 | 0.09 | 55.24 | 6.77 (real 0.08) | 15 | 8298 | 1031 (real 13) |
| random | E11 @1500 | 0.09 | 0.36 | 0.00 (real 0.08) | 15 | 54 | 0 (real 13) |
| net | E5 @20,500 | 3.79 | 5.96 | 2.34 (real 3.68) | 624 | 741 | 385 (real 612) |
| net | E11 @1500 | 3.79 | 10.84 | 1.70 (real 3.68) | 624 | 1362 | 268 (real 612) |
