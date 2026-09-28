<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

E14 arms, 1.70M dream steps each unless stated; 10 games per side, seeds 50000000+ (mobawm/rl/arena.py).

| our team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| P6 base (MLP, = P4b) | radiant | 10 | 0 | 0 | 12.80 ± 0.58 | 22.1 | 5.66 | 0.22 |
| P6 base (MLP, = P4b) | dire | 10 | 0 | 100 | 0.00 ± 0.00 | 0.0 | 0.00 | 0.00 |
| P6 MLP + role heads | radiant | 10 | 0 | 0 | 9.19 ± 0.72 | 17.6 | 4.64 | 0.06 |
| P6 MLP + role heads | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P7 B: PufferLib MinGRU + old-repo PPO | radiant | 10 | 0 | 0 | 6.47 ± 0.52 | 9.7 | 2.96 | 0.14 |
| P7 B: PufferLib MinGRU + old-repo PPO | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| P7 C: PufferLib MinGRU + our PPO | radiant | 10 | 0 | 0 | 17.98 ± 0.58 | 28.4 | 6.66 | 0.32 |
| P7 C: PufferLib MinGRU + our PPO | dire | 10 | 0 | 100 | 0.62 ± 0.21 | 0.5 | 0.72 | 0.00 |
| P8 C, entropy 0.003 | radiant | 10 | 0 | 0 | 8.15 ± 0.29 | 15.6 | 4.52 | 0.08 |
| P8 C, entropy 0.003 | dire | 10 | 0 | 100 | 0.09 ± 0.05 | 0.2 | 0.28 | 0.00 |
| P8 C, entropy 0.03 | radiant | 10 | 0 | 0 | 5.53 ± 0.42 | 12.2 | 3.74 | 0.10 |
| P8 C, entropy 0.03 | dire | 10 | 0 | 100 | 0.70 ± 0.22 | 0.7 | 1.12 | 0.00 |
| P8 C, entropy 0.05 | radiant | 10 | 0 | 0 | 7.40 ± 0.61 | 14.7 | 4.12 | 0.14 |
| P8 C, entropy 0.05 | dire | 10 | 0 | 100 | 0.75 ± 0.17 | 0.6 | 1.12 | 0.00 |
| P8 C, entropy 0.1 | radiant | 10 | 0 | 0 | 8.89 ± 0.54 | 15.2 | 4.60 | 0.20 |
| P8 C, entropy 0.1 | dire | 10 | 0 | 100 | 0.99 ± 0.33 | 1.8 | 1.46 | 0.00 |
| P9 C at 2x steps (3.42M) | radiant | 10 | 0 | 0 | 3.03 ± 0.33 | 1.9 | 2.22 | 0.02 |
| P9 C at 2x steps (3.42M) | dire | 10 | 0 | 100 | 4.01 ± 0.73 | 2.0 | 3.26 | 0.00 |
| P10 C in the E12 @4000 dream | radiant | 10 | 0 | 0 | 12.42 ± 0.54 | 27.2 | 5.62 | 0.16 |
| P10 C in the E12 @4000 dream | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| random actions | radiant | 10 | 0 | 0 | 2.82 ± 0.59 | 4.0 | 2.26 | 0.06 |
| random actions | dire | 10 | 0 | 100 | 0.24 ± 0.06 | 0.8 | 0.86 | 0.00 |
| shipped net (mirror) | radiant | 10 | 100 | 0 | 14.56 ± 1.19 | 15.0 | 5.94 | 0.98 |
| shipped net (mirror) | dire | 10 | 0 | 100 | 7.86 ± 1.02 | 7.7 | 4.60 | 0.00 |
