<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Real game, seeds 50000000+, 10 games per side, max 6,000 ticks (mobawm/rl/arena.py, the P2/P3 harness).

| our team (vs shipped net) | games | win % | return/hero | deaths | level | creeps | towers |
|---|---|---|---|---|---|---|---|
| P3 self-play (E5 dream, 4.5 min) | 20 | 0 | 0.01 | 0.00 | 0.02 | 0.02 | 0.00 |
| P3 vs net (E5 dream, 4.5 min) | 20 | 0 | 0.02 | 0.01 | 0.08 | 0.03 | 0.00 |
| P4 mix, expert starts (E11 @1500, 4.5 min) | 20 | 0 | 3.38 | 0.33 | 1.96 | 7.15 | 0.04 |
| P4b mix, expert + random starts (E11 @1500, 4.5 min) | 20 | 0 | 6.40 | 2.05 | 2.83 | 11.04 | 0.11 |
| random actions | 20 | 0 | 1.53 | 0.56 | 1.56 | 2.38 | 0.03 |
| shipped net | 20 | 50 | 11.21 | 7.12 | 5.27 | 11.35 | 0.49 |
