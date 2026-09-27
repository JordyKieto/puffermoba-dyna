<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

10 games per side x 2 sides, seeds 50000000+, max 6,000 ticks; same seeds for every team.

| our team | opponent | games | win % | loss % | return/hero | deaths/hero | level |
|---|---|---|---|---|---|---|---|
| dream PPO | vs net | 20 | 0 | 50 | 1.20 | 0.19 | 1.26 |
| dream PPO | vs script | 20 | 0 | 100 | 1.79 | 0.37 | 1.42 |
| untrained (random) | vs net | 20 | 0 | 50 | 1.82 | 0.54 | 1.54 |
| untrained (random) | vs script | 20 | 0 | 100 | 6.38 | 1.02 | 3.61 |
| shipped net (reference) | vs net | 20 | 50 | 50 | 11.21 | 7.12 | 5.27 |
| shipped net (reference) | vs script | 20 | 45 | 55 | 18.89 | 3.73 | 7.10 |
