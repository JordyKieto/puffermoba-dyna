<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Seeds 50000000-50000059 per side, max 6,000 ticks. Branch chosen (net's action / keep plan / new plan): [1031, 38883, 5086] = net's action 2%.

| our team (vs shipped net) | games | win % | loss % | timeout % | return/hero | deaths | level | creeps |
|---|---|---|---|---|---|---|---|---|
| WM planner (60 fps budget) | 120 | 0 | 50 | 50 | 1.84 | 1.40 | 1.84 | 2.0 |
| planner as radiant | 60 | 0 | 0 | 100 | 3.36 | 2.13 | 2.90 | 3.5 |
| planner as dire | 60 | 0 | 100 | 0 | 0.33 | 0.67 | 0.79 | 0.5 |
| shipped net (mirror) | 120 | 50 | 50 | 0 | 10.46 | 6.87 | 5.04 | 10.8 |
| untrained (random) | 120 | 0 | 50 | 50 | 1.47 | 0.57 | 1.49 | 1.7 |
