<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

P11: arm C + dream-only reward shaping, 1.70M steps each; 10 games per side vs the shipped net.

| our team (vs shipped net) | side | games | win % | loss % | return/hero ± s.e. | creeps | level | towers |
|---|---|---|---|---|---|---|---|---|
| arm C: game reward | radiant | 10 | 0 | 0 | 17.98 ± 0.58 | 28.4 | 6.66 | 0.32 |
| arm C: game reward | dire | 10 | 0 | 100 | 0.62 ± 0.21 | 0.5 | 0.72 | 0.00 |
| P11 tower 3 to team | radiant | 10 | 0 | 0 | 6.66 ± 0.84 | 8.6 | 3.90 | 0.10 |
| P11 tower 3 to team | dire | 10 | 0 | 100 | 1.47 ± 0.36 | 2.0 | 1.86 | 0.00 |
| P11 win +-10 | radiant | 10 | 0 | 0 | 2.99 ± 0.36 | 6.8 | 2.66 | 0.02 |
| P11 win +-10 | dire | 10 | 0 | 100 | 3.43 ± 0.58 | 3.2 | 2.94 | 0.02 |
| P11 tower 3 team + win +-10 | radiant | 10 | 0 | 0 | 5.06 ± 0.47 | 5.3 | 3.26 | 0.08 |
| P11 tower 3 team + win +-10 | dire | 10 | 0 | 100 | 0.97 ± 0.37 | 1.1 | 1.26 | 0.00 |
| P11 tower 10 team + win +-30 | radiant | 10 | 0 | 0 | 5.72 ± 0.91 | 10.3 | 3.58 | 0.08 |
| P11 tower 10 team + win +-30 | dire | 10 | 0 | 100 | 0.01 ± 0.01 | 0.0 | 0.04 | 0.00 |
| shipped net (mirror) | radiant | 10 | 100 | 0 | 14.56 ± 1.19 | 15.0 | 5.94 | 0.98 |
| shipped net (mirror) | dire | 10 | 0 | 100 | 7.86 ± 1.02 | 7.7 | 4.60 | 0.00 |
