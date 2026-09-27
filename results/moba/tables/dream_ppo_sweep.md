<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

One seed per config, 4.5 min each, all six in parallel on one H200. Dream reward only.

| config (lr, minibatches) | updates | agent steps | reward/1k first | reward/1k last 3 | deaths/1k last 3 |
|---|---|---|---|---|---|
| lr3e-4_mb2 | 30 | 2.46M | 4.47 | 13.91 | 0.28 |
| lr3e-3_mb10 | 29 | 2.37M | 4.47 | 13.30 | 1.02 |
| lr1e-3_mb10 | 29 | 2.37M | 4.47 | 11.33 | 0.89 |
| lr3e-4_mb10 | 29 | 2.37M | 4.47 | 10.88 | 0.91 |
| lr1e-3_mb2 | 30 | 2.46M | 4.47 | 10.98 | 0.93 |
| lr3e-3_mb2 | 31 | 2.54M | 4.47 | 9.20 | 1.14 |
