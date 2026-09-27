<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Held-out loss@h (games 900-999, same 32 windows; h4096 from the few long games only).

| model | source | h1 | h4 | h16 | h64 | h256 | h1024 | h4096 |
|---|---|---|---|---|---|---|---|---|
| E4 final (teacher-forced) | `results/moba/93m7sktq/` step 0 | 0.78 | 3.18 | 15.50 | 80.45 | 213.49 | 323.44 | 428.79 |
| E5 @20,500 (rollout, kept) | `results/moba/93m7sktq/` step 20500 | 1.70 | 5.14 | 7.70 | 18.99 | 24.64 | 34.84 | 36.84 |
| E5 continued +1,500 (weights lost) | `results/moba/soeosr9h/` step 1500 | 1.51 | 4.65 | 7.18 | 16.75 | 26.43 | 37.88 | 52.32 |
| E6b @3,500 (counterfactual + effect) | `results/moba/d9xnnj0a/` step 3500 | 0.90 | 3.55 | 8.12 | 21.25 | 34.33 | 48.94 | 59.78 |
