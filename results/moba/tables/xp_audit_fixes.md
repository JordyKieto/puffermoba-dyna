<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Same audit (real rows as in xp_audit_e5.md). Fine-tunes: local CPU, batch 32, ~30 updates, single seed: smoke tests, not results.

| world model | our team's actions | measure | deaths/1k | XP/1k | tower kills/1k |
|---|---|---|---|---|---|
| E5 @20,500, clamp off (eval only) | ppo | dream, teacher-forced | -0.00 | 18439 | 0.000 |
| E5 @20,500, clamp off (eval only) | ppo | dream, open loop 128 | 0.07 | 6288 | 0.065 |
| E5 @20,500, clamp off (eval only) | net | dream, teacher-forced | 0.14 | 741 | 1.639 |
| E5 @20,500, clamp off (eval only) | net | dream, open loop 128 | 1.53 | 2155 | 0.163 |
| E5 @20,500, clamp off + signed XP reward | ppo | dream, teacher-forced | -0.00 | 17625 | 0.000 |
| E5 @20,500, clamp off + signed XP reward | ppo | dream, open loop 128 | 0.23 | 1349 | 0.163 |
| E5 @20,500, clamp off + signed XP reward | net | dream, teacher-forced | 0.14 | -3234 | 1.639 |
| E5 @20,500, clamp off + signed XP reward | net | dream, open loop 128 | 1.53 | -4739 | 0.163 |
| +~30 upd, XP loss x1 (control) | ppo | dream, teacher-forced | -0.00 | 11379 | 0.000 |
| +~30 upd, XP loss x1 (control) | ppo | dream, open loop 128 | 0.10 | 5228 | 0.000 |
| +~30 upd, XP loss x1 (control) | net | dream, teacher-forced | 0.11 | 783 | 1.361 |
| +~30 upd, XP loss x1 (control) | net | dream, open loop 128 | 1.76 | 758 | 0.163 |
| +~30 upd, XP loss x100 | ppo | dream, teacher-forced | -0.00 | 3774 | 0.000 |
| +~30 upd, XP loss x100 | ppo | dream, open loop 128 | 0.29 | 8267 | 0.000 |
| +~30 upd, XP loss x100 | net | dream, teacher-forced | 0.08 | 722 | 1.111 |
| +~30 upd, XP loss x100 | net | dream, open loop 128 | 1.40 | 1527 | 0.130 |
| +~30 upd, no XP clamp | ppo | dream, teacher-forced | -0.00 | 30353 | 0.000 |
| +~30 upd, no XP clamp | ppo | dream, open loop 128 | 0.07 | 8994 | 0.000 |
| +~30 upd, no XP clamp | net | dream, teacher-forced | 0.31 | 2650 | 2.889 |
| +~30 upd, no XP clamp | net | dream, open loop 128 | 2.51 | 3641 | 0.228 |
