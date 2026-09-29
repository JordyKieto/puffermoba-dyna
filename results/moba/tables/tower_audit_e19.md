<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Full-run configuration: E12 @4000, full data (expert+random+dyna2), batch 768, lr 1e-3; held-out games 900-909. RULE rows: author-approved smoke test, evaluation-time only (data.TOWER_DEATH_RULE); health <= 0 gave 0 predicted falls.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E12 @4000 (start) | 10 | 87 | 77% | 60% | 70% | 33 | 75 | 119 | 51 | 43% | 68% |
| E19 control: +250 updates, no attack graph | 10 | 87 | 55% | 41% | 60% | 47 | 75 | 128 | 48 | 38% | 64% |
| E19 real attack graph: +250 updates | 10 | 87 | 70% | 60% | 80% | 20 | 75 | 111 | 52 | 47% | 69% |
| E18 real attack graph: +500 updates | 10 | 87 | 48% | 48% | 70% | 26 | 75 | 186 | 48 | 26% | 64% |
| E18 real attack graph: +1000 updates | 10 | 87 | 15% | 50% | 96% | 1 | 75 | 138 | 55 | 40% | 73% |
| E18 real attack graph: +1500 updates | 10 | 87 | 85% | 71% | 83% | 18 | 75 | 506 | 58 | 11% | 77% |
| E12 @4000 + RULE death at <=3% health | 10 | 87 | 37% | 33% | 70% | 29 | 75 | 21 | 9 | 43% | 12% |
| E12 @4000 + RULE death at <=8% health | 10 | 87 | 78% | 57% | 80% | 24 | 75 | 51 | 25 | 49% | 33% |
