<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

E17 update-matched A/B: E11 @1500 fine-tuned exactly 60 updates each (batch 384, same data), held-out games 900-909.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 (no fine-tune) | 10 | 87 | 87% | 38% | 53% | 94 | 75 | 165 | 53 | 32% | 71% |
| real attack graph only | 10 | 87 | 15% | 50% | 81% | 5 | 75 | 85 | 49 | 58% | 65% |
| real + consistency | 10 | 87 | 36% | 63% | 86% | 7 | 75 | 60 | 37 | 62% | 49% |
| real + event sampling | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 109 | 52 | 48% | 69% |
| real + consistency + events (combined) | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 96 | 50 | 52% | 67% |
