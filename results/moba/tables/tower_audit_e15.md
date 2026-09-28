<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

E15 arms: E11 @1500 fine-tuned 5 minutes each (~60-100 updates, batch 384); held-out games 900-909.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 (no fine-tune) | 10 | 87 | 87% | 38% | 53% | 94 | 75 | 165 | 53 | 32% | 71% |
| control: fine-tune only | 10 | 87 | 52% | 43% | 72% | 29 | 75 | 272 | 61 | 22% | 81% |
| real attack graph | 10 | 87 | 24% | 60% | 94% | 2 | 75 | 92 | 55 | 60% | 73% |
| derived attack graph | 10 | 87 | 54% | 51% | 65% | 33 | 75 | 92 | 50 | 54% | 67% |
| consistency loss | 10 | 87 | 48% | 61% | 84% | 11 | 75 | 83 | 45 | 54% | 60% |
| tower-event sampling | 10 | 87 | 0% | 0% | 0% | 0 | 75 | 118 | 53 | 45% | 71% |
| tower health x10 | 10 | 87 | 39% | 42% | 65% | 28 | 75 | 627 | 43 | 7% | 57% |
