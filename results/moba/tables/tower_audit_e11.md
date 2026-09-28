<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->

Held-out games 900-919. Open loop: 128/512-tick windows from real states, real actions; a hit = the same tower falls within +-32 ticks. One step: alarms = runs of consecutive predicted falls.

| world model | games | real falls | one step recall | one step precision (exact) | one step precision (incl. early) | false alarms | open-loop real | open-loop predicted | hits | open-loop precision | open-loop recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 @1500 | 20 | 177 | 88% | 38% | 51% | 197 | 153 | 302 | 107 | 35% | 70% |
| E11 @1500 (open loop 512) | 20 |  |  |  |  |  | 134 | 368 | 48 | 13% | 36% |
