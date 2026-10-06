# Learning in Dreams, Winning in Reality

**A continuous Dyna loop for a ten-hero MOBA.** [Paper (PDF)](docs/moba/paper/paper.pdf) ·
[figures notebook](docs/moba/paper/figures.ipynb) · [experiment log](docs/moba/experiments/README.md) ·
[evaluation protocol](docs/moba/EVALUATION.md)

![A policy trained only inside the learned world model (radiant, cyan) playing a real game against the shipped PufferNet, drawn by the game's own client](docs/moba/paper/media/gameplay_teamfight.gif)

*Everything this team of five knows, it learned inside a 0.66M-parameter model of the game. This is the real game.*

**TL;DR.** We learn a structured world model of PufferLib's MOBA (206 units, ten heroes acting every tick), train a
policy only inside it on 1,400-tick imagined games, and judge it only in the real game against the policy the game
ships with. Run as a continuous Dyna loop, with real data repairing the model where the policy goes and real games
anchoring the policy, it wins **70.2% of real radiant games** (421/600, 95% CI 66.4–73.7, on seeds never used for any
decision), up from 0% for dream training alone. It wins **no dire games**, and neither does the shipped net when it
plays itself. And it plays the game its world model can represent: the macro game of lanes and towers, not the micro
game of stuns, timed heals and coordinated kills.

## What we found

Dream training alone wins nothing. Long imagined episodes with a win reward, and less reward for the experience the
dream overpays, bring the policy to a third of real radiant games. Then every attempt to train further collapses: real
wins fall to near zero within a handful of updates while every number inside the dream looks fine. Only the real game
shows what is happening. Real kills and deaths fall in lockstep with real wins: the policy is learning to stop
fighting. Two remedies turn the collapse into steady improvement. The world model keeps training on the policy's newest
real games, which repairs it exactly where the policy goes. And the policy is tied by a KL penalty to the best real-game
policy so far, an anchor that moves only when a candidate wins on fresh seeds.

![What each ingredient added, measured on the same untouched games](docs/moba/paper/figures/fig9_contribution_ladder.png)

The repair is measurable. On the policy's own games the pre-Dyna world model hallucinated 79% more tower falls than
happened, and was worse than predicting "nothing changes" for hero and tower health. After Dyna it is within 16% on
tower falls, with 2.4× less position error. With the PPO recipe held fixed, the Dyna-trained model alone is worth
**+9.2 points** [3.5, 14.7] of real win rate.

The most interesting result is how the policy wins. Measured mechanic by mechanic, the world model captures towers,
lanes and aggregate damage, but almost no crowd control, the wrong timing for heals and casts, and too few deaths. The
policy mirrors that profile. It takes more towers per game than the shipped net, but its heroes are grouped 9% of the
time (the net: 54%), it stuns before 7% of its kills (26%), its heals almost never reach an ally, and it trades three
deaths per kill. The net marches its five heroes down the middle lane and wins in under half the time. Ours spreads over
the whole map and wins by attrition.

![Where the heroes spend their time: the pre-loop policy, the final policy and the shipped net](docs/moba/paper/figures/fig11_occupancy.png)

| the dream models… | …and the policy |
|---|---|
| tower falls well (1.06× real) | takes ~10 towers per game, more than the net |
| lanes and creep waves | pushes every lane with the waves |
| too few hero deaths (0.42× real) | over-extends: 3.3 deaths per kill |
| almost no stuns (0.17× real) | stuns before 7% of its kills (the net: 26%) |
| heals at the wrong moments | heals itself, not its allies |

## The world-model debugger

Aggregate losses say *that* a world model is wrong, not where, when, about whom or about which mechanic. The debugger
puts the real game and the dream side by side from the same state, on the same actions of all ten heroes, so any
difference is the model's own error. Its timeline scrubs the whole game, any of the 206 units can be decomposed field by
field, an event layer shows casts, deaths and key presses, and a headless mode turns every view into a batch metric.
Every mechanistic claim in the paper started here.

![The real game (left of each pair) and the dream (right), 300 ticks open loop on the policy's own game: before Dyna (top) and after (bottom)](docs/moba/paper/media/debugger_dyna_vs_e38.gif)

![Entity decomposition: the support hero, field by field, real vs dream](docs/moba/paper/media/debugger_entity_tooltip_hero.png)

## Honest scope

The shipped net wins every one of its own radiant games, in less than half the time, so we claim that a policy trained
only in a dream wins most radiant games, not that it is the better player. Nobody wins as dire under this opponent, and
our dire play is barely better than random. The real game provides no gradients, but the loop played about 46,000 real
games to train the model and choose checkpoints, so this is not a sample-efficiency result. The world model uses
hand-coded rules for unit spawns and level-derived statistics. The whole project, failed directions included, cost $382
in attributed cloud spend (plus $214 unattributed on a shared account).

![Tower dive](docs/moba/paper/media/gameplay_dive.gif) ![The final push](docs/moba/paper/media/gameplay_ancient.gif)

## Repository

| | |
|---|---|
| `mobawm/` | the world model (`wm/`), dream PPO and the real-game arena (`rl/`), the FedAvg fleet link |
| `moba/` | deterministic game capture, the data collector, the arena, and the real-client renderer (`moba_render.c`) |
| `tools/` | `moba_viz.py` (the debugger), `pod.py` (RunPod), `fleet.py` (world-model fleets), `build.sh` |
| `scripts/moba/` | evaluation (`seed_eval.py`), the paper's data pipeline (`paper_data.py`), mechanics and behaviour analyses, clip recording; `sweeps/` holds every launcher, including the continuous loop (`dyna_live.py`) |
| `docs/moba/` | the paper, the experiment log (E1–E40), the evaluation protocol, provenance and claims |
| `results/moba/` | every evaluation log and derived table behind the paper |

Every number in the paper regenerates from the logs:

```bash
python scripts/moba/paper_data.py
jupyter nbconvert --execute --inplace docs/moba/paper/figures.ipynb
```

Policies and world models are in `s3://puffertank/moba/`, with md5s in the experiment records. The game is
[PufferLib](https://github.com/PufferAI/PufferLib)'s Ocean MOBA, unmodified.

This is the third report in a line. [*Imitation Learning for Autonomous Driving in CARLA*](https://github.com/JordyKieto/carla-imitation-learning)
asked whether a policy can learn to act; [*Neural Simulator*](https://github.com/JordyKieto/neural-simulators) whether a
small learned model can stand in for a simulator; this one whether an agent taught inside such a model can win in the
real world.
