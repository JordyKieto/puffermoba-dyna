# Experiment log

Every number here is on **holdout tracks (env_idx 112–127)**, which no model
trains on. Corpus `data/corpus`, seed 42, frameskip 4, `max_episode_steps` 2000,
state contract v1 (`74d4ee48e481be34`).

Conventions: **N = 50,000 real transitions** unless stated. "real" means env
transitions consumed; "imagined" means transitions generated inside the world
model. They are always reported as separate axes — the claim worth testing is
real-sample efficiency, not total compute.

---

## E1 — World model, one-step training

`racewm/train_wm.py`, 25 epochs, 817,669 transitions, 352 s, 58k samples/s,
843k params, val NMSE 0.070.

Closed-loop rollout on holdout, geometry gathered from the model's own
predicted pose:

| horizon | persistence | mean-delta | world model | vs persistence |
|---|---|---|---|---|
| @1 | 0.11305 | 0.11270 | 0.00634 | 17.8× |
| @5 | 0.35828 | 0.35328 | 0.01152 | 31.1× |
| @20 | 1.06633 | 0.99406 | 0.11443 | 9.3× |
| @100 | 1.97027 | 1.12956 | **2.24494** | **0.88× (worse)** |

**Finding.** Strong to ~20 steps, diverges by 100 — the compounding-error
signature. Sets the trustworthy planning horizon, and matches where the
literature independently lands (Dreamer imagines 15 steps, PlaNet ~12).

## E2 — World model, multi-step rollout training

`racewm/train_wm_multistep.py`, K=10, discounted per-step loss (γ=0.9), warm
started from E1, 12 epochs, 175 s.

| horizon | persistence | E1 one-step | **E2 multi-step** | vs persistence |
|---|---|---|---|---|
| @1 | 0.11305 | 0.00634 | **0.00479** | 23.6× |
| @5 | 0.35828 | 0.01152 | **0.00335** | 107× |
| @20 | 1.06633 | 0.11443 | **0.01468** | 72.6× |
| @100 | 1.97027 | 2.24494 | **0.11288** | **17.5×** |

**Finding.** Not a trade-off — every horizon improved, including @1. Training
on its own rollouts fixes divergence because the mismatch was distributional,
not capacity.

Two prerequisites, both load-bearing: a torch-native track gather
(`racewm/geometry.py`, cross-checked against numpy to 4.8e-07) and
`project_state` rewritten without in-place writes, which otherwise break
autograd through an unroll.

## E3 — Behaviour cloning

`racewm/train_bc.py`. Val accuracy is offline agreement; the rest is online on
holdout.

| variant | val acc | online return | laps | crash% |
|---|---|---|---|---|
| whisker only | **99.67%** | 5.991 | 1.872 | 74.3 |
| expert only | 84.70% | **11.034** | **3.448** | **39.7** |
| expert only + geometry (privileged) | **95.23%** | not runnable in C | | |
| expert + whisker mix | — | 9.174 | | |
| *reference:* `centerline` (the expert) | — | 10.217 | 3.193 | 45.7 |
| *reference:* `whisker` | — | 5.801 | 1.813 | 75.2 |

**Findings.**
1. **BC beats the expert it cloned** (3.448 vs 3.193 laps, 39.7% vs 45.7%
   crash) on only 84.7% action agreement. The net smooths over the hand-tuned
   controller's chattery failure modes.
2. **An information ceiling exists.** Geometry lifts accuracy 84.70 → 95.23,
   so ~15% of expert actions are not predictable from what the agent sees.
3. **Mixing hurt BC.** Expert-only (11.03) beats expert+whisker (9.17). This
   contradicts the original corpus plan. Whisker data still earns its place in
   the *world-model* slices, where coverage is the point.
4. Whisker-only BC reaches 99.67% and matches whisker exactly online —
   accuracy translates perfectly when there is no information gap.

## E4 — Termination head

`racewm/train_termination.py`, 8 epochs, 16 s.

| metric | value |
|---|---|
| val average precision | **0.9258** |
| base rate (always "no") | 0.0140 |
| positive rate in corpus | 1.51% |

**Why it exists.** The first Dreamer run destroyed the BC policy (0.23 return
vs BC's 8.75, 100% crash). The cause was the corpus, not tuning: `c_step`
zeroes the whiskers, logs, resets, and *then* computes observations, so the
crash state is overwritten before it is ever recorded. The corpus contains
**zero** states with a whisker below 0.05, terminal rows included, while **33%
of imagined states** fell below 1e-3. The actor was optimizing inside a region
the model invents.

A crash state cannot be learned; an impending crash can. Imagined survival went
from a fantasy 1.000 to a realistic 0.03 once the head was wired in.

## E5 — Planning, inside the world model

`racewm/eval_planner.py`, horizon 250, same model for every planner.

| planner | return | sec/decision |
|---|---|---|
| random | 5.894 | 7.5 µs |
| greedy (depth 1) | 7.878 | 21 µs |
| exhaustive (depth 5) | **8.595** | 4558 µs |
| Gumbel MCTS (16 sims) | 7.302 | 233 µs |

**Finding.** Depth-5 exhaustive buys +9% over depth-1 greedy at 214× the cost,
and **Gumbel MCTS loses to greedy** at 11× the cost. With |A|=3 the depth-5
tree is 243 leaves and enumerates in one batched tensor. Gumbel MuZero is the
right modern choice for discrete actions in general; here MCTS is machinery for
avoiding a tree that does not need avoiding.

**Caveat.** This scores planners *inside the model*. See E7 for the real env.

## E6 — Sample budget: real vs imagined

`racewm/sample_budget.py`, online on holdout, with the E4 termination head.

| arm | real | imagined | ratio | return | laps | crash% | lap% |
|---|---|---|---|---|---|---|---|
| A: BC on N | 50,000 | 0 | — | **8.754** | 2.736 | 53.4 | 61.8 |
| B: BC + dream 1:1 | 50,000 | 46,080 | 0.9× | 0.330 | 0.103 | 100.0 | 0.0 |
| C: BC + dream 10:1 | 50,000 | 499,200 | 10× | 6.034 | 1.886 | 74.3 | 49.6 |
| D: BC on 10N | 500,000 | 0 | — | **11.454** | 3.579 | 35.9 | 79.0 |

**Finding — negative.** The termination head lifted C from 1.90 to 6.03, so the
crash signal was real and load-bearing. But imagination still **degrades** the
BC policy it starts from, and more imagination only recovers toward BC without
passing it. On the evidence so far imagination is not buying real-sample
efficiency: 10× real data (D) remains the thing to beat, and BC alone (A) beats
every imagined arm.

Recorded as-is rather than tuned until it looked good. Untested knobs, in the
order worth trying:

1. **Reward proxy.** `progress − crash_probability` has no lap-time term, so
   nothing rewards being *fast*.
2. **Entropy coefficient** (0.01). Plausibly pulling the actor off a good BC
   initialization — untested, and a large lever for actor-critic methods.
3. **REINFORCE variance** over H=15.

## E7 — Online leaderboard, real env

`racewm/leaderboard.py`. Every agent in the actual environment on holdout
tracks, via the batched ctypes binding (`racewm/env.py`, 505k steps/s; live
geometry cross-checks to 0.0 against the dumped holdout tracks).

This exists because a planner cannot be exported to a `.rpo` — it needs the
world model at inference — so it cannot go through `race_eval`. Before this,
planners had *never* been measured outside their own model.

**Censoring.** Episodes still running when the step budget ends are dropped,
which biases every statistic toward whatever crashes fastest. Run with
`--steps` above `max_episode_steps` and check the reported censored fraction. A
first run at `--steps 1500` reported `centerline` at 4.42 return against
`race_eval`'s 10.22 for exactly this reason.

48 parallel envs (16 for planners), 2100 steps, so every episode terminates or
truncates inside the budget.

| agent | eps | cens% | return | laps | crash% | length | cover | idle% | revrs | µs/dec |
|---|---|---|---|---|---|---|---|---|---|---|
| **bc** (expert-only) | 483 | 9.0 | **5.925** | **1.852** | 75.2 | 191.0 | 0.703 | 54.5 | 0.101 | 9 |
| centerline | 612 | 7.3 | 4.817 | 1.505 | 81.4 | 156.0 | 0.648 | 49.8 | 0.120 | 42 |
| bc_mix | 666 | 6.7 | 4.269 | 1.334 | 82.4 | 140.3 | 0.586 | 33.8 | 0.231 | 14 |
| dreamer | 1050 | 4.4 | 2.687 | 0.840 | 95.1 | 90.5 | 0.585 | 42.8 | 0.140 | 16 |
| whisker | 1029 | 4.5 | 2.640 | 0.825 | 94.5 | 91.5 | 0.566 | 4.1 | 0.419 | 10 |
| plan:beam6 | 408 | 3.8 | 2.290 | 0.716 | 97.3 | 79.2 | 0.558 | 23.4 | 0.229 | 1206 |
| plan:gumbel | 487 | 3.2 | 2.034 | 0.636 | 99.4 | 67.1 | 0.532 | 32.5 | 0.256 | 1804 |
| plan:greedy | 1256 | 1.3 | 0.775 | 0.242 | 100.0 | 26.5 | 0.299 | 42.5 | 0.136 | 103 |
| straight | 12453 | 0.3 | 0.221 | 0.069 | 100.0 | 8.1 | 0.129 | 100.0 | 0.000 | 9 |
| random | 14676 | 0.3 | 0.177 | 0.055 | 100.0 | 6.9 | 0.115 | 32.7 | 0.235 | 7 |

**Finding — the headline negative result.** Every planner is **worse than the
`whisker` bot** in the real env, and far worse than BC. Yet inside the world
model (E5) the same planners looked fine, and the model itself is 72x better
than persistence at horizon 20 (E2). Accurate dynamics did not produce a good
planner.

Three candidate causes, in the order worth testing:

1. **The value function is a hand-rolled heuristic**, not a learned critic —
   `GreedyPolicyValue` is `alive x clearance`. Planning deeply against a bad
   terminal value is worse than not planning: `plan:greedy` (depth 1) scores
   0.775, *below random-ish behaviour*, and deeper search recovers only
   partially.
2. **Model exploitation.** A planner actively searches for high-value states,
   which selects for exactly the states where the model is optimistically
   wrong. A one-step accuracy of 0.005 NMSE says nothing about the tail the
   planner deliberately seeks out.
3. **The reward proxy has no speed term** — `progress - crash_probability`.
   Planners idle 23-42% of steps.

**Beam vs exhaustive.** `beam6` matches exhaustive (2.073 vs 2.027 in-model) at
13.5x lower cost (641 vs 8646 µs), so exhaustive is dropped from the default
leaderboard. Pruning appears to act as a mild regulariser against compounding
model error at depth.

**Censoring caveat.** Up to 9% of episodes are still censored at 2100 steps,
because a second episode starting mid-budget may not finish. The bias runs
against the best agents (their episodes are longest), so BC's number is if
anything understated.

---

## Open

- Entropy sweep for the Dreamer actor (E6 knob 2).
- Reward proxy with a lap-time term (E6 knob 1).
- Dreamer proper: alternate world-model and actor-critic updates with fresh
  data. The current pipeline trains the world model once, offline, then freezes
  it — there is no data-collection loop.
- **A learned critic** to replace `GreedyPolicyValue`. This is the single most
  likely explanation for E7 and should be tested before any other planning work.
- Planner distillation, so a planner can be exported and run at C speed.
- Count only episodes that start at t=0, to remove the residual censoring bias.
