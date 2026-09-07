# Experiment log

Long-form reference. The narrative is in [README.md](../README.md), the
transferable method in [METHODOLOGY.md](METHODOLOGY.md), and run identifiers /
checkpoints / evaluation costs in [ARTIFACTS.md](ARTIFACTS.md).

Every number is on **holdout tracks (env_idx 112-127)**, which no model trains
on. Corpus `data/corpus`, seed 42, frameskip 4, `max_episode_steps` 2000, state
contract v1 (`74d4ee48e481be34`).

Conventions: **N = 50,000 real transitions** unless stated. "real" means env
transitions consumed; "imagined" means transitions generated inside the world
model. They are always reported as separate axes -- the claim worth testing is
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

## E8 — Fixing the planner's value function (and what it revealed)

`racewm/train_critic.py`: TD(0) on **real** transitions, replacing the
hand-rolled `alive x clearance` heuristic.

| critic input | V(healthy) | V(about to crash) | gap |
|---|---|---|---|
| state only | 0.516 | 0.351 | 0.165 |
| **state + track geometry** | **0.727** | **0.286** | **0.441** |

Value depends on the track ahead, so state alone was never enough.

The planning reward was also wrong: it subtracted a hand-set `5.0 * p_term`
against a max step reward of `0.2`, a 25:1 imbalance that paralysed planners
into pure crash-avoidance. Termination is now a survival weight on downstream
return, with the critic bootstrapping the tail.

**Result — the opposite of the hypothesis.** Online, the learned critic made
planners *worse*:

| planner | heuristic critic | learned critic | search effort |
|---|---|---|---|
| plan:beam6 | 2.290 | **0.411** | most |
| plan:gumbel | 2.034 | 0.897 | medium |
| plan:greedy | 0.775 | 0.753 | least |

More search now produces worse real outcomes. That inversion is **model
exploitation**: a better critic makes the planner better at finding states
where the *dynamics* are confidently wrong. Planning is parked here rather than
iterated further — it is also 100x slower per decision than a policy.

## E9 — Dreamer: freezing, critic anchoring, and an actor constraint

`racewm/train_joint.py`. All arms: 200,000 real transitions, 400 updates,
H=15, actor warm-started from BC, online on holdout.

**Coupling mode.** "Unfrozen" means the world model keeps training on real
data; actor gradients still never reach it (that is `--mode full`, included to
be measured, not adopted).

| mode | return | laps | crash% |
|---|---|---|---|
| frozen | 5.493 | 1.717 | 79.9 |
| **joint** | **7.568** | **2.365** | **65.5** |

Unfreezing is worth **+38%**.

**Entropy is not the lever.** Sweep at joint mode: 0.0 → 7.451, 0.003 → 6.519,
0.01 → 7.547, 0.03 → 6.444. Spread with no trend; 0.01 was already best.

**Critic anchoring.** Dreamer's critic is fitted purely on imagined
lambda-returns, so it inherits the model's optimism — the same pathology E8
found in planners. Adding `anchor_coef * |V_imagined - V_real|` against the
real-data critic:

| anchor_coef | 0.0 | 0.5 | 1.0 | **2.0** | 5.0 |
|---|---|---|---|---|---|
| return | 7.547 | 8.347 | 8.576 | **8.973** | 8.414 |

Saturates at 2.0 and regresses by 5.0. The ceiling is knowable: as the
coefficient grows the critic simply *becomes* the real-data critic. **~9.0 is
this lever's limit, and BC is 11.0 — anchoring alone cannot close the gap.**

**Actor constraint.** Anchoring constrains only the critic; the actor is still
free to walk into states the model mispredicts, which is the standard
offline-RL failure. Adding `bc_coef * KL(pi || pi_BC)` at anchor 2.0:

| bc_coef | 0.0 | **0.3** | 1.0 | 3.0 | 10.0 | 30.0 | 100.0 |
|---|---|---|---|---|---|---|---|
| return | 8.973 | **11.228** | 10.534 | 10.563 | 10.498 | 10.451 | 10.477 |

Peaks at 0.3 and plateaus near 10.5 as the policy collapses back toward BC,
which is the expected large-coefficient limit and confirms the sweep brackets
the question.

**Is the peak real?** Four training seeds of the best config, each evaluated on
3,200 holdout episodes against BC on the identical setting:

| run | return | survive% |
|---|---|---|
| seed 0 | 11.282 | 63.3 |
| seed 1 | 11.278 | 65.9 |
| seed 2 | 11.383 | 66.1 |
| seed 3 | 11.004 | 61.6 |
| **mean** | **11.237 ± 0.16** | **64.2** |
| BC | 11.016 | 60.3 |

**Verdict: a small real effect, not a win.** Return is +2.0%, roughly 1.4
sigma, and seed 3 ties BC. Survival is the sturdier signal — all four seeds beat
BC by ~4 points. Imagination now at least does not *harm* a good policy, which
is where it started (0.33 return against BC's 8.75), but it is not yet buying
the real-sample efficiency that would justify it.

The honest summary of the whole imagination line: three separate fixes
(termination head, critic anchoring, actor constraint) were each necessary to
get from catastrophic to parity. Each was found by an online measurement
disagreeing with an in-model one.

## E10 — Planning with the actor-critic as prior and value

Two omissions from E5/E8, found by asking whether the planner had ever been
given the Dreamer actor-critic:

1. **`prior_logits` was defined but never passed.** Every planner searched
   blind. `plan_exhaustive` and `plan_beam` did not even accept a prior.
   AlphaZero and MuZero both *guide* search with a policy prior; unguided
   search re-derives the policy from scratch at every step, through a model it
   can exploit.
2. **Planners used `critic_geom`** — a critic fitted by TD(0) on the
   *behaviour* data, valuing states under the scripted bots rather than under
   the policy the search would follow afterwards.

Both fixed: `ActorCriticHead` exposes a trained actor-critic as both the search
prior and the leaf value.

All agents in ONE harness (mixing `race_eval` and leaderboard numbers is
invalid — the leaderboard censors long episodes, which is why it reports BC at
5.925 against `race_eval`'s 11.016 for the identical policy):

| agent | return | laps | crash% | µs/decision |
|---|---|---|---|---|
| actor (Dreamer policy) | **6.346** | 1.983 | 71.2 | 10 |
| plan:greedy + prior | 6.296 | 1.968 | 71.8 | 127 |
| bc | 5.925 | 1.852 | 75.2 | 10 |
| centerline | 4.817 | 1.505 | 81.4 | 16 |
| whisker | 2.640 | 0.825 | 94.5 | 3 |
| plan:beam6 + prior | 0.432 | 0.135 | 100.0 | 877 |

Prior strength sweep, `plan:greedy`: 0.792 (no prior) → 6.296 / 6.218 / 6.347
at coef 1 / 3 / 10. An 8x jump, then flat.

**Finding.** Depth-1 search with a prior is not merely close to the actor, it
*is* the actor: identical laps, crash rate, idle fraction and reversal rate, at
12.7x the cost. The scales explain why — the critic's V spread across states is
0.5 while the prior's spread across actions is 7.4, so the prior outvotes the
search 15:1 and the model contributes nothing to the decision.

Depth-5 beam remains catastrophic (0.432) even with the prior, because the
prior is applied only at the root while the expansion stays unguided. Five
levels of model exploitation swamp one root correction.

**Conclusion: planning through this world model does not beat following the
policy.** With a good prior it degenerates to the policy; without one it is
actively harmful. The untested variant is a PUCT-style prior at every node
rather than root-only — which is what AlphaZero actually does, and the only
version that could plausibly change this verdict.

**Bug found here.** The actor-critic was rebound to the variable `head`, which
already held the termination head, so the *second* planner in each run received
the actor-critic as its termination model and got 77-dim world-model inputs.
The first planner always worked, which made it look like a beam-specific
failure.

## E11 — Does a dynamics-pretrained backbone help imitation learning?

`racewm/transfer.py`. Pretrain a world model, initialise a BC policy from its
encoder, compare against BC from scratch.

### Setup

A **separable encoder**, because the world model consumes 77 inputs and a
policy consumes 30:

    encoder(obs)                -> z (256)   shared, transferable, 73k params
    dynamics([z, action, geom]) -> delta     world-model only
    policy(z)                   -> 3 logits  BC only

Also the shape a vision encoder would take into a policy, which is why it was
built this way rather than transferring hidden layers by index.

**How the transfer is done, exactly.** Load the 2-layer encoder
(30→256→256, SiLU) from the dynamics checkpoint; attach a **freshly
initialised** head (256→256→SiLU→256→3); train **all** parameters with
cross-entropy. The `frozen` variant instead freezes the encoder and trains only
the head. The pretrainer's observation normalization travels with the encoder,
so it sees inputs from the distribution it was fitted on.

**A "label" is one (state, expert_action) pair** — the cross-entropy target.
In simulation these are free; in the sim-to-real setting they are expert
demonstrations and they are the scarce resource. Dynamics data is *not*
label-free either: the model is action-conditioned and consumes (s, a, s').
The distinction is between *any* action data (cheap, any policy) and *expert*
action labels (expensive).

### Sample accounting

| | pretraining | BC finetuning |
|---|---|---|
| data | 60,000 windows of length K+1 | `budget` labelled pairs |
| gradient steps | 702 (6 epochs, batch 512) | **2,000** |
| transitions consumed | **359,424** | 2,000 × min(1024, budget) |
| actions used | any policy's | expert only |

**Controlled:** architecture, gradient steps (2,000 every arm), batch size,
optimizer, learning rate and schedule, evaluation protocol (1,600 holdout
episodes), 3 seeds.

**Not controlled:** total data exposure. The pretrained arm has seen 359,424
extra transitions — at a 2,000-label budget that is a 180x head start in raw
samples. Pretraining also costs ~60s of wall clock the scratch arm does not pay.

### Result

Three seeds, online return on holdout. `K1_all` pretrains on all four corpus
slices; `K1_noexpert` pretrains on whisker/noisy/random only, so no expert
action ever reaches the encoder.

| labels | scratch | K1_all | K1_noexpert |
|---|---|---|---|
| 2,000 | 8.015 ± 0.208 | **8.703 ± 0.427** (+8.6%, 2.0σ) | 8.554 ± 0.441 (+6.7%, 1.6σ) |
| 5,000 | 8.983 ± 0.126 | 9.168 ± 0.158 (+2.1%, 1.3σ) | 9.068 ± 0.443 (+1.0%, 0.3σ) |
| 10,000 | 9.990 ± 0.095 | 10.175 ± 0.130 (+1.9%, 1.6σ) | 10.081 ± 0.149 (+0.9%, 0.7σ) |
| 20,000 | 11.146 ± 0.255 | 11.207 ± 0.166 (+0.5%) | 11.269 ± 0.177 (+1.1%) |
| 50,000 | 11.278 ± 0.217 | 11.387 ± 0.160 (+1.0%) | 11.559 ± 0.216 (+2.5%, 1.3σ) |

Validation accuracy is indistinguishable across arms at every budget
(0.7885–0.7894 at 2k; 0.8480–0.8503 at 50k).

**Finding: dynamics pretraining buys at most a small advantage at very low
label counts, and nothing beyond.** +8.6% at 2σ with 2,000 labels; everything
else is under 1.6σ. Excluding the expert slice from pretraining barely changes
it, so the effect is not an expert-action leak.

### RETRACTION — and the confound that caused it

A first version of this experiment reported **+23% at 2k and +20% at 10k, both
above 7σ**, and concluded that dynamics pretraining transfers strongly. **That
was wrong.**

The training loop ran a fixed 30 *epochs* with a capped batch size, so the
number of gradient steps scaled with the dataset: ~120 updates at 2k–20k
labels against 720 at 100k. The scratch arm was simply **undertrained** at
small budgets, and the pretrained arm — starting from a usable representation —
needed fewer updates to reach the same place. The tell was in the curve and I
nearly missed it: scratch was flat at ~8.8 from 5k to 20k labels, then jumped
to 11.4 at 30k, and 30k beat 50k. Data curves do not have step changes.

With a fixed 2,000-step budget for every arm, scratch at 10k labels goes from
8.805 to 9.990 and most of the gap disappears.

**Any claim about sample efficiency has to hold optimization fixed**, or it is
a claim about optimization.

Offline accuracy, once again, failed to separate the arms — but here it was
telling the truth and the online number was the misleading one.

## E12 — Scaled multi-task pretraining: what actually transfers

E11 handicapped pretraining (702 gradient steps against BC's 2,000, on 60k of
817k transitions, one objective). Unlimited cheap dynamics data is the premise
of the method, so here it gets its best shot: **12,000 steps x 1024 batch =
12.3M samples** over all 830k transitions, four heads, five ablations. The
FINETUNING side stays fixed — 2,000 steps, identical architecture, same seeds
and eval — because blurring that is what produced E11's spurious 7σ.

Heads, all reading `z = encoder(obs)`:

| head | signal |
|---|---|
| `fwd` | `[z, a, geom] → Δs` |
| `inv` | `[z_t, z_{t+1}] → a_t` |
| `geom` | `z → geom(44)` — privileged distillation |
| `term` | `[z, a] → done` |

### Ablation, 3 seeds

| labels | scratch | all | fwd_only | **geom_only** | inv_only | no_geom |
|---|---|---|---|---|---|---|
| 500 | 3.18 | 4.33 (+36%) | 2.49 (−22%) | **5.48 (+72%, 3.1σ)** | 2.34 (−27%) | 3.25 (+2%) |
| 1,000 | 4.49 | 5.25 (+17%) | 4.31 (−4%) | 5.29 (+18%) | 3.68 (−18%) | 5.08 (+13%) |
| 2,000 | 8.02 | 8.85 (+10%) | 8.45 (+5%) | 8.68 (+8%, 2.5σ) | 8.15 (+2%) | 8.72 (+9%) |
| 5,000 | 8.98 | 9.06 | 8.86 | 8.64 | 8.69 | 9.00 |
| 10,000 | 9.99 | **9.43 (−5.6%)** | 9.64 | 9.65 | 9.94 | 9.62 |

### Confirmation, 8 seeds

| labels | scratch | no_geom | geom_only |
|---|---|---|---|
| 500 | 3.452 ± 0.740 | 3.732 (+8.1%, **0.4σ**) | **5.395 (+56.3%, 2.4σ)** |
| 2,000 | 7.727 ± 0.798 | 8.278 (+7.1%, **0.7σ**) | 8.646 (+11.9%, 1.2σ) |

### Findings

**Dynamics pretraining does not help.** `no_geom` — forward + inverse +
termination, 12.3M samples — is statistically indistinguishable from random
init (0.4σ, 0.7σ). `fwd_only` and `inv_only` are *worse* than scratch at 500
labels.

**Privileged-information distillation is the entire effect.** Teaching the
encoder to predict track geometry it cannot observe is worth +56.3% at 2.4σ.
The encoder recovers ~84% of geometry variance from 30 floats (`geom` loss
0.157 normalized).

**Inverse dynamics is degenerate here.** Its loss goes to 0.0000: the heading
change between consecutive states reveals the action exactly, so the task is
solvable without learning anything transferable. Worth checking before
assuming inverse dynamics is a useful auxiliary — in a harder environment it
would not collapse like this.

**Pretraining becomes a liability once labels are sufficient.** At 10,000
labels every pretrained arm loses to scratch, some at −5.8σ. A pretrained
representation is a constraint: useful when data is scarce, harmful when it
is not.

**It does not substitute for labels.** Pretraining consumed 830k transitions;
had those carried expert labels, BC on them scores ~11.3 against geom_only's
5.4. More bluntly: **scratch with 2,000 labels (7.727) beats geom_only with
500 labels (5.395)** — 1,500 extra expert labels are worth more than 830,000
unlabelled transitions.

### Why this still matters for RGB

Those economics are specific to this setting, where a scripted bot produces
labels at ~1M/s. For real RGB the ratio inverts: unlabelled video is abundant
and expert demonstrations are expensive. The result to carry forward is that
**the mechanism works and it is not the dynamics** — what transfers is learning
to infer privileged semantic state from an observation that does not contain
it, which is the sim-to-real problem restated.

## E13 — World-model objectives as auxiliary losses during BC

Different mechanism from E11/E12. There the representation is frozen before the
policy starts, which is why it became a liability once labels were plentiful.
Here the auxiliary loss shapes the encoder *while* the policy trains, so the
task loss can outvote it as data grows.

Heads, all on `z = encoder(obs)`: `next_state`, `next_latent`, `multi_state`
(K=5), `multi_latent` (K=5), `geom`, and `state_latent` (both 1-step heads).
The auxiliary loss uses the **same batch** as the policy loss, so no arm sees
more data than another — this isolates the objective from the data.

Latent targets carry a **stop-gradient**. Predicting `encoder(s_{t+1})` with a
live encoder has a trivial solution (constant z), which is collapse rather than
learning; BYOL and JEPA avoid it the same way. Latent variance is reported so a
collapse would show up as a number.

Four seeds, 2,000 gradient steps every arm, lambda = 1.0:

| labels | none | next_state | next_latent | multi_state | multi_latent | geom | state_latent |
|---|---|---|---|---|---|---|---|
| 500 | 3.34±1.33 | +13.6% (0.4σ) | +4.5% | +22.8% (0.6σ) | +3.8% | +7.8% | +12.6% |
| 2,000 | 7.55±0.50 | −1.6% | −0.5% | −4.7% | −0.8% | +2.3% | −1.4% |
| 10,000 | 10.31±0.18 | −3.0% (−1.1σ) | +0.4% | −2.0% | +0.3% | **+2.4% (+1.5σ)** | −2.5% |

**Finding: no auxiliary objective helps.** The largest effect is 0.6σ, and at
2,000 and 10,000 labels most arms are flat or slightly negative. Only `geom` is
consistently non-negative, and only at 1.5σ.

**The auxiliary losses are not being ignored.** Latent variance rises from 0.32
(no aux) to 0.75 for `next_state` and `state_latent` — more than double. The
objectives demonstrably reshape the representation; the reshaping just does not
help the policy.

**Why this differs from E12.** `geom` as *pretraining* was worth +56% (5.395 vs
3.452 at 500 labels). The identical head as an *auxiliary* is worth nothing:

| | geometry supervision seen |
|---|---|
| pretraining (E12) | 830,000 transitions |
| auxiliary (E13) | 500 states — the label budget |

Sharing the policy's batch is what isolates the objective, and it is also what
starves it. The E12 gain came from the **extra unlabelled data the objective
allowed us to consume**, not from the objective regularising the policy.

That distinction is the useful one: a world-model objective is a way to *spend
unlabelled data*, not a better inductive bias for BC. The untested variant that
follows from it is semi-supervised — policy loss on the small labelled batch,
auxiliary loss on a large separate unlabelled batch, jointly.

---

## Open

- **A data-collection loop.** Every result above is offline: the corpus is
  fixed and the policy never gathers its own data. Dreamer's sample-efficiency
  claim is about *online* interaction, so the central claim has not actually
  been tested. This is the biggest remaining gap.
- Reward proxy with a lap-time term (nothing currently rewards being fast).
- Ensemble pessimism for planning (started, then parked).
- PUCT-style prior at every search node, the one untested variant that could
  change E10's verdict.
- Semi-supervised variant of E13: auxiliary loss on a large separate
  unlabelled batch rather than the policy's batch. E13 shows the objective is
  worthless when starved of data and E12 shows it works when fed; this is the
  test that separates those.
- **Do not compare numbers across harnesses.** `race_eval` and the leaderboard
  censor differently; the same BC policy reads 11.016 and 5.925. Always put
  competing agents in one run.
- **A learned critic** to replace `GreedyPolicyValue`. This is the single most
  likely explanation for E7 and should be tested before any other planning work.
- Planner distillation, so a planner can be exported and run at C speed.
- Count only episodes that start at t=0, to remove the residual censoring bias.
