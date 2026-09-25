# World-model methodology

The order below is the point. Each step exists because skipping it makes a
later measurement meaningless, and every rule here was paid for by a failure in
this repository rather than adopted on principle.

Written to transfer: the target is a multimodal model that steps in symbolic
state *or* RGB. Where a rule has an RGB analogue it is named.

---

## 0. Audit the environment before modelling it

A learning curve cannot show you that your environment is broken. Read the
dynamics and ask two questions:

- **What does the transition function actually read?** Everything it reads must
  be in the model's input. Race's `compute_whiskers` intersects track edge
  polylines and `update_nearest_point` walks a centerline — none of it was in
  the observation.
- **Where does randomness come from?** Race seeded the *process-global* libc
  RNG, so every parallel env generated the identical track and episode starts
  depended on reset ordering. Reproduced in `tests/upstream_probe.c`.

Also worth cataloguing: config that does nothing, metrics that divide by an
unassigned zero, documented features that are dead code. All three were present.

## 1. Pin a state contract, hashed, checked from both sides

One definition of the state layout, hashed, asserted against the source of
truth in the other language, and stamped into every dataset shard and every
checkpoint. Loading a checkpoint under a changed hash should be a hard error,
not a warning: the columns no longer mean the same thing and nothing downstream
will look wrong.

## 2. Gate it — with a positive control

Two gates before any learning.

**Determinism.** Thread count must not change results; reruns must be
bit-identical; different seeds must diverge; N distinct configurations must
produce N distinct trajectories. Hash whole trajectories, not summary
statistics.

**Markov sufficiency.** Train the same model, same capacity, same budget, to
predict `s_{t+1}` from `(s_t, a_t)` and then from `(s_{t-H..t}, a_t)`. If
history helps, something the dynamics depend on is missing.

> **Always include a positive control.** Run the identical test on a state you
> *know* is insufficient. If the test cannot detect that, its verdict on your
> real state is worthless. Ours: the upstream 3-float observation, which showed
> a +22.3% history gain against +4.0% for the full contract.

## 3. Split by whatever determines the dynamics

Not by row. Consecutive rows are near-duplicates, so a random row split puts
step *t* in train and *t+1* in validation and every model looks excellent.

Split by the thing that changes the physics — here the track. Write the splits
as **separate files over disjoint ranges**, not a runtime filter: a filter is
one refactor from leaking, and a leaked holdout is invisible in every
downstream metric because the numbers merely improve. Re-derive the split from
the data and assert disjointness before training.

*RGB analogue:* split by scene, lighting condition, or capture session.

## 4. Baselines that can lose

Before any world model:

- **Persistence** (`s_{t+1} = s_t`). Often far stronger than expected at short
  horizons, which is exactly why one-step error is a weak metric.
- **Mean-delta** — how much of your model's skill is knowing the average step.
- **Scripted policies**, ideally one privileged and one restricted to what the
  agent can actually sense. The gap between them measures the information
  ceiling, which is not a training failure.
- **Behaviour cloning.** It beat the expert it cloned here. Any method that
  cannot beat BC is not earning its complexity.

## 5. Train on rollouts, not single steps

A one-step model is only ever asked to predict from a *real* state, then at
evaluation must predict from its own output — a distribution it never saw. Ours
was 9.3× better than persistence at horizon 20 and 0.88× (worse) at 100.

Unroll K steps during training, take the loss at every step, and gather any
state-dependent inputs from the model's *own* predicted state. Discount the
per-step losses so late steps do not swamp near-term accuracy.

This fixed @100 **and improved @1**. It was never a trade-off.

Two implementation details that are load-bearing:

- **Project predictions back onto the state manifold each step.** Angles
  encoded as `(sin, cos)` must be renormalized or rollouts leave the valid set
  within a few steps and the model spends capacity undoing its own invalid
  encodings.
- **Write the projection without in-place tensor writes.** Clone-and-assign
  works for a single forward pass and breaks autograd through an unroll.

## 6. Terminal states may not be observable

Check whether your corpus contains the states you think it does.

Race zeroes the whiskers on a crash, logs, resets, and *then* computes
observations — so the crash frame is overwritten before it is ever recorded.
The corpus contained **zero** states with a whisker below 0.05, while **33% of
imagined states** fell below 1e-3. The model had never seen a crash and was
freely inventing them; an actor trained in imagination optimized inside that
invention and scored 0.33 against BC's 8.75.

A crash *state* cannot be learned from data that lacks it. An *impending* crash
can be predicted from the state and action that cause it. This is what Dreamer's
`continue` head is for, and it is not optional.

## 7. Judge online, behaviourally

In-model metrics measure the model. A planner scored inside its own world model
is the exact comparison that hides a planner which looks good in imagination
and fails in the environment — ours did, by a wide margin.

Report grouped behavioural metrics, not a single return: outcome, failure and
its causes, progress rate and completion time, control smoothness, safety
margin. `centerline` idles 52.8% of steps and `whisker` 4.2% — that difference
is diagnostic and invisible in return alone.

**Censoring.** Episodes still running when a step budget ends must not be
silently dropped: the best agent's episodes are exactly the ones that fail to
finish, so dropping them biases every statistic toward whatever crashes
fastest. Run past the episode limit and report the censored fraction.

**Anything that cannot be exported still has to run in the real environment.**
A planner needs the model at inference, so it cannot be compiled into the fast
evaluator. Build a binding rather than accepting an in-model proxy.

## 8. Real and imagined samples are separate axes

The claim worth making is real-sample efficiency, not total compute. Track both
independently and plot performance against *real* steps.

Design the comparison to bracket the answer: a small real budget, that budget
plus imagination at 1:1 and 10:1, and a 10× real budget with no imagination. If
the imagined arms approach the large-real arm, imagination bought sample
efficiency. Ours did not — until three separate fixes landed.

## 9. Offline, constrain the actor

Every result here is offline: a fixed corpus, no data collection. In that
regime the standard failure is the policy drifting to states the model
mispredicts.

Fixing the *value function* is necessary and not sufficient. We anchored the
imagined critic to one fitted on real transitions, which helped and then
**saturated well short of BC** — in the limit the critic simply becomes the
real-data critic. The actor also needs a behaviour constraint (a KL toward the
behaviour policy). Both together reached parity with BC; either alone did not.

## 10. Do not assume search helps

More optimization against a learned model is not automatically better. Ours
inverted: with a good critic, the planner that searched hardest scored *worst*
online. A planner actively searches for high-value states, which selects for
exactly the states where the model is optimistically wrong. Low one-step error
says nothing about that tail.

Check the cheap thing first. With a small discrete action space, exhaustive
enumeration to modest depth is a batched tensor operation and beam search
matched it at 13× lower cost, while MCTS lost to depth-1 greedy. Modern MCTS
(Gumbel MuZero) is the right choice for discrete actions *in general*; it was
the wrong choice here, and only measurement showed that.

## 11. Audit the accounting before reporting a headline

Not after. This is the rule that would have caught the worst error in this
project.

A dream-PPO run was written up as reaching "91% of BC's return with 200k real
transitions". The `--real-budget 200000` flag capped the normalization
statistics and the reset-anchor buffer — and nothing else. The world model that
run depended on had been trained on the full 817,669-transition split. True
cost: **830,245 unique real transitions, 4.2x the claim and 1.8x MORE than the
BC baseline it was being favourably compared against.** The claim inverted.

No code was wrong. The accounting was, and no test could fail because no test
existed.

**A method's real-sample cost is the union over every component it depends on**
— pretraining, world model, termination head, critic, calibration rollouts,
reset anchors — not the budget flag of the last script you ran. Components
sharing a corpus overlap, so the honest figure is *unique* transitions, capped
by corpus size, not the sum.

Track labelled and unlabelled separately. They are different currencies:
`(s, a, s')` from any policy is cheap, expert demonstrations are not, and a
method that trades 2x more of the former for none of the latter is making a
real claim that summing them together destroys.

```bash
python3 -m racewm.accounting --run <run> --corpus-size <N> --claim <what you were about to publish>
```

Exits non-zero when the claim is off by more than 5%. Run it in the same breath
as the result.

### The fast checks worth running every time

None of these take more than a minute, and each caught a real error here.

| check | what it catches | caught |
|---|---|---|
| `racewm.accounting --claim` | sample-cost claims that don't survive the dependency chain | a 4.2x overstatement that inverted a headline |
| thread-count invariance | shared mutable state in "read-only" inference | policy activations shared across threads: 6.4 return on 1 thread, 0.29 on 8 |
| same policy, both environments | evaluating two things and calling them one | dream-vs-real compared *different track sets* at *different truncation limits* |
| scripted bot on train vs holdout | a holdout split that is not difficulty-matched | holdout is 1.4x easier than train for every agent |
| equal gradient steps across arms | optimization differences reported as data efficiency | a null transfer result presented as 7 sigma |
| positive control on any gate | a test with no power | confirmed the Markov gate could detect a state it should reject |

The pattern behind all of them: **every apparent world-model pathology in this
project turned out to be an evaluation error.** Model exploitation was
diagnosed twice and was wrong twice — first a GAE time-limit bug, then a
track-set mismatch. Suspect the harness before the model.

---

## The pattern worth internalising

Three of the most consequential bugs in this project produced entirely
plausible numbers:

| bug | how it looked | what caught it |
|---|---|---|
| policy activations shared across threads | 6.4 return on 1 thread, 0.29 on 8 | thread-invariance gate |
| corpus contained no crash states | imagination "survived" 100% of rollouts | online eval disagreeing |
| BC warm start silently failed | a clean frozen-vs-joint comparison of nothing | reading the state-dict keys |
| real-sample budget flag capped the wrong thing | a headline claiming 4.2x less data than it used | auditing the dependency chain |
| holdout tracks 1.4x easier than train | every absolute number inflated | running a scripted bot on both splits |

None crashed. None failed a test that existed at the time. Each was found
because an online, behavioural measurement disagreed with an offline or
in-model one — which is the argument for making that measurement cheap enough
to run constantly.
