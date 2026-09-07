# Race World Model

**Scalable world models for reinforcement learning: how well can a policy be
trained inside a learned model instead of the environment, and what actually
limits it.**

A world model that is accurate enough to train in turns environment
interaction — usually the scarce, expensive, sometimes dangerous resource —
into compute. This repository builds that end to end on a small, fast
environment and measures every step of it: the state contract, the model, the
policy trained purely in imagination, and the real-environment behaviour that
is the only thing that counts.

Small and fast is the point. The environment runs at ~1M steps/s, so every
experiment reaches significance in seconds and negative results are cheap.
Partial observability, compounding error, model exploitation, covariate shift
and calibration failure all show up for real at this scale — and being able to
run each question to a conclusion in a minute is what surfaced the six
measurement bugs catalogued in the methodology, most of which had produced
plausible, publishable-looking numbers.

The testbed is **Race** (PufferLib Ocean's `whisker_racer`): a car on a
procedural Bézier track, two whisker rangefinders, three discrete steering
actions.

Multimodal RGB and sim-to-real are natural extensions of this and are noted
where the results bear on them, but they are extensions. The core question is
whether a learned model can carry RL training, and where it stops.

---

## The method, in one pass

The order matters more than any individual technique. Each step exists because
skipping it makes a later measurement meaningless.

1. **Audit the environment before modelling it.** Race's observation was 3
   floats while its transition depended on heading, position, a nearest-segment
   index and sector bookkeeping. Its RNG was process-global, so all envs shared
   one track. Neither is visible from a learning curve.
2. **Pin a state contract in code**, hashed, checked from both the C and Python
   sides, and stamped into every dataset and checkpoint.
3. **Gate it.** Determinism (thread-invariance, bit-exact replay) and a
   Markov test *with a positive control* — if the test cannot detect a state
   you know is insufficient, its verdict on your real state means nothing.
4. **Split by whatever determines the dynamics**, into separate files. Here
   that is the track, not the row.
5. **Get baselines that can lose.** Persistence, scripted bots, behaviour
   cloning — before any world model.
6. **Train the model on its own rollouts**, not single steps.
7. **Judge everything online, behaviourally.** In-model metrics measure the
   model, not the agent.
8. **Report real and imagined samples as separate axes.**

Written up in full, with the failures that motivated each rule, in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

---

## What we found

**The world model works.** Trained on its own 10-step rollouts, closed-loop on
held-out tracks, geometry gathered from its own predicted pose:

| horizon | vs persistence |
|---|---|
| @1 | 23.6× |
| @5 | **107×** |
| @20 | 72.6× |
| @100 | 17.5× |

One-step training *diverged* past ~20 steps (0.88× at @100). Multi-step
training fixed it and improved every shorter horizon too — the failure was
distributional, not capacity.

**Behaviour cloning is a very strong baseline.** BC on the expert beats the
privileged expert it cloned (3.45 vs 3.19 laps, 39.7% vs 45.7% crash) on only
84.7% action agreement — the network smooths over the hand-tuned controller's
chattery failures.

**Planning through the model loses.** Every planner scored worse than a
two-whisker reflex bot in the real environment, despite the model being 72×
better than persistence at the horizon they plan over. Giving them a *better*
critic made them *worse*, and the planner that searched hardest scored last —
the signature of model exploitation.

**Dynamics pretraining does not help imitation learning — privileged
distillation does.** With pretraining scaled to 12.3M samples and finetuning
held fixed, forward + inverse + termination dynamics is statistically
indistinguishable from random initialisation (+8.1%, 0.4σ). Training the same
encoder to predict *track geometry it cannot observe* is worth **+56.3%
(2.4σ) at 500 labels**. It does not substitute for labels, though: 1,500 extra
expert labels beat 830,000 unlabelled transitions here. Those economics invert
for RGB, where unlabelled video is cheap and demonstrations are not — and the
mechanism that worked, inferring hidden semantic state from a partial
observation, *is* the sim-to-real problem.

**PPO trained entirely inside the world model reaches 91% of BC's return with
zero expert labels.** No real environment steps are taken during training; the
200k real transitions fit the world model and anchor imagined resets. Peak
**10.52 return / 60.2% survival** on holdout against BC's ~11.3 — and BC's
samples are expert-annotated while the world model's are `(s, a, s')` from any
policy. Gains saturate around 10M imagined steps and are flat to 28M: the
binding constraint is model quality, not imagined compute.

**And the dream turned out to be pessimistic, not optimistic.** The same policy
scores 10.50 in reality against 6.65 in imagination — reward-per-step is exact
(1.03×) but dreamed episodes die at 213 steps against 345 real. Model
exploitation never materialised because a policy cannot exploit a model that
underestimates it. The cause is that termination calibration does not transfer
across policies: the head was calibrated on corpus behaviour, and the trained
policy visits a safer distribution.

**Imagination needed three separate fixesFull numbers, every sweep and every negative result:
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. Run identifiers, checkpoints and
W&B: **[docs/ARTIFACTS.md](docs/ARTIFACTS.md)**.

---

## What limits it

- **Not world-model quality, and not imagined compute.** PPO saturates around
  10M imagined steps. Making the model **8.3x better at horizon 100** (144.8x
  persistence, on 5x the track diversity) moved the ceiling **+3%**. The
  remaining candidate is the observation itself — the policy sees two whisker
  readings, and BC measured a ~10.5 accuracy-point ceiling from exactly that.
- **Whatever the dynamics read must be in the model's input.** Track geometry
  is episode-constant so it looked like configuration; omitting it made the
  state fail a Markov test (+19% history gain). A one-hot track ID did *not*
  substitute — only the actual local geometry did.
- **Terminal states can be unobservable.** Race overwrites the crash frame
  during reset, so the corpus holds zero crash states while 33% of *imagined*
  states looked like crashes. A model cannot learn what it never saw; it can
  learn to predict that termination is coming. Hence an explicit head.
- **Termination calibration is not one number.** Matching the aggregate rate
  still left imagined episodes unimodal where real ones are bimodal — the head
  cannot express "safe for 500 steps", and no rescaling fixes that.
- **Model exploitation did not appear.** Diagnosed twice, wrong twice: a GAE
  time-limit bug, then a track-set mismatch. The dream is 0.94x *pessimistic*
  on return, and a policy cannot exploit a model that underestimates it.
- **A bilevel "teacher world" runs; it changes nothing measurable.**
  Optimising a mutable world so the policy it teaches scores well in a frozen
  reference world works mechanically -- verified second-order gradient, bounded
  memory, a competent student, a teacher that moves 16% from the reference. A
  **paired** test against a control that never updates the teacher (fixed
  student bank, common random numbers, 600 meta-steps, 2 seeds) gives a real-env
  difference of **−1.2%, t = −0.64**, with the two seeds disagreeing in sign.
  Not an underpowered effect: an absent one.
- **Accurate dynamics imply neither a good planner nor a good representation.**
  Search through the model never beat following the policy, and the best world
  model made the worst backbone for control.

## Running it

```bash
git submodule update --init --recursive
tools/build.sh eval && tools/build.sh play && tools/build.sh capi
tests/run_all.sh                      # phase-0 gates; --full adds the Markov gate

tools/make_dataset.sh corpus 20       # ~1.16M transitions, disjoint train/val/holdout
python3 -m racewm.train_wm_multistep --corpus data/corpus --rollout-k 10
python3 -m racewm.train_bc --corpus data/corpus --eval
python3 -m racewm.leaderboard         # every agent, real env, holdout tracks
```

`build/race_eval` runs 3,200 episodes in ~2s (headless C, no Python in the
loop). The ctypes binding does 797k env-steps/s for agents that cannot be
exported, such as planners. Planning itself costs ~1.2ms per decision against
~8µs for a policy — 150×, and the reason any evaluation here is ever slow.

Layout, contract, replay format and capture: **[docs/CONTRACT.md](docs/CONTRACT.md)**.
Corpus design: **[docs/DATASET.md](docs/DATASET.md)**.
Credentials: **[docs/SECRETS.md](docs/SECRETS.md)**.

---

## Status

Phase 0 (audit, determinism, contract) and Phase 1 (world model, BC, baselines)
are complete and gated. Phase 2 (imagination, planning) has honest numbers and
one clear gap: **every result here is offline**. The policy never collects its
own data, and Dreamer's sample-efficiency claim is specifically about online
interaction — so the central claim has not actually been tested yet. That, and
the RGB modality, are next.

## References

The line of work this follows, and what each contributed here.

| | |
|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018) | learn dynamics, train the controller inside them |
| [PlaNet](https://arxiv.org/abs/1811.04551) (Hafner et al., 2019) | latent dynamics + short-horizon planning (~12 steps) |
| [Dreamer](https://arxiv.org/abs/1912.01603) / [V2](https://arxiv.org/abs/2010.02193) / [V3](https://arxiv.org/abs/2301.04104) | actor-critic in imagination, H=15, λ-returns, `continue` head |
| [MuZero](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., 2020) | dynamics trained *only* through value/reward/policy |
| [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) (Danihelka et al., 2022) | policy-improvement guarantee at low simulation counts |
| [EfficientZero](https://arxiv.org/abs/2111.00210) (Ye et al., 2021) | sample efficiency via consistency + value prefix |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) (Hansen et al., 2024) | plan short, bootstrap with a value function |
| [PETS](https://arxiv.org/abs/1805.12114) / [MOPO](https://arxiv.org/abs/2005.13239) / [MOReL](https://arxiv.org/abs/2005.05951) | ensemble pessimism against model exploitation |
| [TD3+BC](https://arxiv.org/abs/2106.06860) / [AWAC](https://arxiv.org/abs/2006.09359) | constrain the actor toward the behaviour policy offline |
| [DAgger](https://arxiv.org/abs/1011.0686) (Ross et al., 2011) | why cloning compounds error off-distribution |
| [I-JEPA](https://arxiv.org/abs/2301.08243) / [V-JEPA](https://arxiv.org/abs/2404.08471) | predict in representation space, not pixels — the RGB half |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) (LeCun, 2022) | the multimodal predictive-architecture framing |

Environment: [PufferLib](https://github.com/pufferai/pufferlib) (`ocean/whisker_racer`, by Kinvert), pinned at `42f70d6`.
