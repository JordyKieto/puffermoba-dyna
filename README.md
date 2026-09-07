# Race World Model

**Building a world model that can step in either modality — symbolic game state
or RGB — and transfer the rules between them.**

The long goal is sim-to-real: a hand-crafted semantic game state exists in
simulation and does not exist in the real world, where you only have pixels. A
model that learns dynamics over structured state *and* over RGB, sharing the
same latent rules, can be trained where the state is free and deployed where it
is not. This repository is the symbolic half, built first and built to be
measured, so the RGB half has something honest to attach to.

The testbed is **Race** (PufferLib Ocean's `whisker_racer`): a car on a
procedural Bézier track, two whisker rangefinders, three discrete steering
actions, ~1M env steps/sec. Small enough to run every experiment to
significance in seconds, rich enough that partial observability, compounding
error and model exploitation all show up for real.

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

**Dynamics pretraining transfers to imitation learning — but only short
horizon.** Initialising a BC policy from a world-model encoder is worth **+23%
return at 2k labels and +20% at 10k (8σ)**, and the gain vanishes by 50k. The
catch: only *one-step* pretraining transfers. The long-horizon model is the
better world model — it is what fixed @100 rollout divergence — and it is the
*worse* backbone for control, statistically indistinguishable from random init.
Rollout stability and action-discriminative structure are not the same
objective.

**Imagination needed three separate fixes to stop being harmful**, each found
by an online measurement disagreeing with an in-model one, ending at rough
parity with BC:

| | return | survival |
|---|---|---|
| first working Dreamer run | 0.33 | — |
| + termination head | 6.03 | — |
| + unfreezing the world model | 7.57 | 34.5% |
| + critic anchored to real data | 8.97 | 47.8% |
| + actor constrained toward BC | **11.24 ± 0.16** | **64.2%** |
| *behaviour cloning alone* | *11.02* | *60.3%* |

Return is +2.0% at ~1.4σ with one seed tying BC — a small real effect, not a
win. Survival is sturdier: all four seeds beat BC by ~4 points.

Full numbers, every sweep and every negative result:
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. Run identifiers, checkpoints and
W&B: **[docs/ARTIFACTS.md](docs/ARTIFACTS.md)**.

---

## What generalizes to the RGB half

- **Whatever the dynamics read must be in the model's input.** Track geometry
  is episode-constant, so it looked like configuration; omitting it made the
  state test fail (+19% history gain). A one-hot track ID did *not* fix it —
  only the actual local geometry did. The RGB analogue is scene structure that
  a single frame does not carry.
- **Terminal states can be unobservable.** Race overwrites the crash frame
  during reset, so the corpus contains zero crash states while 33% of *imagined*
  states looked like crashes. A model cannot learn what it never saw; it can
  learn to predict that termination is coming. This is why Dreamer carries an
  explicit `continue` head.
- **Fit critics on real data, not model output.** A critic fitted inside the
  model inherits its optimism, and the policy then optimizes straight into it.
- **Offline, the actor needs a behaviour constraint.** Fixing the value
  function was necessary and not sufficient.
- **Accurate dynamics do not imply a good planner.** Low one-step error says
  nothing about the tail a planner deliberately seeks out.
- **Accurate dynamics do not imply a good representation either.** The best
  world model here made the worst backbone for control. Choose a pretraining
  objective by measuring transfer, not rollout accuracy — the asymmetry that
  makes pretraining worth it (unlabelled dynamics data is cheap, expert labels
  are not) is far sharper for RGB than it is here.

---

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
