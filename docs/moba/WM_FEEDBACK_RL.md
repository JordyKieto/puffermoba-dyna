# Scoring rollouts, not steps: human feedback and RL for the world model

**Status:** proposal, 2026-09-26. Nothing here is built.

## The problem with the loss

Training scores each predicted tick against the real replay. After a few
hundred ticks that's the wrong question. The game branches, so a rollout can
leave the replay and still be a believable game. The per-tick loss punishes
that as much as nonsense.

What Jordy actually judges when watching the replay next to the world-model
rollout (same actions) is different:

1. **Outcome after x steps, "the circle":** did the hero end up close to where
   it really is? Close enough counts; exact doesn't matter.
2. **Trajectory, "how it got there":** smooth walking at a realistic speed, or
   teleporting, jittering, freezing?

Those two judgments are a scoring rubric for a whole rollout, not for a tick.
This proposal makes the rubric the training signal.

## Part 1: the rubric as an automatic reward

Both judgments can be computed from a rollout and the real replay, per hero:

| term | definition | notes |
|---|---|---|
| **circle** | Reward 1 if the hero's position after x ticks is within radius r of the real position, falling off smoothly outside. Checked at several horizons (e.g. x = 16, 64, 256). | r grows with x: early positions should be tight, later ones only roughly right. Deaths are compared as events (died / didn't), not as positions. |
| **trajectory** | Penalise per-tick moves the game can't make: a step longer than the hero's move speed (teleport), unless it's a real death respawn; direction reversing every tick (jitter); standing still when the actions say walk (freeze). | All checkable against the game's movement rules and the recorded actions. |
| **speed match** | Mean speed over the window close to the real hero's. | Stops "stand still to be smooth" from scoring well. |

The same pattern extends to other effects once motion works:
- health and towers: did it land in a plausible band;
- casts: did the right ability fire within a few ticks of the real one.

## Part 2: "PPO for the world model"

The world model rolls out a trajectory, the rubric scores it, and the model is
pushed towards trajectories that score well. That's RL with the world model as
the policy. Three ways to do it, cheapest first:

1. **A differentiable rollout loss (no RL).** Hero positions are continuous,
   so gradients can flow through a rollout's positions.
   - The circle becomes a hinge loss: zero inside the radius, growing outside.
   - The trajectory terms become penalties on per-tick displacement.
   - Today's rollout training detaches the predicted state each step; this
     would keep the position chain attached over a window.
2. **Group-relative RL (GRPO-style), the "PPO for the world model".**
   - The model is made to *sample* futures: probability distributions over
     position changes and over events (death, cast, hit), not one fixed
     prediction.
   - From the same start and actions, sample K rollouts, score each with the
     rubric, and push up the likelihood of the ones that beat the group
     average. No critic is needed: the group is its own baseline, a natural
     fit because all K share the start.
   - It also lets the model represent *several* plausible futures instead of
     averaging them into mush, which is the root of "sluggish but safe" motion.
3. **A learned realism critic** for what the rubric can't express. See part 3.

**Keep the per-tick loss alongside.** It anchors the model to the real game.
The rubric shapes what "good enough" means over longer horizons.

**Goodhart risk:** the world model will satisfy the rubric in the cheapest way
it can, the way the dream PPO policy exploited the dream. Mitigations:
- the speed-match term;
- keeping the replay loss;
- checking with fresh human judgments now and then (part 3).

## Part 3: human feedback, to calibrate and cover what the rubric can't

The replay viewer already shows the rollout next to the real replay. Add:

- **Pairwise judgments:** two rollouts (e.g. two checkpoints) of the same
  clip; a key for "A is more realistic" or "B is more realistic". About 3–5 s
  per judgment. Much more consistent than 1–10 scores.
- **Short clips:** 5–10 s of rollout, not whole games.
- **Failure tags:** keys for "tower hurt for no reason", "teleport",
  "jitter", "wrong death", "freeze". A tag used often becomes a new automatic
  rubric term (part 1).
- **A log file:** one line per judgment (clip, checkpoints, choice, tags).

**Use of the labels, cheapest first:**
1. **Choosing between models:** E5 vs E6b vs checkpoints, directly by eye.
2. **Calibrating the rubric:** pick the circle radii and trajectory
   thresholds that agree with human judgments.
3. **A realism critic** trained on the pairs (a reward model). It scores
   millions of rollouts at machine speed. With the critic choosing which clips
   to show (the ones it's least sure about), a few hundred to a couple of
   thousand labels should be enough. At ~5 s each that's under 3 hours of
   human time.

**The human's time is not the limit:** the rubric is automatic, and the critic
generalises the labels. Human time goes into calibrating and auditing, not
into scoring every rollout.

## Build order

1. The rubric as **metrics** first (circle hit rate at several horizons,
   teleport/jitter/freeze rates, speed match), logged next to the horizon
   loss. It costs nothing and makes the viewer judgments into numbers for
   every checkpoint.
2. Pairwise judgments and failure tags in the viewer.
3. The differentiable rollout loss (part 2.1), A/B against E5.
4. Probabilistic outputs and group-relative RL (part 2.2).
