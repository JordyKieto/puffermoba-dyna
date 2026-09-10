# Can a neural network learn to skip training?

A plain-language walkthrough of the `optwm/` experiments: the idea, the toy we
built to test it, the one result that genuinely surprised us, exactly how narrow
it is, and the one experiment that would decide whether it matters.

No prior context needed beyond knowing what training and a learning rate are.

---

## 1. What training actually is, mechanically

Take a small neural network. Ours has **771 numbers** in it — the weights. That
is the entire network: 771 floating-point values in a list.

Training means repeating this:

1. Show it 64 examples.
2. See how wrong it is (the **loss**).
3. Compute which direction to nudge each of the 771 numbers (the **gradient**).
   This is the expensive part — a forward pass and a backward pass.
4. Nudge them a little. How little is the **learning rate**.

Do that 800 times and the loss falls from 1.19 to 0.003. The network has
learned.

The expensive part is step 3. Every single one of those 800 steps requires a
real forward and backward pass. On a 771-number network that's instant. On a 7B
model it's a GPU-hour problem. **That cost is what this whole line of work is
poking at.**

---

## 2. What a "trajectory" is

Here's the only new vocabulary you need.

While training, write down all 771 numbers every 5 steps. You end up with a
list of snapshots:

```
step   0:  [ 0.041, -0.118,  0.203, ... ]   (771 numbers)
step   5:  [ 0.039, -0.121,  0.207, ... ]
step  10:  [ 0.036, -0.125,  0.210, ... ]
...
step 800:  [-0.284,  0.512, -0.077, ... ]
```

That list is a **trajectory**. It is the complete recorded history of one
training run — like the frames of a movie, where each frame is the entire
network at that moment.

Train the same network 128 times with different random data orders and
different learning rates, and you have **128 trajectories**. That's our
dataset. Not images, not text — recordings of training runs.

The **delta** is just the difference between two frames: how much the 771
numbers moved over those 5 steps. Small. Usually a nudge in the fourth decimal
place.

---

## 3. The idea

You have watched 128 training runs. You know what training *looks like* for
this network on this problem.

**Could you now predict where training ends up, without running it?**

Concretely: I hand you the network at step 130. Instead of running steps 131
through 330 — 200 real forward/backward passes — you *predict* what the weights
will be at step 330 and jump straight there.

The analogy is a thrown ball. Simulating it properly means computing forces
every millisecond. But you've watched a thousand balls, so you know the arc.
You can skip to where it lands.

We want the same for training: skip the arc, land on the answer.

This is the same shape as a world model in reinforcement learning — learn the
dynamics of an expensive system, then run it cheaply in your head — except the
"expensive system" here is **gradient descent itself**.

### Why it might not work

A game world model wins because the game state is small (positions, health,
cooldowns) while the game engine is big. Here it is the reverse: the "state" is
every weight in the network. For a 7B model that is 7 billion numbers in, 7
billion numbers out. A predictor that reads and writes 7 billion numbers has
probably just reinvented the expensive thing it was trying to avoid.

The whole idea therefore rests on one bet: **training trajectories are much
simpler than the number of weights suggests.** That's the first thing we tested.

---

## 4. Experiment 1 — is there any structure? (`optwm/compress.py`)

**The question.** When training moves the weights, does it move them in a few
consistent directions, or all over the place?

If it's a few directions, we could describe training compactly — a small dial
setting instead of 771 numbers — and predicting a small thing is much easier
than predicting a big one.

**How we asked.** Collect all the deltas from all trajectories and ask: can a
small number of directions describe most of the movement? (That's PCA/SVD.
"90% explained variance at rank 128" means: 128 directions account for 90% of
all observed motion.)

**Result: no.** Fitting on some runs and testing on held-out runs, 128
directions explain **7.8%** of the movement in the runs they hadn't seen. Random
noise scores 0.2%, so it's above nothing — but nowhere near enough. And rank 1
alone already gives 6.1%, so basically *all* the shared structure is a single
overall drift direction; the other 127 directions add 1.7 points.

**Three ways this measurement lies, and what we did about each.** These matter
more than the number, because each one would have produced a fake success:

| The trap | What it would have shown | The fix |
|---|---|---|
| Fit and test on the same runs | **90.0%** — looks like a triumph | Fit on some runs, score on held-out runs → 7.8% |
| Everything drifts one way | rank 1 "explains" it all, meaninglessly | Report the drift direction separately (it's 4.7%) |
| Score by weight accuracy | Flatters any method, since weights barely move | Score by **loss** — did the model actually get better? |

There's a subtler trap we also had to handle. Two networks trained from
different random starting points end up in different "coordinate systems" — you
can shuffle a network's hidden neurons and get an identical network with totally
different weight numbers. So comparing weight directions across runs from
different starts is meaningless *even if* the underlying dynamics were simple.
Forcing every run to start from the same point moved the answer from 0.5% to
7.8% — a 15× swing from fixing the confound alone.

**Verdict:** the compact-description bet fails, at least for a linear
description of raw weights. That could have ended it. But "can't compress
linearly" and "can't predict" are different claims, so we tested the second one
directly.

---

## 5. Experiment 2 — can a function predict one jump? (`optwm/predict.py`)

**The question.** Forget compression. Train a second neural network — call it
**F** — whose entire job is: *given the network's current weights, output the
weights 200 steps later.*

F is a 1.4-million-parameter network predicting a 771-number network. Absurdly
lopsided, and deliberately so. We wanted to know whether this is *possible*
before asking whether it's *cheap*.

**The baselines are the whole experiment.** This is where this kind of work
usually goes wrong. 200 steps barely move the weights, so a "predictor" that
just hands back its input looks accurate. We compare against three cheats:

- **persistence** — return the input unchanged. Predict nothing.
- **mean delta** — add the average training step, ignoring the input entirely.
- **velocity** — assume it keeps moving the way it just moved (straight-line
  extrapolation).

**How to read the score.** Suppose at some point the network's loss is 0.469.
Run the 200 real steps and it becomes 0.441. So the real training gained
**0.028** of loss. Now:

- persistence stays at 0.469 → gained nothing → **0%**
- F predicts weights whose loss is 0.451 → gained 0.018 of the 0.028 → **64%**

That's the "% of true gain" column. 0% = no better than doing nothing. 100% =
matched real training exactly.

**Result, all runs on the same task:**

| real steps jumped | persistence | mean delta | velocity | **learned F** |
|---|---|---|---|---|
| 10 | −0% | −8% | −535% | **82%** |
| 40 | −0% | −10% | −3426% | **70%** |
| 200 | 0% | 7% | −15678% | **87%** |

F recovers **87%** of what 200 real training steps achieve. The cheats get
nothing, and velocity is catastrophic — straight-line extrapolation flies off a
cliff, which tells you training curves bend a lot.

**Two design faults we hit first**, both of which had made the experiment
meaningless before we noticed:

1. **No headroom.** Our first version jumped 10 steps at a point where 10 steps
   change the loss by 0.011. Doing nothing was already 97% as good as the truth,
   so nothing could be distinguished from anything. Fix: pick jump points where
   the real jump actually accomplishes something, and print how much headroom
   there is so the reader can check.
2. **A misread.** We read an early result off 12 samples and concluded F was
   the worst method. At 32 samples it was statistically indistinguishable from
   the best. Retracted.

**Then it broke.** Give each training run a *different* task (a different thing
to learn) and F goes to **−123%** — worse than doing nothing. It can't tell
which problem it's looking at, so it predicts something like the average across
all problems, which is worse than predicting nothing.

We tried the obvious fix: tell F what the task is, cheaply — one forward pass on
a fixed 32-example probe, showing what the model currently says and what this
task wants it to say. **It got worse** (−401%). We suspected too few distinct
tasks to learn from, so we quadrupled them: **−163%.** Better, still useless.

---

## 6. Experiment 3 — the loop, and the narrow WOW (`optwm/loop.py`)

One jump working is interesting. It is not the claim. The claim is the **loop**:

```
jump 200 steps with F  →  run 25 REAL steps  →  jump again  →  run 25 real  →  ...
```

This is the pattern that matters: **dream cheaply, verify occasionally.** Each
jump starts from the *previous jump's output*, not from a real trajectory, so
errors can compound. Chaining is where a learned simulator either holds together
or spirals into nonsense.

**The baseline that decides everything.** Say the loop reaches 800 steps of
progress while only performing 45 real steps. Tempting headline: "94% of
training eliminated." But you must ask:

> What if I had just done 45 real steps and stopped?

If plain 45 real steps get you to the same place, **the jumps did nothing** and
45 steps was simply enough. That's the **budget-matched** baseline, and it's the
one that makes or breaks the result. We also run a **no-op loop** — the identical
schedule with F replaced by "do nothing" — to prove it's F doing the work rather
than the real steps the schedule happens to include.

Everything is scored on **train loss**, because that is what SGD actually
minimises and it falls monotonically here. Test loss on this toy bottoms out
around step 100 and then *rises* (the model starts memorising), so scoring on
test loss would reward a method for refusing to make progress.

### The result

800 steps of progress, jumps of 200 steps, 8 trials:

| real steps eliminated | real steps used | **LOOP** | budget-matched | no-op loop | advantage |
|---|---|---|---|---|---|
| 0% (full real) | 800 | **0.0035** | — | — | — |
| 49% | 405 | 0.0050 | 0.0148 | 0.0203 | 3.0× |
| 74% | 205 | 0.0160 | 0.0681 | 0.0876 | 4.3× |
| 87% | 105 | 0.0207 | 0.1671 | 0.2085 | 8.1× |
| **94%** | **45** | **0.0220** | 0.3097 | 0.3726 | **14.1×** |

**Read the last row.** 45 real training steps plus 4 chained predicted jumps
reaches a train loss of 0.0220. Spending those same 45 real steps directly on
training gets 0.3097 — **14× worse**. The no-op control is 0.3726, so it isn't
the schedule. Full real training reaches 0.0035, so the loop lands about 6×
short of the real thing while doing 1/18th of the work.

**And the jumps chain.** Four in a row, no divergence. That was the main
theoretical worry — small weight errors changing later gradients and compounding
— and at this scale it did not happen.

That is the WOW. It is real, it survives three controls, and it is narrow.

### Exactly how narrow

F was trained on runs that all shared one starting point and used four learning
rates. So we asked what it can be reused for:

| Change | Result at 94% elimination | Reusable? |
|---|---|---|
| A **learning rate F never saw** (0.07) | 19× over budget-matched | **Yes** |
| A **starting point F never saw** | 14× → **1.5×** | Barely |
| Train on **128 different starting points** | **0.99×** — no benefit at all | **No** |

The last row is the important one. We thought init-dependence was a data
coverage problem, so we trained F on 128 different starting points. That made it
**worse**, not better — it collapsed to exactly the budget-matched baseline.

**So what is F actually doing?** It is not learning "how gradient descent
behaves." It is learning **the shape of one family of training runs** — runs
from this starting point, on this task. Widen the family and it learns nothing
useful, because it ends up averaging over incompatible families.

Note this is the *same failure* as the different-task result in Experiment 2:
**one family → large effect; several families → nothing.** Two completely
independent ways of widening the family (change the task, change the starting
point) produce the same collapse. That's consistent with Experiment 1 finding
almost no shared structure across runs.

### The honest cost accounting

The 14× is a quality comparison, not a compute saving.

| | |
|---|---|
| One real training step | ~141,000 multiply-adds |
| 200 real steps (what a jump replaces) | ~28,000,000 |
| One F prediction | ~1,450,000 |
| **Per-jump saving** | **~19×** |
| But: cost to train F | ~1,300,000,000,000 |
| **Break-even** | **~10,000,000 student steps** |
| Steps we actually generated | 102,000 |

So F would need roughly 100× more downstream use than the data it was trained on
before it pays for itself — while only working on the exact family it was
trained for. At this scale it is a feasibility demonstration, full stop.

---

## 7. What the sweep is, and why it's the experiment that matters

Here is the gap in everything above.

We only tested two situations, and they're the two extremes:

- **Identical.** Every training run starts from the same point, learns the same
  task. → F works brilliantly (14×).
- **Completely unrelated.** Every run starts somewhere random, learns a
  different randomly-generated task with nothing in common. → F is useless.

**Real training jobs are nowhere near either end.**

Picture a company fine-tuning the same base model over and over. Every job
starts from *the same pretrained checkpoint*. Every job trains on *related*
data — customer support tickets this week, slightly different tickets next week.
The jobs aren't identical, but they are enormously more similar to each other
than two random networks learning two random tasks.

That middle ground is the entire practical case for this idea, **and we have
not tested a single point in it.**

### The sweep

Instead of starting points that are either *the same* or *fully random*, build
them as:

```
starting point  =  one shared base  +  ε × random perturbation
task            =  one shared base task  +  ε × random perturbation
```

Then sweep ε:

| ε | meaning | what we know |
|---|---|---|
| 0 | all jobs identical | **tested: 14× advantage** |
| 0.01 | like fine-tuning from one checkpoint on near-identical data | untested |
| 0.1 | related jobs, noticeably different data | untested |
| 0.5 | loosely related | untested |
| 1.0 | unrelated | **tested: no advantage** |

The output is one curve: **how similar do training jobs have to be before a
learned simulator pays off?**

That single curve decides the idea:

- If the advantage survives to ε = 0.1, then "learn the training dynamics of a
  family of related fine-tuning jobs" is a real thing, and the interesting
  question becomes scale.
- If it collapses the moment ε > 0, then F only ever memorises one specific
  trajectory and there is nothing here to build on.

It is a small change to how tasks and starting points are generated, and it runs
in a couple of minutes on a laptop. It is the cheapest decisive experiment
available, which is why it should be the next one.

---

## 8. Where this stands

**Established.**
- Weight deltas are not compressible into a small shared description (7.8% at
  rank 128, 0.2% noise floor, essentially all of it one drift direction).
- A learned F *can* predict a 200-step jump within one family, recovering 87% of
  the real loss improvement where every naive baseline gets ~0%.
- Chained jumps work: 94% of real steps eliminated at 14× the quality of the
  same real-step budget, verified against a budget-matched and a no-op control,
  4 jumps deep without divergence.
- Learning rate generalises, including unseen values.
- Starting point and task do **not** generalise, and training on many of either
  removes the benefit entirely.
- No net compute saving at this scale (~100× more use needed to amortise F).

**Not established.**
- Anything in the middle of the similarity range — i.e. the only regime that
  would matter in practice.
- Whether the family-collapse is fundamental or a capacity limit. Two
  independent kinds of family variation collapsing identically argues against a
  simple capacity story, but does not settle it.
- Whether any of it survives a network larger than 771 numbers.

**Next.** The ε sweep. One curve, minutes to run, and it decides whether the
narrow result generalises into something worth scaling.

---

## Running it

```bash
# Experiment 1: is there shared structure in the deltas?
python -m optwm.compress --trajectories 12 --steps 1500 --shared-init

# Experiment 2: can F predict a single jump? (add --vary-task to break it)
python -m optwm.predict --runs 128 --steps 800 --ks 8,40

# Experiment 3: the loop, and the reusability tests
python -m optwm.loop --runs 128 --steps 800 --k 40 --target 800
python -m optwm.loop --runs 128 --steps 800 --k 40 --eval-lr 0.07     # unseen LR
python -m optwm.loop --runs 128 --steps 800 --k 40 --eval-init 77     # unseen start
python -m optwm.loop --runs 128 --steps 800 --k 40 --multi-init --eval-init 77
```

Each takes 30 seconds to 2 minutes on a laptop CPU. Nothing here needs a GPU.

## Related work this sits next to

- **Learning to learn by gradient descent by gradient descent** (Andrychowicz et
  al., 2016) — learns the optimizer's update rule. Still computes the gradient.
- **VeLO** (Metz et al., 2022) — the same idea at scale. Still computes the
  gradient.
- **Decoupled Neural Interfaces / synthetic gradients** (Jaderberg et al., 2017)
  — predicts gradients so parts of a network can update without waiting for
  backprop. Philosophically the closest ancestor: learn what expensive future
  computation would return, then act on the prediction.
- **Dataset Distillation** (Wang et al., 2018) and **trajectory matching**
  (Cazenavette et al., 2022) — learns tiny synthetic data whose few steps land
  where long real training would. Attacks the same object from the data side.
- **Neural Tangent Kernel** (Jacot et al., 2018) — training dynamics as an
  analytical object rather than a learned one.

What's different here: those learn the *update rule* or the *data*. This tries to
learn the *trajectory* — skipping the forward/backward passes entirely rather
than making them cheaper.
