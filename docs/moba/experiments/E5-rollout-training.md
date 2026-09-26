# E5 — autoregressive (rollout) training from E4

**Question:** does training the model on its own predictions keep open-loop
rollouts close to the real game for longer?

- **Status:** running. Relaunched 2026-09-25 16:57 PDT (23:57 UTC) for **6,000,000 updates** with no time limit. It continues from E4's final checkpoint (update 3,877). Checkpoints are kept every 2,000 updates; validation and the horizon metric run every 500.
- **Code:** `--rollout` in `mobawm/wm/train.py` (the `Lanes` class).
  Experimental and **uncommitted**.
- **wandb:** (filled in at launch)

## Why

Until now, training has only been teacher-forced:
- Each sample is a 16-tick slice of a real game.
- The model is always fed the real state and trained on the next one.

The replay viewer instead runs the model on its own predictions for hundreds of
ticks, which is two things it never saw in training:
1. its own errors as input;
2. GRU memory carried for more than 16 ticks.

At update 500, E4's hero position error beat copying the start state at 64
ticks, but not at 300.

## Method

`--rollout` replaces the 16-tick windows with **lanes**:
- **Real game, own predictions:** each of the 768 batch lanes walks forward
  through a real game with its recorded actions. The model is fed its own
  previous prediction.
- **Open-loop spans:** each lane runs open loop for a span of ticks, then
  snaps back to the real state (keeping its GRU memory) and draws a new span.
  Span lengths are drawn from a Gaussian over log2(length): mean 6 (64
  ticks), standard deviation 3, clipped to 1–4,096 ticks.
  - About 2% of spans are a single tick (teacher forcing).
  - Half are under 64 ticks, and about 9% are over 1,000 ticks.
  - A game is 1,200–4,200 ticks, so the longest spans cover a whole game.
  - Flags: `--span-mu`, `--span-sigma`, `--max-span`.
- **Step decay:** a step's loss weight depends on how many steps the lane
  has run open loop since it was last set to the real state (its "age").
  - The weight is 2^(−age/64), with a floor of 0.02. So h1 weighs 1, h64
    weighs 0.5, h256 weighs 0.06, and anything from about h360 on weighs 0.02.
  - Weights are normalised to a mean of 1 across the batch, so they shift
    emphasis between lanes without changing the overall loss scale.
  - Why: late steps of a long rollout have drifted far and carry huge
    losses. Without decay they would dominate the gradient over the
    near-term transitions.
  - Flags: `--decay-halflife` (0 = off), `--decay-floor`.
- **End of a game:** the lane jumps to a random point in another game, with
  fresh memory.
- **Loss:** at every step it compares the real next state with the lane's
  *current input*, which may have drifted. The model learns to pull its
  drift back towards the real game.
- **Gradients:** they flow through the GRU memory within each 16-step chunk,
  and the memory is carried (detached) across chunks. They do not flow
  through the predicted states, which are rounded and clamped.

## Setup

```sh
.venv/bin/python tools/pod.py train e5_rollout -- --baseline --rollout \
    --init runs/token_wm/e4_baseline_v2_1h \
    --data data/moba/mix1k/dense --train-games 0-899 --held-games 900-999 \
    --minutes 0 --max-updates 6000000 --batch 768 --lr 1e-3 --bf16 --chunk 16 --max-span 4096 \
    --log-every 50 --val-every 500 --val-batches 4 --ckpt-every 2000 --wandb
```

## Metric: held-out loss by horizon (the key metric)

For each horizon h in 1, 4, 16, 64, 256, 1024, 4096:

- **Windows:** the same 32 windows from the held-out games (900–999) every
  time, with a fixed seed. The model warms its memory on 8 real ticks, then
  runs on its own predictions.
- **`loss@h`:** at step h, the model predicts state h from its own state
  h−1. `loss@h` is the training loss (`step_loss`, baseline weights) of that
  prediction against the real state h.
  - `loss@1` is the ordinary one-step, teacher-forced loss.
  - Higher h means the model is predicting from inputs that have drifted
    further.
- **`hero_loss@h`:** the hero part of `loss@h`.
- **`hero_pos_L1@h`:** the mean |dx|+|dy| in cells between each predicted hero
  and the real one at step h, over the 10 heroes.
- **Window sets:** horizons up to 1,024 share one window set. h4096 needs
  windows from games longer than 4,104 ticks, so it uses a second set drawn
  only from those few long games.

It is computed at every validation (every 500 updates, and at update 0, which
is the E4 checkpoint). It is logged to wandb as `horizon/loss@h`,
`horizon/hero_loss@h` and `horizon/hero_pos_L1@h`, together with `epoch`
(supervised transitions seen ÷ training-set size).

**Control:** E4's numbered checkpoints on the same metric
(`python -m mobawm.wm.openloop_eval --baseline --stats <E4 dir> <ckpts>`). They
show what more teacher-forced training was doing to each horizon.

## Verdict so far: very promising

After only 500 updates, rollout training fixes the main failure of the
teacher-forced model: rollouts no longer come apart over time. It costs
one-step precision (jitter), which is the next thing to fix.

## Results

### Control: E4's teacher-forced checkpoints (held-out loss by horizon)

| E4 checkpoint | h1 | h4 | h16 | h64 | h256 | h1k | h4k |
|---|---|---|---|---|---|---|---|
| 500 | 1.10 | 4.17 | 18.7 | 86 | 202 | 258 | 321 |
| 1,000 | 0.88 | 3.95 | 17.9 | 76 | 213 | 305 | 354 |
| 1,500 | 0.81 | 3.79 | 17.7 | 79 | 204 | 269 | 312 |
| 2,000 | 0.77 | 3.61 | 18.4 | 93 | 238 | 280 | 324 |
| 2,500 | 0.77 | 3.56 | 18.5 | 83 | 214 | 295 | 410 |
| 3,000 | 0.76 | 3.11 | 16.1 | 74 | 200 | 288 | 373 |
| 3,500 | 0.77 | 3.14 | 19.9 | 100 | 240 | 312 | 403 |
| final (3,877) | 0.78 | 3.19 | 14.7 | 81 | 205 | 314 | 443 |

- **What more teacher-forced training did:**
  - It improved h1 (−30%) and h4 (−24%).
  - It left h16–h256 flat, within noise.
  - It made h1k and h4k worse. At h4k, final E4's heroes average 129 cells
    from the real ones (hero position error).
- **Noise:** neighbouring checkpoints swing ±15–20% at long horizons. Re-running
  the same final checkpoint in E5 at update 0 gave h4k 429, against 443 here.
  h4k comes from only a few long games. So an E5 gain has to be well beyond
  these swings to count.

### E5 (from E4 final)

| update | h1 | h4 | h16 | h64 | h256 | h1k | h4k |
|---|---|---|---|---|---|---|---|
| 0 (= E4 final) | 0.78 | 3.18 | 15.5 | 80 | 213 | 323 | 429 |
| 500 | 4.42 | 8.83 | 19.7 | 38.6 | 50.3 | 60.4 | 70.9 |

Hero position error (cells), E4 final → E5 at update 500:

| h1 | h4 | h16 | h64 | h256 | h1k | h4k |
|---|---|---|---|---|---|---|
| 0.4 → 2.0 | 1.1 → 4.7 | 5.6 → 7.6 | 25.8 → 12.7 | 55.6 → 12.7 | 91.9 → 12.9 | 122.9 → 14.1 |

- **At update 500 it's a trade-off:**
  - h64–h4k improved 2–6×, far beyond the noise.
  - h1–h16 got worse. h1 rose 0.78 → 4.42, and teacher-forced validation rose
    1.11 → 5.67.
  - One-step hero walking went from 50% better than copying to 134% worse.
- **Likely cause:** most training steps come from drifted lanes, where the
  target is a large correction. The model can't tell a real input from a
  drifted one, so it corrects even when it shouldn't.
- **Attractor check: it is not an attractor.** One held-out rollout, game 900
  from tick 500, hero position gap in cells:

  | ticks ahead | 16 | 64 | 256 | 1,000 |
  |---|---|---|---|---|
  | E4: model vs real | 6.0 | 24.5 | 39.7 | 80.5 |
  | E5 @500: model vs real | 7.5 | 9.5 | 6.9 | 12.6 |
  | E5 @500: real actions vs another game's actions | 15.2 | 48.7 | 47.2 | 51.1 |

  - E5 follows the actions: with the wrong actions its heroes end up about
    50 cells from where the real actions put them.
  - But it jitters. Mean hero speed is 2.11 cells/tick, against a real 1.16;
    E4 was 0.60. The model overshoots and corrects every tick, which is the
    h1 loss increase.

- **Speed:** 0.6 updates/s.
- **Training loss:** 118 at update 50, of which creeps 50 and neutrals 10.6.
  That's alive/respawn drift in open loop.

### Qualitative review in the replay viewer (Jordy, 2026-09-25)

**Setup:** checkpoint at update 500, held-out game 900 from tick 500, a
1,000-tick horizon. Compared against the teacher-forced E4 model.

**Better than before rollout training:**
- **Heroes are far more stable.** No more flying off the map. The model gets
  the broad strokes of each hero's movement, or lack of it, much better.
- **No visible degeneration over time,** unlike the E4 teacher-forced model,
  which drifts further and further from the game.
- **Creeps and neutrals move very naturally.** Their exact positions and
  outcomes are not the real ones, but their trajectories look right (A+).
- **Abilities are predicted more often.** In places, the damage to a hero
  lined up with the real game.

**Failures seen:**
- **Tower health doesn't match.** A tower was damaged in the model but fine
  in the real game, and the reverse also happened.
- **Jitter,** from the numbers above: heroes move about 2× the real speed,
  overshooting and correcting each tick.
