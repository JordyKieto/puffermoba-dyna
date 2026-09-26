# E6 — rollout training + counterfactual siblings from the real sim

**Question:** does adding real-sim counterfactuals (what would have happened
under a different action, from the same state) teach the model what each
action causes, on top of E5's rollout training?

- **Status:** E6 ran 100 updates (launched 18:14 PDT), then was replaced on
  the same pod by **E6b** (launched 18:26 PDT, 6,000,000 updates). E6b is
  E6 plus the counterfactual effect auxiliary losses (below).
- **Starts from:** E5's checkpoint at update 3,500.
- **Control:** E5 continuing from that same checkpoint on the first pod.
- **Code:** `--cf` (see [../COUNTERFACTUALS.md](../COUNTERFACTUALS.md)).
  Experimental and **uncommitted**.
- **wandb:** run `e6_cf` in project `moba-wm`.

## Setup

```sh
POD=cf .venv/bin/python tools/pod.py train e6_cf -- --baseline --rollout --cf \
    --cf-forks 4096 --cf-micro 512 --cf-k 7 --cf-stride 8 \
    --init runs/token_wm/e5_rollout_at3500 \
    --data data/moba/mix1k/dense --train-games 0-899 --held-games 900-999 \
    --minutes 0 --max-updates 6000000 --batch 768 --lr 1e-3 --bf16 --chunk 16 --max-span 4096 \
    --log-every 25 --val-every 250 --val-batches 4 --ckpt-every 1000 --wandb
```

- **Each update has two parts:**
  - E5's rollout lanes: 768 lanes × 16 steps.
  - 4,096 fork points × 8 siblings (the real action plus 7 single edits):
    32,768 counterfactual samples, backpropagated in micro-batches of 512 fork
    points.
- **The two losses are simply added** (`--cf-weight 1`).
- **Sim:** 11 worker processes (the container's quota is 13.6 CPUs) replay the
  900 training games from their logs. Held-out games are never forked.

## Sizing run (40 updates, before launch)

| | |
|---|---|
| speed | 0.23 updates/s |
| counterfactual samples trained | ~7,850/s |
| GPU time waiting on the sim | 0%: the sim is not the bottleneck |
| GPU | 93% utilisation, 125 GB of 141 |
| counterfactual loss | 2.61 → 2.17 in 40 updates |

A single 2,048-fork-point pass (16k samples) ran out of GPU memory. The
attention distance bias is the big tensor, hence the micro-batches.

## Metric

The same as E5: held-out loss by open-loop horizon, h1–h4k, plus hero
position error. It is logged every 250 updates as `horizon/*`.

Comparison: E6 at update N against E5 at update 3,500 + N. Wall-clock time is
compared too, since an E6 update does about 3.5× more work.

## What each update trains

1. **Autoregressive rollout lanes, as in E5:** 768 lanes × 16 steps on the
   model's own predictions. Spans run from 1 tick to a whole game, with step
   decay.
2. **Counterfactual siblings:** 4,096 fork points × 8 action sets, one step
   from a real state, after 4 real ticks of memory warm-up. They branch from
   real states, not from the model's own drifted states; the latter needs
   DAgger with valid-state snapping, which is future work.

## E6b: counterfactual effect auxiliary losses

```sh
... same as E6 ... --cf-effect 1.0 --cf-noeffect 0.2 --cf-affected 0.2
```

For each edit sibling s, the **effect** is (edit sibling − real sibling), in
normalised delta units. The subtraction cancels everything the siblings share
(state, memory, the other heroes' actions) and leaves only what the edited
action caused.

| term | weight | what it is |
|---|---|---|
| effect | 1.0 | Huber(predicted effect, true effect) where the true effect is nonzero. Averaged per edited action head, so move / attack / Q / W / E count equally. Hero x/y keep their ×10. |
| noeffect | 0.2 | Huber(predicted effect, 0) where the edit changed nothing: no hallucinated effects. |
| affected | 0.2 | BCE on "does this edit change this entity at all", from a training-only head on h_edit − h_real. Positives and negatives are averaged separately. |

- **Gated fields:** the effect uses the gate as a probability
  (sigmoid(change) × value), so it trains the gate too.
- **Not saved:** the affected head isn't part of the world model or its
  checkpoint.

### Live metrics (logged every 25 updates as `cf/metric/*`)

- **effect explained %,** per edited head: 1 − |pred effect − true effect|₁ /
  |true effect|₁, over entries with a real effect. 100% is perfect; 0% is no
  better than predicting no effect.
- **affected F1 %:** of the affected-entity head.

**Baseline, E5 at update 3,500** (one CPU batch of training-game forks, before
any counterfactual training):

| edited action | move y | move x | attack | Q | W | E |
|---|---|---|---|---|---|---|
| effect explained | −95% | −80% | −2% | +1% | +1% | +1% |

- **Q/W/E:** the rollout-trained model essentially ignores what an ability
  press changes.
- **Movement:** its predicted difference between two move actions is worse
  than predicting none.

## Results

### E6 (plain counterfactual siblings), 100 updates before being replaced

| update | counterfactual (per-sibling) loss |
|---|---|
| 25 | 2.46 |
| 100 | 2.00 |

Held-out horizon table at update 0 (= E5 at update 3,500):

| h1 | h4 | h16 | h64 | h256 | h1k | h4k |
|---|---|---|---|---|---|---|
| 2.19 | 5.13 | 10.8 | 27.5 | 41.0 | 64.0 | 58.6 |

Hero position error (cells): 1.7, 2.8, 4.2, 8.6, 8.8, 9.4, 9.0.

### E6b

Live effect metrics (training-game forks), share of each edit's true effect
predicted:

| update | move y | move x | attack | Q | W | E | affected F1 | per-sibling loss |
|---|---|---|---|---|---|---|---|---|
| 0 (= E5 @3,500) | −95% | −80% | −2% | +1% | +1% | +1% | — | — |
| 25 | +17% | +11% | +1% | +1% | +0% | +0% | 30% | 2.46 (E6: 2.46) |
| 50 | +31% | +30% | +6% | +6% | +8% | +7% | 35% | 2.19 (E6: 2.16) |

- The per-sibling loss matches E6's, so the effect terms cost nothing there.
- The no-effect loss stays around 0.0002: no hallucinated effects.
- Speed and memory are the same as E6: 0.23 updates/s, 125 GB.

### E6b vs E5, held-out loss by horizon (2026-09-25, ~19:30 PDT)

Both runs start from E5 @3,500. An E6b update takes 4.4× longer (4.49 s vs
1.02 s).

**Matched updates since the shared start:**

| updates since start | run | h1 | h4 | h16 | h64 | h256 | h1k | h4k | teacher-forced val |
|---|---|---|---|---|---|---|---|---|---|
| 500 | E5 @4,000 | 2.18 | 5.22 | 10.6 | 24.2 | 43.0 | 50.9 | 46.0 | 2.96 |
| 500 | E6b @500 | 1.10 | 3.52 | 9.4 | 24.0 | 32.7 | 48.9 | 90.0 | 1.66 |
| 1,000 | E5 @4,500 | 1.93 | 4.73 | 9.7 | 22.0 | 32.8 | 35.4 | 41.6 | 2.80 |
| 1,000 | E6b @1,000 | 1.00 | 2.93 | 8.5 | 21.1 | 31.4 | 45.1 | 90.2 | 1.48 |

**Matched wall-clock time.** E6b @1,000 took ~75 min, which buys E5 about
4,400 updates. The E5 row averages its evaluations at 7,500 / 8,000 / 8,500,
because single E5 evaluations swing ±15–30%.

| | h1 | h4 | h16 | h64 | h256 | h1k | h4k | teacher-forced val |
|---|---|---|---|---|---|---|---|---|
| E5 @~8,000 | 2.39 | 5.13 | 8.4 | 20.4 | 28.9 | 42.9 | 78 | 3.31 |
| E6b @1,000 | 1.00 | 2.93 | 8.5 | 21.1 | 31.4 | 45.1 | 90 | 1.48 |

**Verdict so far:**
- **h1–h4 and teacher-forced val:** E6b is 2–2.4× better. One-step precision
  is fixed, with no more jitter.
- **h16–h1k:** a tie within noise.
- **h4k:** E5 is better. E6b's h4k has stayed at 88–92 at every evaluation.
  h4k uses only a few long held-out games.
- **Why:** long-horizon gains come from rollout practice, and E6b gets about 4×
  less of it per second.
- **Next candidate:** rebalance to fewer counterfactual fork points per
  update (e.g. 1,024), keeping the effect losses.
- **Correction:** E5 did not plateau. After a flat stretch around 5,000–7,000
  it improved again at 8,000–9,500, slowly, on long horizons.

### Update, 22:08 PDT: E6b 3.6 h (update 2,900), E5 5.1 h (update 18,000)

Matched wall-clock time since the split, averaged over each run's last hour
of evaluations:

| | h1 | h4 | h16 | h64 | h256 | h1k | h4k | teacher-forced val |
|---|---|---|---|---|---|---|---|---|
| E5 (16,000–18,000) | 1.95 | 4.46 | 7.3 | 18.8 | 28.1 | 40.5 | 52 | 2.78 |
| E6b (2,000–2,750) | 0.98 | 3.57 | 8.4 | 23.3 | 40.0 | 65.4 | 100 | 1.39 |

- **E5** keeps improving slowly from h16 onwards. Since the split, h64 went
  27.5 → 18.8 and h1k 64 → about 40.
- **E6b** has been flat or worse on long horizons since about 1 h: h64
  around 21–24, h4k about 100, against 59 at its start. Where the
  counterfactuals act, it is still improving: h1 about 1.0, teacher-forced
  val 1.48 → 1.37.
- **E6b action understanding** at update 2,900 (training-game forks): the
  share of each edit's effect it predicts is move 59% / 57%, attack 69%,
  Q 80%, W 86%, E 89%.
- **Verdict:** the counterfactual regime starves rollout practice, and long
  horizons pay for it. Next: E6c, with long rollout steps weighted more (lever
  A) plus more lanes via a data store in CPU RAM (lever B).
