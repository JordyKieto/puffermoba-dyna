# E4 — baseline v2, 1 hour

**Question:** how far does the adopted baseline get with an hour of training
at the fast big-batch recipe, and what does its rollout look like next to the
real game?

- **Status:** done. 3,877 updates in 60 min; final checkpoint `model.pt` (23:52 UTC). The end-of-run report was cut off when the session was stopped to start E5, so there is no `report.json`.
- **wandb:** https://wandb.ai/drift-ai-models/moba-wm/runs/kxe2bh1k

## Setup

```sh
.venv/bin/python tools/pod.py train e4_baseline_v2_1h -- --baseline \
    --data data/moba/mix1k/dense --train-games 0-899 --held-games 900-999 \
    --minutes 60 --batch 768 --lr 1e-3 --bf16 \
    --log-every 50 --val-every 250 --val-batches 4 --ckpt-every 500 --wandb
```

- **Model:** baseline v2: change gate on everything but movement, gate
  yes/no loss ×0.2, gated value loss averaged over all ticks, hero walking
  ×10. 0.66M params.
- **Data:** mix1k, 900 training games (2.4M steps, 51.5 GB in VRAM); held-out
  games 900–999.
- **Recipe:** batch 768, bf16, lr 1e-3, about 1 update/s.
- **Checkpoints** on the pod at `runs/token_wm/e4_baseline_v2_1h/`:
  - `model.pt`, rewritten every 250 updates;
  - `model_<step>.pt`, kept every 500 updates.

## Results

- **Final teacher-forced validation loss:** 1.108 at update 3,877.
  - Per kind: hero 0.660, creep 0.428, neutral 0.011, tower 0.009.
- **Open loop, one viewer rollout** (held-out game 900, from tick 500). Hero
  position error in cells:

  | ticks ahead | update 500 | update 3,000 | copying the start state |
  |---|---|---|---|
  | 16 | 2.24 | 6.03 | 10.95 |
  | 64 | 16.47 | 24.55 | 40.59 |
  | 300 | 54.20 | 44.07 | 45.44 |

  More teacher-forced training made the mid-horizon rollout *worse* on this
  game. It is only one rollout; the E4 control in
  [E5](E5-rollout-training.md) measures it properly.

## Watch it

```sh
.venv/bin/python tools/moba_viz.py --run runs/token_wm/e4_baseline_v2_1h \
    --dense data/moba/mix1k/dense/games.0000900.mbs --game 0 --start 500 --horizon 300
```

See [../VIEWER.md](../VIEWER.md) for pulling the latest checkpoint.
