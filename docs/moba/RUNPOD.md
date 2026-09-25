# RunPod: MOBA world-model training

One GPU pod, sized so the **whole training set lives in VRAM**. Windows are
sampled on the GPU by index arithmetic (`mobawm.wm.data.Store`), so training
never waits on a dataloader. Workflow and conventions are adapted from
zelda-RL's measured fleet runbook (`zelda-RL/docs/RUNPOD.md`).

## Commands

Run everything from the repo with the project venv (it has boto3 and requests):

```sh
.venv/bin/python tools/pod.py template              # register RunPod template "puffertank-moba-wm" (once)
.venv/bin/python tools/pod.py create --confirm      # H200, fallback H100 NVL; waits for SSH
.venv/bin/python tools/pod.py bootstrap             # code push, apt, pip, build moba_collect
.venv/bin/python tools/pod.py data --corpus mix100  # S3 logs -> pod -> full-state files -> S3
.venv/bin/python tools/pod.py train smoke -- --train-games 0-90 --held-games 91-99 --minutes 10 --wandb
.venv/bin/python tools/pod.py status smoke          # tmux, GPU memory/util, log tail
.venv/bin/python tools/pod.py sync                  # runs/token_wm/ back to the Mac
.venv/bin/python tools/pod.py kill --confirm        # ALWAYS: the pod bills until deleted
```

`train NAME -- ARGS` pushes the current code, writes
`runs/token_wm/NAME/launch.sh` on the pod, and starts it in tmux session
`NAME`. Output goes to `runs/token_wm/NAME/{train.log,log.jsonl,report.json,model.pt}`.

## Why this shape

- **Pod size.** The full-state entity tensor is 206 × 26 fp32 = 21 KB per
  step. The 100-game corpus (256k steps) is about 5.5 GB. An **H200 (141 GB)**
  holds about 25× that in fp32, roughly 6M steps or 2,400 games, with room
  for the model. That leaves headroom to scale the corpus before touching the
  data path. The fallback is an H100 NVL (94 GB). Beyond that, store
  non-position fields as fp16/int16 (about 2×) before sharding.
- **Data in S3, pulled at datacenter speed.** The Mac's uplink is about
  0.7 MB/s (zelda-RL, measured), so nothing large goes Mac → pod.
  - The **game logs** are the source of truth: ~11 bytes/step, 2.9 MB for
    100 games. They live at `s3://puffertank/moba/<corpus>/logs/`.
  - The pod regenerates the **full-state files** from the logs with
    `moba_collect replay`. That is bit-exact: every replay checks the state
    hash and the obs function.
  - The pod then pushes them to `s3://puffertank/moba/<corpus>/dense/`, so
    the next pod downloads them instead.
  - All transfers use presigned URLs made on the Mac, so the pod never holds
    AWS credentials.
- **Template.** `puffertank-moba-wm` fixes the image
  (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, CUDA torch
  preinstalled), 80 GB container disk, 300 GB volume at `/workspace`, and
  SSH. `create` uses the same spec directly.
- **Persistence.** Everything that must survive a pod restart is under
  `/workspace`: code, data, runs. apt packages are on the container disk; after
  a restart, run `bootstrap` again (fast, nothing to rebuild on the volume).
- **Secrets.** `WANDB_API_KEY` reaches the pod as a mode-600 `.pod_secrets`
  file, sourced by `launch.sh`. It never goes on a command line.

## Scaling the corpus

Collection is CPU work and replay is CPU work. On the pod, both run on all
cores:

```sh
tools/pod.py ssh "cd /workspace/puffertank && DENSE=1 tools/moba_collect.sh 1000 50 \$(nproc) data/moba/mix1k"
```

Then push the new logs to S3. They are the source of truth and small, so pull
them to the Mac or push them from the pod. Train with
`--data data/moba/mix1k/dense`.

## Measured (2026-09-25, pod `ibkmxiyh4t8vmj`)

| item | result |
|---|---|
| Pod | 1× NVIDIA H200 (143,771 MiB), **160 vCPU, 2 TB RAM**, secure cloud, **$4.59/hr** |
| Create → SSH ready | **33 s** |
| Bootstrap (apt + code + pip + build `moba_collect`) | **36 s** warm, ~70 s first time; code push 5.7 MB in 10–14 s |
| `data --corpus mix100` | **121 s**: logs S3 → pod, 100 games replayed into 1.4 GB of full state on the pod, 8 files pushed to S3 |
| Cross-platform determinism | logs recorded on the Mac (ARM) replay on the pod (x86 Linux) with **0 hash mismatches, 0 obs mismatches** over all 100 games |

### Pitfalls hit on the first pod

1. **The image has no `rsync`.** apt must run before the first code push
   (`bootstrap` does this).
2. **`tar` fails on `/workspace`**: the network volume refuses `chown`.
   `tools/build.sh` extracts raylib with `--no-same-owner`.
3. **Piping a build through `| tail` hides its failure.** `bootstrap` now
   fails loudly with the build log's last 20 lines.
