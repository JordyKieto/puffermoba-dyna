# Artifacts: runs, checkpoints, metrics

Long-form reference. The story is in [README.md](../README.md); the numbers are
in [EXPERIMENTS.md](EXPERIMENTS.md); this is where to look up a specific run.

Everything below is reproducible from a clean checkout: the corpus is
regenerated deterministically from a seed, and every checkpoint carries the
state-contract hash it was trained under.

## Metrics

Two sinks, both always written:

- **`logs/<run_name>.jsonl`** — one JSON line per logged step, always on. Every
  line carries `wall_clock_s`, `samples_seen` and `samples_per_sec` alongside
  the losses, so "how long and how much data" is answerable without rerunning.
- **W&B** — project `race-wm`, entity `drift-ai-models`. Opt-in per run with
  `--wandb`.

> **Note on coverage.** The sweeps in EXPERIMENTS.md were run *without*
> `--wandb`, so they exist only as local `logs/*.jsonl`. The integration is
> verified working (`wandb_smoke_test`), but the historical runs are not in
> W&B. Pass `--wandb` from here on.

## Corpus

`data/corpus`, built by `tools/make_dataset.sh corpus 20`, seed 42, frameskip 4,
`max_episode_steps` 2000. ~1.16M transitions, 161 MB, generated in under a
second.

| split | env_idx | tracks | transitions |
|---|---|---|---|
| train | 0–95 | 96 | 817,669 |
| val | 96–111 | 16 | 147,656 |
| holdout | 112–127 | 16 | 190,528 |

Four slices per split (`expert`, `whisker`, `noisy`, `random`) — see
[DATASET.md](DATASET.md). `data/corpus/manifest.json` records the exact
generation parameters.

S3: `s3://puffertank/race-wm/` via `python3 -m racewm.s3sync push data/corpus`.

## Checkpoints

Each is a directory under `checkpoints/` with `latest.pt`, `best.pt` and a
`manifest.json` holding config, metric history and the contract hash.

### World models

| run | what | headline |
|---|---|---|
| `wm_onestep_v1` | one-step, 25 epochs, 818k transitions | @20 9.3× persistence, **@100 diverges (0.88×)** |
| `wm_multistep_warm` | K=10 rollout loss, warm-started from above | **@100 17.5×**, @5 107× |
| `wm_ens_1`, `wm_ens_2` | ensemble members for planner pessimism | started, parked with planning |

### Policies and heads

| run | what | headline |
|---|---|---|
| `bc_expert_only` | BC on the expert slice | **11.02 return** — beats the expert it cloned |
| `bc_whisker_only` | BC on whisker | 99.67% val acc, matches whisker online |
| `bc_expert_priv` | BC with track geometry | 95.23% val acc vs 84.70% — the information ceiling |
| `bc_expert_whisker` | BC on the mix | 9.17 — mixing hurt BC |
| `termination` | P(terminate \| s, a, geometry) | **val AP 0.926** vs 0.014 base rate |
| `critic_geom` | TD(0) on real transitions, with geometry | V gap 0.441 (0.165 without geometry) |
| `critic` | same, state only | superseded by `critic_geom` |

### Imagination sweeps

| family | swept | outcome |
|---|---|---|
| `jointv2_frozen`, `jointv2_joint` | WM frozen vs still training | 5.493 → **7.568** (+38%) |
| `ent_*` | entropy 0.0 / 0.003 / 0.01 / 0.03 | no trend — not a lever |
| `anchor_*` | critic anchoring 0.0 → 5.0 | peaks **8.973** at 2.0, saturates |
| `bck_*` | actor KL 0.0 → 100.0 | peaks **11.228** at 0.3 |
| `seed_1..3` | best config, 3 extra seeds | 11.237 ± 0.16 vs BC 11.016 |
| `budget_*` | the A/B/C/D real-vs-imagined study | see EXPERIMENTS.md E6 |

### Exported policies

`.rpo` files are flat weights with input normalization baked in, loadable by
`race_eval --policy` and by `racewm.leaderboard`. Written by
`racewm/export_policy.py`, which verifies the export reproduces the torch
argmax before returning.

## Evaluation cost

Worth knowing before designing a sweep. 3,200 holdout episodes, 10 threads:

| agent | time | note |
|---|---|---|
| scripted bot | **0.21 s** | pure C, no inference |
| policy, 128×3 (default) | **4.6 s** | NEON dot product |
| policy, 256×3 | 11.7 s | same return as 128×3 — oversized |
| policy, 256×3, scalar loop | 27.2 s | before vectorization |
| planner (beam6) | ~40 s | ~1.2 ms per decision, ~90 model passes each |

Two fixes got policy eval from 27.2s to 4.6s with **no** change in return:
NEON vectorization in `race_policy.h`, and shrinking the default net after
measuring that 128×3 matches 256×3 (64×2 does not).

**Evaluation was never the slow part.** Long waits in this project were model
training and planner rollouts; planning costs ~150× a policy forward pass per
decision, which is why it is parked.

## Reproducing a result

```bash
git submodule update --init --recursive
tools/build.sh eval && tools/build.sh play && tools/build.sh capi
tests/run_all.sh --full                       # all gates incl. Markov (~7 min)

tools/make_dataset.sh corpus 20
python3 -m racewm.train_wm --corpus data/corpus --epochs 25 --run-name wm_onestep_v1
python3 -m racewm.train_wm_multistep --corpus data/corpus --rollout-k 10 \
    --init-from checkpoints/wm_onestep_v1/best.pt --run-name wm_multistep_warm
python3 -m racewm.train_termination --corpus data/corpus
python3 -m racewm.train_critic --corpus data/corpus --run-name critic_geom
python3 -m racewm.train_bc --corpus data/corpus --slices expert --eval

python3 -m racewm.train_joint --mode joint --anchor-coef 2.0 --bc-coef 0.3 \
    --bc-init checkpoints/bc_expert_only/best.pt --eval
python3 -m racewm.leaderboard
```
