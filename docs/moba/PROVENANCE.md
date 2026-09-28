# MOBA track: provenance

Every number, checkpoint, dataset and sample in the MOBA work, traced to where it
came from. A paper about this work must be writable **from this file alone**:
- without re-running anything;
- without W&B staying up;
- without asking the author.

Claims, with their evidence class, live in [paper/CLAIMS.md](paper/CLAIMS.md). The
rules that keep this true are in [EVIDENCE_RULES.md](EVIDENCE_RULES.md). Where to
find everything is in [MAP.md](MAP.md).

Same standard as `../neural-simulators/PROVENANCE.md` and `../zelda-rl/PROVENANCE.md`.
Started 2026-09-27; the earlier work was provenanced retroactively from its W&B
history, local logs and session records. Where the evidence was lost, this file
says so.

**Evidence classes**, as in the sibling repos:
- **M** measured, with a recoverable artefact in this repo;
- **Q** qualitative: a watched replay, an author observation;
- **I** inferred from M or Q;
- **A** author-confirmed;
- **U** unverified. Do not put it in a paper without closing it first (§6).

## The lineage and the question

| repo | question | status |
|---|---|---|
| `carla-imitation-learning` | can a policy learn to act in a world? | finished paper |
| `neural-simulators` | can we simulate a world? | finished paper |
| **this track (MOBA)** | **can we learn a world well enough to teach an agent in it?** | in progress |
| `zelda-rl` | the same question, in Zelda (Ocarina of Time) | in progress |

| track | question | status | doc |
|---|---|---|---|
| **WM** (world model) | can a learned model roll a full 10-hero MOBA game forward from all heroes' actions, over whole games? | in progress. Long-horizon rollouts are stable (M); jitter and tower health are open | [WM_ARCHITECTURE.md](WM_ARCHITECTURE.md), [experiments/](experiments/README.md) |
| **RL** (dream PPO) | can a policy trained only inside that model play the real game? | **first attempt negative** (M): 0 real wins, below an untrained policy | [experiments/E7-dream-ppo.md](experiments/E7-dream-ppo.md) |

The tracks are provenanced separately. A claim about one is not evidence about the other.

## 1. Environment and code

| item | value | class |
|---|---|---|
| Game | PufferLib Ocean MOBA, `third_party/pufferlib` at `42f70d69` (upstream, unmodified) | M |
| Deterministic wrapper | `moba/moba_game.h`: SplitMix64 random stream per game in the state, non-resetting step, restore. Documented in [DATA.md](DATA.md) | M |
| Markov test | `build/moba_collect markov`: a state restored mid-game continues byte-for-byte (0/99 divergence) | M |
| World model | `mobawm/wm/{model,data,train}.py`, TokenWM 0.66M params. [WM_ARCHITECTURE.md](WM_ARCHITECTURE.md) | M |
| **Rules inside the world model** | Spawns, creep constants, level from XP, level-derived stats, counters, targets and clamps are **hand-coded rules R1–R10**, not learned. Every WM result includes them. [WM_ARCHITECTURE.md § Rules vs learned](WM_ARCHITECTURE.md#rules-vs-learned-what-the-model-does-not-predict) | M |
| Counterfactual fork generator | `moba/moba_fork.c`, `mobawm/wm/cf.py`. Byte-exact against the corpus. [COUNTERFACTUALS.md](COUNTERFACTUALS.md) | M |
| Dream PPO | `mobawm/rl/dream_ppo.py` | M |
| Real-game arena | `moba/moba_arena.c`, `mobawm/rl/arena.py`. It reuses the collector's `net_act` / `scripted_action` unchanged | M |
| Replay viewer, the microscope | `tools/moba_viz.py`: the real game next to the model on the same actions, with hover stats, cast/death flashes and a headless `Replay` API. [VIEWER.md](VIEWER.md). Recordings: `tools/moba_viz_capture.py` → `media/moba/<timestamp>/` (videos + PNGs, local only, md5s in its MANIFEST.json) | M |

### Code commit status (read this before re-running anything)

| code | committed? | where the exact version is |
|---|---|---|
| E0–E4 world model, baseline v2, viewer | yes, up to `efe1003` / `ff9b337` / `bfa68da` / `9e5ea3b` | git |
| E5 rollout training (`--rollout`, `--init`, horizon metric) | yes, `e51dc11` | git |
| E5-continuation additions (`--store-cpu`, rubric metrics), E6/E6b (`CFBatcher`, effect losses, `cf.py`, `moba_fork.c`), dream PPO, arena, S3 persistence | **no, deliberately.** New architectures stay uncommitted until the author says commit | `s3://puffertank/moba/runs/code_snapshots/code_snapshot_20260927_1323.tar.gz` (md5 `6975755a1b7e`, [MANIFEST](../../artifacts/moba/MANIFEST.json)): HEAD `dad8c7e` + tracked diff + untracked files |
| E14: `dream_ppo --arch puffer` (recurrent PufferLib policy), `--role-heads`, reward shaping flags, arena recurrent memory | **no, deliberately** | `s3://puffertank/moba/runs/code_snapshots/code_snapshot_20260928_0700.tar.gz` (md5 `1975f63caedb`, MANIFEST) |
| E11–E13: `--held2-games`, anonymous-mmap loader, `dream_ppo --opponent mix`, `collect_policy_games --matchups/--slice`, CPU pods | **no, deliberately** (pod.py: committed) | `s3://puffertank/moba/runs/code_snapshots/code_snapshot_20260928_0215.tar.gz` (md5 `2b46ba452a3e`, MANIFEST) |
| E8–E11: collector `random` controller, planner (`plan_mpc.py` realtime/batched), XP clamp flags, `--held2-games`, audit/Dyna/calibration scripts | **no, deliberately** | `s3://puffertank/moba/runs/code_snapshots/code_snapshot_20260928_0015.tar.gz` (md5 `bdb61cfa11ac`, MANIFEST): HEAD `173cfc1` + tracked diff + files |

## 2. Datasets

| dataset | contents | location | size | class |
|---|---|---|---|---|
| `mix1k` | 1,000 games. Radiant/dire controllers cycle net/net, net/script, script/net, by game index. Logs = seed + actions; dense = the full state every tick, regenerated from the logs (hash-verified) | `s3://puffertank/moba/mix1k/{logs,dense}/` | 100 log files (30 MB) + 100 dense files (15 GB) | M |
| mix1k split | train games 0–899 (**2,395,549** transitions); held-out games 900–999 (**267,768** transitions). Split by game, never by tick | `mobawm/wm/train.py --train-games 0-899 --held-games 900-999` | — | M |
| mix1k controllers | hero-ticks 64.2% net, 35.8% script; ε = 0 (no random actions) | `results/moba/data/controller_census.json` | — | M |
| `mixrand1k` | 1,000 games, indices 1000–1999, seeds continue mix1k's; 3,000-tick cap; 2,608,592 ticks. Contiguous matchup blocks: random/net 1000–1159, net/random 1160–1329, random/random 1330–1659, random/script 1660–1829, script/random 1830–1999. Hero-ticks: random 69.0%, net 12.7%, script 18.3%. `random` = uniform per action head (collector controller 3) | `s3://puffertank/moba/mixrand1k/{logs,dense}/` | 100 logs (36 MB) + 100 dense (~14 GB) | M |
| mixrand1k split (E11) | **held-out random games** (whole files, last of each block): 1140–1159, 1310–1329, 1620–1659, 1810–1829, 1980–1999 (100 games); the rest trains. mix1k 900–999 stays held out | `train.py --held2-games` | — | M |
| `dyna1` | 100 real games of the two P3 dream-PPO policies vs the shipped net (ids 5000–5099), recorded as standard logs, replay-verified with 0 mismatches | `s3://puffertank/moba/dyna1/`; `results/moba/dyna/dyna1_games.json` | 198,131 ticks | M |
| `dyna2` | 810 real games of the P3 dream policies vs random / the net / each other (DATA.md), collected on a 32-vCPU RunPod CPU pod with `scripts/moba/collect_policy_games.py --matchups --slice`, replay-verified (0 mismatches in all 23 parts) | `s3://puffertank/moba/dyna2/{logs,dense,games.json}` | 81 + 81 files, 12.3 GB | M |
| E12 split | train: E11's list + dyna2 minus 6260–6269, 6530–6539, 6800–6809 (reserved); held-out: mix1k 900–999 and the E11 random held-out | `train.py` args in E12 record | — | M |
| `mix100` | games 0–99, identical to mix1k 0–99 (the same seeds). Local only | `data/moba/mix100/` (gitignored) | 2.7 MB logs, 1.4 GB dense | M |
| Counterfactual stream | not stored. Regenerated from (game, tick, edit seed) by `moba_fork.c`; only training games are forked | — | ~240M distinct single-edit transitions available (estimate, **I**) | M (generator) / I (count) |

**Samples** (committed, with field names): `samples/moba/state_excerpt.{npz,json}` (16
ticks of held-out game 900) and `samples/moba/counterfactual_excerpt.{npz,json}` (one fork
point, 8 siblings). Regenerate with `scripts/moba/make_samples.py`.

## 3. WM track

### 3.1 Phases

All W&B runs are in `drift-ai-models/moba-wm`, exported to `results/moba/<run_id>/`.
Cite the directory, not the link.

| exp | dates (UTC) | what | runs → `results/moba/` | wall clock | verdict | doc |
|---|---|---|---|---|---|---|
| E1 | 2026-09-25 15:14 | first TokenWM smoke runs, mix100 | `vuzjgdh8` (crashed), `ox4cx47i` | 0.2 h | pipeline works | EXPERIMENTS.md (legacy) |
| E2 | 09-25 15:51 | first full mix1k run | `asdg1vl1` | 0.5 h | trains; one-step metrics only | EXPERIMENTS.md |
| E3 | 09-25 16:55 | change gate + event sampling | `65j98z3b`, `s3ilbx02` (crashed) | 0.4 h | gate helps rare fields | EXPERIMENTS.md |
| ablations 1–17 | 09-25 17:27–22:24 | 5-min A/B rounds: gate, sparse gate, action tokens, geometry, navgrid, walk isolation, health, regen, cast input | 36 runs, groups `ablate1`…`ablate17` | 5 min each | walking needed ×10 loss (loss competition); action tokens, geometry, navgrid neutral or negative. Metric later found misleading (cooldown countdown) | EXPERIMENTS.md |
| **E4** | 09-25 22:52 | baseline v2, full mix1k, 1 h, teacher-forced, batch 768, H200 | `kxe2bh1k` | 1.0 h | one-step good; open loop flies apart (hero error 92 cells @1k) | [E4](experiments/E4-baseline-v2-1h.md) |
| **E5** | 09-25 23:59 → 09-26 ~06:00 | autoregressive rollout training from E4 final, 20,750 updates | `93m7sktq` (main); `zll4trqi` (aborted launch) | 5.97 h | long horizons 5–12× better than E4; one-step ~2× worse; jitter | [E5](experiments/E5-rollout-training.md) |
| **E6 / E6b** | 09-26 01:16 → ~06:00 | E5 @3,500 + real-sim counterfactual siblings (32k/update); E6b adds effect auxiliary losses | `uo6sal6q` (E6, 75 updates), `d9xnnj0a` (E6b, 3,550) | 0.1 + 4.5 h | E6b: h1 about 2× better than E5 and action effects learned; long horizons stalled at matched wall clock | [E6](experiments/E6-counterfactual-siblings.md) |
| E8 | 09-27 | dream-vs-real reward audit; XP ×100 and clamp-off smoke tests | xpsmoke runs (local, `results/moba/local_logs/xpsmoke_*`) | ~0.5 h | XP hallucinated off-distribution; neither fix works | [E8](experiments/E8-xp-hallucination.md) |
| E9 | 09-27 ~23:40 | Dyna smoke: + 100 real P3-policy games, H100 NVL | `usrc41qc`, `tnnlmx2e` | 2 × 10 min | 3–7× less XP hallucination vs control | [E9](experiments/E9-dyna-smoke.md) |
| E10 | 09-27 | WM planner at 60 fps, 120 games vs net (eval only) | — (`results/moba/planner/`) | — | 0 wins, ≈ random | [E10](experiments/E10-wm-planner-60fps.md) |
| E11 | 09-28 00:20 → 01:51 | E5 @20,500 resumed on mix1k + mixrand1k, 2,300 updates | `mlbh5d3l` | 1.5 h | random-play hallucination fixed; expert h1/h4096 worse | [E11](experiments/E11-random-data.md) |
| E12 | 09-28 ~02:00 → ~04:45 | E11 @2000 + dyna2 | `xfldazic` | running | — | [E12](experiments/E12-dyna2.md) |
| E13 | 09-28 01:28 → ~04:45 | dream PPO in E11 @1500: P4, P4b (4.5 min), P5 (100×) | `msjwyq7r`, `ahoexos3`, `gqwfjvhm`, `jfw1p3b5` | 2 × 4.5 min + running | first dream policies above random in the real game; 0 wins | [E13](experiments/E13-dream-ppo-e11.md) |
| E14 | 09-28 ~05:00 → ~07:00 | dream PPO policy ablations P6–P11 on one H200 (arm C = best) | 18 runs, groups `P6-roleheads` … `P11-reward-sweep` | ~1 h GPU | recurrent PufferLib policy best; over-optimisation hurts; shaping negative | [E14](experiments/E14-dream-ppo-policy.md) |
| E5 continued | 09-27 03:20 → ~05:45 | E5 from 20,500, H100 NVL, `--store-cpu` | `hm9mnnoy` (OOM), `soeosr9h` (1,700 updates) | 0.93 h | best one-step E5 so far (h1 1.51 @+1,500); **all its checkpoints lost** | [E5](experiments/E5-rollout-training.md) |

Figures and tables, regenerated with `results/moba/make_figures.py`:
- `results/moba/figures/horizon_loss.png` + `tables/horizon_loss.md`
- `figures/horizon_over_training.png` + `tables/horizon_over_training.md`

### 3.2 Resume lineage

```
E4 kxe2bh1k (from scratch, 3,877 updates)
 └─ final model.pt ──► E5 93m7sktq (--init; Adam fresh) ── 20,750 updates
                        ├─ @3,500 model.pt ──► E6 uo6sal6q (75) ─ replaced by ─► E6b d9xnnj0a (--init @3,500; 3,550 updates)
                        └─ @20,500 model.pt ─► E5-continued soeosr9h (--init; Adam fresh; 1,700 updates)
```

**Step-axis discontinuities:**
- Every `--init` restarts `_step` at 0 and restarts Adam, with a 100-update warm-up.
- E6b step S = E5-equivalent update 3,500 + S.
- The continuation's step S = E5 update 20,500 + S. Its step-0 evaluation reproduces E5 @20,500 exactly (h1 1.705 … h4k 36.84).
- An E6b update does about 4.4× the work of an E5 update (4.49 s vs 1.02 s). Compare the two at matched wall clock as well as matched updates.

### 3.3 Checkpoints

md5s are from [artifacts/moba/MANIFEST.json](../../artifacts/moba/MANIFEST.json). Every
kept file has a byte-identical S3 copy under `s3://puffertank/moba/runs/` (ETag = md5 checked).

| file | md5 | bytes | what | class |
|---|---|---|---|---|
| `runs/token_wm/e4_baseline_v2_1h/model.pt` | `b66fae7234b9` | 2,673,474 | E4 pulled mid-run, **~update 3,000**. Not the final checkpoint | M (file) / I (the update number, from pull time) |
| `runs/token_wm/e4_baseline_v2_1h/stats.pt` | `7c1a1c5b9942` | 7,994 | E4 stats + config. The same file serves E5 and E6 | M |
| `runs/token_wm/e5_rollout_at3500/model.pt` | `a4992005f6d5` | 2,673,474 | E5 @3,500, the start of E6/E6b | M |
| `runs/token_wm/e5_rollout/model.pt` | `a61d64aed9c2` | 2,673,474 | **E5 @20,500, the best surviving long-horizon model** | M |
| `runs/token_wm/e6b_cf_effect/model_00{1,2,3}000.pt` | `56e8acb31a0e` / `8cfd5945017c` / `8880a501c96c` | 2,679,074 each | E6b @1,000 / 2,000 / 3,000 | M |
| `runs/token_wm/e6b_cf_effect/model.pt` | `98e71c23a57e` | 2,673,474 | **E6b @3,500, the best one-step model** | M |

**Lost:** checkpoints lived only on pod disks until 2026-09-27, and the pods were
deleted externally. Gone for good:
- E4's final checkpoint (3,877) and its numbered checkpoints;
- E5's numbered checkpoints (every 2,000 updates);
- all of E5-continued's checkpoints, including its best (+1,500).

Their **metrics** survive in the W&B exports. Since 2026-09-27, `tools/pod.py train`
uploads every checkpoint to S3 as it is written ([RUNPOD.md](RUNPOD.md)).

### 3.4 Cost

RunPod H200 (secure) at $4.59/hr; H100 NVL at $3.19/hr.

| item | pod-hours | approx. cost |
|---|---|---|
| E1–E4 and ablations (pod `ibkmxiyh4t8vmj`) | ~8 h | ~$37 |
| E5 | ~6.5 h | ~$30 |
| E6/E6b (pod `hlqd4lm2nqdkmq`, while E5 also ran) | ~4.7 h | ~$22 |
| Two H200 pods that never started (deleted) | ~0.6 h | ~$3 |
| E5-continued (H100 NVL) | ~2.5 h | ~$8 |

All costs are **I**: from the recorded rates and run times, not from invoices, and
they exclude idle pod time between runs, which was not logged.

## 4. RL track (dream PPO)

W&B project `drift-ai-models/moba-dream-rl`, backfilled from local `log.jsonl` by
`scripts/moba/wandb_backfill_dream.py`; the runs logged only locally at the time.
The raw logs are also in `results/moba/dream/`.

| exp | when | what | runs → `results/moba/` | verdict |
|---|---|---|---|---|
| P0 | 2026-09-25, late evening PDT | first self-play in E5's dream, Mac MPS, 5 min, 317k agent steps | `x18k5q1z` | deaths halved, critic learned (EV 0.8), reward flat; the episode phase confounds per-update reward |
| P1-mac | same night | LR × minibatch sweep, 6 configs in parallel on the Mac GPU | `is9phlur jd4ghbkn za7afmv0 8z73kb8d f5m7o9fk s978nel3` | **inconclusive**: 2–4 updates each |
| P1-h200 | same night, E5 pod | the same sweep, 6 in parallel, 4.5 min, about 2.4M agent steps / 30 updates each | `cfuvqm95 k8m39dqw 3xhtvpo3 v22f5ut6 m4zqm6gl xegabn7s` | every config's **dream** reward rises 2–3×; best slope lr 3e-4, 2 minibatches; single seed |
| **P2** | after P1 | the best P1 policy (`v22f5ut6`) in the **real C game**, 120 games | `81jwi4lw` (+ `dream/arena_real_game_lr3e-4_mb2.jsonl`) | **negative:** 0 wins; lower return than an untrained policy |
| **P3** | 2026-09-27 | self-play rerun vs **shipped-net opponent** inside the dream, 4.5 min each in parallel, H100 NVL pod `k9rbiphntinbl2` | `1q729jk7`, `iluni10r` (+ `dream/arena_real_game_p3_*.jsonl`) | **negative:** 0 wins; both below random vs net; more dream reward → worse real play |

Policy checkpoints (md5 in MANIFEST, byte-identical in `s3://puffertank/moba/runs/dream_ppo/`):
- P0: `runs/dream_ppo/e5_selfplay_5min/policy.pt` `9af7af55dec6`;
- the best P1 policy: `runs/dream_ppo/sweep1_gpu/lr3e-4_mb2/policy.pt` `6c4aeb7e8d32`;
- the other five P1-h200 finals: see MANIFEST.

**The dream all P0–P2 policies trained in** was E5 @20,500 (`a61d64aed9c2`), including the
world-model rules R1–R10. Figures:
- `results/moba/figures/dream_ppo_sweep.png`
- `results/moba/figures/arena_real_game.png`

## 5. Qualitative evidence (class Q)

The author watched each of these in the replay viewer (held-out game 900 from tick 500).
These are observations, not rates.

| what | checkpoint | source |
|---|---|---|
| Heroes far more stable than pre-rollout training: no flying off the map, broad movement right, no visible degeneration over time. Creeps and neutrals move very naturally (A+ trajectories, not the real positions). Abilities predicted more often. Failures: tower damaged in one panel but not the other | E5 @500 | [E5 record](experiments/E5-rollout-training.md) |
| E5 vs E6b side by side; tower health tracks better in E6b | E5 @18,000–20,500, E6b @1,000–3,500 | session record |

**Viewer recordings** (local, gitignored, not in S3). The set is
`media/moba/20260927_1332/` (manifest md5 `a94e4787688e95187f64ec47415d46f2`):
- **Models:** E4 ~3,000, E5 @3,500, E5 @20,500 and E6b @3,500, each checkpoint
  identified by md5 against `artifacts/moba/MANIFEST.json`.
- **Data:** held-out game 900 from tick 500, 1,000 ticks.
- **Contents:** 12 videos (rollout, ghost, hover tour) and 52 PNGs.
- **Per file:** the metrics at the steps shown, a finding and its claim ids. The
  Figure 1 candidate is `E4_teacherforced_u3000_ghost_step0300.png`.
- **Rebuild the metadata** without re-recording:
  `tools/moba_viz_capture.py --backfill <dir>`.

## 6. Open items: the U list

| item | why it is not M | track |
|---|---|---|
| E4 control horizon table (E4 checkpoints 500 … final) | Computed on the pod by `mobawm/wm/openloop_eval.py`; the output file was lost with the pod. The numbers survive only as transcribed into the E5 record | WM |
| Single-rollout viewer numbers (E4/E5/E6b hero error on game 900 @ tick 500) | One rollout each, printed by the viewer, not saved. Regenerable only for the kept checkpoints | WM |
| "Not an attractor" check (E5 @500: real vs wrong actions) | The E5 @500 checkpoint is gone, and the script that ran it was a scratch file that wasn't kept. It needs re-running on E5 @20,500 with a committed script | WM |
| Every single-seed comparison (E5 vs E6b; the dream sweep) | One seed each; E5's own evaluations swing ±15–30% | both |
| Long-horizon numbers at h4096 | Only a few held-out games are longer than 4,104 ticks | WM |
| The ~240M counterfactual count | An estimate (states × ~100 single edits), not a count of distinct outcomes | WM |
| Why dream PPO fails in the real game (model exploitation vs policy/opponent shift) | Hypothesis only; the diagnosis (replay the policy's real actions through the WM) was not run | RL |
| Costs | From rates and run times, not invoices | both |

## 7. Regenerate

```sh
.venv/bin/python scripts/moba/wandb_organize.py --apply     # W&B groups/tags/notes (idempotent)
.venv/bin/python scripts/moba/wandb_backfill_dream.py       # dream-PPO runs -> W&B (skips existing)
.venv/bin/python scripts/moba/export_wandb.py               # W&B -> results/moba/<run_id>/ + INDEX.json
.venv/bin/python scripts/moba/make_manifest.py              # artifacts/moba/MANIFEST.json (md5 + S3 check)
.venv/bin/python scripts/moba/make_samples.py               # samples/moba/
.venv/bin/python results/moba/make_figures.py               # results/moba/figures + tables
tools/pod.py s3ls <run> / s3pull <run>                      # a run's persisted checkpoints
```
