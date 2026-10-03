# MOBA: evidence rules

The rules that keep [PROVENANCE.md](PROVENANCE.md) and [paper/CLAIMS.md](paper/CLAIMS.md)
true. The same standard as `../zelda-rl/AGENTS.md` §27 and
`../neural-simulators/PROVENANCE.md`. Each rule names the incident that made it a rule.

## 1. Every result is provenanced as it happens, or it is lost

At the moment a number is produced, write down:
- its **run id** and `results/moba/<run_id>/` directory;
- the **checkpoint md5** (`scripts/moba/make_manifest.py`);
- the **resume lineage** (`--init` of what, at which update);
- **step-axis discontinuities** (every `--init` restarts `_step` and Adam);
- **wall clock and cost**;
- **what the number is not** (single seed? a training-set metric? one rollout?).

*Incident:* the E4 control table and the "not an attractor" check were computed in
scratch files on a pod and survive only as transcriptions (PROVENANCE §6, class U).

## 2. Checkpoints persist off the pod automatically, verified

`tools/pod.py train` gives the run an S3 upload policy, and the trainer uploads every
checkpoint, stats file and log the moment it is written, verified by MD5
(`mobawm/s3persist.py`, `../RUNPOD.md`). Never rely on pulling from a pod later.

*Incident:* 2026-09-27. The pods were deleted externally. Lost for good: E4's final
checkpoint, E5's numbered checkpoints, and all of E5-continued, including its best.

## 3. W&B is organised, then exported; documents cite the export

- **Organise:** every run gets an experiment group, tags (track, regime, outcome)
  and a note pointing at its record (`scripts/moba/wandb_organize.py`, the mapping in
  code).
- **Log to W&B from the start:** runs that logged locally only are backfilled
  (`scripts/moba/wandb_backfill_dream.py`).
- **Export:** everything goes to `results/moba/<run_id>/`
  (`scripts/moba/export_wandb.py`). CLAIMS cites the directory, never a link.

*Incident:* the dream-PPO runs logged only to local `log.jsonl`. The main runs had no
groups or tags, so finding "which E5 was the real one" meant retracing.

## 4. Numbers are generated, not typed

Figures and tables come from `results/moba/make_figures.py`, reading committed files
only. A number in a doc must match a generated table. If they disagree, the doc is
stale.

## 5. Evidence classes on every claim, and retractions stay in the file

Every CLAIMS row carries M / Q / I / A / U. "Not claimed" and "U → withdrawn" rows
stay. When a measurement kills an earlier claim, say so in the next message and
record the retraction next to the original, with what survives.

*Incident:* the ablation-era "better than copying / extra error with wrong actions"
numbers were mostly the cooldown countdown. They are recorded as misleading in the
legacy log, not deleted.

## 6. No hand-coded game logic in the world model without an explicit yes

Every rule in the dream's state update is a design decision. List it in
`../WM_ARCHITECTURE.md` § Rules vs learned, and state it next to every result that
depends on it.

*Incident:* spawns, level, level-derived stats, counters, targets and clamps (R1–R10)
were filled in by rule from the first commit and never raised. The viewer's "perfect
spawns" were rules.

## 7. The barometer is a ground-truth rollout, and the real game

- **World model:** the held-out loss by open-loop horizon (h1 … h4096) plus the
  rubric metrics (circle hit, jitter, speed ratio), not the training loss.
- **Dream RL:** the real C game (`mobawm/rl/arena.py`) against the untrained policy
  and the shipped net on the same seeds, not dream reward.
- **Towers:** `scripts/moba/tower_eval_big.py` on all 100 held-out games (900–999): 128-tick
  open-loop windows every 64 ticks (4,017 windows, 1,561 real falls), precision and recall of
  tower falls (±32 ticks), siege damage (dream / real HP for towers losing ≥ 800), with
  per-game bootstrap 90% intervals. About 7–9 minutes per checkpoint on one H200;
  `scripts/moba/eval_watch.py` scores every checkpoint of a run as it is written. The
  in-training tower line (64 windows, about 23 real falls) is a smoke signal only.

*Incident:* dream reward rose 3× while the real game got worse than random (E7).

*Incident:* the 64-window in-training tower eval swung 13% → 48% precision between
checkpoints of one run and ranked arms wrongly (A1b looked better than the control). On the
100-game eval, A1b was not better; the same model also predicts 9 vs 11 falls on one game
on CPU vs MPS, so tower outcomes in the dream are chaotic and need many games.

## 8. One seed is not a measurement

Report ranges, or say "single seed". E5's own evaluations swing ±15–30% at long
horizons, and h4096 rests on only a few long games.

## 9. Run length is the author's number, and the controls come with it

Launch exactly the steps or updates asked for; funds are the author's call. Every A/B
names its control arm, and compares at matched updates **and** matched wall clock.

## 10. The session-end check

Before ending any session that produced a number:
- does PROVENANCE let someone else re-derive it without asking?
- does CLAIMS say what class it is?
- are its checkpoints in S3, with md5s in the manifest?

If not, the session is not finished.
