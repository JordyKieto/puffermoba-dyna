"""Give every MOBA W&B run an experiment group, tags and a note pointing at its record.

The mapping below is the source of truth for "which run belongs to which
experiment" (docs/moba/PROVENANCE.md). Re-running is idempotent.

    .venv/bin/python scripts/moba/wandb_organize.py            # dry run: print the plan
    .venv/bin/python scripts/moba/wandb_organize.py --apply    # write group/tags/notes to W&B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ENTITY, PROJECT = "drift-ai-models", "moba-wm"
REC = "docs/moba/experiments/"

# run id -> (group, tags, note). Tags: track, regime, outcome.
E = {
    "vuzjgdh8": ("E1-smoke", ["wm", "teacher-forced", "crashed"], "first smoke run; crashed. docs/moba/EXPERIMENTS.md E1"),
    "ox4cx47i": ("E1-smoke", ["wm", "teacher-forced", "finished"], "smoke2, mix100. docs/moba/EXPERIMENTS.md E1"),
    "asdg1vl1": ("E2-mix1k", ["wm", "teacher-forced", "finished"], "first full mix1k run. docs/moba/EXPERIMENTS.md E2"),
    "65j98z3b": ("E3-gate-events", ["wm", "teacher-forced", "gate", "finished"], "change gate + event sampling. docs/moba/EXPERIMENTS.md E3b"),
    "s3ilbx02": ("E3-gate-events", ["wm", "teacher-forced", "gate", "crashed"], "E3c 30 min; crashed. docs/moba/EXPERIMENTS.md E3c"),
    "kxe2bh1k": ("E4-baseline-v2", ["wm", "teacher-forced", "baseline-v2", "finished", "final-ckpt-lost"],
                 "baseline v2, 1 h, H200. Final report never written (session stopped for E5); final checkpoint "
                 "lost with the pod; a ~update-3000 copy survives. " + REC + "E4-baseline-v2-1h.md"),
    "zll4trqi": ("E5-rollout", ["wm", "rollout", "aborted"], "E5 first launch (30-min budget), aborted before training. " + REC + "E5-rollout-training.md"),
    "93m7sktq": ("E5-rollout", ["wm", "rollout", "baseline-v2", "main"],
                 "E5 autoregressive rollout training from E4 final, updates 0-20,750. Final checkpoint (20,500) kept; "
                 "numbered checkpoints lost with the pod. " + REC + "E5-rollout-training.md"),
    "hm9mnnoy": ("E5-rollout", ["wm", "rollout", "oom"], "E5 continuation on H100 NVL; OOM before training (fixed by --store-cpu). " + REC + "E5-rollout-training.md"),
    "soeosr9h": ("E5-rollout", ["wm", "rollout", "store-cpu", "ckpts-lost"],
                 "E5 continuation from 20,500 (steps restart at 0 = E5 20,500+), H100 NVL, --store-cpu. Pod deleted "
                 "externally at ~1,700: ALL checkpoints lost; metrics survive here only. " + REC + "E5-rollout-training.md"),
    "uo6sal6q": ("E6-counterfactual", ["wm", "rollout", "counterfactual", "replaced"],
                 "E6 plain counterfactual siblings, 100 updates, replaced by E6b. " + REC + "E6-counterfactual-siblings.md"),
    "d9xnnj0a": ("E6-counterfactual", ["wm", "rollout", "counterfactual", "effect-loss", "main"],
                 "E6b: E5@3500 + counterfactual siblings + effect aux losses, 3,550 updates. Checkpoints kept. "
                 + REC + "E6-counterfactual-siblings.md"),
}


def plan(run):
    if run.id in E:
        return E[run.id]
    if run.name.startswith("ablate"):          # 5-minute ablation rounds: already grouped ablate<N>
        return (run.group or run.name.split("/")[0], ["wm", "teacher-forced", "ablation-5min", run.state],
                f"5-minute A/B ablation {run.name}. docs/moba/EXPERIMENTS.md (legacy log), ablation rounds")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    from racewm import secrets
    secrets.export_to_env("WANDB_API_KEY")
    import wandb
    api = wandb.Api()
    missing = []
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        p = plan(run)
        if p is None:
            missing.append(run.id)
            continue
        group, tags, note = p
        print(f"{run.id}  {run.name:<32} -> group {group:<20} tags {tags}")
        if args.apply:
            run.group, run.tags, run.notes = group, sorted(set(tags)), note
            run.update()
    if missing:
        print(f"NOT MAPPED (add them to E): {missing}")
        sys.exit(1)
    print("applied" if args.apply else "dry run; --apply to write")


if __name__ == "__main__":
    main()
