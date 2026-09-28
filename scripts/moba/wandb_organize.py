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
    "usrc41qc": ("E9-dyna-smoke", ["wm", "rollout", "dyna", "control", "store-cpu"],
                 "Dyna smoke control: E5 @20,500 + ~125 updates on mix1k only, batch 384, H100 NVL. " + REC + "E9-dyna-smoke.md"),
    "tnnlmx2e": ("E9-dyna-smoke", ["wm", "rollout", "dyna", "policy-games", "store-cpu"],
                 "Dyna smoke: E5 @20,500 + ~125 updates on mix1k + 100 real P3-policy games (game ids 5000-5099). " + REC + "E9-dyna-smoke.md"),
    "mlbh5d3l": ("E11-random-data", ["wm", "rollout", "random-data", "store-cpu"],
                 "E11: E5 @20,500 resumed on mix1k + mixrand1k (random-controller games), 2,300 updates; held-out "
                 "random games (random/...) are the main metric. Stopped for E12. " + REC + "E11-random-data.md"),
    "xfldazic": ("E12-dyna2", ["wm", "rollout", "random-data", "dyna", "store-cpu", "main"],
                 "E12: E11 @2000 + dyna2 (810 real games of the dream policies vs random/net/self). " + REC + "E12-dyna2.md"),
    "j6c6g2ox": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 control: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "skk7a1wh": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 derived: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "grd66rl7": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 real: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "9ttsnv28": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 consist: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "3ifyreey": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 tower10: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "ti1m431j": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 events: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "r8r9tnzz": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 consist: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "b3cl5bq3": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 tower10: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "6krtubiy": ("E15-tower-dynamics", ["wm", "tower", "aborted"], "E15 events: aborted launch (1,000 updates / relaunch), replaced by the 5-minute arm. " + REC + "E15-tower-dynamics.md"),
    "jox03h6v": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: control (fine-tune only); E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "1xflezs2": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: derived attack graph; E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "wt4kffxm": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: real attack graph; E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "zroavbb9": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: tower consistency loss; E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "0ab2b4gu": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: tower health x10; E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "eaqg898j": ("E15-tower-dynamics", ["wm", "tower", "5min", "main"], "E15 arm: tower-event sampling; E11 @1500 fine-tuned 5 min (~60-100 updates). State failed = the end-of-run report crashed after the model was saved (fixed). " + REC + "E15-tower-dynamics.md"),
    "qsh9x6mg": ("E16-towers", ["wm", "smoke", "oom"], "E16 smoke #1: CUDA OOM (eval_horizons lost @torch.no_grad when tower_eval was inserted; fixed). " + REC + "E15-tower-dynamics.md"),
    "uj692w3j": ("E16-towers", ["wm", "smoke"], "E16 smoke #2: combined recipe 3 min, passes; tower OL128 precision 41% -> 62%. " + REC + "E15-tower-dynamics.md"),
    "vele1qb8": ("E16-towers", ["wm", "rollout", "attack-graph", "tower-consist", "tower-events", "dyna", "store-cpu", "main"],
                 "E16: E12 @4000 + real attack graph + tower consistency + tower-event sampling, full data. Stopped at ~150 updates: recipe not yet validated (E17 then showed the additions don't pay). " + REC + "E15-tower-dynamics.md"),
    "02klseww": ("E17-attack-ab", ["wm", "tower", "attack-graph", "60-updates", "main"], "E17: real attack graph only, E11 @1500 + 60 updates. " + REC + "E15-tower-dynamics.md"),
    "08wrb6xw": ("E17-attack-ab", ["wm", "tower", "attack-graph", "60-updates"], "E17: real attack graph + consistency loss, 60 updates. " + REC + "E15-tower-dynamics.md"),
    "sdmvluu5": ("E17-attack-ab", ["wm", "tower", "attack-graph", "60-updates"], "E17: real attack graph + tower-event sampling, 60 updates. " + REC + "E15-tower-dynamics.md"),
    "h5jt4gw8": ("E17-attack-ab", ["wm", "tower", "attack-graph", "60-updates"], "E17: real attack graph + consistency + event sampling (combined), 60 updates. " + REC + "E15-tower-dynamics.md"),
    "tz9mc91w": ("E18-attack", ["wm", "rollout", "attack-graph", "dyna", "store-cpu", "main"], "E18: E12 @4000 + real attack graph only, full data (expert+random+dyna2), open-ended. " + REC + "E15-tower-dynamics.md"),
    "d9xnnj0a": ("E6-counterfactual", ["wm", "rollout", "counterfactual", "effect-loss", "main"],
                 "E6b: E5@3500 + counterfactual siblings + effect aux losses, 3,550 updates. Checkpoints kept. "
                 + REC + "E6-counterfactual-siblings.md"),
}

# moba-dream-rl runs made after the backfill (their group is set at launch; this adds tags/notes).
DREAM = {
    "msjwyq7r": ("P4-mix-e11", ["dream-ppo", "opponent-mix", "e11"],
                 "P4: dream PPO in E11 @1500, 50% self-play / 50% vs net, expert start states, 4.5 min. " + REC + "E13-dream-ppo-e11.md"),
    "gqwfjvhm": ("P4-mix-e11", ["dream-ppo", "crashed"],
                 "P4b first launch: crashed at wandb init (thread limit, an evaluation started at the same time). " + REC + "E13-dream-ppo-e11.md"),
    "ahoexos3": ("P4-mix-e11", ["dream-ppo", "opponent-mix", "e11", "mixed-starts"],
                 "P4b: as P4 + start states from expert AND random-play games, 4.5 min. " + REC + "E13-dream-ppo-e11.md"),
    "zdq4qs2a": ("P6-roleheads", ["dream-ppo", "mlp", "e11"], "P6 base: P4b recipe rerun (reproduces P4b exactly). " + REC + "E14-dream-ppo-policy.md"),
    "74m80azw": ("P6-roleheads", ["dream-ppo", "mlp", "role-heads", "e11"], "P6: + role embedding and per-role action/value heads. " + REC + "E14-dream-ppo-policy.md"),
    "1zl7yomf": ("P7-arch", ["dream-ppo", "puffer-mingru", "puffer-hparams", "e11"], "P7 B: PufferLib Encoder+6-layer MinGRU, old-repo PPO settings. " + REC + "E14-dream-ppo-policy.md"),
    "s2aec2an": ("P7-arch", ["dream-ppo", "puffer-mingru", "e11", "best", "main"], "P7 C (BEST): PufferLib MinGRU + our PPO, 1.70M steps. Kept: s3 moba/runs/keep/p7_armC_puffer_ours_1.70M/. " + REC + "E14-dream-ppo-policy.md"),
    "5z05e3kt": ("P8-ent-sweep", ["dream-ppo", "puffer-mingru", "e11"], "P8: arm C, ent 0.003. " + REC + "E14-dream-ppo-policy.md"),
    "gqha5cv0": ("P8-ent-sweep", ["dream-ppo", "puffer-mingru", "e11"], "P8: arm C, ent 0.03. " + REC + "E14-dream-ppo-policy.md"),
    "znomih0f": ("P8-ent-sweep", ["dream-ppo", "puffer-mingru", "e11"], "P8: arm C, ent 0.05. " + REC + "E14-dream-ppo-policy.md"),
    "ps9yzz2j": ("P8-ent-sweep", ["dream-ppo", "puffer-mingru", "e11"], "P8: arm C, ent 0.1. " + REC + "E14-dream-ppo-policy.md"),
    "wcqdkrnr": ("P9-puffer-2x", ["dream-ppo", "puffer-mingru", "e11"], "P9: arm C recipe at 2x steps (3.42M); first 1.70M identical to C. " + REC + "E14-dream-ppo-policy.md"),
    "jjdb2ezj": ("P10-e12-dream", ["dream-ppo", "puffer-mingru", "e12"], "P10: arm C recipe in the E12 @4000 dream. " + REC + "E14-dream-ppo-policy.md"),
    "4gytv3k0": ("P11-reward-sweep", ["dream-ppo", "puffer-mingru", "e11", "reward-shaping"], "P11: arm C + dream-only reward shaping: tower 3 to the attacking team. " + REC + "E14-dream-ppo-policy.md"),
    "z04it194": ("P11-reward-sweep", ["dream-ppo", "puffer-mingru", "e11", "reward-shaping"], "P11: arm C + dream-only reward shaping: win/loss +-10 to each hero. " + REC + "E14-dream-ppo-policy.md"),
    "7il3e2up": ("P11-reward-sweep", ["dream-ppo", "puffer-mingru", "e11", "reward-shaping"], "P11: arm C + dream-only reward shaping: tower 3 team + win +-10. " + REC + "E14-dream-ppo-policy.md"),
    "dr5qofw7": ("P11-reward-sweep", ["dream-ppo", "puffer-mingru", "e11", "reward-shaping"], "P11: arm C + dream-only reward shaping: tower 10 team + win +-30. " + REC + "E14-dream-ppo-policy.md"),
    "lz7opfch": ("P12-tower-shaping", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P12 tower damage k2 boost2: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "r1titdgl": ("P12-tower-shaping", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P12 falling-tower bonus 5 r8: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "jd1xie1g": ("P12-tower-shaping", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P12 damage + fall: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "n9o3hwu0": ("P12-tower-shaping", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P12 damage + fall big: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "6l122crp": ("P13-fall-sweep", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P13 fall bonus 15: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "ojvhv6dm": ("P13-fall-sweep", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P13 fall bonus 30: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "iuwujj5g": ("P13-fall-sweep", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P13 fall bonus 60: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "j75qsm75": ("P13-fall-sweep", ["dream-ppo", "puffer-mingru", "e11", "tower-shaping"], "P13 fall bonus 120: arm C + dream-only tower shaping. " + REC + "E14-dream-ppo-policy.md"),
    "jfw1p3b5": ("P5-100x", ["dream-ppo", "opponent-mix", "e11", "mixed-starts", "main"],
                 "P5: the P4b recipe for 100x P4b's steps (169.5M). " + REC + "E13-dream-ppo-e11.md"),
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
    for run in api.runs(f"{ENTITY}/moba-dream-rl"):
        if run.id in DREAM:
            group, tags, note = DREAM[run.id]
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
