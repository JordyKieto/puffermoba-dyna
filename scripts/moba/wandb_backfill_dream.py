"""Backfill the dream-PPO runs (which logged only to local log.jsonl) and the real-game
arena evaluation into W&B project drift-ai-models/moba-dream-rl.

Groups:
  P0-first-5min      the first 5-min self-play run (Mac MPS)
  P1-sweep-mac       LR x minibatch sweep, 6 configs in parallel on the Mac GPU: inconclusive (2-4 updates each)
  P1-sweep-h200      the same sweep on an H200, 4.5 min, ~30 updates / 2.4M agent steps per config
  P2-arena-real      the best sweep checkpoint in the REAL C game vs the shipped net and the scripted bots

Created run ids are written to results/moba/dream/wandb_ids.json; a run already there is skipped.

    .venv/bin/python scripts/moba/wandb_backfill_dream.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ENTITY, PROJECT = "drift-ai-models", "moba-dream-rl"
IDS = ROOT / "results/moba/dream/wandb_ids.json"
DOC = "docs/moba/experiments/E7-dream-ppo.md"

RUNS = [("P0-first-5min", "runs/dream_ppo/e5_selfplay_5min", ["dream-ppo", "self-play", "mps", "5min"])] + \
    [("P1-sweep-mac", f"runs/dream_ppo/sweep1/{c}", ["dream-ppo", "sweep", "mps", "inconclusive"])
     for c in ("lr3e-4_mb10", "lr1e-3_mb10", "lr3e-3_mb10", "lr3e-4_mb2", "lr1e-3_mb2", "lr3e-3_mb2")] + \
    [("P1-sweep-h200", f"runs/dream_ppo/sweep1_gpu/{c}", ["dream-ppo", "sweep", "h200"])
     for c in ("lr3e-4_mb10", "lr1e-3_mb10", "lr3e-3_mb10", "lr3e-4_mb2", "lr1e-3_mb2", "lr3e-3_mb2")]


def main():
    from mobawm import secrets
    secrets.export_to_env("WANDB_API_KEY")
    import torch
    import wandb
    ids = json.loads(IDS.read_text()) if IDS.exists() else {}
    IDS.parent.mkdir(parents=True, exist_ok=True)
    for group, d, tags in RUNS:
        d = ROOT / d
        key = str(d.relative_to(ROOT))
        if key in ids or not (d / "log.jsonl").exists():
            continue
        rows = [json.loads(l) for l in open(d / "log.jsonl")]
        cfg = {}
        if (d / "policy.pt").exists():
            ck = torch.load(d / "policy.pt", map_location="cpu", weights_only=False)
            cfg = ck.get("args", {}) if isinstance(ck, dict) else {}
        best = "lr3e-4_mb2" in key and "gpu" in key
        run = wandb.init(entity=ENTITY, project=PROJECT, group=group, name=f"{group}/{d.name}",
                         tags=tags + (["best-sweep", "arena-evaluated"] if best else []),
                         config=dict(cfg, source_dir=key, backfilled=True),
                         notes=f"Backfilled from {key}/log.jsonl (the run logged locally only). {DOC}",
                         reinit=True)
        for r in rows:
            run.log({k: v for k, v in r.items() if v is not None and k != "update"}, step=r["update"])
        ids[key] = run.id
        run.finish()
        IDS.write_text(json.dumps(ids, indent=1))
        print(f"{key} -> {run.id}", flush=True)

    arena = ROOT / "runs/dream_ppo/sweep1_gpu/lr3e-4_mb2/arena.jsonl"
    if "arena" not in ids and arena.exists():
        games = [json.loads(l) for l in open(arena)]
        run = wandb.init(entity=ENTITY, project=PROJECT, group="P2-arena-real", name="P2-arena-real/lr3e-4_mb2",
                         tags=["arena", "real-game", "negative-result"],
                         config=dict(policy="runs/dream_ppo/sweep1_gpu/lr3e-4_mb2/policy.pt", games_per_cell=10,
                                     max_ticks=6000, seeds="50000000+", teams=["ppo", "random", "net"],
                                     opponents=["net", "script"]),
                         notes=f"Dream-trained policy in the real C game: 0 wins, below the untrained policy. {DOC}",
                         reinit=True)
        cols = list(games[0].keys())
        run.log({"games": wandb.Table(columns=cols, data=[[g[c] for c in cols] for g in games])})
        for team in ("ppo", "random", "net"):
            for opp in ("net", "script"):
                g = [x for x in games if x["team"] == team and x["opp"] == opp]
                if g:
                    run.summary[f"{team}_vs_{opp}/win_pct"] = 100 * sum(x["won"] for x in g) / len(g)
                    run.summary[f"{team}_vs_{opp}/loss_pct"] = 100 * sum(x["lost"] for x in g) / len(g)
                    run.summary[f"{team}_vs_{opp}/return_ours"] = sum(x["return_ours"] for x in g) / len(g)
        ids["arena"] = run.id
        run.finish()
        IDS.write_text(json.dumps(ids, indent=1))
        print(f"arena -> {run.id}")
    print(json.dumps(ids, indent=1))


if __name__ == "__main__":
    main()
