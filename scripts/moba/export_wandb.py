"""Export every MOBA W&B run into results/moba/<run_id>/ (maintainer tool).

Documents cite these files, never W&B links: a number in docs/moba/paper/CLAIMS.md
points at results/moba/<run_id>/, so it survives W&B going away.

    results/moba/<run_id>/history.csv    one row per logged step (all scalar metrics)
    results/moba/<run_id>/config.json    run config (secrets stripped)
    results/moba/<run_id>/summary.json   final summary scalars
    results/moba/<run_id>/run.json       entity/project/run, name, group, tags, notes, state, timestamps
    results/moba/INDEX.json              one entry per exported run

Adapted from neural-simulators/tools/export_wandb_run.py (same file layout).

    .venv/bin/python scripts/moba/export_wandb.py                 # both projects, all runs
    .venv/bin/python scripts/moba/export_wandb.py --only d9xnnj0a # one run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "moba"
ENTITY = "drift-ai-models"
PROJECTS = ("moba-wm", "moba-dream-rl")
SECRET_HINTS = ("key", "secret", "token", "password")


def _scalar(v):
    if isinstance(v, bool) or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    return None


def _clean(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_") and not any(h in k.lower() for h in SECRET_HINTS)}


def export(run, project):
    out = OUT / run.id
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in run.scan_history(page_size=1000):
        r = {k: _scalar(v) for k, v in row.items() if _scalar(v) is not None}
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r.get("_step", 0))
    cols = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith("_"), k))
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summary = {k: _scalar(v) for k, v in run.summary._json_dict.items() if _scalar(v) is not None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "config.json").write_text(json.dumps(_clean(run.config), indent=2, sort_keys=True, default=str))
    meta = {"entity_project_run": f"{ENTITY}/{project}/{run.id}", "url": run.url, "name": run.name,
            "group": run.group, "tags": list(run.tags), "notes": run.notes, "state": run.state,
            "created_at": run.created_at, "heartbeat_at": getattr(run, "heartbeat_at", None),
            "exported_rows": len(rows), "max_step": max((r.get("_step", 0) for r in rows), default=0)}
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    return {"project": project, "run_id": run.id, "name": run.name, "group": run.group, "tags": list(run.tags),
            "state": run.state, "created_at": run.created_at, "history_rows": len(rows),
            "max_step": meta["max_step"], "dir": f"results/moba/{run.id}/"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    from racewm import secrets
    secrets.export_to_env("WANDB_API_KEY")
    import wandb
    api = wandb.Api(timeout=120)
    index_path = OUT / "INDEX.json"
    index = {e["run_id"]: e for e in json.loads(index_path.read_text())} if index_path.exists() else {}
    for project in PROJECTS:
        for run in api.runs(f"{ENTITY}/{project}"):
            if args.only and run.id not in args.only:
                continue
            e = export(run, project)
            index[run.id] = e
            print(f"{project}/{run.id} {run.name:<36} {e['history_rows']:6d} rows  group {run.group}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(sorted(index.values(), key=lambda e: (e["project"], e["created_at"])), indent=1))
    print(f"{len(index)} runs in {index_path}")


if __name__ == "__main__":
    main()
