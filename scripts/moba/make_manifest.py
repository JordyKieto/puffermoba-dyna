"""Write artifacts/moba/MANIFEST.json: every MOBA checkpoint identified by md5.

Checkpoints are gitignored; this manifest is what lets a stray .pt be matched to a
run later. For each artifact: local path, present, bytes, md5, the S3 copy (and
whether S3's ETag equals the local md5, i.e. the copy is byte-identical), the W&B
run it came from, and what it is.

    .venv/bin/python scripts/moba/make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

# (name, local path, wandb run, description)
ARTIFACTS = [
    ("wm/E4_baseline_v2_u3000", "runs/token_wm/e4_baseline_v2_1h/model.pt", "kxe2bh1k",
     "E4 teacher-forced baseline v2, pulled mid-run (~update 3,000 of 3,877). The final checkpoint was lost with the pod."),
    ("wm/E4_stats", "runs/token_wm/e4_baseline_v2_1h/stats.pt", "kxe2bh1k", "E4 stats + model config (shared by E5/E6)."),
    ("wm/E5_rollout_u3500", "runs/token_wm/e5_rollout_at3500/model.pt", "93m7sktq",
     "E5 at update 3,500: the starting point of E6/E6b."),
    ("wm/E5_rollout_u20500", "runs/token_wm/e5_rollout/model.pt", "93m7sktq",
     "E5 final kept checkpoint, update 20,500 (the best surviving world model for long horizons)."),
    ("wm/E5_stats", "runs/token_wm/e5_rollout/stats.pt", "93m7sktq", "E5 stats + config."),
    ("wm/E6b_u1000", "runs/token_wm/e6b_cf_effect/model_001000.pt", "d9xnnj0a", "E6b counterfactual + effect losses, update 1,000."),
    ("wm/E6b_u2000", "runs/token_wm/e6b_cf_effect/model_002000.pt", "d9xnnj0a", "E6b, update 2,000."),
    ("wm/E6b_u3000", "runs/token_wm/e6b_cf_effect/model_003000.pt", "d9xnnj0a", "E6b, update 3,000."),
    ("wm/E6b_u3500", "runs/token_wm/e6b_cf_effect/model.pt", "d9xnnj0a", "E6b final kept checkpoint, update 3,500."),
    ("rl/P0_first_5min", "runs/dream_ppo/e5_selfplay_5min/policy.pt", "x18k5q1z", "first dream self-play policy, 5 min on the Mac."),
    ("rl/P1_h200_lr3e-4_mb2", "runs/dream_ppo/sweep1_gpu/lr3e-4_mb2/policy.pt", "v22f5ut6",
     "best dream-PPO sweep policy (reward slope); evaluated in the real game: 0 wins, below random (P2)."),
] + [(f"rl/P1_h200_{c}", f"runs/dream_ppo/sweep1_gpu/{c}/policy.pt", None, f"dream-PPO sweep config {c}, final policy.")
     for c in ("lr3e-4_mb10", "lr1e-3_mb10", "lr3e-3_mb10", "lr1e-3_mb2", "lr3e-3_mb2")]
ARTIFACTS += [("rl/P3_vs_net", "runs/dream_ppo/p3/vs_net/policy.pt", "iluni10r",
               "E7-P3 dream PPO trained AGAINST the shipped net (4.5 min, 1.10M steps); real game: 0 wins, below random."),
              ("rl/P3_selfplay_rerun", "runs/dream_ppo/p3/selfplay_rerun/policy.pt", "1q729jk7",
               "E7-P3 self-play rerun of the P1 best config (4.5 min, 9.74M steps); real game: 0 wins, below random vs net.")]
ARTIFACTS += [("code/code_snapshot_20260927_1539", "runs/code_snapshots/code_snapshot_20260927_1539.tar.gz", None,
               "Snapshot of the uncommitted code behind E7-P3 (dream PPO vs shipped net: ma_pool/ma_pools in "
               "moba_arena.c, dream_ppo --opponent net) and the world-model planner (plan_mpc.py), 2026-09-27.")]
ARTIFACTS += [("code/code_snapshot_20260927_1323", "runs/code_snapshots/code_snapshot_20260927_1323.tar.gz", None,
               "Snapshot of every uncommitted source file behind E5-continuation/E6/E6b/dream-PPO/arena/S3-persistence "
               "(git HEAD dad8c7e + the tracked diff + untracked files), taken 2026-09-27 because that code is "
               "deliberately not committed yet (docs/moba/EVIDENCE_RULES.md).")]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import pod
    c, bucket = pod.s3()
    out = []
    for name, rel, run, desc in ARTIFACTS:
        p = ROOT / rel
        e = {"name": name, "path": rel, "present": p.exists(), "wandb_run": run, "description": desc}
        if p.exists():
            e["bytes"], e["md5"] = p.stat().st_size, md5(p)
            # training uploads go to moba/runs/<run>/ (tools/pod.py train); dream runs to moba/runs/dream_ppo/...
            key = "moba/runs/" + rel[len("runs/token_wm/"):] if rel.startswith("runs/token_wm/") else "moba/" + rel
            try:
                h = c.head_object(Bucket=bucket, Key=key)
                e["s3"] = f"s3://{bucket}/{key}"
                e["s3_byte_identical"] = h["ETag"].strip('"') == e["md5"]
            except Exception:
                e["s3"], e["s3_byte_identical"] = None, False
        out.append(e)
        print(f"{name:<28} {e.get('md5', '-')[:12]}  s3 identical: {e.get('s3_byte_identical')}")
    dest = ROOT / "artifacts/moba/MANIFEST.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "note": "Checkpoints are NOT committed (gitignored). This manifest identifies them by md5 so a file found "
                "later can be matched to a run. Durable copies: the s3 key per entry (ETag == md5 checked). "
                "Provenance: docs/moba/PROVENANCE.md.",
        "artifacts": out}, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
