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
ARTIFACTS += [(f"wm/xpsmoke_{a}", f"runs/token_wm/xpsmoke_{a}/model.pt", None,
                f"XP smoke test (local CPU, ~30 updates from E5 @20,500, batch 32): {d}")
              for a, d in (("xp1", "control, XP loss x1"), ("xp100", "XP loss x100"), ("noclamp", "no XP clamp (rule R7 off)"))]
ARTIFACTS += [("wm/dyna1_control", "runs/token_wm/dyna1_control/model.pt", "usrc41qc",
               "Dyna smoke control: E5 @20,500 fine-tuned ~125 updates on mix1k only (batch 384, H100 NVL)"),
              ("wm/dyna1_policy_games", "runs/token_wm/dyna1_policy_games/model.pt", "tnnlmx2e",
               "Dyna smoke: E5 @20,500 fine-tuned ~125 updates on mix1k + 100 real games played by the P3 policies")]
ARTIFACTS += [("rl/P3_vs_net", "runs/dream_ppo/p3/vs_net/policy.pt", "iluni10r",
               "E7-P3 dream PPO trained AGAINST the shipped net (4.5 min, 1.10M steps); real game: 0 wins, below random."),
              ("rl/P3_selfplay_rerun", "runs/dream_ppo/p3/selfplay_rerun/policy.pt", "1q729jk7",
               "E7-P3 self-play rerun of the P1 best config (4.5 min, 9.74M steps); real game: 0 wins, below random vs net.")]
ARTIFACTS += [("wm/e11_at1500", "runs/token_wm/e11_at1500/model.pt", "mlbh5d3l",
               "E11 @1500 (E5 resumed on mix1k + mixrand1k): the world model the P4/P4b/P5 dream policies trained in"),
              ("wm/e11_at2000", "runs/token_wm/e11_random_resume/model_002000.pt", "mlbh5d3l",
               "E11 @2000, E11's last checkpoint; E12 starts from it"),
              ("rl/P4_mix_e11", "runs/dream_ppo/p4/mix_e11/policy.pt", "msjwyq7r",
               "P4: dream PPO in E11 @1500, 50% self-play / 50% vs net, expert starts, 4.5 min, 2.12M steps"),
              ("rl/P4b_mix_e11_mixstarts", "runs/dream_ppo/p4/mix_e11_mixstarts/policy.pt", "ahoexos3",
               "P4b: as P4 + expert and random-play start states, 4.5 min, 1.70M steps")]
ARTIFACTS += [("rl/P7_C_puffer_ours_1.70M", "runs/dream_ppo/p7/puffer_ours/policy.pt", None,
               "BEST dream policy so far (arm C): PufferLib Encoder + 6-layer MinGRU (recurrent) + our PPO, E11 @1500 "
               "dream, mix curriculum, mixed start states, 1.70M steps. Real vs shipped net: return 9.30, creeps 14.5, "
               "0 wins (W&B moba-dream-rl group P7-arch)")]
# uploads that do not follow the path -> key rule in main()
S3KEY = {"rl/P7_C_puffer_ours_1.70M": "moba/runs/keep/p7_armC_puffer_ours_1.70M/policy.pt",
         "wm/e11_at1500": "moba/runs/e11_at1500/model.pt",
         "wm/e11_at2000": "moba/runs/e11_random_resume/model_002000.pt",
         "rl/P4_mix_e11": "moba/runs/p4_mix_e11/policy.pt",
         "rl/P4b_mix_e11_mixstarts": "moba/runs/p4_mix_e11_mixstarts/policy.pt"}
ARTIFACTS += [(f"wm/e15_{k}", f"runs/token_wm/e15_{k}/model.pt", i, f"E15 arm {k}: E11 @1500 fine-tuned 5 min (tower fix ablation)")
              for k, i in (("control", "jox03h6v"), ("derived", "1xflezs2"), ("real", "wt4kffxm"), ("consist", "zroavbb9"),
                           ("tower10", "0ab2b4gu"), ("events", "eaqg898j"))]
ARTIFACTS += [(f"wm/e17_{k}", f"runs/token_wm/e17_{k}/model.pt", i, f"E17 A/B arm {k}: E11 @1500 + 60 updates")
              for k, i in (("real", "02klseww"), ("real_consist", "08wrb6xw"), ("real_events", "sdmvluu5"),
                           ("real_consist_events", "h5jt4gw8"))]
ARTIFACTS += [("wm/e19_control", "runs/token_wm/e19_control/model.pt", "y9yxv13e", "E19 control: E12 @4000 + 250 updates, no attack graph"),
              ("wm/e19_real", "runs/token_wm/e19_real/model.pt", "rer7ogt1", "E19: E12 @4000 + 250 updates, real attack graph (best tower checkpoint at full config)")]
ARTIFACTS += [(f"wm/e18_at{i}", f"runs/token_wm/e18_attack/model_{i}.pt", "tz9mc91w", f"E18 checkpoint @{int(i)} (tower precision degrades)")
              for i in ("000500", "001000", "001500")]
S3KEY.update({f"wm/e18_at{i}": f"moba/runs/e18_attack/model_{i}.pt" for i in ("000500", "001000", "001500")})
ARTIFACTS += [("code/code_snapshot_20260929_0100", "runs/code_snapshots/code_snapshot_20260929_0100.tar.gz", None,
               "Snapshot of the uncommitted code behind E18/E19 and the tower-death rule smoke test (data.TOWER_DEATH_RULE, "
               "tower_event_audit --tower-death-rule, tower_health_diag.py).")]
ARTIFACTS += [("code/code_snapshot_20260928_2300", "runs/code_snapshots/code_snapshot_20260928_2300.tar.gz", None,
               "Snapshot of the uncommitted code behind E15/E16 and P12/P13: attack graph (data.derived_targets, "
               "attack_inputs, model atk_in), tower consistency/weight/event lanes, tower_eval, tower_event_audit.py, "
               "dream_ppo tower shaping (tower_push), report generator fix.")]
ARTIFACTS += [("code/code_snapshot_20260928_0700", "runs/code_snapshots/code_snapshot_20260928_0700.tar.gz", None,
               "Snapshot of the uncommitted code behind E14: dream_ppo --arch puffer (PufferPolicy, recurrent PPO update), "
               "--role-heads, --r-tower/--tower-team/--r-win shaping, Adam/clip flags; arena recurrent memory.")]
ARTIFACTS += [("code/code_snapshot_20260928_0215", "runs/code_snapshots/code_snapshot_20260928_0215.tar.gz", None,
               "Snapshot of the uncommitted code behind E11-E13: train.py --held2-games, anonymous-mmap loader "
               "(data.py), dream_ppo --opponent mix, collect_policy_games --matchups/--slice, pod.py CPU pods. "
               "HEAD baf8d7c + tracked diff + files.")]
ARTIFACTS += [("code/code_snapshot_20260928_0015", "runs/code_snapshots/code_snapshot_20260928_0015.tar.gz", None,
               "Snapshot of the uncommitted code behind E8-E11: collector random controller (moba_collect.c, "
               "shards.CONTROLLERS), ma_multi_final_hash, plan_mpc realtime/batched planner, XP clamp flags, "
               "train.py --held2-games, audit/Dyna/calibration scripts. HEAD 173cfc1 + tracked diff + files.")]
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
            key = S3KEY.get(name, key)
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
