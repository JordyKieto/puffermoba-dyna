"""Dyna metric (E40): the world model's tower predictions on the CURRENT policy's real games, collected after every
evaluated checkpoint was taken (so no checkpoint trained on them), for each world-model version. Runs on dl0 (A100).
usage: python runs/pod/dyna_tower_eval.py"""
import json, os, sys; sys.path.insert(0, "tools")
os.environ["POD"] = "dl0"
import pod
sel = json.load(open("runs/pod/dyna_eval_set.json"))
R = pod.R_REPO; D = f"{R}/data/moba/dyna_eval"
games = set(g for f in sel["first_ids"] for g in range(f, f + 10))
pod.s3_sync_to_pod("moba/runs/dyna_live/dense/", f"{D}/dense", games)
WMS = ["fleet20e_at500", "wm_r1", "wml_at3600", "wml2_at3400"]
for w in WMS:
    pod.run(f"mkdir -p {R}/runs/token_wm/dyev_{w}")
    pod.rsync_to([f"runs/token_wm/{w}/model.pt", f"runs/token_wm/{w}/stats.pt"], f"{R}/runs/token_wm/dyev_{w}/")
spec = ",".join(f"{f}-{f + 9}" for f in sel["first_ids"])
cmds = " ; ".join(f"python -u scripts/moba/tower_eval_big.py --wm runs/token_wm/dyev_{w} --dense {D}/dense --games {spec} --device cuda "
                  f"--out {D}/tower_{w}.json > {D}/tower_{w}.log 2>&1" for w in WMS)
pod.run(f"cat > {D}/run.sh <<'EOT'\ncd {R}\n{cmds}\nEOT")
pod.run(f"tmux new-session -d -s dyev 'bash {D}/run.sh'")
print("STARTED", spec)
