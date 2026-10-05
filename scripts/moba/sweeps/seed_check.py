"""300 games per side (5 x 60: a single 300-game arena hangs) on seeds BASE + 1000k + i for each policy, on the CPU pod POD.
usage: POD=evc1 [SEED_BASE=70000000] python scripts/moba/sweeps/seed_check.py name=path [name=path ...]"""
import sys, json; sys.path.insert(0, "tools"); import pod
R = pod.R_REPO
pod.push_code(); pod.run(f"mkdir -p {R}/holdout")
import os
BASE = int(os.environ.get("SEED_BASE", 70_000_000))          # 60000000: fresh (anchor selection); 70000000: holdout
pols = dict(a.split("=", 1) for a in sys.argv[1:])            # name=path ...
import time
for name, p in pols.items():
    pod.rsync_to([p], f"{R}/holdout/{name}.pt")
    g = []
    for k in range(5):
        t0 = time.time()
        pod.run(f"cd {R} && timeout 300 python -u -m mobawm.rl.arena --policy holdout/{name}.pt --games 60 --workers 32 --seed-base {BASE + 1000 * k} "
                f"--teams ppo --opponents net --out holdout/{name}_{k}.jsonl > holdout/{name}_{k}.log 2>&1", check=False)
        g += [json.loads(l) for l in pod.run(f"cat {R}/holdout/{name}_{k}.jsonl 2>/dev/null", capture=True, check=False).splitlines() if l.strip()]
        print(name, k, len(g), f"{time.time() - t0:.0f}s", flush=True)
    for side in ("radiant", "dire"):
        x = [r for r in g if r["side"] == side]
        print("RESULT", name, side, f"{sum(r['won'] for r in x)}/{len(x)} wins", f"towers {5 * sum(r['ours_towers_killed'] for r in x) / max(len(x), 1):.2f}", flush=True)
