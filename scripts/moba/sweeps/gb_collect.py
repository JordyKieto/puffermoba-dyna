"""Collect the reward sweep: dream towers/episode (ours vs net, mean of the last 5 updates), progress, and the real arena."""
import os, re, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "runs/pod")
from gen2 import ARMS
def get(ia):
    i, arm = ia
    d = f"/root/puffertank/runs/dream_ppo/rw/{arm}"
    cmd = f"grep -E '^upd' {d}/train.log | tail -5; echo ===; cat {d}/arena.jsonl 2>/dev/null"
    for attempt in range(4):                        # 24 parallel SSH calls: some fail transiently
        try:
            r = subprocess.run([".venv/bin/python", "tools/pod.py", "ssh", cmd], env=dict(os.environ, POD=f"gb{i}"), capture_output=True, text=True, timeout=90)
            if "upd" in r.stdout:
                return arm, r.stdout
        except subprocess.TimeoutExpired:
            pass
    return arm, ""
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, enumerate(ARMS)))
rows = []
for arm, out in res:
    up, _, ar = out.partition("===")
    ups = [l for l in up.splitlines() if l.startswith("upd")]
    m = [re.search(r"ours ([\d.]+) vs net ([\d.]+)", l) for l in ups]; m = [x for x in m if x]
    steps = re.search(r"([\d.]+)M steps", ups[-1]).group(1) if ups else "0"
    do = sum(float(x.group(1)) for x in m) / max(len(m), 1); dn = sum(float(x.group(2)) for x in m) / max(len(m), 1)
    G = [json.loads(l) for l in ar.splitlines() if l.startswith("{")]
    real = {}
    for side in ("radiant", "dire"):
        g = [x for x in G if x["side"] == side]
        if g:
            real[side] = (sum(x["won"] for x in g), len(g), 5 * sum(x["ours_towers_killed"] for x in g) / len(g),
                          5 * sum(x["theirs_towers_killed"] for x in g) / len(g))
    rows.append((arm, steps, do, dn, real))
print("| arm | steps (M) | dream towers/ep ours | net | radiant: wins | radiant: our towers/game | dire: wins | dire: towers ours / net |")
print("|---|---|---|---|---|---|---|---|")
key = lambda r: -(r[4].get("radiant", (0, 0, 0, 0))[2] + r[4].get("dire", (0, 0, 0, 0))[2] + 10 * (r[4].get("radiant", (0,))[0] + r[4].get("dire", (0,))[0]))
for arm, st, do, dn, real in sorted(rows, key=key):
    ra, di = real.get("radiant"), real.get("dire")
    print(f"| {arm} | {st} | {do:.3f} | {dn:.3f} | {f'{ra[0]}/{ra[1]}' if ra else '…'} | {f'{ra[2]:.1f}' if ra else '…'} | "
          f"{f'{di[0]}/{di[1]}' if di else '…'} | {f'{di[2]:.1f} / {di[3]:.1f}' if di else '…'} |")
