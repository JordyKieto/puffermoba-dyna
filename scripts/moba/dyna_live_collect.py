"""Continuous Dyna collector: one worker process ON a CPU pod. Forever: snapshot <dir>/pols/current.pt (the controller
swaps it in atomically whenever the PPO fleet publishes a new policy), play 10 real games of it vs the shipped net
(alternating sides), record + replay + verify them in a private scratch dir, then move the dense/log/meta files into
<dir>/{dense,logs,meta} and upload them (<dir>/.s3_policy.json -> s3://.../moba/runs/<run>/...). Game ids: --base + 10*k.
usage: python scripts/moba/dyna_live_collect.py --dir <dir> --worker <w> --base <first game id>"""
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mobawm.s3persist import persist  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--worker", type=int, required=True)
ap.add_argument("--base", type=int, required=True); ap.add_argument("--max-files", type=int, default=1000)
a = ap.parse_args()
d = Path(a.dir).resolve()
for sub in ("dense", "logs", "meta"):
    (d / sub).mkdir(parents=True, exist_ok=True)
for k in range(a.max_files):
    first = a.base + 10 * k
    if (d / "meta" / f"games.{first:07d}.json").exists():          # restarted worker: skip what is already done (uploaded)
        continue
    cur = d / "pols" / "current.pt"
    while not cur.exists():
        time.sleep(10)
    w = d / f"w{a.worker}"
    shutil.rmtree(w, ignore_errors=True); w.mkdir()
    snap = w / "policy.pt"; shutil.copy(cur, snap)
    import torch
    ck = torch.load(snap, map_location="cpu", weights_only=False)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "scripts/moba/collect_policy_games.py", "--policy", str(snap),
                        "--games-per-policy", "10", "--first-game", str(first), "--seed-base", str(91_000_000 + first),
                        "--slice", "0:10", "--replay-jobs", "2", "--template-log", str(d / "games.0000900.mbl"),
                        "--out", str(w)], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) with hash/obs mismatches", out)
    if r.returncode or m is None or int(m.group(1)):
        print(f"games {first}: FAILED (rc {r.returncode}, mismatches {m.group(1) if m else '?'}): {out[-400:]}", flush=True)
        continue
    meta = json.loads((w / "games_0_10.json").read_text())
    for g in meta:
        g.update(policy_update=ck.get("update"), policy_wm=ck.get("wm"))
    name = f"games.{first:07d}"
    (d / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=1))
    shutil.move(w / "logs" / f"{name}.mbl", d / "logs" / f"{name}.mbl")
    shutil.move(w / "dense" / f"{name}.mbs", d / "dense" / f"{name}.mbs")     # dense last: its arrival marks a complete file
    for f in (d / "logs" / f"{name}.mbl", d / "dense" / f"{name}.mbs", d / "meta" / f"{name}.json"):
        persist(f, wait=True)                       # synchronous, md5-verified (an upload failure is printed)
    for f in (d / "logs" / f"{name}.mbl", d / "dense" / f"{name}.mbs"):
        f.unlink()                                  # in S3 now: the pod's disk filled up at ~400 files when they were kept
    print(f"games {first}: policy update {ck.get('update')}, {len(meta)} games, ticks {sum(g['ticks'] for g in meta)}, "
          f"{time.time() - t0:.0f}s", flush=True)
