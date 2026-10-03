"""Score every checkpoint of a run with the 100-game tower evaluation as it appears (runs next to training on
the same GPU). Results go to <run>/tower_big_<ckpt>.json and are uploaded with the run's S3 policy.

    python scripts/moba/eval_watch.py --run runs/token_wm/g2_grid_long --until 1500
"""
import argparse, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from mobawm.s3persist import persist

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--until", type=int, required=True)
ap.add_argument("--device", default="cuda"); a = ap.parse_args()
run = ROOT / a.run; done = set()
while True:
    for ck in sorted(run.glob("model_*.pt")):
        n = int(ck.stem.split("_")[1])
        if n in done:
            continue
        time.sleep(5)                                              # let the trainer finish writing it
        ev = run.parent / f"_ev_{run.name}_{n:06d}"
        ev.mkdir(exist_ok=True); shutil.copy(ck, ev / "model.pt"); shutil.copy(run / "stats.pt", ev / "stats.pt")
        out = run / f"tower_big_{n:06d}.json"
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/moba/tower_eval_big.py"), "--wm", str(ev.relative_to(ROOT)),
                        "--device", a.device, "--out", str(out)], cwd=ROOT, check=False)
        if out.exists():
            persist(out, wait=True)
        done.add(n)
        print(f"scored checkpoint {n}", flush=True)
    if a.until in done:
        break
    time.sleep(30)
