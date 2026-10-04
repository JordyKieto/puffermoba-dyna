"""Generate the 100-game tower tables (docs/moba) from results/moba/tower_big: one row per scored checkpoint.
    .venv/bin/python scripts/moba/tower_tables.py > results/moba/tower_big/TABLE.md"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results/moba/tower_big"
def row(name, r):
    p, rc, s = r["precision"], r["recall"], r["siege_ratio"]
    return (f"| {name} | {r['pred']:,} | {100 * p:.0f}% [{100 * r['precision_ci'][0]:.0f}–{100 * r['precision_ci'][1]:.0f}] | "
            f"{100 * rc:.0f}% [{100 * r['recall_ci'][0]:.0f}–{100 * r['recall_ci'][1]:.0f}] | {s:.2f}× [{r['siege_ci'][0]:.2f}–{r['siege_ci'][1]:.2f}] |")
print("# 100-game tower evaluation (scripts/moba/tower_eval_big.py)\n")
print("Held-out games 900–999, 128-tick open-loop windows every 64 ticks (4,017 windows), 1,561 real tower falls "
      "(overlapping windows count a fall up to twice). Precision/recall of falls within ±32 ticks; siege damage = dream/real HP "
      "for towers losing ≥ 800 HP in a window. 90% per-game bootstrap intervals.\n")
hdr = "| checkpoint | predicted falls | precision | recall | siege damage |\n|---|---|---|---|---|"
print("## Single checkpoints\n\n" + hdr)
for f in sorted(R.glob("*.json")):
    print(row(f.stem, json.loads(f.read_text())))
for d in sorted(p for p in R.iterdir() if p.is_dir()):
    print(f"\n## {d.name} (every checkpoint)\n\n" + hdr)
    for f in sorted(d.glob("tower_big_*.json")):
        print(row(f"{d.name} @{int(f.stem.split('_')[-1])}", json.loads(f.read_text())))
