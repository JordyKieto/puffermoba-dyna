"""Generated table of the dream-PPO reward sweep (results/moba/dream/{rw,gb,g3}/<arm>/): real games vs the shipped net
(radiant / dire: wins, towers per game for the team) and the last dream towers/episode (ours vs net).
    .venv/bin/python scripts/moba/ppo_sweep_table.py > results/moba/dream/SWEEP_TABLE.md"""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results/moba/dream"
print("# Dream-PPO reward sweep: real games vs the shipped net\n")
print("60 games per side per arm (arena seeds 50000000+, max 6,000 ticks). Towers = team towers per game "
      "(the arena logs a per-hero mean; x5). Dream towers = towers per dream episode, mean of the last 5 updates.\n")
print("| generation | arm | radiant wins | radiant towers/game | dire wins | dire towers ours / net | dream towers/ep ours vs net | steps (M) |")
print("|---|---|---|---|---|---|---|---|")
tot = {"radiant": [0, 0], "dire": [0, 0]}
for gen in ("rw", "gb", "g3"):
    for d in sorted((R / gen).glob("*")):
        f = d / "arena.jsonl"
        if not f.exists():
            continue
        G = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        cell = {}
        for side in ("radiant", "dire"):
            g = [x for x in G if x["side"] == side]
            if g:
                w = sum(x["won"] for x in g); tot[side][0] += w; tot[side][1] += len(g)
                cell[side] = (w, len(g), 5 * sum(x["ours_towers_killed"] for x in g) / len(g), 5 * sum(x["theirs_towers_killed"] for x in g) / len(g))
        log = (d / "train.log").read_text().splitlines() if (d / "train.log").exists() else []
        ups = [l for l in log if l.startswith("upd")][-5:]
        m = [re.search(r"ours ([\d.]+) vs net ([\d.]+)", l) for l in ups]; m = [x for x in m if x]
        dream = f"{sum(float(x.group(1)) for x in m) / len(m):.3f} vs {sum(float(x.group(2)) for x in m) / len(m):.3f}" if m else "—"
        st = re.search(r"([\d.]+)M steps", ups[-1]).group(1) if ups else "—"
        ra, di = cell.get("radiant"), cell.get("dire")
        print(f"| {gen} | {d.name} | {f'{ra[0]}/{ra[1]}' if ra else '—'} | {f'{ra[2]:.1f}' if ra else '—'} | {f'{di[0]}/{di[1]}' if di else '—'} | "
              f"{f'{di[2]:.1f} / {di[3]:.1f}' if di else '—'} | {dream} | {st} |")
print(f"\n**Total:** radiant {tot['radiant'][0]} wins in {tot['radiant'][1]} games; dire {tot['dire'][0]} wins in {tot['dire'][1]} games.")
