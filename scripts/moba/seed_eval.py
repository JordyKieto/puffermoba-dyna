"""Real-game evaluation on a named seed set, for the paper: N games per side against the shipped net, played on a CPU
pod in chunks of 60 per side (one arena per pod at a time: two concurrent arenas, or one 300-game arena, hang), one
structured row per (team, seed set) appended to results/moba/paper/seed_evals.jsonl, per-game records kept in
results/moba/paper/games/<label>_<set>.jsonl.

Seed sets (game seed = base + 1000*chunk + i; the radiant and dire game of a pair share the seed):
  fixed    50000000, 60 per side   used after EVERY PPO update (selection signal; biased for the best checkpoint)
  fresh    60000000, 300 per side  used to confirm an anchor candidate (selection on it is repeated: mildly biased)
  holdout  70000000, 300 per side  never used for any choice: the reported number
  holdout2 80000000                a second untouched set (e.g. for a final 600-game estimate)

usage: POD=evc5 python scripts/moba/seed_eval.py --label ppol4_u69 --team ppo --policy <path> --set holdout [--per-side 300]
       POD=evc5 python scripts/moba/seed_eval.py --label shipped_net --team net --set holdout"""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import pod  # noqa: E402

SETS = {"fixed": 50_000_000, "fresh": 60_000_000, "holdout": 70_000_000, "holdout2": 80_000_000}
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--team", default="ppo", choices=("ppo", "net", "random", "script"))
ap.add_argument("--policy", default=""); ap.add_argument("--set", default="holdout", choices=tuple(SETS))
ap.add_argument("--per-side", type=int, default=300); ap.add_argument("--note", default="")
a = ap.parse_args()
R = pod.R_REPO
out = ROOT / "results/moba/paper"; (out / "games").mkdir(parents=True, exist_ok=True)
pod.run(f"mkdir -p {R}/seed_eval")
pol_arg, md5 = "--policy unused", None                  # the arena requires --policy; only team ppo loads it
if a.team == "ppo":
    md5 = hashlib.md5(Path(a.policy).read_bytes()).hexdigest()
    pod.rsync_to([a.policy], f"{R}/seed_eval/{a.label}.pt")
    pol_arg = f"--policy seed_eval/{a.label}.pt"
games, t0 = [], time.time()
for k in range(a.per_side // 60):
    base = SETS[a.set] + 1000 * k
    f = f"seed_eval/{a.label}_{a.set}_{k}.jsonl"
    pod.run("ps -eo pid,ppid,cmd | awk '$2==1 && /spawn_main/ {print $1}' | xargs -r kill", check=False)
    pod.run(f"cd {R} && timeout 600 python -u -m mobawm.rl.arena {pol_arg} --games 60 --workers 32 --seed-base {base} "
            f"--teams {a.team} --opponents net --out {f} > {f}.log 2>&1", check=False)
    g = [json.loads(l) for l in pod.run(f"cat {R}/{f} 2>/dev/null", capture=True, check=False).splitlines() if l.strip()]
    if len(g) != 120:
        sys.exit(f"chunk {k}: {len(g)} games instead of 120 (see {R}/{f}.log)")
    games += g
    print(f"{a.label} {a.set} chunk {k}: {len(games)} games, {time.time() - t0:.0f}s", flush=True)
(out / "games" / f"{a.label}_{a.set}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in games))
row = dict(label=a.label, team=a.team, policy=a.policy or None, policy_md5=md5, set=a.set, seed_base=SETS[a.set],
           opponent="shipped net", date=time.strftime("%Y-%m-%d %H:%M"), note=a.note)
for side in ("radiant", "dire"):
    x = [r for r in games if r["side"] == side]
    m = lambda key: round(sum(r[key] for r in x) / len(x), 3)
    row.update({f"{side}_wins": sum(r["won"] for r in x), f"{side}_losses": sum(r["lost"] for r in x),
                f"{side}_games": len(x), f"{side}_towers_ours": round(5 * m("ours_towers_killed"), 3),
                f"{side}_towers_theirs": round(5 * m("theirs_towers_killed"), 3), f"{side}_kills": m("ours_kills"),
                f"{side}_deaths": m("ours_deaths"), f"{side}_ticks": m("ticks")})
with (out / "seed_evals.jsonl").open("a") as fh:
    fh.write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)
