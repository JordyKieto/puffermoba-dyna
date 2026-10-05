"""Real-game evaluation of several PPO fleets at once (one CPU pod, one arena at a time: two concurrent arenas hang).
Round-robin: for each arm, if rank 0 published a policy not yet evaluated, play 60 games per side vs the shipped net
(the fixed seeds every dyna-live eval uses) and append wins + behaviour (kills, deaths, creeps, game length) to
results/moba/dream/<arm>_eval.jsonl. usage: python runs/pod/abl_eval.py <eval pod> <arm> [<arm> ...]"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "tools")
os.environ["POD"] = sys.argv[1]
import pod
import torch
arms = sys.argv[2:]
c, bucket = pod.s3()
R = pod.R_REPO
last = {a: None for a in arms}
pod.push_code(); pod.run(f"mkdir -p {R}/eval_abl")
while True:
    did = False
    for a in arms:
        try:
            h = c.head_object(Bucket=bucket, Key=f"moba/runs/{a}_r0/policy.pt")
        except Exception:
            continue
        if h["LastModified"] == last[a]:
            continue
        last[a] = h["LastModified"]; did = True
        loc = Path(f"results/moba/dream/{a}_policies"); loc.mkdir(parents=True, exist_ok=True)
        tmp = loc / "latest.pt"; c.download_file(bucket, f"moba/runs/{a}_r0/policy.pt", str(tmp))
        ck = torch.load(tmp, map_location="cpu", weights_only=False); u = ck["update"]
        keep = loc / f"policy_u{u:04d}.pt"; tmp.rename(keep)
        t0 = time.time(); name = f"{a}_u{u:04d}"
        pod.rsync_to([str(keep)], f"{R}/eval_abl/{name}.pt")
        pod.run("ps -eo pid,ppid,cmd | awk '$2==1 && /spawn_main/ {print $1}' | xargs -r kill", check=False)
        pod.run(f"cd {R} && timeout 600 python -u -m mobawm.rl.arena --policy eval_abl/{name}.pt --games 60 --workers 32 --teams ppo "
                f"--opponents net --out eval_abl/{name}.jsonl > eval_abl/{name}.log 2>&1", check=False)
        g = [json.loads(l) for l in pod.run(f"cat {R}/eval_abl/{name}.jsonl 2>/dev/null", capture=True, check=False).splitlines() if l.strip()]
        rec = dict(update=u, fleet_steps=ck.get("agent_steps"), eval_s=round(time.time() - t0), t=time.strftime("%H:%M:%S"))
        for side in ("radiant", "dire"):
            x = [r for r in g if r["side"] == side]
            if not x:
                continue
            m = lambda k: round(sum(r[k] for r in x) / len(x), 2)
            rec.update({f"{side}_wins": sum(r["won"] for r in x), f"{side}_games": len(x), f"{side}_towers_ours": round(5 * m("ours_towers_killed"), 2),
                        f"{side}_kills": m("ours_kills"), f"{side}_deaths": m("ours_deaths"), f"{side}_creeps": m("ours_creeps_killed"),
                        f"{side}_ticks": m("ticks"), f"{side}_net_level": m("theirs_level")})
        with open(f"results/moba/dream/{a}_eval.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['t']} {a:11s} u{u:3d}: radiant {rec.get('radiant_wins')}/60 towers {rec.get('radiant_towers_ours')} kills {rec.get('radiant_kills')} "
              f"deaths {rec.get('radiant_deaths')} creeps {rec.get('radiant_creeps')} ticks {rec.get('radiant_ticks')} | dire {rec.get('dire_wins')}/60", flush=True)
    if not did:
        time.sleep(15)
