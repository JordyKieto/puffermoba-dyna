"""Real-game evaluation alongside a dream-PPO fleet: every time rank 0 uploads a new policy.pt to S3, download it,
keep a copy (policy_u<update>.pt), play 60 games per side vs the shipped net on the CPU eval pod, and append
wins / towers per side to results/moba/dream/<exp>_eval.jsonl. Runs on the Mac, in the background.
usage: python runs/pod/eval_loop.py <exp> [eval pod]"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "tools")
exp = sys.argv[1]; epod = sys.argv[2] if len(sys.argv) > 2 else "evalcpu"
os.environ["POD"] = epod
import pod
import torch
c, bucket = pod.s3()
key = f"{pod.S3_PREFIX}/runs/{exp}_r0/policy.pt"
loc = Path(f"results/moba/dream/{exp}_policies"); loc.mkdir(parents=True, exist_ok=True)
out = Path(f"results/moba/dream/{exp}_eval.jsonl")
R = pod.R_REPO
last = None
pod.push_code()
while True:
    try:
        head = c.head_object(Bucket=bucket, Key=key)
    except Exception:
        time.sleep(30); continue
    if head["LastModified"] == last:
        time.sleep(30); continue
    last = head["LastModified"]
    tmp = loc / "latest.pt"; c.download_file(bucket, key, str(tmp))
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    u, steps = ck.get("update"), ck.get("agent_steps")
    keep = loc / f"policy_u{u:04d}.pt"; tmp.rename(keep)
    t0 = time.time()
    pod.run(f"mkdir -p {R}/eval_{exp}")
    pod.rsync_to([str(keep)], f"{R}/eval_{exp}/")
    pod.run(f"cd {R} && python -u -m mobawm.rl.arena --policy eval_{exp}/{keep.name} --games 60 --workers 32 --teams ppo --opponents net "
            f"--out eval_{exp}/arena_u{u:04d}.jsonl > eval_{exp}/arena_u{u:04d}.log 2>&1", check=False)
    games = [json.loads(l) for l in pod.run(f"cat {R}/eval_{exp}/arena_u{u:04d}.jsonl", capture=True).splitlines() if l.strip()]
    rec = dict(update=u, fleet_steps=steps, eval_s=round(time.time() - t0))
    for side in ("radiant", "dire"):
        g = [x for x in games if x["side"] == side]
        if g:
            rec[f"{side}_wins"] = sum(x["won"] for x in g); rec[f"{side}_games"] = len(g)
            rec[f"{side}_towers_ours"] = round(5 * sum(x["ours_towers_killed"] for x in g) / len(g), 2)
            rec[f"{side}_towers_theirs"] = round(5 * sum(x["theirs_towers_killed"] for x in g) / len(g), 2)
            rec[f"{side}_return"] = round(sum(x["return_ours"] for x in g) / len(g), 2)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
