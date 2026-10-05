"""Real-game eval for many dream-PPO arms at once: whenever an arm's run (moba/runs/<gen>_<arm>/) uploads a new
policy.pt, play 60 games per side vs the shipped net on one of the CPU eval pods (one worker thread per pod) and append
to results/moba/dream/<gen>_eval.jsonl. usage: python runs/pod/eval_multi.py <gen> <arms module> <cpu pod> [cpu pod ...]"""
import json, os, queue, subprocess, sys, threading, time, importlib
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, "runs/pod")
gen, mod, cpus = sys.argv[1], sys.argv[2], sys.argv[3:]
ARMS = list(importlib.import_module(mod).ARMS)
import pod
import torch
c, bucket = pod.s3()
out = Path(f"results/moba/dream/{gen}_eval.jsonl"); pol_dir = Path(f"results/moba/dream/{gen}_policies")
q, seen, lock = queue.Queue(), {}, threading.Lock()
def worker(cpu):
    env = dict(os.environ, POD=cpu)
    subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'tools'); import pod; pod.push_code()"], env=env, capture_output=True)
    while True:
        arm, path, u, steps = q.get()
        R = pod.R_REPO; t0 = time.time()
        code = (f"import sys, json; sys.path.insert(0,'tools'); import pod; R=pod.R_REPO; pod.run(f'mkdir -p {{R}}/ev/{arm}'); "
                f"pod.rsync_to(['{path}'], f'{{R}}/ev/{arm}/'); "
                f"pod.run(f'cd {{R}} && python -u -m mobawm.rl.arena --policy ev/{arm}/{path.name} --games 60 --workers 32 --teams ppo --opponents net "
                f"--out ev/{arm}/u{u:04d}.jsonl > ev/{arm}/u{u:04d}.log 2>&1', check=False); print(pod.run(f'cat {{R}}/ev/{arm}/u{u:04d}.jsonl', capture=True))")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        games = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        rec = dict(arm=arm, update=u, steps=steps, eval_s=round(time.time() - t0), pod=cpu)
        for side in ("radiant", "dire"):
            g = [x for x in games if x["side"] == side]
            if g:
                rec.update({f"{side}_wins": sum(x["won"] for x in g), f"{side}_games": len(g),
                            f"{side}_towers_ours": round(5 * sum(x["ours_towers_killed"] for x in g) / len(g), 2),
                            f"{side}_towers_theirs": round(5 * sum(x["theirs_towers_killed"] for x in g) / len(g), 2),
                            f"{side}_return": round(sum(x["return_ours"] for x in g) / len(g), 2)})
        with lock, out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
for cpu in cpus:
    threading.Thread(target=worker, args=(cpu,), daemon=True).start()
while True:
    for arm in ARMS:
        key = f"{pod.S3_PREFIX}/runs/{gen}_{arm}/policy.pt"
        try:
            h = c.head_object(Bucket=bucket, Key=key)
        except Exception:
            continue
        if seen.get(arm) == h["LastModified"] or any(x[0] == arm for x in list(q.queue)):
            continue
        seen[arm] = h["LastModified"]
        d = pol_dir / arm; d.mkdir(parents=True, exist_ok=True)
        tmp = d / "latest.pt"; c.download_file(bucket, key, str(tmp))
        ck = torch.load(tmp, map_location="cpu", weights_only=False)
        keep = d / f"policy_u{ck.get('update', 0):04d}.pt"; tmp.rename(keep)
        q.put((arm, keep.resolve(), ck.get("update", 0), ck.get("agent_steps", 0)))
    time.sleep(30)
