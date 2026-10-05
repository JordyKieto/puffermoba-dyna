"""Dyna loop, stage 1: real games of the given policies vs the shipped net (alternating sides), recorded in the corpus
format on the CPU pods (several collector processes per pod), replayed + verified to dense, uploaded to
s3://.../moba/runs/<run>/{dense,logs}. usage: python runs/pod/dyna_collect.py <run> <first game> <games per policy>
<cpu pod>[,<cpu pod>...] <policy> [<policy> ...]"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "tools")
run, first, per, cpus, pols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(","), sys.argv[5:]
G = per * len(pols); PROCS = 6                          # collector processes per CPU pod
chunks = [(i, min(G, i + 10)) for i in range(0, G, 10)]  # 10-game slices = one log/dense file each
jobs_per_pod = [chunks[i::len(cpus)] for i in range(len(cpus))]
def on_pod(ip):
    i, cpu = ip
    env = dict(os.environ, POD=cpu)
    code = f'''
import sys, os, json; sys.path.insert(0, "tools"); import pod
R = pod.R_REPO; out = f"{{R}}/dyna/{run}"
pod.push_code(); pod.run(f"mkdir -p {{out}}/pols")
pod.rsync_to({pols!r}, f"{{out}}/pols/")
pod.rsync_to(["data/moba/mix1k/logs/games.0000900.mbl"], f"{{out}}/")              # header template for the logs
pol = pod.REPO / "runs/pod/.s3_policy_{run}_{i}.json"; pol.write_text(json.dumps(pod.s3_upload_policy("{run}", days=3))); pol.chmod(0o600)
pod.rsync_to([pol], f"{{out}}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {{out}}/.s3_policy.json")
pp = " ".join("--policy " + out + "/pols/" + os.path.basename(p) for p in {pols!r})
cmds = [f"python -u scripts/moba/collect_policy_games.py {{pp}} --games-per-policy {per} --first-game {first} --seed-base {91_000_000 + first} "
        f"--slice {{lo}}:{{hi}} --replay-jobs 2 --template-log {{out}}/games.0000900.mbl --out {{out}} > {{out}}/collect_{{lo}}.log 2>&1" for lo, hi in {jobs_per_pod[i]!r}]
batches = [cmds[k::{PROCS}] for k in range({PROCS})]
script = " & ".join("(" + " ; ".join(b) + ")" for b in batches if b) + " ; wait"
pod.run(f"cd {{R}} && bash -c '{{script}}'", check=False)
up = "from pathlib import Path; from mobawm.s3persist import persist, flush; o = Path('" + out + "'); fs = sorted(o.glob('dense/*.mbs')) + sorted(o.glob('logs/*.mbl')) + sorted(o.glob('games_*.json')); [persist(f) for f in fs]; flush(); print(len(fs), 'files uploaded')"
print(pod.run(f"cd {{R}} && python -c \\"{{up}}\\"", capture=True))
print(pod.run(f"grep -h 'state hash mismatches' {{out}}/collect_*.log | grep -vc 'mismatches 0 '", capture=True, check=False).strip(), "files with mismatches")
'''
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    return cpu, (r.stdout + r.stderr)[-600:]
t0 = time.time()
with ThreadPoolExecutor(len(cpus)) as ex:
    for cpu, out in ex.map(on_pod, enumerate(cpus)):
        print(f"[{cpu}] {out.strip()}", flush=True)
print(f"{run}: {G} games ({per} per policy x {len(pols)}), games {first}-{first + G - 1}, {time.time() - t0:.0f}s", flush=True)
