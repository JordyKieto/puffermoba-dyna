"""Dyna loop, stage 3: distributed dream PPO (FedAvg via Upstash) on the GPU pool that just trained the world model
(each pod starts dream episodes from its own shard of real games, already on the pod). Rank 0 uploads policy.pt every
update; runs/pod/eval_loop.py plays it in the real game. usage:
python runs/pod/ppo_pool.py <exp> <pool prefix> <N pods> <local WM dir> <local init policy> <updates> <envs> [reward flags]
env: POD_OFFSET (first pod index), FLEET_SHARDS + EXTRA_CORPORA (which shard each pod holds, as in tools/fleet.py)"""
import os, sys, json, time, subprocess
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from mobawm.wm.train import parse_games
from mobawm.fleet_link import clear_sync_keys
exp, prefix, N, wm, init, updates, envs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
reward = " ".join(sys.argv[8:])
EP = 1400
per = updates * envs * 5 * EP                                   # agent steps per pod (ours = 5 heroes per game)
TRAIN = parse_games("0-899,1000-1139,1160-1309,1330-1619,1660-1809,1830-1979,6000-6259,6270-6529,6540-6799")
# the shards the pods already hold: same file list as tools/fleet.py (EXTRA_CORPORA included), pod <prefix><OFF + r>
# holds shard OFF + r of SHARDS (POD_OFFSET / FLEET_SHARDS env, default 0 / N)
EXTRA = [(c.split(":")[0], parse_games(c.split(":")[1])) for c in os.environ.get("EXTRA_CORPORA", "").split(";") if c]
for _, g in EXTRA:
    TRAIN |= g
extra_firsts = sorted({g - g % 10 for _, gs in EXTRA for g in gs})
files = [f for f in list(range(0, 1000, 10)) + list(range(1000, 2000, 10)) + list(range(6000, 6810, 10)) + extra_firsts
         if set(range(f, f + 10)) <= TRAIN]
OFF, NS = int(os.environ.get("POD_OFFSET", 0)), int(os.environ.get("FLEET_SHARDS", N))
launch = time.strftime("%Y%m%d%H%M%S")
print(f"{exp}: {N} pods x {updates} updates x {envs} games ({per / 1e6:.0f}M steps/pod); launch {launch}; cleared {clear_sync_keys(exp)}", flush=True)
procs = []
for r in range(N):
    games = ",".join(f"{f}-{f + 9}" for f in files[OFF + r::NS])
    code = f'''
import sys, os, json; sys.path.insert(0, "tools"); import pod
R = pod.R_REPO; out = f"{{R}}/runs/dream_ppo/{exp}/r{r}"
pod.run("tmux kill-server 2>/dev/null; pkill -f [m]obawm.wm.train; pkill -f [e]val_watch; pkill -f [t]ower_eval_big; pkill -f [m]obawm.rl.dream_ppo; true", check=False)
pod.push_code(); pod.run(f"mkdir -p {{out}} {{R}}/runs/token_wm/wm_{exp}/ready {{R}}/runs/token_wm/wm_{exp}/incoming {{R}}/runs/dream_ppo/init_{exp}/reset {{R}}/runs/dream_ppo/init_{exp}/anchor")
pod.rsync_to(["{wm}/model.pt", "{wm}/stats.pt"], f"{{R}}/runs/token_wm/wm_{exp}/")
pod.rsync_to(["{init}"], f"{{R}}/runs/dream_ppo/init_{exp}/policy.pt")
if {r} == 0:
    pol = pod.REPO / "runs/pod/.s3_policy_{exp}.json"; pol.write_text(json.dumps(pod.s3_upload_policy("{exp}_r0", days=3))); pol.chmod(0o600)
    pod.rsync_to([pol], f"{{out}}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {{out}}/.s3_policy.json")
cmd = (f"cd {{R}} && . ./.pod_secrets && python -u -m mobawm.rl.dream_ppo --wm runs/token_wm/wm_{exp} --dense data/moba/full/dense "
       f"--games {games} --envs {envs} --episode {EP} --rollout {EP} --minutes 0 --steps {per} --device cuda --opponent net --arch puffer "
       f"--lr 0.0003 --epochs 2 --minibatches 16 --gpu-opp {reward} --init-policy runs/dream_ppo/init_{exp}/policy.pt --seed {r} "
       f"--wm-watch runs/token_wm/wm_{exp}/ready --reset-watch runs/dream_ppo/init_{exp}/reset "
       f"--fleet-exp {exp} --fleet-rank {r} --fleet-world {N} --fleet-launch {launch} --name {exp}/r{r}"
       + (" --wandb --group dyna-loop" if {r} == 0 else "") + f" > {{out}}/train.log 2>&1")
pod.run(f"cat > {{out}}/launch.sh <<'EOT'\\n{{cmd}}\\nEOT")
pod.run(f"tmux new-session -d -s ppo 'bash {{out}}/launch.sh'")
print("rank {r} LAUNCHED", flush=True)
'''
    procs.append(subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, POD=f"{prefix}{OFF + r}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
bad = 0
for r, p in enumerate(procs):
    o, _ = p.communicate()
    ok = "LAUNCHED" in o
    bad += not ok
    if not ok: print(f"rank {r} FAILED: {o[-300:]}", flush=True)
print(f"{exp}: {N - bad}/{N} ranks launched", flush=True)
