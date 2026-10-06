"""Distributed dream PPO (zelda-RL FedAvg via Upstash) on prepared sweep pods: the winning formula (game reward + win
+-100, 1,024-tick episodes), GPU obs + GPU shipped net, from the best gen-3 policy. Rank 0 evaluates in the real game.
usage: python runs/pod/ppo_fleet.py <exp> <total steps> <pod> [pod ...]"""
import os, sys, json, time; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from mobawm.fleet_link import clear_sync_keys
exp, total, pods = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
N = len(pods); per = total // N; launch = time.strftime("%Y%m%d%H%M%S")
INIT = "results/moba/dream/g3/J_envs128_4x_s2/policy.pt"
print(f"{exp}: {N} pods x {per / 1e6:.0f}M steps = {total / 1e6:.0f}M; launch {launch}; cleared {clear_sync_keys(exp)} sync keys", flush=True)
import subprocess
for r, p in enumerate(pods):
    code = f'''
import sys, os, json; sys.path.insert(0, "tools"); import pod
from mobawm import secrets
R = pod.R_REPO; out = f"{{R}}/runs/dream_ppo/{exp}/r{r}"
pod.push_code(); pod.run(f"mkdir -p {{out}} {{R}}/runs/dream_ppo/init_{exp}")
pod.rsync_to(["{INIT}"], f"{{R}}/runs/dream_ppo/init_{exp}/")
sec = pod.REPO / "runs/pod/.pod_secrets_{exp}_{r}"; sec.write_text("".join(          # from the Keychain
    f"export {{k}}='{{secrets.get(k)}}'\\n" for k in ("WANDB_API_KEY", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN") if secrets.get(k)))
sec.chmod(0o600); pod.rsync_to([sec], f"{{R}}/.pod_secrets"); sec.unlink(); pod.run(f"chmod 600 {{R}}/.pod_secrets")
pol = pod.REPO / "runs/pod/.s3_policy_{exp}_{r}.json"; pol.write_text(json.dumps(pod.s3_upload_policy("{exp}_r{r}", days=3))); pol.chmod(0o600)
pod.rsync_to([pol], f"{{out}}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {{out}}/.s3_policy.json")
cmd = (f"cd {{R}} && . ./.pod_secrets && python -u -m mobawm.rl.dream_ppo --wm runs/token_wm/wm --dense data/moba/starts_mix "
       f"--games 0-99,1000-1019,1160-1179,1330-1349,1660-1679,1830-1849 --envs 512 --episode 1024 --rollout 1024 --minutes 0 "
       f"--steps {per} --device cuda --opponent net --arch puffer --lr 0.0003 --epochs 2 --minibatches 16 --r-win 100 --gpu-opp "
       f"--init-policy runs/dream_ppo/init_{exp}/policy.pt --seed {r} --fleet-exp {exp} --fleet-rank {r} --fleet-world {N} --fleet-launch {launch} "
       f"--name {exp}/r{r}" + (" --wandb --group PPO-fleet" if {r} == 0 else "") + f" 2>&1 | tee {{out}}/train.log"
       + (f"; python -u -m mobawm.rl.arena --policy {{out}}/policy.pt --games 60 --workers 8 --teams ppo --opponents net --out {{out}}/arena.jsonl > {{out}}/arena.log 2>&1; echo EVAL DONE >> {{out}}/arena.log" if {r} == 0 else ""))
pod.run(f"cat > {{out}}/launch.sh <<'EOT'\\n{{cmd}}\\nEOT")
pod.run("tmux kill-server 2>/dev/null; pkill -f mobawm.rl.dream_ppo; true", check=False)
pod.run(f"tmux new-session -d -s ppo 'bash {{out}}/launch.sh'")
print("rank {r} LAUNCHED on {p}", flush=True)
'''
    subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, POD=p), stdout=open(f"runs/pod/{exp}_r{r}.log", "w"), stderr=subprocess.STDOUT)
print("launchers started", flush=True)
