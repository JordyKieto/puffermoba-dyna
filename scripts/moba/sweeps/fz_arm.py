"""Frozen-world-model control (E40): one rank of a 2-pod dream-PPO FedAvg fleet, the ppol4 recipe (lr 5e-5, death
penalty 0, KL 0.3 toward the start policy, which is also the fixed anchor; no resets, no anchor moves), in a world model
that is FROZEN for the whole run. Arms differ only in the world model (and the seed).
create -> bootstrap -> 100 start games (mix1k 0-99) + the arm's WM + u71 -> train.
usage: POD=<pod> python runs/pod/fz_arm.py <exp> <rank> <world> <launch> <local WM dir> <seed> <updates>"""
import sys, os, json, argparse, time; sys.path.insert(0, "tools")
from pathlib import Path
import pod
from mobawm import secrets
exp, rank, world, launch, wm, seed, updates = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
INIT = "results/moba/dream/ppol_policies/policy_u0071.pt"
REWARD = ("--r-win 100 --dmg-k 5 --dmg-boost 2 --fall-bonus 10 --fall-radius 8 --kill-k 2 --siege-k 0.02 --anc-k 50 "
          "--xp-scale 0.5 --death-scale 0")
EP, ENVS = 1400, 128
t0 = time.time()
if os.environ.get("SKIP_SETUP") != "1":
    pod.cmd_create(argparse.Namespace(gpu=os.environ.get("GPUS", "NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S"), name=os.environ["POD"], confirm=True, cpu=0))
    pod.cmd_bootstrap(argparse.Namespace())
    pod.run("command -v s5cmd >/dev/null || (cd /tmp && curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd && mv s5cmd /usr/local/bin/)")
R = pod.R_REPO; D = f"{R}/data/moba"
pod.push_code()
pod.s3_sync_to_pod("moba/mix1k/", f"{D}/mix1k", set(range(100)))
out = f"{R}/runs/dream_ppo/{exp}/r{rank}"
pod.run(f"mkdir -p {out} {R}/runs/token_wm/wm_{exp} {R}/runs/dream_ppo/init_{exp}")
pod.rsync_to([f"{wm}/model.pt", f"{wm}/stats.pt"], f"{R}/runs/token_wm/wm_{exp}/")
pod.rsync_to([INIT], f"{R}/runs/dream_ppo/init_{exp}/policy.pt")
sec = pod.REPO / f"runs/pod/.pod_secrets_{exp}_{rank}"           # W&B + Upstash (FedAvg between the ranks), from the Keychain
sec.write_text("".join(f"export {k}='{secrets.get(k)}'\n" for k in ("WANDB_API_KEY", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN") if secrets.get(k)))
sec.chmod(0o600)
try:
    pod.rsync_to([sec], f"{R}/.pod_secrets")
finally:
    sec.unlink(missing_ok=True)
pod.run(f"chmod 600 {R}/.pod_secrets")
if rank == 0:
    pol = pod.REPO / f"runs/pod/.s3_policy_{exp}.json"; pol.write_text(json.dumps(pod.s3_upload_policy(f"{exp}_r0", days=3))); pol.chmod(0o600)
    pod.rsync_to([pol], f"{out}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {out}/.s3_policy.json")
steps = updates * ENVS * 5 * EP
cmd = (f"cd {R} && . ./.pod_secrets && python -u -m mobawm.rl.dream_ppo --wm runs/token_wm/wm_{exp} --dense {D}/mix1k/dense "
       f"--games 0-99 --envs {ENVS} --episode {EP} --rollout {EP} --minutes 0 --steps {steps} --device cuda --opponent net "
       f"--arch puffer --lr 5e-5 --epochs 2 --minibatches 16 --gpu-opp {REWARD} --init-policy runs/dream_ppo/init_{exp}/policy.pt "
       f"--kl-anchor runs/dream_ppo/init_{exp}/policy.pt --kl-coef 0.3 --seed {seed * 100 + rank} "
       f"--fleet-exp {exp} --fleet-rank {rank} --fleet-world {world} --fleet-launch {launch} --name {exp}/r{rank}"
       + (f" --wandb --group E40-frozen-wm" if rank == 0 else "") + f" > {out}/train.log 2>&1")
pod.run(f"cat > {out}/launch.sh <<'EOT'\n{cmd}\nEOT")
pod.run("tmux kill-session -t ppo 2>/dev/null; true", check=False)
pod.run(f"tmux new-session -d -s ppo 'bash {out}/launch.sh'")
print(f"{exp} rank {rank} LAUNCHED in {time.time() - t0:.0f}s", flush=True)
