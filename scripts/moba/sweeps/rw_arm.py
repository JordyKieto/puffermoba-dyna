"""One reward-sweep arm on its own pod: create -> bootstrap -> starts data + calibrated dream WM + arm C -> dream PPO
(vs the net, ours on a random side) -> 120 real games vs the net (60 per side). usage: POD=<pod> python runs/pod/rw_arm.py <arm> [flags]"""
import sys, os, json, argparse, time; sys.path.insert(0, 'tools')
import pod
from racewm import secrets
arm, flags = sys.argv[1], " ".join(sys.argv[2:]); t0 = time.time()
SETUP = os.environ.get("SKIP_SETUP") != "1"          # relaunch on an existing pod: data, WM and arm C are already there
if SETUP:
    pod.cmd_create(argparse.Namespace(gpu=os.environ.get("GPUS", "NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S"), name=os.environ["POD"], confirm=True, cpu=0))
    pod.cmd_bootstrap(argparse.Namespace())
R = pod.R_REPO; D = f"{R}/data/moba"
pod.push_code()                                        # in-place relaunches need the current code too
rand = set(); [rand.update(range(a, a + 20)) for a in (1000, 1160, 1330, 1660, 1830)]
if SETUP: pod.run("command -v s5cmd >/dev/null || (cd /tmp && curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd && mv s5cmd /usr/local/bin/)")
if SETUP: pod.s3_sync_to_pod("moba/mixrand1k/", f"{D}/mixrand1k", rand)
if SETUP: pod.s3_sync_to_pod("moba/mix1k/", f"{D}/mix1k", set(range(100)))
pod.run(f"cd {D} && mkdir -p starts_mix && ln -sf {D}/mix1k/dense/*.mbs {D}/mixrand1k/dense/*.mbs starts_mix/ && mkdir -p {R}/runs/token_wm/wm {R}/runs/dream_ppo/armC {R}/results/moba/dream")
if SETUP: pod.rsync_to(["runs/token_wm/fleet20e_at500/model.pt", "runs/token_wm/fleet20e_at500/stats.pt"], f"{R}/runs/token_wm/wm/")
if SETUP: pod.rsync_to(["runs/dream_ppo/p7/puffer_ours/policy.pt"], f"{R}/runs/dream_ppo/armC/")
sec = pod.REPO / f"runs/pod/.pod_secrets_rw_{arm}"; sec.write_text(f"export WANDB_API_KEY='{secrets.get('WANDB_API_KEY')}'\n"); sec.chmod(0o600)
pod.rsync_to([sec], f"{R}/.pod_secrets"); sec.unlink(); pod.run(f"chmod 600 {R}/.pod_secrets")
episode = "128"
fl = flags.split()
if "--episode" in fl:
    i = fl.index("--episode"); episode = fl[i + 1]; del fl[i:i + 2]
init = [] if "--scratch" in fl else ["--init-policy", "runs/dream_ppo/armC/policy.pt"]
if "--init-local" in fl:                              # start from a local policy file (e.g. a previous generation's best)
    i = fl.index("--init-local"); src = fl[i + 1]; del fl[i:i + 2]
    pod.run(f"mkdir -p {R}/runs/dream_ppo/init_{arm}")
    pod.rsync_to([src], f"{R}/runs/dream_ppo/init_{arm}/")
    init = ["--init-policy", f"runs/dream_ppo/init_{arm}/" + os.path.basename(src)]
fl = [x for x in fl if x != "--scratch"]
opp = "net"
if "--opponent" in fl:
    i = fl.index("--opponent"); opp = fl[i + 1]; del fl[i:i + 2]
common = (f"--wm runs/token_wm/wm --dense data/moba/starts_mix --games 0-99,1000-1019,1160-1179,1330-1349,1660-1679,1830-1849 "
          f"--envs 64 --episode {episode} --rollout {episode} --minutes 0 --steps 1695435 --device cuda --opponent {opp} --arch puffer "
          "--lr 0.0003 --epochs 2 --minibatches 2 ")
name = f"{os.environ.get('GEN', 'rw')}_{arm}"; out = f"{R}/runs/dream_ppo/rw/{arm}"
pod.run(f"mkdir -p {out}")
pol = pod.REPO / f"runs/pod/.s3_policy_rw_{arm}.json"; pol.write_text(json.dumps(pod.s3_upload_policy(name, days=3))); pol.chmod(0o600)
pod.rsync_to([pol], f"{out}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {out}/.s3_policy.json")
cmd = (f"cd {R} && . ./.pod_secrets && python -u -m mobawm.rl.dream_ppo {common}{' '.join(init + fl)} --name rw/{arm} --wandb --group RW-reward-sweep "
       f"2>&1 | tee {out}/train.log; python -u -m mobawm.rl.arena --policy {out}/policy.pt --games 60 --workers 8 --teams ppo --opponents net "
       f"--out {out}/arena.jsonl > {out}/arena.log 2>&1; echo EVAL DONE >> {out}/arena.log")
pod.run(f"cat > {out}/launch.sh <<'EOT'\n{cmd}\nEOT")
pod.run(f"tmux kill-server 2>/dev/null; true", check=False)
pod.run(f"tmux new-session -d -s {name} 'bash {out}/launch.sh'")
print(f"{arm} LAUNCHED in {time.time() - t0:.0f}s", flush=True)
