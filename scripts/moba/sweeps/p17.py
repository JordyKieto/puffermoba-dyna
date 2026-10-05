"""P16: dream PPO in fleet20c @1000 (tower false-death penalty WM) vs the shipped net (ours on a random side each game),
tower/win shaping; then 60 real games vs the net per arm (both sides). One A100, arms in parallel."""
import sys, os, json, argparse; sys.path.insert(0, 'tools')
os.environ["POD"] = sys.argv[1]
import pod
from racewm import secrets
pod.cmd_create(argparse.Namespace(gpu="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H200", name=sys.argv[1], confirm=True, cpu=0))
pod.cmd_bootstrap(argparse.Namespace())
R = pod.R_REPO; D = f"{R}/data/moba"
rand = set(); [rand.update(range(a, a + 20)) for a in (1000, 1160, 1330, 1660, 1830)]
pod.run("command -v s5cmd >/dev/null || (cd /tmp && curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd && mv s5cmd /usr/local/bin/)")
pod.s3_sync_to_pod("moba/mixrand1k/", f"{D}/mixrand1k", rand)
pod.s3_sync_to_pod("moba/mix1k/", f"{D}/mix1k", set(range(100)))
pod.run(f"cd {D} && mkdir -p starts_mix && ln -sf {D}/mix1k/dense/*.mbs {D}/mixrand1k/dense/*.mbs starts_mix/ && mkdir -p {R}/runs/token_wm/wm {R}/runs/dream_ppo/armC {R}/results/moba/dream")
pod.rsync_to(["runs/token_wm/" + sys.argv[2] + "/model.pt", "runs/token_wm/" + sys.argv[2] + "/stats.pt"], f"{R}/runs/token_wm/wm/")
pod.rsync_to(["runs/dream_ppo/p7/puffer_ours/policy.pt"], f"{R}/runs/dream_ppo/armC/")
sec = pod.REPO / "runs/pod/.pod_secrets_p16"; sec.write_text(f"export WANDB_API_KEY='{secrets.get('WANDB_API_KEY')}'\n"); sec.chmod(0o600)
pod.rsync_to([sec], f"{R}/.pod_secrets"); sec.unlink(); pod.run(f"chmod 600 {R}/.pod_secrets")
common = ("--wm runs/token_wm/wm --dense data/moba/starts_mix --games 0-99,1000-1019,1160-1179,1330-1349,1660-1679,1830-1849 "
          "--envs 64 --episode 128 --rollout 128 --minutes 0 --steps 1695435 --device cuda --opponent net --arch puffer "
          "--lr 0.0003 --epochs 2 --minibatches 2 ")
moderate = "--dmg-k 2 --dmg-boost 2 --fall-bonus 5 --fall-radius 8 --r-win 10 --def-scale 1"
strong = "--dmg-k 5 --dmg-boost 2 --fall-bonus 15 --fall-radius 8 --r-win 30 --def-scale 1"
T = sys.argv[1]
arms = {"armC_mod": f"--init-policy runs/dream_ppo/armC/policy.pt {moderate}",
        "armC_strong": f"--init-policy runs/dream_ppo/armC/policy.pt {strong}",
}
arena = "--games 60 --workers 20 --teams ppo --opponents net"
for arm, flags in arms.items():
    name = f"{T}_{arm}"; out = f"{R}/runs/dream_ppo/{T}/{arm}"
    pod.run(f"mkdir -p {out}")
    pol = pod.REPO / f"runs/pod/.s3_policy_{T}_{arm}.json"; pol.write_text(json.dumps(pod.s3_upload_policy(name, days=3))); pol.chmod(0o600)
    pod.rsync_to([pol], f"{out}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {out}/.s3_policy.json")
    cmd = (f"cd {R} && . ./.pod_secrets && python -u -m mobawm.rl.dream_ppo {common}{flags} --name {T}/{arm} --wandb --group P17-calibrated-WM-vsnet "
           f"2>&1 | tee {out}/train.log; python -u -m mobawm.rl.arena --policy {out}/policy.pt {arena} "
           f"--out results/moba/dream/arena_real_game_{T}_{arm}.jsonl > {out}/arena.log 2>&1; echo EVAL DONE >> {out}/arena.log")
    pod.run(f"cat > {out}/launch.sh <<'EOT'\n{cmd}\nEOT")
    pod.run(f"tmux new-session -d -s {name} 'bash {out}/launch.sh'")
base = f"{R}/runs/dream_ppo/{T}/armC_baseline"
pod.run(f"mkdir -p {base} && cd {R} && tmux new-session -d -s {T}_base 'python -u -m mobawm.rl.arena --policy runs/dream_ppo/armC/policy.pt {arena} "
        f"--out results/moba/dream/arena_real_game_{T}_armC_baseline.jsonl > {base}/arena.log 2>&1; echo EVAL DONE >> {base}/arena.log'")
print("P16 LAUNCHED", flush=True)
