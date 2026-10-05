"""Tower false-death penalty smoke: one A100, two arms in parallel (control vs --tower-fp-weight W), 300 updates from
P1 @1000 on one fleet shard, then both scored by the 100-game tower eval.  usage: python runs/pod/fp_smoke.py <POD> <W>"""
import sys, os, argparse, time; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
os.environ["POD"] = sys.argv[1]
import pod
from mobawm.wm.train import parse_games
W = sys.argv[2]; t0 = time.time()
pod.cmd_create(argparse.Namespace(gpu="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe", name=sys.argv[1], confirm=True, cpu=0))
pod.cmd_bootstrap(argparse.Namespace())
R = pod.R_REPO; D = f"{R}/data/moba"
T = parse_games("0-899,1000-1139,1160-1309,1330-1619,1660-1809,1830-1979,6000-6259,6270-6529,6540-6799")
files = [f for f in list(range(0, 1000, 10)) + list(range(1000, 2000, 10)) + list(range(6000, 6810, 10)) if set(range(f, f + 10)) <= T]
games = ",".join(f"{f}-{f + 9}" for f in files[0::20])
need = parse_games(games) | parse_games("900-999")
pod.run("command -v s5cmd >/dev/null || (cd /tmp && curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd && mv s5cmd /usr/local/bin/)")
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(3) as ex:
    list(ex.map(lambda c: pod.s3_sync_to_pod(f"moba/{c}/", f"{D}/{c}", need), ("mix1k", "mixrand1k", "dyna2")))
pod.run(f"cd {D} && mkdir -p full/dense && for c in mix1k mixrand1k dyna2; do ls $c/dense/*.mbs >/dev/null 2>&1 && ln -sf {D}/$c/dense/*.mbs full/dense/; done; mkdir -p {R}/runs/token_wm/init")
pod.rsync_to(["runs/token_wm/p1_at1000/model.pt", "runs/token_wm/p1_at1000/stats.pt"], f"{R}/runs/token_wm/init/")
print(f"ready in {time.time() - t0:.0f}s", flush=True)
base = (f"--data data/moba/full/dense --train-games {games} --held-games 900-909 --init runs/token_wm/init --baseline --rollout --bf16 "
        "--batch 384 --chunk 16 --window 16 --burn 4 --max-span 4096 --span-mu 6 --span-sigma 3 --decay-halflife 64 --decay-floor 0.02 "
        "--lr 1e-4 --compile --ckpt-every 300 --val-every 100000 --val-batches 1 --eval-batches 1 --rollouts 1 --log-every 25 "
        "--max-updates 300 --minutes 0 --skip-val0")
for name, extra in (("fp_control", ""), (f"fp_w{W}", f" --tower-fp-weight {W}")):
    pod.cmd_train(argparse.Namespace(name=name, train_args=(base + extra).split()))
    pod.run(f"cd {R} && tmux new-session -d -s w_{name} 'python -u scripts/moba/eval_watch.py --run runs/token_wm/{name} --until 300 > runs/token_wm/{name}/eval_watch.log 2>&1'")
print(f"LAUNCHED both arms in {time.time() - t0:.0f}s", flush=True)
