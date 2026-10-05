"""More --tower-fp-weight arms on the fp_smoke pod (data + init already there), 2 at a time, 300 updates, 100-game eval."""
import sys, os, argparse; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
os.environ["POD"] = sys.argv[1]
import pod
from mobawm.wm.train import parse_games
T = parse_games("0-899,1000-1139,1160-1309,1330-1619,1660-1809,1830-1979,6000-6259,6270-6529,6540-6799")
files = [f for f in list(range(0, 1000, 10)) + list(range(1000, 2000, 10)) + list(range(6000, 6810, 10)) if set(range(f, f + 10)) <= T]
games = ",".join(f"{f}-{f + 9}" for f in files[0::20])
R = pod.R_REPO; pod.push_code()
base = (f"--data data/moba/full/dense --train-games {games} --held-games 900-909 --init runs/token_wm/init --baseline --rollout --bf16 "
        "--batch 384 --chunk 16 --window 16 --burn 4 --max-span 4096 --span-mu 6 --span-sigma 3 --decay-halflife 64 --decay-floor 0.02 "
        "--lr 1e-4 --compile --ckpt-every 300 --val-every 100000 --val-batches 1 --eval-batches 1 --rollouts 1 --log-every 25 "
        "--max-updates 300 --minutes 0 --skip-val0")
for W in sys.argv[2:]:
    name = f"fp_w{W}"
    pod.cmd_train(argparse.Namespace(name=name, train_args=(base + f" --tower-fp-weight {W}").split()))
    pod.run(f"cd {R} && (tmux kill-session -t w_{name} 2>/dev/null; true) && tmux new-session -d -s w_{name} 'python -u scripts/moba/eval_watch.py --run runs/token_wm/{name} --until 300 > runs/token_wm/{name}/eval_watch.log 2>&1'")
print("LAUNCHED", sys.argv[2:], flush=True)
