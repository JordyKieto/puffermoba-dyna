"""One sweep arm on an existing pod: stop everything, add shard 0 (the fp_smoke shard) + P1 @1000, train <updates>
with the given flags, score with the 100-game tower eval. usage: POD=<pod> python runs/pod/sweep_rank.py <name> <updates> [flags]"""
import sys, os, argparse; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import pod
from mobawm.wm.train import parse_games
name, n, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
T = parse_games("0-899,1000-1139,1160-1309,1330-1619,1660-1809,1830-1979,6000-6259,6270-6529,6540-6799")
files = [f for f in list(range(0, 1000, 10)) + list(range(1000, 2000, 10)) + list(range(6000, 6810, 10)) if set(range(f, f + 10)) <= T]
games = ",".join(f"{f}-{f + 9}" for f in files[0::20])
R = pod.R_REPO; D = f"{R}/data/moba"
pod.run("tmux kill-server 2>/dev/null; pkill -f mobawm.wm.train; pkill -f eval_watch; pkill -f tower_eval_big; true", check=False)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(3) as ex:
    list(ex.map(lambda c: pod.s3_sync_to_pod(f"moba/{c}/", f"{D}/{c}", parse_games(games) | parse_games("900-999")), ("mix1k", "mixrand1k", "dyna2")))
pod.run(f"cd {D} && for c in mix1k mixrand1k dyna2; do ln -sf {D}/$c/dense/*.mbs full/dense/ 2>/dev/null; done; mkdir -p {R}/runs/token_wm/init_p1")
pod.rsync_to(["runs/token_wm/p1_at1000/model.pt", "runs/token_wm/p1_at1000/stats.pt"], f"{R}/runs/token_wm/init_p1/")
pod.push_code()
base = (f"--data data/moba/full/dense --train-games {games} --held-games 900-909 --init runs/token_wm/init_p1 --baseline --rollout --bf16 "
        "--batch 384 --chunk 16 --window 16 --burn 4 --max-span 4096 --span-mu 6 --span-sigma 3 --decay-halflife 64 --decay-floor 0.02 "
        f"--compile --ckpt-every {n} --val-every 100000 --val-batches 1 --eval-batches 1 --rollouts 1 --log-every 25 "
        f"--max-updates {n} --minutes 0 --skip-val0")
os.environ["LOAD_WORKERS_REMOTE"] = "4"
pod.cmd_train(argparse.Namespace(name=name, train_args=base.split() + extra))
pod.run(f"cd {R} && tmux new-session -d -s watch 'python -u scripts/moba/eval_watch.py --run runs/token_wm/{name} --until {n} > runs/token_wm/{name}/eval_watch.log 2>&1'")
print(f"{name} LAUNCHED", flush=True)
