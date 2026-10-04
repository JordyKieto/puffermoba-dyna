"""One pod of a world-model fleet (runs/pod/fleet.py starts one of these per rank, each with its own POD name).
create -> bootstrap -> data (its shard of training files + the held-out games) -> train as fleet rank R.
usage: POD=<exp><r> python tools/fleet_rank.py <exp> <rank> <world> <gpu> <batch> <updates> <launch> <games> <init> [extra]"""
import sys, os, argparse, time; sys.path.insert(0, 'tools')
import pod
exp, rank, world, gpu, batch, n, launch, games, init = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], \
    int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8], sys.argv[9]
extra = sys.argv[10:]
t0 = time.time()
SETUP = os.environ.get("SKIP_SETUP") != "1"          # relaunch on the same pods: skip create/bootstrap/data
if SETUP:
    pod.cmd_create(argparse.Namespace(gpu=gpu, name=f"{exp}{rank}", confirm=True, cpu=0))
    pod.cmd_bootstrap(argparse.Namespace())
R = pod.R_REPO; D = f"{R}/data/moba"
from mobawm.wm.train import parse_games
held = parse_games("900-999,1140-1159,1310-1329,1620-1659,1810-1829,1980-1999")
need = parse_games(games) | held
if SETUP: pod.run("command -v s5cmd >/dev/null || (cd /tmp && curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/"
        "s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd && mv s5cmd /usr/local/bin/)")
from concurrent.futures import ThreadPoolExecutor
jobs = [(f"moba/{c}/", f"{D}/{c}", need) for c in ("mix1k", "mixrand1k", "dyna2")]
if rank == 0:
    jobs.append(("moba/cache/scenarios_heldout_v7/", f"{D}/scenarios_heldout", None))
if SETUP:
    with ThreadPoolExecutor(len(jobs)) as ex:
        list(ex.map(lambda j: pod.s3_sync_to_pod(*j), jobs))
pod.run(f"cd {D} && mkdir -p full/dense && for c in mix1k mixrand1k dyna2; do ls $c/dense/*.mbs >/dev/null 2>&1 && ln -sf {D}/$c/dense/*.mbs full/dense/; done; mkdir -p {R}/runs/token_wm/init")
pod.rsync_to([f"{init}/model.pt", f"{init}/stats.pt"], f"{R}/runs/token_wm/init/")
if rank == 0:
    pod.run(f"cd {R} && tools/build.sh moba-fork > /tmp/fork_build.log 2>&1")
print(f"rank {rank}: ready in {time.time() - t0:.0f}s", flush=True)
name = f"{exp}_r{rank}"
# rank 0 validates on the full held-out sets; followers never validate, so they load only 2 held-out games
val = ("--held-games 900-999 --held2-games 1140-1159,1310-1329,1620-1659,1810-1829,1980-1999 --held2-name random "
       "--scen-heldout data/moba/scenarios_heldout --event-eval") if rank == 0 else "--held-games 900-901"
main = (f"--data data/moba/full/dense --train-games {games} {val} --init runs/token_wm/init "
        f"--baseline --rollout --bf16 --batch {batch} --chunk 16 --window 16 --burn 4 --max-span 4096 "
        "--span-mu 6 --span-sigma 3 --decay-halflife 64 --decay-floor 0.02 --ckpt-every 250 --val-every 250 --val-batches 4 "
        f"--eval-batches 4 --rollouts 4 --log-every 25 --max-updates {n} --minutes 0 --skip-val0 "
        f"--fleet-exp {exp} --fleet-rank {rank} --fleet-world {world} --fleet-launch {launch}" + (" --wandb" if rank == 0 else ""))
os.environ["LOAD_WORKERS_REMOTE"] = "4"
pod.cmd_train(argparse.Namespace(name=name, train_args=main.split() + extra))
if rank == 0:
    pod.run(f"cd {R} && (tmux kill-session -t watch 2>/dev/null; true) && tmux new-session -d -s watch 'python -u scripts/moba/eval_watch.py --run runs/token_wm/{name} --until {n} > runs/token_wm/{name}/eval_watch.log 2>&1'")
print(f"rank {rank}: {name} LAUNCHED in {time.time() - t0:.0f}s", flush=True)
