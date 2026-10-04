"""Launch a world-model fleet: N pods, each training on its own shard of the training files (every 10-game file goes to
rank i mod N, so every corpus is in every shard), averaging weights every --fleet-every updates via Upstash (FedAvg,
zelda-RL's pattern). Rank 0 also validates, checkpoints and runs the 100-game tower eval watcher.
usage: python tools/fleet.py <exp> <N> <gpu> <batch per pod> <updates> <init run dir> [extra train flags]"""
import sys, os, subprocess, time
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from mobawm.wm.train import parse_games
from mobawm.fleet_link import clear_sync_keys
exp, N, gpu, batch, n, init = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
extra = sys.argv[7:]
TRAIN = parse_games("0-899,1000-1139,1160-1309,1330-1619,1660-1809,1830-1979,6000-6259,6270-6529,6540-6799")
files = [f for f in list(range(0, 1000, 10)) + list(range(1000, 2000, 10)) + list(range(6000, 6810, 10))
         if set(range(f, f + 10)) <= TRAIN]
launch = time.strftime("%Y%m%d%H%M%S")
NS = int(os.environ.get("FLEET_SHARDS", N))       # shards (a smoke test can run 2 pods with the 20-pod shard size)
print(f"{exp}: {N} pods x {gpu}, batch {batch}/pod, {n} updates, {len(files)} training files ({len(files) // NS}-{-(-len(files) // NS)} per pod, {NS} shards), "
      f"launch {launch}; cleared {clear_sync_keys(exp)} old sync keys", flush=True)
Path("runs/pod/fleet").mkdir(parents=True, exist_ok=True)
procs = []
for r in range(N):
    games = ",".join(f"{f}-{f + 9}" for f in files[r::NS])
    env = dict(os.environ, POD=f"{os.environ.get('FLEET_PODS', exp)}{r}")   # FLEET_PODS: reuse another run's pods
    g = os.environ.get("RANK0_GPU", gpu) if r == 0 else gpu          # rank 0 also holds the held-out sets: bigger GPU
    procs.append(subprocess.Popen([sys.executable, "tools/fleet_rank.py", exp, str(r), str(N), g, batch, n, launch, games, init, *extra],
                                  env=env, stdout=open(f"runs/pod/fleet/{exp}_r{r}.log", "w"), stderr=subprocess.STDOUT))
    time.sleep(2)                                                   # stagger RunPod create calls
print(f"started {N} rank launchers; logs runs/pod/fleet/{exp}_r*.log", flush=True)
