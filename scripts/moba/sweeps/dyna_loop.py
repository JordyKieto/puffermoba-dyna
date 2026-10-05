"""Distributed Dyna loop: real games of the current dream policies -> world model fine-tuned on them -> dream PPO in
the corrected dream -> the real game picks the best policy -> repeat. Real-game evaluation is the only selection signal.

  stage 1 collect   CPU pods: <PER> games per policy vs the shipped net (alternating sides) for the round's policies
                    (best so far + the latest, drifted checkpoints), recorded + verified, uploaded to moba/runs/dyna3_r<k>/
                    (last 10% of each round's games held out of world-model training).
  stage 2 world model  GPU pool (tools/fleet.py, FedAvg): fine-tune the current dream on all real data + every round's
                    new games: lr 3e-4, false-tower-death penalty x1.5, WM_UPDATES updates; take rank 0's last checkpoint.
  stage 3 PPO       the same GPU pool (runs/pod/ppo_pool.py): distributed dream PPO from the best policy in the new dream,
                    PPO_UPDATES updates; runs/pod/eval_loop.py plays every rank-0 checkpoint in the real game (60 per side).
  select            the round's best = most real wins (radiant + dire), then most radiant towers.

State: runs/pod/dyna_state.json (resumable: stages already done are skipped). usage: python runs/pod/dyna_loop.py <rounds>"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "tools")
ROUNDS = int(sys.argv[1])
POOL, NPODS, CPUS = "dl", 20, ["evc1", "evc4"]
PER, WM_UPDATES, PPO_UPDATES, ENVS = 100, 500, 15, 128
REWARD = "--r-win 100 --dmg-k 5 --dmg-boost 2 --fall-bonus 10 --fall-radius 8 --kill-k 2 --siege-k 0.02 --anc-k 50 --xp-scale 0.5"
STATE = Path("runs/pod/dyna_state.json")
st = json.loads(STATE.read_text()) if STATE.exists() else dict(
    round=1, wm="runs/token_wm/fleet20e_at500", best="results/moba/dream/g4_policies/all_handcrafted/policy_u0017.pt",
    collect=["results/moba/dream/g4_policies/all_handcrafted/policy_u0017.pt",
             "results/moba/dream/g4_policies/all_handcrafted/policy_u0021.pt",
             "results/moba/dream/g4_policies/all_handcrafted/policy_u0029.pt"], corpora=[], done={}, history=[])
save = lambda: STATE.write_text(json.dumps(st, indent=1))
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
import pod
c, bucket = pod.s3()

def s3_has(key):
    try:
        c.head_object(Bucket=bucket, Key=key); return True
    except Exception:
        return False

while st["round"] <= ROUNDS:
    k = st["round"]; d = st["done"].setdefault(str(k), {})
    first = 7000 + (k - 1) * 1000; G = PER * len(st["collect"]); n_train = int(G * 0.9)
    run = f"dyna3_r{k}"
    # 1. collect
    if not d.get("collect"):
        log(f"round {k}: collecting {G} real games of {len(st['collect'])} policies -> {run}")
        subprocess.run([sys.executable, "runs/pod/dyna_collect.py", run, str(first), str(PER), ",".join(CPUS), *st["collect"]], check=True)
        st["corpora"].append(f"runs/{run}:{first}-{first + n_train - 1}")
        d["collect"] = True; save()
    # 2. world model
    wm_exp = f"wm_r{k}"
    if not d.get("wm"):
        log(f"round {k}: world model {wm_exp} from {st['wm']} with corpora {st['corpora']}")
        env = dict(os.environ, EXTRA_CORPORA=";".join(st["corpora"]), FLEET_PODS=POOL,
                   RANK0_GPU="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe")
        if Path(f"runs/pod/state_{POOL}0.json").exists():
            env["SKIP_SETUP"] = "1"                                  # the pool exists: relaunch in place (new corpora still sync)
        subprocess.run([sys.executable, "tools/fleet.py", wm_exp, str(NPODS), "NVIDIA A40", "384", str(WM_UPDATES), st["wm"],
                        "--lr", "3e-4", "--tower-fp-weight", "1.5", "--compile", "--fleet-every", "50"], env=env, check=True)
        key = f"{pod.S3_PREFIX}/runs/{wm_exp}_r0/model_{WM_UPDATES:06d}.pt"
        while not s3_has(key):
            time.sleep(60)
        out = Path(f"runs/token_wm/{wm_exp}"); out.mkdir(parents=True, exist_ok=True)
        c.download_file(bucket, key, str(out / "model.pt")); c.download_file(bucket, f"{pod.S3_PREFIX}/runs/{wm_exp}_r0/stats.pt", str(out / "stats.pt"))
        st["wm"] = str(out); d["wm"] = True; save()
        log(f"round {k}: world model ready: {out}")
    # 3. PPO + real-game evaluation
    ppo = f"ppo_r{k}"
    evf = Path(f"results/moba/dream/{ppo}_eval.jsonl")
    if not d.get("ppo"):
        log(f"round {k}: PPO {ppo} from {st['best']} in {st['wm']}")
        subprocess.run([sys.executable, "runs/pod/ppo_pool.py", ppo, POOL, str(NPODS), st["wm"], st["best"], str(PPO_UPDATES), str(ENVS),
                        *REWARD.split()], check=True)
        ev = subprocess.Popen([sys.executable, "runs/pod/eval_loop.py", ppo, CPUS[0]], stdout=open(f"runs/pod/eval_loop_{ppo}.log", "w"),
                              stderr=subprocess.STDOUT)
        while True:                                                   # until the last update has been evaluated
            rows = [json.loads(l) for l in evf.read_text().splitlines() if l.strip()] if evf.exists() else []
            if rows and max(r["update"] for r in rows) >= PPO_UPDATES:
                break
            time.sleep(60)
        ev.terminate(); d["ppo"] = True; save()
    # select
    rows = [json.loads(l) for l in evf.read_text().splitlines() if l.strip()]
    best = max(rows, key=lambda r: (r["radiant_wins"] + r["dire_wins"], r["radiant_towers_ours"]))
    pol_dir = Path(f"results/moba/dream/{ppo}_policies")
    latest = sorted(rows, key=lambda r: r["update"])[-2:]
    st["history"].append(dict(round=k, best_update=best["update"], radiant_wins=best["radiant_wins"], dire_wins=best["dire_wins"],
                              radiant_towers=best["radiant_towers_ours"], wm=st["wm"]))
    st["best"] = str(pol_dir / f"policy_u{best['update']:04d}.pt")
    st["collect"] = [st["best"]] + [str(pol_dir / f"policy_u{r['update']:04d}.pt") for r in latest if r["update"] != best["update"]]
    log(f"round {k} done: best update {best['update']}: radiant {best['radiant_wins']}/60, dire {best['dire_wins']}/60, "
        f"towers {best['radiant_towers_ours']}")
    st["round"] = k + 1; save()
log("all rounds done")
