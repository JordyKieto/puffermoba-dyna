"""Reward sweep generation 4: 1,400-tick episodes, from the best real-game policy (pf1 update 7), GPU path, 512 games
per pod, 100M steps per arm; rewards from sparse (win / towers only) to handcrafted (dive, damage, champion kills).
Real-game curves come from runs/pod/eval_multi.py. Run directly to launch."""
import os, subprocess, sys, time
INIT = "--init-local results/moba/dream/pf1_policies/policy_u0007.pt"
COMMON = f"--episode 1400 --envs 512 --minibatches 16 --steps 100000000 --gpu-opp {INIT}"
G = "--r-win 100"                                                       # the formula: game reward + win
DIVE = "--dmg-k 5 --dmg-boost 2 --fall-bonus 10 --fall-radius 8"
KILL = "--kill-k 2"
ARMS = {
    "win_only": "--xp-scale 0 --death-scale 0 --r-win 100",
    "win_towers": "--xp-scale 0 --death-scale 0 --r-win 100 --r-tower 10 --tower-team",
    "win_towers_xp": f"{G} --r-tower 10 --tower-team",
    "formula": G, "formula_xp0.5": f"{G} --xp-scale 0.5", "formula_xp0.25": f"{G} --xp-scale 0.25",
    "dive": f"{G} {DIVE}", "dive_kills": f"{G} {DIVE} {KILL}", "kills_win": f"{G} {KILL}",
    "dive_kills_xp0.5": f"{G} {DIVE} {KILL} --xp-scale 0.5", "dive_kills_nodeath": f"{G} {DIVE} {KILL} --death-scale 0",
    "dive_kills_zs": f"{G} {DIVE} {KILL} --zero-sum",
    "all_handcrafted": f"{G} {DIVE} {KILL} --siege-k 0.02 --anc-k 50 --xp-scale 0.5",
    "ancient_bigwin": "--r-win 200 --anc-k 100",
    "towerteam_kills": f"{G} --r-tower 10 --tower-team {KILL} --xp-scale 0.5",
    "formula_s2": f"{G} --seed 2", "dive_kills_s2": f"{G} {DIVE} {KILL} --seed 2", "win_towers_xp_s2": f"{G} --r-tower 10 --tower-team --seed 2",
    "dive_kills_defend": f"{G} {DIVE} {KILL} --def-scale 1",
    "bigwin_bigtowers": "--r-win 300 --r-tower 20 --tower-team --xp-scale 0.5",
}
if __name__ == "__main__":
    os.makedirs("runs/pod/g4", exist_ok=True)
    for i, (arm, flags) in enumerate(ARMS.items()):
        subprocess.Popen([sys.executable, "runs/pod/rw_arm.py", arm, *f"{flags} {COMMON}".split()], env=dict(os.environ, POD=f"g4p{i}", GEN="g4"),
                         stdout=open(f"runs/pod/g4/{arm}.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(2)
    print(f"{len(ARMS)} gen-4 arms starting", flush=True)
