"""Reward-structure sweep: one pod per arm, all in parallel."""
import os, subprocess, sys, time
M = "--dmg-k 2 --dmg-boost 2 --fall-bonus 5 --fall-radius 8 --r-win 10 --def-scale 1"
ARMS = {
    "game_only": "", "no_xp": "--xp-scale 0",
    "towers_only": "--xp-scale 0 --death-scale 0 --r-tower 5 --tower-team",
    "win_sparse_512": "--xp-scale 0 --death-scale 0 --r-win 100 --episode 512",
    "win_game_1024": "--r-win 100 --episode 1024",
    "dmg_dense": "--dmg-k 10 --xp-scale 0.2", "dmg_dense_team": "--dmg-k 10 --xp-scale 0.2 --team-share",
    "push": "--push-k 0.05", "push_dmg": "--push-k 0.05 --dmg-k 5", "siege": "--siege-k 0.02",
    "zero_sum_mod": f"{M} --zero-sum", "zero_sum_dmg": "--dmg-k 5 --zero-sum", "team_mod": f"{M} --team-share",
    "aggressive": "--death-scale 0 --dmg-k 5 --fall-bonus 10 --fall-radius 8", "ancient": "--anc-k 50 --r-win 50",
    "moderate": M, "tower_team20": "--r-tower 20 --tower-team",
    "tower_team20_win_512": "--r-tower 20 --tower-team --r-win 50 --episode 512",
    "everything_512": "--push-k 0.05 --dmg-k 5 --anc-k 50 --r-win 50 --team-share --zero-sum --xp-scale 0.2 --episode 512",
    "selfplay_mix_mod": f"{M} --opponent mix", "scratch_push_dmg": "--push-k 0.05 --dmg-k 5 --scratch",
    "moderate_512": f"{M} --episode 512", "siege_push_team": "--siege-k 0.02 --push-k 0.05 --team-share --xp-scale 0.2",
    "no_xp_tower_win": "--xp-scale 0 --r-tower 10 --tower-team --r-win 50",
}
if __name__ == "__main__":                 # launching only when run directly: rw_collect imports ARMS
  os.makedirs("runs/pod/rw", exist_ok=True)
  for i, (arm, flags) in enumerate(ARMS.items()):
    subprocess.Popen([sys.executable, "runs/pod/rw_arm.py", arm, *flags.split()], env=dict(os.environ, POD=f"rw{i}"),
                     stdout=open(f"runs/pod/rw/{arm}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(2)
  print(f"{len(ARMS)} arms starting", flush=True)
