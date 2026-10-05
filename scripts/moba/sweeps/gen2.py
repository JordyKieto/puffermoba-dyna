"""Reward sweep generation 2: the structure that worked in gen 1 (death penalty off, siege presence, zero-sum tower
damage, XP kept) at several strengths, episode lengths and seeds. One pod per arm. Run directly to launch."""
import os, subprocess, sys, time
def B(s, d, zs=True, extra=""):
    return f"--death-scale 0 --siege-k {s} --dmg-k {d}" + (" --zero-sum" if zs else "") + (f" {extra}" if extra else "")
ARMS = {}
for s in (0.01, 0.02, 0.05):
    for d in (2, 5, 10):
        ARMS[f"zs_s{s}_d{d}"] = B(s, d)
ARMS.update({
    "zs_s0.02_d5_ep512": B(0.02, 5, extra="--episode 512"), "zs_s0.05_d5_ep512": B(0.05, 5, extra="--episode 512"),
    "zs_s0.02_d10_ep512": B(0.02, 10, extra="--episode 512"),
    "zs_s0.02_d5_ep1024": B(0.02, 5, extra="--episode 1024"), "zs_s0.05_d5_ep1024": B(0.05, 5, extra="--episode 1024"),
    "nozs_s0.02_d5": B(0.02, 5, zs=False), "nozs_s0.02_d5_ep512": B(0.02, 5, zs=False, extra="--episode 512"),
    "zs_s0.02_d5_fall10": B(0.02, 5, extra="--fall-bonus 10 --fall-radius 8"),
    "zs_s0.02_d5_fall10_ep512": B(0.02, 5, extra="--fall-bonus 10 --fall-radius 8 --episode 512"),
    "zs_s0.02_d5_seed1": B(0.02, 5, extra="--seed 1"), "zs_s0.02_d5_seed2": B(0.02, 5, extra="--seed 2"),
    "zs_s0.02_d5_seed3": B(0.02, 5, extra="--seed 3"),
    "zs_s0.02_d5_death0.5": B(0.02, 5).replace("--death-scale 0", "--death-scale 0.5"),
    "zs_s0.02_d5_2x": B(0.02, 5, extra="--steps 3390870"),
    "zs_s0.02_d5_win_ep512": B(0.02, 5, extra="--r-win 50 --episode 512"),
})
if __name__ == "__main__":
    os.makedirs("runs/pod/gb", exist_ok=True)
    for i, (arm, flags) in enumerate(ARMS.items()):
        subprocess.Popen([sys.executable, "runs/pod/rw_arm.py", arm, *flags.split()], env=dict(os.environ, POD=f"gb{i}", GEN="gb"),
                         stdout=open(f"runs/pod/gb/{arm}.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(2)
    print(f"{len(ARMS)} gen-2 arms starting", flush=True)
