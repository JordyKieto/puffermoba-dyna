"""Reward sweep generation 3: long episodes + win reward (what produced real wins), 4x training, continuation from the
best policy (wg1024_2x), dire-only training. Relaunches in place on the 29 existing sweep pods. Run directly to launch."""
import os, subprocess, sys, time
X4, X2 = "--steps 6781740", "--steps 3390870"
W = "--r-win 100 --episode 1024"
F = "--death-scale 0 --siege-k 0.02 --dmg-k 5 --zero-sum"
BEST = "--init-local results/moba/dream/rw/wg1024_2x/policy.pt"
ARMS = {}
for s in (1, 2, 3): ARMS[f"A_wg1024_4x_s{s}"] = f"{W} {X4} --seed {s}"
for s in (1, 2, 3): ARMS[f"B_wg1024_formula_4x_s{s}"] = f"{W} {F} {X4} --seed {s}"
for s in (1, 2): ARMS[f"C_zsd10_win_ep512_4x_s{s}"] = f"--death-scale 0 --siege-k 0.02 --dmg-k 10 --zero-sum --r-win 100 --episode 512 {X4} --seed {s}"
for s in (1, 2, 3): ARMS[f"D_best_cont_2x_s{s}"] = f"{W} {X2} {BEST} --seed {s}"
for s in (1, 2): ARMS[f"E_best_cont_formula_2x_s{s}"] = f"{W} {F} {X2} {BEST} --seed {s}"
for s in (1, 2): ARMS[f"F_win300_4x_s{s}"] = f"--r-win 300 --episode 1024 {X4} --seed {s}"
for s in (1, 2, 3): ARMS[f"G_dire_armC_4x_s{s}"] = f"{W} {X4} --side dire --seed {s}"
for s in (1, 2): ARMS[f"G_dire_best_2x_s{s}"] = f"{W} {X2} {BEST} --side dire --seed {s}"
for s in (1, 2): ARMS[f"H_lr1e-4_4x_s{s}"] = f"{W} {X4} --lr 1e-4 --seed {s}"
for s in (1, 2): ARMS[f"I_death0_siege_4x_s{s}"] = f"{W} --death-scale 0 --siege-k 0.02 {X4} --seed {s}"
for s in (1, 2): ARMS[f"J_envs128_4x_s{s}"] = f"{W} {X4} --envs 128 --seed {s}"
for s in (1, 2): ARMS[f"K_ancient_4x_s{s}"] = f"{W} --anc-k 50 {X4} --seed {s}"
ARMS["L_game_ep1024_nowin_4x"] = f"--episode 1024 {X4}"
PODS = [f"gb{i}" for i in range(24)] + [f"rw{i}" for i in (3, 4, 17, 18, 21)]
if __name__ == "__main__":
    assert len(ARMS) == len(PODS), (len(ARMS), len(PODS))
    os.makedirs("runs/pod/g3", exist_ok=True)
    for (arm, flags), pod_name in zip(ARMS.items(), PODS):
        subprocess.Popen([sys.executable, "runs/pod/rw_arm.py", arm, *flags.split()], env=dict(os.environ, POD=pod_name, GEN="g3", SKIP_SETUP="1"),
                         stdout=open(f"runs/pod/g3/{arm}.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(1)
    print(f"{len(ARMS)} gen-3 arms starting", flush=True)
