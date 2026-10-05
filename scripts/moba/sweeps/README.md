# Dream-PPO and world-model sweep launchers

Copies of the launch scripts used for E35-E39 (they run from `runs/pod/`, which is git-ignored, and expect to be run from the repo root with `tools/pod.py` state in `runs/pod/`).

- `rw_arm.py`: one reward-sweep arm on its own pod (create/bootstrap or `SKIP_SETUP=1` in place).
- `rw_sweep.py`, `gen2.py`, `gen3.py`, `gen4.py`: the arm definitions of generations 1-4 (launch only when run directly).
- `eval_multi.py`, `eval_loop.py`: real-game evaluation on CPU pods alongside training.
- `ppo_fleet.py`: distributed dream PPO (FedAvg via Upstash).
- `fp_smoke.py`, `fp_sweep.py`, `sweep_rank.py`: tower false-death penalty sweeps (E35).
- `p16.py`, `p17.py`: dream PPO in the penalised world models (E35).
- `dyna_live.py`: the continuous Dyna loop (E39): `launch` starts the WM fleet, PPO fleet and collectors; `relay` runs the relays, the evaluator, the anchor selection and the drift guard; `data` runs only the real-games relay.
- `ppo_pool.py`: a dream-PPO FedAvg fleet on existing pods (`POD_OFFSET`, `FLEET_SHARDS`, `EXTRA_CORPORA` pick the pods and the shards they hold).
- `abl_eval.py`: round-robin real-game evaluation of several PPO fleets on one CPU pod (the E39 ablation).
- `seed_check.py`: 300 games per side on fresh (`SEED_BASE=60000000`) or holdout (70000000) seeds.
- `dyna_collect.py`, `dyna_loop.py`: the sequential Dyna loop that E39 replaced (one round ran: `dyna3_r1`, `wm_r1`).
