# Dream-PPO and world-model sweep launchers

Copies of the launch scripts used for E35-E38 (they run from `runs/pod/`, which is git-ignored, and expect to be run from the repo root with `tools/pod.py` state in `runs/pod/`).

- `rw_arm.py`: one reward-sweep arm on its own pod (create/bootstrap or `SKIP_SETUP=1` in place).
- `rw_sweep.py`, `gen2.py`, `gen3.py`, `gen4.py`: the arm definitions of generations 1-4 (launch only when run directly).
- `eval_multi.py`, `eval_loop.py`: real-game evaluation on CPU pods alongside training.
- `ppo_fleet.py`: distributed dream PPO (FedAvg via Upstash).
- `fp_smoke.py`, `fp_sweep.py`, `sweep_rank.py`: tower false-death penalty sweeps (E35).
- `p16.py`, `p17.py`: dream PPO in the penalised world models (E35).
