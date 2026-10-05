# E37: distributed dream PPO at 600M steps (pf1): 26× throughput, but real-game play peaks early and then declines

**Status:** concluded 2026-10-05.

**Verdict:**
- A 5-pod PPO fleet trained the winning formula (E36) for 613M steps at about 116k steps/s.
- Real-game performance, evaluated alongside every update, **peaked at the first point** (update 7, 91M steps: 4/60
  radiant wins, 3.25 towers per game) **and then declined**: 0 wins from 144M steps on, fewer towers, lower dire return.
- The policy overfits the dream. More dream steps alone don't lead to winning.

The curve is in `results/moba/dream/pf1_EVAL_TABLE.md`, generated from `pf1_eval.jsonl`.
Every evaluated policy is in `results/moba/dream/pf1_policies/policy_u<update>.pt`.

## 1. Throughput: the policy side moved to the GPU
- **Profiling:** on an A40, 83% of a dream-PPO update (122 of 148 s) was the policy side. Two CPU pieces dominated:
  - the 510-byte ego observations, built in numpy (a full 128×128 map per game);
  - the shipped net, run as a C library on about 8 CPU cores for all 1,280 hero slots.
- **`mobawm/rl/gpu_net.py`:**
  - `ego_obs_gpu`: byte-identical to `mobawm/obs.py` (0 of 51,000 bytes differ);
  - `ShippedNet`: `moba_weights.bin` in torch, with the same MinGRU maths as `puffernet.h`. Its logits are within
    5e-6 of the C pool's over 5 recurrent steps (`ma_pool_logits`).
- **Enabled by** `--gpu-opp` in `dream_ppo.py`.

| one A40, 256-tick episodes | steps/s | dream WM / policy / PPO update (s) |
|---|---|---|
| 128 games, CPU path | 4,074 | 6 / 34 / 1 |
| 128 games, `--gpu-opp` | 15,666 | 6 / 3 / 1 |
| 512 games, `--gpu-opp` | 24,732 | 20 / 3 / 3 |
| 1,024 games, `--gpu-opp` | 26,866 | 39 / 3 / 6 |

The dream world model is now the bottleneck. `torch.compile` on it is the next likely speed-up; it gave 2× in
world-model training.

## 2. Distributed PPO (zelda-RL FedAvg)
- **How it works:** every pod runs the full PPO loop on its own dream games, with its own seed. After every update the
  pods average their policy weights through Upstash (`--fleet-exp/-rank/-world/-launch/-every`, `mobawm/fleet_link.py`).
  Each pod posts a heartbeat with its steps/s and its dream towers, ours vs the net.
- **Local test:** 2 ranks reached identical weight hashes after every round.
- **Launcher:** `runs/pod/ppo_fleet.py`.

## 3. Run pf1
- **Setup:**
  - 5 A40 pods;
  - start from the best gen-3 policy (`J_envs128_4x_s2`);
  - the winning formula: game reward + win ±100, 1,024-tick episodes, calibrated dream (fleet20e ×1.5);
  - 512 games per pod, 16 minibatches, `--gpu-opp`;
  - 47 updates, 122M steps per pod, 613M in total.
- **Speed:** 21,600–25,500 steps/s per pod, **≈116k steps/s for the fleet (≈26× one pod before)**, 5–25 s of sync wait per
  update. 600M steps took about 85 minutes instead of about 37 hours.
- **Evaluation alongside training** (`runs/pod/eval_loop.py`):
  - a 32-core CPU pod ($1.12/hr) played 60 real games per side against the shipped net;
  - it ran on every new rank-0 `policy.pt` and took about 40 s per point, so there are 41 points.

| | first third of the run | last third |
|---|---|---|
| radiant towers per game | 2.85 | 2.31 |
| radiant return | 22.2 | 20.8 |
| dire return | 7.9 | 3.6 |
| radiant wins | 4 (all at update 7) | 0 |

- **Dire:** 0 towers and 0 wins at every point. The dire return was 15.8 at update 7 against about 1 for arm C, and then
  fell back.

## Lessons
- **Select checkpoints by real games, not training length.** The best policy is update 7 (`pf1_policies/policy_u0007.pt`).
- **Next:**
  - a Dyna loop: the policy's own real games go into the world model's data, then the WM is retrained before more PPO
    (E9 showed this cuts dream exploitation 3–7×);
  - and/or keeping the policy close to a good real-game checkpoint (a KL penalty or a lower learning rate).

## Incidents
- **Duplicate pods:** a results collector imported the sweep launcher, which ran at import, and spawned 72 duplicate pods
  ($56/hr for about 15 minutes, roughly $5–7). Launchers now launch only under `__main__`.
- **Stale code:** in-place relaunches did not push code, so the dire arms ran stale code; the launcher now always pushes.
- **One dire arm-C result lost:** `G_dire_armC_4x_s1` was not copied before its pod was killed.
