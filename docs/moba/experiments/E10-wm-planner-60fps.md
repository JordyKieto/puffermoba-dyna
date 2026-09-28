# E10 — World-model planner at 60 fps vs the shipped net: no better than random (NEGATIVE)

**The one-line result.** An anytime planner (lookahead in E5's dream) plays 120 real
games against the shipped net and wins none. Its return (1.84) is near the untrained
policy's (1.47) and far below the net's (10.46). It picks the net's own action only 2% of
the time: the dream scores its held team commands higher, consistent with E8's XP
hallucination.

- **Status:** concluded, 2026-09-27.
- **Code:** `mobawm/rl/plan_mpc.py` (`play_realtime` / `--batched`), uncommitted; see the
  code snapshot in the manifest.

## The planner (per game)

- **Every frame, exactly one batched world-model call:**
  - row 0 keeps the model's memory on the real game;
  - rows 1..K advance the K branches of the current search by one tick.
- **Branches (K = 3):**
  1. the shipped net's current action for our 5 heroes;
  2. the previous plan;
  3. one new coordinated or mixed team plan (shared move/attack for the team in half of
     proposals; abilities with p = 0.3).
- **Lookahead H = 16:** the search completes every 16 frames, the best branch by our
  team's summed game reward is executed until the next one completes, and inside the
  search the enemy holds its last real action.
- **Budget:** K and 1 tick per frame were chosen so that a **single game on a laptop CPU
  fits 60 fps (16.7 ms per frame)**.
  - Component costs: `results/moba/planner/calibration_mac.json` (e.g. a world-model
    step plus reward at batch 4 on CPU: p95 11.5 ms).
  - Measured whole frames: `results/moba/planner/realtime_single_game_timing.jsonl`,
    p50 11.5–11.7 ms, p95 12.6–14.2 ms, 0–1.2% of frames over budget.
  - MPS was slower than CPU at these batch sizes.
- **Evaluation:** all 120 games batched in one model call per frame on the H100 NVL, for
  throughput only; each game's search is fixed by construction. Baselines on the same
  seeds with `mobawm/rl/arena.py`.

## Result

Seeds 50000000-50000059 per side, max 6,000 ticks. Branch chosen (net's action / keep plan / new plan): [1031, 38883, 5086] = net's action 2%.

| our team (vs shipped net) | games | win % | loss % | timeout % | return/hero | deaths | level | creeps |
|---|---|---|---|---|---|---|---|---|
| WM planner (60 fps budget) | 120 | 0 | 50 | 50 | 1.84 | 1.40 | 1.84 | 2.0 |
| planner as radiant | 60 | 0 | 0 | 100 | 3.36 | 2.13 | 2.90 | 3.5 |
| planner as dire | 60 | 0 | 100 | 0 | 0.33 | 0.67 | 0.79 | 0.5 |
| shipped net (mirror) | 120 | 50 | 50 | 0 | 10.46 | 6.87 | 5.04 | 10.8 |
| untrained (random) | 120 | 0 | 50 | 50 | 1.47 | 0.57 | 1.49 | 1.7 |

**Conclusions:**
1. **Planning in this dream does not beat the policy it could have copied:** branch 0,
   the net's own action, was always available and was chosen 2% of the time.
2. **The search is fooled by the same XP hallucination as dream PPO (E8):** held team
   commands look rewarding in the dream but not in the real game.
3. **The 60 fps budget on a laptop CPU allows a tiny search** (3 branches, 1 tick per
   frame). The main problem is the model's reward signal, not the search size.

**Data:**
- per-game rows in `results/moba/planner/realtime_vs_net.jsonl` and
  `results/moba/planner/baselines_vs_net.jsonl`;
- logs `planner.log` and `baselines.log`.

No W&B run: evaluation only.
