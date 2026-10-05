# E38: reward sweep gen 4: dream policies win 28–45% of real radiant games at their peak

**Status:** running (2026-10-05). The arms are at 40–80M of their 100M steps.

**Verdict so far:**
- **Turning XP down is the key change.** Two arms now win real games against the shipped net at double-digit rates:
  - `all_handcrafted`: **27/60 (45%)** at 61M steps;
  - `formula_xp0.25`: **17/60 (28%)** at 25M steps.
  - The previous best was 4/60.
- **Both arms peak and then decline** with more dream training. Choosing checkpoints by real games is part of the recipe.
- Dire: 0 wins everywhere.

The per-arm curves are in `results/moba/dream/g4_EVAL_TABLE.md`, generated from `g4_eval.jsonl`. The policies at every
evaluated point are in `s3://puffertank/moba/dream_policies/g4_policies/<arm>/policy_u<update>.pt`, with the same paths
locally (git-ignored).

## Setup (every arm)
- **Episodes:** 1,400 ticks (`--episode 1400 --rollout 1400`).
- **Start:** the best real-game policy so far, pf1 update 7 (E37: 4/60).
- **Dream:** fleet20e ×1.5 (calibrated), 512 games per pod, `--gpu-opp`, 16 minibatches, 100M steps per arm, one pod per arm
  (A40, or RTX PRO 6000 when A40s ran out).
- **New reward term:** `--kill-k`, champion kills. When an enemy hero dies, the nearest of our living heroes within 8 cells
  gets +k.
- **Real-game evaluation during training:** `runs/pod/eval_multi.py` watches every arm's S3 `policy.pt` and plays 60 games
  per side against the shipped net on CPU pods (2 × 32 cores), about 50 s per point.

## The 20 arms
They span sparse rewards (win only; win + towers), the current formula (game reward + win ±100) with less XP, and
handcrafted rewards (tower damage, fall bonus, champion kills, siege, ancient, zero-sum, team rewards, a defence
penalty), plus seeds. The definitions are in `scripts/moba/sweeps/gen4.py` (run from `runs/pod/`).

## Results so far
| arm | reward | peak (real radiant games) | curve |
|---|---|---|---|
| **all_handcrafted** | win ±100 + tower damage (k 5, boost 2) + fall bonus 10 + champion kills (k 2) + siege 0.02 + ancient 50; **XP ×0.5** | **27/60, 3.80 towers/game @61M** | 5 → 5 → 22 → 27 → 12 |
| **formula_xp0.25** | game reward + win ±100; **XP ×0.25** | **17/60, 3.72 towers/game @25M** | 2 → 17 → 11 → 4 → 1 |
| win_towers | win ±100 + 10 per tower to the team; no XP, no death | 5/60 | 1 → 5 → 5 → 2 → 1 |
| formula (XP ×1, control) | game reward + win ±100 | 1/60 | 1 → 0 → 0 → 0 → 0, towers 2.9 → 2.5 |
| full-XP dive / kills arms | | 0–1/60 | towers fall to 1.9–2.7 |

## Reading
- **XP is the dream's exploit.** Arms that keep full XP drift into farming and decline, the same pattern as in pf1 (E37).
  Cutting XP to ×0.25–×0.5 keeps the heroes moving and makes winning pay.
- **Removing XP entirely is worse:** `win_only` and `win_towers_xp` without XP decline, and gen 1 showed that no XP means no
  movement.
- **Champion-kill and dive rewards help only together with reduced XP** (`all_handcrafted`); with full XP they decline.
- **Every curve peaks and then falls.** The dream keeps rewarding things the real game doesn't, so the best policy is chosen by
  the real-game evaluation:
  - `g4_policies/all_handcrafted/policy_u0017.pt` (27/60);
  - `g4_policies/formula_xp0.25/policy_u0007.pt` (17/60).
- **Next:**
  - confirm the peaks with more games (wins of 17–27 per 60 carry ±5–6 of sampling noise);
  - keep XP low and stop on the real-game curve;
  - a Dyna loop, so the world model stops overpaying the policy's habits;
  - dire.
