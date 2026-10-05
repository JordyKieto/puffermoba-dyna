# MOBA evaluation service

Every claim that a dream-trained policy "plays the real game" rests on this page. A policy is only ever judged on
**real games** of the C MOBA (PufferLib's `ocean/moba`, unchanged), played against the **shipped PufferNet**
(`third_party/pufferlib/resources/moba/moba_weights.bin`). The world model never scores anything.

## 1. The game and the opponent

- **Engine:** `build/libmobaarena` (`moba/moba_arena.c`). It includes the corpus collector (`moba/moba_collect.c`)
  unchanged, so the game, the shipped net's forward pass and the action decoding are exactly those that produced the
  training data.
- **Ours:** a dream-trained `PufferPolicy` (encoder + 6-layer MinGRU) controls all 5 heroes of one side from the
  game's own 510-byte ego observations. It samples its actions with a `torch.Generator` seeded by the game seed.
- **The other side:** the shipped net, with its own seeded sampling stream.
- **A game ends** when an ancient falls (win/loss), or at 6,000 ticks (no winner; counted as not won).
- **Determinism:** the same policy, seed and side always give the same game. The fixed-seed evaluation of `all_handcrafted`
  u0017 reproduced E38's 27/60 exactly, and the clip recorder re-checks it at every clip start.

## 2. Seed sets: what each number is allowed to mean

| set | seeds | games | used for | bias |
|---|---|---|---|---|
| **fixed** | 50,000,000 + i | 60 per side | after **every** PPO update (training curves, drift guard, anchor candidates) | the best of many checkpoints on the same 60 games is optimistic (E38's u0017: 45% fixed vs 29–34% elsewhere) |
| **fresh** | 60,000,000 + 1000k + i | 300 per side | confirming an anchor candidate (a new anchor needs ≥ 9 more radiant wins than the current one, about 1 SE) | mild: every candidate is scored on the same 600 games |
| **holdout** | 70,000,000 + 1000k + i | 300 per side | **reporting only**. No checkpoint, setting or decision has ever been chosen on it | none by construction |
| **holdout2** | 80,000,000 + 1000k + i | 300 per side | a second untouched set, to pool to 600 radiant games for the headline policy | none |

- **The rule:** a number from the fixed or fresh set is never reported as a win rate. The paper's win rates come only from
  holdout and holdout2.
- **When a new anchor appears:** it is evaluated on holdout (and holdout2), and the headline is recomputed from those rows
  (§5).

## 3. Statistics

- **Win rate:** wins / games for one side. A game that reaches 6,000 ticks counts as not won.
- **95% Wilson score intervals** for every rate; **Newcombe's interval** for differences between two policies
  (`scripts/moba/paper_data.py`).
- **Side:** radiant and dire are always reported separately. The game is not symmetric: see the shipped net's own
  radiant and dire rates in `seed_evals.jsonl`, row `shipped_net`.
- **Behaviour per game,** reported with every evaluation: towers taken (out of 5 per lane-side, as the logs count them),
  kills, deaths, game length.

## 4. The services

| service | where | what | output |
|---|---|---|---|
| **after-update evaluator** | CPU pod evc1, inside `scripts/moba/sweeps/dyna_live.py relay` | every new rank-0 PPO policy: 60 per side on the fixed seeds | `results/moba/dream/<run>_eval.jsonl` |
| **anchor gate** | same | a policy at or above the anchor's fixed-seed score gets 300 per side on fresh seeds and becomes the KL anchor if it wins by ≥ 9 | `results/moba/dream/<run>_fresh.jsonl`, `dyna_live_state.json` |
| **drift guard** | same | 3 consecutive evaluations at ≤ half the anchor's fixed-seed wins → every PPO rank restarts from the anchor at one update | `dyna_live_state.json` (`resets*`) |
| **ablation evaluator** | a CPU pod, `scripts/moba/sweeps/abl_eval.py` | round-robin over several PPO fleets, the newest policy of each, fixed seeds | `results/moba/dream/<arm>_eval.jsonl` |
| **paper evaluation** | a CPU pod, `scripts/moba/seed_eval.py` | any policy (or the shipped net, or random) on a named seed set, 60 per side per chunk | `results/moba/paper/seed_evals.jsonl` + per-game `results/moba/paper/games/` |
| **collector** | CPU pod evc4, `scripts/moba/dyna_live_collect.py` | real games of the current PPO policy, recorded + verified for world-model training; every game's winner is kept | `results/moba/dream/dyna_live_meta/` → `results/moba/paper/collector.csv` |
| **clip recorder** | the Mac, `scripts/moba/record_clips.py` | 60-s videos of real games, drawn by the actual client (`moba/moba_render.c`) | `media/moba/clips/` + `s3://puffertank/moba/media/clips/` |

**Operational rules, learned the hard way:**
- **One arena per CPU pod at a time.** Two concurrent arenas hang each other, and so does a single 300-game arena. Every
  evaluation runs as 60-per-side chunks and first kills orphaned workers.
- **Collector disk:** a full disk truncated dense files that then crashed world-model training. The collector now deletes
  each file after its md5-verified upload, and the world model skips (and logs) unreadable files.

## 5. Evaluating a new policy and updating the headline

```bash
POD=evc5 python scripts/moba/seed_eval.py --label ppol4_u<N> --policy results/moba/dream/ppol4_policies/policy_u<N>.pt --set holdout
POD=evc5 python scripts/moba/seed_eval.py --label ppol4_u<N> --policy ... --set holdout2
python scripts/moba/paper_data.py                 # recomputes results/moba/paper/numbers.json (headline = best holdout policy)
jupyter nbconvert --execute --inplace docs/moba/paper/figures.ipynb
```

- **The headline** is the PPO policy with the highest holdout radiant win rate, with its 95% interval, and its pooled
  holdout + holdout2 rate when both exist.
- **The cost of selection:** picking the best of k holdout-evaluated policies would itself bias the holdout. Every policy
  that has ever been evaluated on holdout is listed in `seed_evals.jsonl`, and the paper reports how many there were.

## 6. Artifacts

- **Per-update evaluations:** `results/moba/dream/*_eval.jsonl`. Every policy they score is in
  `s3://puffertank/moba/dream_policies/dyna_live/<run>/policy_u<update>.pt`.
- **Seed-set evaluations:** `results/moba/paper/seed_evals.jsonl`, plus per-game rows in `results/moba/paper/games/`.
- **Collector games:** `results/moba/dream/dyna_live_meta/`. Dense + logs are in `s3://puffertank/moba/runs/dyna_live/`.
- **Derived tables:** `results/moba/paper/*.csv` and `numbers.json`, from `scripts/moba/paper_data.py`.
- **Videos:** `media/moba/clips/<date>_<policy>/` (git-ignored; mirrored to S3) with one metadata JSON per clip:
  policy md5, seed, ticks, events, camera track and renderer.
