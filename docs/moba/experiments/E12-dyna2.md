# E12 — E11 continued with the dream policies' own real games (dyna2)

- **Status:** stopped by the author at update ~4,350 (2026-09-28, ~2.8 h). Checkpoints
  every 500 updates (`model_000500` … `model_004000`) are in S3 and on the Mac.
- **W&B:** `moba-wm/xfldazic` (group `E12-dyna2`).
- **S3:** `s3://puffertank/moba/runs/e12_dyna2/`. Every eval checkpoint (every 500 updates) is
  kept as `model_<it>.pt` and uploaded with md5 verification. The Mac mirrors it every 10 min
  (`runs/pod/mirror.sh`).

## Setup

- **Init:** E11 @2000 (`model_002000.pt`, md5 `a67d4c41`). Same recipe as E5/E11,
  `--ckpt-every 500`.
- **Data:** mix1k + mixrand1k + **dyna2**, 2,560 training games, 6,766,710 transitions
  (145.5 GB in host memory).
- **dyna2** (DATA.md): 810 real games of the two P3 dream-PPO policies:
  - vs uniform random, 270;
  - vs the shipped net, 270;
  - vs each other, 270;
  - 2,154,814 ticks, 0 replay mismatches.
  - Training uses all but one 10-game file per matchup: 6260–6269, 6530–6539 and 6800–6809
    are kept out for later audits.
- **Held-out sets:** the same two as E11 (random held-out = main metric; mix1k 900–999).
- **Update 0 (= E11 @2000):**
  - mix1k: h1 2.33, h16 7.91, h256 30.4, h1024 39.6, h4096 50.3;
  - random: h1 3.64, h16 8.23, h256 35.3, h1024 60.4.

## Loader change this run needed

The training store (145.5 GB) no longer fits through `/dev/shm`, which is a 117 GB tmpfs
on this pod, so the first launch died (BrokenProcessPool).
- `mobawm/wm/data.py` now decompresses into anonymous shared mappings made before the
  workers fork. These are limited only by the container's 251 GB.
- A CPU store uses them in place, with no second copy.
- Checked on the pod: CPU and GPU stores built from the same files are identical in
  entities, actions, ticks and spans.

## Result

See the generated table `results/moba/tables/e12_horizons.md` (x = relative to update 0 =
E11 @2000).

- **Random held-out:** h1024 fell from 60.4 to 44.1 (update 4000; E5 182) and hero error at
  1,024 ticks from 21.5 to 14.8 cells (E5 72).
- **Expert held-out:** around E5's level from update 1000, with evaluations swinging a lot
  (h4096 35–75).
- **Update 4000 has a short-horizon spike on both sets** (h4 ~9.3–9.8).
- **Not measured:** whether dyna2 reduced the dream's over-payment of the policy. That needs
  `scripts/moba/dream_vs_real_reward.py` on `model_004000`.
