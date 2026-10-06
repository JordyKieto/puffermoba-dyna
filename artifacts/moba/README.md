# Released checkpoints

The policies and world models behind every result in the paper ([docs/moba/paper/paper.pdf](../../docs/moba/paper/paper.pdf)).
Files are byte-identical to the evaluated ones; the md5s match the experiment records.

## Policies

PufferLib's MOBA policy (encoder + 6-layer MinGRU), one network for the five heroes of a side, trained only inside a
world model. Radiant wins against the shipped PufferNet on the holdout seeds (300 games; dire is 0 for every policy).

| file | what | radiant wins (holdout) | md5 |
|---|---|---|---|
| `policies/ppol4_u69.pt` | **headline**: Dyna loop + KL anchor, update 69 | 213/300 holdout, 208/300 holdout2 (70.2% pooled) | `3fe7818b0ae9` |
| `policies/ppol4_u49.pt` | Dyna loop + KL anchor, update 49 (an anchor) | 203/300 | `e75274c65cb7` |
| `policies/ppol4_u10.pt` | Dyna loop + KL anchor, update 10 (an anchor) | 167/300 | `07f1dc98ea91` |
| `policies/ppol_u71.pt` | first continuous-Dyna run; start of the anchored run and of the E40 control | 159/300 | `e42aac2d72d8` |
| `policies/e38_u0017.pt` | ladder rung 4: shaped reward, XP x0.5; start of the loop | 101/300 | `a077dc64ef28` |
| `policies/e37_pf1_u7.pt` | ladder rung 3: 5-pod PPO fleet | 17/300 | `297e6ac37c83` |
| `policies/e36_J_envs128_4x_s2.pt` | ladder rung 2: long episodes + win reward | 4/300 | `ab2b762645ac` |
| `policies/e14_armC.pt` | ladder rung 1: dream PPO, game reward, short episodes | 0/300 | `3874281e9b40` |

Evaluate one in the real game (needs `tools/build.sh moba-arena`):

```bash
python -m mobawm.rl.arena --policy artifacts/moba/policies/ppol4_u69.pt --games 60 --teams ppo --opponents net --seed-base 70000000
```

`scripts/moba/seed_eval.py` runs the full 300-per-side protocol of [EVALUATION.md](../../docs/moba/EVALUATION.md).

## World models

TokenWM, 0.66M parameters (`model.pt`: state dict; `stats.pt`: normalisation statistics and config). The four versions
compared in the paper's fidelity tables (Section 5.3 and Table 7).

| directory | what | model md5 |
|---|---|---|
| `world_models/pre_dyna_fleet20e_at500/` | the E38 dream (calibrated, never saw a policy game); E40 control arm | `d843b9ffdceb` |
| `world_models/dyna_round1_wm_r1/` | first Dyna round (+270 policy games) | `097387d50906` |
| `world_models/dyna_wml_at3600/` | continuous Dyna, first fleet (one pod for most of its life) | `47e2f633bc83` |
| `world_models/dyna_wml2_at3400/` | continuous Dyna, healthy 7-pod fleet; E40 control arm | `19f8df99b4ce` |

Open one in the world-model debugger on a real game:

```bash
python tools/moba_viz.py --run artifacts/moba/world_models/dyna_wml2_at3400 --dense <a .mbs game file> --start 1500 --horizon 300
```
