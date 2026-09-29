# MOBA: experiments index

The research question: **can we learn a world well enough to teach an agent in it?**
This follows `carla-imitation-learning` (learning to act) and `neural-simulators`
(learning to simulate).

Evidence standard: [../EVIDENCE_RULES.md](../EVIDENCE_RULES.md). Claims with evidence
classes: [../paper/CLAIMS.md](../paper/CLAIMS.md). Where every number comes from:
[../PROVENANCE.md](../PROVENANCE.md). Doc map: [../MAP.md](../MAP.md).

## Headline (2026-09-28)

- **⚠ The world model is not fully learned.** Spawns, creep constants, hero level
  and level-derived stats, counters, targets and clamps are **hand-coded rules**
  (R1–R10), and every rollout, viewer result and dream depends on them. Everything
  dynamic (movement, damage, cooldowns, deaths, towers) is learned.
  [WM_ARCHITECTURE.md § Rules vs learned](../WM_ARCHITECTURE.md#rules-vs-learned-what-the-model-does-not-predict).
  Not signed off; decision pending.
- **World model, best so far:**
  - **Long horizons:** E5 (autoregressive rollout training), @20,500. Held-out loss is
    5–12× lower than teacher-forced from h64 to h4k, and heroes are ~5.5 cells off
    after 1,024 ticks (W3, W5).
  - **One step:** E6b (real-sim counterfactual siblings + effect losses). h1 is about
    2× better than E5, and it predicts 81–90% of an ability press's effect on
    training forks (W8, W10).
  - **Open problems:** jitter (31% extra direction reversals, speed 1.3× real; W7),
    tower health (W14), and the rules (R1–R10).
- **Dream RL: first attempt negative.** PPO in E5's dream more than doubles dream
  reward, but wins 0 of 120 real games and earns less than an untrained policy (R2).
  See [E7](E7-dream-ppo.md).
- **The dream invents most tower deaths, and the fix is the attack graph (E15).** In a
  128-tick dream, 2 of 3 predicted tower falls never happen (35% precision). The game records
  who attacks whom every tick, but the model threw it away. Feeding the real attack graph
  raises precision to 58–60%. An update-matched A/B (E17) shows the consistency loss and
  event sampling don't add to it. **At the full-run configuration the effect is modest** (E19:
  47% vs control's 38%), and **tower prediction degrades with longer training** (E18: 11%
  at +1,500 updates). The root cause is that the dream under-predicts tower damage (13% vs
  2% health at a real fall).
- **Best dream policy: arm C (E14).** PufferLib's recurrent MinGRU + our PPO, 1.70M steps in
  E11's dream. Over 120 real games vs the shipped net: as radiant return 18.1 vs the net's
  14.3 with 2× creeps, but 0% wins (no pushing); as dire 100% lost. Longer or harder
  optimisation makes it worse. Neither the game nor the dream rewards winning, and dream-only
  win/tower bonuses did not fix that (R16, R17).
- **Random-play data fixes it for random play (E11), and dream PPO now beats random in the
  real game (E13).** E11 cuts held-out random-game loss 4–8× at h16–h256 and removes the
  fake XP for random actions (8,298 → 54 per 1k). PPO in E11's dream: return 3.38 (P4) and
  6.40 (P4b, random-play start states) vs the shipped net, against random's 1.53 and the
  net's 11.21. 0 wins. The dream still ranks the PPO policy above the net (W23). Running:
  E12 (+ dyna2) and P5 (100× steps).
- **Why: the dream hallucinates XP off-distribution** (E8). The training data was
  expert/scripted only (64% net, 36% script, no random actions; D1). For the dream
  policy's actions E5 pays ~17k XP per 1k hero-steps where the real game pays 0. A
  60 fps WM planner is fooled the same way: 0/120 wins, ≈ random (E10). The policy's
  own real games cut the hallucination 3–7× (E9). Now training on 1,000 added
  random-play games (E11).
- **Checkpoints lost on 2026-09-27** (the pods were deleted externally): E4 final, E5
  numbered, and all of E5-continued. Their metrics survive; see
  [PROVENANCE §3.3](../PROVENANCE.md). Every run now uploads checkpoints to S3 as it
  trains.
- **Nothing is running.** No pods.

## Experiments

| id | what | status | W&B group → `results/moba/` | record |
|---|---|---|---|---|
| E1 | first TokenWM smoke runs (mix100) | done | `E1-smoke`: `vuzjgdh8`, `ox4cx47i` | [../EXPERIMENTS.md](../EXPERIMENTS.md) (legacy) |
| E2 | first full mix1k run | done | `E2-mix1k`: `asdg1vl1` | legacy |
| E3 | change gate + event sampling | done | `E3-gate-events`: `65j98z3b`, `s3ilbx02` | legacy |
| ablations 1–17 | 5-minute A/B rounds (gate, action tokens, geometry, navgrid, walking ×10, health, regen, cast input) | done | `ablate1` … `ablate17` (36 runs) | legacy |
| E4 | baseline v2, full mix1k, 1 h, teacher-forced | done; final checkpoint lost | `E4-baseline-v2`: `kxe2bh1k` | [E4-baseline-v2-1h.md](E4-baseline-v2-1h.md) |
| E5 | autoregressive rollout training from E4, 20,750 updates; then continued 1,700 more from 20,500 | stopped (pod deleted); @20,500 kept, continuation lost | `E5-rollout`: `93m7sktq` (main), `soeosr9h` (continued), `zll4trqi`, `hm9mnnoy` | [E5-rollout-training.md](E5-rollout-training.md) |
| E6 / E6b | E5 @3,500 + real-sim counterfactual siblings; E6b adds the effect losses | stopped; E6b checkpoints kept | `E6-counterfactual`: `uo6sal6q`, `d9xnnj0a` | [E6-counterfactual-siblings.md](E6-counterfactual-siblings.md) |
| **E8** | Dream-vs-real reward audit; XP loss ×100; clamp off | **concluded: XP hallucinated off-distribution** | local (`results/moba/local_logs/xpsmoke_*`) | [E8-xp-hallucination.md](E8-xp-hallucination.md) |
| **E9** | Dyna smoke: + the dream policies' own real games | concluded (smoke): 3–7× less XP hallucination | `E9-dyna-smoke`: `usrc41qc`, `tnnlmx2e` | [E9-dyna-smoke.md](E9-dyna-smoke.md) |
| **E10** | WM planner at 60 fps vs the shipped net, 120 games | **concluded: negative** (0 wins, ≈ random) | — (eval only) | [E10-wm-planner-60fps.md](E10-wm-planner-60fps.md) |
| **E11** | E5 @20,500 resumed on mix1k + mixrand1k (random play); held-out random games are the main metric | stopped @2,300 for E12: random-play hallucination fixed | `E11-random-data`: `mlbh5d3l` | [E11-random-data.md](E11-random-data.md) |
| E12 | E11 @2000 + dyna2 (the dream policies' own real games) | stopped @~4,350; random h1024 60 → 44 | `E12-dyna2`: `xfldazic` | [E12-dyna2.md](E12-dyna2.md) |
| **E20** | Scenario mining: 2M real-simulator tower-death slices (1M died / 1M survived) from 12,794 training events | **corpus done** (S3); training loader next | — | [E20-scenario-mining.md](E20-scenario-mining.md) |
| **E15** | Tower deaths in the dream: audit, attack-graph analysis, 6-arm fix ablation; E16 combined run | concluded: at full scale the attack graph beats control modestly (47% vs 38%, E19), but tower prediction degrades with longer training (11% at +1,500, E18); dream tower damage lags reality (13% vs 2% health at death); a death-from-health rule doesn't fix it | `E15-tower-dynamics`, `E16-towers`, `E17-attack-ab`, `E18-attack`, `E19-full-config-ab` | [E15-tower-dynamics.md](E15-tower-dynamics.md) |
| **E14** | Dream PPO policy: role heads, PufferLib MinGRU vs MLP, entropy, 2× steps, E12 dream, reward shaping (P6–P11) | concluded: **arm C best** (radiant 18.1 vs the net's 14.3, 0 wins; dire 100% lost); shaping negative | `P6-roleheads` … `P11-reward-sweep` | [E14-dream-ppo-policy.md](E14-dream-ppo-policy.md) |
| **E13** | Dream PPO in E11: P4 (mix curriculum), P4b (+ random start states), P5 (100× steps) | P4/P4b concluded; P5 stopped @83.8M: real return fell to 2.34 (P4b 6.40) | `moba-dream-rl` `P4-mix-e11`, `P5-100x` | [E13-dream-ppo-e11.md](E13-dream-ppo-e11.md) |
| **E7** | PPO in E5's dream (P0 first run, P1 LR × minibatch sweep, P2 real-game evaluation) | **concluded: negative** | `moba-dream-rl`: `P0-first-5min`, `P1-sweep-mac`, `P1-sweep-h200`, `P2-arena-real` | [E7-dream-ppo.md](E7-dream-ppo.md) |

The pre-E4 work stays in the single legacy log, [../EXPERIMENTS.md](../EXPERIMENTS.md),
which is not extended. Every run since E4 has one file here.

## Proposed next

- **Rubric loss A/B** (circle hinge + trajectory penalties) from E5 @20,500, targeting
  jitter: [../WM_FEEDBACK_RL.md](../WM_FEEDBACK_RL.md). The code is ready;
  `--rubric-circle` and `--rubric-traj` are uncommitted.
- **Diagnose E7's transfer failure:** replay real actions through the WM.
- **A decision on rules R1–R10:** audit / remove / keep.
