# MOBA: experiments index

The research question: **can we learn a world well enough to teach an agent in it?**
This follows `carla-imitation-learning` (learning to act) and `neural-simulators`
(learning to simulate).

Evidence standard: [../EVIDENCE_RULES.md](../EVIDENCE_RULES.md). Claims with evidence
classes: [../paper/CLAIMS.md](../paper/CLAIMS.md). Where every number comes from:
[../PROVENANCE.md](../PROVENANCE.md). Doc map: [../MAP.md](../MAP.md).

## Headline (2026-10-05, gen 4)

- **Dream policies now win 28–45% of real radiant games against the shipped net at their peak (E38).**
  - `all_handcrafted` (win + tower damage + fall + champion kills + siege + ancient, XP ×0.5) won **27/60** at 61M steps.
  - `formula_xp0.25` (game reward + win, XP ×0.25) won **17/60** at 25M steps.
  - The key is **turning XP down**: the dream overpays for farming. Every curve peaks and then declines, so checkpoints are
    chosen by the real-game evaluation that runs alongside training. Dire: 0 wins.

## Headline (2026-10-05)

- **Distributed dream PPO runs 26× faster, but scale alone doesn't win (E37).**
  - **Speed:** the policy side (observations + the shipped net) now runs on the GPU and is exact, giving ≈25k steps/s per A40.
    A 5-pod FedAvg fleet reaches ≈116k steps/s.
  - **Real games, evaluated after every update:** play peaked at 91M steps (4/60 wins, 3.25 towers/game) and declined over
    613M steps. The policy overfits the dream.
  - **Next:** a Dyna loop (the policy's real games into the world model's data), plus choosing checkpoints by real games.
- **Dream policies won real games against the shipped net for the first time (E36):** 25 wins in 4,380 radiant games.
  - The key ingredient is **long dream episodes (512–1,024 ticks) plus a win reward on top of the game's own reward**.
  - The wins are rare (best arms 3–4/60) and swing with the seed. Dire is 0 wins in 4,380.
- **Tower precision finally moved (E35):** a false-tower-death penalty in the loss takes precision from 40% to 71–77%.
  At ×1.5 the dream predicts tower falls at the real rate (1,513 vs 1,561).

## Headline (2026-10-04)

- **New best world model: P1 @1000 (E33)**, which is E12 @4000 plus 1,000 updates with only the learning rate
  lowered to 1e-4.
  - **vs E12 on the 100-game tower evaluation:** precision equal (40%), recall 71% vs 65%, siege damage 0.77×
    vs 0.65×.
  - **Other measures:** events 60% vs 57%; hero position error 3.85 vs 4.59.
  - It is the default dream.
  - The architecture additions (predicted attack edges E29, stored target lists E30, grid CNN E31) did not
    beat it. The instability at lr 1e-3 was the learning rate (E32).
- **Training is 11× faster per pod and runs on 20 pods (E34).**
  - **Per pod:** one attention-bias einsum backward was 88% of GPU time; replacing it with masked adds, plus
    `--compile`, cut A40 updates from 11 s to 0.94 s.
  - **Fleet:** 20 pods (zelda-RL FedAvg over Upstash) run at about 6,950 lanes/s, about 43× the old
    single-H200 run. `fleet20` is training 20,000 updates per pod from P1 @1000.
- **Tower metric:** use the 100-game evaluation (`scripts/moba/tower_eval_big.py`, EVIDENCE_RULES §7). The
  64-window in-training line ranked arms wrongly.
- **Not yet written up:** E21–E28 (scenario training, LR/freeze sweeps, P14/P15, event sampling,
  E26/E27 strata, E28 attack-graph from scratch). Their runs are in S3 and W&B.

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
- **Running:** gen-4 reward sweep (20 PPO pods + 2 CPU eval pods, E38).

## Experiments

| id | what | status | W&B group → `results/moba/` | record |
|---|---|---|---|---|
| **E38** | Reward sweep gen 4: 20 rewards (sparse → handcrafted, XP down, champion kills), 1,400-tick episodes, real-game eval during training | **running; peaks 27/60 (all_handcrafted) and 17/60 (formula_xp0.25) radiant wins**; decline after peak | `g4_*` | [E38-reward-sweep-gen4.md](E38-reward-sweep-gen4.md) |
| **E37** | Distributed dream PPO, 613M steps, 5-pod FedAvg fleet; GPU observations + shipped net (`--gpu-opp`); real-game eval after every update | concluded: ≈116k steps/s; real play peaks at 91M then declines (dream overfitting) | `PPO-fleet` (pf1) | [E37-ppo-fleet-600m.md](E37-ppo-fleet-600m.md) |
| **E36** | Dream-PPO reward-structure sweep, 3 generations, 73 arms (A40 pod per arm) | **first real wins (25/4,380 radiant)**; long episodes + win reward; dire 0 | `RW-reward-sweep` | [E36-reward-sweep.md](E36-reward-sweep.md) |
| **E35** | False-tower-death penalty (`--tower-fp-weight`), calibrated dream (fleet20c/d/e), dream PPO P16/P17 | concluded: precision 40% → 71–77%; ×1.5 calibrated; PPO didn't improve | `fleet20c_r0` … | [E35-tower-penalty-calibrated-dream.md](E35-tower-penalty-calibrated-dream.md) |
| **E34** | Throughput: einsum-backward fix + `--compile` (11× per pod); 20-pod FedAvg fleet via Upstash | **running** (`fleet20`, 20,000 updates/pod) | `fleet20_r0` | [E34-throughput-fleet.md](E34-throughput-fleet.md) |
| **E33** | Plain E12 @4000 at lr 1e-4, 1,000 updates (P1) | **concluded: new best; default dream** | `p1_e12_lr1e-4` | [E33-e12-lr1e-4.md](E33-e12-lr1e-4.md) |
| **E32** | Stability sweep: lr 1e-3/3e-4/1e-4 × EMA (S1–S4, G2) | concluded: lr 1e-4 stable; EMA doesn't fix 1e-3 | `s1`…`s4`, `g2_grid_long` | [E32-lr-ema-sweep.md](E32-lr-ema-sweep.md) |
| E31 | Global 64×64 map + CNN encoder (G1, G2) | concluded: no stable tower gain | `g1_e12_grid` (`6odugwtq`) | [E31-grid-cnn.md](E31-grid-cnn.md) |
| E30 | Stored target lists as state + aggro inputs (A0 vs A1b) | concluded: no tower gain on the 100-game eval | `a0_e12_control` (`pvgc94tc`), `a1b_e12_lists` | [E30-target-lists.md](E30-target-lists.md) |
| E29 | Predicted attack edges: decay, distance bias, rebuilt kill edges (K0–K3) | concluded: edges improve, towers don't | `k0`…`k3` | [E29-attack-edges.md](E29-attack-edges.md) |
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
| **E20** | Scenario mining: 2M real-simulator tower-death slices (1M died / 1M survived) from 12,794 training events | corpus done; **E21 trained with it (lr 1e-3): dives not learned, towers degrade**; E22 (lr 3e-4) running | — | [E20-scenario-mining.md](E20-scenario-mining.md) |
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
