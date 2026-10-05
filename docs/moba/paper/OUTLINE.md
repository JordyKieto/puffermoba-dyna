> **Superseded (2026-10-05)** by [paper.md](paper.md), *Learning in Dreams, Winning in Reality*. Kept as the record of
> the 2026-09-27 outline, written when transfer was still negative.

# Paper outline (draft for review)

**Working title.** *Learning a World to Teach an Agent: A Full-State World Model of a
Ten-Hero MOBA, and What Its Dream Could Not Yet Teach*

**Alternatives.**
- *Rollouts, Not Steps: Training and Inspecting a Multi-Agent Game World Model*
- *A Microscope for Learned Worlds: Side-by-Side Rollouts of a Ten-Agent Game Simulator*

**Format.** A technical report in the same structure and evidence grading as the two
companion reports:
- `carla-imitation-learning` asked whether a policy can learn to act in a world;
- `neural-simulators` asked whether we can simulate a world;
- this report asks whether we can **learn a world well enough to teach an agent in it**.

**Spine.**
1. **A world model is judged by its rollouts, not its steps.** One-step loss and
   open-loop fidelity pull in different directions (W1–W4).
2. **Training on your own rollouts fixes long horizons but costs one-step
   precision.** Real-simulator counterfactuals win the precision back and teach what
   actions do, but at a wall-clock price for long horizons (W8–W10).
3. **Dream reward can lie.** A policy that more than doubles its dream reward loses
   every real game (R1–R2).
4. **The instrument is part of the contribution:** a side-by-side replay microscope,
   a byte-exact state capture, and a real-simulator counterfactual generator.

**Evidence tags** (as in [CLAIMS.md](CLAIMS.md)): M measured · Q qualitative · I
inferred · A author-confirmed · U unverified. U never appears as a result.

---

## Abstract (~200 words)

- **Question:** can a learned model of a full 10-hero MOBA game be good enough that
  a policy trained only inside it plays the real game?
- **Setup:**
  - PufferLib's Ocean MOBA, made deterministic and fully capturable (T1).
  - A 0.66M-parameter token world model over all 206 units, conditioned on all 10
    heroes' actions.
  - 1,000 recorded games.
  - Evaluation by open-loop rollouts on held-out games, and side by side with the
    real game in a replay viewer.
- **Findings:**
  - one-step training fails open loop (W1);
  - autoregressive rollout training makes long horizons 5–12× better but one step
    2× worse (W3–W4);
  - real-simulator counterfactual siblings recover one-step precision and action
    effects (W8, W10);
  - PPO in the dream wins 0 real games (R2).
- **Honesty:** spawns and level-derived stats are rules, not learned (W12, W15).
- **Contribution:**
  - the regimes and their trade-off;
  - a negative transfer result;
  - the instruments.

## 1. Introduction

- Learned simulators are increasingly proposed as places to train agents. The
  evidence is mostly one-step error, short imagined rollouts or pixels, rarely
  multi-agent full state, and rarely checked against the real game on the same
  actions.
- **The distinction the paper is built on:** predicting the next state vs being an
  environment. An environment consumes its own predictions for thousands of ticks,
  under actions it did not choose.
- **The concrete question:** a 10-hero MOBA with creeps, neutrals, towers and
  abilities. Can a small learned model roll a whole game forward from all 10 heroes'
  actions, and teach a policy?
- **Contributions:**
  1. **A deterministic, capturable MOBA,** so every state can be restored exactly
     (T1). It enables byte-exact observations from dreamed states (T2) and
     counterfactual forks (T3).
  2. **The one-step vs rollout trade-off, measured by horizon** (W1–W5, Figure 2).
  3. **Real-simulator counterfactual training** and its trade-off (W8–W11).
  4. **A negative dream-to-real transfer result** with a sound harness (R1–R3).
  5. **The replay microscope** (Figure 1, §5).
- **Figure 1:** the viewer at step 300 of a held-out game.
  - Left panel: E4 (teacher-forced), ghost overlay on. Heroes drift toward the edges,
    and ghost lines reach back to the true positions.
    `media/moba/20260927_1332/E4_teacherforced_u3000_ghost_step0300.png`.
  - Right panel: E5 at the same step (`E5_rollout_u20500_ghost_step0300.png`).

## 2. Related work

Draws on [../PRIOR_WORK.md](../PRIOR_WORK.md).
- Latent world models and model-based RL: World Models, Dreamer, IRIS, DIAMOND.
- Object- and token-centric dynamics.
- Exposure bias, scheduled sampling, DAgger for sequence models.
- Counterfactual data augmentation.
- Playable neural games.
- Where this sits: full symbolic state, many agents, action-conditioned,
  autoregressive, checked against the source simulator on identical actions.

## 3. Problem formulation

- The state has 206 entities × 26 fields plus globals, and all 10 heroes act every
  tick.
- A model is an environment if it maps (state, all actions) to the next state and
  stays faithful when fed its own output.
- **Metrics** (defined before use, as in the E5 record):
  - held-out loss@h for h = 1 … 4,096;
  - hero position error at each horizon;
  - rubric metrics: circle hit, jitter, speed ratio.

## 4. Environment and data

- PufferLib Ocean MOBA:
  - the per-game SplitMix64 random stream and the non-resetting step (§1 of
    PROVENANCE);
  - the Markov test (T1);
  - byte-exact ego observations (T2).
- The mix1k corpus: 1,000 games, net and script controllers, 900/100 split by game,
  2.4M training transitions.
- Samples: `samples/moba/`.

## 5. Instruments

- **The replay microscope:** the real game next to the model on the same actions;
  hover stats, cast and death flashes, ghost overlay; a headless API; recordings.
  Figure 1; supplementary videos.
- **The counterfactual fork generator:** byte-exact (T3), ~30–50k samples/s per core
  (T4), with its edit statistics (T5).
- **The real-game arena:** reuses the shipped policy and the scripted bots unchanged
  (R2 harness check).

## 6. Model

- TokenWM: 207 tokens, actions added to their heroes' tokens, attention with a
  distance/team bias, a per-unit GRU, typed delta heads, a change gate; 0.66M params.
- **Rules vs learned** (R1–R10), stated as a design limitation, not hidden (W12,
  W15).
- Training regimes:
  - teacher-forced (E4);
  - rollout lanes with log-normal spans and step decay (E5);
  - counterfactual siblings plus effect, no-effect and affected losses (E6b).

## 7. Results

- **7.1 One step vs sustained rollout** (W1–W5). Figure 2 = `results/moba/figures/horizon_loss.png`.
- **7.2 Training dynamics by horizon** (W2, W8, W9). Figure 3 = `horizon_over_training.png`.
- **7.3 Trajectory quality:** jitter, speed, circle hit (W7).
- **7.4 What actions do:** effect explained on counterfactuals (W10, W11). *A held-out
  version is needed before this becomes M.*
- **7.5 Teaching an agent in the dream** (R1–R3). Figure 4 = `dream_ppo_sweep.png`;
  Figure 5 = `arena_real_game.png`.

## 8. Failure analysis

- Jitter (W7), tower health (W14), transfer (R2–R4), and the lost checkpoints as a
  methods lesson (PROVENANCE §3.3).

## 9. Limitations and threats to validity

- Single seeds.
- Rules R1–R10.
- One data mix (net and script players).
- h4096 from a few long games.
- Training-set effect metrics.
- Retroactive provenance for the early work.

## 10. Future work

- The rubric loss ([../WM_FEEDBACK_RL.md](../WM_FEEDBACK_RL.md)).
- A Dyna loop: real games played by the current policy feed the world model.
- Counterfactuals from the model's own drifted states.
- Human pairwise realism judgments.
- A decision on the rules.

## Appendices

- **A:** provenance, from [../PROVENANCE.md](../PROVENANCE.md).
- **B:** the claim audit.
- **C:** reproducibility commands (PROVENANCE §7).

## To close before submission (from CLAIMS)

- **W6:** re-run the attractor check on E5 @20,500.
- **T2:** a committed byte-exact observation test.
- **W10/W11:** a held-out counterfactual evaluation.
- **Seeds:** ≥2 per arm for E5 vs E6b and for the sweep.
- **R2:** real-game evaluation of every sweep config.
- **W2:** the E4 control table, re-run from surviving checkpoints or dropped.
