# Learning a World to Teach an Agent: A Full-State World Model of a Ten-Hero MOBA, and What Its Dream Could Not Yet Teach

*Draft, started 2026-09-27. Every number cites a claim in [CLAIMS.md](CLAIMS.md);
every claim cites an artefact. Sections marked TODO are outlined in
[OUTLINE.md](OUTLINE.md).*

## Abstract

Learned simulators are increasingly proposed as places to train agents. We ask
whether a small learned model of a full ten-hero MOBA game (heroes, creeps, neutral
camps and towers; 206 units, all ten heroes acting every tick) can be faithful
enough that a policy trained only inside it plays the real game.

We make the game deterministic and fully capturable, so that any recorded state can
be restored exactly. We train a 0.66M-parameter token world model on 1,000 recorded
games, and judge it the way it would be used: fed its own predictions for up to 4,096
ticks on held-out games, and watched next to the real game on identical actions.

Training on one-step targets gives low one-step error, but rollouts fall apart, with
hero positions no better than copying the start state after a few hundred ticks.
Training on the model's own rollouts cuts long-horizon error 5–12×, at the cost of
twice the one-step error. Counterfactual siblings from the real simulator, the same
state under edited actions, recover one-step precision and teach the model most of
what an ability press changes, but they buy no long-horizon gain at matched
compute. A policy trained with PPO inside the best model more than doubles its
dream reward, yet wins none of 120 real games.

Spawning and level-derived statistics are supplied by rules, not learned. We
release the instruments with the results: the deterministic capture, a side-by-side
replay microscope, and a byte-exact counterfactual generator.

## 1. Introduction

A model that predicts the next state well is not yet an environment. An environment
has to accept its own output as input, tick after tick, for as long as an agent
keeps acting, under actions the model did not choose. The gap between the two is
the subject of this report.

Two earlier reports in this line asked the preceding questions. The first asked
whether a policy can learn to act in a world from demonstrations
(`carla-imitation-learning`). The second asked whether a compact model can stand in
for a world as an interactive simulator (`neural-simulators`). Here we ask the
question that joins them: **can we learn a world well enough to teach an agent in
it?**

We study it in a setting that is hard for world models in specific ways. PufferLib's
Ocean MOBA is a ten-hero, two-team game:
- **Many units:** 206 units whose interactions are local but dense (lanes of creeps,
  neutral camps, towers, abilities with cooldowns and mana).
- **Every hero acts every tick:** a model must be conditioned on all of their
  actions at once.
- **Long games:** 1,200–4,200 ticks. A learned environment for this game must stay
  coherent over thousands of self-fed steps.

Figure 1 shows the failure this report starts from and the instrument we use to see
it. A model trained on one-step targets, run on its own predictions, lets the heroes
drift toward the edges of the map within a few hundred ticks. By step 300 its hero
positions are no better than copying the starting state (claim W1).

> **Figure 1** *(to compose from the viewer recordings)*: the replay microscope at
> step 300 of a held-out game. Left: the real game. Right: the teacher-forced model
> (E4) on the same actions, with ghost markers at the true hero positions. Lines join
> each predicted hero to where it really is.
> Source: `media/moba/20260927_1332/E4_teacherforced_u3000_ghost_step0300.png`
> (md5 in that directory's MANIFEST.json). Metrics at that frame: hero position
> error 44.1 cells vs 45.4 for copying the start state.

Our contributions:
1. **A deterministic, fully capturable MOBA.** A recorded state restores and
   continues byte-for-byte (T1). This makes three things possible: byte-exact agent
   observations from model-generated states (T2), counterfactual forks with the real
   simulator (T3), and a real-game arena that shares code with the data collector.
2. **The one-step vs rollout trade-off, measured by horizon.** We report held-out
   loss at horizons 1 to 4,096 for three training regimes (W1–W5, W8–W9), with
   trajectory-quality metrics (W7).
3. **Real-simulator counterfactual training.** Siblings share state and memory and
   differ in one hero's action. Auxiliary losses on their differences teach the
   model what actions do (W10–W11).
4. **A negative dream-to-real transfer result.** We use a harness validated by
   reference players and an untrained baseline on the same seeds (R1–R3).
5. **The replay microscope.** The real game and the model side by side on the same
   actions, with per-unit inspection. We treat it as part of the method: every
   qualitative claim here was made by watching it (W13–W14).

We are explicit about what the model does *not* learn. Unit spawns, creep
statistics, hero level and level-derived statistics, attack targets and the tick
counters are supplied by hand-coded rules (§6). Every result includes them.

## 2. Related work

TODO: see OUTLINE §2 and [../PRIOR_WORK.md](../PRIOR_WORK.md).

## 3. Problem formulation

TODO (OUTLINE §3).
- **Metrics:** held-out loss@h (h = 1, 4, …, 4,096) on the same 32 windows from 100
  held-out games; hero position error at each h; the trajectory rubric.

## 4. Environment and data

TODO (OUTLINE §4). The data: 1,000 games, split 900/100 by game, 2,395,549 training
transitions and 267,768 held-out (PROVENANCE §2).

## 5. Instruments

TODO (OUTLINE §5).

## 6. Model and training regimes

TODO (OUTLINE §6). Include the rules table R1–R10 from
[../WM_ARCHITECTURE.md](../WM_ARCHITECTURE.md).

## 7. Results

### 7.1 One step versus sustained rollout

Held-out loss at horizon h: generated table `results/moba/tables/horizon_loss.md`,
Figure 2 = `results/moba/figures/horizon_loss.png`.

| model | h1 | h4 | h16 | h64 | h256 | h1024 | h4096 |
|---|---|---|---|---|---|---|---|
| teacher-forced (E4 final) | 0.78 | 3.18 | 15.50 | 80.45 | 213.49 | 323.44 | 428.79 |
| rollout-trained (E5 @20,500) | 1.70 | 5.14 | 7.70 | 18.99 | 24.64 | 34.84 | 36.84 |
| rollout + counterfactual (E6b @3,500) | 0.90 | 3.55 | 8.12 | 21.25 | 34.33 | 48.94 | 59.78 |

**What the table shows** (single seed per regime):
- **Teacher-forced training has the best one-step loss,** and its loss grows ~550×
  from h1 to h4,096 (W1).
- **Rollout training cuts the loss 5–12× from h64 on,** and doubles it at h1 (W3,
  W4).
- **Counterfactual siblings recover most of the one-step precision** (W8). At
  matched wall clock they do not improve h16–h1024 over rollout training alone (W9).

TODO: 7.2 training dynamics; 7.3 trajectory rubric; 7.4 action effects (needs the
held-out evaluation).

### 7.5 Teaching an agent in the dream

TODO. Figure 4 = `results/moba/figures/dream_ppo_sweep.png`, Figure 5 =
`results/moba/figures/arena_real_game.png`; claims R1–R3.

## 8. Failure analysis

TODO.

## 9. Limitations and threats to validity

TODO (OUTLINE §9).

## 10. Future work

TODO.

## Appendix A. Artifact provenance

See [../PROVENANCE.md](../PROVENANCE.md). Checkpoints are identified by md5 in
`artifacts/moba/MANIFEST.json`; W&B histories are exported to `results/moba/`.
