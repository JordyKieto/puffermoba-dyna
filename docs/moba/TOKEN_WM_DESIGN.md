# Token world model for Ocean MOBA: design

Next-state prediction from the full game state and all 10 heroes' actions:
`s_{t+1} = f(s_t, a_t^{1..10})`, on the Markov-tested `MobaState` of
[DATA.md](DATA.md). General principles are in [TOKEN_WM_GUIDE.md](TOKEN_WM_GUIDE.md).
This page is what was taken from the guide for this game, what was not,
and why. Code: `mobawm/wm/`.

The target is neither an MVP that repeats the naive mistakes nor a system
built ahead of evidence. Every component below answers a specific failure
mode the guide describes, or a specific property of this game.

## Facts about the data that shape the design

Measured on the corpus:

- **206 entities in fixed slots,** 1 map, 128×128. Heroes 0–9, creeps
  10–109, neutrals 110–181, towers 182–205.
- **Changes are sparse.** Per tick, hero x/y change about 60% of the time,
  creep positions about 22%, neutral positions 1.4%, tower health 0.9%,
  alive flags under 0.5%. Copying the state forward is a strong baseline.
- **Heroes never appear dead.** They respawn in the same tick, so a death is
  a position jump plus a health reset, with spawn jitter drawn from the RNG.
- **Several fields are fixed functions of other fields** (see Targets).
  Predicting them wastes capacity and invites inconsistency.
- **Target lists are proven state** (Markov test, 90/99), but they are a
  deterministic function of positions at the last rescan tick (every 5 ticks
  for creeps and neutrals, every 3 for towers).

## Tokens

One token per entity (206), plus one global token.

**Unified part (every entity, same encoder).**
- A shared learned "entity" vector.
- Embeddings of entity kind (hero / creep / neutral / tower) and team
  (radiant / dire / neutral).
- A shared MLP over the fields every entity has: alive, health fraction,
  health / 1000, damage, move modifier, stun / move / attack timers, is_hit.
- Position through Fourier features.
- A 9×9 local wall patch from the static map, through an MLP.

This is the "language" every unit shares, so attention compares a hero and a
creep on the same axes.

**Class-specific part (added on top).**
- A class embedding: hero class × team (10 classes), creep, neutral, tower.
- A per-class MLP over that class's own fields:
  - heroes: mana fraction, level, XP, max health, Q/W/E timers;
  - creeps: lane embedding, waypoint;
  - towers: max health, which stands in for tier.

**Global token.** Tick phase features (tick mod 3 / 5 / 150 / 600, as
Fourier and one-hot) and `creep_idx`. These drive spawns and rescans. The
global token is always a valid key, so no query ever has an empty key row.

**Why absolute position is fed.** It is one fixed map: position encodes
walls, lanes and waypoints. It is normalized and Fourier-encoded, never raw.
Relative geometry goes in the attention bias.

**Not used:** slot/identity embeddings. They invite memorizing "slot 191
changes", and generalization across identities was an explicit lesson of
the prior study.

## Actions

Each hero's 6 heads (7, 7, 3, 2, 2, 2) are embedded and summed into one
action vector, added to that hero's token. The action is a condition, never
part of the state features. All 10 heroes' actions are always present.

## Interaction

- **2 pre-LN transformer blocks,** d = 128, 4 heads, over 207 tokens.
- **Keys are alive entities plus the global token.** Dead entities still
  query, because they need context to predict their own respawn. They are
  not keys.
- **Learned logit bias,** no hard radius:
  - `−softplus(w_h) · dist_ij / 16`, per head, so heads choose between local
    and global attention;
  - a learned per-head team-relation bias (same / enemy / neutral).
- **Full attention is kept.** With 207 tokens it's cheap. With the distance
  bias and clean masking, the guide's pathologies (empty rows, scale
  saturation, no locality prior) are all addressed.
- **No latent bottleneck or entmax yet.** Add them only if attention
  diagnostics show a problem.

## Memory: a GRU per entity

A GRU cell per entity token (shared weights) carries a hidden state across
ticks. The previous hidden state is added to the token **before**
interaction, so memory influences what is attended to. The interaction
output then updates the hidden state.

This is also why the target lists are left out of the input. They are a
deterministic function of positions at the last rescan (≤ 5 ticks ago), and
the GRU sees those ticks. Feeding the lists directly would require predicting
them in rollouts (pointer-set prediction), which is more machinery than the
evidence justifies now.

Training uses windows of 16 ticks: the first 4 are burn-in (no loss), and
the loss is on the remaining 12. Hidden state starts at zero.

## Targets

**Predicted** by typed heads (zero-initialized, so the model starts at
persistence):

| field | classes | head |
|---|---|---|
| x, y | hero, creep, neutral | Δ regression |
| health | all | Δ regression |
| mana, xp, Q/W/E timers | hero | Δ regression |
| move modifier, stun / move / attack timers | hero, creep, neutral, tower (as applicable) | Δ regression |
| waypoint | creep | Δ regression |
| alive | creep, neutral, tower | BCE on "flips this tick" |
| is_hit | all | BCE on next value |
| lane, team | creep | CE on next value; the current value is added to the logits, so the default is to keep it |

- **Regression targets** are divided by a per-(class, field) scale: the std
  of non-zero deltas in training data. Huber loss.
- **Each (class, field) loss is averaged on its own and then summed,** so
  rare fields weigh as much as common ones.
- **Supervision masks:** a (class, field) is supervised only where it can
  change. Entities dead at both t and t+1 are not supervised.

**Derived, not predicted.** In rollouts these are filled by game rules or
lookups fitted on training data:

- level ← XP (upstream's `XP_FOR_LEVEL` table);
- hero max health, max mana, damage ← (class, level);
- creep type, attack cooldown, XP-on-kill, damage, max health, tile id ←
  alive + team;
- a unit that dies gets upstream's `kill_entity` values (position 0, health 0);
- tick += 1; `creep_idx` += 30 (mod 100) on wave ticks;
- `target_pid`: not predicted (reset every step; an event label only).

**Not modelled yet:**
- **Ego observations inside the dream need `grid` and `pids`.** A
  follow-up is to rebuild them from walls + predicted entities and measure
  the gap from stale tiles.
- **The RNG.** Respawn and spawn jitter is irreducible noise. The model
  predicts the mean and is evaluated knowing that.

## Evaluation (smoke test)

- **Split by game:** train on games 0–25 (both configs of each triple),
  hold out games 91–99 (9 games, all three configs).
- **Training health:** loss curve and its slope over the last quarter,
  gradient norm, NaN count.
- **One-step, held-out and train:** per-field MAE in natural units, model vs
  persistence, on all supervised entries and on entries that changed. Alive
  event recall.
- **Open-loop horizon:** after 8 burn-in ticks on real states, roll out H =
  1, 4, 16, 64 feeding predictions back. Compare hero position / health,
  creep position and alive, and tower health against truth, model vs
  persistence.
- **Action control:** the same one-step metrics with actions shuffled
  across the batch. Hero movement error must get worse.
