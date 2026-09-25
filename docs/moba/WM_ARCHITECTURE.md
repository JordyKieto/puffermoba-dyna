# MOBA world model: architecture as built

This describes the model that exists today, from the code
(`mobawm/wm/model.py`, `mobawm/wm/data.py`, `mobawm/wm/train.py`), not the
plan. [TOKEN_WM_DESIGN.md](TOKEN_WM_DESIGN.md) has the reasoning behind it,
[GAME_MECHANICS.md](GAME_MECHANICS.md) the game it models.

**One step:** turn the full game state at tick t into tokens, add the 10
heroes' actions and each unit's memory, let tokens attend to each other, then
predict what changes by tick t+1. Size: d = 128, 2 layers, 4 heads, about
0.66M parameters.

## At a glance

| question | answer |
|---|---|
| Architecture | Transformer over a set of tokens (2 pre-LayerNorm blocks), then a GRU per unit, then typed output heads |
| Tokens | **one per unit** (206) + **one global token** = 207 |
| Unified entity model? | **Yes.** Every unit is built from the same shared base, plus a class-specific add-on. All units share the attention, the GRU and the output heads |
| Global attention? | **Yes.** Full attention: every token can attend to every living unit and the global token |
| Hierarchical? | **No.** One flat set of tokens. No local pooling, no latent bottleneck |
| Timers in the state? | **Yes, per unit:** stun, slow/boost, attack cooldown; Q/W/E cooldowns for heroes. They are inputs and prediction targets |
| Global clock? | **Yes, in the global token:** the phase of the 150-tick creep waves, the 600-tick neutral respawns and the 3/5-tick rescans |
| Navigation grid? | **No.** Geometry is a 9×9 wall patch around each unit, plus its absolute position and pairwise distances in attention. No pathfinding graph, no whole-map encoder |
| Memory | a GRU hidden state per unit, carried across ticks |

## Tokenization

Each of the 206 units becomes one 128-number token, the **sum** of these
parts.

**Shared by every unit** (same layers for a hero, a creep, a tower):

| part | how it's encoded |
|---|---|
| "this is a unit" | one learned vector |
| kind | embedding: hero / creep / neutral / tower |
| team | embedding: radiant / dire / neutral |
| common stats | small MLP over 9 scaled numbers: alive, health ÷ max health, health ÷ 1000, damage ÷ 100, speed modifier ÷ 2, stun timer ÷ 20, slow/boost timer ÷ 30, attack cooldown ÷ 10, got-hit flag |
| position | x and y (÷ 128) through Fourier features (8 frequencies each), then an MLP |
| terrain | the 9×9 patch of the wall map centred on the unit (81 walls/not-walls), through an MLP |

**Class-specific add-on** (only that class has these layers):

| class | extra part |
|---|---|
| all | class embedding: 10 hero classes (5 classes × 2 teams), creep, neutral, tower |
| heroes | MLP over mana ÷ max mana, level ÷ 30, XP ÷ 5000, max health ÷ 2000, Q/W/E cooldowns ÷ 100 |
| creeps | MLP over waypoint index ÷ 12, plus a lane embedding |
| towers | MLP over max health ÷ 4500 (stands in for the tower's tier) |
| neutrals | nothing extra |

**Global token** (one, shared by the whole game):
- **creep waves:** sin/cos of `tick mod 150`;
- **neutral respawns:** sin/cos of `tick mod 600`;
- **rescan phase:** one-hot of `tick mod 3` (towers) and `tick mod 5`
  (creeps and neutrals);
- **which creep slots the next wave reuses:** `creep_idx ÷ 100`;
- **game time:** `tick ÷ 5000`.

**Deliberately absent:**
- **raw unscaled numbers:** every number is scaled first;
- **per-slot identity embeddings:** they invite memorizing "slot 191".

**Actions.** Each hero's 6 action heads (move y, move x, target mode, Q, W,
E) are embedded (one table per head) and summed into one vector. That vector
is added to that hero's token. Actions are a condition on the hero token,
never part of the state features.

**Memory.** Each unit's GRU hidden state from the previous tick goes through
a linear layer and is added to its token before attention.

## Attention

- **2 blocks.** Each is pre-LayerNorm self-attention with 4 heads, then an
  MLP (4× width).
- **Who can be attended to (keys):** every **living** unit plus the global
  token.
  - Dead units are not keys, but they still query: they need context to
    predict their own respawn.
  - The global token is always a valid key, so no query ever has an empty
    attention row (the NaN failure mode in the guide).
- **Learned bias added to the attention scores,** with no hard radius:
  - **Distance:** `−softplus(w_head) × distance / 16`, with one learned
    `w` per head. Each head chooses how local it is: a large `w` means
    nearby units only, `w ≈ 0` means map-wide.
  - **Team relation:** a learned value per head for "same team", "enemy" and
    "involves a neutral".
- **Full attention, no sparsity.** Every living unit can attend to every other
  (207 × 207 per head). No bottleneck, no local-first stage, no entmax.

## Geometry

**The model gets:**
- **absolute position** of every unit (Fourier features). There is only one
  fixed map, so absolute position also says where the lanes and bases are;
- **the walls immediately around each unit:** a 9×9 patch;
- **distance between every pair of units:** only through the attention bias.

**It does not get:**
- **no navigation grid or pathfinding.** Creeps follow a precomputed shortest
  path (BFS) in the simulator; the model never sees paths;
- **no whole-map terrain encoder** (e.g. a CNN over the 128×128 wall map);
- **no occupancy grid input.** Which cells are occupied has to be inferred
  from unit positions. The simulator's own grid, with its stale tiles, is in
  the saved state but not fed to the model;
- **no lane or waypoint graph tokens.**

So walls further than 4 cells away, and routes around them, have to be
learned implicitly from positions and experience.

## Timers

**Saved in the game state (`MobaState`) per unit:**
- stun timer, slow/boost timer and speed modifier;
- attack cooldown;
- Q/W/E cooldowns (heroes).

**Surfaced to the model:** all of them, scaled, as token inputs (listed
above). Each is also a prediction target:
- predicted as a change from this tick to the next;
- with `--gate`, behind a "does it change" gate.

**Global timers:** the tick, via its phases in the global token.

**Not surfaced:**
- **the target lists** that creeps, neutrals and towers act on between
  rescans. They're proven state, but they're left to the GRU's memory of the
  last few ticks;
- **the random-number state** (respawn jitter). It is noise to the model;
- **a hero's constant attack cooldown length** (8 ticks, the same for every
  hero).

There are no respawn timers in this game; respawn is instant.

## Outputs

One set of heads, shared by every unit.

**Change regression:** for 13 fields, the predicted change divided by that
field's typical change size:
- x, y, waypoint;
- health, mana, XP;
- Q/W/E cooldowns and the attack cooldown;
- stun, slow/boost timer and speed modifier.

It is zero-initialized, so an untrained model predicts "nothing changes".

**With `--gate` (copy by default):** a per-field "does this field change this
tick" output. It starts at that field's observed change rate. The change is
only applied where it says yes. `--gate-fields` limits the gate to chosen
fields.

**The adopted baseline (`--baseline`, v2)** gates every field except
movement. Its loss is balanced so the gated fields don't crowd out walking:
- the gate's yes/no loss ×0.2;
- each gated field's value loss averaged over all supervised ticks;
- the hero x/y terms weighted ×10.

A walking-only test showed that the architecture learns walking as well as a
per-hero MLP. Without that weighting, the full loss buries it. See
EXPERIMENTS.md.

**Other heads:**
- **alive flip:** "this unit dies or spawns this tick";
- **got hit** next tick;
- **creep lane and team.**

**Filled in by game rule, not predicted:**
- level from XP;
- a hero's max health, mana and damage from class and level;
- creep constants;
- wave respawns (which slots, which lane, where);
- the tick and creep-slot counters.

Everything is predicted by one shared head on top of each unit's GRU state;
there are no separate per-class decoders.

## Where this matters for the scoreboard

Gaps between what the model is given and what the abilities depend on
([GAME_MECHANICS.md](GAME_MECHANICS.md)):

- **Ability validity is never given directly.** Whether a pressed Q actually
  fires depends on:
  - cooldown = 0: given;
  - mana ≥ cost: the cost isn't given; it has to be learned per class;
  - a target within 5 cells: only implicit, via attention;
  - class-specific conditions, e.g. the assassin's Q works only on creeps.

  A near-exact model has to learn all of this from data. **Candidate
  addition:** derived "could cast Q/W/E this tick" features. They're pure
  functions of the state, so the model stays honest.
- **Priority order.** Q beats W beats E beats the basic attack. It's implicit,
  and has to be learned too.
- **Targets are the nearest enemy in an 11×11 box.** The attention distance
  bias is Euclidean with a learned slope, not that box, so the model has to
  approximate the rule.
- **Walking collides with occupied cells,** but occupancy isn't an input.
