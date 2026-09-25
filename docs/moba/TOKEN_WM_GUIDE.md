# Building token-based world models for structured game state

A guide for world models whose input is structured simulator state (units,
stats, cooldowns, terrain) rather than pixels. It covers how to tokenize that
state, how to wire attention, how to condition on actions, and what to predict.
It is written from the failure modes that recur when NLP transformer recipes
are applied to game telemetry unchanged.

The short version: **game state is not a sentence.** An entity is a bundle of
heterogeneous attributes plus relationships to other entities and to terrain.
Tokenization and attention should encode that structure. When they don't, a
transformer trains unstably and often ends up no better than an MLP.

---

## 1. Why naive entity transformers fail

The naive recipe is one flat vector per entity (floats, flags and IDs
concatenated), `nn.Linear` to `d_model`, dense self-attention over all
entities, and one loss. It fails for three separate reasons. Tuning LayerNorm,
init or learning rate fixes none of them, because none of them is an
optimization-hyperparameter problem.

### 1.1 Degenerate attention rows (the NaN / gradient-spike source)

Entity counts vary, so batches are padded and attention is masked.

- **A row with every key masked is ill-defined.** With `-inf`,
  softmax is 0/0 = NaN, and the NaN propagates through the whole batch.
- **`-1e9` is not a real fix.** A fully masked row becomes a uniform
  distribution over padding, so the query attends to garbage. In fp16,
  `-1e9` overflows to `-inf` and you are back to NaN.
- **Queries from padded or dead slots** produce outputs that nothing should
  consume. If a loss touches them, gradients flow through meaningless paths.

**Fix:** no query should ever have an empty key set. Always give every query
at least one valid key: itself, or a dedicated null/global token. Use a
mask fill that is finite in your precision (e.g. `-1e4` for fp16). Zero padded
outputs after attention, and exclude padded slots from every loss.

### 1.2 Scale leaking into dot products (softmax saturation)

Raw game quantities span orders of magnitude: map coordinates in the
thousands, health in the thousands, flags in {0, 1}. Projected linearly, they
produce query and key norms far from unit scale. Logits of ±50 saturate
softmax. Gradients are then near zero almost everywhere, and spike when a small
input change flips which key wins, e.g. a unit stepping one tile closer.

**Fix:** no raw float enters a projection.
- Continuous values are normalized to roughly [-1, 1] or [0, 1] first
  (health as a fraction of max, timers as a fraction of their cooldown).
- Coordinates enter as normalized or relative values through Fourier features.
- IDs and discrete states go through embedding tables.

### 1.3 Unstructured all-to-all attention

Dense self-attention lets every entity attend to every other, including pairs
with no causal link: a minion in one lane and a turret in another. With a
sparse supervision signal, the model binds to incidental correlations, and
gradients pass through O(N²) irrelevant paths. The model is also left to
discover space from scratch, with no locality prior.

**Fix:** give attention the structure of the game:
- a relational bias (distance, team, known interaction edges) added to the
  logits;
- restricted participation where the physics justifies it;
- or a bottleneck (see 3.3).

---

## 2. Tokenization

### 2.1 Typed, factorized entity tokens

Build each entity token as a sum of sub-embeddings, one per kind of
information:

```
token = E_unified                      # shared "this is an entity" basis
      + E_class[class_id]              # unit type: hero class, creep, tower tier, ...
      + E_state[discrete_state_ids]    # team, lane, action/aggro state, flags
      + MLP_shared(common_scalars)     # fields every entity has (hp%, position, ...)
      + MLP_class[class](class_scalars)# fields only this class has (mana, cooldowns, ...)
      + MLP_pos(fourier(position))     # position features
      + MLP_terrain(local_terrain)     # what the terrain looks like here
```

- **Keep a unified part and a class-specific part.** Every entity shares the
  same "language" for what they have in common (health, position, team). Then
  a champion and a minion are comparable in attention, while class-specific
  fields don't pollute the shared space.
- **Discrete fields go through embedding tables, never through a linear layer
  as floats.** An integer state ID tells attention immediately what a unit is
  doing, without having to infer it from velocity.
- **Normalize continuous fields per field,** with ranges chosen from the game
  or the data, not learned. Use health fraction, not raw health.

### 2.2 Coordinates

- **Varying maps, or translation invariance matters:** do not feed absolute
  coordinates. Represent positions relative to the query entity (Δx, Δy,
  distance, optionally rotated into its frame), in the attention bias or in
  per-pair features.
- **One fixed map:** absolute position is real information. It tells you
  where the walls, lanes and objectives are. It is then legitimate to feed,
  but only normalized and through Fourier features, never as raw floats.
  Relative geometry still belongs in the attention bias.

### 2.3 Terrain and other spatial fields

Terrain, walls and zones are fields, not entities. Don't stuff "distance to
wall k" features into entity tokens. Use one of these:

- a low-resolution 2D map processed by a small CNN, which entities sample at
  their position (bilinear lookup or cross-attention);
- a local patch around each entity, encoded into its token;
- a small set of static region tokens that entities attend to.

Static context never sends queries.

### 2.4 Hierarchical tokenization

When an entity carries structure of its own (a trajectory history, a polyline
such as a lane), encode it locally first (1D conv, PointNet, small GRU) into
one summary token, and let only summary tokens interact globally. Global
attention should see tens to a few hundred tokens, not raw telemetry slices.

---

## 3. Attention

### 3.1 Learned relational bias instead of hard radii

Do not hand-code a distance cutoff. Global effects (map-wide abilities,
teleports, objective state) break any fixed radius. Instead, add a learned
per-head bias to the logits:

```
score_ij = q_i·k_j / √d  −  softplus(w_h) · ‖p_i − p_j‖ / σ  +  b_h(relation_ij)
```

- Heads with a large `w_h` become local combat heads; heads with `w_h ≈ 0`
  become global heads. The data decides; it is fully differentiable.
- `b_h(relation)` is a learned bias per discrete relation: same team, enemy,
  "j is in i's target list", "j attacked i last step". Known interaction
  edges are cheap and strong priors.

### 3.2 Sparse attention, if you need hard zeros

Softmax gives every key non-zero weight. If distractors must be exactly
excluded, use:
- α-entmax (α = 1.5), or sparsemax (α = 2): both give exact zeros, learned
  per query;
- or class-level causal filtering: measure whether class C influences target
  T at all, e.g. by ablating C and measuring the change in predictive
  log-likelihood, and cut the edge if it doesn't.

Start with softmax + relational bias. Add sparsity only when there's a
measured problem.

### 3.3 Bottlenecks

- **Latent queries** (Perceiver/Wayformer style): M learned latents
  cross-attend to all entities, self-attend among themselves, then are read
  back out. This scales to large N and filters pairwise noise.
- **Bipartite cross-attention:** active actors (heroes) query the context
  pool (minions, towers), but the pool does not attend to itself.
- **Pointer heads** for target selection: one query over candidate keys,
  1 × N, instead of N × N reasoning.

Use a bottleneck when N is large or most pairs are irrelevant. With a few
hundred entities and a good relational bias, full attention is affordable
and often fine.

---

## 4. Action conditioning

- **State and action are separate inputs.** Tokens encode the state s_t.
  The action a_t enters strictly as a condition: added to the acting entity's
  token, or as separate action tokens that entities cross-attend to.
- **Never put `prev_action` (or anything derived from the future) inside
  state features.** Autoregressive rollouts then feed the model its own
  actions in two places, and the model finds shortcuts.
- **Record the joint action.** One global transition with only one actor's
  action is non-identifiable: the same s_t → s_{t+1} appears with different
  "actions". Either record every actor's action, or make sure that unrecorded
  actors are deterministic functions of the state.
- **Test that the model uses actions:** evaluate with actions shuffled across
  the batch. If the loss barely moves, the model ignores them.

---

## 5. What to predict

### 5.1 Deltas, starting at identity

- **Predict Δs = s_{t+1} − s_t, not s_{t+1}.** Absolute targets make the
  model spend capacity on reproducing the input.
- **Zero-init the output layer,** so step-0 predictions are exactly "no
  change". The model starts at the persistence baseline instead of emitting
  garbage that blows up the first losses.

### 5.2 Typed heads, balanced losses

One unweighted MSE over all fields lets the largest-scale fields dominate:
coordinates in the hundreds swamp cooldowns and flags. Instead:

- **Continuous fields:** regression on deltas normalized per field (e.g.
  divided by the per-field standard deviation of non-zero deltas); Huber for
  robustness.
- **Binary fields** (alive, flags): BCE on the next value.
- **Categorical fields** (team, lane, type, state): cross-entropy on the next
  value.
- **Supervise only what can change.** Mask fields that are constant for a
  class (e.g. mana on a tower).

### 5.3 Copy by default

In most game ticks only a few percent of values change. Dense losses then
reward predicting "nothing changes". Counter this with:

- change-aware metrics (see 6);
- optionally, an explicit change head: P(field changes) gating the delta;
- event-balanced sampling for rare mechanics (deaths, level-ups, objectives).

---

## 6. Evaluation

- **Persistence (copy s_t) is the baseline that matters.** It is strong.
  Report every metric against it, in natural units.
- **Change-aware metrics:** error on entries that actually changed; recall
  of events (deaths, spawns, level-ups); false-change rate.
- **One-step vs open-loop:** teacher-forced one-step error says little about
  rollouts. Report open-loop error at several horizons (e.g. 1, 4, 16, 64),
  against persistence at the same horizon.
- **Held-out splits by what determines the data** (game seed, map,
  opponent), never by row: adjacent rows are near-duplicates.
- **Action controls:** real vs shuffled actions.
- **Training health:** loss slope over the last window (still improving?),
  gradient norm, NaN/Inf counts.

---

## 7. Before training: the state must be sufficient

No architecture can learn a target that the input does not determine. Before
modelling, prove that the state is Markov: restore it into a fresh
simulator and check the future is reproduced. Omit each part and check that
the future changes. Scramble what you left out and check that it doesn't.
Hidden timers, stale target lists, spawn counters and RNG state are the usual
omissions.

---

## Checklist

- [ ] State proven sufficient (restore / omit / perturb).
- [ ] No raw floats into projections; per-field normalization; IDs through embeddings.
- [ ] Unified + class-specific token parts.
- [ ] Coordinates normalized through Fourier features (fixed map), or relative (varying maps).
- [ ] Terrain as a field (patch or map), not per-entity distance features.
- [ ] Attention: learned distance bias + relation bias; no empty key rows; finite mask values.
- [ ] Actions as a separate condition; joint actions recorded; shuffled-action control.
- [ ] Delta targets, zero-init output, typed heads, per-field loss scaling, constant fields masked.
- [ ] Metrics vs persistence; change-aware; open-loop horizons; held-out by seed.
