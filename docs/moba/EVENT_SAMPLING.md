# Event-sampled training for the MOBA world model (design)

**Status:** design, 2026-09-30. No sampling code yet.
**Why:** world-model training samples windows almost uniformly, so it is dominated by
walking. The decisive events the dream gets wrong are a tiny share of what it trains on:
- tower deaths: 35% precision in a 128-tick dream (E15);
- dive outcomes: at chance (E21–E25);
- deaths, as the dream reward sees them: 1.8× too many (E14);
- XP: over-paid for off-expert play (E8, E11).

This doc lists the key events for **both sides**, how each is detected from recorded
states, how rare it is, and how training should sample it: from real windows *and* from
simulator-mined variants, the way the tower dives were mined (E20).

## 1. How rare the key events are

Measured on 130 expert games, 339,188 ticks: mix1k 0–99 plus held-out 900–929
(`results/moba/data/event_rates_expert130.json`).
- **"% of ticks"** = share of ticks where the event happens for that side at least once.
- **Random-play and dyna2 games** will differ (for example 4.7 and 2.2 tower deaths per game
  vs 9.1 here; `tower_counts.json`). The index in § 3 measures every corpus.

| # | event (per side) | detected from the state (tick t → t+1) | radiant: per game / % ticks | dire: per game / % ticks | known dream error |
|---|---|---|---|---|---|
| E1 | **game won** | an ancient (entity 204 / 205) goes `alive` 1 → 0 | 1.0 / 0.04% (over both sides) | | wins never happen in 128-tick dream episodes |
| E2 | **own tower destroyed** | tower `alive` 1 → 0, tower's team = side | 1.0 / 0.04% | 7.9 / 0.30% | 35% precision / 70% recall at 128 ticks |
| E3 | **own tower damaged** | tower health drops while alive | 66 / 2.5% | 350 / 12.7% | dream damage lags (13% vs 2% health at death) |
| E4 | **hero attacks tower** | hero's `target_pid` is an enemy tower | 104 / 3.8% | 14 / 0.5% | attacker info blanked in the dream (rule R9); derived targets 64% right for towers |
| E5 | **hero death** | the game's own death reward component ≠ 0 | 15.4 / 0.58% | 30.7 / 1.17% | the dream reward's death detector fires 1.8× |
| E6 | **hero kill** | a hero death where an enemy hero's `target_pid` was the victim that tick | ≤ E5 (not counted yet) | | not measured |
| E7 | **hero attacks hero** (fights) | hero `target_pid` is an enemy hero | 403 / 13.8% | 201 / 7.1% | — |
| E8 | **ability cast** (Q/W/E) | that ability's cooldown jumps up | 159 / 6.0% | 171 / 6.4% | E6b: effects learned on forks only |
| E9 | **hero hit** | `is_hit` = 1 on a hero | 750 / 25% | 541 / 17% | — |
| E10 | **level up** | hero level goes up | 27 / 1.0% | 20 / 0.76% | level comes from XP by rule (R4); XP over-paid off-expert |
| E11 | **hero attacks creep** (farming) | hero `target_pid` is a creep | 262 / 9.2% | 180 / 6.3% | — |
| E12 | **own creep dies** | creep `alive` 1 → 0, creep's team at t = side | 248 / 9.0% | 255 / 9.2% | — |
| E13 | **hero attacks neutral** / neutral dies | `target_pid` is a neutral / neutral `alive` 1 → 0 | 77 / 2.9% | 17 / 0.64% | — |
| E14 | **whole team idle** | every hero of the side moves < 0.01 cell | 0.8% | 0.9% | — |
| — | **hero moving** (the default) | 0.01 < move ≤ 12 cells | 99.2% of ticks | 99.0% | dominates what training sees |

**How to read it:**
- **Decisive events are 0.04–1.2% of ticks:** wins, tower deaths, hero deaths, level-ups.
- **Combat is common but local:** hits 17–25%, fights 7–14%.
- **Walking is in 99% of ticks.** A uniformly sampled 16-tick window almost never contains a
  tower death (about 1 in 20) or a win (about 1 in 160).
- **The sides are very asymmetric in expert data:** radiant (the net) attacks towers 7× more
  often, and dire loses 8× more towers. Sampling must balance **by side**, or the model learns
  radiant attacks and dire defends.

## 2. Two sources of event data

### A. Real windows centred on events (cheap, exact, limited)
- Every training corpus (mix1k, mixrand1k, dyna2; 2,560 games, 6.77M transitions) gets an
  **event index**: one pass over the full states, recording (game, tick, event type, side,
  entities) for E1–E14.
- The rollout lanes then **start** their open-loop spans at a chosen offset *before* an event.
  This generalises `--tower-event-p` (E15) from towers to all event types.
- **Limit:** every event exists once. Rare events (wins: 1,452 in the whole dataset) repeat
  heavily.

### B. Simulator-mined variants (for rare, decisive events)
- The E20 machinery: restore the real state before the event, vary it (random stream, small
  state nudges, random action stretches), simulate with the real game, and keep both outcomes
  (it happens / a near-miss where it doesn't).
- **Tower deaths are done:** 2.01M slices, 1.01M died / 1.00M survived.
- **To add with the same generator:**
  - hero deaths (E5), both sides;
  - game wins (E1): about 1,300 events;
  - team fights: several heroes fighting within a few ticks.
- **Near-misses matter:** they are what teaches the dream *not* to invent the event, which is
  its main failure on towers.

## 3. Sampling plan

1. **Event index** (`scripts/moba/index_events.py`): on a CPU pod over the S3 corpora.
   - Output: `s3://puffertank/moba/event_index/<corpus>.npz` (type, side, game, tick, entity)
     plus a per-type summary.
   - Only training games; held-out games get their own index for validation.
2. **A sampling mix per update,** configurable. Proposed first version:

| share of rollout lanes | what they start near |
|---|---|
| 40% | uniform, as today (keeps walking and ordinary play calibrated) |
| 30% | real event windows: **equal share per event type**, each split 50/50 radiant / dire |
| 30% | mined variants: equal share of tower deaths, hero deaths and wins, each 50/50 happened / near-miss |

3. **Window alignment:** start 8–96 ticks *before* the event (random within that range), so
   the model has to predict the build-up, not just the moment.
4. **Loss weighting:** none. Every lane counts once, as in normal training, so the gain comes
   only from *what* is sampled, not from reweighting (the author's rule for E21).
5. **Coverage logging:** every log line shows, per event type, how many events and variants
   have been seen, and an epoch per source (as E21 did for scenarios).

## 4. Validation per event type (held-out games only)

For each event type, both sides, on a fixed held-out set:
- **Does the dream produce the event when it really happens,** and not when it doesn't?
  Precision and recall in a 128-tick open-loop rollout from real states. This is the tower
  audit (E15), generalised.
- **For mined events:** outcome accuracy on held-out variants, the way the tower dives are
  scored.
- **The normal held-out horizon losses** (expert and random), so the event mix doesn't cost
  ordinary play.

## 5. Lessons this plan builds on

- **Test at training scale first.** Small-scale ablations (E15/E17) mis-predicted the
  full-scale result (E19). A quick A/B uses the full configuration for a few hundred updates.
- **A 14% share of scenario lanes did nothing** (E21), and scenario-only fine-tunes damaged
  normal play (E25). So keep the uniform lanes, and make the event share large.
- **Towers already drift under any continued training** (E18, E21–E25). Per-event validation
  every 250 updates, and choosing checkpoints by it, is part of the plan, not an afterthought.

## 6. Decisions for the author

- **The mix:** 40 / 30 / 30 above, or different?
- **Which events get mined variants** beyond tower deaths: hero deaths? wins? team fights?
- **Is "both sides" 50/50 per event type right,** or weighted toward what the policy plays?
- **A first A/B:** the event-sampled mix vs uniform, E12 @4000, full data, about 500 updates,
  judged on the § 4 metrics.
