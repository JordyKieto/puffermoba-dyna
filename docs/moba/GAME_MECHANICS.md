# Ocean MOBA: how heroes act

What each hero action does, straight from the simulator
(`third_party/pufferlib/ocean/moba/moba.h`: `step_players`, the `skill_*`
functions, `attack`, `basic_attack`, `spawn_player`). One **tick** is one
simulator step. Everything below applies to all 10 heroes: in our data both
teams act through the same action interface.

## What one hero chooses every tick

Six choices, made together every tick:

| head | options | what it is |
|---|---|---|
| move y | 7 | vertical stick position: −1, −⅔, −⅓, 0, +⅓, +⅔, +1 |
| move x | 7 | horizontal stick position, same 7 levels |
| target mode | 3 | which enemies count as targets: 0 or 1 = any enemy; 2 = only heroes and towers |
| Q | 2 | try to cast Q (yes / no) |
| W | 2 | try to cast W (yes / no) |
| E | 2 | try to cast E (yes / no) |

There is no mouse click and no explicit target selection. The target is
chosen automatically (see Targeting).

## Order of events in one tick, per hero

1. **Regenerate:** +2 health and +2 mana, capped at the maximum.
2. **Timers tick down by 1:** stun, slow, Q, W, E and attack cooldowns.
   When the slow timer reaches 0, speed returns to normal.
3. **If stunned** (stun timer > 0), the hero does nothing else this tick: no
   move, no attack, no ability. Cooldowns still ran in step 2.
4. **Targeting:** find enemies in the 11×11 window around the hero (5 cells
   each way), filtered by target mode. The target is the nearest one (by
   L1 distance), or none.
5. **At most one of these, in this priority order:**
   1. **Q**, if pressed, Q is off cooldown, and Q's own condition holds
      (below);
   2. otherwise **W**, same checks;
   3. otherwise **E**, same checks;
   4. otherwise a **basic attack**, if there is a target and the attack
      cooldown is ready.
6. **Move:** always, whatever happened in step 5.

## Walking

- **Input:** the two move heads work like an analog stick, a direction
  vector with 7 levels per axis (−1 … +1 in thirds). Diagonals are
  normalized, so no direction is faster than 1.
- **Step:** each tick the hero tries to move by `speed × stick`. Speed is 1
  normally, 2 under the assassin's E buff, and 0.5 while slowed.
- **Blocked moves:** the move only happens if the destination cell is free
  (not a wall, not another unit). Otherwise the hero stays put.
- **No cooldown, no cost.** A stunned hero can't move.

## Attacking (basic attack)

- **Automatic.** There is no "attack" button. Whenever a hero has a target
  (nearest enemy in its 11×11 window) and hasn't cast an ability this tick,
  it basic-attacks.
- **Can't be turned off.** The target-mode head only chooses *which*
  enemies count as targets (anything vs heroes and towers only). To avoid
  attacking, a hero has to walk away.
- **Damage:** 50 + per-level gain (10 per level for most classes, 15 tank,
  25 carry).
- **Cooldown:** 8 ticks. The cooldown restarts on every attempt, even when
  the hit itself fails, e.g. a target more than 12 cells away (L1).
- **Kills give XP,** shared with allied heroes within 7 cells.

## Abilities: Q, W, E

- **Q, W and E differ by hero class.** Each team has one of each class, in
  slots 0–4 / 5–9: support, assassin, burst, tank, carry.
- **Each has its own cooldown timer.** Pressing it does nothing while the
  cooldown is running.
- **A cast only happens when every condition holds.** Then its mana is spent
  and its cooldown starts. If a condition fails, nothing happens: no mana
  spent, no cooldown. The assassin's W is the one exception, noted in the
  table.
- **"Needs a target"** means the automatic target from step 4 must exist.
- **Area abilities** hit enemies within the given radius **around the
  caster**, not around the target. The target only has to exist.

`L` is the caster's level.

| class | key | effect | mana | cooldown | valid only if |
|---|---|---|---|---|---|
| **support** | Q | **hook**: pulls the target 1.5 + 0.1L cells toward the caster | 100 | 15 | a target exists |
| | W | **area heal**: heals allies within 5 cells (caster included) by 350 + 50L | 100 | 50 | enough mana (always has someone to heal: itself) |
| | E | **stun**: 50 + 20L damage, target stunned 15 + 0.5L ticks | 75 | 60 | a target exists and the hit lands |
| **assassin** | Q | **clear**: 100 + 20L damage to enemies within 3 cells | 100 | 40 | the target is a **creep or neutral** (not a hero or tower), and at least one enemy is within 3 |
| | W | **blink strike**: teleports next to the target, then 250 + 50L damage | 150 | 60 | a target exists and there is a free cell next to it. Mana is spent on the teleport even if the hit then fails; the cooldown only starts if it lands |
| | E | **speed boost**: double speed for 25 ticks | 100 | 100 | enough mana (no target needed) |
| **burst** | Q | **nuke**: 250 + 40L damage to the target | 200 | 70 | a target exists and the hit lands |
| | W | **blast**: 100 + 40L damage to enemies within 2 cells | 100 | 40 | a target exists and at least one enemy within 2 |
| | E | **area stun**: stuns enemies within 2 cells for 10 + 0.5L ticks, no damage | 75 | 50 | a target exists and at least one enemy within 2 |
| **tank** | Q | **burn**: 25 + 2L damage to enemies within 2 cells | 5 | **none**: castable every tick | at least one enemy within 2 |
| | W | **self heal**: heals itself 400 + 125L | 100 | 70 | enough mana |
| | E | **engage**: jumps next to the target, pulls enemies within 4 toward itself (2 + 0.1L cells) | 50 | 40 | a target exists and there is a free cell next to it |
| **carry** | Q | **knockback + slow**: pushes the target away 3 + 0.1L cells, up to 3 times (25 mana each); slows it to half speed for 15 ticks | 25 per push | 40 | a target exists and a push succeeds |
| | W | **crippling shot**: 50 + 20L damage, slows the target to half speed for 20 + L ticks | 150 | 40 | a target exists and the hit lands |
| | E | **blast**: 100 + 20L damage to enemies within 2 cells | 100 | 40 | a target exists and at least one enemy within 2 |

"The hit lands" means the target is alive, an enemy, and within 12 cells.

## Mana

- **Maximum:** a class base + a per-level gain.

  | class | base max mana | gain per level |
  |---|---|---|
  | support | 250 | +50 |
  | assassin | 300 | +65 |
  | burst | 300 | +90 |
  | tank | 200 | +50 |
  | carry | 250 | +50 |

  Levelling up raises the maximum, not the current mana.
- **Goes down** only when an ability is cast, by that ability's cost.
- **Goes up** by +2 every tick (the regeneration in step 1), up to the
  maximum, and resets to full on respawn.
- **No timer of its own.** Only the abilities have cooldowns. Regeneration
  continues while stunned.

## Health

- **Maximum:** a class base + a per-level gain.

  | class | base max health | gain per level |
  |---|---|---|
  | support | 500 | +100 |
  | assassin | 400 | +100 |
  | burst | 400 | +75 |
  | tank | 700 | +150 |
  | carry | 300 | +50 |

  Levelling up raises the maximum, not the current health.
- **Goes down** when hit by:

  | source | damage |
  |---|---|
  | enemy hero basic attack | that hero's damage |
  | enemy hero ability | see the ability table |
  | tower | 110–175 per hit, every 5 ticks |
  | creep | 22 per hit, every 5 ticks |
  | neutral monster (attacks any hero within 3 cells of it) | 22 per hit, every 5 ticks |

- **Goes up** by:
  - +2 every tick of regeneration, up to the maximum;
  - the support's W (area heal) and the tank's W (self heal);
  - resetting to full on respawn.
- **No timer of its own.**
- **At 0 the hero dies:**
  - the killer's team gets XP;
  - the victim's death penalty is applied (only when a hero kills it; see
    DATA.md);
  - **the hero respawns in the same tick** near its team's base (random
    ±7 cells), with full health and mana, all timers cleared and the same
    level.

  So in the data a death looks like a jump back to base plus a health reset,
  never a dead hero.

## Status effects

- **Stun** (support E, burst E): the hero does nothing but regenerate and
  let cooldowns run, for the stun duration. It freezes creeps and neutrals
  too; towers keep firing.
- **Slow** (carry Q and W): half speed for the duration.
- **Speed boost** (assassin E): double speed for 25 ticks.

## What this means for the world model

- **Walking is a direct function of the two move heads,** blocked only by
  occupied cells. It should be near-exact.
- **Basic attacks are automatic.** Whether one happens depends on the state
  (is there a target, is the cooldown ready) and on whether an ability took
  priority. The target-mode head only filters targets.
- **A pressed Q/W/E does nothing unless its conditions hold.** So "Q
  pressed" alone doesn't determine a Q cooldown reset: the model also has to
  check cooldown, mana, target existence and class-specific conditions.
  That's part of why the ability rows are harder than walking.
- **At most one of Q, W, E and the basic attack happens per tick,** in that
  priority order.
- **Mana and cooldowns change deterministically:** −cost and a cooldown
  reset on a cast; otherwise +2 mana and −1 cooldown every tick. That makes
  them the most learnable of the red rows.
