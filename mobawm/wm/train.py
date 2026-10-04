"""Train and evaluate the token world model.

  python3 -m mobawm.wm.train --data data/moba/mix100/dense --train-games 0-90 \
      --held-games 91-99 --minutes 20 --out runs/token_wm/<name>

The whole dataset is loaded once into device tensors (mobawm.wm.data.Store)
and windows are sampled on the device. Splits are by game (seed), never by
row. Progress is printed every --log-every iterations and appended to
OUT/log.jsonl (and wandb with --wandb); the final report is OUT/report.json.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import shards as S
from .data import (F, KIND, N_KIND, REG_FIELDS, REG_IDX, Featurizer, Stats, Store, apply_prediction,
                   fit_stats, load_store, rebuild_kill_edges, rebuild_kill_edges_store, regen_part, wave_respawns)
from .model import TokenWM



def parse_games(spec: str) -> set[int]:
    """'0-25,40,50-60' -> set of game indices."""
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def featurize_window(fz, ent, tick, cidx):
    """(B,K+1,...) raw -> list of K+1 feature dicts."""
    B, T = ent.shape[:2]
    f = fz(ent.flatten(0, 1), tick.flatten(), cidx.flatten())
    return [{k: v.view(B, T, *v.shape[1:])[:, t] for k, v in f.items()} for t in range(T)]


KIND_NAMES = ("hero", "creep", "neutral", "tower")
# Delta fields grouped by what they mean in the game. Losses and the skill
# metrics are reported per effect, so "does it understand abilities" is a
# curve, not a question for the end-of-run report.
EFFECTS = {
    "movement": ("x", "y", "waypoint"),
    "abilities": ("q_timer", "w_timer", "e_timer", "mana"),
    "attack": ("basic_attack_timer",),
    "health": ("health",),
    "status": ("stun_timer", "move_timer", "move_modifier"),
    "xp": ("xp",),
}
EFFECT_OF = {f: e for e, fs in EFFECTS.items() for f in fs}
# The target scoreboard's rows (docs/moba/EXPERIMENTS.md), for hero skill metrics:
# abilities are split per ability so Q, W, E and mana each get their own curve.
HERO_ROWS = {"walking": ("x", "y"), "attack": ("basic_attack_timer",), "q": ("q_timer",),
             "w": ("w_timer",), "e": ("e_timer",), "mana": ("mana",), "health": ("health",),
             "status": ("stun_timer", "move_timer", "move_modifier"), "xp": ("xp",)}
HERO_ROW_OF = {f: r for r, fs in HERO_ROWS.items() for f in fs}
# Semantic loss: per-effect weights (--effect-weights). 1.0 = plain sum.
# The adopted baseline gates every delta field except movement (EXPERIMENTS.md).
BASELINE_GATE_FIELDS = ("q_timer", "w_timer", "e_timer", "mana", "health", "basic_attack_timer",
                        "stun_timer", "move_timer", "move_modifier", "xp")
EFFECT_WEIGHT = {}
GATE_LOSS_WEIGHT = 1.0   # weight on the change gate's yes/no (BCE) loss
WALK_ONLY = False        # diagnostic: train on hero x/y changes only
HERO_WALK_WEIGHT = 1.0   # weight on the hero x/y loss terms (walking)
HERO_HEALTH_WEIGHT = 1.0 # weight on the hero health loss term
ONLY_HERO_FIELDS = ()    # diagnostic: train on these hero fields' loss terms only (e.g. ("health",))
# gated fields' value loss averaged over "changed" entries only, or over "all"
# supervised entries (= scaled by the field's change rate, as without the gate)
GATE_VALUE_NORM = "changed"
BF16 = False


def _wmean(x, m, w):
    """Mean of x over mask m (both (B, ...)), each batch row weighted by w (B,) or unweighted."""
    mf = m.float() if w is None else m.float() * w.view(-1, *([1] * (m.dim() - 1)))
    return (x * mf).sum() / mf.sum().clamp(min=1e-6)


# Tower fixes (E15): tower health (and tower deaths) loss weight; a consistency loss tying a tower's
# predicted death to its own predicted next health ("dead <=> health ~ 0"); event-balanced lanes
# (at each span restart, with probability TOWER_EVENT_P jump to a real state <= 64 ticks before a
# tower loses health or falls).
TOWER_HEALTH_WEIGHT = 1.0
TOWER_CONSIST = 0.0
TOWER_EVENT_P = 0.0
TARGET_WEIGHT = 1.0   # loss weight of the attack edges (who attacks whom, --target-weight)
TARGET_DECAY = False  # --target-decay: the edge loss gets the same rollout-depth step weights as every other loss
LIST_WEIGHT = 1.0     # --list-weight: loss weight of next-tick stored-list membership (LIST_STATE)
KILL_EDGES = False    # --kill-edges: rebuild the killing-blow edges the game doesn't record (data.rebuild_kill_edges)
# Stratified lanes (docs/moba/EVENT_SAMPLING.md): {stratum: weight}; "uniform" = a uniformly random row,
# any other stratum = an event of mobawm.wm.events (50/50 per side), started 8-96 ticks before it.
STRATA = None


def parse_strata(spec):
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = float(v)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items() if v > 0}


def list_txt(p):
    """Stored-list metrics for a log line ('' without LIST_STATE)."""
    if "metric/list_exact" not in p:
        return ""
    g = lambda k: 100 * float(p[f"metric/{k}"])
    return f"  | lists: exact {g('list_exact'):.1f}%  changes rec {g('list_chg_rec'):.0f}% prec {g('list_chg_prec'):.0f}%"


def edge_txt(p):
    """Attack-edge metrics for a log line ('' when the model has no target head)."""
    if "metric/fires_prec" not in p:
        return ""
    g = lambda k: 100 * float(p[f"metric/{k}"])
    return (f"  | edges: fires prec {g('fires_prec'):.0f}% rec {g('fires_rec'):.0f}%  right unit {g('who_given_fires'):.0f}%"
            f"  exact {g('attack_acc'):.0f}%")


def step_loss(pred, e0, e1, stats, kind, tick, cidx, w=None):
    """Total loss and its parts, per entity kind and per head.

    parts["<kind>/<effect>"]  Huber on scaled deltas of that effect's fields (EFFECTS),
                              each (kind, field) averaged separately then summed
    parts["<kind>/deaths"]    BCE on "alive flips this tick" (creep, neutral, tower)
    parts["<kind>/got_hit"]   BCE on next is_hit
    parts["creep/lane"], parts["creep/team"]  CE on next value
    w: optional (B,) per-sample weights (normalized to mean 1 here); None = plain means
    """
    sw = None if w is None else w / w.mean().clamp(min=1e-12)   # per-sample weights, mean 1
    wv = 1.0 if sw is None else sw.view(-1, 1, 1)
    dev = e0.device
    alive0, alive1 = e0[..., F["alive"]] > 0, e1[..., F["alive"]] > 0
    scale = stats.reg_scale.to(dev)[kind]
    delta = e1[..., REG_IDX] - e0[..., REG_IDX]
    if stats.regen_rule:
        delta = delta - regen_part(e0)            # the model predicts what regen doesn't explain
    target = delta / scale
    gated = "change" in pred
    if gated:
        # copy by default: BCE on "did this field change" everywhere it is
        # supervised, value loss only where it really changed
        changed = target != 0
        gate_err = Fn.binary_cross_entropy_with_logits(pred["change"], changed.float(), reduction="none")
        err = Fn.huber_loss(pred["value"], target, reduction="none", delta=1.0)
    else:
        err = Fn.huber_loss(pred["reg"], target, reduction="none", delta=1.0)
    # units respawned in place by a wave are applied by rule, not regressed (a teleport to spawn)
    wc, _, wn = wave_respawns(tick, cidx)
    rule = torch.zeros_like(alive0)
    rule[:, 10:110] = wc
    rule[:, 110:182] = wn.unsqueeze(-1)
    mask = (alive0 & alive1 & ~rule).unsqueeze(-1) & stats.reg_mask.to(dev)[kind]
    can_flip = stats.alive_mask.to(dev)
    flip = (alive0 != alive1).float()
    alive_bce = Fn.binary_cross_entropy_with_logits(pred["alive_flip"], flip, reduction="none")
    hit_bce = Fn.binary_cross_entropy_with_logits(pred["is_hit"], e1[..., F["is_hit"]], reduction="none")
    if WALK_ONLY:
        # hero x, y only (fields 0, 1), same Huber on scaled deltas, same masks
        hs = kind == 0
        m = mask[:, hs, 0:2].float()
        e = (pred["value"] if gated else pred["reg"])[:, hs, 0:2]
        walk = Fn.huber_loss(e, target[:, hs, 0:2], reduction="none", delta=1.0)
        l = ((walk * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)).sum()
        return l, {"hero/walking": l}
    parts = {}
    for k, name in enumerate(KIND_NAMES):
        sel = kind == k
        m, e = mask[:, sel].float() * wv, err[:, sel]         # weighted mask (== bool mask when w is None)
        if gated:
            mc = (mask[:, sel] & changed[:, sel]).float() * wv
            value_den = (mc if GATE_VALUE_NORM == "changed" else m).sum((0, 1)).clamp(min=1)
            gated_f = GATE_LOSS_WEIGHT * (gate_err[:, sel] * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1) \
                + (e * mc).sum((0, 1)) / value_den
            plain_f = (e * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
            per_field = torch.where(pred["gate_mask"], gated_f, plain_f)
        else:
            per_field = (e * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
        for j, fname in enumerate(REG_FIELDS):
            if stats.reg_mask[k, j]:
                eff = EFFECT_OF[fname]
                key = f"{name}/{eff}"
                w = EFFECT_WEIGHT.get(eff, 1.0) * (HERO_WALK_WEIGHT if k == 0 and fname in ("x", "y") else 1.0) \
                    * (HERO_HEALTH_WEIGHT if k == 0 and fname == "health" else 1.0) \
                    * (TOWER_HEALTH_WEIGHT if k == 3 and fname == "health" else 1.0)
                parts[key] = parts.get(key, 0.0) + w * per_field[j]
                if k == 0 and fname in ONLY_HERO_FIELDS:
                    parts[f"only/hero/{fname}"] = per_field[j]
        if can_flip[k]:
            parts[f"{name}/deaths"] = EFFECT_WEIGHT.get("deaths", 1.0) * (TOWER_HEALTH_WEIGHT if k == 3 else 1.0) * \
                _wmean(alive_bce[:, sel], torch.ones_like(alive_bce[:, sel], dtype=torch.bool), sw)
        a1 = alive1[:, sel]
        if a1.any():
            parts[f"{name}/got_hit"] = _wmean(hit_bce[:, sel], a1, sw)
    if "target_logits" in pred:
        # who each unit attacks during t -> t+1 (recorded target_pid of the next state); 206 = nobody
        tgt = e1[..., F["target_pid"]].long()
        tgt = torch.where(tgt < 0, torch.full_like(tgt, S.N_ENTITIES), tgt)
        m = alive0 & alive1 & ~rule
        ce = Fn.cross_entropy(pred["target_logits"].transpose(1, 2), tgt, reduction="none")
        # ~92% of units attack nobody: weight the units that DO attack x10, so "nobody everywhere" can't win
        cw = torch.where(tgt < S.N_ENTITIES, torch.full_like(ce, 10.0), torch.ones_like(ce)) * m
        if TARGET_DECAY and sw is not None:
            cw = cw * sw.view(-1, 1)
        parts["attack/targets"] = TARGET_WEIGHT * (ce * cw).sum() / cw.sum().clamp(min=1)
        with torch.no_grad():
            hit = (pred["target_logits"].argmax(-1) == tgt) & m
            attacking = m & (tgt < S.N_ENTITIES)
            parts["metric/attack_acc"] = (hit & attacking).sum().float() / attacking.sum().clamp(min=1)   # not in the loss
            # the two halves: does it fire (top choice is a unit, not "nobody"), and is it the right unit
            fires = m & (pred["target_logits"].argmax(-1) < S.N_ENTITIES)
            parts["metric/fires_prec"] = (fires & attacking).sum().float() / fires.sum().clamp(min=1)
            parts["metric/fires_rec"] = (fires & attacking).sum().float() / attacking.sum().clamp(min=1)
            who = pred["target_logits"][..., :S.N_ENTITIES].argmax(-1) == tgt
            parts["metric/who_given_fires"] = (who & attacking).sum().float() / attacking.sum().clamp(min=1)
    if "list_logits" in pred:
        # next-tick stored target lists (creeps, towers): BCE per (owner, member); bits that change weigh x10
        from .data import OWNERS, unpack_lists
        own = torch.as_tensor(OWNERS, device=dev)
        cur, nxt = unpack_lists(e0), unpack_lists(e1)
        mo = (alive0 & alive1)[:, own].float()
        if sw is not None:
            mo = mo * sw.view(-1, 1)
        chg = cur != nxt
        bce = Fn.binary_cross_entropy_with_logits(pred["list_logits"], nxt.float(), reduction="none")
        bw = (1.0 + 9.0 * chg.float()) * mo.unsqueeze(-1)
        parts["list/members"] = LIST_WEIGHT * (bce * bw).sum() / bw.sum().clamp(min=1)
        with torch.no_grad():
            pb = pred["list_logits"] > 0
            mb = mo.unsqueeze(-1) > 0
            pch = pb != cur
            parts["metric/list_chg_rec"] = ((pb == nxt) & chg & mb).sum().float() / (chg & mb).sum().clamp(min=1)
            parts["metric/list_chg_prec"] = ((pb == nxt) & pch & mb).sum().float() / (pch & mb).sum().clamp(min=1)
            parts["metric/list_exact"] = (((pb == nxt).all(-1)) & (mo > 0)).sum().float() / (mo > 0).sum().clamp(min=1)
    if TOWER_CONSIST:
        # towers alive now: predicted P(dies this tick) must agree with the predicted next health
        tw = kind == 3
        jh = REG_FIELDS.index("health")
        v = (pred["value"] if gated else pred["reg"])[:, tw, jh]
        if gated:
            v = torch.sigmoid(pred["change"][:, tw, jh]) * v
        nh = e0[:, tw, F["health"]] + v * scale[tw, jh]
        frac = (nh / e0[:, tw, F["max_health"]].clamp(min=1)).clamp(0, 1)
        logit = pred["alive_flip"][:, tw]
        dead_by_health = (frac < 0.01).float().detach()
        cons = Fn.binary_cross_entropy_with_logits(logit, dead_by_health, reduction="none") \
            + torch.sigmoid(logit).detach() * frac                     # "you said dead: health should be ~0"
        parts["tower/consistency"] = TOWER_CONSIST * _wmean(cons, alive0[:, tw], sw)
    c = slice(10, 110)
    ca = alive1[:, c]
    if ca.any():
        lane = Fn.cross_entropy(pred["lane"].transpose(1, 2), e1[:, c, F["lane"]].long(), reduction="none")
        team = Fn.cross_entropy(pred["team"].transpose(1, 2), e1[:, c, F["team"]].long().clamp(0, 2), reduction="none")
        parts["creep/lane"] = _wmean(lane, ca, sw)
        parts["creep/team"] = _wmean(team, ca, sw)
    if ONLY_HERO_FIELDS:
        only = {k: v for k, v in parts.items() if k.startswith("only/")}
        return sum(only.values()), only
    return sum(v for k, v in parts.items() if not k.startswith("metric/")), parts


def run_window(model, fz, batch, stats, kind, burn, dev, loss=True, shuffle_actions=False):
    ent, act, tick, cidx = batch
    if shuffle_actions:
        act = act[torch.randperm(act.shape[0], device=dev)]
    feats = featurize_window(fz, ent, tick, cidx)
    B, K = act.shape[:2]
    h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
    total, parts, preds = 0.0, {}, []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and dev.type == "cuda") else nullcontext()
    for t in range(K):
        with amp:
            pred, h = model.step(feats[t], act[:, t], h)
        pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
        h = h.float()
        if t >= burn:
            if loss:
                l, p = step_loss(pred, ent[:, t], ent[:, t + 1], stats, kind, tick[:, t], cidx[:, t])
                total = total + l
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v
            preds.append((t, pred))
    n = K - burn
    if loss:
        return total / n, {k: v / n for k, v in parts.items()}, preds, ent
    return None, None, preds, ent


def summarize_parts(parts):
    """Per-kind totals plus every head, as floats."""
    out = {k: float(v) for k, v in parts.items()}
    for name in KIND_NAMES:
        out[f"{name}/total"] = sum(v for k, v in out.items() if k.startswith(name + "/") and k != f"{name}/total")
    return out


@torch.no_grad()
def validate(model, fz, store, stats, kind, dev, args, seed=123):
    """Loss on the SAME held-out windows every time (fixed seed), averaged over batches."""
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    model.eval()
    tot, agg = 0.0, {}
    for _ in range(args.val_batches):
        batch = store.sample(args.batch, args.window, gen)
        l, parts, _, _ = run_window(model, fz, batch, stats, kind, args.burn, dev)
        tot += float(l)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    model.train()
    n = args.val_batches
    return tot / n, summarize_parts({k: v / n for k, v in agg.items()})


@torch.no_grad()
def skill_eval(model, fz, store, stats, kind, dev, args, seed=123):
    """The two questions that tell whether an effect is understood, per effect,
    on the same fixed held-out windows as validate():

      better_than_copy_pct  = 100 * (1 - model error / persistence error)
                              > 0: better than copying the last frame
      action_dependence_pct = 100 * (error with wrong actions / error with real actions - 1)
                              how much the prediction depends on the actions actually taken:
                              the real actions are swapped for another window's; 0 = ignores them

    Errors are on RMS-scaled deltas (unitless), summed over the effect's
    fields; only entities alive at t and t+1, excluding wave respawns. Deaths:
    recall and false positives of alive flips."""
    names = KIND_NAMES
    err = {}          # (kind, effect) -> [model, persist, shuffled]
    deaths = [0, 0, 0]  # tp, fn, fp
    model.eval()
    for shuffle in (False, True):
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        for _ in range(args.val_batches):
            batch = store.sample(args.batch, args.window, gen)
            _, _, preds, ent = run_window(model, fz, batch, stats, kind, args.burn, dev, loss=False,
                                          shuffle_actions=shuffle)
            scale = stats.reg_scale.to(dev)[kind]
            for t, p in preds:
                e0, e1 = ent[:, t], ent[:, t + 1]
                both = (e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] > 0)
                wc, _, wn = wave_respawns(batch[2][:, t], batch[3][:, t])
                both[:, 10:110] &= ~wc
                both[:, 110:182] &= ~wn.unsqueeze(-1)
                true_d = (e1[..., REG_IDX] - e0[..., REG_IDX]) / scale
                pred_d = p["reg"] + (regen_part(e0) / scale if stats.regen_rule else 0)
                ae = (pred_d - true_d).abs()
                pe = true_d.abs()
                for k, kn in enumerate(names):
                    m = both[:, kind == k]
                    for j, fname in enumerate(REG_FIELDS):
                        if not stats.reg_mask[k, j]:
                            continue
                        group = HERO_ROW_OF[fname] if kn == "hero" else EFFECT_OF[fname]
                        a = err.setdefault((kn, group), [0.0, 0.0, 0.0])
                        if shuffle:
                            a[2] += ae[:, kind == k, j][m].sum().item()
                        else:
                            a[0] += ae[:, kind == k, j][m].sum().item()
                            a[1] += pe[:, kind == k, j][m].sum().item()
                if not shuffle:
                    can = stats.alive_mask.to(dev)[kind].expand_as(both)
                    flip = ((e0[..., F["alive"]] > 0) != (e1[..., F["alive"]] > 0)) & can
                    pf = (torch.sigmoid(p["alive_flip"]) > 0.5) & can
                    deaths[0] += int((flip & pf).sum()); deaths[1] += int((flip & ~pf).sum())
                    deaths[2] += int((~flip & pf).sum())
    model.train()
    out = {}
    for (kn, ef), (m, pz, sh) in err.items():
        if pz <= 0:
            continue
        out[f"{kn}/{ef}/better_than_copy_pct"] = 100 * (1 - m / pz)
        out[f"{kn}/{ef}/action_dependence_pct"] = 100 * (sh / max(m, 1e-9) - 1)
    out["deaths/recall_pct"] = 100 * deaths[0] / max(deaths[0] + deaths[1], 1)
    out["deaths/false_pos_per_true"] = deaths[2] / max(deaths[0] + deaths[1], 1)
    return out


EVENT_ROWS = ("attack", "q", "w", "e", "mana", "health", "death")


@torch.no_grad()
def event_eval(model, fz, store, stats, kind, dev, args, seed=123):
    """The scoreboard (docs/moba/EXPERIMENTS.md): per hero and tick, did an
    event happen, and did the model predict it. Held-out, one step ahead.

    Events (true side from the real next state; predicted side from the model's
    predicted change, in game units):
      attack / q / w / e : that cooldown resets, i.e. jumps up    (predicted: change > +0.5)
      mana               : mana spent, drop > 3 beyond the +2 regen (costs are >= 5)
      health             : damage taken, drop > 3 beyond regen (hits are >= 22), not a death
      death              : the hero teleports > 12 cells (respawn at base)
    For each row: caught % (of real events), false alarms per real event, and
    caught % when the actions are swapped for another window's (if it stays
    equal to caught %, the model is not using the actions for that event).
    Walking: position error vs copying, on non-death ticks.
    The tank's Q has no cooldown, so it never appears as a Q event."""
    cols = {r: REG_FIELDS.index(c) for r, c in (("attack", "basic_attack_timer"), ("q", "q_timer"),
                                                 ("w", "w_timer"), ("e", "e_timer"))}
    ix, iy, ih, im = (REG_FIELDS.index(c) for c in ("x", "y", "health", "mana"))
    tally = {r: [0, 0, 0, 0] for r in EVENT_ROWS}      # real, caught, false alarms, caught with wrong actions
    walk = [0.0, 0.0, 0.0]                              # model err, copy err, model err with wrong actions
    model.eval()
    for shuffle in (False, True):
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        for _ in range(args.val_batches):
            batch = store.sample(args.batch, args.window, gen)
            _, _, preds, ent = run_window(model, fz, batch, stats, kind, args.burn, dev, loss=False,
                                          shuffle_actions=shuffle)
            scale = stats.reg_scale.to(dev)[kind]
            for t, p in preds:
                e0, e1 = ent[:, t, :10], ent[:, t + 1, :10]
                true_d = (e1[..., REG_IDX] - e0[..., REG_IDX])
                regen = regen_part(ent[:, t])[:, :10]
                pred_d = p["reg"][:, :10] * scale[:10] + (regen if stats.regen_rule else 0)
                jump = (true_d[..., ix].abs() + true_d[..., iy].abs())
                died = jump > 12
                pjump = (pred_d[..., ix].abs() + pred_d[..., iy].abs())
                real, pred = {}, {}
                for r, c in cols.items():
                    real[r], pred[r] = (true_d[..., c] > 0) & ~died, (pred_d[..., c] > 0.5) & ~died
                real["mana"] = ((true_d[..., im] - regen[..., im]) < -3) & ~died
                pred["mana"] = ((pred_d[..., im] - regen[..., im]) < -3) & ~died
                real["health"] = ((true_d[..., ih] - regen[..., ih]) < -3) & ~died
                pred["health"] = ((pred_d[..., ih] - regen[..., ih]) < -3) & ~died
                real["death"], pred["death"] = died, pjump > 12
                for r in EVENT_ROWS:
                    tl = tally[r]
                    if shuffle:
                        tl[3] += int((real[r] & pred[r]).sum())
                    else:
                        tl[0] += int(real[r].sum()); tl[1] += int((real[r] & pred[r]).sum())
                        tl[2] += int((~real[r] & pred[r]).sum())
                ok = ~died
                err = (pred_d[..., ix] - true_d[..., ix]).abs() + (pred_d[..., iy] - true_d[..., iy]).abs()
                if shuffle:
                    walk[2] += err[ok].sum().item()
                else:
                    walk[0] += err[ok].sum().item(); walk[1] += jump[ok].sum().item()
    model.train()
    out = {"walking/better_than_copy_pct": 100 * (1 - walk[0] / max(walk[1], 1e-9)),
           "walking/wrong_actions_extra_error_pct": 100 * (walk[2] / max(walk[0], 1e-9) - 1)}
    for r, (n, c, fa, cw) in tally.items():
        out[f"{r}/real_events"] = n
        out[f"{r}/caught_pct"] = 100 * c / max(n, 1)
        out[f"{r}/false_alarms_per_real"] = fa / max(n, 1)
        out[f"{r}/caught_wrong_actions_pct"] = 100 * cw / max(n, 1)
    return out


@torch.no_grad()
def eval_onestep(model, fz, store, stats, kind, dev, gen, n_batches=8, B=16, K=16, burn=4, shuffle=False):
    """Natural-unit MAE of predicted deltas vs persistence (delta 0), per (kind, field)."""
    names = ["hero", "creep", "neutral", "tower"]
    acc = {}
    flips = {"tp": 0, "fn": 0, "fp": 0}
    for _ in range(n_batches):
        batch = store.sample(B, K, gen)
        _, _, preds, ent = run_window(model, fz, batch, stats, kind, burn, dev, loss=False, shuffle_actions=shuffle)
        scale = stats.reg_scale.to(dev)[kind]
        rmask = stats.reg_mask.to(dev)[kind]
        for t, p in preds:
            e0, e1 = ent[:, t], ent[:, t + 1]
            both = (e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] > 0)
            wc, _, wn = wave_respawns(batch[2][:, t], batch[3][:, t])
            both[:, 10:110] &= ~wc
            both[:, 110:182] &= ~wn.unsqueeze(-1)
            true_d = e1[..., REG_IDX] - e0[..., REG_IDX]
            pred_d = p["reg"] * scale + (regen_part(e0) if stats.regen_rule else 0)
            for k in range(N_KIND):
                for j, fname in enumerate(REG_FIELDS):
                    if not stats.reg_mask[k, j]:
                        continue
                    m = both[:, kind == k]
                    td, pd = true_d[:, kind == k, j][m], pred_d[:, kind == k, j][m]
                    ch = td != 0
                    a = acc.setdefault((names[k], fname), [0.0, 0.0, 0, 0.0, 0.0, 0])
                    a[0] += (pd - td).abs().sum().item(); a[1] += td.abs().sum().item(); a[2] += len(td)
                    a[3] += (pd - td)[ch].abs().sum().item(); a[4] += td[ch].abs().sum().item(); a[5] += int(ch.sum())
            can = stats.alive_mask.to(dev)[kind].expand_as(both)
            flip = ((e0[..., F["alive"]] > 0) != (e1[..., F["alive"]] > 0)) & can
            pf = (torch.sigmoid(p["alive_flip"]) > 0.5) & can
            flips["tp"] += int((flip & pf).sum()); flips["fn"] += int((flip & ~pf).sum())
            flips["fp"] += int((~flip & pf).sum())
    out = {}
    for (kn, fn), a in sorted(acc.items()):
        out[f"{kn}.{fn}"] = dict(model_mae=a[0] / max(a[2], 1), persist_mae=a[1] / max(a[2], 1),
                                 changed_model_mae=a[3] / max(a[5], 1), changed_persist_mae=a[4] / max(a[5], 1),
                                 n=a[2], n_changed=a[5])
    out["alive_flip"] = dict(recall=flips["tp"] / max(flips["tp"] + flips["fn"], 1), **flips)
    return out


# Rollout rubric (docs/moba/WM_FEEDBACK_RL.md): "did the hero land inside the circle"
# with a radius that grows with steps since the last real state, and "was the path
# walkable" (speed like the real hero's, no reversals the real hero didn't make).
RUBRIC_R0, RUBRIC_SLOPE, RUBRIC_CAP = 1.0, 0.05, 8.0     # radius (cells) = min(r0 + slope * steps, cap)


def rubric_radius(steps):
    return (RUBRIC_R0 + RUBRIC_SLOPE * steps).clamp(max=RUBRIC_CAP)


class Lanes:
    """Autoregressive training: B lanes, each walking forward through one real game
    with its recorded actions, the model fed its OWN predictions.

    Each lane runs open loop for a span drawn from a Gaussian in log2(ticks) (median
    2**span_mu, clipped to 1..`max_span`),
    then snaps back to the real state (keeping its GRU memory) and draws a new span;
    at the end of a game it jumps to a random row of a random game with a fresh
    memory. So one batch mixes 1-step (teacher forcing) up to whole-game rollouts.
    The loss at every step is the real next state against the lane's current input
    state, so the model learns to steer its own drift back to the truth.
    """

    def __init__(self, store, B, d, gen, max_span, span_mu=6.0, span_sigma=3.0, halflife=0.0, floor=0.0,
                 w_circle=0.0, w_traj=0.0):
        self.st, self.B, self.gen, self.max_span = store, B, gen, max_span
        self.span_mu, self.span_sigma = span_mu, span_sigma
        self.halflife, self.floor = halflife, floor
        self.w_circle, self.w_traj = w_circle, w_traj
        # the store may live in CPU RAM (--store-cpu); lanes, rows and outputs live on `self.dev`
        dev = self.dev = getattr(store, "out_dev", None) or store.device
        self.age = torch.zeros(B, dtype=torch.long, device=dev)   # steps since the last real state
        self._starts1 = store.starts(1).to(dev)
        self.row_end = torch.empty(len(store.ent), dtype=torch.long, device=dev)
        for off, n in store.spans:
            self.row_end[off:off + n + 1] = off + n            # terminal row of each game
        self.row = torch.zeros(B, dtype=torch.long, device=dev)
        self.left = torch.zeros(B, dtype=torch.long, device=dev)
        self.h = torch.zeros(B, S.N_ENTITIES, d, device=dev)
        self.cur = self.ct = self.cc = None
        self.row_start = torch.empty(len(store.ent), dtype=torch.long, device=dev)
        for off, n in store.spans:
            self.row_start[off:off + n + 1] = off
        self.strata = None
        if STRATA:
            from .events import index_store
            t_i = time.time()
            ix = index_store(store, self.row_end)
            self.strata_names = list(STRATA)
            self.strata_w = torch.tensor([STRATA[k] for k in self.strata_names], device=dev)
            self.strata = {k: [r.to(dev) for r in ix[k]] for k in self.strata_names if k != "uniform"}
            self.strata_seen = {k: 0 for k in self.strata_names}
            print("strata " + ", ".join(f"{k} {100 * STRATA[k]:.0f}%" + (f" ({len(ix[k][0])}+{len(ix[k][1])} events)" if k != "uniform" else "")
                                          for k in self.strata_names) + f"  [index {time.time() - t_i:.0f}s]", flush=True)
        self.event_starts = None
        if TOWER_EVENT_P > 0:
            th = store.ent[:, 182:, F["health"]]                      # (rows, 24), wherever the store lives
            ta = store.ent[:, 182:, F["alive"]]
            ev = ((th[1:] < th[:-1] - 1e-3) | (ta[1:] != ta[:-1])).any(-1).to(dev)
            ev &= self.row_end[:-1] > torch.arange(len(ev), device=dev)   # not across a game boundary
            w = torch.zeros(len(store.ent), dtype=torch.bool, device=dev)
            for k in range(65):                                       # rows up to 64 ticks before an event
                w[:len(ev) - k] |= ev[k:]
            valid = torch.zeros_like(w)
            valid[self._starts1] = True
            self.event_starts = torch.nonzero(w & valid).flatten()
            print(f"tower-event lanes: {int(ev.sum())} event rows, {len(self.event_starts)} start rows "
                  f"(jump p={TOWER_EVENT_P})", flush=True)
        self.new_game(torch.ones(B, dtype=torch.bool, device=dev))

    def span(self, n):
        """Open-loop span lengths: Gaussian in log2(ticks), mean span_mu, std span_sigma, clipped to 1..max_span."""
        z = torch.randn(n, device=self.dev, generator=self.gen)
        return torch.exp2(self.span_mu + self.span_sigma * z).round().long().clamp(1, self.max_span)

    def sync(self, m):
        """Lanes in mask m take the real state of their current row."""
        if self.strata is not None and m.any():
            lanes = torch.nonzero(m).flatten()
            pick = torch.multinomial(self.strata_w, len(lanes), replacement=True, generator=self.gen)
            for si, name in enumerate(self.strata_names):
                sel = lanes[pick == si]
                if len(sel) == 0:
                    continue
                self.strata_seen[name] += len(sel)
                if name == "uniform":
                    st = self._starts1
                    self.row[sel] = st[torch.randint(len(st), (len(sel),), device=self.dev, generator=self.gen)]
                else:
                    rad, dire = self.strata[name]
                    side = torch.rand(len(sel), device=self.dev, generator=self.gen) < 0.5
                    for rows, msk in ((rad, side), (dire, ~side)):
                        k = int(msk.sum())
                        if k == 0:
                            continue
                        src = rows if len(rows) else (dire if rows is rad else rad)
                        ev = src[torch.randint(len(src), (k,), device=self.dev, generator=self.gen)]
                        off = torch.randint(8, 97, (k,), device=self.dev, generator=self.gen)
                        self.row[sel[msk]] = torch.maximum(ev - off, self.row_start[ev])
        if self.event_starts is not None and len(self.event_starts):
            j = m & (torch.rand(m.shape, device=self.dev, generator=self.gen) < TOWER_EVENT_P)
            if j.any():
                es = self.event_starts
                self.row[j] = es[torch.randint(len(es), (int(j.sum()),), device=self.dev, generator=self.gen)]
                self.h[j] = 0
        r = self.row[m]
        if self.cur is None:
            self.cur, self.ct, self.cc = self._g("ent", self.row), self._g("tick", self.row), self._g("cidx", self.row)
        self.cur[m], self.ct[m], self.cc[m] = self._g("ent", r), self._g("tick", r), self._g("cidx", r)
        self.left[m] = self.span(int(m.sum()))
        self.age[m] = 0

    def new_game(self, m):
        st = self._starts1
        self.row[m] = st[torch.randint(len(st), (int(m.sum()),), device=self.dev, generator=self.gen)]
        self.h[m] = 0
        self.sync(m)

    def _g(self, name, rows):
        """store.<name>[rows] on self.dev, wherever the store lives."""
        t = getattr(self.st, name)
        if t.device == rows.device:
            return t[rows]
        return t[rows.to(t.device)].to(self.dev, non_blocking=True)

    def chunk(self, model, fz, stats, kind, C):
        """C autoregressive steps; returns the mean loss (grad through the GRU over the chunk) and parts."""
        total, parts = 0.0, {}
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and self.dev.type == "cuda") else nullcontext()
        h = self.h
        rubric = self.w_circle > 0 or self.w_traj > 0
        xy = [F["x"], F["y"]]
        ix, iy = REG_FIELDS.index("x"), REG_FIELDS.index("y")
        pos = prev_d = prev_rd = None                 # differentiable hero position chain within the chunk
        rub = {"circle": [0.0, 0.0], "speed": [0.0, 0.0], "jitter": [0.0, 0.0]}   # weighted sum, weight
        for _ in range(C):
            act = self._g("act", self.row).long()
            with amp:
                pred, h = model.step(fz(self.cur, self.ct, self.cc), act, h)
            pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
            h = h.float()
            w = None
            if self.halflife > 0:   # step decay: h1 after a real state weighs most, then halves every `halflife`
                w = torch.exp2(-self.age.float() / self.halflife).clamp(min=self.floor)
            nxt = self._g("ent", self.row + 1)
            l, p = step_loss(pred, self.cur, nxt, stats, kind, self.ct, self.cc, w=w)
            total = total + l
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v
            if rubric:
                sc = stats.reg_scale.to(self.cur.device)[kind][:10]
                d = torch.stack([pred["reg"][:, :10, ix] * sc[:, ix], pred["reg"][:, :10, iy] * sc[:, iy]], -1)
                real0, real1 = self._g("ent", self.row)[:, :10][..., xy], nxt[:, :10][..., xy]
                if pos is None:
                    pos = self.cur[:, :10][..., xy].detach()
                pos_new = pos + d                                            # (B,10,2), grad through all moves
                rd = real1 - real0
                ok = (rd.abs().sum(-1) <= 12) & (d.detach().abs().sum(-1) <= 12)    # no death/respawn jump
                wv = (w if w is not None else torch.ones(len(d), device=d.device))[:, None] * ok
                circ = Fn.relu((pos_new - real1).norm(dim=-1) - rubric_radius(self.age.float() + 1)[:, None])
                spd = (d.norm(dim=-1) - rd.norm(dim=-1)).abs()
                for name, term, m in (("circle", circ, wv), ("speed", spd, wv)):
                    rub[name][0] = rub[name][0] + (term * m).sum()
                    rub[name][1] += float(m.sum())
                if prev_d is not None:                                   # reversals the real hero didn't make
                    jit = Fn.relu(-(d * prev_d).sum(-1)) * ((rd * prev_rd).sum(-1) >= 0)
                    rub["jitter"][0] = rub["jitter"][0] + (jit * wv).sum()
                    rub["jitter"][1] += float(wv.sum())
            with torch.no_grad():
                self.cur, self.ct, self.cc = apply_prediction(self.cur, self.ct, self.cc,
                                                              {k: v.detach() for k, v in pred.items()}, stats, fz)
                self.row += 1
                self.left -= 1
                self.age += 1
                done = self.row >= self.row_end[self.row]          # reached the terminal state
                resync = (self.left <= 0) & ~done
                if resync.any():
                    self.sync(resync)
                    if self.strata is not None:            # jumped to a new place: fresh memory (out of place: h is in the graph)
                        h = torch.where(resync[:, None, None], torch.zeros_like(h), h)
                if done.any():                                  # next game: fresh memory
                    h = torch.where(done[:, None, None], torch.zeros_like(h), h)
                    st = self._starts1
                    self.row[done] = st[torch.randint(len(st), (int(done.sum()),), device=self.dev,
                                                      generator=self.gen)]
                    self.sync(done)
            if rubric:        # the chain follows the rollout; it restarts where the rollout jumped or re-synced
                brk = (resync | done)[:, None] | ~ok
                cur_xy = self.cur[:, :10][..., xy]
                pos = torch.where(brk[..., None], cur_xy, pos_new)
                prev_d = torch.where(brk[..., None], torch.zeros_like(d), d)
                prev_rd = rd
        self.h = h.detach()
        out = {k: v / C for k, v in parts.items()}
        total = total / C
        if rubric:
            r = {k: v[0] / max(v[1], 1.0) for k, v in rub.items()}
            total = total + self.w_circle * r["circle"] + self.w_traj * (r["speed"] + r["jitter"])
            out.update({f"rubric/{k}": v for k, v in r.items()})
        return total, out


def unpack_codes(codes):
    """uint16/int action codes (...) -> (..., 6) long head indices (moba_unpack_action)."""
    c = codes.long()
    out = torch.empty(*c.shape, len(S.ACTION_SIZES), dtype=torch.long, device=c.device)
    for h in range(len(S.ACTION_SIZES) - 1, -1, -1):
        out[..., h] = c % S.ACTION_SIZES[h]
        c = c // S.ACTION_SIZES[h]
    return out


class EffectHead(nn.Module):
    """Training-only auxiliary head: from the difference of two siblings' new memories
    (h_edit - h_real, per entity), does this edit change this entity at all? Not part of
    the world model or its checkpoint; it exists to shape the trunk's action understanding."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, dh):
        return self.net(dh).squeeze(-1)


def soft_delta(pred):
    """Predicted normalized deltas with the change gate as a probability (differentiable)."""
    if "change" not in pred:
        return pred["reg"]
    return torch.where(pred["gate_mask"], torch.sigmoid(pred["change"]) * pred["value"], pred["value"])


class CFBatcher:
    """Counterfactual sibling batches from the real sim (mobawm/wm/cf.py CFStream).

    Each fork point: `burn` real states warm the memory, then the model steps the
    real state S_t under K+1 action sets -- the recorded one and K single edits
    (one hero, one action head) -- against the sim's true next state for each.
    Siblings share state and memory, so the only thing that explains their
    different outcomes is the action: dense counterfactual supervision.
    """

    def __init__(self, logs_dir, games, dev, forks, K, stride, workers, burn=4, seed=0,
                 w_effect=0.0, w_noeffect=0.0, w_affected=0.0, d=128):
        from . import cf
        files = sorted(Path(logs_dir).glob("games.*.mbl"))
        g = cf.load_log_games(files, games)
        assert g, f"no training games found in {logs_dir}"
        self.stream = cf.CFStream(g, workers=workers or None, K=K, stride=stride, burn=burn, seed=seed)
        self.dev, self.forks, self.K, self.burn = dev, forks, K, burn
        self.n_games, self.served, self.wait_s = len(g), 0, 0.0
        self.w_effect, self.w_noeffect, self.w_affected = w_effect, w_noeffect, w_affected
        self.aux = w_effect > 0 or w_noeffect > 0 or w_affected > 0
        self.head = EffectHead(d).to(dev) if w_affected > 0 else None

    def next(self):
        """Whole chunks until >= `forks` fork points, as GPU tensors."""
        keys = ("hist", "hist_tick", "hist_cidx", "hist_codes", "ent_in", "tick_in", "cidx_in", "codes",
                "ent_next", "tick_next", "cidx_next", "edit_head")
        parts, have = {k: [] for k in keys}, 0
        while have < self.forks:
            t0 = time.time()
            w, sl, n, v = self.stream.get()
            self.wait_s += time.time() - t0
            for k in keys:
                parts[k].append(torch.from_numpy(v[k][:n].astype(np.int64) if v[k].dtype == np.uint16
                                                 else v[k][:n]).to(self.dev, non_blocking=False))
            self.stream.release(w, sl)
            have += n
        self.served += have
        return {k: torch.cat(v) for k, v in parts.items()}

    def loss_backward(self, model, fz, stats, kind, weight, micro):
        """Fetch a batch and backprop weight x its mean loss in micro-batches of `micro` fork
        points (gradient accumulation: any batch size at a fixed memory cost). Returns the
        detached mean loss, its parts, and the number of samples."""
        b = self.next()
        n = b["codes"].shape[0]
        tot, parts, samples = 0.0, {}, 0
        for i in range(0, n, micro):
            mb = {k: v[i:i + micro] for k, v in b.items()}
            frac = mb["codes"].shape[0] / n
            l, p, m = self._loss(model, fz, stats, kind, mb)
            (weight * frac * l).backward()
            tot += frac * float(l)
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + frac * float(v)
            samples += m
        return torch.tensor(tot), parts, samples

    def loss(self, model, fz, stats, kind):
        return self._loss(model, fz, stats, kind, self.next())

    def _loss(self, model, fz, stats, kind, b):
        n, Sb = b["codes"].shape[:2]
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and self.dev.type == "cuda") else nullcontext()
        h = torch.zeros(n, S.N_ENTITIES, model.d, device=self.dev)
        with amp:
            for t in range(self.burn):
                _, h = model.step(fz(b["hist"][:, t], b["hist_tick"][:, t].long(), b["hist_cidx"][:, t].long()),
                                  unpack_codes(b["hist_codes"][:, t]), h)
            rep = lambda x: x.repeat_interleave(Sb, 0)
            e0, t0, c0 = rep(b["ent_in"]), rep(b["tick_in"].long()), rep(b["cidx_in"].long())
            pred, h1 = model.step(fz(e0, t0, c0), unpack_codes(b["codes"].flatten(0, 1)), rep(h.float()))
        pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
        e1 = b["ent_next"].flatten(0, 1)
        l, p = step_loss(pred, e0, e1, stats, kind, t0, c0)
        if self.aux:
            l, p = self._effect_terms(l, p, pred, h1.float(), e0, e1, stats, kind, n, Sb, b["edit_head"])
        return l, p, n * Sb

    def _effect_terms(self, l, p, pred, h1, e0, e1, stats, kind, n, Sb, edit_head):
        """Counterfactual difference terms: the effect of an edit is (edit sibling - real
        sibling), which cancels everything the siblings share (state, memory, the other
        heroes' actions) and leaves only what the edited action caused.

          effect     Huber(pred effect, true effect) where the true effect is nonzero,
                     averaged per edited action head (move / attack / Q / W / E count equally)
          noeffect   Huber(pred effect, 0) where the edit changed nothing
          affected   BCE: does this edit change this entity at all, from h_edit - h_real
                     (EffectHead), positives and negatives averaged separately
        """
        dev = e0.device
        scale = stats.reg_scale.to(dev)[kind]
        target = (e1[..., REG_IDX] - e0[..., REG_IDX]) / scale
        if stats.regen_rule:
            target = target - regen_part(e0) / scale
        pd = soft_delta(pred)
        E, R = target.shape[1:]
        target, pd = target.view(n, Sb, E, R), pd.view(n, Sb, E, R)
        alive = (e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] > 0)
        valid = (alive.view(n, Sb, E, 1) & stats.reg_mask.to(dev)[kind].view(1, 1, E, R))
        valid = valid[:, 1:] & valid[:, :1]                      # alive in the edit and the real sibling
        eff = target[:, 1:] - target[:, :1]                      # true effect of each edit
        peff = pd[:, 1:] - pd[:, :1]                             # predicted effect
        present = valid & (eff.abs() > 1e-6)
        absent = valid & ~present
        fw = torch.ones(R, device=dev)                           # hero walking weighted as in step_loss
        fw[[REG_FIELDS.index("x"), REG_FIELDS.index("y")]] = HERO_WALK_WEIGHT
        fw = torch.where(torch.from_numpy(KIND).to(dev).view(1, 1, E, 1) == 0, fw.view(1, 1, 1, R),
                         torch.ones_like(fw).view(1, 1, 1, R))
        err = Fn.huber_loss(peff, eff, reduction="none", delta=1.0) * fw
        head = edit_head[:, 1:].long().to(dev)                   # (n, K) edited action head 0..5
        l_eff, heads = 0.0, 0
        for hd in range(6):
            m = present & (head == hd).view(n, Sb - 1, 1, 1)
            if m.any():
                l_eff = l_eff + (err * m).sum() / m.sum()
                heads += 1
                # live metric: share of the true effect explained (1 - |pred - true|_1 / |true|_1)
                p[f"metric/effect_explained_pct/{('move_y', 'move_x', 'attack', 'q', 'w', 'e')[hd]}"] = \
                    float(100 * (1 - ((peff - eff).abs() * m).sum() / (eff.abs() * m).sum().clamp(min=1e-9)))
        l_eff = l_eff / max(heads, 1)
        l_no = (Fn.huber_loss(peff, torch.zeros_like(peff), reduction="none", delta=1.0) * fw * absent).sum() \
            / absent.sum().clamp(min=1)
        p["effect/sibling"] = float(l)                           # the plain per-sibling loss, comparable with E6
        total = l + self.w_effect * l_eff + self.w_noeffect * l_no
        p["effect/effect"], p["effect/noeffect"] = l_eff, l_no
        if self.head is not None:
            h1 = h1.view(n, Sb, E, -1)
            logit = self.head(h1[:, 1:] - h1[:, :1])            # (n, K, E)
            tgt = (present.any(-1) | ((e1[..., F["alive"]].view(n, Sb, E)[:, 1:] > 0)
                                      != (e1[..., F["alive"]].view(n, Sb, E)[:, :1] > 0))).float()
            bce = Fn.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
            pos, neg = tgt > 0.5, tgt < 0.5
            l_aff = 0.5 * (bce[pos].mean() if pos.any() else 0.0) + 0.5 * (bce[neg].mean() if neg.any() else 0.0)
            total = total + self.w_affected * l_aff
            p["effect/affected"] = l_aff
            with torch.no_grad():
                pr = logit > 0
                tp, fp_, fn_ = (pr & pos).sum(), (pr & neg).sum(), (~pr & pos).sum()
                p["metric/affected_f1_pct"] = float(100 * 2 * tp / (2 * tp + fp_ + fn_).clamp(min=1))
        return total, p


HORIZONS = (1, 4, 16, 64, 256, 1024, 4096)


# --- scenario slices (E20 corpus) -----------------------------------------------
# Slices = (perturbed start state, 128 x 10 action codes, outcome) mined from the real sim
# (scripts/moba/mine_scenarios.py). A background process regenerates each slice's states
# exactly with moba_fork.c ms_replay; ScenarioLanes rolls the model open loop through
# them (memory warmed on the first `burn` real ticks), with the same step loss and decay
# as Lanes. Sampling: 50/50 tower died / survived, full-length (128-tick) slices only.
SCEN_HEADS = (7, 7, 3, 2, 2, 2)


def _unpack_codes(codes):
    """(..., 10) uint16 action codes -> (..., 10, 6) head indices (moba_unpack_action)."""
    c = codes.astype(np.int64)
    out = np.zeros(codes.shape + (6,), np.uint8)
    for h in range(5, -1, -1):
        out[..., h] = c % SCEN_HEADS[h]
        c //= SCEN_HEADS[h]
    return out


def _replay(L, st, codes, ent_out, tk, cc, n=128):
    """Regenerate one scenario slice into ent_out (n+1, 206, len(F)); with LIST_STATE also the stored lists."""
    from .data import LIST_NF, N_BASE, pack_lists_np
    if ent_out.shape[-1] == N_BASE:
        return L.ms_replay(st.ctypes.data, codes.ctypes.data, n, -1, ent_out.ctypes.data, tk.ctypes.data, cc.ctypes.data)
    e = np.zeros((n + 1, S.N_ENTITIES, N_BASE), np.float32); sc = np.zeros((n + 1, S.N_ENTITIES, 64), np.uint8)
    r = L.ms_replay_scan(st.ctypes.data, codes.ctypes.data, n, -1, e.ctypes.data, tk.ctypes.data, cc.ctypes.data, sc.ctypes.data)
    ent_out[..., :N_BASE] = e
    ent_out[..., N_BASE:N_BASE + LIST_NF] = pack_lists_np(sc)
    return r


def _scen_worker(meta, q, B, seed):
    import ctypes
    import zlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "moba"))
    from mine_scenarios import lib as fork_lib
    L = fork_lib()
    rng = np.random.default_rng(seed)
    died_idx, live_idx = meta["died_idx"], meta["live_idx"]
    NE_, NF_ = S.N_ENTITIES, len(F)
    while True:
        pick = np.concatenate([rng.choice(died_idx, B // 2), rng.choice(live_idx, B - B // 2)])
        ent = np.zeros((B, 129, NE_, NF_), np.float32); tk = np.zeros((B, 129), np.int32); cc = np.zeros((B, 129), np.int32)
        act = np.zeros((B, 128, 10, 6), np.uint8)
        for j, g in enumerate(pick):
            sh, i = meta["where"][g]
            z = meta["shards"][sh]
            st = np.frombuffer(zlib.decompress(z["blob"][z["off"][i]:z["off"][i + 1]].tobytes()), np.uint8)
            codes = np.ascontiguousarray(z["codes"][i])
            _replay(L, st, codes, ent[j], tk[j], cc[j])
            act[j] = _unpack_codes(codes)
        q.put((ent, act, tk, cc))


class ScenarioFeeder:
    """Loads every shard (compressed) once, then worker processes regenerate batches of slices."""

    def __init__(self, root, B, workers=2, prefetch=3):
        import multiprocessing as mp
        files = sorted(Path(root).rglob("shard_*.npz"))
        shards, where, died = [], [], []
        for fi, f in enumerate(files):
            z = np.load(f)
            full = z["len"] == 128
            shards.append(dict(blob=z["state_blob"], off=z["state_off"], codes=z["codes"]))
            idx = np.nonzero(full)[0]
            where += [(fi, int(i)) for i in idx]
            died += [bool(z["died"][i]) for i in idx]
        died = np.array(died)
        self.meta = dict(shards=shards, where=where, died_idx=np.nonzero(died)[0], live_idx=np.nonzero(~died)[0])
        print(f"scenarios: {len(files)} shards, {len(where)} full-length slices "
              f"({int(died.sum())} died / {int((~died).sum())} survived)", flush=True)
        ctx = mp.get_context("fork")
        self.q = ctx.Queue(prefetch)
        self.procs = [ctx.Process(target=_scen_worker, args=(self.meta, self.q, B, 1000 + w), daemon=True)
                      for w in range(workers)]
        for p in self.procs:
            p.start()

    def get(self):
        return self.q.get()


class ScenarioLanes:
    def __init__(self, feeder, d, dev, halflife, floor, burn=4):
        self.fd, self.d, self.dev, self.halflife, self.floor, self.burn = feeder, d, dev, halflife, floor, burn
        self.k = None
        self.seen = 0

    def _new(self, model, fz):
        ent, act, tk, cc = self.fd.get()
        self.seen += len(ent)
        self.ent = torch.from_numpy(ent).to(self.dev, non_blocking=True)
        self.act = torch.from_numpy(act).to(self.dev).long()
        self.tk, self.cc = torch.from_numpy(tk).to(self.dev).long(), torch.from_numpy(cc).to(self.dev).long()
        B = self.ent.shape[0]
        if KILL_EDGES:      # here on the device, not in the forked workers (torch after fork deadlocks)
            for j in range(B):
                rebuild_kill_edges(self.ent[j])
        h = torch.zeros(B, S.N_ENTITIES, self.d, device=self.dev)
        with torch.no_grad():
            for k in range(self.burn):
                _, h = model.step(fz(self.ent[:, k], self.tk[:, k], self.cc[:, k]), self.act[:, k], h)
        self.h = h.float()
        self.k = self.burn
        self.cur, self.ct, self.cc_cur = self.ent[:, self.burn].clone(), self.tk[:, self.burn].clone(), self.cc[:, self.burn].clone()

    def chunk(self, model, fz, stats, kind, C):
        if self.k is None or self.k >= 128:
            self._new(model, fz)
        total, parts, n = 0.0, {}, 0
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and self.dev.type == "cuda") else nullcontext()
        h = self.h
        while n < C and self.k < 128:
            with amp:
                pred, h = model.step(fz(self.cur, self.ct, self.cc_cur), self.act[:, self.k], h)
            pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
            h = h.float()
            w = None
            if self.halflife > 0:
                age = torch.full((len(self.cur),), float(self.k - self.burn), device=self.dev)
                w = torch.exp2(-age / self.halflife).clamp(min=self.floor)
            l, p = step_loss(pred, self.cur, self.ent[:, self.k + 1], stats, kind, self.ct, self.cc_cur, w=w)
            total = total + l
            for key, v in p.items():
                parts[key] = parts.get(key, 0.0) + v
            with torch.no_grad():
                self.cur, self.ct, self.cc_cur = apply_prediction(self.cur, self.ct, self.cc_cur,
                                                                  {k: v.detach() for k, v in pred.items()}, stats, fz)
            self.k += 1
            n += 1
        self.h = h.detach()
        return total / max(n, 1), {k: v / max(n, 1) for k, v in parts.items()}


def load_scen_heldout(root, n=128, seed=5):
    """A fixed held-out batch of scenario slices (n/2 died, n/2 survived), regenerated once."""
    import zlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "moba"))
    from mine_scenarios import lib as fork_lib
    L = fork_lib()
    rows = []
    for f in sorted(Path(root).rglob("shard_*.npz")):
        z = np.load(f)
        for i in np.nonzero(z["len"] == 128)[0]:
            rows.append((z, int(i), bool(z["died"][i])))
    rng = np.random.default_rng(seed)
    d = [r for r in rows if r[2]]; lv = [r for r in rows if not r[2]]
    nd, nl = min(n // 2, len(d)), min(n - n // 2, len(lv))
    pick = [d[i] for i in rng.choice(len(d), nd, replace=False)] + [lv[i] for i in rng.choice(len(lv), nl, replace=False)]
    n = len(pick)
    ent = np.zeros((n, 129, S.N_ENTITIES, len(F)), np.float32); tk = np.zeros((n, 129), np.int32); cc = np.zeros((n, 129), np.int32)
    act = np.zeros((n, 128, 10, 6), np.uint8); tower = np.zeros(n, np.int64); died = np.zeros(n, bool)
    for j, (z, i, dd) in enumerate(pick):
        st = np.frombuffer(zlib.decompress(z["state_blob"][z["state_off"][i]:z["state_off"][i + 1]].tobytes()), np.uint8)
        codes = np.ascontiguousarray(z["codes"][i])
        _replay(L, st, codes, ent[j], tk[j], cc[j])
        act[j] = _unpack_codes(codes); tower[j] = int(z["tower"][i]); died[j] = dd
        if KILL_EDGES:
            rebuild_kill_edges(torch.from_numpy(ent[j]))
    print(f"held-out scenarios: {n} slices ({int(died.sum())} died / {int((~died).sum())} survived) from {root}", flush=True)
    return dict(ent=ent, act=act, tk=tk, cc=cc, tower=tower, died=died)


@torch.no_grad()
def scen_eval(model, fz, stats, kind, dev, hb, burn=4, mb=32):
    """Held-out tower-dive slices, open loop from tick `burn`: mean step loss (all, tower part) and
    whether the target tower dies in the dream (precision/recall of 'dies', accuracy)."""
    was = model.training
    model.eval()
    n = len(hb["died"])
    tot, tow, steps, pred_died = 0.0, 0.0, 0, np.zeros(n, bool)
    for s0 in range(0, n, mb):
        sl = slice(s0, min(n, s0 + mb))
        ent = torch.from_numpy(hb["ent"][sl]).to(dev); act = torch.from_numpy(hb["act"][sl]).to(dev).long()
        tk = torch.from_numpy(hb["tk"][sl]).to(dev).long(); cc = torch.from_numpy(hb["cc"][sl]).to(dev).long()
        tw = torch.from_numpy(hb["tower"][sl]).to(dev) + 182
        B = len(ent)
        h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
        for k in range(burn):
            _, h = model.step(fz(ent[:, k], tk[:, k], cc[:, k]), act[:, k], h)
        cur, ct, c2 = ent[:, burn].clone(), tk[:, burn].clone(), cc[:, burn].clone()
        dead = torch.zeros(B, dtype=torch.bool, device=dev)
        for k in range(burn, 128):
            pred, h = model.step(fz(cur, ct, c2), act[:, k], h)
            l, p = step_loss(pred, cur, ent[:, k + 1], stats, kind, ct, c2)
            tot += float(l) * B; steps += B
            tow += sum(float(v) for key, v in p.items() if key.startswith("tower/")) * B
            cur, ct, c2 = apply_prediction(cur, ct, c2, pred, stats, fz)
            cur = torch.nan_to_num(cur)
            dead |= cur[torch.arange(B, device=dev), tw, F["alive"]] <= 0
        pred_died[sl] = dead.cpu().numpy()
    if was:
        model.train()
    real = hb["died"]
    tp = int((pred_died & real).sum())
    nan = float("nan")
    return {"loss": tot / steps, "tower_loss": tow / steps, "died_precision": tp / pred_died.sum() if pred_died.sum() else nan,
            "died_recall": tp / real.sum(), "outcome_acc": float((pred_died == real).mean()),
            "survivor_false_death": float((pred_died & ~real).sum() / max((~real).sum(), 1))}


@torch.no_grad()
def tower_eval(model, fz, store, stats, dev, n=64, H=128, burn=8, tol=32, seed=11):
    """Tower deaths on held-out windows (the same n windows every time): open loop H ticks from a
    real state (memory warmed on `burn` real ticks) and one step at a time over the same ticks.
    A predicted open-loop fall is a hit if the same tower really falls within +-tol ticks.
    Returns precision/recall for both, and one-step false alarms per real fall."""
    was = model.training
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    starts = store.starts(burn + H + 1).cpu()
    rows = starts[torch.randint(len(starts), (n,), generator=g)]
    idx = (rows.unsqueeze(1) + torch.arange(burn + H + 1)).to(store.ent.device)
    ent, tick, cidx = store.ent[idx].to(dev), store.tick[idx].to(dev), store.cidx[idx].to(dev)
    act = store.act[idx[:, :-1]].long().to(dev)
    alive = lambda e: e[..., 182:, F["alive"]] > 0
    h = torch.zeros(n, S.N_ENTITIES, model.d, device=dev)
    for k in range(burn):
        _, h = model.step(fz(ent[:, k], tick[:, k], cidx[:, k]), act[:, k], h)
    h_tf = h.clone()
    cur, ct, cc = ent[:, burn].clone(), tick[:, burn].clone(), cidx[:, burn].clone()
    real_ev, pred_ev = [], []
    tf_real = tf_pred = tf_hit = 0
    for k in range(burn, burn + H):
        # open loop
        p, h = model.step(fz(cur, ct, cc), act[:, k], h)
        nxt, ct, cc = apply_prediction(cur, ct, cc, p, stats, fz)
        nxt = torch.nan_to_num(nxt)
        pf = alive(cur) & ~alive(nxt)
        rf = alive(ent[:, k]) & ~alive(ent[:, k + 1])
        pred_ev += [(int(w), int(t), k) for w, t in torch.nonzero(pf).tolist()]
        real_ev += [(int(w), int(t), k) for w, t in torch.nonzero(rf).tolist()]
        cur = nxt
        # one step from the real state
        p1, h_tf = model.step(fz(ent[:, k], tick[:, k], cidx[:, k]), act[:, k], h_tf)
        n1, _, _ = apply_prediction(ent[:, k], tick[:, k], cidx[:, k], p1, stats, fz)
        pf1 = alive(ent[:, k]) & ~alive(torch.nan_to_num(n1))
        tf_real += int(rf.sum()); tf_pred += int(pf1.sum()); tf_hit += int((rf & pf1).sum())
    used, hits = set(), 0
    for w, t, k in real_ev:
        c = [(abs(k - k2), j) for j, (w2, t2, k2) in enumerate(pred_ev) if w2 == w and t2 == t and j not in used
             and abs(k - k2) <= tol]
        if c:
            used.add(min(c)[1]); hits += 1
    if was:
        model.train()
    nan = float("nan")
    return {"ol_real": len(real_ev), "ol_pred": len(pred_ev), "ol_precision": hits / len(pred_ev) if pred_ev else nan,
            "ol_recall": hits / len(real_ev) if real_ev else nan, "onestep_precision": tf_hit / tf_pred if tf_pred else nan,
            "onestep_recall": tf_hit / tf_real if tf_real else nan,
            "onestep_false_per_fall": (tf_pred - tf_hit) / tf_real if tf_real else nan}


@torch.no_grad()
def eval_horizons(model, fz, store, stats, kind, dev, horizons=HORIZONS, n=32, burn=8, seed=7):
    """THE key metric: held-out transition loss at open-loop horizon h.

    From the same n windows every time (seed), warm the memory on `burn` real
    states, then run open loop. At step h the model predicts state h from its own
    state h-1; loss@h is the training loss (step_loss) of that prediction against
    the real state h. loss@1 is the ordinary one-step (teacher-forced) loss.
    Horizons up to 1024 share one set of windows; longer ones need windows from
    games long enough (4096 + burn ticks), a second, smaller-choice set.
    Also hero_pos_L1@h: mean |dx|+|dy| (cells) of the 10 heroes at step h.
    """
    was = model.training
    model.eval()
    out = {}
    for hs in ([h for h in horizons if h <= 1024], [h for h in horizons if h > 1024]):
        if not hs or max(n for _, n in store.spans) < burn + max(hs):   # no game long enough
            continue
        H = max(hs)
        g = torch.Generator(device=dev); g.manual_seed(seed)
        ent, act, tick, cidx = store.sample(n, burn + H, g)
        h = torch.zeros(n, S.N_ENTITIES, model.d, device=dev)
        for t in range(burn):
            _, h = model.step(fz(ent[:, t], tick[:, t], cidx[:, t]), act[:, t], h)
        cur, ct, cc = ent[:, burn].clone(), tick[:, burn].clone(), cidx[:, burn].clone()
        xy = [F["x"], F["y"]]
        track = [cur[:, :10][..., xy]]                       # predicted hero positions, for the rubric
        for k in range(1, H + 1):
            pred, h = model.step(fz(cur, ct, cc), act[:, burn + k - 1], h)
            if k in hs:
                l, parts = step_loss(pred, cur, ent[:, burn + k], stats, kind, ct, cc)
                sp = summarize_parts({a: float(b) for a, b in parts.items()})
                out[f"loss@{k}"] = float(l)
                out[f"hero_loss@{k}"] = sp["hero/total"]
            cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
            track.append(cur[:, :10][..., xy])
            if k in hs:
                out[f"hero_pos_L1@{k}"] = _hero_pos(cur, ent[:, burn + k])
                out[f"windows@{k}"] = n
                # circle: share of heroes within the rubric radius of the real hero
                dist = (cur[:, :10][..., xy] - ent[:, burn + k, :10][..., xy]).norm(dim=-1)
                out[f"circle_hit_pct@{k}"] = float(100 * (dist <= rubric_radius(torch.tensor(float(k)))).float().mean())
        if max(hs) <= 1024:                                  # trajectory rubric over the first 256 steps
            P = torch.stack(track[:257], 1)                  # (n, T+1, 10, 2) predicted
            R = ent[:, burn:burn + 257, :10][..., xy]        # real
            dp, dr = P[:, 1:] - P[:, :-1], R[:, 1:] - R[:, :-1]
            ok = (dr.abs().sum(-1) <= 12) & (dp.abs().sum(-1) <= 12)
            sp, sr = dp.norm(dim=-1), dr.norm(dim=-1)
            out["teleport_pct"] = float(100 * ((dp.abs().sum(-1) > 3) & (dr.abs().sum(-1) <= 3)).float().mean())
            rev_p = ((dp[:, 1:] * dp[:, :-1]).sum(-1) < 0) & ok[:, 1:] & ok[:, :-1]
            rev_r = ((dr[:, 1:] * dr[:, :-1]).sum(-1) < 0)
            out["jitter_pct"] = float(100 * (rev_p & ~rev_r).float().mean())
            out["speed_ratio"] = float(sp[ok].mean() / sr[ok].mean().clamp(min=1e-6))
            out["freeze_pct"] = float(100 * ((sp < 0.05) & (sr > 0.5) & ok).float().mean())
    model.train(was)
    return out


@torch.no_grad()
def eval_rollout(model, fz, store, stats, kind, dev, gen, n=16, burn=8, H=64, marks=(1, 4, 16, 64)):
    """Open loop: burn-in on real states, then feed predictions back for H steps."""
    B = n
    ent, act, tick, cidx = store.sample(B, burn + H, gen)
    h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
    for t in range(burn):
        f = fz(ent[:, t], tick[:, t], cidx[:, t])
        _, h = model.step(f, act[:, t], h)
    cur, ct, cc = ent[:, burn].clone(), tick[:, burn].clone(), cidx[:, burn].clone()
    start = cur.clone()
    res = {}
    for k in range(1, H + 1):
        f = fz(cur, ct, cc)
        pred, h = model.step(f, act[:, burn + k - 1], h)
        cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
        if k in marks:
            truth = ent[:, burn + k]
            res[k] = {name: (metric(cur, truth), metric(start, truth)) for name, metric in METRICS.items()}
    return res


def _hero_pos(s, t):
    return ((s[:, :10, F["x"]] - t[:, :10, F["x"]]).abs() + (s[:, :10, F["y"]] - t[:, :10, F["y"]]).abs()).mean().item()


def _hero_hp(s, t):
    return (s[:, :10, F["health"]] - t[:, :10, F["health"]]).abs().mean().item()


def _creep_pos(s, t):
    m = (s[:, 10:110, F["alive"]] > 0) & (t[:, 10:110, F["alive"]] > 0)
    d = (s[:, 10:110, F["x"]] - t[:, 10:110, F["x"]]).abs() + (s[:, 10:110, F["y"]] - t[:, 10:110, F["y"]]).abs()
    return d[m].mean().item() if m.any() else float("nan")


def _creep_alive(s, t):
    return ((s[:, 10:110, F["alive"]] > 0) == (t[:, 10:110, F["alive"]] > 0)).float().mean().item()


def _tower_hp(s, t):
    return (s[:, 182:, F["health"]] - t[:, 182:, F["health"]]).abs().mean().item()


METRICS = {"hero_pos_L1": _hero_pos, "hero_hp": _hero_hp, "creep_pos_L1": _creep_pos,
           "creep_alive_acc": _creep_alive, "tower_hp": _tower_hp}


def log_report(run, report):
    """End-of-run evaluation as wandb tables and summary numbers."""
    import wandb
    he, sh = report["onestep_heldout"], report["onestep_heldout_shuffled_actions"]
    rows = []
    for f, v in he.items():
        if f == "alive_flip" or v["n_changed"] < 20:
            continue
        kn, fname = f.split(".", 1)
        rows.append([kn, fname, EFFECT_OF.get(fname, "?"), v["model_mae"], v["persist_mae"], sh[f]["model_mae"],
                     100 * (1 - v["model_mae"] / max(v["persist_mae"], 1e-9)),
                     100 * (sh[f]["model_mae"] / max(v["model_mae"], 1e-9) - 1),
                     v["changed_model_mae"], v["changed_persist_mae"], v["n_changed"]])
    run.log({"final/onestep_heldout": wandb.Table(
        columns=["unit", "field", "effect", "model_err", "copy_err", "wrong_actions_err", "better_than_copy_pct",
                 "action_dependence_pct", "changed_model_err", "changed_copy_err", "n_changed"], data=rows)})
    for split in ("rollout_heldout", "rollout_train"):
        rrows = [[int(h), name, m[0], m[1]] for h, ms in report[split].items() for name, m in ms.items()]
        run.log({f"final/{split}": wandb.Table(columns=["horizon", "metric", "model", "copy"], data=rrows)})
    run.summary.update({"final/loss_last_100": report["loss_last_100"],
                        "final/slope_per_1k_it": report["log_loss_slope_per_1k_it_last_quarter"],
                        "final/deaths_recall": he["alive_flip"]["recall"]})


def backfill(run_dir: Path, wandb_id: str):
    """Attach an older run's report.json to its wandb run (tables + summary)."""
    import wandb
    report = json.loads((run_dir / "report.json").read_text())
    run = wandb.init(project="moba-wm", id=wandb_id, resume="must")
    log_report(run, report)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/moba/mix100/dense")
    ap.add_argument("--train-games", default="0-25")
    ap.add_argument("--held-games", default="91-99")
    ap.add_argument("--held2-games", default="", help="a second held-out set, evaluated like the first and logged "
                    "under <--held2-name>/... (e.g. games with random-action controllers)")
    ap.add_argument("--held2-name", default="random")
    ap.add_argument("--scenarios", default="", help="scenario slice corpus dir (E20); adds ScenarioLanes to every update")
    ap.add_argument("--scen-batch", type=int, default=64)
    ap.add_argument("--freeze", default="none", choices=("none", "heads", "gru_heads", "no_embed", "direct"),
                    help="train only these parameter groups (see FREEZE in main)")
    ap.add_argument("--scen-only", action="store_true", help="train on scenario lanes only (no regular data loss)")
    ap.add_argument("--val-light", action="store_true", help="validation = tower dives + tower windows only")
    ap.add_argument("--scen-weight", type=float, default=None,
                    help="default: per-lane (scen_batch / batch), i.e. no special weighting")
    ap.add_argument("--scen-heldout", default="", help="held-out scenario slices dir (from held-out games) for validation")
    ap.add_argument("--scen-workers", type=int, default=3)
    ap.add_argument("--strata", default="", help="stratified lanes, e.g. 'uniform:0.1,tower_destroyed:0.2,hero_death:0.2,...'")
    ap.add_argument("--event-eval", action="store_true", help="per-event evaluation suite on held-out games at each validation")
    ap.add_argument("--skip-val0", action="store_true", help="no validation at update 0")
    ap.add_argument("--tower-health-weight", type=float, default=1.0)
    ap.add_argument("--tower-consist", type=float, default=0.0)
    ap.add_argument("--tower-event-p", type=float, default=0.0)
    ap.add_argument("--target-weight", type=float, default=1.0, help="loss weight of the attack edges")
    ap.add_argument("--predict-targets", action="store_true", help="predict who each unit attacks (attack graph in the dream)")
    ap.add_argument("--target-dist", action="store_true", help="learned distance bias on the target scores")
    ap.add_argument("--target-decay", action="store_true", help="rollout-depth step decay on the edge loss")
    ap.add_argument("--kill-edges", action="store_true", help="rebuild the unrecorded killing-blow edges in all data")
    ap.add_argument("--list-state", action="store_true", help="stored target lists as state + aggro inputs (data.LIST_STATE)")
    ap.add_argument("--list-weight", type=float, default=1.0, help="loss weight of next-tick list membership")
    # Fleet (many pods, zelda-RL style): each pod trains on its own shard; every --fleet-every updates the pods
    # average their weights through Upstash Redis (mobawm/fleet_link.py). Fleet rank 0 also validates/checkpoints.
    ap.add_argument("--fleet-exp", default="", help="fleet experiment name (empty = single pod)")
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world", type=int, default=1)
    ap.add_argument("--fleet-launch", default="0", help="launch id: sync keys are per launch")
    ap.add_argument("--fleet-every", type=int, default=10, help="updates between weight averages")
    ap.add_argument("--fleet-timeout", type=float, default=300.0)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model step (about 2x on A100/A40)")
    ap.add_argument("--pack-only", action="store_true", help="load the training store, write --packed-dir, exit (before a multi-GPU run)")
    ap.add_argument("--ema", type=float, default=0.0, help="keep an EMA of the weights with this decay; checkpoints "
                    "model_*.pt / model.pt are the EMA weights, raw_*.pt the raw ones (0 = off)")
    ap.add_argument("--packed-dir", default="", help="raw copy of the training store: memory-mapped if present, else written once")
    ap.add_argument("--grid-cnn", action="store_true", help="global 64x64 map of the state through a CNN into every token")
    ap.add_argument("--direct-damage", action="store_true", help="incoming_damage feeds the health prediction directly")
    ap.add_argument("--direct-lr", type=float, default=0.05, help="learning rate of the direct-damage coefficients")
    ap.add_argument("--attack-mode", default=None, choices=(None, "derived", "real", "state"),
                    help="attack-graph inputs (who each unit attacks / is attacked by); see data.ATTACK_MODE")
    ap.add_argument("--minutes", type=float, default=6.0, help="time budget (0 = none; stop on --max-updates)")
    ap.add_argument("--max-updates", type=int, default=0, help="stop after this many updates (0 = none)")
    ap.add_argument("--cf", action="store_true", help="add counterfactual sibling batches from the real sim (CFBatcher)")
    ap.add_argument("--cf-logs", default="", help="--cf: dir of .mbl logs (default <data>/../logs)")
    ap.add_argument("--cf-forks", type=int, default=1024, help="--cf: fork points per update (x (K+1) samples)")
    ap.add_argument("--cf-k", type=int, default=7, help="--cf: counterfactual edits per fork point")
    ap.add_argument("--cf-stride", type=int, default=8, help="--cf: fork every N ticks of a replayed game")
    ap.add_argument("--cf-workers", type=int, default=0, help="--cf: sim processes (0 = all cores but 2)")
    ap.add_argument("--cf-weight", type=float, default=1.0, help="--cf: weight of the counterfactual loss")
    ap.add_argument("--cf-effect", type=float, default=0.0,
                    help="--cf: weight of the effect loss (predicted vs true edit-minus-real difference)")
    ap.add_argument("--cf-noeffect", type=float, default=0.0, help="--cf: weight of the no-hallucinated-effect loss")
    ap.add_argument("--cf-affected", type=float, default=0.0,
                    help="--cf: weight of the affected-entity aux head (training only)")
    ap.add_argument("--cf-micro", type=int, default=512,
                    help="--cf: fork points per forward/backward (gradient accumulation; memory, not batch size)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--burn", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=0, help="also keep model_<it>.pt every N iterations")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/token_wm/smoke")
    ap.add_argument("--gate", action="store_true", help="copy by default: per-field change gate")
    ap.add_argument("--gate-fields", default="", help="comma list of gated fields (empty = all)")
    ap.add_argument("--gate-loss-weight", type=float, default=1.0, help="weight on the gate's yes/no loss")
    ap.add_argument("--gate-value-norm", default="changed", choices=("changed", "all"))
    ap.add_argument("--hero-walk-weight", type=float, default=1.0, help="weight on the hero x/y loss terms")
    ap.add_argument("--baseline", action="store_true",
                    help="the adopted baseline (EXPERIMENTS.md): gate on all fields but movement, "
                         "yes/no loss x0.2, value loss averaged over all entries, hero walking x10")
    ap.add_argument("--effect-weights", default="", help="semantic loss, e.g. abilities=3,health=3,deaths=3")
    ap.add_argument("--event-frac", type=float, default=0.0,
                    help="fraction of each batch drawn from windows with a cast, damage or death")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--init", help="run dir (or .pt) to continue from; its stats.pt is reused")
    ap.add_argument("--rollout", action="store_true",
                    help="autoregressive training: lanes fed their own predictions (see Lanes)")
    ap.add_argument("--max-span", type=int, default=4096, help="--rollout: longest open-loop span (4096 = whole game)")
    ap.add_argument("--chunk", type=int, default=16, help="--rollout: steps per update (gradient through the GRU)")
    ap.add_argument("--span-mu", type=float, default=6.0, help="--rollout: mean log2 open-loop span (6 = 64 ticks)")
    ap.add_argument("--span-sigma", type=float, default=3.0, help="--rollout: std of log2 span")
    ap.add_argument("--decay-halflife", type=float, default=64.0,
                    help="--rollout: loss weight halves every N open-loop steps since the last real state (0 = off)")
    ap.add_argument("--decay-floor", type=float, default=0.02, help="--rollout: minimum step weight")
    ap.add_argument("--no-xp-clamp", action="store_true",
                    help="drop rule R7 (XP never decreases) from the dream's state update")
    ap.add_argument("--store-cpu", action="store_true",
                    help="keep the training store in CPU RAM and copy rows to the GPU per step (fits smaller GPUs)")
    ap.add_argument("--rubric-circle", type=float, default=0.0,
                    help="--rollout: weight of the circle hinge (hero lands within the growing radius)")
    ap.add_argument("--rubric-traj", type=float, default=0.0,
                    help="--rollout: weight of the trajectory terms (speed match + no extra reversals)")
    ap.add_argument("--ol-every", type=int, default=0,
                    help="held-out loss-by-horizon eval (h1..h4096) every N its; 0 = with --val-every")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--backfill", help="run dir: attach its report.json to --wandb-id and exit")
    ap.add_argument("--wandb-id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.backfill:
        return backfill(Path(args.backfill), args.wandb_id)
    # Multi-GPU (torchrun): every process trains on its own GPU with batch/WORLD lanes (same global batch, own lane
    # seed); gradients are averaged after each backward (an all-reduce of one flat buffer: the model is 0.7M params
    # and is called `chunk` times per update, which DDP's wrapper does not fit). Rank 0 alone logs, validates,
    # checkpoints and talks to W&B; the others write nothing but their own log under <out>/rank<r>.
    import os
    WORLD, RANK = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    MAIN = RANK == 0
    dist = None
    if WORLD > 1:
        import datetime
        import torch.distributed as dist
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            args.device = f"cuda:{local}"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(minutes=120))
        assert args.batch % WORLD == 0, f"--batch {args.batch} must divide by {WORLD} GPUs"
        args.batch //= WORLD
        if args.scenarios:
            args.scen_batch //= WORLD
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        if not MAIN:
            args.wandb, args.event_eval, args.scen_heldout = False, False, None
            args.out = str(Path(args.out) / f"rank{RANK}")
    if args.fleet_exp and args.fleet_rank != 0:     # followers only train: no validation, checkpoints or W&B
        args.wandb, args.event_eval, args.scen_heldout, args.skip_val0 = False, False, None, True
        args.val_every, args.ckpt_every, args.ol_every = 10 ** 9, 0, 0
    LEAD = MAIN and args.fleet_rank == 0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gen = torch.Generator(device=dev); gen.manual_seed(1000 * RANK)          # own lanes per GPU
    egen = torch.Generator(device=dev); egen.manual_seed(1)

    files = sorted(Path(args.data).glob("games.*.mbs"))
    tr_idx, he_idx = parse_games(args.train_games), parse_games(args.held_games)
    assert not tr_idx & he_idx, "train and held-out games overlap"
    he2_idx = parse_games(args.held2_games) if args.held2_games else set()
    assert not tr_idx & he2_idx, "train and held2 games overlap"
    t_load = time.time()
    if args.list_state:
        from . import data as _data
        _data.enable_list_state()           # before any store: entity rows get the list fields
    print(f"loading from {args.data} ({len(files)} files) ...", flush=True)
    if args.no_xp_clamp:
        from . import data as _data
        _data.XP_MONOTONE = False
    tr_store = load_store(files, tr_idx, torch.device("cpu") if args.store_cpu else dev,
                          packed=args.packed_dir or None)
    if args.pack_only:
        print(f"packed store ready under {args.packed_dir}", flush=True)
        return
    if args.store_cpu:
        tr_store.out_dev = dev              # Lanes / sample() hand everything back on the GPU
    he_store = load_store(files, he_idx, dev)
    he2_store = load_store(files, he2_idx, dev) if he2_idx else None
    global KILL_EDGES, TARGET_DECAY
    KILL_EDGES, TARGET_DECAY = args.kill_edges, args.target_decay
    if KILL_EDGES:
        t_k = time.time()
        n_k = [rebuild_kill_edges_store(st, dev) for st in (tr_store, he_store, he2_store) if st is not None]
        print(f"kill edges rebuilt: {n_k} (train, held, held2) in {time.time() - t_k:.0f}s", flush=True)
    if he2_store:
        print(f"held2 ({args.held2_name}): {len(he2_store.spans)} games / {he2_store.n_steps} steps", flush=True)
    t_f = time.time()
    init = Path(args.init) if args.init else None
    if init and (init if init.is_dir() else init.parent).joinpath("stats.pt").exists():
        sd = torch.load((init if init.is_dir() else init.parent) / "stats.pt", map_location="cpu", weights_only=False)
        stats = Stats(**sd["stats"])            # same scales the checkpoint was trained with
    else:
        stats = fit_stats(tr_store)
    print(f"decompress + store {t_f - t_load:.1f}s, fit_stats {time.time() - t_f:.1f}s", flush=True)
    print(f"train {len(tr_store.spans)} games / {tr_store.n_steps} steps ({tr_store.nbytes / 1e9:.2f} GB on {dev}); "
          f"held-out {len(he_store.spans)} games / {he_store.n_steps} steps; loaded in {time.time() - t_load:.0f}s",
          flush=True)
    fz = Featurizer(dev)
    kind = torch.from_numpy(KIND).to(dev)
    global BF16, GATE_LOSS_WEIGHT, GATE_VALUE_NORM, HERO_WALK_WEIGHT
    if args.baseline:
        args.gate, args.gate_fields = True, ",".join(BASELINE_GATE_FIELDS)
        args.gate_loss_weight, args.gate_value_norm, args.hero_walk_weight = 0.2, "all", 10.0
    BF16 = args.bf16
    GATE_LOSS_WEIGHT, GATE_VALUE_NORM = args.gate_loss_weight, args.gate_value_norm
    HERO_WALK_WEIGHT = args.hero_walk_weight
    EFFECT_WEIGHT.update({k: float(v) for k, v in (kv.split("=") for kv in args.effect_weights.split(",") if kv)})
    gate_fields = [f for f in args.gate_fields.split(",") if f] or None
    global TOWER_HEALTH_WEIGHT, TOWER_CONSIST, TOWER_EVENT_P, STRATA, TARGET_WEIGHT, LIST_WEIGHT
    TARGET_WEIGHT, LIST_WEIGHT = args.target_weight, args.list_weight
    TOWER_HEALTH_WEIGHT, TOWER_CONSIST, TOWER_EVENT_P = args.tower_health_weight, args.tower_consist, args.tower_event_p
    STRATA = parse_strata(args.strata) if args.strata else None
    if args.attack_mode:
        from . import data as _data
        _data.ATTACK_MODE = args.attack_mode
    model = TokenWM(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate, attack_mode=args.attack_mode,
                    predict_targets=args.predict_targets, target_dist=args.target_dist,
                    list_state=args.list_state, direct_damage=args.direct_damage, grid_cnn=args.grid_cnn,
                    gate_fields=gate_fields).to(dev)
    model.set_change_prior(stats.change_rate)
    if init:
        missing, unexpected = model.load_state_dict(torch.load(init / "model.pt" if init.is_dir() else init,
                                                                map_location=dev), strict=False)
        assert not unexpected and all(k.startswith(("atk_in.", "tq.", "tk.", "t_none", "t_dist", "agg_in.", "lq.", "lk.", "l_tab", "dd_", "h_scale", "g_cnn.", "g_out.")) for k in missing), (missing, unexpected)
        print(f"continuing from {init}", flush=True)
    model.set_health_scale(stats)             # direct_damage: after the init load (old checkpoints lack h_scale)
    if WORLD > 1:                             # identical start on every GPU
        for v in model.state_dict().values():
            dist.broadcast(v, 0)
    if args.compile:                          # fixed shapes per run (batch, 206 entities): one graph, no recompiles
        model.step = torch.compile(model.step, dynamic=False)
    link = None
    if args.fleet_exp and MAIN:
        from ..fleet_link import FleetLink
        link = FleetLink(args.fleet_exp, args.fleet_rank, args.fleet_world, args.fleet_launch)
        print(f"fleet {args.fleet_exp}: rank {args.fleet_rank}/{args.fleet_world}, average every {args.fleet_every} updates", flush=True)
    # Everything needed to load and roll out this model elsewhere (e.g. the viewer).
    torch.save(dict(stats=vars(stats), config=dict(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate,
                                                   attack_mode=args.attack_mode, predict_targets=args.predict_targets,
                                                   target_dist=args.target_dist, list_state=args.list_state,
                                                   direct_damage=args.direct_damage, grid_cnn=args.grid_cnn,
                                                   gate_fields=gate_fields),
                    train_games=args.train_games, held_games=args.held_games), out / "stats.pt")
    # Off-pod persistence: every file below is uploaded to S3 as soon as it is written
    # (mobawm/s3persist.py; a no-op when the run has no upload policy, e.g. local runs).
    from ..s3persist import flush as persist_flush, persist
    persist(out / "stats.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params", flush=True)
    FREEZE = {  # parameter groups left TRAINABLE per --freeze (everything else frozen)
        "heads": ("reg", "change", "alive_flip", "is_hit", "lane", "team", "atk_in"),
        "gru_heads": ("reg", "change", "alive_flip", "is_hit", "lane", "team", "atk_in", "gru", "mem_in", "ln"),
        "direct": ("dd_val", "dd_gate"),
        "no_embed": ("reg", "change", "alive_flip", "is_hit", "lane", "team", "atk_in", "gru", "mem_in", "ln", "blocks"),
    }
    if args.freeze != "none":
        keep = FREEZE[args.freeze]
        for n, p in model.named_parameters():
            p.requires_grad_(n.split(".")[0] in keep)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze {args.freeze}: {n_tr} of {n_params} parameters trainable ({', '.join(keep)})", flush=True)
    dd = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("dd_")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("dd_")]
    opt = torch.optim.AdamW([g for g in ({"params": rest}, {"params": dd, "lr": args.direct_lr, "weight_decay": 0.0}) if g["params"]],
                            lr=args.lr, weight_decay=0.01)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="moba-wm", name=out.name, config=vars(args) | dict(params=n_params,
                         train_steps=tr_store.n_steps, held_steps=he_store.n_steps))

    logf = open(out / "log.jsonl", "w")

    def emit(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if run:
            run.log({k: v for k, v in rec.items() if k != "it"}, step=rec["it"])

    steps_per_it = args.batch * WORLD * (args.chunk if args.rollout else args.window - args.burn)   # supervised transitions

    def open_loop(it):
        """Held-out transition loss at open-loop horizons (eval_horizons): the key metric."""
        for store, pre, lab in ((he_store, "horizon", "HELD-OUT"),) + (
                ((he2_store, f"{args.held2_name}/horizon", f"HELD-OUT {args.held2_name}"),) if he2_store else ()):
            open_loop_one(it, store, pre, lab)

    def open_loop_one(it, store, pre, lab):
        r = eval_horizons(model, fz, store, stats, kind, dev)
        emit(dict(it=it, epoch=round(it * steps_per_it / tr_store.n_steps, 3),
                  **{f"{pre}/{k}": v for k, v in r.items() if not k.startswith("windows")}))
        hs = [int(k.split("@")[1]) for k in r if k.startswith("loss@")]
        print(f"    {lab} loss by horizon:  " + "  ".join(f"h{h} {r[f'loss@{h}']:.3f}" for h in hs), flush=True)
        print("         hero pos L1 (cells):  " + "  ".join(f"h{h} {r[f'hero_pos_L1@{h}']:.1f}" for h in hs),
              flush=True)
        print("         rubric: circle hit %  " + "  ".join(f"h{h} {r[f'circle_hit_pct@{h}']:.0f}" for h in hs)
              + f"  | teleport {r.get('teleport_pct', float('nan')):.2f}%  jitter {r.get('jitter_pct', float('nan')):.2f}%"
              f"  freeze {r.get('freeze_pct', float('nan')):.2f}%  speed ratio {r.get('speed_ratio', float('nan')):.2f}",
              flush=True)

    ev_ctx = None
    if args.event_eval:
        from .events import index_store as _ix
        he_end = torch.empty(len(he_store.ent), dtype=torch.long); he_start = torch.empty(len(he_store.ent), dtype=torch.long)
        for off, n in he_store.spans:
            he_end[off:off + n + 1] = off + n; he_start[off:off + n + 1] = off
        ev_ctx = dict(idx=_ix(he_store, he_end), start=he_start, end=he_end)

    def run_event_eval(it):
        from .events import event_eval
        r = event_eval(model, fz, he_store, ev_ctx["idx"], ev_ctx["start"], ev_ctx["end"], stats, dev, apply_prediction)
        emit(dict(it=it, **{f"events/{k}": v for k, v in r.items()}))
        rec = "  ".join(f"{k.split('/', 1)[1]} {100 * v:.0f}%" for k, v in r.items() if k.startswith("recall/"))
        print(f"    EVENTS (held-out, open loop 32, +-4 ticks) recall macro {100 * r['recall_macro']:.0f}% | hero pos err "
              f"{r['hero_pos_err_h32']:.2f} | {rec}", flush=True)
        fp = "  ".join(f"{e} {r[f'dream_per_1k/{e}/radiant'] + r[f'dream_per_1k/{e}/dire']:.1f}/"
                       f"{r[f'real_per_1k/{e}/radiant'] + r[f'real_per_1k/{e}/dire']:.1f}"
                       for e in ("tower_destroyed", "hero_death", "level_up", "ability_cast"))
        print(f"    EVENTS on ordinary windows, dream/real per 1k ticks: {fp}", flush=True)

    def run_val(it):
        if args.val_light:                 # towers + held-out dives only (fast)
            if scen_hb is not None:
                se = scen_eval(model, fz, stats, kind, dev, scen_hb)
                emit(dict(it=it, **{f"scen_val/{k}": v for k, v in se.items()}))
                print(f"    TOWER DIVES ({len(scen_hb['died'])} held-out slices): outcome acc {100 * se['outcome_acc']:.0f}% | "
                      f"survivors killed {100 * se['survivor_false_death']:.0f}% | 'dies' precision {100 * se['died_precision']:.0f}% "
                      f"recall {100 * se['died_recall']:.0f}% | tower loss {se['tower_loss']:.3f}", flush=True)
            if ev_ctx is not None:
                run_event_eval(it)
            tw = tower_eval(model, fz, he_store, stats, dev)
            emit(dict(it=it, **{f"tower/{k}": v for k, v in tw.items()}))
            print(f"    TOWERS (held-out windows): open loop 128 precision {100 * tw['ol_precision']:.0f}% recall "
                  f"{100 * tw['ol_recall']:.0f}% ({tw['ol_pred']} predicted / {tw['ol_real']} real)", flush=True)
            return 0.0
        if not args.ol_every:
            open_loop(it)
        vl, vp = validate(model, fz, he_store, stats, kind, dev, args)
        sk = skill_eval(model, fz, he_store, stats, kind, dev, args)
        emit(dict(it=it, **{"val/loss": vl}, **{f"val/{k}": v for k, v in vp.items()},
                  **{f"skill/{k}": v for k, v in sk.items()}))
        if scen_hb is not None:
            se = scen_eval(model, fz, stats, kind, dev, scen_hb)
            emit(dict(it=it, **{f"scen_val/{k}": v for k, v in se.items()}))
            print(f"    TOWER DIVES (held-out scenarios, {len(scen_hb['died'])} slices): loss {se['loss']:.3f} tower loss "
                  f"{se['tower_loss']:.3f} | outcome acc {100 * se['outcome_acc']:.0f}% | 'dies' precision "
                  f"{100 * se['died_precision']:.0f}% recall {100 * se['died_recall']:.0f}% | survivors killed "
                  f"{100 * se['survivor_false_death']:.0f}%", flush=True)
        if ev_ctx is not None:
            run_event_eval(it)
        tw = tower_eval(model, fz, he_store, stats, dev)
        emit(dict(it=it, **{f"tower/{k}": v for k, v in tw.items()}))
        print(f"    TOWERS (held-out, 64 windows): open loop 128 precision {100 * tw['ol_precision']:.0f}% recall "
              f"{100 * tw['ol_recall']:.0f}% ({tw['ol_pred']} predicted / {tw['ol_real']} real) | one step precision "
              f"{100 * tw['onestep_precision']:.0f}% recall {100 * tw['onestep_recall']:.0f}% "
              f"false/fall {tw['onestep_false_per_fall']:.2f}", flush=True)
        kinds = "  ".join(f"{n} {vp[n + '/total']:.3f}" for n in KIND_NAMES)
        print(f"VAL it {it:6d}  loss {vl:.4f}  | {kinds}" + edge_txt(vp) + list_txt(vp), flush=True)
        hero = "  ".join(f"{e} {sk.get(f'hero/{e}/better_than_copy_pct', float('nan')):+.0f}%/"
                         f"{sk.get(f'hero/{e}/action_dependence_pct', float('nan')):+.0f}%"
                         for e in ("walking", "attack", "q", "w", "e", "mana", "health"))
        print(f"    hero better-than-copy / action-dependence: {hero}  | deaths recall {sk['deaths/recall_pct']:.0f}%", flush=True)
        if he2_store:
            vl2, vp2 = validate(model, fz, he2_store, stats, kind, dev, args)
            sk2 = skill_eval(model, fz, he2_store, stats, kind, dev, args)
            n2 = args.held2_name
            emit(dict(it=it, **{f"{n2}/val/loss": vl2}, **{f"{n2}/val/{k}": v for k, v in vp2.items()},
                      **{f"{n2}/skill/{k}": v for k, v in sk2.items()}))
            print(f"VAL {n2} it {it:6d}  loss {vl2:.4f}  | deaths recall {sk2['deaths/recall_pct']:.0f}%", flush=True)
        return vl

    scen = (ScenarioLanes(ScenarioFeeder(args.scenarios, args.scen_batch, args.scen_workers), args.d, dev,
                          args.decay_halflife, args.decay_floor) if args.scenarios else None)
    scen_hb = load_scen_heldout(args.scen_heldout) if args.scen_heldout else None
    lanes = Lanes(tr_store, args.batch, args.d, gen, args.max_span, args.span_mu, args.span_sigma,
                  args.decay_halflife, args.decay_floor, args.rubric_circle, args.rubric_traj) if args.rollout else None
    cfb = CFBatcher(args.cf_logs or Path(args.data).parent / "logs", tr_idx, dev, args.cf_forks, args.cf_k,
                    args.cf_stride, args.cf_workers, burn=4, seed=0, w_effect=args.cf_effect,
                    w_noeffect=args.cf_noeffect, w_affected=args.cf_affected, d=args.d) if args.cf else None
    if cfb and cfb.head is not None:
        opt.add_param_group({"params": list(cfb.head.parameters())})
    # Step 0: the zero-initialized model predicts "no change", so this is the persistence baseline
    # (or, with --init, the starting checkpoint).
    base = run_val(0) if (not args.skip_val0 and MAIN) else 0.0
    if run:
        run.summary["val/persistence_loss"] = base
    ema_sd = None
    if args.ema:      # slow-moving average of every float tensor (buffers copied), warm-up corrected decay
        ema_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def save_ckpt(path, raw_path=None):
        if ema_sd is None:
            torch.save(model.state_dict(), path)
        else:
            torch.save(ema_sd, path)
            if raw_path is not None:
                torch.save(model.state_dict(), raw_path)

    t0 = time.time()
    it, losses, nan_count = 0, [], 0
    part_acc, part_n = {}, 0
    while (not args.minutes or time.time() - t0 < args.minutes * 60) and \
            (not args.max_updates or it < args.max_updates):
        warm = min(1.0, (it + 1) / 100)
        for g in opt.param_groups:                  # warm-up scales each group's own base lr (direct-damage group: --direct-lr)
            g.setdefault("base_lr", g["lr"])
            g["lr"] = g["base_lr"] * warm
        opt.zero_grad()
        if lanes:
            if args.scen_only:
                loss, parts = scen.chunk(model, fz, stats, kind, args.chunk)
                parts = dict(parts, **{"scen/loss": loss})
            else:
                loss, parts = lanes.chunk(model, fz, stats, kind, args.chunk)
            if scen is not None and not args.scen_only:
                sl, sp = scen.chunk(model, fz, stats, kind, args.chunk)
                loss = loss + (args.scen_batch / args.batch if args.scen_weight is None else args.scen_weight) * sl
                parts = dict(parts, **{f"scen/{k}": v for k, v in sp.items()}, **{"scen/loss": sl})
        else:
            batch = tr_store.sample(args.batch, args.window, gen, event_frac=args.event_frac, burn=args.burn)
            loss, parts, _, _ = run_window(model, fz, batch, stats, kind, args.burn, dev)
        loss.backward()
        if cfb:                                     # counterfactual siblings, backpropagated in micro-batches
            cl, cp, cf_n = cfb.loss_backward(model, fz, stats, kind, args.cf_weight, args.cf_micro)
            loss = loss.detach() + args.cf_weight * cl.to(loss.device)
            parts = dict(parts, **{f"cf/{k}": v for k, v in cp.items()}, **{"cf/total": float(cl)})
        if WORLD > 1:                           # average gradients (and the loss) over GPUs
            ps = [q for q in model.parameters() if q.requires_grad]
            flat = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten().float() for q in ps]
                             + [loss.detach().float().reshape(1)])
            dist.all_reduce(flat)
            flat /= WORLD
            o = 0
            for q in ps:
                g = flat[o:o + q.numel()].view_as(q).to(q.dtype)
                if q.grad is None:
                    q.grad = g.clone()
                else:
                    q.grad.copy_(g)
                o += q.numel()
            loss = flat[-1]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if not math.isfinite(loss.item()) or not math.isfinite(gn):
            nan_count += 1
            continue
        opt.step()
        it += 1
        if ema_sd is not None:
            dcy = min(args.ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        ema_sd[k].mul_(dcy).add_(v, alpha=1 - dcy)
                    else:
                        ema_sd[k].copy_(v)
        losses.append(loss.item())
        for k, v in parts.items():
            part_acc[k] = part_acc.get(k, 0.0) + (float(v.detach()) if torch.is_tensor(v) else float(v))
        part_n += 1
        if args.fleet_exp and it % args.fleet_every == 0:
            rnd = it // args.fleet_every
            if link is not None:
                fi = link.average_weights(model, rnd, timeout=args.fleet_timeout, aggregator=0)
                link.heartbeat({"update": it, "round": rnd, "loss": float(np.mean(losses[-args.fleet_every:])),
                                "it_per_s": it / (time.time() - t0), **fi})
                if rnd <= 3 or it % args.log_every == 0:
                    print(f"fleet round {rnd}: averaged {fi['sync_ranks']} pods (wait {fi['sync_wait_s']:.1f}s, "
                          f"from round {fi['sync_from_update']}, hash {fi['weight_hash']})", flush=True)
            if WORLD > 1:                       # this pod's other GPUs take the averaged weights
                for v in model.state_dict().values():
                    dist.broadcast(v, 0)
        if it % args.log_every == 0 and MAIN:
            tp = summarize_parts({k: v / part_n for k, v in part_acc.items()})
            part_acc, part_n = {}, 0
            rec = dict(it=it, **{"train/loss": float(np.mean(losses[-args.log_every:]))},
                       **{f"train/{k}": v for k, v in tp.items()}, grad_norm=gn,
                       elapsed_s=round(time.time() - t0, 1), nan=nan_count,
                       it_per_s=round(it / (time.time() - t0), 2),
                       **(dict(cf_samples_per_s=round(cfb.served * (args.cf_k + 1) / (time.time() - t0)),
                               cf_wait_frac=round(cfb.wait_s / (time.time() - t0), 3)) if cfb else {}))
            kinds = "  ".join(f"{n} {tp[n + '/total']:.3f}" for n in KIND_NAMES)
            ep_reg = it * steps_per_it / tr_store.n_steps
            ep_txt = f"epoch {ep_reg:.3f}" + (f" scen-epoch {scen.seen / len(scen.fd.meta['where']):.3f}" if scen is not None else "")
            rec["epoch"] = ep_reg
            if scen is not None:
                rec["scen_epoch"] = scen.seen / len(scen.fd.meta["where"])
            emit(rec)
            print(f"it {it:6d}  {ep_txt}  loss {rec['train/loss']:.4f}  | {kinds}  | grad {gn:.2f}  "
                  f"{rec['it_per_s']} it/s  {rec['elapsed_s']:.0f}s" + edge_txt(tp) + list_txt(tp) + (
                  "  | direct: val " + " ".join(f"{v:+.2f}" for v in model.dd_val.tolist()) + "  gate " + " ".join(f"{v:+.2f}" for v in model.dd_gate.tolist())
                  if getattr(model, "direct_damage", False) else "") + (f"  | cf loss {tp.get('cf/total', 0):.3f}  "
                  f"{rec['cf_samples_per_s']} cf samples/s  wait {100 * rec['cf_wait_frac']:.0f}%" if cfb else "")
                  + (f"\n    cf sibling loss {tp.get('cf/effect/sibling', 0):.3f}  effect {tp.get('cf/effect/effect', 0):.3f}  "
                     f"noeffect {tp.get('cf/effect/noeffect', 0):.4f}  affected {tp.get('cf/effect/affected', 0):.3f}"
                     f" (F1 {tp.get('cf/metric/affected_f1_pct', 0):.0f}%)  | effect explained: " + "  ".join(
                         f"{h} {tp.get(f'cf/metric/effect_explained_pct/{h}', float('nan')):+.0f}%"
                         for h in ("move_y", "move_x", "attack", "q", "w", "e")) if cfb and cfb.aux else ""),
                  flush=True)
        if it % args.val_every == 0 and MAIN:
            run_val(it)
            save_ckpt(out / "model.pt")
            for f in ("model.pt", "log.jsonl", "train.log"):
                if (out / f).exists():
                    persist(out / f)
        if args.ol_every and it % args.ol_every == 0 and MAIN:
            open_loop(it)
        if args.ckpt_every and it % args.ckpt_every == 0 and MAIN:
            save_ckpt(out / f"model_{it:06d}.pt", out / f"raw_{it:06d}.pt")
            persist(out / f"model_{it:06d}.pt")
    if not LEAD:                             # other GPUs and follower pods: done
        if WORLD > 1:
            dist.destroy_process_group()
        return
    run_val(it)
    save_ckpt(out / "model.pt", out / "raw.pt")
    persist(out / "model.pt")

    # loss slope over the last quarter (per 1000 iterations, in log-loss)
    n = len(losses)
    q = np.array(losses[3 * n // 4:])
    slope = float(np.polyfit(np.arange(len(q)), np.log(q), 1)[0] * 1000) if len(q) > 10 else float("nan")

    print("evaluating ...", flush=True)
    model.eval()
    ev = dict(n_batches=args.eval_batches)
    report = dict(
        args=vars(args), params=n_params, iterations=it, nan_skipped=nan_count,
        train_steps=tr_store.n_steps, held_steps=he_store.n_steps,
        loss_first_100=float(np.mean(losses[:100])), loss_last_100=float(np.mean(losses[-100:])),
        log_loss_slope_per_1k_it_last_quarter=slope,
        # the train store may live in CPU RAM (--store-cpu); its sampler needs a generator on its device
        onestep_train=eval_onestep(model, fz, tr_store, stats, kind, dev,
                                   torch.Generator(device=tr_store.device).manual_seed(1), **ev),
        onestep_heldout=eval_onestep(model, fz, he_store, stats, kind, dev, egen, **ev),
        onestep_heldout_shuffled_actions=eval_onestep(model, fz, he_store, stats, kind, dev, egen, shuffle=True, **ev),
        rollout_train=eval_rollout(model, fz, tr_store, stats, kind, dev,
                                   torch.Generator(device=tr_store.device).manual_seed(1), n=args.rollouts),
        rollout_heldout=eval_rollout(model, fz, he_store, stats, kind, dev, egen, n=args.rollouts),
    )
    if he2_store:
        report[f"onestep_{args.held2_name}"] = eval_onestep(model, fz, he2_store, stats, kind, dev, egen, **ev)
        report[f"rollout_{args.held2_name}"] = eval_rollout(model, fz, he2_store, stats, kind, dev, egen, n=args.rollouts)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    for f in ("report.json", "log.jsonl", "train.log"):
        if (out / f).exists():
            persist(out / f)
    persist_flush()
    if WORLD > 1:
        dist.destroy_process_group()
    if run:
        log_report(run, report)
        run.finish()
    print(f"wrote {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
