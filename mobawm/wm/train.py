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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .. import shards as S
from .data import (F, KIND, N_KIND, REG_FIELDS, REG_IDX, Featurizer, Stats, Store, apply_prediction,
                   fit_stats, load_store, regen_part, wave_respawns)
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
                    * (HERO_HEALTH_WEIGHT if k == 0 and fname == "health" else 1.0)
                parts[key] = parts.get(key, 0.0) + w * per_field[j]
                if k == 0 and fname in ONLY_HERO_FIELDS:
                    parts[f"only/hero/{fname}"] = per_field[j]
        if can_flip[k]:
            parts[f"{name}/deaths"] = EFFECT_WEIGHT.get("deaths", 1.0) * \
                _wmean(alive_bce[:, sel], torch.ones_like(alive_bce[:, sel], dtype=torch.bool), sw)
        a1 = alive1[:, sel]
        if a1.any():
            parts[f"{name}/got_hit"] = _wmean(hit_bce[:, sel], a1, sw)
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
    return sum(parts.values()), parts


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

    def __init__(self, store, B, d, gen, max_span, span_mu=6.0, span_sigma=3.0, halflife=0.0, floor=0.0):
        self.st, self.B, self.gen, self.max_span = store, B, gen, max_span
        self.span_mu, self.span_sigma = span_mu, span_sigma
        self.halflife, self.floor = halflife, floor
        self.age = torch.zeros(B, dtype=torch.long, device=store.device)   # steps since the last real state
        dev = store.device
        self.row_end = torch.empty(len(store.ent), dtype=torch.long, device=dev)
        for off, n in store.spans:
            self.row_end[off:off + n + 1] = off + n            # terminal row of each game
        self.row = torch.zeros(B, dtype=torch.long, device=dev)
        self.left = torch.zeros(B, dtype=torch.long, device=dev)
        self.h = torch.zeros(B, S.N_ENTITIES, d, device=dev)
        self.cur = self.ct = self.cc = None
        self.new_game(torch.ones(B, dtype=torch.bool, device=dev))

    def span(self, n):
        """Open-loop span lengths: Gaussian in log2(ticks), mean span_mu, std span_sigma, clipped to 1..max_span."""
        z = torch.randn(n, device=self.st.device, generator=self.gen)
        return torch.exp2(self.span_mu + self.span_sigma * z).round().long().clamp(1, self.max_span)

    def sync(self, m):
        """Lanes in mask m take the real state of their current row."""
        r = self.row[m]
        if self.cur is None:
            self.cur, self.ct, self.cc = self.st.ent[self.row].clone(), self.st.tick[self.row].clone(), \
                self.st.cidx[self.row].clone()
        self.cur[m], self.ct[m], self.cc[m] = self.st.ent[r], self.st.tick[r], self.st.cidx[r]
        self.left[m] = self.span(int(m.sum()))
        self.age[m] = 0

    def new_game(self, m):
        st = self.st.starts(1)
        self.row[m] = st[torch.randint(len(st), (int(m.sum()),), device=self.st.device, generator=self.gen)]
        self.h[m] = 0
        self.sync(m)

    def chunk(self, model, fz, stats, kind, C):
        """C autoregressive steps; returns the mean loss (grad through the GRU over the chunk) and parts."""
        total, parts = 0.0, {}
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and self.st.device.type == "cuda") else nullcontext()
        h = self.h
        for _ in range(C):
            act = self.st.act[self.row].long()
            with amp:
                pred, h = model.step(fz(self.cur, self.ct, self.cc), act, h)
            pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
            h = h.float()
            w = None
            if self.halflife > 0:   # step decay: h1 after a real state weighs most, then halves every `halflife`
                w = torch.exp2(-self.age.float() / self.halflife).clamp(min=self.floor)
            l, p = step_loss(pred, self.cur, self.st.ent[self.row + 1], stats, kind, self.ct, self.cc, w=w)
            total = total + l
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v
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
                if done.any():                                  # next game: fresh memory
                    h = torch.where(done[:, None, None], torch.zeros_like(h), h)
                    st = self.st.starts(1)
                    self.row[done] = st[torch.randint(len(st), (int(done.sum()),), device=self.st.device,
                                                      generator=self.gen)]
                    self.sync(done)
        self.h = h.detach()
        return total / C, {k: v / C for k, v in parts.items()}


HORIZONS = (1, 4, 16, 64, 256, 1024, 4096)


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
        for k in range(1, H + 1):
            pred, h = model.step(fz(cur, ct, cc), act[:, burn + k - 1], h)
            if k in hs:
                l, parts = step_loss(pred, cur, ent[:, burn + k], stats, kind, ct, cc)
                sp = summarize_parts({a: float(b) for a, b in parts.items()})
                out[f"loss@{k}"] = float(l)
                out[f"hero_loss@{k}"] = sp["hero/total"]
            cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
            if k in hs:
                out[f"hero_pos_L1@{k}"] = _hero_pos(cur, ent[:, burn + k])
                out[f"windows@{k}"] = n
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
    ap.add_argument("--minutes", type=float, default=6.0, help="time budget (0 = none; stop on --max-updates)")
    ap.add_argument("--max-updates", type=int, default=0, help="stop after this many updates (0 = none)")
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
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    egen = torch.Generator(device=dev); egen.manual_seed(1)

    files = sorted(Path(args.data).glob("games.*.mbs"))
    tr_idx, he_idx = parse_games(args.train_games), parse_games(args.held_games)
    assert not tr_idx & he_idx, "train and held-out games overlap"
    t_load = time.time()
    print(f"loading from {args.data} ({len(files)} files) ...", flush=True)
    tr_store = load_store(files, tr_idx, dev)
    he_store = load_store(files, he_idx, dev)
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
    model = TokenWM(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate,
                    gate_fields=gate_fields).to(dev)
    model.set_change_prior(stats.change_rate)
    if init:
        model.load_state_dict(torch.load(init / "model.pt" if init.is_dir() else init, map_location=dev))
        print(f"continuing from {init}", flush=True)
    # Everything needed to load and roll out this model elsewhere (e.g. the viewer).
    torch.save(dict(stats=vars(stats), config=dict(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate,
                                                   gate_fields=gate_fields),
                    train_games=args.train_games, held_games=args.held_games), out / "stats.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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

    steps_per_it = args.batch * (args.chunk if args.rollout else args.window - args.burn)   # supervised transitions

    def open_loop(it):
        """Held-out transition loss at open-loop horizons (eval_horizons): the key metric."""
        r = eval_horizons(model, fz, he_store, stats, kind, dev)
        emit(dict(it=it, epoch=round(it * steps_per_it / tr_store.n_steps, 3),
                  **{f"horizon/{k}": v for k, v in r.items() if not k.startswith("windows")}))
        hs = [int(k.split("@")[1]) for k in r if k.startswith("loss@")]
        print("    HELD-OUT loss by horizon:  " + "  ".join(f"h{h} {r[f'loss@{h}']:.3f}" for h in hs), flush=True)
        print("         hero pos L1 (cells):  " + "  ".join(f"h{h} {r[f'hero_pos_L1@{h}']:.1f}" for h in hs),
              flush=True)

    def run_val(it):
        if not args.ol_every:
            open_loop(it)
        vl, vp = validate(model, fz, he_store, stats, kind, dev, args)
        sk = skill_eval(model, fz, he_store, stats, kind, dev, args)
        emit(dict(it=it, **{"val/loss": vl}, **{f"val/{k}": v for k, v in vp.items()},
                  **{f"skill/{k}": v for k, v in sk.items()}))
        kinds = "  ".join(f"{n} {vp[n + '/total']:.3f}" for n in KIND_NAMES)
        print(f"VAL it {it:6d}  loss {vl:.4f}  | {kinds}", flush=True)
        hero = "  ".join(f"{e} {sk.get(f'hero/{e}/better_than_copy_pct', float('nan')):+.0f}%/"
                         f"{sk.get(f'hero/{e}/action_dependence_pct', float('nan')):+.0f}%"
                         for e in ("walking", "attack", "q", "w", "e", "mana", "health"))
        print(f"    hero better-than-copy / action-dependence: {hero}  | deaths recall {sk['deaths/recall_pct']:.0f}%", flush=True)
        return vl

    lanes = Lanes(tr_store, args.batch, args.d, gen, args.max_span, args.span_mu, args.span_sigma,
                  args.decay_halflife, args.decay_floor) if args.rollout else None
    # Step 0: the zero-initialized model predicts "no change", so this is the persistence baseline
    # (or, with --init, the starting checkpoint).
    base = run_val(0)
    if run:
        run.summary["val/persistence_loss"] = base
    t0 = time.time()
    it, losses, nan_count = 0, [], 0
    part_acc, part_n = {}, 0
    while (not args.minutes or time.time() - t0 < args.minutes * 60) and \
            (not args.max_updates or it < args.max_updates):
        lr = args.lr * min(1.0, (it + 1) / 100)
        for g in opt.param_groups:
            g["lr"] = lr
        if lanes:
            loss, parts = lanes.chunk(model, fz, stats, kind, args.chunk)
        else:
            batch = tr_store.sample(args.batch, args.window, gen, event_frac=args.event_frac, burn=args.burn)
            loss, parts, _, _ = run_window(model, fz, batch, stats, kind, args.burn, dev)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if not math.isfinite(loss.item()) or not math.isfinite(gn):
            nan_count += 1
            continue
        opt.step()
        it += 1
        losses.append(loss.item())
        for k, v in parts.items():
            part_acc[k] = part_acc.get(k, 0.0) + float(v.detach())
        part_n += 1
        if it % args.log_every == 0:
            tp = summarize_parts({k: v / part_n for k, v in part_acc.items()})
            part_acc, part_n = {}, 0
            rec = dict(it=it, **{"train/loss": float(np.mean(losses[-args.log_every:]))},
                       **{f"train/{k}": v for k, v in tp.items()}, grad_norm=gn,
                       elapsed_s=round(time.time() - t0, 1), nan=nan_count,
                       it_per_s=round(it / (time.time() - t0), 2))
            emit(rec)
            kinds = "  ".join(f"{n} {tp[n + '/total']:.3f}" for n in KIND_NAMES)
            print(f"it {it:6d}  loss {rec['train/loss']:.4f}  | {kinds}  | grad {gn:.2f}  "
                  f"{rec['it_per_s']} it/s  {rec['elapsed_s']:.0f}s", flush=True)
        if it % args.val_every == 0:
            run_val(it)
            torch.save(model.state_dict(), out / "model.pt")
        if args.ol_every and it % args.ol_every == 0:
            open_loop(it)
        if args.ckpt_every and it % args.ckpt_every == 0:
            torch.save(model.state_dict(), out / f"model_{it:06d}.pt")
    run_val(it)
    torch.save(model.state_dict(), out / "model.pt")

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
        onestep_train=eval_onestep(model, fz, tr_store, stats, kind, dev, egen, **ev),
        onestep_heldout=eval_onestep(model, fz, he_store, stats, kind, dev, egen, **ev),
        onestep_heldout_shuffled_actions=eval_onestep(model, fz, he_store, stats, kind, dev, egen, shuffle=True, **ev),
        rollout_train=eval_rollout(model, fz, tr_store, stats, kind, dev, egen, n=args.rollouts),
        rollout_heldout=eval_rollout(model, fz, he_store, stats, kind, dev, egen, n=args.rollouts),
    )
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if run:
        log_report(run, report)
        run.finish()
    print(f"wrote {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
