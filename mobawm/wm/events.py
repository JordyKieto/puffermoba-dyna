"""Key game events, both sides: one set of detectors for the event index (stratified
training), the per-event evaluation suite, and nothing else.

detect(a, b): entity states at t and t+1, (..., 206, F) -> {event: (..., 2) bool}, where the
last axis is the side the event belongs to (0 radiant, 1 dire):

  win                 an ancient falls; side = the winner
  tower_destroyed     a tower (or ancient) falls; side = the tower's team (it lost it)
  tower_damaged       a tower loses health; side = the tower's team
  hero_attacks_tower  a hero's target is an enemy tower (needs real target_pid; index only)
  hero_death          a hero jumps > 12 cells with health reset to full; side = the hero's team
  fight               a hero's target is an enemy hero (index only)
  ability_cast        a hero's Q/W/E cooldown jumps up
  level_up            a hero's level goes up
  farming             a hero's target is a creep (index only)
  hero_hit            a hero is hit (is_hit)

Heroes 0-4 are radiant, 5-9 dire. Targets are the recorded target_pid, which only real
states have (rule R9 blanks it in the dream), so the evaluation suite uses EVAL_EVENTS.
See docs/moba/EVENT_SAMPLING.md.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import shards as S

F = S.F
EVENTS = ("win", "tower_destroyed", "tower_damaged", "hero_attacks_tower", "hero_death", "fight",
          "ability_cast", "level_up", "farming", "hero_hit")
EVAL_EVENTS = ("win", "tower_destroyed", "tower_damaged", "hero_death", "ability_cast", "level_up", "hero_hit")
KIND = torch.from_numpy(np.array([0] * 10 + [1] * 100 + [2] * 72 + [3] * 24))
HERO_SIDE = torch.tensor([0] * 5 + [1] * 5)


def detect(a, b):
    """a, b: (..., 206, F) torch float states at t and t+1 -> {event: (..., 2) bool per side}."""
    out = {}
    side_mask = torch.stack([HERO_SIDE == 0, HERO_SIDE == 1], -1).to(a.device)          # (10, 2)
    per_side = lambda x: (x.unsqueeze(-1) & side_mask).any(-2)                          # (..., 10) -> (..., 2)
    ta0, ta1 = a[..., 182:, F["alive"]] > 0, b[..., 182:, F["alive"]] > 0
    tteam = a[..., 182:, F["team"]].long().clamp(0, 1)
    tside = torch.stack([tteam == 0, tteam == 1], -1)                                   # (..., 24, 2)
    fell = ta0 & ~ta1
    out["tower_destroyed"] = (fell.unsqueeze(-1) & tside).any(-2)
    dmg = ta0 & ta1 & (b[..., 182:, F["health"]] < a[..., 182:, F["health"]] - 1e-3)
    out["tower_damaged"] = (dmg.unsqueeze(-1) & tside).any(-2)
    anc = fell[..., 22:24]                                                               # 204 dire ancient, 205 radiant
    out["win"] = torch.stack([anc[..., 0], anc[..., 1]], -1)                            # radiant wins when dire's ancient falls
    h0, h1 = a[..., :10, :], b[..., :10, :]
    jump = (h1[..., F["x"]] - h0[..., F["x"]]).abs() + (h1[..., F["y"]] - h0[..., F["y"]]).abs()
    death = (jump > 12) & (h1[..., F["health"]] >= h1[..., F["max_health"]] - 0.5) & \
        (h0[..., F["health"]] < h0[..., F["max_health"]] - 0.5)
    out["hero_death"] = per_side(death)
    cast = torch.zeros_like(death)
    for ab in ("q_timer", "w_timer", "e_timer"):
        cast = cast | (h1[..., F[ab]] > h0[..., F[ab]] + 0.5)
    out["ability_cast"] = per_side(cast)
    out["level_up"] = per_side(h1[..., F["level"]] > h0[..., F["level"]] + 0.5)
    out["hero_hit"] = per_side(h1[..., F["is_hit"]] > 0)
    tgt = b[..., :10, F["target_pid"]].long()                                            # attacks made during t -> t+1
    has = tgt >= 0
    tk = KIND.to(a.device)[tgt.clamp(0, 205)]
    tteam_all = torch.gather(b[..., F["team"]], -1, tgt.clamp(0, 205)).long()
    enemy = tteam_all != HERO_SIDE.to(a.device)
    out["hero_attacks_tower"] = per_side(has & (tk == 3) & enemy)
    out["fight"] = per_side(has & (tk == 0) & enemy)
    out["farming"] = per_side(has & (tk == 1))
    return out


def index_store(store, row_end, chunk=100_000):
    """Event index of a Store: {event: [rows of radiant, rows of dire]} where row r means the
    transition r -> r+1 inside one game. Uses only the fields the detectors need."""
    need = [F[k] for k in ("alive", "x", "y", "health", "max_health", "level", "q_timer", "w_timer", "e_timer",
                           "is_hit", "target_pid", "team")]
    n = len(store.ent) - 1
    rows = {e: [[], []] for e in EVENTS}
    remap = {f: i for i, f in enumerate(need)}
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        blk = store.ent[lo:hi + 1][..., need].float().cpu()
        full = torch.zeros(len(blk), blk.shape[1], len(F))
        for f, i in remap.items():
            full[..., f] = blk[..., i]
        ok = (row_end[lo:hi].cpu() > torch.arange(lo, hi))                               # r+1 in the same game
        ev = detect(full[:-1], full[1:])
        for e, m in ev.items():
            for s in (0, 1):
                idx = torch.nonzero(m[:, s] & ok).flatten() + lo
                rows[e][s].append(idx)
    return {e: [torch.cat(v[0]) if v[0] else torch.zeros(0, dtype=torch.long),
                torch.cat(v[1]) if v[1] else torch.zeros(0, dtype=torch.long)] for e, v in rows.items()}


@torch.no_grad()
def event_eval(model, fz, store, idx, row_start, row_end, stats, dev, apply_prediction, n=32, H=32, burn=8, tol=4, seed=13):
    """Per event and side: recall = share of held-out windows containing the real event where
    the dream (open loop from `burn` real ticks before) also produces it within +-tol ticks;
    false positives per 1k ticks on ordinary (uniform) windows vs the real rate there.
    Also hero position error (cells) at the end of the ordinary windows."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    starts_ok = lambda r: r - burn >= row_start[r]
    wins = []                                               # (start row, event, side, real tick in window or -1)
    for e in EVAL_EVENTS:
        for s in (0, 1):
            rows = idx[e][s]
            if len(rows) == 0:
                continue
            pick = rows[torch.randint(len(rows), (n,), generator=g)]
            off = torch.randint(4, H - tol, (n,), generator=g)
            for r, o in zip(pick.tolist(), off.tolist()):
                st = min(max(r - o, int(row_start[r]) + burn), int(row_end[r]) - H - 1)   # window inside the game
                if st - burn < int(row_start[r]) or not (0 <= r - st < H):
                    continue
                wins.append((st, e, s, r - st))
    L = len(store.ent)
    uni = torch.randint(0, L - H - 2, (4 * n,), generator=g).tolist()
    uni = [u for u in uni if u - burn >= int(row_start[u]) and u + H + 1 <= int(row_end[u])][:2 * n]
    wins += [(u, "uniform", -1, -1) for u in uni]
    res = {}
    fp_pred = {e: [0, 0] for e in EVAL_EVENTS}; fp_real = {e: [0, 0] for e in EVAL_EVENTS}
    pos_err = []
    for lo in range(0, len(wins), 128):
        W = wins[lo:lo + 128]
        st = torch.tensor([w[0] for w in W])
        rr = (st.unsqueeze(1) + torch.arange(-burn, H + 1)).to(store.ent.device)
        ent = store.ent[rr].to(dev).float(); tick = store.tick[rr].to(dev); cidx = store.cidx[rr].to(dev)
        act = store.act[rr[:, :-1]].to(dev).long()
        B = len(W)
        h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
        for k in range(burn):
            _, h = model.step(fz(ent[:, k], tick[:, k], cidx[:, k]), act[:, k], h)
        cur, ct, cc = ent[:, burn].clone(), tick[:, burn].clone(), cidx[:, burn].clone()
        pred_ev = {e: [] for e in EVAL_EVENTS}; real_ev = {e: [] for e in EVAL_EVENTS}
        for k in range(H):
            p, h = model.step(fz(cur, ct, cc), act[:, burn + k], h)
            nxt, ct, cc = apply_prediction(cur, ct, cc, p, stats, fz)
            nxt = torch.nan_to_num(nxt)
            dp = detect(cur, nxt); dr = detect(ent[:, burn + k], ent[:, burn + k + 1])
            for e in EVAL_EVENTS:
                pred_ev[e].append(dp[e]); real_ev[e].append(dr[e])
            cur = nxt
        for e in EVAL_EVENTS:
            P = torch.stack(pred_ev[e], 1).cpu(); Rl = torch.stack(real_ev[e], 1).cpu()    # (B, H, 2)
            for j, w in enumerate(W):
                if w[1] == e:
                    s, t = w[2], w[3]
                    lo_t, hi_t = max(0, t - tol), min(H, t + tol + 1)
                    key = f"{e}/{'radiant' if s == 0 else 'dire'}"
                    hit = bool(P[j, lo_t:hi_t, s].any())
                    res.setdefault(key, []).append(hit)
                elif w[1] == "uniform":
                    for s in (0, 1):
                        fp_pred[e][s] += int(P[j, :, s].sum()); fp_real[e][s] += int(Rl[j, :, s].sum())
        for j, w in enumerate(W):
            if w[1] == "uniform":
                d = (cur[j, :10, F["x"]] - ent[j, burn + H, :10, F["x"]]).abs() + (cur[j, :10, F["y"]] - ent[j, burn + H, :10, F["y"]]).abs()
                pos_err.append(float(d.mean()))
    if was:
        model.train()
    out = {f"recall/{k}": float(np.mean(v)) for k, v in res.items()}
    ticks = max(1, len(uni) * H) / 1000
    for e in EVAL_EVENTS:
        for s, nm in ((0, "radiant"), (1, "dire")):
            out[f"dream_per_1k/{e}/{nm}"] = fp_pred[e][s] / ticks
            out[f"real_per_1k/{e}/{nm}"] = fp_real[e][s] / ticks
    out["hero_pos_err_h32"] = float(np.mean(pos_err)) if pos_err else float("nan")
    rec = [v for k, v in out.items() if k.startswith("recall/")]
    out["recall_macro"] = float(np.mean(rec)) if rec else float("nan")
    return out
