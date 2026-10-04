"""How well does the world model predict tower deaths? Precision and recall on real games.

Event: a tower or ancient (entities 182-205) goes alive -> dead. For held-out real games,
the recorded actions are fed to the world model in two ways:

  teacher-forced  each tick predicted from the REAL state (one-step); a hit = the same tower
                  falls at the same tick
  open loop       from real start states every H ticks (memory warmed on `burn` real ticks),
                  the model rolls H ticks on its own predictions; a hit = the same tower falls
                  within +-tol ticks of the real fall, inside the window

precision = hits / predicted falls, recall = hits / real falls.

    .venv/bin/python scripts/moba/tower_event_audit.py --wm runs/token_wm/e11_at1500 \\
        --dense data/moba/mix1k/dense/games.0000900.mbs --horizons 128,512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mobawm import shards as S  # noqa: E402
from mobawm.rl.plan_mpc import load_wm  # noqa: E402
from mobawm.wm.data import Featurizer, apply_prediction  # noqa: E402

F = S.F
T0 = 182                                   # first tower entity; 204/205 are the ancients


def falls(alive_prev, alive_next):
    """(..., 24) bool: towers that are alive before and dead after."""
    return (alive_prev > 0) & (alive_next <= 0)


@torch.no_grad()
def teacher_forced(model, stats, fz, ent, tick, cidx, act, dev):
    """Real falls and predicted falls per (tick, tower) for one game, one step at a time."""
    T = len(act)
    h = torch.zeros(1, 206, model.d, device=dev)
    real, pred = [], []
    for t in range(T):
        e = ent[t:t + 1]
        p, h = model.step(fz(e, tick[t:t + 1], cidx[t:t + 1]), act[t:t + 1], h)
        nxt, _, _ = apply_prediction(e, tick[t:t + 1], cidx[t:t + 1], p, stats, fz)
        a0 = e[0, T0:, F["alive"]]
        real.append(falls(a0, ent[t + 1, T0:, F["alive"]]))
        pred.append(falls(a0, torch.nan_to_num(nxt)[0, T0:, F["alive"]]))
    return torch.stack(real).cpu().numpy(), torch.stack(pred).cpu().numpy()          # (T, 24)


@torch.no_grad()
def open_loop(model, stats, fz, ent, tick, cidx, act, dev, H, burn=8):
    """(real events, predicted events) as lists of (window, tower, tick-in-window)."""
    T = len(act)
    starts = list(range(burn, T - H, H))
    if not starts:
        return [], [], 0
    B = len(starts)
    h = torch.zeros(B, 206, model.d, device=dev)
    for k in range(burn):                                    # warm memory on real ticks
        idx = [s - burn + k for s in starts]
        _, h = model.step(fz(ent[idx], tick[idx], cidx[idx]), act[idx], h)
    cur, ct, cc = ent[starts].clone(), tick[starts].clone(), cidx[starts].clone()
    real_ev, pred_ev = [], []
    for k in range(H):
        idx = [s + k for s in starts]
        p, h = model.step(fz(cur, ct, cc), act[idx], h)
        nxt, ct, cc = apply_prediction(cur, ct, cc, p, stats, fz)
        nxt = torch.nan_to_num(nxt)
        pf = falls(cur[:, T0:, F["alive"]], nxt[:, T0:, F["alive"]]).cpu().numpy()
        rf = falls(ent[idx][:, T0:, F["alive"]], ent[[i + 1 for i in idx]][:, T0:, F["alive"]]).cpu().numpy()
        for w, tw in zip(*np.nonzero(pf)):
            pred_ev.append((w, tw, k))
        for w, tw in zip(*np.nonzero(rf)):
            real_ev.append((w, tw, k))
        cur = nxt
    return real_ev, pred_ev, B


def match(real_ev, pred_ev, tol):
    """Greedy one-to-one matching by (window, tower) within tol ticks -> hits."""
    used, hits, lags = set(), 0, []
    for w, tw, k in real_ev:
        cands = [(abs(k - k2), j) for j, (w2, tw2, k2) in enumerate(pred_ev)
                 if w2 == w and tw2 == tw and j not in used and abs(k - k2) <= tol]
        if cands:
            d, j = min(cands)
            used.add(j)
            hits += 1
            lags.append(pred_ev[j][2] - k)
    return hits, lags


def pr(hits, n_pred, n_real):
    return dict(real=n_real, predicted=n_pred, hits=hits,
                precision=hits / n_pred if n_pred else float("nan"), recall=hits / n_real if n_real else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="runs/token_wm/e11_at1500")
    ap.add_argument("--dense", nargs="+", default=["data/moba/mix1k/dense/games.0000900.mbs"])
    ap.add_argument("--horizons", default="128,512")
    ap.add_argument("--tol", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tower-death-rule", type=float, default=None, help="RULE: tower dies iff predicted health <= x * max health")
    ap.add_argument("--out", default="results/moba/dream/tower_event_audit.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.tower_death_rule is not None:
        import mobawm.wm.data as _wd
        _wd.TOWER_DEATH_RULE = a.tower_death_rule if a.tower_death_rule > 0 else 1e-9
    torch.set_num_threads(8)
    model, stats = load_wm(a.wm, dev)
    fz = Featurizer(dev)
    Hs = [int(x) for x in a.horizons.split(",")]
    tf_real = tf_pred = tf_hit = 0
    tf = dict(real=0, alarms=0, alarm_hit=0, early=0, false=0, caught=0)   # alarm = run of consecutive predicted falls
    ol = {H: dict(real=0, pred=0, hit=0, lags=[], windows=0) for H in Hs}
    games = 0
    t0 = time.time()
    for path in a.dense:
        d = S.load_dense(path)
        for gi in range(len(d.games)):
            r = d.load_game(gi)
            ent = torch.from_numpy(r["state"]["entity"].astype(np.float32)).to(dev)
            tick = torch.from_numpy(r["state"]["tick"].astype(np.int64)).to(dev)
            cidx = torch.from_numpy(r["state"]["creep_idx"].astype(np.int64)).to(dev)
            act = torch.from_numpy(r["actions"][:-1].astype(np.int64)).to(dev)
            rf, pf = teacher_forced(model, stats, fz, ent, tick, cidx, act, dev)
            tf_real += int(rf.sum()); tf_pred += int(pf.sum()); tf_hit += int((rf & pf).sum())
            real_t = {tw: np.nonzero(rf[:, tw])[0] for tw in range(24)}
            for tw in range(24):
                ticks = np.nonzero(pf[:, tw])[0]
                runs = np.split(ticks, np.nonzero(np.diff(ticks) > 1)[0] + 1) if len(ticks) else []
                rt = real_t[tw]
                for run in runs:
                    tf["alarms"] += 1
                    lo, hi = run[0], run[-1]
                    if any(lo <= x <= hi for x in rt):
                        tf["alarm_hit"] += 1                          # the run contains the real fall tick
                    elif any(0 < x - hi <= a.tol for x in rt):
                        tf["early"] += 1                              # ends at most tol ticks before the real fall
                    else:
                        tf["false"] += 1
                tf["real"] += len(rt)
                tf["caught"] += sum(1 for x in rt if x in set(ticks.tolist()))
            for H in Hs:
                re, pe, B = open_loop(model, stats, fz, ent, tick, cidx, act, dev, H)
                hits, lags = match(re, pe, a.tol)
                o = ol[H]
                o["real"] += len(re); o["pred"] += len(pe); o["hit"] += hits; o["lags"] += lags; o["windows"] += B
            games += 1
            print(f"game {r['game_index'][0]}: {len(act)} ticks | teacher-forced: real {tf_real} caught {tf['caught']} | "
                  f"alarms {tf['alarms']} (on the fall tick {tf['alarm_hit']}, early {tf['early']}, false {tf['false']}) | " + " | ".join(f"open loop {H}: real {ol[H]['real']} pred {ol[H]['pred']} hits {ol[H]['hit']}"
                                                     for H in Hs) + f"  ({time.time() - t0:.0f}s)", flush=True)
    res = dict(wm=a.wm, games=games, tol=a.tol, teacher_forced=pr(tf_hit, tf_pred, tf_real),
               teacher_forced_alarms=dict(tf, precision_exact=tf["alarm_hit"] / max(tf["alarms"], 1),
                                          precision_incl_early=(tf["alarm_hit"] + tf["early"]) / max(tf["alarms"], 1),
                                          recall=tf["caught"] / max(tf["real"], 1)),
               open_loop={H: dict(**pr(o["hit"], o["pred"], o["real"]), windows=o["windows"],
                                  mean_abs_lag=float(np.mean(np.abs(o["lags"]))) if o["lags"] else None) for H, o in ol.items()})
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
