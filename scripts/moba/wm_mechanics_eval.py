"""How well does each world model reproduce each game MECHANIC? Event-level scoring of open-loop dreams against the
real game, on the current policy's real games (collected after every evaluated checkpoint was taken).

Windows: every STRIDE ticks of every game, the dream starts from the real state (BURN real ticks warm its memory) and
runs H ticks open loop, driven by the real recorded actions of all ten heroes. In truth and in the dream, per hero:
  cast_Q/W/E   an ability's cooldown resets (the debugger's rule; the tank's Q by its mana cost), per class
  heal         health rises by >= 250 in one tick without a respawn (support W area heal, tank W self heal)
  displaced    the hero moves 3-12 cells (L1) in one tick without a respawn (assassin blink strike, support hook,
               carry knockback; a hero walks at most 2 cells/tick)
  stunned      stun timer starts (support E, burst E area stun)
  slowed       the move multiplier becomes 0.5 (carry Q / W)
  speed_buff   the move multiplier becomes 2 (assassin E)
  hero_death   a > 12-cell jump with health reset to full (death + same-tick respawn)
  tower_fall   a tower's alive flag goes 1 -> 0
A dream event matches a real one of the same hero (tower) and kind within +-TOL ticks. Reported per model and kind:
real count, dream count, matched; precision = matched / dream, recall = matched / real, ratio = dream / real.
Also per window: hero damage taken (health lost, deaths excluded), dream / real.
Output: results/moba/paper/wm_mechanics.csv (+ _windows.csv).
usage: python scripts/moba/wm_mechanics_eval.py --dense <dir> --models label=run,... [--device cuda]"""
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import moba_viz as V  # noqa: E402
from mobawm import shards as S  # noqa: E402
F = S.F
CLASSES = ("support", "assassin", "burst", "tank", "carry")


def events(fr):
    """dict kind -> list of (step, entity) in frames fr (N, 206, 26)."""
    ev = {}
    c = V.casts(fr); d = V.deaths(fr)
    for k, h, j in zip(*np.nonzero(c)):
        ev.setdefault(f"cast_{'QWE'[j]}_{CLASSES[h % 5]}", []).append((k, h))
    a, b = fr[:-1, :10], fr[1:, :10]
    died = d[1:, :10]
    move = np.abs(b[..., F["x"]] - a[..., F["x"]]) + np.abs(b[..., F["y"]] - a[..., F["y"]])
    rules = {"heal": (b[..., F["health"]] - a[..., F["health"]] >= 250) & ~died,
             "displaced": (move > 3.0) & (move <= 12) & ~died,   # > 2.83: a speed-buffed diagonal step
             "stunned": (a[..., F["stun_timer"]] <= 0) & (b[..., F["stun_timer"]] > 0),
             # move_modifier is a speed multiplier: 1 normal, 0.5 slowed (carry Q/W), 2 the assassin's E speed buff
             "slowed": (np.abs(a[..., F["move_modifier"]] - 0.5) > 0.1) & (np.abs(b[..., F["move_modifier"]] - 0.5) <= 0.1),
             "speed_buff": (a[..., F["move_modifier"]] < 1.5) & (b[..., F["move_modifier"]] >= 1.5)}
    for kind, m in rules.items():
        ev[kind] = [(k + 1, h) for k, h in zip(*np.nonzero(m))]
    ev["hero_death"] = [(k, h) for k, h in zip(*np.nonzero(d[:, :10]))]
    ev["tower_fall"] = [(k, 182 + t) for k, t in zip(*np.nonzero(d[:, 182:]))]
    return ev


@torch.no_grad()
def rollout_batch(model, stats, recs, starts, H, burn, dev):
    """V.rollout (the debugger's open loop) for many start records of one game at once: (B, H+1, 206, 26)."""
    from mobawm.wm.data import Featurizer, apply_prediction
    fz = Featurizer(dev)
    st = recs["state"]
    ent = torch.from_numpy(np.ascontiguousarray(st["entity"], np.float32)).to(dev)
    tick = torch.from_numpy(st["tick"].astype(np.int64)).to(dev)
    cidx = torch.from_numpy(st["creep_idx"].astype(np.int64)).to(dev)
    act = torch.from_numpy(recs["actions"].astype(np.int64)).to(dev)
    s = torch.tensor(starts, device=dev)
    h = torch.zeros(len(starts), S.N_ENTITIES, model.d, device=dev)
    for o in range(-burn, 0):
        _, h = model.step(fz(ent[s + o], tick[s + o], cidx[s + o]), act[s + o], h)
    cur, ct, cc = ent[s].clone(), tick[s].clone(), cidx[s].clone()
    out = [cur.cpu().numpy()]
    for k in range(1, H + 1):
        pred, h = model.step(fz(cur, ct, cc), act[s + k - 1], h)
        cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
        out.append(cur.cpu().numpy())
    return np.stack(out, 1)


def match(real, pred, tol):
    used, tp = set(), 0
    for k, h in pred:
        for i, (k2, h2) in enumerate(real):
            if i not in used and h2 == h and abs(k2 - k) <= tol:
                used.add(i); tp += 1; break
    return tp


ap = argparse.ArgumentParser()
ap.add_argument("--dense", default="data/moba/dyna_eval/dense"); ap.add_argument("--models", required=True)
ap.add_argument("--H", type=int, default=64); ap.add_argument("--stride", type=int, default=200)
ap.add_argument("--burn", type=int, default=8); ap.add_argument("--tol", type=int, default=4)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default="results/moba/paper/wm_mechanics.csv")
a = ap.parse_args()
dev = torch.device(a.device)
files = sorted(Path(a.dense).glob("games.*.mbs"))
tot, win_rows = {}, []
for spec in a.models.split(","):
    label, run = spec.split("=")
    model, stats, _ = V.load_model(Path(run), files[0], dev)
    for f in files:
        D = S.load_dense(f)
        for g in range(len(D.games)):
            recs = D.load_game(g)
            n = len(recs) - 1
            ours_radiant = g % 2 == 0                                  # the collector alternates sides: ours radiant in even games
            starts = list(range(max(a.burn, 200), n - a.H, a.stride))
            if not starts:
                continue
            P = rollout_batch(model, stats, recs, starts, a.H, a.burn, dev)
            for start, pred in zip(starts, P):
                truth = np.ascontiguousarray(recs["state"]["entity"][start:start + a.H + 1], np.float32)
                er, ep = events(truth), events(pred)
                for kind in set(er) | set(ep):
                    r, p = er.get(kind, []), ep.get(kind, [])
                    t = tot.setdefault((label, kind), [0, 0, 0])
                    t[0] += len(r); t[1] += len(p); t[2] += match(r, p, a.tol)
                dmg = lambda fr: float(np.clip(fr[:-1, :10, F["health"]] - fr[1:, :10, F["health"]], 0, None)[~V.deaths(fr)[1:, :10]].sum())
                win_rows.append(dict(model=label, file=f.name, game=g, ours_radiant=ours_radiant, start=start,
                                     dmg_real=dmg(truth), dmg_dream=dmg(pred)))
        print(label, f.name, len(win_rows), flush=True)
rows = [dict(model=l, kind=k, real=r, dream=p, matched=m, precision=round(m / p, 4) if p else None,
             recall=round(m / r, 4) if r else None, ratio=round(p / r, 3) if r else None) for (l, k), (r, p, m) in sorted(tot.items())]
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
with open(out.with_name(out.stem + "_windows.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(win_rows[0])); w.writeheader(); w.writerows(win_rows)
print("wrote", out, len(rows), "rows,", len(win_rows), "windows")
