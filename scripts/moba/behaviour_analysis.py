"""How does a policy play the REAL game, and does its team cooperate? Plays real games (as radiant, vs the shipped net)
with the arena's exact controllers (scripts/moba/record_clips.py Game: same library, same sampling), records every hero
every tick, and measures:

  team shape   mean pairwise distance of our live heroes; share of ticks with >= 3 of them within 8 cells of each other
  focus fire   of the ticks where >= 2 of our heroes attack enemy heroes, the share where >= 2 attack the SAME hero;
               and the same for towers
  pushes       our heroes within 8 cells of each enemy tower as it falls
  abilities    casts that LAND per class and ability (cooldown resets; the tank's Q by its mana), per hero-minute
  support      support W casts (area heal) with >= 1 ally within 5 cells
  combos       enemy hero deaths with that hero stunned in the previous 20 ticks
  map          where our heroes are (64 x 64 occupancy, for heatmaps)
For comparison: --policy net plays the shipped net on our side (the net against itself).
Output: results/moba/paper/behaviour.jsonl (one row per policy x game) and behaviour_maps.npz.
usage: python scripts/moba/behaviour_analysis.py --label ppol4_u69 --policy <path|net> --seeds 70000000-70000019"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/moba")); sys.path.insert(0, str(ROOT))
import record_clips as RC  # noqa: E402

CLS = ("support", "assassin", "burst", "tank", "carry")
X, Y, HP, MHP, TGT, MANA, STUN, Q, W, E = 3, 4, 5, 6, 7, 10, 11, 13, 14, 15
TOW0 = RC.TOWER_OFFSET


class NetGame(RC.Game):
    """The shipped net on both sides (the arena's team net vs net)."""
    def __init__(self, L, seed, max_ticks=6000):
        self.L, self.seed, self.max_ticks = L, seed, max_ticks
        self.act = np.zeros((10, 6), np.float32); self.E = np.zeros((L.mr_num_entities(), L.mr_fields()), np.float32)
        L.ma_start(seed); self.t, self.result = 0, 0

    def step(self):
        self.L.ma_ctrl_actions(0, self.act.ctypes.data)
        self.result = self.L.ma_step(self.act.ctypes.data); self.t += 1
        return self.done


def play(L, policy, seed):
    g = NetGame(L, seed) if policy == "net" else RC.Game(L, policy, seed, "radiant")
    keep = list(range(10)) + list(range(TOW0, TOW0 + 24))
    S, A = [g.entities()[keep].copy()], []
    while not g.done:
        g.step(); S.append(g.entities()[keep].copy()); A.append(g.act[:10, 3:6].copy())
    return np.stack(S), np.stack(A), {1: "radiant", 2: "dire"}.get(g.result, "none")


def analyse(S, A, winner):
    ours, theirs = range(0, 5), range(5, 10)
    T = len(S) - 1
    a, b = S[:-1], S[1:]
    move = np.abs(b[:, :10, X] - a[:, :10, X]) + np.abs(b[:, :10, Y] - a[:, :10, Y])
    died = (move > 12) & (b[:, :10, HP] >= b[:, :10, MHP] - 0.5) & (a[:, :10, HP] < a[:, :10, MHP] - 0.5)
    cast = np.zeros((T, 10, 3), bool)
    for j, f in enumerate((Q, W, E)):
        cast[:, :, j] = (b[:, :10, f] - a[:, :10, f] > 3) & ~died
    tank_q = (b[:, :10, MANA] - a[:, :10, MANA] <= -2.5) & ~cast[:, :, 1] & ~cast[:, :, 2] & ~died
    cast[:, 3, 0] |= tank_q[:, 3]
    xy = S[:, :10, X:Y + 1]
    d = np.linalg.norm(xy[:, :5, None] - xy[:, None, :5], axis=-1)                      # (T+1, 5, 5) ours
    iu = np.triu_indices(5, 1)
    spread = float(d[:, iu[0], iu[1]].mean())
    grouped = float(np.mean([(np.sort((d[t] < 8).sum(1))[-1] >= 3) for t in range(len(d))]))
    tgt = S[:, :5, TGT].astype(int)
    on_hero = [[x for x in row if 5 <= x < 10] for row in tgt]
    on_tower = [[x for x in row if x >= TOW0] for row in tgt]
    multi_h = [r for r in on_hero if len(r) >= 2]; multi_t = [r for r in on_tower if len(r) >= 2]
    focus_h = float(np.mean([len(set(r)) < len(r) for r in multi_h])) if multi_h else float("nan")
    focus_t = float(np.mean([len(set(r)) < len(r) for r in multi_t])) if multi_t else float("nan")
    falls = [t for t in range(1, len(S)) for k in range(24) if S[t - 1, 10 + k, 0] and not S[t, 10 + k, 0] and S[0, 10 + k, 2] == 1]
    at_fall = []
    for t in range(1, len(S)):
        for k in range(24):
            if S[t - 1, 10 + k, 0] and not S[t, 10 + k, 0] and S[0, 10 + k, 2] == 1:
                tx, ty = S[0, 10 + k, X], S[0, 10 + k, Y]
                at_fall.append(int(max(sum(np.hypot(S[s, r, X] - tx, S[s, r, Y] - ty) < 8 for r in ours) for s in range(max(0, t - 30), t))))
    heals = cast[:, 0, 1]                                                                   # support W
    heal_allies = [int(sum(np.hypot(*(a[t, r, X:Y + 1] - a[t, 0, X:Y + 1])) < 5 for r in range(1, 5))) for t in np.nonzero(heals)[0]]
    stunned = (a[:, :10, STUN] <= 0) & (b[:, :10, STUN] > 0)
    enemy_deaths = [(t, h) for t, h in zip(*np.nonzero(died)) if h >= 5]
    combos = sum(stunned[max(0, t - 20):t + 1, h].any() for t, h in enemy_deaths)
    minutes = T / 5 / 60                                                                    # the client's pace: 5 ticks/s
    row = dict(ticks=T, won=int(winner == "radiant"), spread=spread, grouped=grouped, focus_hero=focus_h, focus_tower=focus_t,
               enemy_towers=len(falls), heroes_at_fall=float(np.mean(at_fall)) if at_fall else float("nan"),
               kills=len(enemy_deaths), deaths=int(died[:, :5].sum()), support_heals=int(heals.sum()),
               heal_allies_mean=float(np.mean(heal_allies)) if heal_allies else float("nan"),
               stun_before_kill=float(combos / len(enemy_deaths)) if enemy_deaths else float("nan"))
    for h in ours:
        for j in range(3):
            row[f"cast_{CLS[h]}_{'QWE'[j]}_per_min"] = float(cast[:, h, j].sum() / minutes)
            row[f"press_{CLS[h]}_{'QWE'[j]}_per_min"] = float((A[:, h, j] > 0).sum() / minutes)
    occ = np.zeros((64, 64));
    for r in ours:
        live = S[:, r, 0] > 0
        np.add.at(occ, (np.clip(S[live, r, Y] / 2, 0, 63).astype(int), np.clip(S[live, r, X] / 2, 0, 63).astype(int)), 1)
    return row, occ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", default="70000000-70000019")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    pol = a.policy if a.policy == "net" else str(Path(a.policy).resolve())
    L = RC.lib()
    out = ROOT / "results/moba/paper/behaviour.jsonl"
    maps = ROOT / "results/moba/paper/behaviour_maps.npz"
    M = dict(np.load(maps)) if maps.exists() else {}
    occ_sum = np.zeros((64, 64))
    for seed in range(lo, hi + 1):
        S, A, winner = play(L, pol, seed)
        row, occ = analyse(S, A, winner)
        occ_sum += occ
        with out.open("a") as f:
            f.write(json.dumps(dict(label=a.label, seed=seed, **row)) + "\n")
        print(a.label, seed, winner, row["ticks"], f"grouped {row['grouped']:.2f} focus {row['focus_hero']:.2f}", flush=True)
    M[a.label] = occ_sum / occ_sum.sum()
    np.savez_compressed(maps, **M)
