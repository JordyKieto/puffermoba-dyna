"""Real-game evaluation: a dream-trained policy plays one team of the actual C game.

The other team is the shipped PufferNet ("net", the policy that played the
corpus) or the handwritten bots ("script"). Our team sees the game's own ego
observations (the same bytes mobawm.obs reproduces in the dream) and acts through
the same 6 heads. Games run in parallel processes (build/libmobaarena).

    .venv/bin/python -m mobawm.rl.arena --policy runs/dream_ppo/sweep1_gpu/lr3e-4_mb2/policy.pt --games 10

Teams compared, same seeds and sides:
  ppo     the given checkpoint (actions sampled, as trained)
  random  the same network untrained (uniform actions): the floor
  net     the shipped net playing our side: the reference
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "build" / ("libmobaarena.so" if sys.platform.startswith("linux") else "libmobaarena.dylib")
WEIGHTS = ROOT / "third_party/pufferlib/resources/moba/moba_weights.bin"
CTRL = {"net": 0, "script": 1}
LOG_FIELDS = ("episode_return", "reward_death", "reward_xp", "reward_distance", "reward_tower", "level", "kills",
              "deaths", "damage_dealt", "damage_received", "healing_dealt", "healing_received", "creeps_killed",
              "neutrals_killed", "towers_killed", "usage_auto", "usage_q", "usage_w", "usage_e")


def _lib():
    L = ctypes.CDLL(str(LIB))
    P = ctypes.c_void_p
    L.ma_init.argtypes = [ctypes.c_char_p]
    L.ma_start.argtypes = [ctypes.c_uint64]
    L.ma_obs.argtypes = [P]
    L.ma_ctrl_actions.argtypes = [ctypes.c_int, P]
    L.ma_step.argtypes = [P]
    L.ma_rewards.argtypes = [P]
    L.ma_logs.argtypes = [P]
    if L.ma_init(str(WEIGHTS).encode()) != 0:
        raise SystemExit(f"could not load {WEIGHTS}")
    return L


def play(job):
    """One game. job = (team, policy_path, opponent, our_side, seed, max_ticks)."""
    team, policy_path, opp, side, seed, max_ticks = job
    torch.set_num_threads(1)
    L = _lib()
    pol = None
    if team in ("ppo", "random"):
        from .dream_ppo import Policy
        pol = Policy.load(policy_path) if team == "ppo" else Policy()
        pol.eval()
        if pol.recurrent:                                           # memory per hero, carried tick to tick
            pol.reset_state(5, torch.device("cpu"))
    ours = slice(0, 5) if side == "radiant" else slice(5, 10)
    obs = np.zeros((10, 510), np.uint8)
    act = np.zeros((10, 6), np.float32)
    ctrl = np.zeros((10, 6), np.float32)
    netact = np.zeros((10, 6), np.float32)
    rew = np.zeros(10, np.float32)
    total = np.zeros(10, np.float64)
    L.ma_start(seed)
    result, t = 0, 0
    g = torch.Generator().manual_seed(seed)
    while not result and t < max_ticks:
        L.ma_ctrl_actions(CTRL[opp], ctrl.ctypes.data)             # the opponent's controller
        act[:] = ctrl
        if team == "net":                                           # the shipped net on our side
            if opp != "net":
                L.ma_ctrl_actions(CTRL["net"], netact.ctypes.data)
                act[ours] = netact[ours]
        else:
            L.ma_obs(obs.ctypes.data)
            with torch.no_grad():
                x = torch.from_numpy(obs[ours].copy())
                logits, _ = pol.step(x) if pol.recurrent else pol(x)
                a = torch.stack([torch.multinomial(torch.softmax(lg, -1), 1, generator=g).squeeze(-1)
                                 for lg in logits], -1)
            act[ours] = a.numpy().astype(np.float32)
        result = L.ma_step(act.ctypes.data)
        L.ma_rewards(rew.ctypes.data)
        total += rew
        t += 1
    logs = np.zeros((10, 19), np.float32)
    L.ma_logs(logs.ctypes.data)
    win = {1: "radiant", 2: "dire", 3: "both"}.get(result, "none")
    return dict(team=team, opp=opp, side=side, seed=seed, ticks=t, winner=win, won=win == side,
                lost=win in ("radiant", "dire") and win != side, return_ours=float(total[ours].mean()),
                return_theirs=float(np.delete(total, np.r_[ours]).mean()),
                **{f"ours_{k}": float(logs[ours, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                **{f"theirs_{k}": float(np.delete(logs, np.r_[ours], 0)[:, i].mean()) for i, k in enumerate(LOG_FIELDS)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--games", type=int, default=10, help="per (team, opponent, side)")
    ap.add_argument("--teams", default="ppo,random,net")
    ap.add_argument("--opponents", default="net,script")
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    jobs = [(team, args.policy, opp, side, 50_000_000 + i, args.max_ticks)
            for team in args.teams.split(",") for opp in args.opponents.split(",")
            for side in ("radiant", "dire") for i in range(args.games)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers or max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    print(f"{len(rows)} games in {time.time() - t0:.0f}s   (seeds 50000000+, max {args.max_ticks} ticks)\n")
    if args.out:
        import json
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    cols = [("games", lambda g: len(g)), ("win%", lambda g: 100 * np.mean([r["won"] for r in g])),
            ("loss%", lambda g: 100 * np.mean([r["lost"] for r in g])),
            ("timeout%", lambda g: 100 * np.mean([r["winner"] == "none" for r in g])),
            ("ticks", lambda g: np.mean([r["ticks"] for r in g])),
            ("ret ours", lambda g: np.mean([r["return_ours"] for r in g])),
            ("ret theirs", lambda g: np.mean([r["return_theirs"] for r in g]))] + \
        [(k, (lambda k: lambda g: np.mean([r[f"ours_{k}"] for r in g]))(k))
         for k in ("kills", "deaths", "towers_killed", "creeps_killed", "level", "usage_q", "usage_w", "usage_e")]
    print(f"{'team':<8}{'vs':<8}" + "".join(f"{n:>11}" for n, _ in cols))
    for team in args.teams.split(","):
        for opp in args.opponents.split(","):
            g = [r for r in rows if r["team"] == team and r["opp"] == opp]
            print(f"{team:<8}{opp:<8}" + "".join(f"{f(g):>11.2f}" for _, f in cols))

if __name__ == "__main__":
    main()
