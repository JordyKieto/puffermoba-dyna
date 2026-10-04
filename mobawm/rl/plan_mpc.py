"""World-model planner (MPC) for a whole team, played in the REAL game.

For each real game our team (5 heroes, one side) is controlled by lookahead search in
the world model; the other side is the shipped PufferNet or the scripted bots.

Every `--replan` ticks, per game:
  1. the world model's memory has been kept in sync by stepping it on the REAL states and
     actions every tick (teacher-forced tracking);
  2. K coordinated team plans are proposed -- a command (6 action heads) per hero, held for
     the lookahead:
       branch 0      what the shipped net would do now (the baseline),
       branch 1      the plan executed last round (temporal consistency),
       coordinated   the whole team shares one move direction and attack mode,
       mixed         per-hero random commands;
     abilities are pressed with probability --p-cast per hero;
  3. all G x K branches roll H ticks forward in the world model at once, from the real
     state; the enemy keeps its last real action in every branch (cheap; no observation
     rebuild inside the search);
  4. each branch is scored by our team's summed game reward over the lookahead
     (death -0.164, XP x 0.0067, tower +0.642: mobawm.rl.dream_ppo.reward), and the best
     branch's command is executed in the real game for the next `--replan` ticks
     (branch 0 executes the net's live actions).

    .venv/bin/python -m mobawm.rl.plan_mpc --smoke            # branches x horizon throughput table
    .venv/bin/python -m mobawm.rl.plan_mpc --games-per-cell 30 --branches 32 --horizon 16 --replan 4
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from .. import shards as S  # noqa: E402
from ..wm.data import Featurizer, apply_prediction  # noqa: E402
from .dream_ppo import reward  # noqa: E402

F = S.F
HEADS = S.ACTION_SIZES
WEIGHTS = ROOT / "third_party/pufferlib/resources/moba/moba_weights.bin"
LOG_FIELDS = ("episode_return", "reward_death", "reward_xp", "reward_distance", "reward_tower", "level", "kills",
              "deaths", "damage_dealt", "damage_received", "healing_dealt", "healing_received", "creeps_killed",
              "neutrals_killed", "towers_killed", "usage_auto", "usage_q", "usage_w", "usage_e")


class Arena:
    """build/libmobaarena: many real games in one process + shipped-net pools."""

    def __init__(self, n_games):
        lib = ROOT / "build" / ("libmobaarena.so" if sys.platform.startswith("linux") else "libmobaarena.dylib")
        L = ctypes.CDLL(str(lib))
        P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        for f, a in (("ma_multi_init", [I]), ("ma_multi_start", [I, U]), ("ma_multi_obs", [I, P]),
                     ("ma_multi_state", [I, P, P, P]), ("ma_multi_script", [I, P]), ("ma_multi_step", [I, P]),
                     ("ma_multi_rewards", [I, P]), ("ma_multi_logs", [I, P]),
                     ("ma_pools_new", [ctypes.c_char_p, I, U]), ("ma_pools_reset_rows", [I, I, I]),
                     ("ma_pools_act", [I, P, P])):
            getattr(L, f).argtypes = a
        L.ma_multi_init(n_games)
        self.L, self.G = L, n_games
        self.pool = L.ma_pools_new(str(WEIGHTS).encode(), n_games * 10, 0)   # the shipped net, all 10 heroes/game
        self._ent = np.zeros((206, 26), np.float32)
        self._t, self._c = np.zeros(1, np.int32), np.zeros(1, np.int32)

    def start(self, g, seed):
        self.L.ma_multi_start(g, seed)
        self.L.ma_pools_reset_rows(self.pool, g * 10, g * 10 + 10)

    def states(self):
        ent = np.zeros((self.G, 206, 26), np.float32)
        tick, cidx = np.zeros(self.G, np.int64), np.zeros(self.G, np.int64)
        for g in range(self.G):
            self.L.ma_multi_state(g, self._ent.ctypes.data, self._t.ctypes.data, self._c.ctypes.data)
            ent[g], tick[g], cidx[g] = self._ent, self._t[0], self._c[0]
        return ent, tick, cidx

    def net_actions(self):
        """(G, 10, 6) the shipped net's actions for every hero of every game (one recurrent step)."""
        obs = np.zeros((self.G * 10, 510), np.uint8)
        o = np.zeros((10, 510), np.uint8)
        for g in range(self.G):
            self.L.ma_multi_obs(g, o.ctypes.data)
            obs[g * 10:g * 10 + 10] = o
        out = np.zeros((self.G * 10, 6), np.float32)
        self.L.ma_pools_act(self.pool, obs.ctypes.data, out.ctypes.data)
        return out.reshape(self.G, 10, 6)

    def script_actions(self, g):
        out = np.zeros((10, 6), np.float32)
        self.L.ma_multi_script(g, out.ctypes.data)
        return out

    def step(self, g, act):
        a = np.ascontiguousarray(act, np.float32)
        return self.L.ma_multi_step(g, a.ctypes.data)

    def rewards(self, g):
        r = np.zeros(10, np.float32)
        self.L.ma_multi_rewards(g, r.ctypes.data)
        return r

    def logs(self, g):
        out = np.zeros((10, 19), np.float32)
        self.L.ma_multi_logs(g, out.ctypes.data)
        return out


def load_wm(run, dev):
    import moba_viz as V
    model, stats, _ = V.load_model(Path(run), ROOT / "data/moba/mix1k/dense/games.0000900.mbs", dev)
    model.requires_grad_(False)
    return model, stats


def propose(net_ours, last_plan, K, p_cast, gen):
    """(G, K, 5, 6) long team commands for our 5 heroes."""
    G = net_ours.shape[0]
    plans = torch.zeros(G, K, 5, 6, dtype=torch.long)
    plans[:, 0] = net_ours                                   # the shipped net's current actions
    plans[:, 1] = last_plan
    n = K - 2
    rnd = lambda hi, *shape: torch.randint(hi, shape, generator=gen)
    coord = torch.rand(G, n, generator=gen) < 0.5            # half the branches: one command for the team
    my, mx, tm = rnd(7, G, n, 5), rnd(7, G, n, 5), rnd(3, G, n, 5)
    my = torch.where(coord[..., None], my[..., :1].expand(-1, -1, 5), my)
    mx = torch.where(coord[..., None], mx[..., :1].expand(-1, -1, 5), mx)
    tm = torch.where(coord[..., None], tm[..., :1].expand(-1, -1, 5), tm)
    casts = (torch.rand(G, n, 5, 3, generator=gen) < p_cast).long()
    plans[:, 2:] = torch.cat([my[..., None], mx[..., None], tm[..., None], casts], -1)
    return plans


@torch.no_grad()
def search(model, stats, fz, ent, tick, cidx, h, ours, enemy_act, plans, H, dev):
    """Roll every (game, branch) H ticks in the world model; return our team's summed reward (G, K)."""
    G, K = plans.shape[:2]
    rep = lambda x: x.repeat_interleave(K, 0)
    cur, ct, cc, hh = rep(ent), rep(tick), rep(cidx), rep(h)
    act = rep(enemy_act).clone()                                                  # (G*K, 10, 6)
    om = rep(ours)                                                                 # (G*K, 10) bool
    idx = om.nonzero()
    act[idx[:, 0], idx[:, 1]] = plans.reshape(G * K, 5, 6).reshape(-1, 6).to(dev)
    total = torch.zeros(G * K, device=dev)
    for _ in range(H):
        pred, hh = model.step(fz(cur, ct, cc), act, hh)
        nxt, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
        nxt = torch.nan_to_num(nxt)
        r, _ = reward(cur, nxt)
        total += (r * om).sum(-1)
        cur = nxt
    return total.view(G, K)


def smoke(args, dev):
    """Throughput: WM steps/s at a range of batch sizes (= games x branches)."""
    model, stats = load_wm(args.wm, dev)
    fz = Featurizer(dev)
    d = S.load_dense(ROOT / "data/moba/mix1k/dense/games.0000900.mbs").load_game(0)
    print(f"device {dev}; one planning round = H world-model steps at batch G*K")
    print(f"{'batch (G*K)':>12}{'ms / WM step':>14}{'branch-ticks/s':>16}")
    for B in (64, 256, 1024, 2048, 4096):
        ent = torch.from_numpy(np.ascontiguousarray(d["state"]["entity"][500:500 + 1].repeat(B, 0), np.float32)).to(dev)
        tick = torch.full((B,), int(d["state"]["tick"][500]), device=dev)
        cidx = torch.full((B,), int(d["state"]["creep_idx"][500]), device=dev)
        act = torch.from_numpy(d["actions"][500:501].repeat(B, 0).astype(np.int64)).to(dev)
        h = torch.zeros(B, 206, model.d, device=dev)
        try:
            with torch.no_grad():
                for i in range(6):
                    if i == 2:
                        torch.cuda.synchronize() if dev.type == "cuda" else None
                        t0 = time.time()
                    pred, h = model.step(fz(ent, tick, cidx), act, h)
                    ent2, _, _ = apply_prediction(ent, tick, cidx, pred, stats, fz)
                    reward(ent, ent2)
            torch.cuda.synchronize() if dev.type == "cuda" else None
            ms = (time.time() - t0) / 4 * 1000
            print(f"{B:>12}{ms:>14.1f}{B / ms * 1000:>16.0f}")
        except torch.OutOfMemoryError:
            print(f"{B:>12}{'OOM':>14}")
            break
    print("per game tick with G games, K branches, H horizon, replan R: cost ~ (H / R) WM steps at batch G*K")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="runs/token_wm/p1_at1000")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--games-per-cell", type=int, default=30, help="games per (opponent, side)")
    ap.add_argument("--opponents", default="net,script")
    ap.add_argument("--parallel", type=int, default=40, help="real games stepped together")
    ap.add_argument("--branches", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--p-cast", type=float, default=0.3)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/plan_mpc/run.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(args.device)
    if args.smoke:
        return smoke(args, dev)

    model, stats = load_wm(args.wm, dev)
    fz = Featurizer(dev)
    jobs = [(opp, side, 50_000_000 + i) for opp in args.opponents.split(",") for side in ("radiant", "dire")
            for i in range(args.games_per_cell)]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")
    gen = torch.Generator().manual_seed(args.seed)
    t_all = time.time()
    for b0 in range(0, len(jobs), args.parallel):
        batch = jobs[b0:b0 + args.parallel]
        G = len(batch)
        A = Arena(G)
        for g, (opp, side, seed) in enumerate(batch):
            A.start(g, seed)
        ours = torch.zeros(G, 10, dtype=torch.bool)
        for g, (_, side, _) in enumerate(batch):
            ours[g, :5] = side == "radiant"
            ours[g, 5:] = side == "dire"
        ours_d = ours.to(dev)
        h = torch.zeros(G, 206, model.d, device=dev)
        plan = torch.zeros(G, 5, 6, dtype=torch.long)
        plan[..., :2] = 3                                     # "stand still" until the first plan
        use_net = torch.zeros(G, dtype=torch.bool)
        running = np.ones(G, bool)
        result = np.zeros(G, int)
        ret = np.zeros((G, 10))
        t, t_search, n_search = 0, 0.0, 0
        chosen = np.zeros(args.branches, int)
        while running.any() and t < args.max_ticks:
            ent, tick, cidx = A.states()
            net = A.net_actions()                              # the net for all 10 heroes of every game
            enemy = np.zeros((G, 10, 6), np.float32)
            for g, (opp, _, _) in enumerate(batch):
                enemy[g] = net[g] if opp == "net" else A.script_actions(g)
            e_t = torch.from_numpy(ent).to(dev)
            tk, ci = torch.from_numpy(tick).to(dev), torch.from_numpy(cidx).to(dev)
            net_ours = torch.from_numpy(net[ours.numpy()].reshape(G, 5, 6).astype(np.int64))
            if t % args.replan == 0:                           # plan
                ts = time.time()
                plans = propose(net_ours, plan, args.branches, args.p_cast, gen)
                score = search(model, stats, fz, e_t, tk, ci, h, ours_d,
                               torch.from_numpy(enemy.astype(np.int64)).to(dev), plans, args.horizon, dev)
                best = score.argmax(1).cpu()
                plan = plans[torch.arange(G), best]
                use_net = best == 0
                np.add.at(chosen, best.numpy(), 1)
                t_search += time.time() - ts
                n_search += 1
            act = enemy.copy()
            mine = np.where(use_net[:, None, None].numpy(), net_ours.numpy(), plan.numpy())
            act[ours.numpy()] = mine.reshape(-1, 6)
            with torch.no_grad():                              # keep the WM memory on the real trajectory
                _, h = model.step(fz(e_t, tk, ci), torch.from_numpy(act.astype(np.int64)).to(dev), h)
            for g in range(G):
                if running[g]:
                    res = A.step(g, act[g])
                    ret[g] += A.rewards(g)
                    if res:
                        running[g], result[g] = False, res
            t += 1
            if t % 500 == 0:
                print(f"  batch {b0 // args.parallel}: tick {t}, {running.sum()}/{G} running, "
                      f"search {t_search / max(n_search, 1):.2f}s/round", flush=True)
        for g, (opp, side, seed) in enumerate(batch):
            lg = A.logs(g)
            om = ours[g].numpy()
            win = {1: "radiant", 2: "dire", 3: "both"}.get(int(result[g]), "none")
            row = dict(team="planner", opp=opp, side=side, seed=seed, ticks=t, winner=win, won=win == side,
                       lost=win in ("radiant", "dire") and win != side, return_ours=float(ret[g][om].mean()),
                       return_theirs=float(ret[g][~om].mean()),
                       **{f"ours_{k}": float(lg[om, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                       **{f"theirs_{k}": float(lg[~om, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                       planner=dict(branches=args.branches, horizon=args.horizon, replan=args.replan,
                                    p_cast=args.p_cast, wm=args.wm))
            f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"batch done: {G} games, branch choice histogram (0 = net baseline, 1 = keep plan): "
              f"{chosen[:4].tolist()} ... others {int(chosen[4:].sum())}; {(time.time() - t_all) / 60:.1f} min",
              flush=True)
    print(f"-> {out}")




# --- real-time (60 fps) anytime planner --------------------------------------------------

def play_realtime(job):
    """One real game with our team driven by an ANYTIME planner that must fit the frame
    budget: every frame does exactly ONE batched world-model call -- row 0 keeps the
    model's memory on the real game, rows 1..K advance the K branches of the current search
    by one tick. After H frames the search is complete, the best branch's command is
    executed from the next frame, and a new search starts from the then-current real state.
    Per-frame compute (game step + shipped net + model call) is timed and returned."""
    seed, side, opp, cfg = job
    torch.set_num_threads(cfg["threads"])
    dev = torch.device("cpu")
    model, stats = load_wm(cfg["wm"], dev)
    fz = Featurizer(dev)
    K, H, budget = cfg["branches"], cfg["horizon"], cfg["budget_ms"]
    gen = torch.Generator().manual_seed(seed)
    A = Arena(1)
    A.start(0, seed)
    ours = np.zeros(10, bool)
    ours[:5] = side == "radiant"
    ours[5:] = side == "dire"
    om = torch.from_numpy(ours)
    h = torch.zeros(1, 206, model.d)
    plan = torch.zeros(5, 6, dtype=torch.long)
    plan[:, :2] = 3
    use_net = True                       # until the first search completes, play the net's action
    search = None
    frame_ms, chosen = [], np.zeros(K, int)
    ret = np.zeros(10)
    result, t = 0, 0
    while not result and t < cfg["max_ticks"]:
        t0 = time.perf_counter()
        ent, tick, cidx = A.states()
        net = A.net_actions()[0]
        enemy = net if opp == "net" else A.script_actions(0)
        net_ours = torch.from_numpy(net[ours].astype(np.int64))
        act = enemy.copy()
        act[ours] = net_ours.numpy() if use_net else plan.numpy()
        e, tk, ci = torch.from_numpy(ent), torch.from_numpy(tick), torch.from_numpy(cidx)
        a_real = torch.from_numpy(act.astype(np.int64))[None]
        if search is None:               # start a search from the current real state
            plans = propose(net_ours[None], plan[None], K, cfg["p_cast"], gen)[0]          # (K, 5, 6)
            ba = torch.from_numpy(enemy.astype(np.int64))[None].repeat(K, 1, 1)
            ba[:, om] = plans
            search = dict(cur=e.repeat(K, 1, 1), ct=tk.repeat(K), cc=ci.repeat(K), hh=h.repeat(K, 1, 1),
                          act=ba, total=torch.zeros(K), step=0, plans=plans)
        s = search
        with torch.no_grad():
            pred, hb = model.step(fz(torch.cat([e, s["cur"]]), torch.cat([tk, s["ct"]]), torch.cat([ci, s["cc"]])),
                                  torch.cat([a_real, s["act"]]), torch.cat([h, s["hh"]]))
            h = hb[:1]
            sp = {k: (v[1:] if torch.is_tensor(v) and v.dim() and v.shape[0] == K + 1 else v) for k, v in pred.items()}
            nxt, s["ct"], s["cc"] = apply_prediction(s["cur"], s["ct"], s["cc"], sp, stats, fz)
            nxt = torch.nan_to_num(nxt)
            r, _ = reward(s["cur"], nxt)
            s["total"] += (r * om).sum(-1)
            s["cur"], s["hh"] = nxt, hb[1:]
            s["step"] += 1
        if s["step"] == H:               # search done: commit the best branch
            best = int(s["total"].argmax())
            plan, use_net = s["plans"][best], best == 0
            chosen[best] += 1
            search = None
        res = A.step(0, act)
        ret += A.rewards(0)
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if res:
            result = res
        t += 1
    lg = A.logs(0)
    fm = np.array(frame_ms)
    win = {1: "radiant", 2: "dire", 3: "both"}.get(int(result), "none")
    return dict(team="planner", opp=opp, side=side, seed=seed, ticks=t, winner=win, won=win == side,
                lost=win in ("radiant", "dire") and win != side, return_ours=float(ret[ours].mean()),
                return_theirs=float(ret[~ours].mean()),
                **{f"ours_{k}": float(lg[ours, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                **{f"theirs_{k}": float(lg[~ours, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                frame_ms_p50=float(np.percentile(fm, 50)), frame_ms_p95=float(np.percentile(fm, 95)),
                frame_ms_max=float(fm.max()), frames_over_budget_pct=float(100 * (fm > budget).mean()),
                branch_chosen=chosen.tolist(), planner=cfg)


def realtime_main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="runs/token_wm/p1_at1000")
    ap.add_argument("--games-per-side", type=int, default=60)
    ap.add_argument("--opponent", default="net", choices=("net", "script"))
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=16, help="frames per search = lookahead ticks")
    ap.add_argument("--budget-ms", type=float, default=1000 / 60)
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads per game process")
    ap.add_argument("--procs", type=int, default=4, help="games run at once (each its own process)")
    ap.add_argument("--p-cast", type=float, default=0.3)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/plan_mpc/realtime.jsonl")
    a = ap.parse_args(argv)
    cfg = dict(wm=a.wm, branches=a.branches, horizon=a.horizon, budget_ms=a.budget_ms, threads=a.threads,
               p_cast=a.p_cast, max_ticks=a.max_ticks, device="cpu", mode="realtime-anytime",
               procs=a.procs)
    jobs = [(50_000_000 + a.first_seed + i, side, a.opponent, cfg) for side in ("radiant", "dire")
            for i in range(a.games_per_side)]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "w") as f, mp.get_context("spawn").Pool(a.procs) as pool:
        for n, row in enumerate(pool.imap_unordered(play_realtime, jobs), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{n}/{len(jobs)} seed {row['seed']} {row['side']:<7} {row['winner']:<7} ticks {row['ticks']}  "
                  f"ret {row['return_ours']:.2f}  frame p50 {row['frame_ms_p50']:.1f} p95 {row['frame_ms_p95']:.1f} "
                  f"ms, over budget {row['frames_over_budget_pct']:.1f}%  branches {row['branch_chosen']}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"-> {out}")



def batched_realtime_main(argv=None):
    """The same anytime planner as play_realtime (per game: K branches, ONE world-model tick
    per frame, lookahead H, settings calibrated for a single game at 60 fps on a laptop CPU),
    evaluated for many games at once: every frame is ONE batched model call over all games'
    tracking rows and branches. Batching changes only evaluation throughput, not what each
    game's planner is allowed to compute."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="runs/token_wm/p1_at1000")
    ap.add_argument("--games-per-side", type=int, default=60)
    ap.add_argument("--opponent", default="net", choices=("net", "script"))
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--p-cast", type=float, default=0.3)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calibration", default="", help="per-game 60 fps calibration note to record with the results")
    ap.add_argument("--out", default="runs/plan_mpc/realtime_vs_net.jsonl")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    model, stats = load_wm(a.wm, dev)
    fz = Featurizer(dev)
    K, H = a.branches, a.horizon
    jobs = [(50_000_000 + i, side) for side in ("radiant", "dire") for i in range(a.games_per_side)]
    G = len(jobs)
    A = Arena(G)
    for g, (seed, _) in enumerate(jobs):
        A.start(g, seed)
    ours = torch.zeros(G, 10, dtype=torch.bool)
    for g, (_, side) in enumerate(jobs):
        ours[g, :5], ours[g, 5:] = side == "radiant", side == "dire"
    om = ours.to(dev)
    gen = torch.Generator().manual_seed(0)
    h = torch.zeros(G, 206, model.d, device=dev)
    plan = torch.zeros(G, 5, 6, dtype=torch.long)
    plan[..., :2] = 3
    use_net = torch.ones(G, dtype=torch.bool)
    running, result, ticks = np.ones(G, bool), np.zeros(G, int), np.zeros(G, int)
    ret = np.zeros((G, 10))
    chosen = np.zeros((G, K), int)
    t0, t, s = time.time(), 0, None
    while running.any() and t < a.max_ticks:
        ent, tick, cidx = A.states()
        net = A.net_actions()
        enemy = net if a.opponent == "net" else np.stack([A.script_actions(g) for g in range(G)])
        net_ours = torch.from_numpy(net[ours.numpy()].reshape(G, 5, 6).astype(np.int64))
        act = enemy.copy()
        mine = torch.where(use_net[:, None, None], net_ours, plan).numpy()
        act[ours.numpy()] = mine.reshape(-1, 6)
        e, tk, ci = (torch.from_numpy(x).to(dev) for x in (ent, tick, cidx))
        if t % H == 0:                                   # every game starts a search from its real state
            plans = propose(net_ours, plan, K, a.p_cast, gen)                               # (G, K, 5, 6)
            ba = torch.from_numpy(enemy.astype(np.int64)).repeat_interleave(K, 0)            # (G*K, 10, 6)
            ob = ours.repeat_interleave(K, 0)
            ba[ob] = plans.reshape(-1, 6)
            s = dict(cur=e.repeat_interleave(K, 0), ct=tk.repeat_interleave(K, 0), cc=ci.repeat_interleave(K, 0),
                     hh=h.repeat_interleave(K, 0), act=ba.to(dev), total=torch.zeros(G * K, device=dev), plans=plans)
        with torch.no_grad():
            pred, hb = model.step(fz(torch.cat([e, s["cur"]]), torch.cat([tk, s["ct"]]), torch.cat([ci, s["cc"]])),
                                  torch.cat([torch.from_numpy(act.astype(np.int64)).to(dev), s["act"]]),
                                  torch.cat([h, s["hh"]]))
            h = hb[:G]
            sp = {k: (v[G:] if torch.is_tensor(v) and v.dim() and v.shape[0] == G * (K + 1) else v)
                  for k, v in pred.items()}
            nxt, s["ct"], s["cc"] = apply_prediction(s["cur"], s["ct"], s["cc"], sp, stats, fz)
            nxt = torch.nan_to_num(nxt)
            r, _ = reward(s["cur"], nxt)
            s["total"] += (r * om.repeat_interleave(K, 0)).sum(-1)
            s["cur"], s["hh"] = nxt, hb[G:]
        if t % H == H - 1:                               # searches complete: commit each game's best branch
            best = s["total"].view(G, K).argmax(1).cpu()
            plan = s["plans"][torch.arange(G), best]
            use_net = best == 0
            chosen[np.arange(G), best.numpy()] += 1
        for g in np.nonzero(running)[0]:
            res = A.step(g, act[g])
            ret[g] += A.rewards(g)
            ticks[g] += 1
            if res:
                running[g], result[g] = False, res
        t += 1
        if t % 500 == 0:
            print(f"  tick {t}: {running.sum()}/{G} games running, {(time.time() - t0):.0f}s", flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(wm=a.wm, branches=K, horizon=H, p_cast=a.p_cast, mode="anytime, 1 WM tick/frame/game",
               calibration=a.calibration, eval_device=str(dev), eval_batched_games=G)
    with open(out, "w") as f:
        for g, (seed, side) in enumerate(jobs):
            lg, o = A.logs(g), ours[g].numpy()
            win = {1: "radiant", 2: "dire", 3: "both"}.get(int(result[g]), "none")
            f.write(json.dumps(dict(team="planner", opp=a.opponent, side=side, seed=seed, ticks=int(ticks[g]),
                                    winner=win, won=win == side, lost=win in ("radiant", "dire") and win != side,
                                    return_ours=float(ret[g][o].mean()), return_theirs=float(ret[g][~o].mean()),
                                    **{f"ours_{k}": float(lg[o, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                                    **{f"theirs_{k}": float(lg[~o, i].mean()) for i, k in enumerate(LOG_FIELDS)},
                                    branch_chosen=chosen[g].tolist(), planner=cfg)) + "\n")
    print(f"{G} games in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    if "--batched" in sys.argv:
        sys.argv.remove("--batched")
        batched_realtime_main()
    elif "--realtime" in sys.argv:
        sys.argv.remove("--realtime")
        realtime_main()
    else:
        main()
