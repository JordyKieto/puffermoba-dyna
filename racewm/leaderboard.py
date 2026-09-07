"""Online behavioral leaderboard: every agent, in the REAL env, on holdout tracks.

This is the only comparison that means anything. A planner scored inside its
own world model is measuring the model, not the planner -- and that is exactly
the failure mode where imagination looks great and the env disagrees.

Scripted bots and exported policies are also runnable through race_eval, which
is faster; they are included here so every row comes from one code path and one
set of counters.

    python3 -m racewm.leaderboard --wm checkpoints/wm_multistep_warm/best.pt \\
        --policies bc=checkpoints/bc/expert_only.rpo --steps 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import contract, geometry, model as M, planning, shards
from .env import RacePool

LEFT, NOOP, RIGHT = 0, 1, 2
I = contract.INDEX


# ------------------------------------------------------------------- agents
class Agent:
    name = "agent"
    def act(self, obs, pool): raise NotImplementedError


class RandomAgent(Agent):
    name = "random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, obs, pool): return self.rng.integers(0, 3, len(obs))


class StraightAgent(Agent):
    name = "straight"
    def act(self, obs, pool): return np.full(len(obs), NOOP)


class WhiskerAgent(Agent):
    """Port of bot_whisker. flw points geometric RIGHT, so a near reading on
    flw calls for LEFT -- see docs/CONTRACT.md."""
    name = "whisker"
    def act(self, obs, pool):
        d = obs[:, I["flw_length"]] - obs[:, I["frw_length"]]
        a = np.full(len(obs), NOOP)
        a[d < -0.02] = LEFT
        a[d > 0.02] = RIGHT
        return a


class CenterlineAgent(Agent):
    """Port of bot_centerline: pure pursuit with its own persistent progress
    index and a lookahead measured in pixels."""
    name = "centerline"
    def __init__(self, pool, lookahead=60.0):
        tk = pool.tracks()
        self.center = {e: tk.centerline[e] for e in tk.centerline}
        self.idx = np.full(pool.n, -1)
        self.lookahead = lookahead
        self.env_idx = pool.env_idx
        self.frameskip, self.turn = 4, 40.0

    def act(self, obs, pool):
        px = obs[:, I["px_norm"]] * 1080.0
        py = obs[:, I["py_norm"]] * 720.0
        ang = np.arctan2(obs[:, I["sin_ang"]], obs[:, I["cos_ang"]])
        self.idx[obs[:, I["tick_norm"]] == 0] = -1      # new episode

        out = np.full(len(obs), NOOP)
        for k in range(len(obs)):
            c = self.center[int(self.env_idx[k])]
            n = len(c)
            if self.idx[k] < 0:
                self.idx[k] = int(np.argmin((c[:, 0] - px[k])**2 + (c[:, 1] - py[k])**2))
            else:
                w = (self.idx[k] + np.arange(0, 9)) % n
                d2 = (c[w, 0] - px[k])**2 + (c[w, 1] - py[k])**2
                self.idx[k] = int(w[np.argmin(d2)])
            j, acc = int(self.idx[k]), 0.0
            for _ in range(n):
                nx = (j + 1) % n
                acc += float(np.hypot(*(c[nx] - c[j])))
                j = nx
                if acc >= self.lookahead:
                    break
            err = np.arctan2(c[j, 1] - py[k], c[j, 0] - px[k]) - ang[k]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            tol = 0.5 * self.frameskip * np.pi / self.turn
            out[k] = LEFT if err > tol else (RIGHT if err < -tol else NOOP)
        return out


class PolicyAgent(Agent):
    """A trained net. Loads the same .rpo race_eval consumes, so this row and
    the race_eval row describe the identical policy."""
    def __init__(self, name, rpo_path):
        import struct
        self.name = name
        raw = Path(rpo_path).read_bytes()
        _, _, nl, od, ad, ak = struct.unpack_from("<6I", raw, 0)
        off = 24
        self.mu = np.frombuffer(raw, np.float32, od, off).copy(); off += 4 * od
        self.sd = np.frombuffer(raw, np.float32, od, off).copy(); off += 4 * od
        self.layers, self.act_kind = [], ak
        for _ in range(nl):
            i_dim, o_dim = struct.unpack_from("<2I", raw, off); off += 8
            w = np.frombuffer(raw, np.float32, i_dim * o_dim, off).reshape(o_dim, i_dim).copy()
            off += 4 * i_dim * o_dim
            b = np.frombuffer(raw, np.float32, o_dim, off).copy(); off += 4 * o_dim
            self.layers.append((w, b))

    def act(self, obs, pool):
        h = (obs - self.mu) / self.sd
        for k, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if k + 1 < len(self.layers):
                h = h / (1.0 + np.exp(-h))
        return h.argmax(1)


class PlannerAgent(Agent):
    """Plans through the learned world model, acts in the real env."""
    def __init__(self, name, sim, bank, slots, kind, depth, n_sims=16, seed=0):
        self.name = name
        self.sim, self.bank, self.slots = sim, bank, slots
        self.kind, self.depth, self.n_sims, self.seed = kind, depth, n_sims, seed
        self.value = planning.GreedyPolicyValue()
        self.t = 0

    def act(self, obs, pool):
        s = torch.from_numpy(np.ascontiguousarray(obs))
        self.t += 1
        if self.kind == "greedy":
            a, _ = planning.plan_exhaustive(self.sim, s, self.slots, 1, self.value)
        elif self.kind == "exhaustive":
            a, _ = planning.plan_exhaustive(self.sim, s, self.slots, self.depth, self.value)
        elif self.kind.startswith("beam"):
            w = int(self.kind[4:]) if len(self.kind) > 4 else 6
            a, _ = planning.plan_beam(self.sim, s, self.slots, self.depth, w, self.value)
        elif self.kind == "gumbel":
            a, _ = planning.plan_gumbel(self.sim, s, self.slots, self.depth,
                                        self.n_sims, self.value, seed=self.seed + self.t)
        else:
            raise ValueError(self.kind)
        return a.cpu().numpy()


# ------------------------------------------------------------------ metrics
def run_agent(agent, pool, steps, sectors=16, reward_yellow=0.2):
    """Behavioral metrics matching race_eval's groups."""
    obs = pool.reset()
    n = pool.n
    ep = {k: dict(ret=0.0, steps=0, acts=np.zeros(3), rev=0, prev=NOOP,
                  dist=0.0, minw=1.0, sect=set()) for k in range(n)}
    done_eps = []
    tracks = pool.tracks()
    t0 = time.time()

    for _ in range(steps):
        a = np.asarray(agent.act(obs, pool), dtype=np.int32)
        for k in range(n):
            e = ep[k]
            e["acts"][a[k]] += 1
            if (a[k] == LEFT and e["prev"] == RIGHT) or (a[k] == RIGHT and e["prev"] == LEFT):
                e["rev"] += 1
            if a[k] != NOOP:
                e["prev"] = a[k]
            e["minw"] = min(e["minw"], float(min(obs[k, I["flw_length"]],
                                                 obs[k, I["frw_length"]])))
            ph = np.arctan2(obs[k, I["sin_sector"]], obs[k, I["cos_sector"]])
            e["sect"].add(int(round((ph % (2 * np.pi)) / (2 * np.pi) * sectors)) % sectors)
        obs, rew, term, trunc = pool.step(a)
        for k in range(n):
            e = ep[k]
            e["ret"] += float(rew[k]); e["steps"] += 1
            if term[k] or trunc[k]:
                crossings = e["ret"] / reward_yellow
                done_eps.append(dict(
                    ret=e["ret"], steps=e["steps"], crashed=bool(term[k]),
                    laps=crossings / sectors, coverage=len(e["sect"]) / sectors,
                    idle=e["acts"][NOOP] / e["steps"], left=e["acts"][LEFT] / e["steps"],
                    right=e["acts"][RIGHT] / e["steps"], rev=e["rev"] / e["steps"],
                    minw=e["minw"]))
                ep[k] = dict(ret=0.0, steps=0, acts=np.zeros(3), rev=0, prev=NOOP,
                             dist=0.0, minw=1.0, sect=set())
    secs = time.time() - t0
    if not done_eps:
        return None

    # Episodes still running when the budget ends are CENSORED. Dropping them
    # silently biases every statistic toward whatever crashes fastest -- a good
    # agent's episodes are exactly the ones that fail to finish. Run with
    # steps >= max_episode_steps so truncation always fires, and report the
    # censored count so a biased run is visible rather than plausible.
    censored = sum(1 for k in range(n) if ep[k]["steps"] > 0)
    agg = lambda key: float(np.mean([d[key] for d in done_eps]))
    return dict(episodes=len(done_eps), censored=censored,
                censored_frac=censored / (len(done_eps) + censored),
                ret=agg("ret"), length=agg("steps"),
                crash=agg("crashed"), laps=agg("laps"), coverage=agg("coverage"),
                idle=agg("idle"), left=agg("left"), right=agg("right"),
                rev=agg("rev"), minw=agg("minw"),
                us_per_decision=1e6 * secs / (steps * pool.n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="checkpoints/wm_multistep_warm/best.pt")
    ap.add_argument("--termination", default="checkpoints/termination/best.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--policies", nargs="*", default=[],
                    help="name=path.rpo, repeatable")
    ap.add_argument("--planners", default="greedy,beam6,gumbel")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-sims", type=int, default=16)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--planner-envs", type=int, default=16,
                    help="planners cost orders of magnitude more per decision; "
                         "fewer parallel envs keeps episodes complete without "
                         "the wall-clock blowing up")
    # Defaults exceed max_episode_steps so every episode terminates or
    # truncates inside the budget; see the censoring note in run_agent.
    ap.add_argument("--steps", type=int, default=2100)
    ap.add_argument("--planner-steps", type=int, default=2100)
    ap.add_argument("--json-out", default="data/leaderboard.json")
    args = ap.parse_args()

    man = json.loads(Path(args.corpus, "manifest.json").read_text())
    h = man["splits"]["holdout"]
    mk = lambda: RacePool(n=args.envs, seed=man["seed"],
                          track_start=h["track_start"], n_tracks=h["n_tracks"],
                          frameskip=man["frameskip"],
                          max_episode_steps=man["max_episode_steps"])

    rows = []
    print(f"holdout tracks {h['track_start']}..{h['track_start']+h['n_tracks']-1}, "
          f"{args.envs} parallel envs\n")

    for A in (RandomAgent(), StraightAgent(), WhiskerAgent()):
        pool = mk(); r = run_agent(A, pool, args.steps); pool.close()
        if r: rows.append((A.name, r))
        print(f"  {A.name:<22} done")

    pool = mk(); A = CenterlineAgent(pool)
    r = run_agent(A, pool, args.steps); pool.close()
    if r: rows.append((A.name, r))
    print(f"  {A.name:<22} done")

    for spec in args.policies:
        name, path = spec.split("=", 1)
        pool = mk(); A = PolicyAgent(name, path)
        r = run_agent(A, pool, args.steps); pool.close()
        if r: rows.append((name, r))
        print(f"  {name:<22} done")

    # Planners share one world model; each gets its own pool so the slot
    # mapping stays aligned with env_idx.
    if args.planners:
        from . import dataset
        from .train_termination import TerminationHead
        flat = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                                  with_geometry=True, max_rows=150_000)
        x = torch.from_numpy(dataset.wm_inputs(flat)); y = torch.from_numpy(dataset.wm_targets(flat))
        x_mu, x_sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
        y_mu, y_sd = y.mean(0, keepdim=True), y.std(0, keepdim=True).clamp_min(1e-8)

        ck = torch.load(args.wm, map_location="cpu", weights_only=False)
        c = ck["config"]
        net = M.DeltaMLP(c["d_in"], contract.OBS_SIZE, c["hidden"], c["layers"])
        net.load_state_dict(ck["model"]); net.eval()

        head = t_mu = t_sd = None
        if Path(args.termination).exists():
            tk = torch.load(args.termination, map_location="cpu", weights_only=False)
            tc = tk["config"]
            head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"])
            head.load_state_dict(tk["model"]); head.eval()
            ex = tk.get("extra") or {}
            t_mu, t_sd = ex.get("mu", x_mu), ex.get("sd", x_sd)

        for kind in args.planners.split(","):
            pool = RacePool(n=args.planner_envs, seed=man["seed"],
                            track_start=h["track_start"], n_tracks=h["n_tracks"],
                            frameskip=man["frameskip"],
                            max_episode_steps=man["max_episode_steps"])
            bank = geometry.TrackBank(pool.tracks(), flat.header)
            slots = bank.slots(pool.env_idx)
            sim = planning.WorldModelSim(net, bank, x_mu, x_sd, y_mu, y_sd,
                                         head, t_mu, t_sd)
            A = PlannerAgent(f"plan:{kind}", sim, bank, slots, kind,
                             args.depth, args.n_sims)
            r = run_agent(A, pool, args.planner_steps); pool.close()
            if r: rows.append((A.name, r))
            print(f"  {A.name:<22} done")

    rows.sort(key=lambda kv: -kv[1]["ret"])
    print(f"\n{'='*94}\n ONLINE LEADERBOARD -- real env, holdout tracks\n{'='*94}")
    print(f"{'agent':<20}{'eps':>7}{'cens%':>7}{'return':>9}{'laps':>8}{'crash%':>8}"
          f"{'length':>8}{'cover':>8}{'idle%':>7}{'revrs':>8}{'us/dec':>10}")
    for name, r in rows:
        print(f"{name:<20}{r['episodes']:>7}{100*r['censored_frac']:>7.1f}"
              f"{r['ret']:>9.3f}{r['laps']:>8.3f}"
              f"{100*r['crash']:>8.1f}{r['length']:>8.1f}{r['coverage']:>8.3f}"
              f"{100*r['idle']:>7.1f}{r['rev']:>8.3f}{r['us_per_decision']:>10.1f}")
    worst = max(r["censored_frac"] for _, r in rows)
    if worst > 0.05:
        print(f"\n  WARNING: up to {100*worst:.0f}% of episodes were censored. "
              f"Raise --steps above max_episode_steps ({man['max_episode_steps']}).")

    Path(args.json_out).write_text(json.dumps(
        {"holdout": h, "rows": dict(rows)}, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
