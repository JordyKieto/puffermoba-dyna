"""Which reward components does the dream get wrong for a policy's own actions?

Plays REAL games (C arena, many at once) with a team policy on one side and the shipped
PufferNet on the other, records every real state and action, then feeds the SAME actions
to the world model two ways:

  teacher-forced  each tick the WM predicts from the REAL state (one-step error only)
  open loop       128-tick episodes from real start states, exactly how dream PPO trains

and compares, per reward component (the game's reward: death -0.164, XP x 0.0067, tower
+0.642; mobawm.rl.dream_ppo.reward), what the dream pays with what the real game paid,
for our team's heroes. Teams: a dream-trained PPO policy, the shipped net, random actions.

    .venv/bin/python scripts/moba/dream_vs_real_reward.py \\
        --policy runs/dream_ppo/p3/selfplay_rerun/policy.pt --games 8 --ticks 900
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

from mobawm.rl.dream_ppo import Policy, reward  # noqa: E402
from mobawm.rl.plan_mpc import Arena, load_wm  # noqa: E402
from mobawm.wm.data import Featurizer, apply_prediction  # noqa: E402

COMP = ("death", "xp", "distance", "tower")


def play(team, policy_path, G, T, seed0):
    """Real games: `team` on radiant (heroes 0-4), the shipped net on dire. Returns the
    recorded states/actions (T+1 states) and the real per-tick reward components."""
    A = Arena(G)
    for g in range(G):
        A.start(g, seed0 + g)
    pol = None
    if team == "ppo":
        pol = Policy()
        ck = torch.load(policy_path, map_location="cpu", weights_only=False)
        pol.load_state_dict(ck["policy"] if "policy" in ck else ck)
        pol.eval()
    rng = np.random.default_rng(seed0)
    ents, ticks, cidxs, acts = [], [], [], []
    o = np.zeros((10, 510), np.uint8)
    for t in range(T + 1):
        e, tk, ci = A.states()
        ents.append(e), ticks.append(tk), cidxs.append(ci)
        if t == T:
            break
        net = A.net_actions()
        act = net.copy()
        if team == "ppo":
            obs = np.zeros((G, 5, 510), np.uint8)
            for g in range(G):
                A.L.ma_multi_obs(g, o.ctypes.data)
                obs[g] = o[:5]
            with torch.no_grad():
                logits, _ = pol(torch.from_numpy(obs.reshape(-1, 510)))
                a = torch.stack([torch.distributions.Categorical(logits=l).sample() for l in logits], -1)
            act[:, :5] = a.view(G, 5, 6).numpy()
        elif team == "random":
            act[:, :5] = np.stack([rng.integers(0, n, (G, 5)) for n in (7, 7, 3, 2, 2, 2)], -1)
        acts.append(act.astype(np.int64))
        for g in range(G):
            A.step(g, act[g])
    return (np.stack(ents, 1), np.stack(ticks, 1), np.stack(cidxs, 1), np.stack(acts, 1))      # (G, T+1, ...)


@torch.no_grad()
def audit(model, stats, dev, ents, ticks, cidxs, acts, horizon, burn=8):
    fz = Featurizer(dev)
    G, T1 = ents.shape[:2]
    T = T1 - 1
    E = torch.from_numpy(ents).to(dev)
    TK, CI = torch.from_numpy(ticks).to(dev), torch.from_numpy(cidxs).to(dev)
    AC = torch.from_numpy(acts).to(dev)
    real = torch.zeros(4, device=dev)
    tf = torch.zeros(4, device=dev)
    h = torch.zeros(G, 206, model.d, device=dev)
    for t in range(T):                                   # teacher-forced, memory tracked on the real game
        pred, h = model.step(fz(E[:, t], TK[:, t], CI[:, t]), AC[:, t], h)
        nxt, _, _ = apply_prediction(E[:, t], TK[:, t], CI[:, t], pred, stats, fz)
        _, cr = reward(E[:, t], E[:, t + 1])
        _, cd = reward(E[:, t], torch.nan_to_num(nxt))
        real += cr[:, :5].sum((0, 1))
        tf += cd[:, :5].sum((0, 1))
    starts = list(range(burn, T - horizon, horizon))     # open loop, 128-tick episodes as in dream PPO
    ol_real, ol_dream = torch.zeros(4, device=dev), torch.zeros(4, device=dev)
    for s0 in starts:
        h = torch.zeros(G, 206, model.d, device=dev)
        for t in range(s0 - burn, s0):
            _, h = model.step(fz(E[:, t], TK[:, t], CI[:, t]), AC[:, t], h)
        cur, ct, cc = E[:, s0].clone(), TK[:, s0].clone(), CI[:, s0].clone()
        for t in range(s0, s0 + horizon):
            pred, h = model.step(fz(cur, ct, cc), AC[:, t], h)
            nxt, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
            nxt = torch.nan_to_num(nxt)
            _, cd = reward(cur, nxt)
            _, cr = reward(E[:, t], E[:, t + 1])
            ol_dream += cd[:, :5].sum((0, 1))
            ol_real += cr[:, :5].sum((0, 1))
            cur = nxt
    n_tf = G * T * 5 / 1000
    n_ol = G * len(starts) * horizon * 5 / 1000
    unit = torch.tensor([-0.163764, 0.00665677, 1.0, 0.642119], device=dev)      # reward -> events (deaths, XP, -, towers)
    f = lambda v, n: (v / unit / n).tolist()
    return dict(real_per_1k=f(real, n_tf), teacher_forced_dream_per_1k=f(tf, n_tf),
                openloop_real_per_1k=f(ol_real, n_ol), openloop_dream_per_1k=f(ol_dream, n_ol),
                hero_steps_tf=int(n_tf * 1000), hero_steps_ol=int(n_ol * 1000), episodes=G * len(starts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="runs/dream_ppo/p3/selfplay_rerun/policy.pt")
    ap.add_argument("--wm", default="runs/token_wm/p1_at1000")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=900)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--teams", default="ppo,net,random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-xp-clamp", action="store_true", help="dream state update without rule R7")
    ap.add_argument("--signed-xp-reward", action="store_true", help="XP reward = signed XP change")
    ap.add_argument("--out", default="results/moba/dream/dream_vs_real_reward.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    import mobawm.rl.dream_ppo as DP
    import mobawm.wm.data as WD
    WD.XP_MONOTONE = not a.no_xp_clamp
    DP.XP_SIGNED = a.signed_xp_reward
    model, stats = load_wm(a.wm, dev)
    rows = {}
    print(f"{'team':<8}{'measure':<24}{'deaths/1k':>11}{'XP/1k':>10}{'towers/1k':>11}")
    for team in a.teams.split(","):
        t0 = time.time()
        ents, ticks, cidxs, acts = play(team, a.policy, a.games, a.ticks, 70_000_000)
        r = audit(model, stats, dev, ents, ticks, cidxs, acts, a.horizon)
        rows[team] = r
        for k, lab in (("real_per_1k", "REAL game"), ("teacher_forced_dream_per_1k", "dream, teacher-forced"),
                       ("openloop_real_per_1k", "REAL, same windows"), ("openloop_dream_per_1k", "dream, open loop 128")):
            d, x, _, tw = r[k]
            print(f"{team:<8}{lab:<24}{d:>11.2f}{x:>10.0f}{tw:>11.3f}")
        print(f"         ({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / a.out
    out.write_text(json.dumps(dict(args=vars(a), note="per 1k hero-steps of OUR team (radiant, heroes 0-4) vs the "
                                   "shipped net; events: deaths, XP points, tower kills", teams=rows), indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
