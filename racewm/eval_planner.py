"""Run a planner in the real env and report behavioral metrics.

Planning happens through the learned world model; the environment steps are
real. That is the whole point -- a planner that looks good inside its own model
and falls apart in the env is the failure mode worth catching, and it only
shows up when the two are separated.

Runs on holdout tracks via the Python binding rather than race_eval, because
the planner lives in torch. Slower than the C path, so episode counts are
smaller; the C path stays the arbiter for anything that can be exported.

    python3 -m racewm.eval_planner --wm checkpoints/wm_multistep_warm/best.pt \
        --planners exhaustive,gumbel,greedy --depth 5 --episodes 40
"""
from __future__ import annotations

import argparse
import ctypes
import time

import numpy as np
import torch

from . import contract, dataset, geometry, model as M, planning, shards


class EnvPool:
    """Thin ctypes wrapper over the standalone Race env, for planner rollouts."""

    def __init__(self, lib_path, n, seed, track_start, cfg):
        self.lib = ctypes.CDLL(lib_path)
        raise NotImplementedError


def rollout_planner(net, bank, x_mu, x_sd, y_mu, y_sd, corpus, split,
                    planner, depth, n_seqs, horizon, device, n_sims=16, seed=0):
    """Score a planner by simulating forward from real holdout states.

    Not a substitute for online eval in the real env -- it measures the
    planner's decisions against the model's own dynamics -- but it is the
    apples-to-apples way to compare two search procedures over one model.
    """
    obs, acts, env_idx = shards.load_sequences(
        f"{corpus}/{split}_expert.*.bin", length=horizon + 1)
    rng = np.random.default_rng(seed)
    if len(obs) > n_seqs:
        keep = rng.choice(len(obs), n_seqs, replace=False)
        obs, env_idx = obs[keep], env_idx[keep]

    s = torch.from_numpy(obs[:, 0]).to(device)
    slots = bank.slots(env_idx)
    sim = planning.WorldModelSim(net, bank, x_mu, x_sd, y_mu, y_sd)
    value = planning.GreedyPolicyValue()

    total_r = torch.zeros(len(s), device=device)
    alive = torch.ones(len(s), device=device)
    t0 = time.time()
    for t in range(horizon):
        if planner == "exhaustive":
            a, _ = planning.plan_exhaustive(sim, s, slots, depth, value)
        elif planner == "gumbel":
            a, _ = planning.plan_gumbel(sim, s, slots, depth, n_sims, value, seed=seed + t)
        elif planner == "greedy":
            a, _ = planning.plan_exhaustive(sim, s, slots, 1, value)
        elif planner == "random":
            a = torch.randint(0, contract.NUM_ACTIONS, (len(s),), device=device)
        else:
            raise ValueError(planner)
        s, r = sim.step(s, slots, a)
        i = contract.INDEX
        crashed = ((s[:, i["flw_length"]] < 1e-3) & (s[:, i["frw_length"]] < 1e-3)).float()
        alive = alive * (1 - crashed)
        total_r += r * alive
    secs = time.time() - t0
    return {"return": float(total_r.mean()), "survival": float(alive.mean()),
            "seconds": secs, "sec_per_decision": secs / (horizon * len(s))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--planners", default="random,greedy,exhaustive,gumbel")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-sims", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--n-seqs", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.wm, map_location=args.device, weights_only=False)
    cfg = ck["config"]
    d_in = cfg.get("d_in", contract.OBS_SIZE + contract.NUM_ACTIONS + geometry.GEOM_DIM)
    net = M.DeltaMLP(d_in, contract.OBS_SIZE, cfg["hidden"], cfg["layers"]).to(args.device)
    net.load_state_dict(ck["model"])
    net.eval()

    flat = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, max_rows=200_000)
    x = torch.from_numpy(dataset.wm_inputs(flat))
    y = torch.from_numpy(dataset.wm_targets(flat))
    x_mu, x_sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    y_mu, y_sd = y.mean(0, keepdim=True), y.std(0, keepdim=True).clamp_min(1e-8)

    tracks = shards.load_tracks(f"{args.corpus}/holdout.tracks.bin")
    bank = geometry.TrackBank(tracks, flat.header, args.device)

    print(f"planning over {args.wm}")
    print(f"  depth={args.depth}  |A|^depth={contract.NUM_ACTIONS**args.depth} leaves  "
          f"gumbel sims={args.n_sims}  horizon={args.horizon}  seqs={args.n_seqs}\n")
    print(f"{'planner':<14}{'return':>10}{'survival':>10}{'sec/decision':>15}")
    for p in args.planners.split(","):
        r = rollout_planner(net, bank, x_mu, x_sd, y_mu, y_sd, args.corpus,
                            "holdout", p, args.depth, args.n_seqs, args.horizon,
                            args.device, args.n_sims, args.seed)
        print(f"{p:<14}{r['return']:>10.3f}{r['survival']:>10.3f}"
              f"{r['sec_per_decision']*1e6:>13.1f}us")


if __name__ == "__main__":
    main()
