"""Recalibrate the termination head against a SPECIFIC policy.

Calibrating on the corpus aggregate is not enough, and E14 measured how much
not enough. After corpus calibration the head matched the holdout termination
count to 1.01x — yet the trained PPO policy's imagined episodes ran 213 steps
against 345 real, because that policy visits a safer state distribution than
`centerline`/`whisker`/`random` ever did and the head still assigned it too
much crash probability.

Calibration is a property of a *behaviour distribution*, not of a model. So it
has to be refit on rollouts from the policy being trained.

This costs real environment steps, which are reported separately and must be
added to the run's real-sample budget. It is also the first step toward a
genuine online loop: collect with the current policy, refit, continue.

Collection happens on TRAIN tracks. Holdout is never touched.

    python3 -m racewm.recalibrate --policy checkpoints/ppo2_h200/best.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import contract, dataset, geometry, shards
from .env import RacePool
from .train_termination import TerminationHead


def collect(policy_ac, obs_mu, obs_sd, n_envs=256, steps=4000, seed=0,
            track_start=0, n_tracks=96, manifest=None):
    """Roll the policy in the REAL env; return (obs, action, done) it visited."""
    man = manifest or {}
    pool = RacePool(n=n_envs, seed=man.get("seed", 42), track_start=track_start,
                    n_tracks=n_tracks, frameskip=man.get("frameskip", 4),
                    max_episode_steps=man.get("max_episode_steps", 2000))
    obs = pool.reset()
    tracks = pool.tracks()
    O, A, D, E = [], [], [], []
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        with torch.no_grad():
            logits = policy_ac.actor((torch.from_numpy(obs.copy()) - obs_mu) / obs_sd)
            a = torch.distributions.Categorical(logits=logits).sample()
        O.append(obs.copy()); A.append(a.numpy().copy()); E.append(pool.env_idx.copy())
        obs, rew, term, trunc = pool.step(a.numpy())
        # Only real terminations count. A time-limit truncation is not a crash
        # and must not be fitted as one.
        D.append(term.copy())
    pool.close()
    return (np.concatenate(O), np.concatenate(A).astype(np.int64),
            np.concatenate(D).astype(np.float32), np.concatenate(E), tracks,
            n_envs * steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="checkpoints/ppo2_h200/best.pt")
    ap.add_argument("--head", default="checkpoints/termination/best.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="checkpoints/termination/calibration_policy.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from .dream_ppo import ActorCritic
    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    cfg, ex = ck["config"], ck["extra"]
    ac = ActorCritic(cfg["hidden"], cfg["layers"])
    ac.load_state_dict(ck["model"]); ac.eval()
    obs_mu, obs_sd = ex["obs_mu"], ex["obs_sd"]

    man = dataset.load_manifest(args.corpus)
    tr = man["splits"]["train"]
    print(f"collecting with {args.policy} on TRAIN tracks "
          f"{tr['track_start']}..{tr['track_start']+tr['n_tracks']-1}")
    obs, act, done, env_idx, tracks, n_real = collect(
        ac, obs_mu, obs_sd, args.envs, args.steps, args.seed,
        tr["track_start"], tr["n_tracks"], man)
    print(f"  collected {n_real:,} REAL env steps  "
          f"({done.sum():.0f} terminations, {100*done.mean():.3f}%)")
    print(f"  ** these are real samples and must be added to the run's budget **")

    tk = torch.load(args.head, map_location="cpu", weights_only=False)
    tc = tk["config"]
    head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"])
    head.load_state_dict(tk["model"]); head.eval()
    hex_ = tk.get("extra") or {}
    hmu, hsd = hex_["mu"], hex_["sd"]

    hdr = shards.read_header(shards._shard_paths(f"{args.corpus}/train_expert.*.bin")[0])
    bank = geometry.TrackBank(tracks, hdr)
    slots = bank.slots(env_idx)
    ot = torch.from_numpy(obs)
    with torch.no_grad():
        gm = bank.local_window(ot, slots) 
        oh = torch.zeros(len(act), contract.NUM_ACTIONS)
        oh[torch.arange(len(act)), torch.from_numpy(act)] = 1.0
        x = torch.cat([ot, oh, gm], dim=1)
        z = head((x - hmu) / hsd)
    y = torch.from_numpy(done)

    def stats(T, b):
        p = torch.sigmoid(z / T + b)
        return float(p.sum()), float(y.sum())

    old = torch.load("checkpoints/termination/calibration.pt", map_location="cpu",
                     weights_only=False)
    for label, (T, b) in (("raw", (1.0, 0.0)),
                          ("corpus-calibrated", (old["temperature"], old["bias"]))):
        e, a = stats(T, b)
        print(f"  {label:<20} predicted {e:8.0f} vs actual {a:6.0f}  ratio {e/max(a,1):.2f}")

    T = torch.nn.Parameter(torch.ones(1)); b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([T, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(
            z / T.clamp_min(1e-2) + b, y)
        l.backward()
        return l
    opt.step(closure)
    T_, b_ = float(T.clamp_min(1e-2).detach()), float(b.detach())
    e, a = stats(T_, b_)
    print(f"  {'policy-calibrated':<20} predicted {e:8.0f} vs actual {a:6.0f}  "
          f"ratio {e/max(a,1):.2f}   (T={T_:.4f}, b={b_:+.4f})")

    torch.save({"temperature": T_, "bias": b_, "head": args.head,
                "policy": args.policy, "real_steps_used": n_real,
                "schema_hash": contract.schema_hash()}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
