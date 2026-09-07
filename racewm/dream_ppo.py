"""PPO trained entirely inside the learned world model, evaluated in the real one.

The protocol is the point:

  1. Train PPO on imagined rollouts only. No real environment steps are taken
     during training -- the real data budget is fixed, and is spent on fitting
     the world model and on the buffer of real states that imagined episodes
     reset from.
  2. Evaluate periodically in the CANONICAL environment on holdout tracks, at
     checkpoints, so the curve is against a real-behaviour axis rather than an
     imagined-return axis.
  3. Keep going to absurd imagined budgets and watch what happens.

Step 3 is the experiment. E8/E10 measured this world model being exploited by a
planner searching only five steps deep; PPO with millions of steps searches far
harder. So the expectation is that real performance improves, peaks, then
degrades as the policy finds regions where the model is optimistically wrong.
Locating that turning point is a result either way, and imagined return will
keep climbing straight through it -- which is exactly why the evaluation has to
happen in the real environment.

    python3 -m racewm.dream_ppo --iters 200 --horizon 200 --eval-every 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dream, export_policy, tracking


class ActorCritic(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        def trunk(out):
            mods, d = [], contract.OBS_SIZE
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            mods.append(nn.Linear(d, out))
            return nn.Sequential(*mods)
        self.actor = trunk(contract.NUM_ACTIONS)
        self.critic = trunk(1)

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def gae(rew, val, crashed, truncated, last_val, gamma=0.99, lam=0.95):
    """GAE with correct time-limit handling.

    `crashed` zeroes the bootstrap: the episode really ended and there is no
    future value. `truncated` does NOT -- the episode was cut by a time limit
    and its continuation still has value, so the bootstrap flows through while
    the advantage trace is reset (the next state belongs to a different
    episode). Conflating the two makes the policy myopic, in proportion to how
    often the limit fires.
    """
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    run = torch.zeros(N, device=rew.device)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        alive = 1.0 - crashed[t].float()          # value continues unless it crashed
        cut = 1.0 - (crashed[t] | truncated[t]).float()   # trace breaks on either
        delta = rew[t] + gamma * nv * alive - val[t]
        run = delta + gamma * lam * cut * run
        adv[t] = run
    return adv, adv + val


def evaluate(ac, obs_mu, obs_sd, tag, episodes=100):
    rpo = f"/tmp/dreamppo_{tag}.rpo"
    export_policy.export(ac.actor, obs_mu.cpu().numpy(), obs_sd.cpu().numpy(), rpo)
    r = subprocess.run(["./build/race_eval", "--bots", "policy", "--policy", rpo,
                        "--tracks", "16", "--track-start", "112",
                        "--episodes", str(episodes), "--threads", "10"],
                       capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if l.strip().startswith("policy"):
            f = l.split()
            return float(f[2]), float(f[4])
    return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--wm", default="checkpoints/wm_multistep_warm/best.pt")
    ap.add_argument("--termination", default="checkpoints/termination/best.pt")
    ap.add_argument("--calibration", default="checkpoints/termination/calibration.pt")
    ap.add_argument("--real-budget", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    run_name = args.run_name or f"dreamppo_h{args.horizon}_e{args.envs}"

    env, meta = dream.build(args.corpus, args.wm, args.termination, args.calibration,
                            args.real_budget, args.envs, args.horizon, dev, args.seed)
    obs_mu, obs_sd = meta["obs_mu"], meta["obs_sd"]
    print(f"dream env: {args.envs} parallel, horizon {args.horizon}, "
          f"termination {'calibrated' if meta['calibrated'] else 'RAW (biased)'}")
    print(f"real budget: {meta['n_real']:,} transitions (world model + reset anchors)")
    print(f"  no real env steps are taken during training\n")

    ac = ActorCritic(args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(ac.parameters(), lr=args.lr, eps=1e-5)
    cfg = dict(vars(args), n_real=meta["n_real"], calibrated=meta["calibrated"])
    cp = ckpt_mod.Checkpointer("checkpoints", run_name, config=cfg,
                               metric_name="real_return", metric_mode="max")
    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase2", "dream-ppo"])
    trk.watch_params(ac)

    obs = env.reset()
    history, t0 = [], time.time()
    print(f"{'iter':>6}{'imagined':>13}{'imag_ret':>10}{'real_ret':>10}{'surv%':>8}"
          f"{'entropy':>9}{'sec':>7}")

    for it in range(args.iters):
        O = torch.zeros(args.rollout, args.envs, contract.OBS_SIZE, device=dev)
        A = torch.zeros(args.rollout, args.envs, dtype=torch.long, device=dev)
        LP = torch.zeros(args.rollout, args.envs, device=dev)
        R = torch.zeros(args.rollout, args.envs, device=dev)
        C = torch.zeros(args.rollout, args.envs, dtype=torch.bool, device=dev)
        TR = torch.zeros(args.rollout, args.envs, dtype=torch.bool, device=dev)
        V = torch.zeros(args.rollout, args.envs, device=dev)

        with torch.no_grad():
            for t in range(args.rollout):
                z = (obs - obs_mu) / obs_sd
                logits, v = ac(z)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                O[t], A[t], LP[t], V[t] = obs, a, dist.log_prob(a), v
                obs, r, c, tr = env.step(a)
                R[t], C[t], TR[t] = r, c, tr
            _, last_v = ac((obs - obs_mu) / obs_sd)

        adv, ret = gae(R, V, C, TR, last_v, args.gamma, args.lam)
        b_o = ((O - obs_mu) / obs_sd).reshape(-1, contract.OBS_SIZE)
        b_a, b_lp = A.reshape(-1), LP.reshape(-1)
        b_adv = ((adv - adv.mean()) / adv.std().clamp_min(1e-8)).reshape(-1)
        b_ret = ret.reshape(-1)

        n = len(b_o)
        mb = n // args.minibatches
        ent_val = 0.0
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logits, v = ac(b_o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_a[idx])
                ratio = (lp - b_lp[idx]).exp()
                a1 = ratio * b_adv[idx]
                a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]
                pg = -torch.min(a1, a2).mean()
                vf = ((v - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vf - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
                ent_val = float(ent.detach())

        imag_ret = float(R.sum(0).mean())
        real_ret = real_surv = float("nan")
        if (it + 1) % args.eval_every == 0 or it == 0:
            real_ret, real_surv = evaluate(ac, obs_mu, obs_sd, run_name,
                                           args.eval_episodes)
            cp.save(step=it + 1, model=ac, optimizer=opt,
                    extra={"obs_mu": obs_mu.cpu(), "obs_sd": obs_sd.cpu()},
                    metrics={"real_return": real_ret})
            history.append(dict(iter=it + 1, imagined=env.imagined_steps,
                                imag_ret=imag_ret, real_ret=real_ret,
                                real_surv=real_surv))
            print(f"{it+1:>6}{env.imagined_steps:>13,}{imag_ret:>10.3f}"
                  f"{real_ret:>10.3f}{real_surv:>8.1f}{ent_val:>9.3f}"
                  f"{time.time()-t0:>7.0f}", flush=True)
        trk.log(it + 1, samples=args.rollout * args.envs,
                imagined_return=imag_ret, real_return=real_ret,
                entropy=ent_val, imagined_steps=env.imagined_steps,
                real_transitions=meta["n_real"])

    Path(f"data/{run_name}.json").write_text(json.dumps(
        dict(config=cfg, history=history), indent=2))
    trk.summary(imagined_total=env.imagined_steps, real_transitions=meta["n_real"],
                best_real_return=max((h["real_ret"] for h in history), default=float("nan")))
    trk.finish()
    print(f"\nimagined {env.imagined_steps:,} steps against a fixed "
          f"{meta['n_real']:,} real transitions "
          f"({env.imagined_steps/meta['n_real']:.0f}:1)")


if __name__ == "__main__":
    main()
