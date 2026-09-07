"""Dreamer-style actor-critic trained inside the learned world model.

Follows the parts of DreamerV3 that matter here and drops the parts that do not
(we already have a Markov symbolic state, so there is no RSSM to learn):

  Imagination is anchored to REAL states. Every rollout starts from a state
  drawn from the corpus, never from an imagined one. This is the detail that
  keeps imagination near the data manifold, and it is why a short horizon is
  enough -- the model is never asked to extrapolate far from something it saw.

  Short horizon, H=15. Not a limitation to engineer around: our own rollout
  measurement says the model is trustworthy to ~100 steps after multi-step
  training, and the critic bootstraps everything past H anyway.

  lambda-returns as the critic target. Mixes n-step returns geometrically,
  trading the variance of Monte Carlo for the bias of pure one-step TD.

  REINFORCE with a critic baseline for the actor, because the action space is
  discrete. DreamerV3 uses straight-through gradients for continuous actions;
  with 3 actions a score-function estimator plus entropy regularization is the
  simpler correct choice.

Sample accounting is explicit and separate: real transitions consumed and
imagined transitions generated are reported as independent axes, because the
claim worth making is real-sample efficiency, not total compute.

    python3 -m racewm.dreamer --wm checkpoints/wm_multistep_warm/best.pt \\
        --real-budget 200000 --updates 300 --eval
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, export_policy, geometry, model as M
from . import planning, shards, tracking


class ActorCritic(nn.Module):
    def __init__(self, d_obs=contract.OBS_SIZE, n_act=contract.NUM_ACTIONS,
                 hidden=256, layers=3):
        super().__init__()
        def trunk(out):
            mods, d = [], d_obs
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            mods.append(nn.Linear(d, out))
            return nn.Sequential(*mods)
        self.actor = trunk(n_act)
        self.critic = trunk(1)

    def forward(self, s):
        return self.actor(s), self.critic(s).squeeze(-1)


def lambda_returns(rewards, values, bootstrap, gamma=0.97, lam=0.95):
    """DreamerV3 lambda-returns, computed backwards over the imagined rollout.

    rewards, values: (B, H). bootstrap: (B,) value of the state after step H.
    """
    H = rewards.shape[1]
    out = torch.zeros_like(rewards)
    nxt = bootstrap
    for t in reversed(range(H)):
        v_next = values[:, t + 1] if t + 1 < H else bootstrap
        nxt = rewards[:, t] + gamma * ((1 - lam) * v_next + lam * nxt)
        out[:, t] = nxt
    return out


def imagine(sim, ac, s0, slots, H, obs_mu, obs_sd, gen=None):
    """Roll the actor inside the model for H steps from real states s0."""
    s = s0
    states, logps, rewards, entropies, alive = [], [], [], [], []
    live = torch.ones(len(s0), device=s0.device)
    for _ in range(H):
        z = (s - obs_mu) / obs_sd
        logits, _ = ac(z)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        s2, r, p_term = sim.step(s, slots, a)
        states.append(s)
        logps.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        rewards.append(r * live)
        # Soft continue signal from the termination head. Weighting by the
        # survival probability, rather than by a hard threshold on a predicted
        # whisker, is what keeps the actor from optimizing inside a crash
        # region the dynamics model invents.
        live = live * (1 - p_term)
        alive.append(live.clone())
        s = s2
    return (torch.stack(states, 1), torch.stack(logps, 1),
            torch.stack(rewards, 1), torch.stack(entropies, 1),
            torch.stack(alive, 1), s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--termination", default="checkpoints/termination/best.pt",
                    help="termination head; without it imagination has no crash signal")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--real-budget", type=int, default=200_000,
                    help="real transitions the agent is allowed to have seen")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--bc-init", default=None,
                    help="warm start the actor from a BC checkpoint")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="checkpoints/dreamer/policy.rpo")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    H = args.horizon
    run_name = args.run_name or f"dreamer_H{H}_real{args.real_budget//1000}k"

    dataset.assert_disjoint(args.corpus)
    man = dataset.load_manifest(args.corpus)

    # The real-sample budget is enforced here and nowhere else: this is the
    # only real data the agent is allowed to have touched.
    real = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, max_rows=args.real_budget,
                              seed=args.seed)
    print(f"real budget: {len(real):,} transitions ({len(np.unique(real.env_idx))} tracks)")

    x = torch.from_numpy(dataset.wm_inputs(real))
    y = torch.from_numpy(dataset.wm_targets(real))
    x_mu, x_sd = x.mean(0, keepdim=True).to(dev), x.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    y_mu, y_sd = y.mean(0, keepdim=True).to(dev), y.std(0, keepdim=True).clamp_min(1e-8).to(dev)
    obs_t = torch.from_numpy(real.obs).to(dev)
    obs_mu = obs_t.mean(0, keepdim=True)
    obs_sd = obs_t.std(0, keepdim=True).clamp_min(1e-6)

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    wcfg = ck["config"]
    d_in = wcfg.get("d_in", contract.OBS_SIZE + contract.NUM_ACTIONS + geometry.GEOM_DIM)
    wm = M.DeltaMLP(d_in, contract.OBS_SIZE, wcfg["hidden"], wcfg["layers"]).to(dev)
    wm.load_state_dict(ck["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    tracks = shards.load_tracks(f"{args.corpus}/train.tracks.bin")
    bank = geometry.TrackBank(tracks, real.header, dev)
    slots_all = bank.slots(real.env_idx)
    term_head = t_mu = t_sd = None
    if args.termination and Path(args.termination).exists():
        from .train_termination import TerminationHead
        tk = torch.load(args.termination, map_location=dev, weights_only=False)
        tc = tk["config"]
        term_head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"]).to(dev)
        term_head.load_state_dict(tk["model"])
        term_head.eval()
        for q in term_head.parameters():
            q.requires_grad_(False)
        ex = tk.get("extra") or {}
        t_mu = ex.get("mu", x_mu).to(dev)
        t_sd = ex.get("sd", x_sd).to(dev)
        print(f"termination head loaded (val AP {tk['metrics'].get('val_ap', float('nan')):.3f})")
    else:
        print("WARNING: no termination head -- imagination has no crash signal")
    sim = planning.WorldModelSim(wm, bank, x_mu, x_sd, y_mu, y_sd,
                                 term_head, t_mu, t_sd)

    ac = ActorCritic(hidden=args.hidden, layers=args.layers).to(dev)
    if args.bc_init:
        bc = torch.load(args.bc_init, map_location=dev, weights_only=False)
        try:
            ac.actor.load_state_dict(bc["model"])
            print(f"actor warm started from {args.bc_init}")
        except RuntimeError as e:
            print(f"could not warm start actor ({e}); training from scratch")

    opt = torch.optim.AdamW(ac.parameters(), lr=args.lr, weight_decay=1e-5)
    cfg = dict(vars(args), n_real=len(real), corpus_manifest=man,
               params=sum(p.numel() for p in ac.parameters()))
    cp = ckpt_mod.Checkpointer("checkpoints", run_name, config=cfg,
                               metric_name="imagined_return", metric_mode="max")
    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase2", "dreamer"])
    trk.watch_params(ac)

    g = torch.Generator().manual_seed(args.seed)
    imagined_total = 0
    t0 = time.time()

    for it in range(args.updates):
        # Anchor every rollout to real states drawn from the buffer.
        idx = torch.randint(0, len(obs_t), (args.batch,), generator=g).to(dev)
        s0 = obs_t[idx]
        slots = slots_all[idx]

        states, logps, rewards, ents, alive, s_end = imagine(
            sim, ac, s0, slots, H, obs_mu, obs_sd)
        imagined_total += args.batch * H

        z = (states.reshape(-1, contract.OBS_SIZE) - obs_mu) / obs_sd
        _, v_flat = ac(z)
        values = v_flat.view(args.batch, H)
        with torch.no_grad():
            _, boot = ac((s_end - obs_mu) / obs_sd)

        with torch.no_grad():
            ret = lambda_returns(rewards, values, boot, args.gamma, args.lam)
            adv = (ret - values).detach()
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

        # Actor: REINFORCE with the critic as baseline; entropy keeps it from
        # collapsing onto one action, which the LEFT-heavy corpus invites.
        actor_loss = -((logps * adv * alive).mean() + args.ent_coef * ents.mean())
        critic_loss = (((values - ret) ** 2) * alive).mean()
        loss = actor_loss + 0.5 * critic_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        opt.step()

        if (it + 1) % 25 == 0 or it == 0:
            imag_ret = float((rewards * alive).sum(1).mean())
            p = trk.log(it + 1, samples=0,
                        actor_loss=float(actor_loss.detach()),
                        critic_loss=float(critic_loss.detach()),
                        imagined_return=imag_ret,
                        survival=float(alive[:, -1].mean()),
                        entropy=float(ents.mean().detach()),
                        real_transitions=len(real),
                        imagined_transitions=imagined_total)
            print(f"  update {it+1:4d}/{args.updates}  imag_return={imag_ret:7.3f}  "
                  f"survival={float(alive[:,-1].mean()):.3f}  "
                  f"H={float(ents.mean()):.3f}  "
                  f"imagined={imagined_total:,}  ({p['wall_clock_s']:.0f}s)")
            cp.save(step=it + 1, model=ac, optimizer=opt,
                    metrics={"imagined_return": imag_ret})

    secs = time.time() - t0
    ratio = imagined_total / max(len(real), 1)
    print(f"\nsample budget   real {len(real):,}   imagined {imagined_total:,}"
          f"   ratio {ratio:.1f}:1   ({secs:.0f}s)")

    out = export_policy.export(ac.actor, obs_mu.cpu().numpy(), obs_sd.cpu().numpy(), args.out)
    agree = export_policy.verify(ac.actor, obs_mu.cpu().numpy(), obs_sd.cpu().numpy(), out)
    print(f"exported {out}  (C/torch agreement {agree*100:.1f}%)")

    trk.summary(train_seconds=secs, real_transitions=len(real),
                imagined_transitions=imagined_total, imagined_to_real=ratio,
                export_agreement=agree)

    if args.eval:
        h = man["splits"]["holdout"]
        print(f"\nonline eval on HOLDOUT tracks {h['track_start']}.."
              f"{h['track_start']+h['n_tracks']-1}\n")
        subprocess.run([
            "./build/race_eval", "--bots", "policy,centerline,whisker",
            "--policy", str(out), "--tracks", str(h["n_tracks"]),
            "--track-start", str(h["track_start"]),
            "--episodes", str(args.eval_episodes), "--threads", "10",
            "--frameskip", str(man["frameskip"]),
            "--max-steps", str(man["max_episode_steps"]),
        ], check=False)
    trk.finish()


if __name__ == "__main__":
    main()
