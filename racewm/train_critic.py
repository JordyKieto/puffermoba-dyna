"""Learned critic V(s), fitted on REAL transitions by TD(0).

Replaces the hand-rolled `alive x clearance` heuristic that every planner was
using. The evidence that heuristic was the bottleneck: depth-1 greedy search
scored 0.775 online, below the `straight` bot -- and depth-1 search *is* the
value function, with the model barely involved.

Fitted on real data, not in imagination, deliberately. A critic trained inside
the world model inherits the model's optimistic errors, and a planner then
searches for exactly those states. Real transitions give a value function the
model cannot talk out of.

TD(0) with a target network:

    V(s) <- r + gamma * (1 - done) * V_target(s')

`done` comes from the corpus with include_terminal=True, so terminal rows
correctly target just `r` -- which is what teaches the critic that crashing
forfeits all future return. That, rather than an arbitrary constant penalty, is
how "do not crash" should be expressed: the env pays at most 0.2 per step,
while the old hand-set penalty was 5.0, so planners were paralysed into
avoidance and still crashed.

    python3 -m racewm.train_critic --corpus data/corpus --gamma 0.97
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, tracking


class Critic(nn.Module):
    def __init__(self, d_in=contract.OBS_SIZE, hidden=256, layers=3):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--slices", default="expert,whisker",
                    help="which policies' returns to value; the critic is "
                         "on-policy w.r.t. whatever generated these")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--target-tau", type=float, default=0.01)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--no-geometry", action="store_true",
                    help="ablation: value the state without the track ahead")
    ap.add_argument("--run-name", default="critic")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    slices = tuple(args.slices.split(","))
    dataset.assert_disjoint(args.corpus)

    # Geometry matters here for the same reason it matters to the termination
    # head: how much a state is worth depends on the track ahead of it. Without
    # it the critic separated healthy states from about-to-crash states by only
    # 0.165, which is not enough signal for a planner to steer on.
    use_geom = not args.no_geometry
    tr = dataset.load_split(args.corpus, "train", slices=slices, with_geometry=use_geom,
                            include_terminal=True, max_rows=args.max_rows, seed=args.seed)
    va = dataset.load_split(args.corpus, "val", slices=slices, with_geometry=use_geom,
                            include_terminal=True)
    print(f"critic on {slices}: train {len(tr):,} ({tr.done.sum():,} terminal)  "
          f"val {len(va):,}")
    print(f"  mean step reward {tr.reward.mean():.4f}   "
          f"implied V at gamma={args.gamma}: ~{tr.reward.mean()/(1-args.gamma):.3f}")

    # The successor's geometry must come from the successor's own pose, so it
    # is gathered rather than reused from s_t.
    def feats(c, obs):
        if not use_geom:
            return torch.from_numpy(obs.astype(np.float32))
        from . import geometry as geo, shards as sh
        bank = geo.TrackBank(sh.load_tracks(f"{args.corpus}/{c.split}.tracks.bin"),
                             c.header)
        slots = bank.slots(c.env_idx)
        with torch.no_grad():
            g = bank.local_window(torch.from_numpy(obs.astype(np.float32)), slots)
        return torch.cat([torch.from_numpy(obs.astype(np.float32)), g], dim=1)

    s = feats(tr, tr.obs)
    s2 = feats(tr, tr.next_obs)
    r = torch.from_numpy(tr.reward)
    d = torch.from_numpy(tr.done.astype(np.float32))
    vs = feats(va, va.obs)
    vs2 = feats(va, va.next_obs)
    vr = torch.from_numpy(va.reward)
    vd = torch.from_numpy(va.done.astype(np.float32))

    mu, sd = s.mean(0, keepdim=True), s.std(0, keepdim=True).clamp_min(1e-6)
    sn, s2n = (s - mu) / sd, (s2 - mu) / sd
    vsn, vs2n = (vs - mu) / sd, (vs2 - mu) / sd

    d_in = s.shape[1]
    print(f"  critic input dim {d_in} ({'with' if use_geom else 'without'} geometry)")
    net = Critic(d_in, args.hidden, args.layers)
    target = copy.deepcopy(net)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(sn) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    cfg = dict(vars(args), n_train=len(tr), n_val=len(va),
               d_in=int(s.shape[1]), use_geometry=use_geom)
    cp = ckpt_mod.Checkpointer("checkpoints", args.run_name, config=cfg,
                               metric_name="val_td", metric_mode="min")
    trk = tracking.RunTracker(args.run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase2", "critic"])

    g = torch.Generator().manual_seed(args.seed)
    n = len(sn)
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            with torch.no_grad():
                # A terminal row targets just r: crashing forfeits everything
                # after it, which is the whole lesson.
                y = r[idx] + args.gamma * (1 - d[idx]) * target(s2n[idx])
            loss = nn.functional.mse_loss(net(sn[idx]), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for tp, p in zip(target.parameters(), net.parameters()):
                    tp.mul_(1 - args.target_tau).add_(args.target_tau * p)
            tot += float(loss.detach())

        net.eval()
        with torch.no_grad():
            vy = vr + args.gamma * (1 - vd) * target(vs2n)
            vpred = net(vsn)
            val_td = float(nn.functional.mse_loss(vpred, vy))
            v_alive = float(vpred[vd == 0].mean())
            v_term = float(vpred[vd == 1].mean()) if (vd == 1).any() else float("nan")
        p = trk.log(epoch + 1, samples=n, train_loss=tot / steps, val_td=val_td,
                    v_nonterminal=v_alive, v_terminal=v_term)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  td={tot/steps:.5f}  "
              f"val_td={val_td:.5f}  V(alive)={v_alive:6.3f}  V(crash)={v_term:6.3f}  "
              f"({p['wall_clock_s']:.0f}s)")
        cp.save(step=epoch + 1, model=net, optimizer=opt,
                extra={"mu": mu, "sd": sd, "gamma": args.gamma,
                       "use_geometry": use_geom, "d_in": int(d_in)},
                metrics={"val_td": val_td})

    print(f"\ntrained in {time.time()-t0:.0f}s")
    print(f"  V(non-terminal)={v_alive:.3f}  V(about to crash)={v_term:.3f}  "
          f"gap={v_alive - v_term:.3f}")
    print("  A useful critic separates these: a state one step from a crash is "
          "worth much less than a healthy one.")
    trk.summary(val_td=val_td, v_nonterminal=v_alive, v_terminal=v_term)
    trk.finish()


if __name__ == "__main__":
    main()
