"""Scaled multi-task encoder pretraining, for transfer to imitation learning.

The first transfer experiment (E11) handicapped pretraining badly: 702 gradient
steps against BC's 2,000, on 60k of 817k available transitions, with a single
objective. In an environment that generates data at ~1M steps/s that is not
rigour, it is just a worse experiment. Unlimited cheap dynamics data is the
whole premise of pretraining, so here it gets its best shot.

What stays controlled is the FINETUNING side -- equal gradient steps, identical
architecture, same seeds and eval. Blurring that is how a null result was
turned into a spurious 7-sigma one.

Four heads, all reading z = encoder(obs):

  forward     [z, a, geom] -> delta s     how the world moves
  inverse     [z_t, z_t+1] -> a_t         which action explains a transition.
                                          Structurally the closest thing to a
                                          policy there is, and a policy is what
                                          we are transferring to.
  geometry    z -> geom (44)              PRIVILEGED DISTILLATION: learn to
                                          infer the track layout from two
                                          whisker readings alone. The policy
                                          never gets geometry -- it is worth
                                          +10.5 accuracy points when handed to
                                          BC directly -- so teaching the encoder
                                          to reconstruct it is teaching it to
                                          recover information the observation
                                          does not contain.
  termination [z, a] -> done              what kills you

The geometry head is the one that matters beyond this experiment. "Infer
hand-crafted semantic state from a limited observation" is the sim-to-real goal
restated: real RGB has no game state, and something has to infer it. If
privileged distillation works on 30 floats it is direct evidence for the RGB
plan.

    python3 -m racewm.pretrain --steps 10000 --heads fwd,inv,geom,term
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract, dataset, geometry, tracking
from .transfer import Encoder

ALL_HEADS = ("fwd", "inv", "geom", "term")


class MultiTask(nn.Module):
    def __init__(self, hidden=256, enc_layers=2, heads=ALL_HEADS):
        super().__init__()
        self.encoder = Encoder(contract.OBS_SIZE, hidden, enc_layers)
        self.heads = tuple(heads)
        A, G, D = contract.NUM_ACTIONS, geometry.GEOM_DIM, contract.OBS_SIZE

        def mlp(d_in, d_out):
            return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d_out))

        if "fwd" in heads:  self.fwd = mlp(hidden + A + G, D)
        if "inv" in heads:  self.inv = mlp(2 * hidden, A)
        if "geom" in heads: self.geom = mlp(hidden, G)
        if "term" in heads: self.term = mlp(hidden + A, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--heads", default="fwd,inv,geom,term")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--no-expert", action="store_true",
                    help="exclude the expert slice, so no expert action ever "
                         "reaches the encoder")
    ap.add_argument("--out", default="checkpoints/pretrain/mt")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    heads = tuple(h for h in args.heads.split(",") if h)
    slices = (("whisker", "noisy", "random") if args.no_expert
              else dataset.WM_SLICES)

    tr = dataset.load_split(args.corpus, "train", slices=slices,
                            with_geometry=True, include_terminal=True)
    print(f"pretraining on {len(tr):,} transitions  slices={slices}")
    print(f"  heads={heads}  steps={args.steps:,}  batch={args.batch}")
    print(f"  samples presented: {args.steps * args.batch:,}")

    obs = torch.from_numpy(tr.obs)
    nxt = torch.from_numpy(tr.next_obs)
    act = torch.from_numpy(tr.action)
    gm = torch.from_numpy(tr.geom)
    done = torch.from_numpy(tr.done.astype(np.float32))
    delta = nxt - obs
    alive = (1.0 - done)

    mu, sd = obs.mean(0, keepdim=True), obs.std(0, keepdim=True).clamp_min(1e-6)
    gmu, gsd = gm.mean(0, keepdim=True), gm.std(0, keepdim=True).clamp_min(1e-6)
    dmu, dsd = delta.mean(0, keepdim=True), delta.std(0, keepdim=True).clamp_min(1e-8)
    on, nn_ = (obs - mu) / sd, (nxt - mu) / sd
    gn, dn = (gm - gmu) / gsd, (delta - dmu) / dsd
    pos_w = torch.tensor([(len(done) - done.sum()) / done.sum().clamp_min(1)])

    net = MultiTask(args.hidden, args.enc_layers, heads)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    trk = tracking.RunTracker(Path(args.out).name, config=dict(vars(args)),
                              mode="wandb" if args.wandb else "off",
                              tags=["pretrain", "transfer"])

    g = torch.Generator().manual_seed(args.seed)
    n = len(on)
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        a1 = torch.zeros(len(idx), contract.NUM_ACTIONS)
        a1[torch.arange(len(idx)), act[idx]] = 1.0
        z = net.encoder(on[idx])
        m = alive[idx]                      # terminal rows have a stub next_obs
        parts, loss = {}, 0.0

        if "fwd" in heads:
            pred = net.fwd(torch.cat([z, a1, gn[idx]], 1))
            l = (((pred - dn[idx]) ** 2).mean(1) * m).sum() / m.sum().clamp_min(1)
            parts["fwd"] = float(l.detach()); loss = loss + l
        if "inv" in heads:
            z2 = net.encoder(nn_[idx])
            logits = net.inv(torch.cat([z, z2], 1))
            l = (nn.functional.cross_entropy(logits, act[idx], reduction="none")
                 * m).sum() / m.sum().clamp_min(1)
            parts["inv"] = float(l.detach()); loss = loss + l
        if "geom" in heads:
            l = ((net.geom(z) - gn[idx]) ** 2).mean()
            parts["geom"] = float(l.detach()); loss = loss + l
        if "term" in heads:
            l = nn.functional.binary_cross_entropy_with_logits(
                net.term(torch.cat([z, a1], 1)).squeeze(-1), done[idx],
                pos_weight=pos_w)
            parts["term"] = float(l.detach()); loss = loss + 0.1 * l

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % 2000 == 0 or step == 0:
            p = trk.log(step + 1, samples=args.batch, total=float(loss.detach()), **parts)
            msg = "  ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"  step {step+1:6d}/{args.steps}  {msg}  ({p['wall_clock_s']:.0f}s)",
                  flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": net.encoder.state_dict(), "hidden": args.hidden,
                "enc_layers": args.enc_layers, "heads": list(heads),
                "obs_mu": mu, "obs_sd": sd, "slices": list(slices),
                "steps": args.steps, "batch": args.batch,
                "transitions_seen": args.steps * args.batch,
                "schema_hash": contract.schema_hash()}, out / "encoder.pt")
    print(f"\nsaved {out/'encoder.pt'}  ({time.time()-t0:.0f}s)")
    trk.summary(train_seconds=time.time() - t0, **parts)
    trk.finish()


if __name__ == "__main__":
    main()
