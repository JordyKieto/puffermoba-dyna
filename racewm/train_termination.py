"""Termination head: P(episode ends | s_t, a_t, local geometry).

Why this is a separate model and not a whisker threshold on the predicted
state: the crash state is never observable. c_step zeroes the whiskers, logs,
resets, and *then* computes observations, so the post-crash observation
belongs to the next episode. The corpus contains zero states with a whisker
below 0.05 -- terminal rows included.

So the world model cannot learn what a crash looks like. It can learn that one
is about to happen, from the state and action that cause it. That is what this
predicts, and it is why Dreamer carries an explicit `continue` head rather than
reading termination off reconstructed observations.

Without it, an actor trained in imagination optimizes in a region the dynamics
model invents: measured, 33% of imagined states had a whisker below 1e-3, a
value that never occurs in real data.

    python3 -m racewm.train_termination --corpus data/corpus
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, geometry, tracking


class TerminationHead(nn.Module):
    def __init__(self, d_in, hidden=256, layers=3):
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--run-name", default="termination")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dataset.assert_disjoint(args.corpus)

    tr = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                            with_geometry=True, include_terminal=True,
                            max_rows=args.max_rows, seed=args.seed)
    va = dataset.load_split(args.corpus, "val", slices=dataset.WM_SLICES,
                            with_geometry=True, include_terminal=True)
    print(f"train {len(tr):,} ({tr.done.sum():,} terminal, {100*tr.done.mean():.2f}%)")
    print(f"val   {len(va):,} ({va.done.sum():,} terminal)")

    x_tr = torch.from_numpy(dataset.wm_inputs(tr))
    x_va = torch.from_numpy(dataset.wm_inputs(va))
    y_tr = torch.from_numpy(tr.done.astype(np.float32))
    y_va = torch.from_numpy(va.done.astype(np.float32))

    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_trn, x_van = (x_tr - mu) / sd, (x_va - mu) / sd

    # Terminals are ~0.5% of rows. Unweighted, the head would predict "never
    # ends" and score 99.5% accuracy while being useless.
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / y_tr.sum().clamp_min(1)])
    print(f"pos_weight {float(pos_weight):.1f}")

    net = TerminationHead(x_tr.shape[1], args.hidden, args.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(x_trn) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    cfg = dict(vars(args), d_in=int(x_tr.shape[1]), n_train=len(tr),
               pos_rate=float(tr.done.mean()))
    cp = ckpt_mod.Checkpointer("checkpoints", args.run_name, config=cfg,
                               metric_name="val_ap", metric_mode="max")
    trk = tracking.RunTracker(args.run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase2", "termination"])

    g = torch.Generator().manual_seed(args.seed)
    n = len(x_trn)
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(x_trn[idx]), y_tr[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach())

        net.eval()
        with torch.no_grad():
            logit = net(x_van)
            prob = torch.sigmoid(logit)
        # Average precision: accuracy is meaningless at a 0.5% positive rate.
        order = torch.argsort(prob, descending=True)
        hits = y_va[order]
        cum = torch.cumsum(hits, 0)
        prec = cum / torch.arange(1, len(hits) + 1)
        ap = float((prec * hits).sum() / hits.sum().clamp_min(1))
        rec50 = float(hits[:int(0.005 * len(hits))].sum() / hits.sum().clamp_min(1))
        p = trk.log(epoch + 1, samples=n, train_loss=tot / steps,
                    val_ap=ap, recall_at_top_0p5pct=rec50)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss={tot/steps:.4f}  "
              f"val_AP={ap:.4f}  recall@top0.5%={rec50:.3f}  ({p['wall_clock_s']:.0f}s)")
        cp.save(step=epoch + 1, model=net, optimizer=opt,
                extra={"mu": mu, "sd": sd}, metrics={"val_ap": ap})

    print(f"\ntrained in {time.time()-t0:.0f}s   final val AP {ap:.4f}")
    print(f"  (a head that always says 'no' would score AP ~= {float(y_va.mean()):.4f})")
    trk.summary(val_ap=ap, baseline_ap=float(y_va.mean()))
    trk.finish()


if __name__ == "__main__":
    main()
