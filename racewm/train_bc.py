"""Behaviour cloning on the expert + whisker mix, evaluated online on holdout.

Offline accuracy is not the goal. A policy can match held-out actions closely
and still crash immediately, because BC error compounds through the states it
steers itself into -- so the number that matters is behavioral, measured by
running the policy on tracks it has never seen.

    python3 -m racewm.train_bc --corpus data/corpus --epochs 15 --eval
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
from . import contract, dataset, export_policy, tracking


class PolicyMLP(nn.Module):
    def __init__(self, d_in, n_act=contract.NUM_ACTIONS, hidden=256, layers=3):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, n_act))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--slices", default="expert,whisker",
                    help="which corpus slices to clone (never the noisy ones)")
    ap.add_argument("--balance", action="store_true",
                    help="class-weight the loss; LEFT outnumbers RIGHT ~1.8:1 "
                         "because generated tracks share a handedness")
    ap.add_argument("--privileged", action="store_true",
                    help="give the policy the track geometry the expert used")
    ap.add_argument("--out", default="checkpoints/bc/policy.rpo")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--eval", action="store_true", help="online eval on holdout after training")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--real-budget", type=int, default=None,
                    help="cap on real transitions, for sample-efficiency studies")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    slices = tuple(args.slices.split(","))
    run_name = args.run_name or f"bc_{'_'.join(slices)}{'_priv' if args.privileged else ''}"

    dataset.assert_disjoint(args.corpus)
    man = dataset.load_manifest(args.corpus)
    train = dataset.load_split(args.corpus, "train", slices=slices,
                               with_geometry=args.privileged,
                               max_rows=args.real_budget, seed=args.seed)
    val = dataset.load_split(args.corpus, "val", slices=slices,
                             with_geometry=args.privileged)
    print(f"BC on {slices}  train {len(train):,}  val {len(val):,}")
    print(f"  action counts: {dict(zip(contract.ACTION_NAMES, np.bincount(train.action, minlength=3)))}")

    x_tr = torch.from_numpy(dataset.bc_inputs(train, args.privileged))
    y_tr = torch.from_numpy(train.action)
    x_va = torch.from_numpy(dataset.bc_inputs(val, args.privileged))
    y_va = torch.from_numpy(val.action)

    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_trn, x_van = (x_tr - mu) / sd, (x_va - mu) / sd

    weight = None
    if args.balance:
        counts = np.bincount(train.action, minlength=contract.NUM_ACTIONS)
        weight = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                              dtype=torch.float32)
        print(f"  class weights: {weight.tolist()}")

    net = PolicyMLP(x_tr.shape[1], hidden=args.hidden, layers=args.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(x_trn) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    cfg = dict(vars(args), n_train=len(train), n_val=len(val),
               d_in=int(x_tr.shape[1]), corpus_manifest=man,
               params=sum(p.numel() for p in net.parameters()))
    cp = ckpt_mod.Checkpointer("checkpoints", run_name, config=cfg,
                               metric_name="val_acc", metric_mode="max")
    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase1", "bc"])
    trk.watch_params(net)

    g = torch.Generator().manual_seed(args.seed)
    n = len(x_trn)
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.cross_entropy(net(x_trn[idx]), y_tr[idx], weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        net.eval()
        with torch.no_grad():
            logits = net(x_van)
            acc = float((logits.argmax(1) == y_va).float().mean())
        p = trk.log(epoch + 1, samples=n, train_loss=tot / steps, val_acc=acc)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  " + tracking.format_row(
            p, ("step", "train_loss", "val_acc", "samples_seen", "wall_clock_s")))
        cp.save(step=epoch + 1, model=net, optimizer=opt, metrics={"val_acc": acc})
    secs = time.time() - t0

    out = export_policy.export(net, mu.numpy(), sd.numpy(), args.out)
    agree = export_policy.verify(net, mu.numpy(), sd.numpy(), out)
    print(f"\nexported {out}  (C/torch argmax agreement: {agree*100:.1f}%)")
    if agree < 0.99:
        print("  WARNING: export does not reproduce the torch policy")

    trk.summary(train_seconds=secs, val_acc=acc, export_agreement=agree)

    if args.eval:
        if args.privileged:
            print("\n(skipping online eval: a privileged policy needs geometry "
                  "the C harness does not feed it)")
        else:
            h = man["splits"]["holdout"]
            print(f"\nonline eval on HOLDOUT tracks {h['track_start']}.."
                  f"{h['track_start']+h['n_tracks']-1}, {args.eval_episodes} episodes/track\n")
            cmd = [
                "./build/race_eval", "--bots", "policy,centerline,whisker,random",
                "--policy", str(out), "--tracks", str(h["n_tracks"]),
                "--track-start", str(h["track_start"]),
                "--episodes", str(args.eval_episodes), "--threads", "10",
                "--frameskip", str(man["frameskip"]),
                "--max-steps", str(man["max_episode_steps"]),
            ]
            if args.json_out:
                cmd += ["--json", args.json_out]
            subprocess.run(cmd, check=False)
    trk.finish()


if __name__ == "__main__":
    main()
