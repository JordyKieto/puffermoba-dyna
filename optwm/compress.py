"""Gate test for a world model over optimizer trajectories.

The recipe is the same one racewm applies to a game: authoritative dynamics ->
trajectories -> learned dynamics -> long imagined rollout -> authoritative
correction. Here the "environment" is SGD and the "state" is the weight vector.

Before any of that is worth building, one question decides it:

    How compressible are the deltas  D_t = theta_{t+k} - theta_t ?

If a small latent explains the meaningful part of training, that is the analogue
of a structured state and the rest of the recipe has something to stand on. If
not, nothing downstream can work and we stop here.

Three ways this test can lie to you, all of which are controlled for:

  1. Fitting PCA and scoring it on the SAME trajectories. Any basis fits the
     data it was built from. We fit on one set of runs and score on held-out
     runs, which is the only version that means anything.
  2. Every delta pointing the same way. If training is dominated by one global
     drift direction, rank-1 "explains" everything while telling you nothing.
     We report the mean direction's share separately and re-run the analysis
     with it removed.
  3. Weight MSE. Reconstructing weights accurately is not the goal and is
     nearly meaningless. The functional test asks the only question that
     matters: apply a rank-r reconstruction of the delta and does the model's
     LOSS land where the true delta would have put it?

Run:  python -m optwm.compress --trajectories 16 --steps 600
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- task + model

class ModAdd:
    """a + b mod P, presented as a 3-token sequence. Small, clean, and known to
    have non-trivial optimisation dynamics (the grokking setup), which is what
    we want: a task whose trajectory has structure worth predicting."""

    def __init__(self, p=97, device="cpu"):
        self.p, self.device = p, device
        a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
        self.x = torch.stack([a.flatten(), b.flatten(),
                              torch.full((p * p,), p)], dim=1).to(device)
        self.y = ((a.flatten() + b.flatten()) % p).to(device)

    def split(self, frac=0.5, seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.x), generator=g)
        n = int(frac * len(idx))
        return (self.x[idx[:n]], self.y[idx[:n]],
                self.x[idx[n:]], self.y[idx[n:]])


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d=48, n_layer=2, n_head=4, seq=3):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, d)
        self.pos = nn.Parameter(torch.zeros(seq, d))
        layer = nn.TransformerEncoderLayer(
            d, n_head, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos
        h = self.blocks(h)
        return self.head(self.ln(h[:, -1]))


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def unflat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


# ------------------------------------------------------------ trajectory gen

def run_trajectory(task, seed, lr, wd, steps, every, batch, device="cpu",
                   init_seed=None):
    """One authoritative optimizer run. Returns checkpoints as flat vectors.

    `init_seed` separates WHERE the run starts from HOW it proceeds. With a
    shared init_seed every trajectory begins at the same theta_0, so a basis
    shared across runs is at least possible. With distinct inits it is not:
    permutation and scaling symmetries put each run in its own coordinate
    frame, and no weight-space basis can transfer between them even if the
    dynamics are perfectly low-dimensional.
    """
    torch.manual_seed(seed if init_seed is None else init_seed)
    model = TinyTransformer(task.p).to(device)
    xtr, ytr, xte, yte = task.split(0.5, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed + 9999)

    ckpts, losses, tr_losses = [], [], []
    for s in range(steps + 1):
        if s % every == 0:
            ckpts.append(flat(model).clone())
            with torch.no_grad():
                losses.append(float(F.cross_entropy(model(xte), yte)))
                tr_losses.append(float(F.cross_entropy(model(xtr), ytr)))
        idx = torch.randint(0, len(xtr), (batch,), generator=g)
        loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (torch.stack(ckpts), torch.tensor(losses),
            torch.tensor(tr_losses), (xte, yte, xtr, ytr))


# ------------------------------------------------------------------- analysis

def explained(D, basis, mean):
    """Fraction of held-out delta energy captured by `basis`."""
    C = D - mean
    proj = (C @ basis) @ basis.T
    return float(1.0 - (C - proj).pow(2).sum() / C.pow(2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=20, help="checkpoint interval k")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--holdout", type=int, default=5, help="trajectories held out")
    ap.add_argument("--ranks", type=str, default="1,2,4,8,16,32,64,128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared-init", action="store_true",
                    help="all trajectories start from the SAME theta_0, "
                         "varying only data order, lr and weight decay. This "
                         "removes the permutation-symmetry confound that makes "
                         "cross-seed weight bases impossible a priori.")
    args = ap.parse_args()

    task = ModAdd(97)
    lrs = [1e-3, 3e-3, 1e-2]
    wds = [0.0, 1.0]

    print(f"generating {args.trajectories} trajectories x {args.steps} steps "
          f"(checkpoint every {args.every})"
          + ("  [shared theta_0]" if args.shared_init else "  [distinct inits]"))
    t0 = time.time()
    trajs, all_losses, evals, tr_all = [], [], [], []
    for i in range(args.trajectories):
        lr, wd = lrs[i % len(lrs)], wds[(i // len(lrs)) % len(wds)]
        C, L, TL, ev = run_trajectory(task, args.seed * 1000 + i, lr, wd,
                                      args.steps, args.every, args.batch,
                                      init_seed=args.seed if args.shared_init
                                      else None)
        trajs.append(C); all_losses.append(L); evals.append(ev)
        tr_all.append(TL)
        print(f"  traj {i:2d}  lr {lr:<6} wd {wd:<4} "
              f"train {TL[0]:.3f} -> {TL[-1]:.3f}   "
              f"test {L[0]:.3f} -> {L[-1]:.3f}  ({time.time()-t0:.0f}s)")

    P = trajs[0].shape[1]
    print(f"\nmodel: {P:,} parameters;  {len(trajs[0])} checkpoints/trajectory")

    # deltas, tagged by trajectory so the split is by RUN not by sample
    D, tag, ttag = [], [], []
    for i, C in enumerate(trajs):
        D.append(C[1:] - C[:-1])
        tag += [i] * (len(C) - 1)
        ttag += list(range(len(C) - 1))
    D = torch.cat(D); tag = torch.tensor(tag); ttag = torch.tensor(ttag)
    print(f"deltas: {D.shape[0]} x {D.shape[1]:,}")

    def analyse(fit_m, label, note):
        Dfit, Dho = D[fit_m], D[~fit_m]
        print(f"\n--- {label}\n    {note}")
        print(f"    fit {int(fit_m.sum())} deltas / score {int((~fit_m).sum())} deltas")
        mean = Dfit.mean(0, keepdim=True)
        mdir = mean / mean.norm()
        drift = float((Dho @ mdir.T).pow(2).sum() / Dho.pow(2).sum())
        _, S, Vh = torch.linalg.svd(Dfit - mean, full_matrices=False)
        rs = [r for r in map(int, args.ranks.split(",")) if r <= len(S)]
        gn = torch.randn_like(Dfit) * Dfit.std()
        gmean = gn.mean(0, keepdim=True)
        _, _, gVh = torch.linalg.svd(gn - gmean, full_matrices=False)
        gho = torch.randn_like(Dho) * Dho.std()
        print(f"    global-drift direction alone: {100*drift:.1f}% of held-out energy")
        print(f"    {'rank':>6} {'held-out EV':>12} {'in-fit EV':>10} {'gauss ctrl':>11}")
        for r in rs:
            B, gB = Vh[:r].T, gVh[:r].T
            print(f"    {r:>6} {100*explained(Dho, B, mean):>11.1f}% "
                  f"{100*explained(Dfit, B, mean):>9.1f}% "
                  f"{100*explained(gho, gB, gmean):>10.1f}%")
        return mean, Vh

    ho = set(range(args.trajectories - args.holdout, args.trajectories))
    run_m = torch.tensor([t.item() not in ho for t in tag])
    mean, Vh = analyse(run_m, "SPLIT BY RUN (generalise to a new training run)",
                       "the condition a cross-run simulator needs")

    cut = int(0.6 * ttag.max().item())
    time_m = ttag < cut
    analyse(time_m, "SPLIT BY TIME (same runs, predict their later steps)",
            "the most favourable case: jump ahead within a run already seen")

    # ---------------------------------------------------- functional test
    print("\nfunctional test: apply a rank-r reconstruction of the delta and "
          "measure LOSS,\nnot weight error. theta' = theta_t + reconstruct(D_t)")
    ranks = [r for r in map(int, args.ranks.split(",")) if r <= Vh.shape[0]]
    hi = args.trajectories - 1
    C, TL, (xte, yte, xtr, ytr) = trajs[hi], tr_all[hi], evals[hi]
    model = TinyTransformer(task.p)
    # the step where the optimizer actually did the most work -- a step that
    # changes nothing cannot distinguish a good reconstruction from a bad one
    j = int(torch.argmax((TL[:-1] - TL[1:]).abs()))
    theta, true_next, Dtrue = C[j], C[j + 1], (C[j + 1] - C[j]).unsqueeze(0)

    def loss_of(vec):
        unflat(model, vec)
        with torch.no_grad():
            return float(F.cross_entropy(model(xtr), ytr))
    print(f"  (checkpoint {j} of {len(C)-1}, the largest real loss change)")

    l0, l1 = loss_of(theta), loss_of(true_next)
    print(f"  loss at theta_t           {l0:.4f}")
    print(f"  loss at true theta_t+k    {l1:.4f}   (true step: {l1-l0:+.4f})")
    print(f"  {'rank':>6} {'loss':>9} {'vs true':>9} {'delta captured':>15}")
    for r in ranks:
        B = Vh[:r].T
        rec = mean + ((Dtrue - mean) @ B) @ B.T
        lr_ = loss_of(theta + rec.squeeze(0))
        frac = (lr_ - l0) / (l1 - l0) if abs(l1 - l0) > 1e-9 else float("nan")
        print(f"  {r:>6} {lr_:>9.4f} {lr_-l1:>+9.4f} {100*frac:>14.0f}%")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
