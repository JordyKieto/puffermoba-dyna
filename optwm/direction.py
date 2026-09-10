"""Direction as the action: optimise validation THROUGH a learned landscape model.

Selecting a learning rate is one scalar. The real action space is the DIRECTION
of the jump in weight space. So instead of F predicting where SGD goes, learn a
model that EVALUATES a proposed move:

    V(theta, momentum, delta) -> (train loss, val loss) at theta + delta

Then the direction itself becomes optimisable: V is differentiable, so descend
on `delta` to find a move that lowers predicted validation loss, and check
whether it lowers the real one.

THE THING THAT MAKES THIS WORK OR NOT: V must have seen off-trajectory
directions. Trained only on the deltas SGD actually took, it cannot evaluate a
direction SGD would never take, and searching over directions would just walk
off the data. So the training set mixes:

  on-trajectory   delta = theta_{t+k} - theta_t          what SGD did
  scaled          the same direction, 0.25x to 4x        magnitude sensitivity
  random          matched-norm gaussian directions       the landscape around
  mixtures        interpolations of the two              in between

Labels are the TRUE losses at theta + delta, evaluated directly. No SGD needed
to make a label, which is what makes off-trajectory data cheap.

Baselines, and the last one is the one that matters:

  no move         stay at theta                    (the floor)
  SGD direction   what real training would do      (the thing to beat)
  random          matched-norm random direction    (is direction doing anything)
  V-optimised     descend on delta through V

If V-optimised does not beat the SGD direction on REAL validation loss, the
model is not finding anything SGD wasn't already going to give you.

Run:  python -m optwm.direction --runs 16 --steps 600
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .sim import init_w, make_task, numel, rollout_real, shapes_for


def fwd_batch(W, x, shapes):
    """(B, NP) weight vectors on (N, d_in) inputs -> (B, N, d_out)."""
    i, p = 0, []
    for sh in shapes:
        n = int(torch.tensor(sh).prod())
        p.append(W[:, i:i + n].view(-1, *sh)); i += n
    W1, b1, W2, b2 = p
    h = torch.relu(torch.einsum("nd,bdh->bnh", x, W1) + b1[:, None, :])
    return torch.einsum("bnh,bho->bno", h, W2) + b2[:, None, :]


def losses_batch(W, x, y, shapes, chunk=256):
    """True loss of many weight vectors at once."""
    out = []
    with torch.no_grad():
        for s in range(0, len(W), chunk):
            lg = fwd_batch(W[s:s + chunk], x, shapes)
            out.append(F_.cross_entropy(
                lg.reshape(-1, lg.shape[-1]),
                y.repeat(len(lg)), reduction="none").view(len(lg), -1).mean(1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=20, help="jump = k*res steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=24000)
    ap.add_argument("--fwidth", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--opt-steps", type=int, default=200, help="descent on delta")
    ap.add_argument("--opt-lr", type=float, default=0.02)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid); NP = numel(SH)
    task = make_task(SH, 11)
    xtr, ytr, xte, yte = task
    # THREE splits, not two. V is trained on VAL-loss labels and then we
    # optimise a direction against V's val prediction, so improving val loss is
    # partly circular -- the search can fit the val set through the surrogate.
    # TEST is never seen by V or by the optimiser, and is the only unbiased
    # check that a found direction actually generalises.
    h = len(xte) // 2
    xva, yva, xts, yts = xte[:h], yte[:h], xte[h:], yte[h:]
    print(f"splits: {len(xtr)} train / {len(xva)} val (V's labels) / "
          f"{len(xts)} TEST (untouched)")
    w0 = init_w(SH, 0)
    lrs = [0.02, 0.05, 0.1, 0.2]
    print(f"student {NP:,} params | jump {a.k*a.res} steps | "
          f"action = a {NP}-dim direction")

    t0 = time.time()
    TW, TB = [], []
    for i in range(a.runs):
        C, L, B = rollout_real(task, w0, lrs[i % 4], a.steps, a.res, a.batch,
                               SH, seed=i)
        TW.append(C); TB.append(B)
    TW, TB = torch.stack(TW), torch.stack(TB)
    NCK = TW.shape[1]
    print(f"{a.runs} trajectories ({time.time()-t0:.0f}s)")

    # ---- dataset of (state, proposed move) -> true losses at the moved point
    g = torch.Generator().manual_seed(1)
    n = a.samples
    j = torch.randint(0, a.runs, (n,), generator=g)
    i = torch.randint(1, NCK - a.k - 1, (n,), generator=g)
    TH, BF = TW[j, i], TB[j, i]
    on = TW[j, i + a.k] - TH                       # what SGD did
    nrm = on.norm(dim=1, keepdim=True)
    rnd = torch.randn(n, NP, generator=g)
    rnd = rnd / rnd.norm(dim=1, keepdim=True) * nrm
    kind = torch.randint(0, 4, (n,), generator=g)
    scale = torch.exp(torch.rand(n, 1, generator=g) * 2.77 - 1.386)  # 0.25..4
    mixw = torch.rand(n, 1, generator=g)
    DL = torch.where(kind[:, None] == 0, on,
         torch.where(kind[:, None] == 1, on * scale,
         torch.where(kind[:, None] == 2, rnd,
                     mixw * on + (1 - mixw) * rnd)))
    WT = TH + DL
    ytr_l = losses_batch(WT, xtr, ytr, SH)
    yte_l = losses_batch(WT, xva, yva, SH)
    print(f"{n:,} (state, move) pairs; kinds = on/scaled/random/mixed "
          f"({time.time()-t0:.0f}s)")
    print(f"  V's val loss over the dataset: min {float(yte_l.min()):.4f} "
          f"mean {float(yte_l.mean()):.4f} max {float(yte_l.max()):.4f}")

    dsd = DL.std(0).clamp_min(1e-8)
    lmu, lsd = ytr_l.mean(), ytr_l.std().clamp_min(1e-8)
    qmu, qsd = yte_l.mean(), yte_l.std().clamp_min(1e-8)

    V = nn.Sequential(nn.Linear(3 * NP, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, 2))
    opt = torch.optim.AdamW(V.parameters(), lr=1e-3, weight_decay=1e-4)
    ntr = int(0.9 * n)

    def predict(th, bf, dl):
        o = V(torch.cat([th, bf, dl / dsd], 1))
        return o[:, :1] * lsd + lmu, o[:, 1:] * qsd + qmu

    for e in range(a.epochs):
        b = torch.randint(0, ntr, (256,))
        pl, pv = predict(TH[b], BF[b], DL[b])
        loss = (((pl.squeeze(1) - ytr_l[b]) / lsd).pow(2).mean()
                + ((pv.squeeze(1) - yte_l[b]) / qsd).pow(2).mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        pl, pv = predict(TH[ntr:], BF[ntr:], DL[ntr:])
        err = float((pv.squeeze(1) - yte_l[ntr:]).abs().mean())
        vc = pv.squeeze(1) - pv.mean(); rc = yte_l[ntr:] - yte_l[ntr:].mean()
        r = float((vc * rc).sum() / (vc.norm() * rc.norm()))
    print(f"V trained: held-out val-loss MAE {err:.4f}, corr {r:+.3f} "
          f"({time.time()-t0:.0f}s)\n")

    # ---- optimise the direction through V, then check it for real
    hold = rollout_real(task, w0, 0.07, a.steps, a.res, a.batch, SH, seed=777)
    Ch, Lh, Bh = hold
    gg = torch.Generator().manual_seed(9)
    NAMES = ("no move", "SGD dir", "random", "V-optimised")
    res = {k: [] for k in NAMES}
    rest = {k: [] for k in NAMES}
    print(f"  {'':>6}{'------ VAL (V optimised this) ------':>39}"
          f"{'------ TEST (untouched) ------':>33}")
    print(f"  {'state':>6} {'no move':>9} {'SGD':>9} {'rand':>9} {'V-opt':>9} "
          f"{'no move':>9} {'SGD':>9} {'rand':>9} {'V-opt':>9}")
    for t in range(a.trials):
        si = 1 + int(torch.randint(1, NCK - a.k - 1, (1,), generator=gg))
        th, bf = Ch[si].unsqueeze(0), Bh[si].unsqueeze(0)
        sgd = (Ch[si + a.k] - Ch[si]).unsqueeze(0)
        rd = torch.randn(1, NP, generator=gg)
        rd = rd / rd.norm() * sgd.norm()

        d = sgd.clone().requires_grad_(True)
        o2 = torch.optim.Adam([d], lr=a.opt_lr * float(sgd.norm()) / NP ** 0.5)
        for _ in range(a.opt_steps):
            _, pv = predict(th, bf, d)
            o2.zero_grad(set_to_none=True); pv.sum().backward(); o2.step()
        with torch.no_grad():
            _, pv_final = predict(th, bf, d)

        cand = torch.cat([th, th + sgd, th + rd, th + d.detach()])
        vl = losses_batch(cand, xva, yva, SH)
        tl = losses_batch(cand, xts, yts, SH)
        for q, kk in enumerate(NAMES):
            res[kk].append(float(vl[q])); rest[kk].append(float(tl[q]))
        if t < 8:
            print(f"  {si*a.res:>6} " +
                  " ".join(f"{float(vl[q]):>9.4f}" for q in range(4)) + " " +
                  " ".join(f"{float(tl[q]):>9.4f}" for q in range(4)))

    print(f"\n  mean loss over {a.trials} states:")
    print(f"    {'':<14} {'VAL':>9} {'TEST':>9}")
    for kk in NAMES:
        print(f"    {kk:<14} {st.mean(res[kk]):>9.4f} {st.mean(rest[kk]):>9.4f}")
    dv = st.mean(res['SGD dir']) - st.mean(res['V-optimised'])
    dt = st.mean(rest['SGD dir']) - st.mean(rest['V-optimised'])
    dnv = st.mean(rest['no move']) - st.mean(rest['V-optimised'])
    print(f"\n  V-opt beats SGD direction:  VAL {dv:+.4f}   TEST {dt:+.4f}")
    print(f"  V-opt beats no-move on TEST: {dnv:+.4f}")
    print(f"\n  VAL is the metric V was trained on and optimised against, so a")
    print(f"  gain there is expected and partly circular. TEST is untouched by")
    print(f"  both -- only a TEST gain shows the direction genuinely generalises.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
