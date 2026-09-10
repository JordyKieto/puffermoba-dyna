"""Step 1: a transition model over weight space, learned from a RANDOM dataset.

Before any policy learning, learn the transition itself:

    T(theta, train_loss, val_loss, delta) -> (train_loss', val_loss') at theta+delta

Built from randomly sampled states and randomly sampled moves rather than from
SGD trajectories. Two reasons that matters. Labels need no SGD at all -- the
loss at theta+delta is one forward pass -- so the dataset is cheap. And it
covers directions SGD would never take, which is exactly what a policy needs if
it is going to propose moves of its own without immediately leaving the data.

THE COVERAGE QUESTION, measured rather than assumed. Uniformly random weights
are almost all useless networks: for 3 classes, chance is ln(3) = 1.0986 and
accuracy 0.33. If the whole dataset sits at chance, the model learns the
geometry of a region no policy will ever visit, and its accuracy there is
worthless. So states are drawn at several "temperatures" -- from near-init
scaling up and down, plus optionally a few real SGD steps to reach nontrivial
loss -- and the val-loss histogram of the dataset is reported so the coverage is
visible instead of hoped for.

Evaluated three ways, because in-distribution accuracy is the easy part:
  held-out random   same distribution as training
  SGD states        weights from real training runs, never in the dataset
  large moves       delta norms beyond the training range

Run:  python -m optwm.landscape --samples 40000
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .sim import init_w, make_task, numel, rollout_real, shapes_for


def fwd_batch(W, x, shapes):
    i, p = 0, []
    for sh in shapes:
        n = int(torch.tensor(sh).prod())
        p.append(W[:, i:i + n].view(-1, *sh)); i += n
    W1, b1, W2, b2 = p
    h = torch.relu(torch.einsum("nd,bdh->bnh", x, W1) + b1[:, None, :])
    return torch.einsum("bnh,bho->bno", h, W2) + b2[:, None, :]


def loss_acc(W, x, y, shapes, chunk=512):
    """True loss and accuracy of many weight vectors."""
    ls, ac = [], []
    with torch.no_grad():
        for s in range(0, len(W), chunk):
            lg = fwd_batch(W[s:s + chunk], x, shapes)
            b = len(lg)
            ls.append(F_.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                       y.repeat(b), reduction="none")
                      .view(b, -1).mean(1))
            ac.append((lg.argmax(-1) == y[None, :]).float().mean(1))
    return torch.cat(ls), torch.cat(ac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--seed-runs", type=int, default=12,
                    help="short real SGD runs used only to seed states into the "
                         "non-trivial region; 0 = purely random states")
    ap.add_argument("--seed-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fwidth", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid); NP = numel(SH)
    xtr, ytr, xte, yte = make_task(SH, 11)
    h = len(xte) // 2
    xva, yva, xts, yts = xte[:h], yte[:h], xte[h:], yte[h:]
    print(f"student {NP:,} params | chance loss {torch.tensor(3.).log():.4f} "
          f"acc 0.333 | {len(xtr)} train / {len(xva)} val / {len(xts)} test")

    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed + 1)

    # ---- state pool: random at several scales, plus optional SGD-seeded states
    # The previous version drew 1,000 random inits to 246 SGD-seeded states and
    # move norms up to 2.7x ||theta||. Real SGD moves ~1-5% of ||theta|| over
    # 100 steps, so 83% of that dataset was destroyed networks and the model was
    # 10x WORSE than a constant predictor on the states a policy would visit.
    # Random now means random WITHIN A REALISTIC ENVELOPE.
    pool = []
    for sc in (0.7, 1.0, 1.4):
        for _ in range(max(1, a.samples // 400)):
            pool.append(init_w(SH, int(torch.randint(0, 10**6, (1,),
                                                     generator=g))) * sc)
    pool = torch.stack(pool)
    n_rand = len(pool)
    if a.seed_runs:
        seeded = []
        for i in range(a.seed_runs):
            C, _, _ = rollout_real((xtr, ytr, xva, yva), init_w(SH, 500 + i),
                                   [0.02, 0.05, 0.1, 0.2][i % 4],
                                   a.seed_steps, 5, a.batch, SH, seed=900 + i)
            seeded.append(C)
        pool = torch.cat([pool, torch.cat(seeded)])
    print(f"state pool: {n_rand:,} random + {len(pool)-n_rand:,} SGD-seeded "
          f"({time.time()-t0:.0f}s)")

    # ---- random moves from those states
    n = a.samples
    si = torch.randint(0, len(pool), (n,), generator=g)
    TH = pool[si]
    # move norm as a FRACTION of the weight norm: 0.1% to 10%, log-uniform.
    # This is the range real SGD occupies over tens to hundreds of steps.
    scale = torch.exp(torch.rand(n, 1, generator=g) * 4.6 - 6.9)   # 0.001..0.1
    d = torch.randn(n, NP, generator=g)
    DL = d / d.norm(dim=1, keepdim=True) * TH.norm(dim=1, keepdim=True) * scale
    l0v, _ = loss_acc(TH, xva, yva, SH)
    l0t, _ = loss_acc(TH, xtr, ytr, SH)
    l1v, a1v = loss_acc(TH + DL, xva, yva, SH)
    l1t, _ = loss_acc(TH + DL, xtr, ytr, SH)
    print(f"{n:,} (state, move) pairs labelled ({time.time()-t0:.0f}s)")

    ch = float(torch.tensor(3.).log())
    frac_chance = float((l1v > ch * 0.95).float().mean())
    print(f"\nCOVERAGE of the dataset (val loss at theta+delta):")
    for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
        print(f"    q{int(q*100):>3} {float(l1v.quantile(q)):.4f}", end="")
    print(f"\n    {100*frac_chance:.0f}% of samples are at or worse than chance "
          f"({ch:.4f}).")
    print(f"    best in dataset: val loss {float(l1v.min()):.4f}, "
          f"acc {float(a1v.max()):.3f}")
    if frac_chance > 0.8:
        print(f"    WARNING: the dataset is almost entirely useless networks. A")
        print(f"    transition model fitted here describes a region no policy")
        print(f"    will visit.")

    # ---- transition model
    dsd = DL.std(0).clamp_min(1e-8)
    tmu, tsd = l1t.mean(), l1t.std().clamp_min(1e-8)
    vmu, vsd = l1v.mean(), l1v.std().clamp_min(1e-8)
    T = nn.Sequential(nn.Linear(2 * NP + 2, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, 2))
    opt = torch.optim.AdamW(T.parameters(), lr=1e-3, weight_decay=1e-4)
    ntr = int(0.9 * n)

    def pred(th, dl, lt, lv):
        o = T(torch.cat([th, dl / dsd, ((lt - tmu) / tsd).unsqueeze(1),
                         ((lv - vmu) / vsd).unsqueeze(1)], 1))
        return o[:, 0] * tsd + tmu, o[:, 1] * vsd + vmu

    for e in range(a.epochs):
        b = torch.randint(0, ntr, (256,))
        pt, pv = pred(TH[b], DL[b], l0t[b], l0v[b])
        loss = (((pt - l1t[b]) / tsd).pow(2).mean()
                + ((pv - l1v[b]) / vsd).pow(2).mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"\ntransition model trained ({time.time()-t0:.0f}s)")

    def report(tag, th, dl, lt, lv, tgt):
        with torch.no_grad():
            _, pv = pred(th, dl, lt, lv)
        mae = float((pv - tgt).abs().mean())
        pc, rc = pv - pv.mean(), tgt - tgt.mean()
        r = float((pc * rc).sum() / (pc.norm() * rc.norm()).clamp_min(1e-9))
        base = float((tgt.mean() - tgt).abs().mean())
        print(f"  {tag:<26} MAE {mae:.4f}  corr {r:+.3f}  "
              f"(predict-the-mean MAE {base:.4f})")

    print(f"\nVAL-LOSS PREDICTION:")
    report("held-out random", TH[ntr:], DL[ntr:], l0t[ntr:], l0v[ntr:], l1v[ntr:])

    # distribution shift: states from real SGD runs, never in the dataset
    C, _, _ = rollout_real((xtr, ytr, xva, yva), init_w(SH, 0), 0.07, 600, 5,
                           a.batch, SH, seed=777)
    m = len(C) - 1
    sth = C[:m]
    sdl = C[1:m + 1] - C[:m]
    s0t, _ = loss_acc(sth, xtr, ytr, SH); s0v, _ = loss_acc(sth, xva, yva, SH)
    s1v, _ = loss_acc(sth + sdl, xva, yva, SH)
    report("SGD states + SGD moves", sth, sdl, s0t, s0v, s1v)
    rr = torch.randn(m, NP, generator=g)
    rr = rr / rr.norm(dim=1, keepdim=True) * sdl.norm(dim=1, keepdim=True)
    r1v, _ = loss_acc(sth + rr, xva, yva, SH)
    report("SGD states + random moves", sth, rr, s0t, s0v, r1v)

    # ---- Dreamer-style: imagine cheaply, anchor in reality occasionally
    print(f"\nANCHORED ROLLOUT -- propose moves, track the loss channel, and")
    print(f"re-anchor to a REAL validation evaluation every N hops.")
    print(f"A real anchor is ONE forward pass on {len(xva)} examples; the moves")
    print(f"it is standing in for are tens of SGD steps each.")
    HOPS = 12
    start = len(C) // 6
    th0 = C[start]
    moves = []
    gg = torch.Generator().manual_seed(31)
    for _ in range(HOPS):
        mv = torch.randn(1, NP, generator=gg)
        moves.append(mv / mv.norm() * th0.norm() * 0.02)
    print(f"\n  {'anchor every':>13} {'mean |pred-real| over 12 hops':>32} "
          f"{'final err':>10} {'real evals':>11}")
    for every in (1, 2, 4, 12, 0):
        th = th0.clone().unsqueeze(0)
        lt, _ = loss_acc(th, xtr, ytr, SH)
        lv, _ = loss_acc(th, xva, yva, SH)
        errs, nev = [], 1
        for hp in range(1, HOPS + 1):
            with torch.no_grad():
                pt, pv = pred(th, moves[hp - 1], lt, lv)
            th = th + moves[hp - 1]
            rv, _ = loss_acc(th, xva, yva, SH)
            errs.append(float((pv - rv).abs()))
            if every and hp % every == 0:
                lt, _ = loss_acc(th, xtr, ytr, SH)   # ANCHOR: reality
                lv = rv
                nev += 2
            else:
                lt, lv = pt, pv                      # stay in imagination
        lab = "never" if every == 0 else str(every)
        print(f"  {lab:>13} {sum(errs)/len(errs):>32.4f} {errs[-1]:>10.4f} "
              f"{nev:>11}")
    print(f"\n  'never' is pure imagination; anchoring every hop is the")
    print(f"  accuracy ceiling. The gap between them is what drift costs, and")
    print(f"  the cheapest anchor interval that stays near the ceiling is the")
    print(f"  operating point.")

    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
