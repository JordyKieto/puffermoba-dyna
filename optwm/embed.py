"""Learned task embeddings: can F transfer to a function it has never seen?

optwm/predict showed the boundary: F works within one task and collapses when
the task varies. A hand-built context (one forward pass on a probe batch) made
it worse, not better -- a fixed featurisation cannot represent whatever
distinguishes one optimization problem from another.

So learn it. Give every training task a free vector e_i, optimized jointly with
F by backprop:

    F(theta_t, velocity, e_task, lr) -> Delta

Then a NEW task needs no retraining of F. Freeze F, spend a handful of real
steps on the new task, and fit ONLY its embedding -- a few dozen numbers -- on
those observed transitions. That is the adaptation test, and it is the thing a
hand-built context cannot do.

HONEST ACCOUNTING. Adapting costs real training steps: you cannot observe a
transition without producing it. So every arm is charged for what it spent, and
compared against spending the same real steps directly. Reported alongside:

  persistence     predict nothing                     (the floor)
  mean embedding  F with the average training task    (zero-shot, no adaptation)
  adapted         F with an embedding fit on n real transitions
  ORACLE          embedding fit on the EVAL transitions themselves

The oracle is the ceiling: it cheats by seeing the answers. If adapted lands far
below oracle, more adaptation data would help. If oracle itself is low, no
embedding can express this task and the limit is F, not the data.

Run:  python -m optwm.embed --tasks 192 --steps 400
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .predict import NP, evaluate, init_w, make_task, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=192)
    ap.add_argument("--holdout-tasks", type=int, default=32)
    ap.add_argument("--runs-per-task", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=20, help="jump = k*res real steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16, help="embedding width")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--adapt", type=str, default="1,2,5,10,20",
                    help="n real transitions used to fit a new task's embedding")
    ap.add_argument("--adapt-steps", type=int, default=300)
    ap.add_argument("--diagnose", action="store_true",
                    help="run the three diagnostics and skip the slow "
                         "adaptation sweep")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    w0 = init_w(0)
    lrs = [0.05, 0.1]
    ntr = a.tasks - a.holdout_tasks
    print(f"{a.tasks} distinct tasks ({ntr} train / {a.holdout_tasks} held out), "
          f"{a.runs_per_task} run(s) each, embedding dim {a.dim}")

    t0 = time.time()
    tasks, trajs, TID, LRv = [], [], [], []
    for i in range(a.tasks):
        task = make_task(5000 + i)
        tasks.append(task)
        for r in range(a.runs_per_task):
            trajs.append(run(task, w0, lrs[r % len(lrs)], a.steps, a.res,
                             a.batch, seed=i * 10 + r))
            TID.append(i); LRv.append(lrs[r % len(lrs)])
    print(f"generated {len(trajs)} trajectories ({time.time()-t0:.0f}s)")

    kk = a.k
    X, Y, PREV, LR, TI = [], [], [], [], []
    for j, C in enumerate(trajs):
        X.append(C[1:-kk]); Y.append(C[1 + kk:]); PREV.append(C[:-kk - 1])
        m = len(C) - kk - 1
        LR += [LRv[j]] * m; TI += [TID[j]] * m
    X, Y, PREV = torch.cat(X), torch.cat(Y), torch.cat(PREV)
    LR = torch.tensor(LR).float().unsqueeze(1); TI = torch.tensor(TI)
    tr = TI < ntr
    D = Y - X
    mu, sd = D[tr].mean(0), D[tr].std(0).clamp_min(1e-8)
    TGT = (D - mu) / sd
    print(f"pairs {len(X)}  (jump = {kk*a.res} real steps)")

    # ---- F, conditioned on a learned per-task embedding
    emb = nn.Embedding(ntr, a.dim)
    nn.init.normal_(emb.weight, std=0.1)
    net = nn.Sequential(nn.Linear(NP * 2 + a.dim + 1, 512), nn.SiLU(),
                        nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, NP))
    opt = torch.optim.AdamW(list(net.parameters()) + list(emb.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    itr = torch.nonzero(tr).squeeze(1)
    for e in range(a.epochs):
        b = itr[torch.randint(0, len(itr), (256,))]
        inp = torch.cat([X[b], X[b] - PREV[b], emb(TI[b]), LR[b]], 1)
        loss = F_.mse_loss(net(inp), TGT[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"F+embeddings trained (mse {float(loss.detach()):.4f})")
    ebar = emb.weight.detach().mean(0, keepdim=True)

    def predict(x, prev, ev, lr):
        with torch.no_grad():
            inp = torch.cat([x, x - prev, ev.expand(len(x), -1), lr], 1)
            return x + (net(inp) * sd + mu)

    def fit_emb(x, prev, tgt, lr, steps):
        """Fit ONLY the embedding. F is frozen."""
        ev = ebar.clone().requires_grad_(True)
        o = torch.optim.Adam([ev], lr=0.05)
        for _ in range(steps):
            inp = torch.cat([x, x - prev, ev.expand(len(x), -1), lr], 1)
            l = F_.mse_loss(net(inp), tgt)
            o.zero_grad(set_to_none=True); l.backward(); o.step()
        return ev.detach()

    # ---- per-held-out-task adaptation
    print(f"\n% of the true jump's loss gain, on tasks F has NEVER seen")
    print(f"  (mean over {a.holdout_tasks} held-out tasks; 0% = predict nothing,")
    print(f"   100% = match real training; ORACLE sees the eval answers)")
    print(f"\n  {'n adapt':>8} {'real steps':>11} {'persist':>8} {'mean-emb':>9} "
          f"{'ADAPTED':>9} {'oracle':>8}")

    # CONTROL: does this F work on the tasks it TRAINED on, using their own
    # learned embeddings? Without this the held-out numbers are ambiguous -- a
    # multi-task F that simply never learned would look identical to one that
    # learned and failed to transfer.
    indist = []
    for tid in range(max(0, ntr - 24), ntr):
        m = TI == tid
        xs, ps, ys, ls = X[m], PREV[m], Y[m], LR[m]
        task = tasks[tid]
        half = len(xs) // 2
        lx = torch.tensor([evaluate(xs[i], task)[0] for i in range(half, len(xs))])
        ly = torch.tensor([evaluate(ys[i], task)[0] for i in range(half, len(xs))])
        order = (lx - ly).argsort(descending=True)[:12]
        sel = order + half
        head = float((lx[order] - ly[order]).mean())
        if head < 1e-4:
            continue
        base = float(lx[order].mean())
        ev = emb.weight.detach()[tid:tid + 1]
        P = predict(xs[sel], ps[sel], ev, ls[sel])
        ml = sum(evaluate(P[i], task)[0] for i in range(len(P))) / len(P)
        indist.append(100 * (base - ml) / head)
    print(f"\nCONTROL -- TRAINING tasks, own learned embedding: "
          f"{st.mean(indist):+.0f}% (n={len(indist)})")

    if a.diagnose:
        # === 1. Does F work on the tasks it TRAINED on, with their own
        # embeddings? Without this, "learned but cannot transfer" and "never
        # learned anything" are indistinguishable and no other number here can
        # be interpreted.
        indist = []
        for tid in range(max(0, ntr - 24), ntr):
            m = TI == tid
            xs, ps, ys, ls = X[m], PREV[m], Y[m], LR[m]
            task = tasks[tid]
            half = len(xs) // 2
            lx = torch.tensor([evaluate(xs[i], task)[0]
                               for i in range(half, len(xs))])
            ly = torch.tensor([evaluate(ys[i], task)[0]
                               for i in range(half, len(xs))])
            order = (lx - ly).argsort(descending=True)[:12]
            sel = order + half
            head = float((lx[order] - ly[order]).mean())
            if head < 1e-4:
                continue
            base = float(lx[order].mean())
            ev = emb.weight.detach()[tid:tid + 1]
            P = predict(xs[sel], ps[sel], ev, ls[sel])
            ml = sum(evaluate(P[i], task)[0] for i in range(len(P))) / len(P)
            indist.append(100 * (base - ml) / head)
        print(f"\n=== 1. IN-DISTRIBUTION CONTROL")
        print(f"  training tasks, own learned embedding: "
              f"{st.mean(indist):+.0f}% of true gain  (n={len(indist)})")
        print(f"  single-task F reached +87% for reference; persistence is 0%")

        # === 2. Is the embedding channel alive? If swapping in another task's
        # embedding barely changes the prediction, F is not using it and no
        # amount of adaptation can help.
        print(f"\n=== 2. IS THE EMBEDDING CHANNEL ALIVE?")
        b = itr[torch.randint(0, len(itr), (512,))]
        x_, p_, l_ = X[b], PREV[b], LR[b]
        e_true = emb.weight.detach()[TI[b]]
        perm = torch.randint(0, ntr, (len(b),))
        e_swap = emb.weight.detach()[perm]
        with torch.no_grad():
            def raw(ev):
                return net(torch.cat([x_, x_ - p_, ev, l_], 1)) * sd + mu
            d_true, d_swap = raw(e_true), raw(e_swap)
            d_mean = raw(ebar.expand(len(b), -1))
        nd = d_true.norm(dim=1).mean()
        print(f"  predicted delta magnitude               {float(nd):.4e}")
        print(f"  change from swapping in another task's e {float((d_swap-d_true).norm(dim=1).mean()):.4e}")
        print(f"  change from using the MEAN e             {float((d_mean-d_true).norm(dim=1).mean()):.4e}")
        print(f"  --> swap changes the prediction by "
              f"{100*float((d_swap-d_true).norm(dim=1).mean()/nd):.1f}% of its size")
        xg = x_[:64].clone().requires_grad_(True)
        eg = e_true[:64].clone().requires_grad_(True)
        out = net(torch.cat([xg, xg - p_[:64], eg, l_[:64]], 1)).sum()
        gx, ge = torch.autograd.grad(out, [xg, eg])
        print(f"  ||dF/dtheta|| {float(gx.norm()):.3e} over {NP} dims  "
              f"-> {float(gx.norm())/NP**.5:.3e} per dim")
        print(f"  ||dF/de||     {float(ge.norm()):.3e} over {a.dim} dims  "
              f"-> {float(ge.norm())/a.dim**.5:.3e} per dim")

        # === 3. Did similar functions end up near each other? The direct test
        # of the hypothesis. Task similarity is measured functionally -- how
        # often two teachers agree on a fixed probe -- which is invariant to
        # the permutation symmetries that make weight distance meaningless.
        print(f"\n=== 3. DID SIMILAR FUNCTIONS GROUP?")
        py = torch.stack([tasks[i][4] for i in range(ntr)])
        E = emb.weight.detach()
        ii, jj = torch.triu_indices(ntr, ntr, offset=1)
        samp = torch.randperm(len(ii))[:20000]
        ii, jj = ii[samp], jj[samp]
        agree = (py[ii] == py[jj]).float().mean(1)
        edist = (E[ii] - E[jj]).norm(dim=1)
        ac, ec = agree - agree.mean(), edist - edist.mean()
        r = float((ac * ec).sum() / (ac.norm() * ec.norm()))
        print(f"  functional similarity between task pairs (label agreement on"
              f"\n  a fixed probe): min {float(agree.min()):.2f} "
              f"mean {float(agree.mean()):.2f} max {float(agree.max()):.2f}")
        print(f"  embedding distance: mean {float(edist.mean()):.3f} "
              f"(sd {float(edist.std()):.3f})")
        print(f"  correlation(similarity, embedding distance) = {r:+.3f}")
        print(f"  (want NEGATIVE: more similar tasks -> closer embeddings)")
        print(f"\n({time.time()-t0:.0f}s total)")
        return

    ns = [int(v) for v in a.adapt.split(",")]
    rows = {n: {"persist": [], "mean": [], "adapt": [], "oracle": []} for n in ns}
    for tid in range(ntr, a.tasks):
        m = TI == tid
        xs, ps, ys, ls = X[m], PREV[m], Y[m], LR[m]
        tg = TGT[m]
        task = tasks[tid]
        # eval on the SECOND HALF of this task's transitions; adaptation may
        # only look at the first half, so nothing is scored on what it fit
        half = len(xs) // 2
        lx = torch.tensor([evaluate(xs[i], task)[0] for i in range(half, len(xs))])
        ly = torch.tensor([evaluate(ys[i], task)[0] for i in range(half, len(xs))])
        order = (lx - ly).argsort(descending=True)[:12]
        sel = order + half
        head = float((lx[order] - ly[order]).mean())
        if head < 1e-4:
            continue
        base = float(lx[order].mean())

        def gain(P):
            ml = sum(evaluate(P[i], task)[0] for i in range(len(P))) / len(P)
            return 100 * (base - ml) / head

        # the oracle does not depend on n -- fit it once
        ov = fit_emb(xs[sel], ps[sel], tg[sel], ls[sel], a.adapt_steps)
        g_mean = gain(predict(xs[sel], ps[sel], ebar, ls[sel]))
        g_or = gain(predict(xs[sel], ps[sel], ov, ls[sel]))
        for n in ns:
            idx = torch.arange(0, min(n, half))
            ev = fit_emb(xs[idx], ps[idx], tg[idx], ls[idx], a.adapt_steps)
            rows[n]["persist"].append(0.0)
            rows[n]["mean"].append(g_mean)
            rows[n]["adapt"].append(gain(predict(xs[sel], ps[sel], ev, ls[sel])))
            rows[n]["oracle"].append(g_or)

    for n in ns:
        r = rows[n]
        if not r["adapt"]:
            continue
        sem = st.stdev(r["adapt"]) / len(r["adapt"]) ** .5 if len(r["adapt"]) > 1 else 0
        print(f"  {n:>8} {n*kk*a.res:>11} {0:>7.0f}% {st.mean(r['mean']):>8.0f}% "
              f"{st.mean(r['adapt']):>6.0f}%±{sem:<3.0f} {st.mean(r['oracle']):>7.0f}%")

    print(f"\n  'real steps' is what the adaptation cost: n observed transitions")
    print(f"  at {kk*a.res} real steps each. Compare any row against simply")
    print(f"  spending those steps on training.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
