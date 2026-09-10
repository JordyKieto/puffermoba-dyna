"""The fine-tuning regime: does a shared starting point make F transfer?

optwm/encoder found that task identity is not the bottleneck. A task-BLIND F
scores +70% in-distribution and -94% on held-out tasks, a 164-point gap using no
task information at all, so what separates the regimes is the region of weight
space a trajectory visits -- not which function it is learning. theta_t already
tells F which basin it is in, which makes task conditioning redundant where it
works and useless where it does not.

That predicts something specific and testable. Fine-tuning jobs all start from
ONE pretrained checkpoint, so their trajectories stay in one neighbourhood. If
the weight-space story is right, F should transfer to a HELD-OUT job in that
setting -- and fail when runs start from scratch and scatter.

Two factors, crossed:

    start:  pretrained (one shared checkpoint)  vs  scratch (own random init)
    tasks:  related (base teacher + eps*noise)  vs  unrelated (independent)

Everything is scored on tasks F never trained on. F is task-blind by design --
(theta_t, velocity, lr) only -- because the encoder experiment showed conditioning
adds nothing, and dropping it removes a confound.

Also reported: how far held-out weights sit from the training set, relative to
the spread within it. That is the mechanism itself, measured directly rather
than inferred. If the pretrained arms transfer AND show a small distance ratio
while the scratch arms show a large one, the weight-space account is confirmed.

Run:  python -m optwm.finetune --tasks 64 --steps 300
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .predict import (D_IN, D_OUT, NP, PROBE_X, evaluate, init_w, run,
                      sforward)


def task_from_teacher(tw, n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D_IN, generator=g)
    with torch.no_grad():
        y = sforward(tw, x).argmax(1)
        py = sforward(tw, PROBE_X).argmax(1)
    return x[:n // 2], y[:n // 2], x[n // 2:], y[n // 2:], py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=64)
    ap.add_argument("--holdout-tasks", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.15,
                    help="task relatedness: teacher = base + eps*noise")
    ap.add_argument("--pretrain-steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    ntr = a.tasks - a.holdout_tasks
    lrs = [0.05, 0.1]
    kk = a.k

    g0 = torch.Generator().manual_seed(1234)
    tw_base = torch.randn(NP, generator=g0) * 0.6
    base_task = task_from_teacher(tw_base, seed=7)

    # one shared pretrained checkpoint, as every fine-tuning job would use
    pre = run(base_task, init_w(0), 0.1, a.pretrain_steps, a.pretrain_steps,
              a.batch, seed=99)[-1]
    print(f"pretrained checkpoint: base-task train loss "
          f"{float(F_.cross_entropy(sforward(pre, base_task[0]), base_task[1])):.4f}"
          f"  (from {float(F_.cross_entropy(sforward(init_w(0), base_task[0]), base_task[1])):.4f})")
    print(f"eps = {a.eps}  ({a.tasks} tasks, {ntr} train / {a.holdout_tasks} held out)\n")

    def build(related, pretrained):
        tasks, trajs, TID = [], [], []
        for i in range(a.tasks):
            gi = torch.Generator().manual_seed(7000 + i)
            tw = (tw_base + a.eps * torch.randn(NP, generator=gi) * 0.6
                  if related else torch.randn(NP, generator=gi) * 0.6)
            tk = task_from_teacher(tw, seed=7000 + i)
            tasks.append(tk)
            w_start = pre if pretrained else init_w(20_000 + i)
            for r, lr in enumerate(lrs):
                trajs.append(run(tk, w_start, lr, a.steps, a.res, a.batch,
                                 seed=i * 10 + r))
                TID.append(i)
        X, Y, PREV, LR, TI = [], [], [], [], []
        for j, C in enumerate(trajs):
            X.append(C[1:-kk]); Y.append(C[1 + kk:]); PREV.append(C[:-kk - 1])
            m = len(C) - kk - 1
            LR += [lrs[j % len(lrs)]] * m; TI += [TID[j]] * m
        return (tasks, torch.cat(X), torch.cat(Y), torch.cat(PREV),
                torch.tensor(LR).float().unsqueeze(1), torch.tensor(TI))

    print(f"{'arm':<28} {'held-out':>10} {'in-dist':>9} {'dist ratio':>11}")
    results = {}
    for related in (True, False):
        for pretrained in (True, False):
            tasks, X, Y, PREV, LR, TI = build(related, pretrained)
            trm = TI < ntr
            D = Y - X
            mu, sd = D[trm].mean(0), D[trm].std(0).clamp_min(1e-8)
            TGT = (D - mu) / sd
            net = nn.Sequential(nn.Linear(NP * 2 + 1, 512), nn.SiLU(),
                                nn.Linear(512, 512), nn.SiLU(),
                                nn.Linear(512, NP))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3,
                                    weight_decay=1e-4)
            itr = torch.nonzero(trm).squeeze(1)
            xin = torch.cat([X, X - PREV, LR], 1)
            for e in range(a.epochs):
                b = itr[torch.randint(0, len(itr), (256,))]
                loss = F_.mse_loss(net(xin[b]), TGT[b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            def gains(lo, hi):
                out = []
                for tid in range(lo, hi):
                    m = torch.nonzero(TI == tid).squeeze(1)
                    xs, ps, ys, ls = X[m], PREV[m], Y[m], LR[m]
                    task = tasks[tid]
                    half = len(xs) // 2
                    lx = torch.tensor([evaluate(xs[i], task)[0]
                                       for i in range(half, len(xs))])
                    ly = torch.tensor([evaluate(ys[i], task)[0]
                                       for i in range(half, len(xs))])
                    o = (lx - ly).argsort(descending=True)[:12]
                    sel = o + half
                    head = float((lx[o] - ly[o]).mean())
                    if head < 1e-4:
                        continue
                    base = float(lx[o].mean())
                    with torch.no_grad():
                        P = xs[sel] + (net(torch.cat(
                            [xs[sel], xs[sel] - ps[sel], ls[sel]], 1)) * sd + mu)
                    ml = sum(evaluate(P[i], task)[0]
                             for i in range(len(P))) / len(P)
                    out.append(100 * (base - ml) / head)
                return out

            ho, ind = gains(ntr, a.tasks), gains(max(0, ntr - 12), ntr)
            # how far outside the training set do held-out weights sit?
            trX, hoX = X[trm], X[~trm]
            si = torch.randperm(len(hoX))[:128]
            sj = torch.randperm(len(trX))[:512]
            near = torch.cdist(hoX[si], trX[sj]).min(1).values.mean()
            within = torch.cdist(trX[sj[:128]], trX[sj]).median()
            ratio = float(near / within.clamp_min(1e-9))
            nm = (("related" if related else "unrelated") + " + " +
                  ("pretrained" if pretrained else "scratch"))
            sem = st.stdev(ho) / len(ho) ** .5 if len(ho) > 1 else 0
            print(f"{nm:<28} {st.mean(ho):>+7.0f}%±{sem:<3.0f} "
                  f"{st.mean(ind):>+8.0f}% {ratio:>11.3f}")
            results[nm] = st.mean(ho)

    print(f"\n  held-out = tasks F never trained on. 0% = predict nothing,")
    print(f"  100% = match real training. 'dist ratio' is how far held-out")
    print(f"  weights sit from the nearest training weights, over the typical")
    print(f"  spacing inside the training set -- near 1 means same region.")
    print(f"\n({time.time():.0f})")


if __name__ == "__main__":
    main()
