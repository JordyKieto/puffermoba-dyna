"""The full loop: dream cheaply, verify expensively, repeatedly.

optwm.predict established that one jump works within a fixed task. That is not
yet the claim. The claim is the loop:

    jump k steps with F  ->  run m REAL SGD steps  ->  jump again  ->  ...

which is where a learned simulator either holds or diverges, because each jump
starts from the previous jump's output rather than from a real trajectory.

THE BASELINE THAT DECIDES THIS. A loop reaching T steps of progress using R
real steps must be compared against three things, not one:

  full real      T real steps                 -- the quality target
  budget-matched R real steps and nothing else -- THE ONE THAT MATTERS
  no-op loop     the same loop with F replaced by identity

If the loop matches `full real` but `budget-matched` matches it too, the jumps
did nothing and R real steps were simply enough. Only beating budget-matched
shows the jumps performed real optimization work.

Scored on TRAIN loss, which is what SGD actually minimises and is monotone
here. Test loss bottoms out around step 100 on this task and then rises, so it
measures overfitting, not training progress, and would reward a method for
refusing to advance.

Run:  python -m optwm.loop --runs 128 --steps 800
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F_

from .predict import (NP, build_F, evaluate, init_w, make_task, run, sforward)


def train_loss(w, task):
    xtr, ytr, _, _, _ = task
    with torch.no_grad():
        return float(F_.cross_entropy(sforward(w, xtr), ytr))


def sgd_steps(w, task, lr, n, batch, gen, snap_at=None):
    """n authoritative SGD steps. Returns (weights, snapshot)."""
    xtr, ytr, _, _, _ = task
    ww = w.clone().requires_grad_(True)
    opt = torch.optim.SGD([ww], lr=lr, momentum=0.9)
    snap = None
    for i in range(n):
        if snap_at is not None and i == snap_at:
            snap = ww.detach().clone()
        idx = torch.randint(0, len(xtr), (batch,), generator=gen)
        L = F_.cross_entropy(sforward(ww, xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True); L.backward(); opt.step()
    return ww.detach(), snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=40, help="jump = k*res real steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--target", type=int, default=800, help="T: total progress")
    ap.add_argument("--corrections", type=str, default="200,100,50,25,10",
                    help="m: real steps per cycle, must be >= res")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eval-lr", type=float, default=None,
                    help="run the loop at a learning rate F never trained on")
    ap.add_argument("--multi-init", action="store_true",
                    help="generate each training trajectory from a DIFFERENT "
                         "theta_0. Every run sharing one init is what lets F "
                         "memorise a single trajectory family; this tests "
                         "whether the init dependence is fundamental or just "
                         "missing coverage.")
    ap.add_argument("--eval-init", type=int, default=0,
                    help="run the loop from a DIFFERENT theta_0 than every "
                         "training trajectory used. This is the reusability "
                         "test: if F only works from the init it was trained "
                         "on, it has memorised one trajectory family.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    task, w0 = make_task(0), init_w(0)
    wS = init_w(a.eval_init) if a.eval_init else w0
    eval_lr = a.eval_lr if a.eval_lr is not None else a.lr
    lrs = [0.02, 0.05, 0.1, 0.2]

    t0 = time.time()
    trajs, LRv = [], []
    for i in range(a.runs):
        wi = init_w(10_000 + i) if a.multi_init else w0
        trajs.append(run(task, wi, lrs[i % 4], a.steps, a.res, a.batch, seed=i))
        LRv.append(lrs[i % 4])
    print(f"{a.runs} trajectories from "
          f"{a.runs if a.multi_init else 1} distinct init(s) "
          f"({time.time()-t0:.0f}s)")

    kk = a.k
    X, Y, PREV, LR = [], [], [], []
    for i, C in enumerate(trajs):
        X.append(C[1:-kk]); Y.append(C[1 + kk:]); PREV.append(C[:-kk - 1])
        LR += [LRv[i]] * (len(C) - kk - 1)
    X, Y, PREV = torch.cat(X), torch.cat(Y), torch.cat(PREV)
    LR = torch.tensor(LR).float().unsqueeze(1)
    tr = torch.ones(len(X), dtype=torch.bool)
    net, mu, sd = build_F(X, PREV, LR, Y, tr, a.epochs)
    print(f"F trained on jumps of {kk*a.res} real steps\n")

    def jump(w, prev, lr):
        with torch.no_grad():
            x = torch.cat([w.unsqueeze(0), (w - prev).unsqueeze(0),
                           torch.tensor([[lr]])], 1)
            return w + (net(x)[0] * sd + mu)

    def do_loop(m, lr, seed, use_F=True):
        """One loop to T progress. Returns (train loss, real steps used)."""
        g = torch.Generator().manual_seed(seed)
        w, _ = sgd_steps(wS, task, lr, a.res, a.batch, g)
        prev, used, prog = wS.clone(), a.res, a.res
        while prog < a.target:
            w = jump(w, prev, lr) if use_F else w
            prog += kk * a.res
            n = min(m, max(0, a.target - prog) + m)
            w, snap = sgd_steps(w, task, lr, n, a.batch, g, snap_at=n - a.res)
            prev = snap if snap is not None else w.clone()
            used += n; prog += n
        return train_loss(w, task), used

    print(f"TARGET: {a.target} steps of progress, jump size {kk*a.res}, "
          f"lr {eval_lr}{' [UNSEEN]' if a.eval_lr else ''}, "
          f"init {a.eval_init}{' [UNSEEN]' if a.eval_init else ''}, "
          f"{a.trials} trials")
    print(f"{'m real/cycle':>13} {'real used':>10} {'% elim':>7} "
          f"{'LOOP loss':>11} {'budget-matched':>15} {'no-op loop':>12}")

    full = [train_loss(sgd_steps(wS, task, eval_lr, a.target, a.batch,
                                 torch.Generator().manual_seed(500 + s))[0], task)
            for s in range(a.trials)]
    import statistics as st
    print(f"{'--- full real':>13} {a.target:>10} {0:>6}% "
          f"{st.mean(full):>11.4f}")

    for m in map(int, a.corrections.split(",")):
        if m < a.res:
            continue
        L, U, B, N = [], [], [], []
        for s in range(a.trials):
            l, u = do_loop(m, eval_lr, 100 + s, use_F=True)
            L.append(l); U.append(u)
            ln, _ = do_loop(m, eval_lr, 100 + s, use_F=False)
            N.append(ln)
            g = torch.Generator().manual_seed(300 + s)
            B.append(train_loss(sgd_steps(wS, task, eval_lr, u, a.batch, g)[0], task))
        u = int(st.mean(U))
        sem = st.stdev(L)/len(L)**.5 if len(L) > 1 else 0.0
        print(f"{m:>13} {u:>10} {100*(1-u/a.target):>6.0f}% "
              f"{st.mean(L):>8.4f}+/-{sem:<.4f} {st.mean(B):>13.4f} "
              f"{st.mean(N):>12.4f}")

    print(f"\nRead it as: LOOP must beat budget-matched to have done any real")
    print(f"work, and approach full-real to be useful. no-op loop is the same")
    print(f"schedule with F replaced by identity -- it isolates F from the")
    print(f"real steps the schedule happens to include.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
