"""Amortised task encoder: compute the embedding instead of looking it up.

optwm/embed diagnosed the failure precisely. A free per-task embedding table
works in-distribution (+75% of the true jump's gain) and the conditioning
channel is strong (swapping a task's code changes the prediction by 84%), but
the learned space has NO structure: correlation between functional task
similarity and embedding distance was -0.015. F memorised 160 arbitrary ID
codes. It behaves sensibly at those 160 points and nowhere between them, so a
new task has no valid code to find -- even an oracle fit on the answers reached
only -52%.

The objective permitted this. Apply any invertible relabelling to the codebook,
adjust F's first layer to match, and the loss is IDENTICAL. Grouping is not
merely unlikely under that objective, it is un-selected-for.

An encoder removes the freedom. Instead of a lookup table:

    e = Enc({(x_1,y_1) ... (x_N,y_N)})        a shared, continuous function

Three consequences:

  1. No per-task free parameters, so there is no codebook to memorise.
  2. Enc is continuous, so similar task data MUST give similar embeddings.
     Grouping becomes structural rather than hoped-for.
  3. A new task is ZERO-SHOT: run Enc on its data. The table needed n observed
     transitions -- n x 100 real SGD steps -- to fit a code. This needs a
     single forward pass on some labelled examples and no training at all.

Enc is a DeepSets encoder: embed each labelled example, mean-pool, project. Mean
pooling makes it permutation-invariant, which is the right prior -- a task is a
SET of examples and their order carries nothing.

Reported against the same references: persistence (predict nothing), a
task-blind arm (F with the mean training embedding, so it knows dynamics but not
which task), and the in-distribution control that tells us F learned at all.

Run:  python -m optwm.encoder --tasks 192 --steps 400
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .predict import D_IN, D_OUT, NP, evaluate, init_w, make_task, run


class SetEncoder(nn.Module):
    """DeepSets: phi on each example, mean-pool, rho on the pooled vector."""

    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_hid), nn.SiLU(),
                                 nn.Linear(d_hid, d_hid), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(d_hid, d_hid), nn.SiLU(),
                                 nn.Linear(d_hid, d_out))

    def forward(self, S):                      # S: (T, N, d_in)
        return self.rho(self.phi(S).mean(1))   # -> (T, d_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=192)
    ap.add_argument("--holdout-tasks", type=int, default=32)
    ap.add_argument("--runs-per-task", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--enc-hid", type=int, default=128)
    ap.add_argument("--enc-n", type=int, default=32,
                    help="labelled examples the encoder sees per task")
    ap.add_argument("--tasks-per-step", type=int, default=32)
    ap.add_argument("--pairs-per-task", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    w0 = init_w(0)
    lrs = [0.05, 0.1]
    ntr = a.tasks - a.holdout_tasks
    print(f"{a.tasks} tasks ({ntr} train / {a.holdout_tasks} held out), "
          f"encoder sees {a.enc_n} labelled examples/task, dim {a.dim}")

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
    D = Y - X
    trm = TI < ntr
    mu, sd = D[trm].mean(0), D[trm].std(0).clamp_min(1e-8)
    TGT = (D - mu) / sd
    by_task = [torch.nonzero(TI == t).squeeze(1) for t in range(a.tasks)]
    print(f"pairs {len(X)}  (jump = {kk*a.res} real steps)")

    # each task's pool of labelled examples, for the encoder to sample from
    pools = []
    for t in range(a.tasks):
        xtr, ytr, _, _, _ = tasks[t]
        pools.append(torch.cat([xtr, F_.one_hot(ytr, D_OUT).float()], 1))
    d_ex = D_IN + D_OUT

    enc = SetEncoder(d_ex, a.enc_hid, a.dim)
    net = nn.Sequential(nn.Linear(NP * 2 + a.dim + 1, 512), nn.SiLU(),
                        nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, NP))
    opt = torch.optim.AdamW(list(net.parameters()) + list(enc.parameters()),
                            lr=1e-3, weight_decay=1e-4)

    def build_sets(tids, n):
        """Sample n labelled examples per task. Resampled every step, so the
        encoder must produce a stable code for the TASK rather than latch onto
        one particular set of examples."""
        return torch.stack([pools[int(t)][torch.randint(0, len(pools[int(t)]),
                                                        (n,))] for t in tids])

    for e in range(a.epochs):
        tsel = torch.randint(0, ntr, (a.tasks_per_step,))
        E = enc(build_sets(tsel, a.enc_n))
        idx, emap = [], []
        for u, t in enumerate(tsel):
            pool = by_task[int(t)]
            pick = pool[torch.randint(0, len(pool), (a.pairs_per_task,))]
            idx.append(pick); emap += [u] * a.pairs_per_task
        idx = torch.cat(idx); emap = torch.tensor(emap)
        inp = torch.cat([X[idx], X[idx] - PREV[idx], E[emap], LR[idx]], 1)
        loss = F_.mse_loss(net(inp), TGT[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"encoder + F trained (mse {float(loss.detach()):.4f}, "
          f"{time.time()-t0:.0f}s)")

    with torch.no_grad():
        ALL_E = enc(build_sets(torch.arange(a.tasks), 256))
    ebar = ALL_E[:ntr].mean(0, keepdim=True)

    def gain_for(tid, ev):
        """% of the true jump's loss gain recovered on this task."""
        p = by_task[tid]
        xs, ps, ys, ls = X[p], PREV[p], Y[p], LR[p]
        task = tasks[tid]
        half = len(xs) // 2
        lx = torch.tensor([evaluate(xs[i], task)[0] for i in range(half, len(xs))])
        ly = torch.tensor([evaluate(ys[i], task)[0] for i in range(half, len(xs))])
        order = (lx - ly).argsort(descending=True)[:12]
        sel = order + half
        head = float((lx[order] - ly[order]).mean())
        if head < 1e-4:
            return None
        base = float(lx[order].mean())
        with torch.no_grad():
            inp = torch.cat([xs[sel], xs[sel] - ps[sel],
                             ev.expand(len(sel), -1), ls[sel]], 1)
            P = xs[sel] + (net(inp) * sd + mu)
        ml = sum(evaluate(P[i], task)[0] for i in range(len(P))) / len(P)
        return 100 * (base - ml) / head

    def sweep(lo, hi, label):
        enc_g, blind_g = [], []
        for tid in range(lo, hi):
            g = gain_for(tid, ALL_E[tid:tid + 1])
            b = gain_for(tid, ebar)
            if g is not None:
                enc_g.append(g); blind_g.append(b)
        se = st.stdev(enc_g) / len(enc_g) ** .5 if len(enc_g) > 1 else 0
        print(f"  {label:<34} {st.mean(enc_g):>+6.0f}% ±{se:<3.0f} "
              f"{st.mean(blind_g):>+9.0f}%")
        return st.mean(enc_g)

    print(f"\n{'':<36}{'ENCODER':>11} {'task-blind':>10}")
    sweep(max(0, ntr - 24), ntr, "training tasks (control)")
    sweep(ntr, a.tasks, "HELD-OUT tasks, ZERO-SHOT")
    print(f"  persistence is 0%; single-task F reached +87%;")
    print(f"  the embedding TABLE got +75% in-distribution and -52% with an")
    print(f"  ORACLE embedding on held-out tasks (and it cost real SGD steps).")

    # did grouping emerge?
    py = torch.stack([tasks[i][4] for i in range(a.tasks)])
    ii, jj = torch.triu_indices(a.tasks, a.tasks, offset=1)
    s = torch.randperm(len(ii))[:20000]
    ii, jj = ii[s], jj[s]
    agree = (py[ii] == py[jj]).float().mean(1)
    ed = (ALL_E[ii] - ALL_E[jj]).norm(dim=1)
    ac, ec = agree - agree.mean(), ed - ed.mean()
    r = float((ac * ec).sum() / (ac.norm() * ec.norm()))
    print(f"\nDID SIMILAR FUNCTIONS GROUP?")
    print(f"  correlation(functional similarity, embedding distance) = {r:+.3f}")
    print(f"  (want NEGATIVE; the embedding TABLE gave -0.015)")
    print(f"  embedding spread: mean pairwise distance {float(ed.mean()):.3f} "
          f"(sd {float(ed.std()):.3f})")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
