"""Choosing actions by asking F: can we optimise THROUGH the simulator?

F now predicts validation loss, so it can be queried counterfactually: from this
exact optimizer state, which learning rate gives the lowest validation loss k
steps from now? That is one greedy step of planning in optimization space, and
it is the precursor to anything like PPO over training decisions.

It also tests the assumption such a method would rest on. Predicting a
trajectory needs F to be right on average; CHOOSING needs F to be right about
DIFFERENCES between actions -- a much stronger requirement, and the one that
failed earlier when the state carried velocity instead of momentum (F produced
17x too little spread across learning rates and ranked them at chance).

Every counterfactual starts from the same (theta, momentum), and the real
comparison resumes SGD from that same optimizer state, so the question is well
posed rather than comparing against a run that had different history.

Reported:
  rank corr    does F order the actions like reality does
  top-1        does F's argmin match the true argmin
  regret       real val loss of F's pick minus the oracle's pick
  vs fixed     the same regret for always using one default lr
  vs random    and for picking uniformly at random

`vs fixed` is the baseline that matters. A selector that cannot beat "always use
0.1" is not doing anything, however good its correlations look.

Run:  python -m optwm.plan --runs 48 --steps 800
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .sim import init_w, make_task, numel, rollout_real, shapes_for, sfwd


def vloss(w, xte, yte, SH):
    with torch.no_grad():
        return float(F_.cross_entropy(sfwd(w, xte, SH), yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--runs", type=int, default=48)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--k", type=int, default=20, help="jump = k*res steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fwidth", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--mask-state", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=24, help="states to choose at")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid); NP = numel(SH)
    task = make_task(SH, 11)
    xtr, ytr, xte, yte = task
    w0 = init_w(SH, 0)
    CAND = [0.02, 0.05, 0.1, 0.2]
    jump = a.k * a.res
    print(f"student {NP:,} params | jump {jump} steps | actions {CAND}")

    t0 = time.time()
    trajs, LRs = [], []
    for i in range(a.runs):
        lr = CAND[i % len(CAND)]
        C, L, B = rollout_real(task, w0, lr, a.steps, a.res, a.batch, SH, seed=i)
        V = torch.tensor([vloss(C[t], xte, yte, SH) for t in range(len(C))])
        trajs.append((C, L, B, V)); LRs.append(lr)
    TW = torch.stack([x[0] for x in trajs]); TL = torch.stack([x[1] for x in trajs])
    TB = torch.stack([x[2] for x in trajs]); TV = torch.stack([x[3] for x in trajs])
    LRt = torch.tensor(LRs).float(); NCK = TW.shape[1]
    print(f"{a.runs} trajectories ({time.time()-t0:.0f}s)")

    rows = [(j, i) for j in range(a.runs) for i in range(1, NCK - a.k - 1)]
    idx = torch.tensor(rows)
    sm = idx[torch.randperm(len(idx))[:4096]]
    j_, i_ = sm[:, 0], sm[:, 1]
    dmu = (TW[j_, i_ + a.k] - TW[j_, i_]).mean(0)
    dsd = (TW[j_, i_ + a.k] - TW[j_, i_]).std(0).clamp_min(1e-8)
    bmu, bsd = TB[j_, i_].mean(0), TB[j_, i_].std(0).clamp_min(1e-8)
    lmu, lsd = TL[j_, i_].mean(), TL[j_, i_].std().clamp_min(1e-8)
    qmu, qsd = TV[j_, i_].mean(), TV[j_, i_].std().clamp_min(1e-8)

    net = nn.Sequential(nn.Linear(2 * NP + 2 + 2, a.fwidth), nn.SiLU(),
                        nn.Linear(a.fwidth, a.fwidth), nn.SiLU(),
                        nn.Linear(a.fwidth, 2 * NP + 2))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    def step_F(th, bf, lo, vl, act):
        o = net(torch.cat([th, bf, (lo - lmu) / lsd, (vl - qmu) / qsd, act], 1))
        return (th + o[:, :NP] * dsd + dmu, o[:, NP:2 * NP] * bsd + bmu,
                o[:, 2 * NP:2 * NP + 1] * lsd + lmu,
                o[:, 2 * NP + 1:] * qsd + qmu)

    for _ in range(a.epochs):
        b = idx[torch.randint(0, len(idx), (128,))]
        j, i = b[:, 0], b[:, 1]
        act = torch.stack([LRt[j], torch.full_like(LRt[j], jump / 100.0)], 1)
        bf = TB[j, i]
        if a.mask_state > 0:
            bf = bf * (torch.rand(len(j), 1) > a.mask_state).float()
        ph, pb, pl, pv = step_F(TW[j, i], bf, TL[j, i].unsqueeze(1),
                                TV[j, i].unsqueeze(1), act)
        ti = i + a.k
        loss = (((ph - TW[j, ti]) / dsd).pow(2).mean()
                + ((pb - TB[j, ti]) / bsd).pow(2).mean()
                + ((pl - TL[j, ti].unsqueeze(1)) / lsd).pow(2).mean()
                + ((pv - TV[j, ti].unsqueeze(1)) / qsd).pow(2).mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"F trained ({time.time()-t0:.0f}s)\n")

    # ---- greedy action selection at held-out states
    hold = rollout_real(task, w0, 0.07, a.steps, a.res, a.batch, SH, seed=777)
    Ch, Lh, Bh = hold
    Vh = torch.tensor([vloss(Ch[t], xte, yte, SH) for t in range(len(Ch))])

    # CONSTANT-ACTION BASELINE. If one action is optimal everywhere, a selector
    # that ignores the state entirely scores a perfect top-1 and zero regret,
    # and every planning number below is vacuous. This tracks it explicitly.
    ranks, hits, reg_F, reg_fix, reg_rand = [], [], [], [], []
    best_hist, reg_const = [], {c: [] for c in CAND}
    spread_p, spread_r = [], []
    g = torch.Generator().manual_seed(5)
    print(f"{'state':>6} " + " ".join(f"{'lr '+str(c):>16}" for c in CAND)
          + f" {'F picks':>8} {'best':>6}")
    print(f"{'':>6} " + " ".join(f"{'pred / real':>16}" for c in CAND))
    for t in range(a.trials):
        si = 1 + int(torch.randint(1, NCK - a.k - 1, (1,), generator=g))
        th, bf = Ch[si].unsqueeze(0), Bh[si].unsqueeze(0)
        lo = torch.tensor([[float(Lh[si])]]); vl = torch.tensor([[float(Vh[si])]])
        pred, real = [], []
        for c in CAND:
            act = torch.tensor([[c, jump / 100.0]])
            with torch.no_grad():
                _, _, _, pv = step_F(th, bf, lo, vl, act)
            pred.append(float(pv[0, 0]))
            Cr, _, _ = rollout_real(task, Ch[si], c, jump, jump, a.batch, SH,
                                    seed=4242, buf0=Bh[si])
            real.append(vloss(Cr[-1], xte, yte, SH))
        p, r = torch.tensor(pred), torch.tensor(real)
        pc, rc = p - p.mean(), r - r.mean()
        rk = float((pc * rc).sum() / (pc.norm() * rc.norm()).clamp_min(1e-9))
        pick, best = int(p.argmin()), int(r.argmin())
        ranks.append(rk); hits.append(int(pick == best))
        reg_F.append(real[pick] - real[best])
        reg_fix.append(real[CAND.index(0.1)] - real[best])
        reg_rand.append(st.mean(real) - real[best])
        best_hist.append(CAND[best])
        for q, c in enumerate(CAND):
            reg_const[c].append(real[q] - real[best])
        spread_p.append(float(p.max() - p.min()))
        spread_r.append(float(r.max() - r.min()))
        if t < 8:
            cells = " ".join(f"{pred[q]:>7.4f}/{real[q]:<8.4f}"
                             for q in range(len(CAND)))
            print(f"{si*a.res:>6} {cells} {CAND[pick]:>8} {CAND[best]:>6}")

    print(f"\n  over {a.trials} states, jump {jump} steps:")
    print(f"    rank corr(pred, real) across actions   {st.mean(ranks):+.3f}")
    print(f"    top-1 (F's argmin == true argmin)      "
          f"{100*st.mean(hits):.0f}%   (chance {100/len(CAND):.0f}%)")
    print(f"    regret of F's choice                   {st.mean(reg_F):+.4f}")
    print(f"    regret of always lr=0.1                {st.mean(reg_fix):+.4f}")
    print(f"    regret of a random choice              {st.mean(reg_rand):+.4f}")
    from collections import Counter
    cnt = Counter(best_hist)
    print(f"\n  WHICH ACTION IS ACTUALLY BEST, across states: "
          f"{dict(sorted(cnt.items()))}")
    bc = min(reg_const, key=lambda c: st.mean(reg_const[c]))
    print(f"    regret of ALWAYS lr={bc} (best constant)  "
          f"{st.mean(reg_const[bc]):+.4f}")
    print(f"    F's action spread {st.mean(spread_p):.4f} vs real "
          f"{st.mean(spread_r):.4f}  ({st.mean(spread_r)/max(st.mean(spread_p),1e-9):.0f}x too flat)")
    if len(cnt) == 1:
        print(f"\n  DEGENERATE: one action is optimal in every state, so a")
        print(f"  state-blind selector also scores top-1 100% with zero regret.")
        print(f"  Nothing here demonstrates planning.")
    else:
        print(f"\n  F beats the best constant action by "
              f"{st.mean(reg_const[bc]) - st.mean(reg_F):+.4f} -- that gap, not")
        print(f"  the top-1 rate, is the evidence of state-dependent choice.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
