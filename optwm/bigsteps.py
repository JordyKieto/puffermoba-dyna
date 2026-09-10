"""Big jumps: does teacher forcing win once each hop is large?

Autoregressive training exists to stop error compounding across MANY hops. But
compute saved goes as the jump size, so there is a second lever: make each hop
huge and take few of them. Fewer hops means less to compound, and more steps
skipped per F call. Both effects favour big k, and they favour the CHEAP
objective -- teacher forcing costs 1 forward pass per training step where
autoregression costs H.

The comparison is held at FIXED TOTAL PROGRESS. Every arm simulates the same
number of real SGD steps; only the jump size (and hence the hop count) varies.
That is the only way to compare k values without confounding accuracy with how
far the rollout went.

  k*res =  50 real steps -> 16 hops to cover 800
  k*res = 400 real steps ->  2 hops to cover 800

Speedup is computed from MACs, not wall clock, so it is independent of how well
either implementation is optimised:

  one student SGD step   ~ 3 * batch * NP        (fwd + bwd)
  one F call             ~ 4 * NP * H_F + H_F^2
  speedup                = (k*res) * 3*batch*NP / (F call)

Both scale linearly in NP, so the ratio is independent of student size -- which
is why the break-even jump size (~H_F/batch) does not worsen as models grow.

Run:  python -m optwm.bigsteps --hid 128 --total 800
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .sim import (init_w, make_task, numel, rollout_real, shapes_for, sfwd,
                  tloss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--runs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--ks", type=str, default="10,20,40,80",
                    help="jump sizes in units of res")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--total", type=int, default=800,
                    help="real steps every arm must cover")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--fwidth", type=int, default=640)
    ap.add_argument("--mask-state", type=float, default=0.5)
    ap.add_argument("--no-autoreg", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid)
    NP = numel(SH)
    task = make_task(SH, 11)
    xtr, ytr = task[0], task[1]
    w0 = init_w(SH, 0)
    lrs = [0.02, 0.05, 0.1, 0.2]
    print(f"student {NP:,} params (hidden {a.hid})   F width {a.fwidth}   "
          f"covering {a.total} real steps per arm")

    t0 = time.time()
    trajs, LRs = [], []
    for i in range(a.runs):
        lr = lrs[i % len(lrs)]
        trajs.append(rollout_real(task, w0, lr, a.steps, a.res, a.batch, SH,
                                  seed=i))
        LRs.append(lr)
    TW = torch.stack([c for c, _, _ in trajs])
    TL = torch.stack([l for _, l, _ in trajs])
    TB = torch.stack([b for _, _, b in trajs])
    LRt = torch.tensor(LRs).float()
    NCK = TW.shape[1]
    print(f"{a.runs} trajectories x {a.steps} steps ({time.time()-t0:.0f}s)   "
          f"loss {float(TL[0,0]):.3f} -> {float(TL[0,-1]):.3f}")

    # cost model, in MACs
    step_mac = 3 * a.batch * NP
    f_mac = 4 * NP * a.fwidth + a.fwidth ** 2

    hold = rollout_real(task, w0, 0.07, a.steps, a.res, a.batch, SH, seed=777)
    C, L, B = hold
    start = max(1, NCK // 8)

    print(f"\n{'jump':>6} {'hops':>5} {'REAL':>8} {'no-move':>8} "
          f"{'TEACHER-F':>10} {'autoreg':>9} {'speedup':>8}")

    for k in [int(v) for v in a.ks.split(",")]:
        jump = k * a.res
        hops = max(1, round(a.total / jump))
        if start + hops * k >= NCK:
            print(f"{jump:>6} -- trajectory too short, skipped")
            continue

        # rows valid for the deepest rollout this arm needs
        rows = []
        for j in range(a.runs):
            for i in range(1, NCK - k * max(a.horizon, 1) - 1):
                rows.append((j, i, k))
        idx = torch.tensor(rows)
        sm = idx[torch.randperm(len(idx))[:4096]]
        j_, i_ = sm[:, 0], sm[:, 1]
        dmu = (TW[j_, i_ + k] - TW[j_, i_]).mean(0)
        dsd = (TW[j_, i_ + k] - TW[j_, i_]).std(0).clamp_min(1e-8)
        vmu, vsd = TB[j_, i_].mean(0), TB[j_, i_].std(0).clamp_min(1e-8)
        lmu, lsd = TL[j_, i_].mean(), TL[j_, i_].std().clamp_min(1e-8)

        def train(H):
            net = nn.Sequential(
                nn.Linear(2 * NP + 1 + 2, a.fwidth), nn.SiLU(),
                nn.Linear(a.fwidth, a.fwidth), nn.SiLU(),
                nn.Linear(a.fwidth, 2 * NP + 1))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3,
                                    weight_decay=1e-4)

            def step_F(th, bf, lo, act):
                o = net(torch.cat([th, bf, (lo - lmu) / lsd, act], 1))
                return (th + o[:, :NP] * dsd + dmu,
                        o[:, NP:2 * NP] * vsd + vmu,
                        o[:, 2 * NP:] * lsd + lmu)

            for _ in range(a.epochs):
                b = idx[torch.randint(0, len(idx), (128,))]
                j, i = b[:, 0], b[:, 1]
                act = torch.stack([LRt[j],
                                   torch.full_like(LRt[j], jump / 100.0)], 1)
                cth, cbf, clo = TW[j, i], TB[j, i], TL[j, i].unsqueeze(1)
                if a.mask_state > 0:
                    cbf = cbf * (torch.rand(len(j), 1) > a.mask_state).float()
                loss = 0.0
                for h in range(1, H + 1):
                    cth, cbf, clo = step_F(cth, cbf, clo, act)
                    ti = i + h * k
                    loss = loss + (((cth - TW[j, ti]) / dsd).pow(2).mean()
                                   + ((cbf - TB[j, ti]) / vsd).pow(2).mean()
                                   + ((clo - TL[j, ti].unsqueeze(1)) / lsd)
                                   .pow(2).mean())
                opt.zero_grad(set_to_none=True)
                (loss / H).backward(); opt.step()
            return step_F

        def free_run(step_F):
            st = (C[start].unsqueeze(0), B[start].unsqueeze(0),
                  torch.tensor([[L[start]]]))
            act = torch.tensor([[0.07, jump / 100.0]])
            for _ in range(hops):
                with torch.no_grad():
                    st = step_F(*st, act)
            return tloss(st[0][0], xtr, ytr, SH)

        tf = free_run(train(1))
        ar = free_run(train(a.horizon)) if not a.no_autoreg else float("nan")
        ti = min(start + hops * k, NCK - 1)
        speed = jump * step_mac / f_mac
        print(f"{jump:>6} {hops:>5} {float(L[ti]):>8.4f} "
              f"{float(L[start]):>8.4f} {tf:>10.4f} {ar:>9.4f} "
              f"{speed:>7.1f}x")

    print(f"\n  every arm covers {a.total} real steps; 'hops' is how many F")
    print(f"  calls that took. speedup = MACs of the real steps skipped over")
    print(f"  MACs of the F calls that replaced them.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
