"""A simulator of optimization: state = weights, action = the jump.

Everything before this was an extrapolator, not a simulator, for one concrete
reason: F needed a `velocity` input it could not produce, so real SGD steps had
to be interleaved to manufacture its next input. The real steps were feeding F,
not verifying it.

Two changes make it a transition function you can actually step:

  1. F predicts the FULL next state, velocity included. Its output is a valid
     input to itself, so it can free-run with no real computation at all.
  2. It is trained AUTOREGRESSIVELY -- on rollouts of its own predictions, not
     only on real checkpoints. One-step training is teacher forcing, and a model
     that has never seen its own outputs has no reason to be stable on them.

    state   s = (theta, velocity, loss)        weights, recent movement, where we are
    action  a = (lr, k)                        how hard and how far to jump
    F(s, a) -> s'                              one jump of k real steps

`loss` is carried in the state because it is the one cheap observable: it lets a
rollout be sanity-checked without evaluating the real network.

THE MEASUREMENT THAT DECIDES THIS. Free-running rollout: start from a real
state, apply F repeatedly with ZERO real steps, and evaluate the true loss of
the weights it invents at each hop. Compared against the real trajectory at the
same step count (the target) and against never moving (the floor). Teacher-forced
and autoregressive F are trained on identical data so the only difference is the
objective.

Run:  python -m optwm.sim --hid 64 --smoke
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_

D_IN, D_OUT = 20, 3


def shapes_for(hid):
    return [(D_IN, hid), (hid,), (hid, D_OUT), (D_OUT,)]


def numel(shapes):
    return sum(int(torch.tensor(s).prod()) for s in shapes)


def sfwd(w, x, shapes):
    i, p = 0, []
    for s in shapes:
        n = int(torch.tensor(s).prod())
        p.append(w[i:i + n].view(*s)); i += n
    W1, b1, W2, b2 = p
    return torch.relu(x @ W1 + b1) @ W2 + b2


def make_task(shapes, seed, n=2048):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D_IN, generator=g)
    tw = torch.randn(numel(shapes), generator=g) * 0.6
    with torch.no_grad():
        y = sfwd(tw, x, shapes).argmax(1)
    return x[:n // 2], y[:n // 2], x[n // 2:], y[n // 2:]


def init_w(shapes, seed):
    g = torch.Generator().manual_seed(seed)
    w, i = torch.zeros(numel(shapes)), 0
    for s in shapes:
        n = int(torch.tensor(s).prod())
        if len(s) == 2:
            w[i:i + n] = torch.randn(n, generator=g) * (1.0 / s[0] ** 0.5)
        i += n
    return w


def tloss(w, x, y, shapes):
    with torch.no_grad():
        return float(F_.cross_entropy(sfwd(w, x, shapes), y))


def rollout_real(task, w0, lr, steps, res, batch, shapes, seed):
    """One authoritative run. Returns checkpoints and their train losses."""
    xtr, ytr, _, _ = task
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.SGD([w], lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    ck, ls = [], []
    for t in range(steps + 1):
        if t % res == 0:
            ck.append(w.detach().clone())
            ls.append(tloss(w.detach(), xtr, ytr, shapes))
        i = torch.randint(0, len(xtr), (batch,), generator=g)
        L = F_.cross_entropy(sfwd(w, xtr[i], shapes), ytr[i])
        opt.zero_grad(set_to_none=True); L.backward(); opt.step()
    return torch.stack(ck), torch.tensor(ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=64, help="student hidden width")
    ap.add_argument("--runs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--ks", type=str, default="2,4,8", help="jump sizes (x res)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4, help="autoregressive depth")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--eval-hops", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.runs, a.steps, a.epochs, a.horizon = 24, 300, 800, 3

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid)
    NP = numel(SH)
    ks = [int(v) for v in a.ks.split(",")]
    print(f"student: {NP:,} params (hidden {a.hid})   jumps: "
          f"{[k*a.res for k in ks]} real steps")

    task = make_task(SH, 11)
    xtr, ytr = task[0], task[1]
    w0 = init_w(SH, 0)
    lrs = [0.02, 0.05, 0.1, 0.2]

    t0 = time.time()
    trajs, LRs = [], []
    for i in range(a.runs):
        lr = lrs[i % len(lrs)]
        C, L = rollout_real(task, w0, lr, a.steps, a.res, a.batch, SH, seed=i)
        trajs.append((C, L)); LRs.append(lr)
    print(f"{a.runs} trajectories ({time.time()-t0:.0f}s)   "
          f"loss {float(trajs[0][1][0]):.3f} -> {float(trajs[0][1][-1]):.3f}")
    TW = torch.stack([c for c, _ in trajs])          # (runs, ckpts, NP)
    TL = torch.stack([l for _, l in trajs])          # (runs, ckpts)
    LRt = torch.tensor(LRs).float()
    NCK = TW.shape[1]

    # ---------------- the tabular transition dataset -------------------------
    # one row per (state, action, next state). vel is the last `res` steps of
    # movement, which is what makes the state self-sufficient once F predicts it.
    rows = []
    for j, (C, L) in enumerate(trajs):
        for k in ks:
            for i in range(1, len(C) - k):
                rows.append((j, i, k, LRs[j]))
    idx = torch.tensor([(r[0], r[1], r[2]) for r in rows])
    ACT = torch.tensor([[r[3], r[2] * a.res / 100.0] for r in rows]).float()
    print(f"\ntabular dataset: {len(rows):,} rows")
    print(f"  state  s = (theta {NP}, vel {NP}, loss 1)   = {2*NP+1} cols")
    print(f"  action a = (lr 1, k/100 1)                  = 2 cols")
    print(f"  next   s' = (theta' {NP}, vel' {NP}, loss' 1)")

    def gather(sub):
        j, i, k = sub[:, 0], sub[:, 1], sub[:, 2]
        th, pv = TW[j, i], TW[j, i - 1]
        lo = TL[j, i].unsqueeze(1)
        th2, pv2 = TW[j, i + k], TW[j, i + k - 1]
        lo2 = TL[j, i + k].unsqueeze(1)
        return th, th - pv, lo, th2, th2 - pv2, lo2

    sample = idx[torch.randperm(len(idx))[:4096]]
    th, vel, lo, th2, vel2, lo2 = gather(sample)
    dmu, dsd = (th2 - th).mean(0), (th2 - th).std(0).clamp_min(1e-8)
    vmu, vsd = vel2.mean(0), vel2.std(0).clamp_min(1e-8)
    lmu, lsd = lo2.mean(), lo2.std().clamp_min(1e-8)

    def build(autoreg):
        net = nn.Sequential(nn.Linear(2 * NP + 1 + 2, 640), nn.SiLU(),
                            nn.Linear(640, 640), nn.SiLU(),
                            nn.Linear(640, 2 * NP + 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

        def step_F(th_, vel_, lo_, act):
            o = net(torch.cat([th_, vel_, (lo_ - lmu) / lsd, act], 1))
            dth = o[:, :NP] * dsd + dmu
            nv = o[:, NP:2 * NP] * vsd + vmu
            nl = o[:, 2 * NP:] * lsd + lmu
            return th_ + dth, nv, nl

        H = a.horizon if autoreg else 1
        for e in range(a.epochs):
            # sample rows whose trajectory has room for H hops of the same k
            b = idx[torch.randint(0, len(idx), (128,))]
            k = b[:, 2]
            ok = (b[:, 1] + k * H) < (NCK - 1)
            b = b[ok]
            if len(b) < 8:
                continue
            j, i, k = b[:, 0], b[:, 1], b[:, 2]
            act = torch.stack([LRt[j], k.float() * a.res / 100.0], 1)
            cth, cvel, clo = TW[j, i], TW[j, i] - TW[j, i - 1], TL[j, i].unsqueeze(1)
            loss = 0.0
            for h in range(1, H + 1):
                # ROLL ON OWN OUTPUT: no re-anchoring to the real trajectory
                cth, cvel, clo = step_F(cth, cvel, clo, act)
                ti = i + h * k
                tth, tpv = TW[j, ti], TW[j, ti - 1]
                tlo = TL[j, ti].unsqueeze(1)
                loss = loss + (((cth - tth) / dsd).pow(2).mean()
                               + ((cvel - (tth - tpv)) / vsd).pow(2).mean()
                               + ((clo - tlo) / lsd).pow(2).mean())
            opt.zero_grad(set_to_none=True)
            (loss / H).backward(); opt.step()
        return net, step_F

    print(f"\ntraining two F's on identical data, differing only in objective")
    nets = {}
    for tag, ar in (("teacher-forced (H=1)", False),
                    (f"autoregressive (H={a.horizon})", True)):
        nets[tag] = build(ar)
        print(f"  {tag} done ({time.time()-t0:.0f}s)")

    # ------------- free-running rollout: ZERO real steps --------------------
    kk = ks[len(ks) // 2]
    print(f"\nFREE-RUNNING ROLLOUT -- jump {kk*a.res} real steps at a time, "
          f"no corrections\n  true train loss of the weights F invents, "
          f"vs the real run at the same step count")
    hold = rollout_real(task, w0, 0.07, a.steps, a.res, a.batch, SH, seed=777)
    C, L = hold
    start = max(1, len(C) // 6)
    print(f"\n  {'hop':>4} {'real steps':>11} {'REAL':>8} {'never move':>11} "
          f"{'teacher-forced':>15} {'autoregressive':>15}")
    state = {}
    for tag, (net, step_F) in nets.items():
        state[tag] = (C[start].unsqueeze(0), (C[start] - C[start - 1]).unsqueeze(0),
                      torch.tensor([[L[start]]]))
    act = torch.tensor([[0.07, kk * a.res / 100.0]])
    for hop in range(1, a.eval_hops + 1):
        ti = start + hop * kk
        if ti >= len(C):
            break
        out = {}
        for tag, (net, step_F) in nets.items():
            with torch.no_grad():
                state[tag] = step_F(*state[tag], act)
            out[tag] = tloss(state[tag][0][0], xtr, ytr, SH)
        print(f"  {hop:>4} {hop*kk*a.res:>11} {float(L[ti]):>8.4f} "
              f"{float(L[start]):>11.4f} "
              f"{out['teacher-forced (H=1)']:>15.4f} "
              f"{out[f'autoregressive (H={a.horizon})']:>15.4f}")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
