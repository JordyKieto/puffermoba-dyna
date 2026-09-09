"""Can a learned function predict where optimization goes?

Training is a dynamical system: (theta_t, s_t, B_t) -> (theta_{t+1}, s_{t+1}).
The question is whether a cheap learned F can jump it forward:

    F(theta_t, context, k) -> theta_{t+k}

without executing the k forward/backward passes.

This is deliberately shrunk until direct weight prediction is *possible*, so we
learn whether the idea works at all before asking whether it saves compute. The
student is a ~700-parameter MLP; F is far larger than the thing it predicts.
That is fine here. Cheapness is a scaling question and only matters if the
prediction works in the first place.

Restricted family, as the hypothesis requires: same architecture, same theta_0,
same task family. Only data order and learning rate vary.

WHAT MAKES OR BREAKS THIS: the baselines. A predictor that emits theta_t
unchanged already looks good on weight MSE, because k steps move the weights
very little relative to their norm. So every number below is reported against:

  persistence      theta_t             (do nothing)
  mean delta       theta_t + mubar     (the average training step, no input)
  velocity         theta_t + k*(theta_t - theta_{t-1})   (linear extrapolation)

and scored by LOSS, not weight error. Beating persistence on weight MSE is
worth nothing; beating it on loss is the whole claim.

The final test is the one that matters: jump k steps with F, then resume REAL
SGD for m steps, and ask whether you land where k+m real steps would have.

Run:  python -m optwm.predict --runs 256 --steps 300
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F_


# ------------------------------------------------------------ student (tiny)

D_IN, D_HID, D_OUT = 20, 32, 3
SHAPES = [(D_IN, D_HID), (D_HID,), (D_HID, D_OUT), (D_OUT,)]
NP = sum(int(torch.tensor(s).prod()) for s in SHAPES)


def sforward(w, x):
    """Functional MLP so a flat weight vector is the only state."""
    i = 0
    out = []
    for s in SHAPES:
        n = int(torch.tensor(s).prod())
        out.append(w[i:i + n].view(*s)); i += n
    W1, b1, W2, b2 = out
    return torch.relu(x @ W1 + b1) @ W2 + b2


def make_task(seed, n=2048):
    """Teacher-student: a fixed random teacher labels gaussian inputs."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D_IN, generator=g)
    tw = torch.randn(NP, generator=g) * 0.6
    with torch.no_grad():
        y = sforward(tw, x).argmax(1)
    return x[:n // 2], y[:n // 2], x[n // 2:], y[n // 2:]


def init_w(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(NP)
    i = 0
    for s in SHAPES:
        n = int(torch.tensor(s).prod())
        if len(s) == 2:
            w[i:i + n] = torch.randn(n, generator=g) * (1.0 / s[0] ** 0.5)
        i += n
    return w


def run(task, w0, lr, steps, every, batch, seed):
    xtr, ytr, xte, yte = task
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.SGD([w], lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    ck = []
    for t in range(steps + 1):
        if t % every == 0:
            ck.append(w.detach().clone())
        idx = torch.randint(0, len(xtr), (batch,), generator=g)
        loss = F_.cross_entropy(sforward(w, xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return torch.stack(ck)


def evaluate(w, task):
    xtr, ytr, xte, yte = task
    with torch.no_grad():
        lg = sforward(w, xte)
        return (float(F_.cross_entropy(lg, yte)),
                float((lg.argmax(1) == yte).float().mean()))


# ------------------------------------------------------------------- main

def build_F(X, PREV, LR, Y, tr, epochs):
    mu, sd = (Y - X)[tr].mean(0), (Y - X)[tr].std(0).clamp_min(1e-8)
    net = nn.Sequential(nn.Linear(NP * 2 + 1, 512), nn.SiLU(),
                        nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, NP))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xin = torch.cat([X, X - PREV, LR], 1)[tr]
    tgt = (((Y - X) - mu) / sd)[tr]
    for _ in range(epochs):
        i = torch.randint(0, len(xin), (256,))
        loss = F_.mse_loss(net(xin[i]), tgt[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--res", type=int, default=5, help="checkpoint resolution")
    ap.add_argument("--ks", type=str, default="1,2,4,8,16",
                    help="jump sizes, in units of --res steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--resume", type=int, default=150)
    ap.add_argument("--basin-runs", type=int, default=12)
    ap.add_argument("--vary-task", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"student {NP} params | {a.runs} runs x {a.steps} steps | "
          f"{'varying task' if a.vary_task else 'single task'}")

    base_task = make_task(0)
    w0 = init_w(0)
    lrs = [0.02, 0.05, 0.1, 0.2]
    t0 = time.time()
    trajs, ctxs, tasks = [], [], []
    for i in range(a.runs):
        task = make_task(1000 + i) if a.vary_task else base_task
        trajs.append(run(task, w0, lrs[i % 4], a.steps, a.res, a.batch, seed=i))
        ctxs.append(lrs[i % 4]); tasks.append(task)
    print(f"generated ({time.time()-t0:.0f}s)   "
          f"run0 loss {evaluate(trajs[0][0], base_task)[0]:.3f} -> "
          f"{evaluate(trajs[0][-1], base_task)[0]:.3f}")

    print(f"\nKILLER PLOT: real SGD steps eliminated vs quality retained")
    print("  '% of true gain' = fraction of the loss improvement that the true")
    print("  jump achieves, recovered by the method. 0% = no better than doing")
    print("  nothing; 100% = matches real training.\n")
    print(f"  {'k steps':>8} {'headroom':>10} {'persist':>9} {'mean-d':>8} "
          f"{'velocity':>9} {'learned F':>10}")

    for kk in map(int, a.ks.split(",")):
        X, Y, PREV, LR, RUN = [], [], [], [], []
        for i, C in enumerate(trajs):
            if len(C) < kk + 2:
                continue
            X.append(C[1:-kk]); Y.append(C[1 + kk:]); PREV.append(C[:-kk - 1])
            m = len(C) - kk - 1
            LR += [ctxs[i]] * m; RUN += [i] * m
        X, Y, PREV = torch.cat(X), torch.cat(Y), torch.cat(PREV)
        LR = torch.tensor(LR).float().unsqueeze(1); RUN = torch.tensor(RUN)
        ho = RUN >= (a.runs - a.holdout)
        net, mu, sd = build_F(X, PREV, LR, Y, ~ho, a.epochs)

        preds = {"persist": lambda x, p, l: x,
                 "mean-d": lambda x, p, l: x + mu,
                 "velocity": lambda x, p, l: x + (x - p) * kk,
                 "learned F": lambda x, p, l: x + (
                     net(torch.cat([x, x - p, l], 1)).detach() * sd + mu)}

        hx, hp, hl, hy = X[ho], PREV[ho], LR[ho], Y[ho]
        cand = torch.randperm(len(hx))[:192]
        lx = torch.tensor([evaluate(hx[j], base_task)[0] for j in cand])
        ly = torch.tensor([evaluate(hy[j], base_task)[0] for j in cand])
        order = (lx - ly).argsort(descending=True)[:64]
        sel, lx_s, ly_s = cand[order], lx[order], ly[order]
        base_l, head = float(lx_s.mean()), float((lx_s - ly_s).mean())
        row = f"  {kk*a.res:>8} {head:>10.4f}"
        for nm, f in preds.items():
            P = f(hx[sel], hp[sel], hl[sel])
            ml = sum(evaluate(P[i], base_task)[0] for i in range(len(sel))) / len(sel)
            row += f" {100*(base_l-ml)/head:>8.0f}%" if nm != "persist" else \
                   f" {100*(base_l-ml)/head:>8.0f}%"
        print(row)
        if kk == int(a.ks.split(",")[len(a.ks.split(","))//2]):
            keep = (preds, ho, X, PREV, LR, kk)

    # ---------------------------------------- basin test, averaged over runs
    preds, ho, X, PREV, LR, kk = keep
    print(f"\nBASIN TEST (k={kk*a.res} predicted steps, then m={a.resume} REAL "
          f"SGD steps)\n  averaged over {a.basin_runs} held-out runs -- a single "
          f"run is pure noise here")
    agg = {nm: [] for nm in list(preds) + ["TRUE jump"]}
    for r in range(a.runs - a.basin_runs, a.runs):
        C, task, lr = trajs[r], tasks[r], ctxs[r]
        j = max(1, len(C) // 6)
        th, pv = C[j].unsqueeze(0), C[j - 1].unsqueeze(0)
        starts = {nm: f(th, pv, torch.tensor([[lr]]))[0] for nm, f in preds.items()}
        starts["TRUE jump"] = C[min(j + kk, len(C) - 1)]
        for nm, w in starts.items():
            xtr, ytr, _, _ = task
            ww = w.clone().requires_grad_(True)
            o = torch.optim.SGD([ww], lr=lr, momentum=0.9)
            g = torch.Generator().manual_seed(777 + r)
            for _ in range(a.resume):
                i = torch.randint(0, len(xtr), (a.batch,), generator=g)
                L = F_.cross_entropy(sforward(ww, xtr[i]), ytr[i])
                o.zero_grad(set_to_none=True); L.backward(); o.step()
            agg[nm].append(evaluate(ww.detach(), task)[0])
    import statistics as st
    print(f"  {'method':<13} {'loss':>9} {'vs TRUE (paired)':>18} {'t':>7}")
    tj = agg["TRUE jump"]
    for nm in list(preds) + ["TRUE jump"]:
        v = agg[nm]; d = [x - y for x, y in zip(v, tj)]
        m = st.mean(v)
        if nm == "TRUE jump":
            print(f"  {nm:<13} {m:>9.4f} {'--':>18} {'--':>7}")
        else:
            sem = st.stdev(d) / len(d) ** .5 if len(d) > 1 else float("nan")
            t = st.mean(d) / sem if sem > 1e-12 else float("nan")
            print(f"  {nm:<13} {m:>9.4f} {st.mean(d):>+11.4f}+/-{sem:.4f} "
                  f"{t:>+7.2f}")
    dp = [x - y for x, y in zip(agg["learned F"], agg["persist"])]
    sem = st.stdev(dp) / len(dp) ** .5
    print(f"\n  learned F vs persistence (the baseline that matters): "
          f"{st.mean(dp):+.4f} +/- {sem:.4f}  t={st.mean(dp)/sem:+.2f}")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
