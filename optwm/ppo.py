"""PPO inside a learned weight-space transition model, anchored in reality.

The full loop:

  1. a policy proposes a MOVE in weight space
  2. a transition model T predicts the resulting validation loss
  3. PPO trains the policy entirely inside T -- no data forward passes
  4. periodically the policy's moves are EXECUTED for real, the true validation
     loss is measured, and that goes back into T's buffer
  5. T is retrained on it, and the loop repeats

Step 4 is what makes this work rather than the random-dataset version. A random
unit vector in 1,539 dimensions has expected |cosine| ~ 0.02 with any fixed
direction, so random moves are essentially orthogonal to the descent direction
and a model trained on them has no data about the moves that matter -- measured
at correlation +0.073 on real SGD moves. On-policy collection fixes that by
construction: T only ever needs to be right about moves the policy proposes.

ACTION SPACE. The policy does NOT emit a free 1,539-dim direction. That failed
earlier: optimising 1,539 dimensions against a 512-example validation signal is
underdetermined by 3x, and it produced a 24% val-loss gain that was PURE val-set
fitting -- worse than not moving at all on an untouched split. So the action is
a coefficient vector over a small fixed basis (momentum plus K random
directions), turning it into ~9 parameters against 512 examples.

THE CONTROL THAT DECIDES EVERYTHING. The policy is rewarded on validation loss,
so improving validation is circular. TEST is a third split, never seen by T, the
policy, or the reward. Only a TEST gain means anything.

Run:  python -m optwm.ppo --iters 40
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
    i, p = 0, []
    for sh in shapes:
        n = int(torch.tensor(sh).prod())
        p.append(W[:, i:i + n].view(-1, *sh)); i += n
    W1, b1, W2, b2 = p
    h = torch.relu(torch.einsum("nd,bdh->bnh", x, W1) + b1[:, None, :])
    return torch.einsum("bnh,bho->bno", h, W2) + b2[:, None, :]


def losses(W, x, y, shapes, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(W), chunk):
            lg = fwd_batch(W[s:s + chunk], x, shapes)
            out.append(F_.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                        y.repeat(len(lg)), reduction="none")
                       .view(len(lg), -1).mean(1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--basis", type=int, default=8, help="random basis vectors")
    ap.add_argument("--step-frac", type=float, default=0.03,
                    help="max move norm as a fraction of ||theta||")
    ap.add_argument("--horizon", type=int, default=6, help="imagined hops")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--anchor-every", type=int, default=2)
    ap.add_argument("--anchor-n", type=int, default=16, help="real evals/anchor")
    ap.add_argument("--t-epochs", type=int, default=400, help="T refit per anchor")
    ap.add_argument("--seed-runs", type=int, default=10)
    ap.add_argument("--seed-steps", type=int, default=300)
    ap.add_argument("--fwidth", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    SH = shapes_for(a.hid); NP = numel(SH)
    xtr, ytr, xte, yte = make_task(SH, 11)
    h = len(xte) // 2
    xva, yva, xts, yts = xte[:h], yte[:h], xte[h:], yte[h:]
    K = a.basis
    print(f"student {NP:,} params | action = {K+1} coefficients "
          f"(momentum + {K} random) | move <= {a.step_frac:.0%} of ||theta||")
    print(f"splits: {len(xtr)} train / {len(xva)} val (reward) / "
          f"{len(xts)} TEST (untouched)")

    g = torch.Generator().manual_seed(a.seed + 7)
    Brand = torch.randn(K, NP, generator=g)
    Brand = Brand / Brand.norm(dim=1, keepdim=True)

    t0 = time.time()
    # ---- start states from short real runs
    pool_w, pool_b = [], []
    for i in range(a.seed_runs):
        C, _, B = rollout_real((xtr, ytr, xva, yva), init_w(SH, 300 + i),
                               [0.02, 0.05, 0.1, 0.2][i % 4], a.seed_steps, 25,
                               64, SH, seed=700 + i)
        pool_w.append(C); pool_b.append(B)
    PW, PB = torch.cat(pool_w), torch.cat(pool_b)
    print(f"{len(PW)} start states ({time.time()-t0:.0f}s)")

    def make_basis(bf, th):
        """(B, K+1, NP) basis per state: momentum first, then fixed randoms."""
        m = bf / bf.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.cat([m.unsqueeze(1),
                          Brand.unsqueeze(0).expand(len(bf), -1, -1)], 1)

    def to_move(act, bf, th):
        """coefficients -> a weight-space move, norm-capped."""
        Bs = make_basis(bf, th)
        d = torch.einsum("bk,bkn->bn", act, Bs)
        cap = a.step_frac * th.norm(dim=1, keepdim=True)
        nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return d * torch.clamp(cap / nrm, max=1.0)

    def feats(th, bf, lt, lv):
        """low-dim state features for policy and critic"""
        Bs = make_basis(bf, th)
        proj = torch.einsum("bn,bkn->bk", bf, Bs) / th.norm(dim=1, keepdim=True)
        return torch.cat([lt.unsqueeze(1), lv.unsqueeze(1),
                          bf.norm(dim=1, keepdim=True).log1p(), proj], 1)

    NF = 3 + K + 1
    # ---- transition model over (theta, losses, move)
    T = nn.Sequential(nn.Linear(2 * NP + 2, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, a.fwidth), nn.SiLU(),
                      nn.Linear(a.fwidth, 2))
    Topt = torch.optim.AdamW(T.parameters(), lr=1e-3, weight_decay=1e-4)
    pi = nn.Sequential(nn.Linear(NF, 128), nn.Tanh(),
                       nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, K + 1))
    logstd = nn.Parameter(torch.zeros(K + 1) - 0.5)
    vf = nn.Sequential(nn.Linear(NF, 128), nn.Tanh(),
                       nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    popt = torch.optim.Adam(list(pi.parameters()) + [logstd]
                            + list(vf.parameters()), lr=3e-4)

    # ---- seed T's buffer with on-policy-ish moves from the initial policy
    buf = {"th": [], "bf": [], "dl": [], "lt": [], "lv": [], "nt": [], "nv": []}
    norm = {}

    def add_real(n):
        """EXECUTE the policy's moves for real and measure true losses."""
        idx = torch.randint(0, len(PW), (n,))
        th, bf = PW[idx], PB[idx]
        lt = losses(th, xtr, ytr, SH); lv = losses(th, xva, yva, SH)
        with torch.no_grad():
            mu = pi(feats(th, bf, lt, lv))
            act = mu + torch.randn_like(mu) * logstd.exp()
        dl = to_move(act, bf, th)
        nt = losses(th + dl, xtr, ytr, SH); nv = losses(th + dl, xva, yva, SH)
        for k_, v_ in zip(buf, [th, bf, dl, lt, lv, nt, nv]):
            buf[k_].append(v_)
        return float(lv.mean() - nv.mean())   # reduction: positive = better

    def fit_T(epochs):
        TH = torch.cat(buf["th"]); DL = torch.cat(buf["dl"])
        LT, LV = torch.cat(buf["lt"]), torch.cat(buf["lv"])
        NT, NV = torch.cat(buf["nt"]), torch.cat(buf["nv"])
        norm["dsd"] = DL.std(0).clamp_min(1e-8)
        norm["lm"], norm["ls"] = LT.mean(), LT.std().clamp_min(1e-3)
        norm["vm"], norm["vs"] = LV.mean(), LV.std().clamp_min(1e-3)
        for _ in range(epochs):
            b = torch.randint(0, len(TH), (min(256, len(TH)),))
            pt, pv = T_pred(TH[b], DL[b], LT[b], LV[b])
            loss = (((pt - NT[b]) / norm["ls"]).pow(2).mean()
                    + ((pv - NV[b]) / norm["vs"]).pow(2).mean())
            Topt.zero_grad(set_to_none=True); loss.backward(); Topt.step()

    def T_pred(th, dl, lt, lv):
        o = T(torch.cat([th, dl / norm["dsd"],
                         ((lt - norm["lm"]) / norm["ls"]).unsqueeze(1),
                         ((lv - norm["vm"]) / norm["vs"]).unsqueeze(1)], 1))
        return o[:, 0] * norm["ls"] + norm["lm"], o[:, 1] * norm["vs"] + norm["vm"]

    norm["dsd"] = torch.ones(NP); norm["lm"] = torch.tensor(0.)
    norm["ls"] = torch.tensor(1.); norm["vm"] = torch.tensor(0.)
    norm["vs"] = torch.tensor(1.)
    add_real(a.anchor_n * 4)
    fit_T(1500)
    print(f"T seeded on {len(torch.cat(buf['th']))} real transitions "
          f"({time.time()-t0:.0f}s)\n")

    print(f"  {'iter':>5} {'imagined dV':>12} {'REAL dV':>9} {'T MAE':>7} "
          f"{'real val':>9} {'TEST':>8} {'nbuf':>6}")
    for it in range(a.iters):
        # ---------- PPO entirely inside T
        idx = torch.randint(0, len(PW), (a.rollouts,))
        th, bf = PW[idx], PB[idx]
        lt = losses(th, xtr, ytr, SH); lv = losses(th, xva, yva, SH)
        S, A, LP, R, V = [], [], [], [], []
        cth, clt, clv = th, lt, lv
        for hp in range(a.horizon):
            f = feats(cth, bf, clt, clv)
            with torch.no_grad():
                mu = pi(f); sd = logstd.exp()
                act = mu + torch.randn_like(mu) * sd
                lp = (-0.5 * ((act - mu) / sd).pow(2) - logstd
                      - 0.9189385).sum(1)
                v = vf(f).squeeze(1)
                dl = to_move(act, bf, cth)
                nt, nv = T_pred(cth, dl, clt, clv)
            S.append(f); A.append(act); LP.append(lp); V.append(v)
            R.append(clv - nv)                     # reward: val-loss reduction
            cth, clt, clv = cth + dl, nt, nv
        S = torch.stack(S); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.stack(R); V = torch.stack(V)
        with torch.no_grad():
            lastv = vf(feats(cth, bf, clt, clv)).squeeze(1)
        adv = torch.zeros_like(R); run = torch.zeros(a.rollouts)
        for t in reversed(range(a.horizon)):
            nxt = lastv if t == a.horizon - 1 else V[t + 1]
            delta = R[t] + 0.99 * nxt - V[t]
            run = delta + 0.99 * 0.95 * run
            adv[t] = run
        ret = adv + V
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-8)
        Sf, Af, LPf = S.reshape(-1, NF), A.reshape(-1, K + 1), LP.reshape(-1)
        advf, retf = adv.reshape(-1), ret.reshape(-1)
        for _ in range(6):
            mu = pi(Sf); sd = logstd.exp()
            lp = (-0.5 * ((Af - mu) / sd).pow(2) - logstd - 0.9189385).sum(1)
            ratio = (lp - LPf).exp()
            pl = -torch.min(ratio * advf,
                            ratio.clamp(0.8, 1.2) * advf).mean()
            vl = (vf(Sf).squeeze(1) - retf).pow(2).mean()
            popt.zero_grad(set_to_none=True)
            (pl + 0.5 * vl - 0.001 * logstd.sum()).backward(); popt.step()
        imagined = float(R.sum(0).mean())

        # ---------- anchor in reality
        real_dv, mae = float("nan"), float("nan")
        if it % a.anchor_every == 0:
            nb = len(torch.cat(buf["th"]))
            real_dv = add_real(a.anchor_n)
            th2 = torch.cat(buf["th"])[nb:]; dl2 = torch.cat(buf["dl"])[nb:]
            lt2 = torch.cat(buf["lt"])[nb:]; lv2 = torch.cat(buf["lv"])[nb:]
            nv2 = torch.cat(buf["nv"])[nb:]
            with torch.no_grad():
                _, pv2 = T_pred(th2, dl2, lt2, lv2)
            mae = float((pv2 - nv2).abs().mean())
            fit_T(a.t_epochs)

        # ---------- honest evaluation: apply the policy for real, score on TEST
        if it % 4 == 0 or it == a.iters - 1:
            ei = torch.arange(0, len(PW), max(1, len(PW) // 24))
            eth, ebf = PW[ei], PB[ei]
            elt = losses(eth, xtr, ytr, SH); elv = losses(eth, xva, yva, SH)
            cth2, c_lt, c_lv = eth, elt, elv
            for _ in range(a.horizon):
                with torch.no_grad():
                    act = pi(feats(cth2, ebf, c_lt, c_lv))
                    dl = to_move(act, ebf, cth2)
                cth2 = cth2 + dl
                c_lt = losses(cth2, xtr, ytr, SH)
                c_lv = losses(cth2, xva, yva, SH)
            rv = float(c_lv.mean()); tv = float(losses(cth2, xts, yts, SH).mean())
            base_v = float(elv.mean())
            base_t = float(losses(eth, xts, yts, SH).mean())
            # SGD baseline: real training from the same states, matched budget
            sg_v, sg_t = [], []
            for q in range(len(eth)):
                Cq, _, _ = rollout_real((xtr, ytr, xva, yva), eth[q], 0.07,
                                        a.horizon * 20, a.horizon * 20, 64, SH,
                                        seed=int(1000 + q), buf0=ebf[q])
                sg_v.append(float(losses(Cq[-1:], xva, yva, SH)))
                sg_t.append(float(losses(Cq[-1:], xts, yts, SH)))
            print(f"  {it:>5} {imagined:>12.4f} {real_dv:>9.4f} {mae:>7.4f} "
                  f"{rv:>9.4f} {tv:>8.4f} "
                  f"{len(torch.cat(buf['th'])):>6}")
            print(f"        start:  val {base_v:.4f} test {base_t:.4f}   |  "
                  f"policy: val {rv:.4f} test {tv:.4f}   |  "
                  f"SGD: val {st.mean(sg_v):.4f} test {st.mean(sg_t):.4f}")
        else:
            print(f"  {it:>5} {imagined:>12.4f} {real_dv:>9.4f} {mae:>7.4f}")

    print(f"\n  'imagined dV' is the val-loss reduction the policy believes it")
    print(f"  achieves inside T. 'REAL dV' is what its moves actually do. A gap")
    print(f"  between them is model exploitation. TEST is the only unbiased")
    print(f"  number -- the reward uses val, so a val gain is partly circular.")
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
