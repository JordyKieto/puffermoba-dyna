"""Calibrate the termination head.

The head is trained with a large `pos_weight` because terminations are ~1.5% of
rows; without it the head would predict "never ends" and score 98.5% accuracy
while being useless. That weighting deliberately biases it toward predicting
termination, and the bias is large: on holdout it predicts 2,095 terminations
where 688 occurred, a 3.05x over-prediction.

This matters specifically for training a policy inside the model. Imagined
episodes end about three times too early, so the policy sees a truncated
horizon and learns to be over-cautious -- which is what the Dreamer runs
looked like. A well-ranked but miscalibrated probability is fine for detection
and wrong for simulation.

Temperature + bias scaling on the logit, fitted on the validation split by
matching the expected number of terminations. Ranking (and therefore AP) is
untouched; only the scale moves.

    python3 -m racewm.calibrate --head checkpoints/termination/best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import contract, dataset
from .train_termination import TerminationHead


def logits_for(head, mu, sd, corpus, split):
    c = dataset.load_split(corpus, split, slices=dataset.WM_SLICES,
                           with_geometry=True, include_terminal=True)
    x = torch.from_numpy(dataset.wm_inputs(c))
    with torch.no_grad():
        z = head((x - mu) / sd)
    return z, torch.from_numpy(c.done.astype(np.float32))


def report(z, y, T, b, label):
    p = torch.sigmoid(z / T + b)
    exp, act = float(p.sum()), float(y.sum())
    return (f"  {label:<12} predicted {exp:7.0f} vs actual {act:5.0f} "
            f"(ratio {exp/max(act,1):5.2f})   "
            f"P|survivor {float(p[y==0].mean()):.4f}  P|terminal {float(p[y==1].mean()):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", default="checkpoints/termination/best.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="checkpoints/termination/calibration.pt")
    args = ap.parse_args()

    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    head = TerminationHead(cfg["d_in"], cfg["hidden"], cfg["layers"])
    head.load_state_dict(ck["model"]); head.eval()
    ex = ck.get("extra") or {}
    mu, sd = ex["mu"], ex["sd"]

    zv, yv = logits_for(head, mu, sd, args.corpus, "val")
    zh, yh = logits_for(head, mu, sd, args.corpus, "holdout")

    print("before calibration")
    print(report(zv, yv, 1.0, 0.0, "val"))
    print(report(zh, yh, 1.0, 0.0, "holdout"))

    # Fit T and b on VAL only, by minimizing NLL. Fitting on holdout would be
    # tuning on the test set.
    T = torch.nn.Parameter(torch.ones(1))
    b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([T, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            zv / T.clamp_min(1e-2) + b, yv)
        loss.backward()
        return loss
    opt.step(closure)
    T_, b_ = float(T.clamp_min(1e-2)), float(b)

    print(f"\nfitted on val: temperature {T_:.4f}  bias {b_:+.4f}")
    print("after calibration")
    print(report(zv, yv, T_, b_, "val"))
    print(report(zh, yh, T_, b_, "holdout"))

    # Ranking must be unchanged: calibration rescales, it does not reorder.
    def ap(z, y):
        o = torch.argsort(z, descending=True)
        h = y[o]; c = torch.cumsum(h, 0)
        prec = c / torch.arange(1, len(h) + 1)
        return float((prec * h).sum() / h.sum().clamp_min(1))
    print(f"\n  holdout AP before {ap(zh, yh):.4f}  after {ap(zh / T_ + b_, yh):.4f}  "
          f"(monotone rescaling cannot change ranking)")

    # NLL calibration fixes the AGGREGATE termination count. It does not fix
    # the survival DISTRIBUTION, which is what a policy trained inside the
    # model actually experiences -- see racewm/survival.py. An analytic
    # estimate from the mean hazard is useless here because the corpus mixes
    # policies with wildly different crash rates; it has to be measured by
    # rolling a policy in the dream.
    torch.save({"temperature": T_, "bias": b_, "head": args.head,
                "schema_hash": contract.schema_hash()}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
