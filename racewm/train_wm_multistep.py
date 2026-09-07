"""World model trained on its own rollouts, not on single steps.

The one-step model is excellent to ~20 steps and diverges by 100 (measured:
9.3x better than persistence at 20, 0.88x at 100). That is the standard
compounding-error signature and it is a train/test mismatch, not a capacity
problem: the model is only ever asked to predict from a REAL state, then at
evaluation is asked to predict from its own output, which is a distribution it
never saw.

Fix: unroll K steps during training and take the loss at every step, with the
geometry at step k gathered from the model's own predicted pose at step k. The
model sees its own error distribution while it can still learn from it.

Two details that matter:

  Discounting. Later steps carry more error and would otherwise dominate the
  gradient and destabilize early-step accuracy. Losses are weighted by
  `gamma**k`, so near-term accuracy stays anchored.

  Projection in the loop. project_state renormalizes the sin/cos pairs each
  step. Without it, unrolling drifts off the state manifold and the model
  spends capacity learning to undo its own invalid encodings.

    python3 -m racewm.train_wm_multistep --corpus data/corpus --rollout-k 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, geometry, model as M, shards, tracking

WM_SLICE_FILES = ("expert", "whisker", "noisy", "random")


def load_sequences_multi(corpus_dir, split, length, slices=WM_SLICE_FILES,
                         max_seqs=None, seed=0):
    obs, act, env = [], [], []
    for name in slices:
        try:
            o, a, e = shards.load_sequences(f"{corpus_dir}/{split}_{name}.*.bin", length)
        except (FileNotFoundError, ValueError):
            continue
        obs.append(o); act.append(a); env.append(e)
    if not obs:
        raise FileNotFoundError(f"no sequences for {split}")
    obs = np.concatenate(obs); act = np.concatenate(act); env = np.concatenate(env)
    if max_seqs and len(obs) > max_seqs:
        keep = np.random.default_rng(seed).choice(len(obs), max_seqs, replace=False)
        obs, act, env = obs[keep], act[keep], env[keep]
    return obs, act, env


def unroll(net, s0, actions, bank, slots, x_mu, x_sd, y_mu, y_sd, K):
    """Differentiable K-step rollout. Returns (B, K, OBS_SIZE)."""
    s = s0
    outs = []
    n = len(s0)
    ar = torch.arange(n, device=s0.device)
    for k in range(K):
        a = torch.zeros(n, contract.NUM_ACTIONS, device=s0.device)
        a[ar, actions[:, k]] = 1.0
        g = bank.local_window(s, slots)
        x = torch.cat([s, a, g], dim=1)
        delta = net((x - x_mu) / x_sd) * y_sd + y_mu
        s = M.project_state(s + delta)
        outs.append(s)
    return torch.stack(outs, dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--rollout-k", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="per-step loss discount; keeps late steps from dominating")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-seqs", type=int, default=120_000)
    ap.add_argument("--init-from", default=None,
                    help="warm start from a one-step checkpoint (much faster)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    K = args.rollout_k
    run_name = args.run_name or f"wm_multistep_k{K}"

    dataset.assert_disjoint(args.corpus)
    man = dataset.load_manifest(args.corpus)

    # Normalization comes from the one-step transition set: it must describe
    # the same quantities the model consumed during one-step training, so a
    # warm start stays valid.
    flat = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, max_rows=300_000)
    x_flat = torch.from_numpy(dataset.wm_inputs(flat))
    y_flat = torch.from_numpy(dataset.wm_targets(flat))
    x_mu = x_flat.mean(0, keepdim=True).to(dev)
    x_sd = x_flat.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    y_mu = y_flat.mean(0, keepdim=True).to(dev)
    y_sd = y_flat.std(0, keepdim=True).clamp_min(1e-8).to(dev)
    del x_flat, y_flat

    tr_obs, tr_act, tr_env = load_sequences_multi(args.corpus, "train", K + 1,
                                                  max_seqs=args.max_seqs, seed=args.seed)
    va_obs, va_act, va_env = load_sequences_multi(args.corpus, "val", K + 1,
                                                  max_seqs=20_000, seed=args.seed)
    print(f"multi-step K={K}  train seqs {len(tr_obs):,}  val seqs {len(va_obs):,}")

    tracks_tr = shards.load_tracks(f"{args.corpus}/train.tracks.bin")
    tracks_va = shards.load_tracks(f"{args.corpus}/val.tracks.bin")
    bank_tr = geometry.TrackBank(tracks_tr, flat.header, dev)
    bank_va = geometry.TrackBank(tracks_va, flat.header, dev)
    slots_tr = bank_tr.slots(tr_env)
    slots_va = bank_va.slots(va_env)

    T_obs = torch.from_numpy(tr_obs).to(dev)
    T_act = torch.from_numpy(tr_act).to(dev)
    V_obs = torch.from_numpy(va_obs).to(dev)
    V_act = torch.from_numpy(va_act).to(dev)

    d_in = contract.OBS_SIZE + contract.NUM_ACTIONS + geometry.GEOM_DIM
    net = M.DeltaMLP(d_in, contract.OBS_SIZE, args.hidden, args.layers).to(dev)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        print(f"warm started from {args.init_from} (step {ck.get('step')})")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(T_obs) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    # Per-step discount, normalized so the loss scale is comparable across K.
    w = torch.tensor([args.gamma ** k for k in range(K)], device=dev)
    w = w / w.sum()
    dim_sd = T_obs.reshape(-1, contract.OBS_SIZE).std(0).clamp_min(1e-6).to(dev)

    cfg = dict(vars(args), n_train_seqs=len(T_obs), d_in=d_in,
               params=sum(p.numel() for p in net.parameters()), corpus_manifest=man)
    cp = ckpt_mod.Checkpointer("checkpoints", run_name, config=cfg,
                               metric_name="val_rollout", metric_mode="min")
    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase1", "world-model", "multistep"])
    trk.watch_params(net)

    g = torch.Generator().manual_seed(args.seed)
    n = len(T_obs)
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            pred = unroll(net, T_obs[idx, 0], T_act[idx, :K], bank_tr,
                          slots_tr[idx], x_mu, x_sd, y_mu, y_sd, K)
            err = ((pred - T_obs[idx, 1:K + 1]) / dim_sd) ** 2      # (B, K, D)
            loss = (err.mean(dim=(0, 2)) * w).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach())

        net.eval()
        with torch.no_grad():
            vp = unroll(net, V_obs[:, 0], V_act[:, :K], bank_va, slots_va,
                        x_mu, x_sd, y_mu, y_sd, K)
            verr = (((vp - V_obs[:, 1:K + 1]) / dim_sd) ** 2).mean(dim=(0, 2))
            val_roll = float((verr * w).sum())
        p = trk.log(epoch + 1, samples=n * K,
                    train_loss=tot / steps, val_rollout=val_roll,
                    val_step1=float(verr[0]), val_stepK=float(verr[-1]))
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss={tot/steps:.5f}  "
              f"val_rollout={val_roll:.5f}  @1={verr[0]:.5f}  @{K}={verr[-1]:.5f}  "
              f"({p['wall_clock_s']:.0f}s)")
        cp.save(step=epoch + 1, model=net, optimizer=opt, scheduler=sched,
                metrics={"val_rollout": val_roll})
    secs = time.time() - t0

    print("\nHOLDOUT rollout, closed loop, geometry from the model's own pose")
    from .train_wm import evaluate_rollouts
    tracks_ho = shards.load_tracks(f"{args.corpus}/holdout.tracks.bin")
    wm = M.ModelWrapper(net, x_mu, x_sd, y_mu, y_sd, True)
    res = evaluate_rollouts([M.Persistence(), M.MeanDelta(y_mu), wm],
                            args.corpus, "holdout", tracks_ho, flat.header, dev)
    hs = sorted(next(iter(res.values())).keys())
    print(f"\n{'predictor':<16}" + "".join(f"{'@'+str(k):>12}" for k in hs))
    for name, curve in res.items():
        print(f"{name:<16}" + "".join(f"{curve[k]:>12.5f}" for k in hs))
    print()
    for k in hs:
        r = res["persistence"][k] / max(res["world_model"][k], 1e-12)
        print(f"  @{k:<4} vs persistence: {r:6.2f}x {'better' if r > 1 else 'WORSE'}")

    summary = {"train_seconds": secs, "rollout_k": K}
    for name, curve in res.items():
        for k, v in curve.items():
            summary[f"rollout{k}_{'wm' if name == 'world_model' else name}"] = v
    trk.summary(**summary)
    cp.save(step=args.epochs, model=net, optimizer=opt, scheduler=sched,
            metrics={"val_rollout": val_roll, "rollout20_wm": res["world_model"][20]})
    trk.finish()
    print(f"\ntrained {args.epochs} epochs, K={K}, in {secs:.0f}s")


if __name__ == "__main__":
    main()
