"""Phase 1: train the world model and hold it to the multi-step bar.

    python3 -m racewm.train_wm --corpus data/bc --epochs 20 --wandb

What this reports, and why:

  one-step NMSE       easy here -- persistence is already close. Reported for
                      completeness, not as evidence of anything.
  rollout error @K    the real metric. Closed-loop, K steps, geometry gathered
                      from the model's own predicted pose.
  vs persistence      the floor. A model that does not beat s_{t+1}=s_t at
                      long horizons has not learned dynamics.
  vs mean-delta       beats persistence wherever motion is consistent; shows
                      how much apparent skill is just the average step.

Throughput, samples consumed and wall clock are logged next to the losses, so
"how long and how much data" is answerable without rerunning anything.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, model as M, shards, tracking


def evaluate_rollouts(predictors, corpus_dir, split, tracks, header, device,
                      horizons=(1, 5, 20, 100), n_seqs=512, seed=0):
    """Closed-loop rollout error per predictor per horizon, on `split`.

    Always call this with split="holdout": those tracks were generated over a
    disjoint env_idx range and the model has never seen their geometry.
    """
    H = max(horizons)
    obs, acts, env_idx = shards.load_sequences(
        f"{corpus_dir}/{split}_expert.*.bin", length=H + 1)
    rng = np.random.default_rng(seed)
    if len(obs) > n_seqs:
        keep = rng.choice(len(obs), n_seqs, replace=False)
        obs, acts, env_idx = obs[keep], acts[keep], env_idx[keep]

    s0 = torch.from_numpy(obs[:, 0]).to(device)
    true = torch.from_numpy(obs[:, 1:H + 1]).to(device)
    actions = torch.from_numpy(acts[:, :H]).to(device)
    dim_sd = true.reshape(-1, contract.OBS_SIZE).std(0).clamp_min(1e-6)

    out = {}
    for p in predictors:
        pred = M.rollout(p, s0, actions, tracks, env_idx, header, H)
        curve = M.rollout_error(pred, true, dim_sd)
        out[p.name] = {k: float(curve[k - 1]) for k in horizons}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--no-geometry", action="store_true",
                    help="ablation: drop the local track geometry")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    use_geom = not args.no_geometry
    run_name = args.run_name or f"wm_{'geom' if use_geom else 'nogeom'}_h{args.hidden}x{args.layers}"

    man = dataset.load_manifest(args.corpus)
    seen = dataset.assert_disjoint(args.corpus)   # refuses to run on a leaky corpus
    print(f"corpus {args.corpus}  seed={man['seed']}  frameskip={man['frameskip']}")
    for k in ("train", "val", "holdout"):
        v = man["splits"][k]
        print(f"  {k:<8} env_idx {v['track_start']:>3}.."
              f"{v['track_start']+v['n_tracks']-1:<3} ({v['n_tracks']} tracks)")
    print(f"  splits disjoint: OK")

    train = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                               with_geometry=True, max_rows=args.max_rows)
    val = dataset.load_split(args.corpus, "val", slices=dataset.WM_SLICES,
                             with_geometry=True)
    print(f"  train {len(train):,}  slices={train.counts()}")
    print(f"  val   {len(val):,}  (model selection only)")
    print(f"  holdout is untouched until the final rollout report")

    x_tr = torch.from_numpy(dataset.wm_inputs(train, use_geom))
    y_tr = torch.from_numpy(dataset.wm_targets(train))
    x_va = torch.from_numpy(dataset.wm_inputs(val, use_geom))
    y_va = torch.from_numpy(dataset.wm_targets(val))

    x_mu, x_sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    y_mu, y_sd = y_tr.mean(0, keepdim=True), y_tr.std(0, keepdim=True).clamp_min(1e-8)

    x_trn = ((x_tr - x_mu) / x_sd).to(device)
    y_trn = ((y_tr - y_mu) / y_sd).to(device)
    x_van = ((x_va - x_mu) / x_sd).to(device)
    y_van = ((y_va - y_mu) / y_sd).to(device)

    net = M.DeltaMLP(x_tr.shape[1], contract.OBS_SIZE, args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(x_trn) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    cfg = dict(vars(args), n_train=len(train), n_val=len(val),
               corpus_manifest=man,
               d_in=int(x_tr.shape[1]), use_geometry=use_geom,
               params=sum(p.numel() for p in net.parameters()))
    cp = ckpt_mod.Checkpointer(args.checkpoint_dir, run_name, config=cfg,
                               metric_name="rollout20_wm", metric_mode="min")
    start_epoch = cp.resume(net, opt, sched) if args.resume else 0
    if start_epoch:
        print(f"resumed from step {start_epoch}")

    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase1", "world-model"])
    trk.watch_params(net)
    print(f"model: {cfg['params']:,} params, input dim {cfg['d_in']}")

    # Rollouts are scored on holdout tracks, so the geometry comes from the
    # holdout track file -- never the one the model trained on.
    tracks_geo = shards.load_tracks(f"{args.corpus}/holdout.tracks.bin")
    header = train.header
    g = torch.Generator().manual_seed(args.seed)
    n = len(x_trn)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.mse_loss(net(x_trn[idx]), y_trn[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())

        net.eval()
        with torch.no_grad():
            val_nmse = float(((net(x_van) - y_van) ** 2).mean())
        payload = trk.log(epoch + 1, samples=n,
                          train_loss=tot / steps_per_epoch,
                          val_nmse=val_nmse,
                          lr=sched.get_last_lr()[0])
        print(f"  epoch {epoch+1:3d}/{args.epochs}  " + tracking.format_row(payload))
        cp.save(step=epoch + 1, model=net, optimizer=opt, scheduler=sched,
                metrics={"val_nmse": val_nmse})

    train_seconds = time.time() - t0

    # ---------------------------------------------------------- final report
    print("\nrolling out on HOLDOUT tracks "
          f"(env_idx {man['splits']['holdout']['track_start']}.."
          f"{man['splits']['holdout']['track_start']+man['splits']['holdout']['n_tracks']-1}), "
          "closed loop, geometry gathered from the model's own predicted pose...")
    wm = M.ModelWrapper(net, x_mu.to(device), x_sd.to(device),
                        y_mu.to(device), y_sd.to(device), use_geom)
    mean_delta = M.MeanDelta(y_tr.mean(0, keepdim=True).to(device))
    res = evaluate_rollouts([M.Persistence(), mean_delta, wm],
                            args.corpus, "holdout", tracks_geo, header, device)

    horizons = sorted(next(iter(res.values())).keys())
    print(f"\n{'predictor':<16}" + "".join(f"{'@'+str(k):>12}" for k in horizons))
    for name, curve in res.items():
        print(f"{name:<16}" + "".join(f"{curve[k]:>12.5f}" for k in horizons))

    summary = {"train_seconds": train_seconds,
               "samples_total": args.epochs * n,
               "final_val_nmse": val_nmse}
    for name, curve in res.items():
        for k, v in curve.items():
            summary[f"rollout{k}_{'wm' if name=='world_model' else name}"] = v

    base = res["persistence"]
    print()
    verdict = True
    for k in horizons:
        better = res["world_model"][k] < base[k]
        ratio = base[k] / max(res["world_model"][k], 1e-12)
        print(f"  @{k:<4} vs persistence: {ratio:6.2f}x {'better' if better else 'WORSE'}")
        if k >= 5 and not better:
            verdict = False
    summary["beats_persistence"] = verdict
    trk.summary(**summary)
    cp.save(step=args.epochs, model=net, optimizer=opt, scheduler=sched,
            metrics={"val_nmse": val_nmse,
                     "rollout20_wm": res["world_model"][20]})
    trk.finish()

    print(f"\ntrained {args.epochs} epochs on {len(train):,} transitions "
          f"in {train_seconds:.1f}s ({args.epochs*n/train_seconds:,.0f} samples/s)")
    print(f"phase 1 gate: {'PASS' if verdict else 'FAIL'} "
          f"(world model must beat persistence at horizon >= 5)")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
