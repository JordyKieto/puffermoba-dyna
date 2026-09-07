"""Phase 0 Markov gate.

The question this answers: given the pinned state, does knowing the PAST help
predict the future? If it does, the state is not Markovian and something the
dynamics depend on is missing from it.

Method: train the same MLP, same capacity, same budget, to predict s_{t+1} from
(s_t, a_t), then from (s_{t-H..t}, a_t) for H > 0. Compare held-out error. A
history model that does no better means the extra frames carried no information
the current state lacked.

The run includes a POSITIVE CONTROL: the same test on upstream's 3-float
observation [flw, frw, score]. That state is provably non-Markovian, so history
must help there by a wide margin. If it does not, the test has no power and its
verdict on the full state means nothing.

    python3 -m racewm.markov --shards 'data/probe.*.bin'
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from . import contract, shards

# Upstream whisker_racer's compute_observations, expressed in our indices.
LEGACY_DIMS = (contract.INDEX["flw_length"],
               contract.INDEX["frw_length"],
               contract.INDEX["score_norm"])


def build_windows(obs: np.ndarray, act: np.ndarray, history: int, h_max: int, dims):
    """(inputs, target_delta, s_t) for a given history depth."""
    dims = list(dims)
    frames = obs[:, h_max - history:h_max + 1, :][:, :, dims]   # (N, H+1, D)
    s_t = obs[:, h_max, dims]                                   # (N, D)
    s_next = obs[:, h_max + 1, dims]                            # (N, D)
    a = act[:, h_max]                                           # (N,)

    onehot = np.zeros((len(a), contract.NUM_ACTIONS), dtype=np.float32)
    onehot[np.arange(len(a)), a] = 1.0

    x = np.concatenate([frames.reshape(len(frames), -1), onehot], axis=1)
    return x.astype(np.float32), (s_next - s_t).astype(np.float32), s_t.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def fit_and_score(x_tr, y_tr, x_te, y_te, *, epochs=60, batch=1024, lr=1e-3, seed=0,
                  hidden=256, layers=3, device="cpu"):
    """Held-out normalized MSE. Normalized by target variance, so 1.0 == the
    predict-the-mean baseline and lower is better."""
    torch.manual_seed(seed)

    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    ymu, ysd = y_tr.mean(0, keepdims=True), y_tr.std(0, keepdims=True)

    # Drop target dims that never move in this slice. A sector_completed bit
    # that is constant across the whole sample has zero variance, and dividing
    # by it turns a perfectly-predicted constant into a huge normalized error
    # (this produced NMSE ~1e13 on single-track fits). They carry no signal to
    # score, so they are excluded rather than scaled.
    live = (ysd > 1e-6).ravel()
    if live.sum() == 0:
        return float("nan"), np.zeros(0)
    y_tr, y_te = y_tr[:, live], y_te[:, live]
    ymu, ysd = ymu[:, live], ysd[:, live] + 1e-8

    xt = torch.from_numpy((x_tr - mu) / sd).to(device)
    yt = torch.from_numpy((y_tr - ymu) / ysd).to(device)
    xv = torch.from_numpy((x_te - mu) / sd).to(device)
    yv = torch.from_numpy((y_te - ymu) / ysd).to(device)

    model = MLP(xt.shape[1], yt.shape[1], hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(xt)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(xv)
        # Normalized by the variance of the standardized target == 1, so this
        # is directly comparable across history depths and state definitions.
        per_dim = ((pred - yv) ** 2).mean(0)
        return float(per_dim.mean()), per_dim.cpu().numpy()


def _denorm(obs, hdr):
    """Recover raw px, py, ang, near_point_idx from the normalized state."""
    px = obs[:, contract.INDEX["px_norm"]] * hdr["width"]
    py = obs[:, contract.INDEX["py_norm"]] * hdr["height"]
    ang = np.arctan2(obs[:, contract.INDEX["sin_ang"]], obs[:, contract.INDEX["cos_ang"]])
    phase = np.arctan2(obs[:, contract.INDEX["sin_near_point"]],
                       obs[:, contract.INDEX["cos_near_point"]])
    p = int(hdr["total_points"])
    near = np.rint((phase % (2 * np.pi)) / (2 * np.pi) * p).astype(np.int64) % p
    return px, py, ang, near


def _track_features(kind, obs_t, env_idx, tracks, hdr, n_tracks):
    """Episode-constant context to append to the per-step state."""
    if kind == "none":
        return np.zeros((len(obs_t), 0), dtype=np.float32)
    if kind == "id":
        # One-hot track identity. Enough to make the fit Markov on a FIXED set
        # of tracks, but it is a lookup table: it cannot transfer to a track
        # the model has never seen.
        oh = np.zeros((len(obs_t), n_tracks), dtype=np.float32)
        oh[np.arange(len(obs_t)), env_idx] = 1.0
        return oh
    if kind == "local":
        # The actual geometry the dynamics read, in the car's ego frame.
        # Generalizes across tracks because it is what compute_whiskers sees.
        px, py, ang, near = _denorm(obs_t, hdr)
        out = np.zeros((len(obs_t), 44), dtype=np.float32)
        for e in np.unique(env_idx):
            m = env_idx == e
            out[m] = tracks.local_window(int(e), near[m], px[m], py[m], ang[m])
        return out / 100.0     # whisker scale; keeps inputs O(1)
    raise ValueError(kind)


def run(pattern, tracks_path=None, histories=(0, 1, 2, 4), epochs=60, seed=0,
        max_windows=200_000, device="cpu", tracks=None):
    h_max = max(histories)
    obs, act, env_idx = shards.load_sequences(pattern, length=h_max + 2)

    if tracks is not None:
        mask = np.isin(env_idx, np.asarray(tracks))
        obs, act, env_idx = obs[mask], act[mask], env_idx[mask]
        if len(obs) == 0:
            raise ValueError(f"no windows for tracks={tracks}")

    rng = np.random.default_rng(seed)
    if len(obs) > max_windows:
        keep = rng.choice(len(obs), max_windows, replace=False)
        obs, act, env_idx = obs[keep], act[keep], env_idx[keep]

    hdr = shards.read_header(shards._shard_paths(pattern)[0])
    track_geo = shards.load_tracks(tracks_path) if tracks_path else None
    uniq = np.unique(env_idx)
    remap = {int(e): i for i, e in enumerate(uniq)}
    env_slot = np.array([remap[int(e)] for e in env_idx])

    idx = rng.permutation(len(obs))
    cut = int(0.85 * len(obs))
    tr_i, te_i = idx[:cut], idx[cut:]

    arms = [
        ("contract_v1 (no track context)", range(contract.OBS_SIZE), "none"),
        ("contract_v1 + track_id",         range(contract.OBS_SIZE), "id"),
    ]
    if track_geo is not None:
        arms.append(("contract_v1 + local geometry", range(contract.OBS_SIZE), "local"))
    arms.append(("legacy upstream 3-float (control)", LEGACY_DIMS, "none"))

    results = {}
    for label, dims, ctx in arms:
        arm = {}
        for h in histories:
            x, y, s_t = build_windows(obs, act, h, h_max, dims)
            extra = _track_features(ctx, obs[:, h_max, :], env_slot, track_geo,
                                    hdr, len(uniq))
            x = np.concatenate([x, extra], axis=1)
            nmse, _ = fit_and_score(x[tr_i], y[tr_i], x[te_i], y[te_i],
                                    epochs=epochs, seed=seed, device=device)
            arm[h] = nmse
        base = arm[0]
        best_h = min((h for h in histories if h > 0), key=lambda h: arm[h])
        gain = (base - arm[best_h]) / base if base > 0 else 0.0
        results[label] = {
            "n_windows": int(len(obs)),
            "n_dims": len(list(dims)),
            "track_context": ctx,
            "by_history": {str(h): arm[h] for h in histories},
            "best_history": best_h,
            "history_gain": gain,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/probe.*.bin")
    ap.add_argument("--tracks-file", default=None,
                    help="PREFIX.tracks.bin, enabling the local-geometry arm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=200_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tracks", default=None,
                    help="comma separated env_idx values to restrict the fit to")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--control-min-gain", type=float, default=0.20)
    args = ap.parse_args()

    tracks = [int(x) for x in args.tracks.split(",")] if args.tracks else None
    res = run(args.shards, tracks_path=args.tracks_file, epochs=args.epochs,
              seed=args.seed, max_windows=args.max_windows, device=args.device,
              tracks=tracks)

    print(f"Markov gate   schema={contract.schema_hash()}  shards={args.shards}")
    print("  Does knowing the past improve prediction? If yes, the state is "
          "missing something.\n")
    for label, r in res.items():
        print(f"  {label}  ({r['n_dims']} dims + ctx={r['track_context']}, "
              f"{r['n_windows']} windows)")
        for h, v in r["by_history"].items():
            tag = "   <- no history" if h == "0" else ""
            print(f"    history={h}  held-out NMSE {v:.5f}{tag}")
        print(f"    best history={r['best_history']}   gain: "
              f"{r['history_gain']*100:+.2f}%\n")

    control = res["legacy upstream 3-float (control)"]["history_gain"]
    powered = control >= args.control_min_gain

    # The state is Markov GIVEN the track. Prefer the geometry arm when we have
    # it, since track_id only works on a fixed track set.
    if "contract_v1 + local geometry" in res:
        key = "contract_v1 + local geometry"
    else:
        key = "contract_v1 + track_id"
    gain = res[key]["history_gain"]
    passed = gain < args.tolerance

    print("  verdict")
    print(f"    control gain {control*100:+.2f}% (needs >= {args.control_min_gain*100:.0f}%)"
          f"  -> test {'HAS' if powered else 'has NO'} power")
    print(f"    {key}: gain {gain*100:+.2f}% (needs < {args.tolerance*100:.0f}%)"
          f"  -> {'MARKOV' if passed else 'NOT MARKOV'}")
    no_ctx = res["contract_v1 (no track context)"]["history_gain"]
    print(f"    for reference, without track context: {no_ctx*100:+.2f}% "
          f"-- the track is part of the state")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"schema_hash": contract.schema_hash(), "results": res,
                       "passed": bool(passed and powered)}, f, indent=2)
        print(f"\n  wrote {args.json}")

    ok = passed and powered
    print(f"\nmarkov gate: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
