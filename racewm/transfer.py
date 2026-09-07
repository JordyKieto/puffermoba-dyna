"""Does a dynamics-pretrained backbone help imitation learning?

The question: train a world model, then initialise a BC policy from its
encoder. Does that beat training BC from scratch -- and does it depend on how
long a horizon the world model was trained over?

This needs a **separable encoder**. The world model consumes 77 inputs (state,
action, local geometry) while a policy consumes 30, so a monolithic MLP cannot
transfer its first layer. Splitting it as

    encoder(obs)              -> z          shared, transferable
    dynamics([z, action, geom]) -> delta     world-model only
    policy(z)                 -> 3 actions   BC only

is also the structure that would carry a vision encoder into a policy, which
is where this is going.

Representation transfer usually pays off in the LOW-DATA regime, so the sweep
varies the BC label budget. World-model data is effectively free here (the env
runs at ~1M steps/s and needs no action labels), which is precisely the
asymmetry that would make pretraining worth it.

    python3 -m racewm.transfer --stage wm      # pretrain encoders at each horizon
    python3 -m racewm.transfer --stage bc      # the transfer sweep
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract, dataset, export_policy, geometry, model as M, shards, tracking

HORIZONS = (1, 5, 20)
BC_BUDGETS = (2_000, 10_000, 50_000, 450_000)


class Encoder(nn.Module):
    def __init__(self, d_in=contract.OBS_SIZE, hidden=256, layers=2):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.net = nn.Sequential(*mods)
        self.d_out = hidden

    def forward(self, x):
        return self.net(x)


class EncoderDynamics(nn.Module):
    """encoder(obs) -> z ; dynamics([z, action, geometry]) -> delta."""

    def __init__(self, hidden=256, enc_layers=2, dyn_layers=2,
                 d_extra=contract.NUM_ACTIONS + geometry.GEOM_DIM):
        super().__init__()
        self.encoder = Encoder(contract.OBS_SIZE, hidden, enc_layers)
        mods, d = [], hidden + d_extra
        for _ in range(dyn_layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, contract.OBS_SIZE))
        self.dynamics = nn.Sequential(*mods)

    def forward(self, obs, extra):
        return self.dynamics(torch.cat([self.encoder(obs), extra], dim=1))


class EncoderPolicy(nn.Module):
    """encoder(obs) -> z ; policy(z) -> logits. Module order keeps the
    Linear sequence exportable to the .rpo format race_eval reads."""

    def __init__(self, hidden=256, enc_layers=2, head_layers=1,
                 n_act=contract.NUM_ACTIONS):
        super().__init__()
        self.encoder = Encoder(contract.OBS_SIZE, hidden, enc_layers)
        mods, d = [], hidden
        for _ in range(head_layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, n_act))
        self.head = nn.Sequential(*mods)

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------- stage: WM
def train_wm(corpus, horizon, epochs, hidden, out_dir, device="cpu", seed=0,
             max_seqs=60_000):
    """Multi-step dynamics training with a separable encoder."""
    from .train_wm_multistep import load_sequences_multi
    torch.manual_seed(seed)
    K = horizon

    flat = dataset.load_split(corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, max_rows=200_000)
    x = torch.from_numpy(dataset.wm_inputs(flat))
    y = torch.from_numpy(dataset.wm_targets(flat))
    obs_mu = torch.from_numpy(flat.obs).mean(0, keepdim=True).to(device)
    obs_sd = torch.from_numpy(flat.obs).std(0, keepdim=True).clamp_min(1e-6).to(device)
    ex_mu = x[:, contract.OBS_SIZE:].mean(0, keepdim=True).to(device)
    ex_sd = x[:, contract.OBS_SIZE:].std(0, keepdim=True).clamp_min(1e-6).to(device)
    y_mu = y.mean(0, keepdim=True).to(device)
    y_sd = y.std(0, keepdim=True).clamp_min(1e-8).to(device)

    obs, act, env = load_sequences_multi(corpus, "train", K + 1,
                                         max_seqs=max_seqs, seed=seed)
    bank = geometry.TrackBank(shards.load_tracks(f"{corpus}/train.tracks.bin"),
                              flat.header, device)
    slots = bank.slots(env)
    O = torch.from_numpy(obs).to(device)
    A_ = torch.from_numpy(act).to(device)
    dim_sd = O.reshape(-1, contract.OBS_SIZE).std(0).clamp_min(1e-6)

    net = EncoderDynamics(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 512
    steps = max(1, len(O) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    w = torch.tensor([0.9 ** k for k in range(K)], device=device)
    w = w / w.sum()

    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(O), generator=g).to(device)
        tot = 0.0
        for i in range(0, len(O) - batch + 1, batch):
            idx = perm[i:i + batch]
            s = O[idx, 0]
            sl = slots[idx]
            loss = 0.0
            for k in range(K):
                a = torch.zeros(len(s), contract.NUM_ACTIONS, device=device)
                a[torch.arange(len(s), device=device), A_[idx, k]] = 1.0
                gm = bank.local_window(s, sl)
                extra = (torch.cat([a, gm], 1) - ex_mu) / ex_sd
                delta = net((s - obs_mu) / obs_sd, extra) * y_sd + y_mu
                s = M.project_state(s + delta)
                loss = loss + w[k] * (((s - O[idx, k + 1]) / dim_sd) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach())
        print(f"    K={K} epoch {ep+1}/{epochs} loss={tot/steps:.5f}", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": net.encoder.state_dict(), "hidden": hidden,
                "horizon": K, "obs_mu": obs_mu.cpu(), "obs_sd": obs_sd.cpu(),
                "schema_hash": contract.schema_hash()}, out / "encoder.pt")
    return out / "encoder.pt"


# ---------------------------------------------------------------- stage: BC
def train_bc_transfer(corpus, budget, encoder_path, freeze, epochs, hidden,
                      out_rpo, device="cpu", seed=0):
    torch.manual_seed(seed)
    tr = dataset.load_split(corpus, "train", slices=("expert",), with_geometry=False,
                            max_rows=budget, seed=seed)
    va = dataset.load_split(corpus, "val", slices=("expert",), with_geometry=False)

    x_tr = torch.from_numpy(tr.obs); y_tr = torch.from_numpy(tr.action)
    x_va = torch.from_numpy(va.obs); y_va = torch.from_numpy(va.action)

    net = EncoderPolicy(hidden=hidden).to(device)
    if encoder_path:
        ck = torch.load(encoder_path, map_location=device, weights_only=False)
        assert ck["schema_hash"] == contract.schema_hash(), "contract drift"
        net.encoder.load_state_dict(ck["encoder"])
        # Reuse the pretrainer's normalization so the encoder sees the inputs
        # it was fitted on.
        mu, sd = ck["obs_mu"].to(device), ck["obs_sd"].to(device)
        if freeze:
            for q in net.encoder.parameters():
                q.requires_grad_(False)
    else:
        mu = x_tr.mean(0, keepdim=True); sd = x_tr.std(0, keepdim=True).clamp_min(1e-6)

    xn, xvn = (x_tr - mu.cpu()) / sd.cpu(), (x_va - mu.cpu()) / sd.cpu()
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    batch = min(4096, max(64, len(xn) // 4))
    g = torch.Generator().manual_seed(seed)

    t0 = time.time()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xn), generator=g)
        for i in range(0, max(1, len(xn) - batch + 1), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.cross_entropy(net(xn[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(xvn).argmax(1) == y_va).float().mean())
    export_policy.export(net, mu.cpu().numpy(), sd.cpu().numpy(), out_rpo)
    return acc, time.time() - t0


def online(rpo, episodes=100):
    r = subprocess.run(["./build/race_eval", "--bots", "policy", "--policy", str(rpo),
                        "--tracks", "16", "--track-start", "112",
                        "--episodes", str(episodes), "--threads", "10"],
                       capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if l.strip().startswith("policy"):
            f = l.split()
            return float(f[2]), float(f[4])   # return, survive%
    return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["wm", "bc", "all"], default="all")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wm-epochs", type=int, default=6)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="data/transfer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path("checkpoints/transfer")
    if args.stage in ("wm", "all"):
        for K in HORIZONS:
            print(f"  pretraining dynamics encoder, horizon K={K}", flush=True)
            train_wm(args.corpus, K, args.wm_epochs, args.hidden, root / f"wm_K{K}",
                     seed=args.seed)

    if args.stage in ("bc", "all"):
        arms = [("scratch", None, False)]
        for K in HORIZONS:
            arms.append((f"K{K}_finetune", root / f"wm_K{K}" / "encoder.pt", False))
        arms.append((f"K{max(HORIZONS)}_frozen",
                     root / f"wm_{max(HORIZONS)}" / "encoder.pt", True))
        arms[-1] = (f"K{max(HORIZONS)}_frozen",
                    root / f"wm_K{max(HORIZONS)}" / "encoder.pt", True)

        results = {}
        print(f"\n{'arm':<16}{'labels':>9}{'val_acc':>9}{'return':>9}{'surv%':>8}{'train_s':>9}")
        for name, enc, freeze in arms:
            for b in BC_BUDGETS:
                rpo = Path(f"/tmp/transfer_{name}_{b}.rpo")
                acc, secs = train_bc_transfer(args.corpus, b, enc, freeze,
                                              args.bc_epochs, args.hidden, rpo,
                                              seed=args.seed)
                ret, surv = online(rpo, args.episodes)
                results.setdefault(name, {})[b] = dict(val_acc=acc, ret=ret, surv=surv)
                print(f"{name:<16}{b:>9,}{acc:>9.4f}{ret:>9.3f}{surv:>8.1f}{secs:>8.1f}s",
                      flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
