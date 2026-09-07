"""Do world-model objectives help behaviour cloning when trained ALONGSIDE it?

Different mechanism from pretrain-then-finetune (E11/E12). There the
representation is fixed before the policy starts, which is why it became a
liability once labels were plentiful — a frozen prior that stops paying rent.
Here the auxiliary loss shapes the encoder *while* the policy trains, so it
behaves as a regularizer that can be outvoted by the task loss as data grows.

Auxiliary heads, all reading `z = encoder(obs)`:

    next_state    [z, a]        -> Δs_{t+1}              observation space
    next_latent   [z, a]        -> sg(encoder(s_{t+1}))  latent space
    multi_state   [z, a_{t..}]  -> Δs over K steps
    multi_latent  [z, a_{t..}]  -> sg(encoder(s_{t+k}))
    geom          z             -> local track geometry  (E12's winner)

**Latent prediction needs a stop-gradient on the target.** Predicting
`encoder(s_{t+1})` with a live encoder has a trivial solution — make z
constant and the loss is zero — which is representation collapse, not
learning. BYOL and JEPA avoid it with a stop-gradient (and usually an EMA
target). We use the stop-gradient and report the latent variance, so a collapse
shows up as a number rather than as a quietly useless run.

Same controls as E12: identical architecture, a fixed 2,000 gradient steps for
every arm, same seeds, same evaluation. The auxiliary loss uses the SAME batch
as the policy loss, so no arm sees more data than another — this isolates the
objective from the data.

    python3 -m racewm.bc_aux --aux next_state --budget 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract, dataset, export_policy, geometry, shards
from .transfer import Encoder, online

AUX = ("none", "next_state", "next_latent", "multi_state", "multi_latent",
       "geom", "state_latent")


class AuxPolicy(nn.Module):
    def __init__(self, aux="none", hidden=256, enc_layers=2, K=5):
        super().__init__()
        self.encoder = Encoder(contract.OBS_SIZE, hidden, enc_layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, contract.NUM_ACTIONS))
        self.aux, self.K, self.hidden = aux, K, hidden
        A = contract.NUM_ACTIONS

        def mlp(d_in, d_out):
            return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d_out))

        if aux in ("next_state", "state_latent"):
            self.h_state = mlp(hidden + A, contract.OBS_SIZE)
        if aux in ("next_latent", "state_latent"):
            self.h_latent = mlp(hidden + A, hidden)
        if aux == "multi_state":
            self.h_state = mlp(hidden + A * K, contract.OBS_SIZE * K)
        if aux == "multi_latent":
            self.h_latent = mlp(hidden + A * K, hidden * K)
        if aux == "geom":
            self.h_geom = mlp(hidden, geometry.GEOM_DIM)

    def forward(self, x):
        return self.head(self.encoder(x))


def run(corpus, aux, budget, steps=2000, lam=1.0, hidden=256, K=5, seed=0,
        episodes=100, out_rpo=None):
    torch.manual_seed(seed)
    L = K + 1
    obs, act, env = shards.load_sequences(f"{corpus}/train_expert.*.bin", length=L)
    rng = np.random.default_rng(seed)
    if len(obs) > budget:
        keep = rng.choice(len(obs), budget, replace=False)
        obs, act, env = obs[keep], act[keep], env[keep]

    O = torch.from_numpy(obs)                    # (N, L, D)
    Aidx = torch.from_numpy(act)                 # (N, L)
    mu = O[:, 0].mean(0, keepdim=True)
    sd = O[:, 0].std(0, keepdim=True).clamp_min(1e-6)
    On = (O - mu.unsqueeze(0)) / sd.unsqueeze(0)
    y = Aidx[:, 0]

    gm = None
    if aux == "geom":
        hdr = shards.read_header(shards._shard_paths(f"{corpus}/train_expert.*.bin")[0])
        bank = geometry.TrackBank(shards.load_tracks(f"{corpus}/train.tracks.bin"), hdr)
        with torch.no_grad():
            gm = bank.local_window(O[:, 0], bank.slots(env))
        gm = (gm - gm.mean(0, keepdim=True)) / gm.std(0, keepdim=True).clamp_min(1e-6)

    dn = (O[:, 1:] - O[:, :-1])
    dmu, dsd = dn.mean((0, 1), keepdim=True), dn.std((0, 1), keepdim=True).clamp_min(1e-8)

    net = AuxPolicy(aux, hidden, 2, K)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    batch = min(1024, len(On))
    n = len(On)
    zvar = float("nan")

    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        z = net.encoder(On[idx, 0])
        loss = nn.functional.cross_entropy(net.head(z), y[idx])

        a1 = torch.zeros(batch, contract.NUM_ACTIONS)
        a1[torch.arange(batch), Aidx[idx, 0]] = 1.0
        aK = torch.zeros(batch, contract.NUM_ACTIONS * K)
        for k in range(K):
            aK[torch.arange(batch), k * contract.NUM_ACTIONS + Aidx[idx, k]] = 1.0

        if aux in ("next_state", "state_latent"):
            tgt = ((O[idx, 1] - O[idx, 0]) - dmu[:, 0]) / dsd[:, 0]
            loss = loss + lam * ((net.h_state(torch.cat([z, a1], 1)) - tgt) ** 2).mean()
        if aux in ("next_latent", "state_latent"):
            with torch.no_grad():                       # stop-grad: no collapse
                zt = net.encoder(On[idx, 1])
            loss = loss + lam * ((net.h_latent(torch.cat([z, a1], 1)) - zt) ** 2).mean()
        if aux == "multi_state":
            tgt = ((O[idx, 1:] - O[idx, :-1]) - dmu) / dsd
            pred = net.h_state(torch.cat([z, aK], 1)).view(batch, K, contract.OBS_SIZE)
            loss = loss + lam * ((pred - tgt) ** 2).mean()
        if aux == "multi_latent":
            with torch.no_grad():
                zt = torch.stack([net.encoder(On[idx, k + 1]) for k in range(K)], 1)
            pred = net.h_latent(torch.cat([z, aK], 1)).view(batch, K, net.hidden)
            loss = loss + lam * ((pred - zt) ** 2).mean()
        if aux == "geom":
            loss = loss + lam * ((net.h_geom(z) - gm[idx]) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

    net.eval()
    with torch.no_grad():
        va = dataset.load_split(corpus, "val", slices=("expert",), with_geometry=False)
        xv = (torch.from_numpy(va.obs) - mu) / sd
        acc = float((net(xv).argmax(1) == torch.from_numpy(va.action)).float().mean())
        # Latent spread: near zero would mean the encoder collapsed.
        zvar = float(net.encoder(On[:2048, 0]).var(0).mean())

    rpo = out_rpo or f"/tmp/aux_{aux}_{budget}_{seed}.rpo"
    # Export only the inference path. export_policy walks every nn.Linear in
    # the module, so passing `net` directly would serialize the auxiliary heads
    # too and produce a policy whose last layer is an auxiliary output.
    export_policy.export(nn.Sequential(net.encoder.net, net.head),
                         mu.numpy(), sd.numpy(), rpo)
    ret, surv = online(rpo, episodes)
    return dict(acc=acc, ret=ret, surv=surv, zvar=zvar)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--aux", default=",".join(AUX))
    ap.add_argument("--budgets", default="500,2000,10000")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="data/bc_aux.json")
    args = ap.parse_args()

    auxes = args.aux.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]
    print(f"BC + auxiliary world-model objectives  lam={args.lam}  "
          f"steps={args.steps}  seeds={args.seeds}")
    print("aux loss uses the SAME batch as the policy loss: no arm sees more data\n")
    print(f"{'labels':>7} " + "".join(f"{a:>14}" for a in auxes), flush=True)

    res = {}
    for b in budgets:
        row = []
        for a in auxes:
            rs, zs = [], []
            for s in range(args.seeds):
                r = run(args.corpus, a, b, args.steps, args.lam, seed=s)
                rs.append(r["ret"]); zs.append(r["zvar"])
            res.setdefault(a, {})[b] = dict(ret=float(np.mean(rs)), sd=float(np.std(rs)),
                                            zvar=float(np.mean(zs)))
            row.append(f"{np.mean(rs):>9.2f}±{np.std(rs):<4.2f}")
        print(f"{b:>7,} " + "".join(row), flush=True)

    base = res["none"] if "none" in res else None
    if base:
        print(f"\n{'labels':>7} " + "".join(f"{a:>14}" for a in auxes if a != "none"))
        for b in budgets:
            cells = []
            for a in auxes:
                if a == "none":
                    continue
                x, y0 = res[a][b], base[b]
                p = np.sqrt((x["sd"]**2 + y0["sd"]**2) / 2) + 1e-9
                cells.append(f"{100*(x['ret']-y0['ret'])/y0['ret']:>+8.1f}%{(x['ret']-y0['ret'])/p:>+5.1f}s")
            print(f"{b:>7,} " + "".join(cells))
    print("\nlatent variance (collapse check, ~0 would be bad):")
    for a in auxes:
        print(f"  {a:<14} {res[a][budgets[-1]]['zvar']:.4f}")
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
