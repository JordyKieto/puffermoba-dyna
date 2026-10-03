"""Tower evaluation on many held-out games, for comparing checkpoints without the noise of the 64-window
in-training eval. Open loop: windows of H ticks starting every `stride` ticks in every game (memory warmed
on `burn` real ticks); a predicted tower fall matches a real fall of the same tower in the same window within
+-tol ticks. Also siege damage (towers losing >= 800 HP in the window). Per-game bootstrap 90% intervals.

    .venv/bin/python scripts/moba/tower_eval_big.py --wm runs/token_wm/e12_at4000 --games 900-999 --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from mobawm import shards as S
import mobawm.wm.data as D
F = S.F; T0 = 182

def parse(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-"); out |= set(range(int(a), int(b or a) + 1))
    return out

@torch.no_grad()
def game(model, stats, fz, r, dev, H, stride, burn, tol, bs=64):
    st = r["state"]
    e = st["entity"].astype(np.float32)
    if getattr(D, "LIST_STATE", False):              # models whose state carries the stored target lists
        x = np.zeros(e.shape[:-1] + (len(F),), np.float32); x[..., :D.N_BASE] = e; x[..., D.N_BASE:] = D.pack_lists_np(st["scan"]); e = x
    ent = torch.from_numpy(e).to(dev); tick = torch.from_numpy(st["tick"].astype(np.int64)).to(dev)
    cidx = torch.from_numpy(st["creep_idx"].astype(np.int64)).to(dev); act = torch.from_numpy(r["actions"][:-1].astype(np.int64)).to(dev)
    T = len(act); starts = list(range(burn, T - H, stride))
    real_n = pred_n = hit = 0; siege_r = siege_d = 0.0
    for s0 in range(0, len(starts), bs):
        S0 = starts[s0:s0 + bs]; B = len(S0)
        h = torch.zeros(B, 206, model.d, device=dev)
        for k in range(burn):
            idx = [s - burn + k for s in S0]; _, h = model.step(fz(ent[idx], tick[idx], cidx[idx]), act[idx], h)
        cur, ct, cc = ent[S0].clone(), tick[S0].clone(), cidx[S0].clone()
        hp0 = (cur[:, T0:, F["health"]] * (cur[:, T0:, F["alive"]] > 0)).clone()
        rf, pf = [], []
        for k in range(H):
            idx = [s + k for s in S0]
            p, h = model.step(fz(cur, ct, cc), act[idx], h)
            nxt, ct, cc = D.apply_prediction(cur, ct, cc, p, stats, fz); nxt = torch.nan_to_num(nxt)
            pf.append((cur[:, T0:, F["alive"]] > 0) & (nxt[:, T0:, F["alive"]] <= 0))
            rf.append((ent[idx][:, T0:, F["alive"]] > 0) & (ent[[i + 1 for i in idx]][:, T0:, F["alive"]] <= 0))
            cur = nxt
        rf, pf = torch.stack(rf, 1).cpu(), torch.stack(pf, 1).cpu()                      # (B, H, 24)
        real_n += int(rf.sum()); pred_n += int(pf.sum())
        for b, t, w in torch.nonzero(rf).tolist():                                     # one-to-one-ish: any predicted fall of that tower within tol
            lo, hi = max(0, t - tol), min(H, t + tol + 1)
            hit += int(pf[b, lo:hi, w].any())
        end = [s + H for s in S0]
        rl = (hp0 - ent[end][:, T0:, F["health"]] * (ent[end][:, T0:, F["alive"]] > 0)).clamp(min=0).cpu()
        dl = (hp0 - cur[:, T0:, F["health"]] * (cur[:, T0:, F["alive"]] > 0)).clamp(min=0).cpu()
        m = rl >= 800; siege_r += float(rl[m].sum()); siege_d += float(dl[m].sum())
    return dict(real=real_n, pred=pred_n, hit=hit, siege_real=siege_r, siege_dream=siege_d, windows=len(starts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True); ap.add_argument("--games", default="900-999")
    ap.add_argument("--dense", default="data/moba/mix1k/dense"); ap.add_argument("--device", default="mps")
    ap.add_argument("--H", type=int, default=128); ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--burn", type=int, default=8); ap.add_argument("--tol", type=int, default=32)
    ap.add_argument("--out", default=""); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_num_threads(4)
    import moba_viz as V
    model, stats, _ = V.load_model(ROOT / a.wm, ROOT / a.dense / "games.0000900.mbs", dev)
    model.requires_grad_(False); fz = D.Featurizer(dev)
    want = parse(a.games); per = []; t0 = time.time()
    for f in sorted((ROOT / a.dense).glob("games.*.mbs")):
        d = S.load_dense(str(f))
        for k, (gi, *_ ) in enumerate(d.games):
            if gi in want:
                per.append(dict(game=gi, **game(model, stats, fz, d.load_game(k), dev, a.H, a.stride, a.burn, a.tol)))
                if len(per) % 10 == 0: print(f"  {a.wm}: {len(per)} games ({time.time() - t0:.0f}s)", flush=True)
    R = {k: np.array([g[k] for g in per], float) for k in ("real", "pred", "hit", "siege_real", "siege_dream")}
    stat = lambda idx: (R["hit"][idx].sum() / max(R["pred"][idx].sum(), 1), R["hit"][idx].sum() / max(R["real"][idx].sum(), 1),
                        R["siege_dream"][idx].sum() / max(R["siege_real"][idx].sum(), 1))
    rng = np.random.default_rng(0); n = len(per)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(1000)])
    p, r_, sg = stat(np.arange(n)); lo, hi = np.percentile(boots, [5, 95], 0)
    res = dict(wm=a.wm, games=n, windows=int(sum(g["windows"] for g in per)), real=int(R["real"].sum()), pred=int(R["pred"].sum()),
               precision=p, recall=r_, siege_ratio=sg, precision_ci=[lo[0], hi[0]], recall_ci=[lo[1], hi[1]], siege_ci=[lo[2], hi[2]])
    print(f"{a.wm}: {n} games, {res['windows']} windows | tower falls real {res['real']} predicted {res['pred']} | "
          f"precision {100*p:.0f}% [{100*lo[0]:.0f}-{100*hi[0]:.0f}] recall {100*r_:.0f}% [{100*lo[1]:.0f}-{100*hi[1]:.0f}] | "
          f"siege damage {sg:.2f}x [{lo[2]:.2f}-{hi[2]:.2f}] | {time.time() - t0:.0f}s", flush=True)
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
