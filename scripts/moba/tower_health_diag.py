"""At each real tower fall: the predicted health fraction (open loop from the window start, and one step)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from mobawm import shards as S
from mobawm.rl.plan_mpc import load_wm
from mobawm.wm.data import Featurizer, apply_prediction
F = S.F
if __name__ == "__main__":
    wm, H, burn = sys.argv[1], 128, 8
    dev = torch.device("cuda")
    model, stats = load_wm(wm, dev); fz = Featurizer(dev)
    d = S.load_dense("data/moba/mix1k/dense/games.0000900.mbs")
    ol, tf, real1, real8 = [], [], [], []
    for gi in range(10):
        r = d.load_game(gi)
        ent = torch.from_numpy(r["state"]["entity"].astype(np.float32)).to(dev)
        tick = torch.from_numpy(r["state"]["tick"].astype(np.int64)).to(dev)
        cidx = torch.from_numpy(r["state"]["creep_idx"].astype(np.int64)).to(dev)
        act = torch.from_numpy(r["actions"][:-1].astype(np.int64)).to(dev)
        T = len(act); starts = list(range(burn, T - H, H)); B = len(starts)
        if not B: continue
        with torch.no_grad():
            h = torch.zeros(B, 206, model.d, device=dev)
            for k in range(burn):
                idx = [s - burn + k for s in starts]
                _, h = model.step(fz(ent[idx], tick[idx], cidx[idx]), act[idx], h)
            h1 = h.clone()
            cur, ct, cc = ent[starts].clone(), tick[starts].clone(), cidx[starts].clone()
            for k in range(H):
                idx = [s + k for s in starts]; nx = [i + 1 for i in idx]
                p, h = model.step(fz(cur, ct, cc), act[idx], h)
                nxt, ct, cc = apply_prediction(cur, ct, cc, p, stats, fz)
                p1, h1 = model.step(fz(ent[idx], tick[idx], cidx[idx]), act[idx], h1)
                n1, _, _ = apply_prediction(ent[idx], tick[idx], cidx[idx], p1, stats, fz)
                rf = (ent[idx][:, 182:, F["alive"]] > 0) & (ent[nx][:, 182:, F["alive"]] <= 0)
                for w, t in torch.nonzero(rf).tolist():
                    mh = float(ent[idx][w, 182 + t, F["max_health"]])
                    # predicted health BEFORE rule R8 zeroes a dead unit: use the pre-death state's health + predicted change
                    ol.append(float(cur[w, 182 + t, F["health"]]) / mh)          # predicted health going into the fatal tick
                    tf.append(float(ent[idx][w, 182 + t, F["health"]]) / mh)     # real health going into it
                    j = max(0, idx[w] - 7)
                    real8.append(float(ent[j, 182 + t, F["health"]]) / mh)
                cur = torch.nan_to_num(nxt)
    ol, tf, real8 = map(np.array, (ol, tf, real8))
    q = lambda a: " ".join(f"{np.percentile(a, x):.2f}" for x in (10, 25, 50, 75, 90))
    print(f"{wm}: {len(ol)} real falls")
    print(f"  real health fraction on the tick before the fall  p10/25/50/75/90: {q(tf)}")
    print(f"  real health fraction 8 ticks before               p10/25/50/75/90: {q(real8)}")
    print(f"  OPEN-LOOP predicted health fraction before fall    p10/25/50/75/90: {q(ol)}  | share <= 5%: {(ol<=0.05).mean():.0%}, <= 20%: {(ol<=0.2).mean():.0%}")
