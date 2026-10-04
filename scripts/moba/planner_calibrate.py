"""Per-frame cost of the world-model planner's parts on THIS machine (the 60 fps budget is 16.7 ms).
Writes results/moba/planner/calibration_<host>.json.

    .venv/bin/python scripts/moba/planner_calibrate.py
"""
import sys, time, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json, platform
OUT = {}
from mobawm.rl.plan_mpc import Arena, load_wm
from mobawm.wm.data import Featurizer, apply_prediction
from mobawm.rl.dream_ppo import reward
def p(x): return f"p50 {np.percentile(x,50):5.1f}  p95 {np.percentile(x,95):5.1f} ms"
A = Arena(1); A.start(0, 50_000_000)
act = np.zeros((10, 6), np.float32); act[:, :2] = 3
for _ in range(300): A.step(0, act)                      # mid-game-ish state
t_game, t_net, t_state = [], [], []
for _ in range(100):
    t = time.perf_counter(); net = A.net_actions(); t_net.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter(); ent, tick, cidx = A.states(); t_state.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter(); A.step(0, net[0]); t_game.append((time.perf_counter() - t) * 1e3)
print(f"game step        {p(t_game)}\nshipped net (10) {p(t_net)}\nstate capture    {p(t_state)}")
for k, v in (("game_step", t_game), ("shipped_net_10_heroes", t_net), ("state_capture", t_state)):
    OUT[k] = dict(p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95)))
for devn in ("cpu", "mps"):
    dev = torch.device(devn); model, stats = load_wm(str(ROOT / "runs/token_wm/e5_rollout"), dev); fz = Featurizer(dev)
    e = torch.from_numpy(ent).to(dev); tk = torch.from_numpy(tick).to(dev); ci = torch.from_numpy(cidx).to(dev)
    a1 = torch.from_numpy(net.astype(np.int64)).to(dev)
    for B in (1, 2, 4, 8, 16, 32):
        eb, tb, cb, ab = e.repeat(B, 1, 1), tk.repeat(B), ci.repeat(B), a1.repeat(B, 1, 1)
        h = torch.zeros(B, 206, model.d, device=dev); ts = []
        with torch.no_grad():
            for i in range(25):
                t = time.perf_counter()
                pred, h = model.step(fz(eb, tb, cb), ab, h)
                nxt, _, _ = apply_prediction(eb, tb, cb, pred, stats, fz); r, _ = reward(eb, nxt); float(r.sum())
                if i >= 5: ts.append((time.perf_counter() - t) * 1e3)
        print(f"{devn} WM step+reward batch {B:2d}: {p(ts)}")
        OUT[f"wm_step_reward_{devn}_batch{B}"] = dict(p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)))

OUT["machine"] = dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__, torch_threads=torch.get_num_threads())
(ROOT / "results/moba/planner").mkdir(parents=True, exist_ok=True)
(ROOT / "results/moba/planner/calibration_mac.json").write_text(json.dumps(OUT, indent=1))
print("-> results/moba/planner/calibration_mac.json")
