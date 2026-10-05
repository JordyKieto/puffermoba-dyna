"""The world-model debugger's divergence metrics (tools/moba_viz.py Replay: the HUD table), in batch, for the paper.
Open-loop rollouts from real states of the CURRENT policy's real games (collected after every evaluated checkpoint
was taken: no model trained on them), driven by the real recorded actions of all 10 heroes, for several world-model
versions. Per (model, game, start): hero position L1 (cells), hero HP MAE, creep alive agreement (%), tower HP MAE
at steps 16 / 64 / 128 / 200, and the same for persistence (the start state held fixed).
Output: results/moba/paper/debugger_eval.csv.
usage: python scripts/moba/debugger_eval.py [--files games.1000840.mbs,...] [--device cpu]"""
import argparse, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import moba_viz as V  # noqa: E402

MODELS = {"E38 dream (no policy games)": "runs/token_wm/fleet20e_at500", "Dyna round 1 (wm_r1)": "runs/token_wm/wm_r1",
          "wml @3600": "runs/token_wm/wml_at3600", "wml2 @3400": "runs/token_wm/wml2_at3400"}
STEPS = (16, 64, 128, 200)
STARTS = (500, 1500, 3000, 4500)
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/moba/dyna_eval/dense")
ap.add_argument("--files", default="games.1000840.mbs,games.1010840.mbs,games.1040850.mbs")
ap.add_argument("--device", default="cpu"); ap.add_argument("--horizon", type=int, default=200)
a = ap.parse_args()
out = ROOT / "results/moba/paper/debugger_eval.csv"
rows = []
for label, run in MODELS.items():
    for fname in a.files.split(","):
        dense = ROOT / a.dir / fname
        for start in STARTS:
            rp = None
            for g in range(10):
                if rp is None:
                    rp = V.Replay(ROOT / run, dense, game=g, start=start, horizon=a.horizon, device=a.device)
                else:
                    rp.load(g)
                if rp.start != start or rp.Hz < a.horizon:          # the game is too short for this start
                    continue
                ours = "radiant" if g % 2 == 0 else "dire"            # the collector alternates sides: ours radiant in even games
                for k in STEPS:
                    d = rp.divergence(k)
                    rows.append(dict(model=label, file=fname, game=g, game_index=rp.gi, our_side=ours, start=start, step=k,
                                     **{f"model_{n}": round(v[0], 4) for n, v in d.items()},
                                     **{f"persist_{n}": round(v[1], 4) for n, v in d.items()}))
        print(label, fname, len(rows), flush=True)
    with open(out, "w", newline="") as f:                              # rewrite after every model: partial results survive
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out, len(rows))
