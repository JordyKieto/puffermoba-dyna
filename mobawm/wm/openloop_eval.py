"""Held-out loss by open-loop horizon (train.eval_horizons, the key metric) for a list
of checkpoints: the same windows and loss as train.py logs during a run, so earlier
checkpoints (e.g. E4's teacher-forced ones) sit on the same scale.

    python -m mobawm.wm.openloop_eval --stats runs/token_wm/e4_baseline_v2_1h \
        runs/token_wm/e4_baseline_v2_1h/model_000500.pt runs/token_wm/e4_baseline_v2_1h/model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import Featurizer, Stats, load_store
from .model import TokenWM
from . import train as T
from .train import KIND, eval_horizons, parse_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--stats", required=True, help="run dir whose stats.pt (scales + config) the checkpoints share")
    ap.add_argument("--data", default="data/moba/mix1k/dense")
    ap.add_argument("--held-games", default="900-999")
    ap.add_argument("--baseline", action="store_true", help="loss weights of --baseline (as E4/E5 trained)")
    args = ap.parse_args()
    if args.baseline:
        T.GATE_LOSS_WEIGHT, T.GATE_VALUE_NORM, T.HERO_WALK_WEIGHT = 0.2, "all", 10.0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(Path(args.stats) / "stats.pt", map_location="cpu", weights_only=False)
    stats = Stats(**sd["stats"])
    files = sorted(Path(args.data).glob("games.*.mbs"))
    he = load_store(files, parse_games(args.held_games), dev)
    fz, kind = Featurizer(dev), torch.from_numpy(KIND).to(dev)
    rows = []
    for c in args.ckpts:
        model = TokenWM(**sd["config"]).to(dev)
        model.set_change_prior(stats.change_rate)
        model.load_state_dict(torch.load(c, map_location=dev))
        model.eval()
        r = eval_horizons(model, fz, he, stats, kind, dev)
        row = dict(ckpt=c, **{k: round(v, 3) for k, v in r.items() if not k.startswith("windows")})
        rows.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
