"""Train several recipe arms side by side, on identical batches, in one process.

  python -m mobawm.wm.ablate --name ablate1 --minutes 5 --wandb

Arms (each its own model and optimizer, same init seed):
  base     plain loss, uniform batches                       (E2's recipe)
  gate     + copy by default (change gate)                    framing 3 alone
  events   + effect weights + event-oversampled batches       framing 1 alone
  both     gate + effect weights + event batches              framings 1 + 3
  gate_sparse  change gate on everything except movement (x, y, waypoint)
  gate_sparse_w02  gate_sparse with the gate's yes/no loss weighted x0.2
  gate_sparse_w02_all  + gated value loss averaged over all entries (scaled by change rate)
  baseline_v1  = gate_sparse_w02_all
  baseline     = baseline_v1 + hero walking loss x10, the adopted baseline
  walk_only    baseline model and inputs, loss on hero x/y only (diagnostic)
  hero_mlp     per-hero MLP on its own stats + action + wall patch, walk-only loss (reference)
  walk_x5/x10  baseline with the hero x/y loss terms weighted x5 / x10

Every round draws one uniform batch and one event batch; arms without event
sampling train on the uniform batch, arms with it on the event batch, so
arms differing only in the gate see exactly the same data. All arms are
scored on the same fixed held-out windows every --eval-every rounds, per
scoreboard row (docs/moba/EXPERIMENTS.md): better than copying, and error
with wrong actions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import train as T
from .data import KIND, Featurizer, fit_stats, load_store
from .model import TokenWM
from ..shards import ACTION_SIZES

class HeroMLP(torch.nn.Module):
    """Reference: each hero predicts its own next change from its own stats,
    position, 9x9 wall patch and its action. No attention, no memory, no other
    units. Everything else is predicted as unchanged (persistence)."""

    def __init__(self, d=256):
        super().__init__()
        from .model import Fourier
        self.d = 1
        self.fourier = Fourier(8)
        n_in = 9 + 7 + 32 + 81 + sum(ACTION_SIZES)
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, d), torch.nn.GELU(), torch.nn.Linear(d, d),
                                       torch.nn.GELU(), torch.nn.Linear(d, 13))
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)

    def set_change_prior(self, rate):
        pass

    def step(self, f, a, h):
        B = a.shape[0]
        onehot = torch.cat([torch.nn.functional.one_hot(a[..., i], n).float() for i, n in enumerate(ACTION_SIZES)], -1)
        x = torch.cat([f["shared"][:, :10], f["hero"], self.fourier(f["pos"][:, :10]), f["patch"][:, :10], onehot], -1)
        reg = torch.zeros(B, 206, 13, device=a.device)
        reg[:, :10] = self.net(x)
        c = slice(10, 110)
        return dict(reg=reg, alive_flip=torch.full((B, 206), -6.0, device=a.device),
                    is_hit=torch.zeros(B, 206, device=a.device),
                    lane=5 * torch.nn.functional.one_hot(f["lane"], 6).float(),
                    team=5 * torch.nn.functional.one_hot(f["team"][:, c], 3).float()), h


SPARSE = ("q_timer", "w_timer", "e_timer", "mana", "health", "basic_attack_timer",
          "stun_timer", "move_timer", "move_modifier", "xp")   # everything but movement
ARMS = {
    "base": dict(gate=False, weights={}, events=False),
    "gate_sparse": dict(gate=True, gate_fields=SPARSE, weights={}, events=False),
    "gate": dict(gate=True, weights={}, events=False),
    "gate_sparse_w02": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, weights={}, events=False),
    "gate_sparse_w02_all": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False),
    "baseline_v1": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False),
    "baseline": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False,
                     walk_w=10.0),
    "walk_only": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False,
                      walk_only=True),
    "hero_mlp": dict(model="hero_mlp", gate=False, weights={}, events=False, walk_only=True),
    "walk_x5": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False, walk_w=5.0),
    "walk_x10": dict(gate=True, gate_fields=SPARSE, gate_w=0.2, value_norm="all", weights={}, events=False, walk_w=10.0),
    "events": dict(gate=False, weights=dict(abilities=3, health=3, deaths=3, attack=2), events=True),
    "both": dict(gate=True, weights=dict(abilities=3, health=3, deaths=3, attack=2), events=True),
}
ROWS = ("walking", "attack", "q", "w", "e", "mana", "health")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/moba/mix1k/dense")
    ap.add_argument("--train-games", default="0-299")
    ap.add_argument("--held-games", default="900-929")
    ap.add_argument("--arms", default="base,gate,events,both")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0 = time only)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--burn", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--event-frac", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-batches", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.BF16 = args.bf16
    out = Path("runs/ablate") / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    files = sorted(Path(args.data).glob("games.*.mbs"))
    tr = load_store(files, T.parse_games(args.train_games), dev)
    he = load_store(files, T.parse_games(args.held_games), dev)
    stats = fit_stats(tr)
    fz = Featurizer(dev)
    kind = torch.from_numpy(KIND).to(dev)
    print(f"loaded {tr.n_steps} train / {he.n_steps} held-out steps in {time.time() - t0:.0f}s", flush=True)

    arms = {}
    for name in args.arms.split(","):
        cfg = ARMS[name]
        torch.manual_seed(0)                              # identical init across arms
        if cfg.get("model") == "hero_mlp":
            m = HeroMLP().to(dev)
        else:
            m = TokenWM(gate=cfg["gate"], gate_fields=cfg.get("gate_fields")).to(dev)
        m.set_change_prior(stats.change_rate)
        run = None
        if args.wandb:
            import wandb
            run = wandb.init(project="moba-wm", group=args.name, name=f"{args.name}/{name}",
                             config=vars(args) | dict(arm=name, **{k: str(v) for k, v in cfg.items()}),
                             reinit="create_new")
        arms[name] = dict(cfg=cfg, model=m, opt=torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01),
                          run=run, hist=[])

    val_args = argparse.Namespace(val_batches=args.val_batches, batch=args.batch, window=args.window, burn=args.burn)

    def evaluate(rnd):
        line = []
        for name, a in arms.items():
            T.EFFECT_WEIGHT.clear(); T.EFFECT_WEIGHT.update(a["cfg"]["weights"])
            T.GATE_LOSS_WEIGHT = a["cfg"].get("gate_w", 1.0)
            T.GATE_VALUE_NORM = a["cfg"].get("value_norm", "changed")
            T.WALK_ONLY = a["cfg"].get("walk_only", False)
            T.HERO_WALK_WEIGHT = a["cfg"].get("walk_w", 1.0)
            sk = T.skill_eval(a["model"], fz, he, stats, kind, dev, val_args)
            a["hist"].append(dict(round=rnd, **sk))
            if a["run"]:
                a["run"].log({f"skill/{k}": v for k, v in sk.items()}, step=rnd)
            line.append(f"{name:6s} " + " ".join(
                f"{r} {sk.get(f'hero/{r}/better_than_copy_pct', float('nan')):+4.0f}/"
                f"{sk.get(f'hero/{r}/action_dependence_pct', float('nan')):+4.0f}" for r in ROWS)
                + f" deaths {sk['deaths/recall_pct']:.0f}%")
        print(f"--- round {rnd}  ({time.time() - t_train:.0f}s)  better-than-copy / error-with-wrong-actions, %", flush=True)
        print("\n".join(line), flush=True)

    gen = torch.Generator(device=dev); gen.manual_seed(0)
    t_train = time.time()
    evaluate(0)
    rnd = 0
    while time.time() - t_train < args.minutes * 60 and not (args.rounds and rnd >= args.rounds):
        uniform = tr.sample(args.batch, args.window, gen)
        event = tr.sample(args.batch, args.window, gen, event_frac=args.event_frac, burn=args.burn)
        rnd += 1
        for name, a in arms.items():
            T.EFFECT_WEIGHT.clear(); T.EFFECT_WEIGHT.update(a["cfg"]["weights"])
            T.GATE_LOSS_WEIGHT = a["cfg"].get("gate_w", 1.0)
            T.GATE_VALUE_NORM = a["cfg"].get("value_norm", "changed")
            T.WALK_ONLY = a["cfg"].get("walk_only", False)
            T.HERO_WALK_WEIGHT = a["cfg"].get("walk_w", 1.0)
            lr = args.lr * min(1.0, rnd / 50)
            for g in a["opt"].param_groups:
                g["lr"] = lr
            loss, _, _, _ = T.run_window(a["model"], fz, event if a["cfg"]["events"] else uniform,
                                         stats, kind, args.burn, dev)
            a["opt"].zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(a["model"].parameters(), args.clip).item()
            a["opt"].step()
            if a["run"] and rnd % 5 == 0:
                a["run"].log({"train/loss": loss.item(), "grad_norm": gn}, step=rnd)
        if rnd % args.eval_every == 0:
            evaluate(rnd)
    if rnd % args.eval_every:
        evaluate(rnd)
    (out / "history.json").write_text(json.dumps({n: a["hist"] for n, a in arms.items()}, indent=1))
    for a in arms.values():
        if a["run"]:
            a["run"].finish()
    print(f"done: {rnd} rounds in {time.time() - t_train:.0f}s -> {out}/history.json", flush=True)


if __name__ == "__main__":
    main()
