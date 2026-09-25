"""Train and evaluate the token world model.

  python3 -m mobawm.wm.train --data data/moba/mix100/dense --train-games 0-90 \
      --held-games 91-99 --minutes 20 --out runs/token_wm/<name>

The whole dataset is loaded once into device tensors (mobawm.wm.data.Store)
and windows are sampled on the device. Splits are by game (seed), never by
row. Progress is printed every --log-every iterations and appended to
OUT/log.jsonl (and wandb with --wandb); the final report is OUT/report.json.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .. import shards as S
from .data import (F, KIND, N_KIND, REG_FIELDS, REG_IDX, Featurizer, Store, apply_prediction,
                   fit_stats, load_store, wave_respawns)
from .model import TokenWM



def parse_games(spec: str) -> set[int]:
    """'0-25,40,50-60' -> set of game indices."""
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def featurize_window(fz, ent, tick, cidx):
    """(B,K+1,...) raw -> list of K+1 feature dicts."""
    B, T = ent.shape[:2]
    f = fz(ent.flatten(0, 1), tick.flatten(), cidx.flatten())
    return [{k: v.view(B, T, *v.shape[1:])[:, t] for k, v in f.items()} for t in range(T)]


KIND_NAMES = ("hero", "creep", "neutral", "tower")
# Delta fields grouped by what they mean in the game. Losses and the skill
# metrics are reported per effect, so "does it understand abilities" is a
# curve, not a question for the end-of-run report.
EFFECTS = {
    "movement": ("x", "y", "waypoint"),
    "abilities": ("q_timer", "w_timer", "e_timer", "mana"),
    "attack": ("basic_attack_timer",),
    "health": ("health",),
    "status": ("stun_timer", "move_timer", "move_modifier"),
    "xp": ("xp",),
}
EFFECT_OF = {f: e for e, fs in EFFECTS.items() for f in fs}
# The target scoreboard's rows (docs/moba/EXPERIMENTS.md), for hero skill metrics:
# abilities are split per ability so Q, W, E and mana each get their own curve.
HERO_ROWS = {"walking": ("x", "y"), "attack": ("basic_attack_timer",), "q": ("q_timer",),
             "w": ("w_timer",), "e": ("e_timer",), "mana": ("mana",), "health": ("health",),
             "status": ("stun_timer", "move_timer", "move_modifier"), "xp": ("xp",)}
HERO_ROW_OF = {f: r for r, fs in HERO_ROWS.items() for f in fs}
# Semantic loss: per-effect weights (--effect-weights). 1.0 = plain sum.
EFFECT_WEIGHT = {}
BF16 = False


def step_loss(pred, e0, e1, stats, kind, tick, cidx):
    """Total loss and its parts, per entity kind and per head.

    parts["<kind>/<effect>"]  Huber on scaled deltas of that effect's fields (EFFECTS),
                              each (kind, field) averaged separately then summed
    parts["<kind>/deaths"]    BCE on "alive flips this tick" (creep, neutral, tower)
    parts["<kind>/got_hit"]   BCE on next is_hit
    parts["creep/lane"], parts["creep/team"]  CE on next value
    """
    dev = e0.device
    alive0, alive1 = e0[..., F["alive"]] > 0, e1[..., F["alive"]] > 0
    scale = stats.reg_scale.to(dev)[kind]
    target = (e1[..., REG_IDX] - e0[..., REG_IDX]) / scale
    gated = "change" in pred
    if gated:
        # copy by default: BCE on "did this field change" everywhere it is
        # supervised, value loss only where it really changed
        changed = target != 0
        gate_err = Fn.binary_cross_entropy_with_logits(pred["change"], changed.float(), reduction="none")
        err = Fn.huber_loss(pred["value"], target, reduction="none", delta=1.0)
    else:
        err = Fn.huber_loss(pred["reg"], target, reduction="none", delta=1.0)
    # units respawned in place by a wave are applied by rule, not regressed (a teleport to spawn)
    wc, _, wn = wave_respawns(tick, cidx)
    rule = torch.zeros_like(alive0)
    rule[:, 10:110] = wc
    rule[:, 110:182] = wn.unsqueeze(-1)
    mask = (alive0 & alive1 & ~rule).unsqueeze(-1) & stats.reg_mask.to(dev)[kind]
    can_flip = stats.alive_mask.to(dev)
    flip = (alive0 != alive1).float()
    alive_bce = Fn.binary_cross_entropy_with_logits(pred["alive_flip"], flip, reduction="none")
    hit_bce = Fn.binary_cross_entropy_with_logits(pred["is_hit"], e1[..., F["is_hit"]], reduction="none")
    parts = {}
    for k, name in enumerate(KIND_NAMES):
        sel = kind == k
        m, e = mask[:, sel], err[:, sel]
        if gated:
            mc = m & changed[:, sel]
            gated_f = (gate_err[:, sel] * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1) \
                + (e * mc).sum((0, 1)) / mc.sum((0, 1)).clamp(min=1)
            plain_f = (e * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
            per_field = torch.where(pred["gate_mask"], gated_f, plain_f)
        else:
            per_field = (e * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
        for j, fname in enumerate(REG_FIELDS):
            if stats.reg_mask[k, j]:
                eff = EFFECT_OF[fname]
                key = f"{name}/{eff}"
                parts[key] = parts.get(key, 0.0) + EFFECT_WEIGHT.get(eff, 1.0) * per_field[j]
        if can_flip[k]:
            parts[f"{name}/deaths"] = EFFECT_WEIGHT.get("deaths", 1.0) * alive_bce[:, sel].mean()
        a1 = alive1[:, sel]
        if a1.any():
            parts[f"{name}/got_hit"] = hit_bce[:, sel][a1].mean()
    c = slice(10, 110)
    ca = alive1[:, c]
    if ca.any():
        parts["creep/lane"] = Fn.cross_entropy(pred["lane"][ca], e1[:, c, F["lane"]][ca].long())
        parts["creep/team"] = Fn.cross_entropy(pred["team"][ca], e1[:, c, F["team"]][ca].long().clamp(0, 2))
    return sum(parts.values()), parts


def run_window(model, fz, batch, stats, kind, burn, dev, loss=True, shuffle_actions=False):
    ent, act, tick, cidx = batch
    if shuffle_actions:
        act = act[torch.randperm(act.shape[0], device=dev)]
    feats = featurize_window(fz, ent, tick, cidx)
    B, K = act.shape[:2]
    h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
    total, parts, preds = 0.0, {}, []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (BF16 and dev.type == "cuda") else nullcontext()
    for t in range(K):
        with amp:
            pred, h = model.step(feats[t], act[:, t], h)
        pred = {k: (v.float() if v.is_floating_point() else v) for k, v in pred.items()}
        h = h.float()
        if t >= burn:
            if loss:
                l, p = step_loss(pred, ent[:, t], ent[:, t + 1], stats, kind, tick[:, t], cidx[:, t])
                total = total + l
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v
            preds.append((t, pred))
    n = K - burn
    if loss:
        return total / n, {k: v / n for k, v in parts.items()}, preds, ent
    return None, None, preds, ent


def summarize_parts(parts):
    """Per-kind totals plus every head, as floats."""
    out = {k: float(v) for k, v in parts.items()}
    for name in KIND_NAMES:
        out[f"{name}/total"] = sum(v for k, v in out.items() if k.startswith(name + "/") and k != f"{name}/total")
    return out


@torch.no_grad()
def validate(model, fz, store, stats, kind, dev, args, seed=123):
    """Loss on the SAME held-out windows every time (fixed seed), averaged over batches."""
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    model.eval()
    tot, agg = 0.0, {}
    for _ in range(args.val_batches):
        batch = store.sample(args.batch, args.window, gen)
        l, parts, _, _ = run_window(model, fz, batch, stats, kind, args.burn, dev)
        tot += float(l)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    model.train()
    n = args.val_batches
    return tot / n, summarize_parts({k: v / n for k, v in agg.items()})


@torch.no_grad()
def skill_eval(model, fz, store, stats, kind, dev, args, seed=123):
    """The two questions that tell whether an effect is understood, per effect,
    on the same fixed held-out windows as validate():

      better_than_copy_pct  = 100 * (1 - model error / persistence error)
                              > 0: better than copying the last frame
      action_dependence_pct = 100 * (error with wrong actions / error with real actions - 1)
                              how much the prediction depends on the actions actually taken:
                              the real actions are swapped for another window's; 0 = ignores them

    Errors are on RMS-scaled deltas (unitless), summed over the effect's
    fields; only entities alive at t and t+1, excluding wave respawns. Deaths:
    recall and false positives of alive flips."""
    names = KIND_NAMES
    err = {}          # (kind, effect) -> [model, persist, shuffled]
    deaths = [0, 0, 0]  # tp, fn, fp
    model.eval()
    for shuffle in (False, True):
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        for _ in range(args.val_batches):
            batch = store.sample(args.batch, args.window, gen)
            _, _, preds, ent = run_window(model, fz, batch, stats, kind, args.burn, dev, loss=False,
                                          shuffle_actions=shuffle)
            scale = stats.reg_scale.to(dev)[kind]
            for t, p in preds:
                e0, e1 = ent[:, t], ent[:, t + 1]
                both = (e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] > 0)
                wc, _, wn = wave_respawns(batch[2][:, t], batch[3][:, t])
                both[:, 10:110] &= ~wc
                both[:, 110:182] &= ~wn.unsqueeze(-1)
                true_d = (e1[..., REG_IDX] - e0[..., REG_IDX]) / scale
                ae = (p["reg"] - true_d).abs()
                pe = true_d.abs()
                for k, kn in enumerate(names):
                    m = both[:, kind == k]
                    for j, fname in enumerate(REG_FIELDS):
                        if not stats.reg_mask[k, j]:
                            continue
                        group = HERO_ROW_OF[fname] if kn == "hero" else EFFECT_OF[fname]
                        a = err.setdefault((kn, group), [0.0, 0.0, 0.0])
                        if shuffle:
                            a[2] += ae[:, kind == k, j][m].sum().item()
                        else:
                            a[0] += ae[:, kind == k, j][m].sum().item()
                            a[1] += pe[:, kind == k, j][m].sum().item()
                if not shuffle:
                    can = stats.alive_mask.to(dev)[kind].expand_as(both)
                    flip = ((e0[..., F["alive"]] > 0) != (e1[..., F["alive"]] > 0)) & can
                    pf = (torch.sigmoid(p["alive_flip"]) > 0.5) & can
                    deaths[0] += int((flip & pf).sum()); deaths[1] += int((flip & ~pf).sum())
                    deaths[2] += int((~flip & pf).sum())
    model.train()
    out = {}
    for (kn, ef), (m, pz, sh) in err.items():
        if pz <= 0:
            continue
        out[f"{kn}/{ef}/better_than_copy_pct"] = 100 * (1 - m / pz)
        out[f"{kn}/{ef}/action_dependence_pct"] = 100 * (sh / max(m, 1e-9) - 1)
    out["deaths/recall_pct"] = 100 * deaths[0] / max(deaths[0] + deaths[1], 1)
    out["deaths/false_pos_per_true"] = deaths[2] / max(deaths[0] + deaths[1], 1)
    return out


@torch.no_grad()
def eval_onestep(model, fz, store, stats, kind, dev, gen, n_batches=8, B=16, K=16, burn=4, shuffle=False):
    """Natural-unit MAE of predicted deltas vs persistence (delta 0), per (kind, field)."""
    names = ["hero", "creep", "neutral", "tower"]
    acc = {}
    flips = {"tp": 0, "fn": 0, "fp": 0}
    for _ in range(n_batches):
        batch = store.sample(B, K, gen)
        _, _, preds, ent = run_window(model, fz, batch, stats, kind, burn, dev, loss=False, shuffle_actions=shuffle)
        scale = stats.reg_scale.to(dev)[kind]
        rmask = stats.reg_mask.to(dev)[kind]
        for t, p in preds:
            e0, e1 = ent[:, t], ent[:, t + 1]
            both = (e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] > 0)
            wc, _, wn = wave_respawns(batch[2][:, t], batch[3][:, t])
            both[:, 10:110] &= ~wc
            both[:, 110:182] &= ~wn.unsqueeze(-1)
            true_d = e1[..., REG_IDX] - e0[..., REG_IDX]
            pred_d = p["reg"] * scale
            for k in range(N_KIND):
                for j, fname in enumerate(REG_FIELDS):
                    if not stats.reg_mask[k, j]:
                        continue
                    m = both[:, kind == k]
                    td, pd = true_d[:, kind == k, j][m], pred_d[:, kind == k, j][m]
                    ch = td != 0
                    a = acc.setdefault((names[k], fname), [0.0, 0.0, 0, 0.0, 0.0, 0])
                    a[0] += (pd - td).abs().sum().item(); a[1] += td.abs().sum().item(); a[2] += len(td)
                    a[3] += (pd - td)[ch].abs().sum().item(); a[4] += td[ch].abs().sum().item(); a[5] += int(ch.sum())
            can = stats.alive_mask.to(dev)[kind].expand_as(both)
            flip = ((e0[..., F["alive"]] > 0) != (e1[..., F["alive"]] > 0)) & can
            pf = (torch.sigmoid(p["alive_flip"]) > 0.5) & can
            flips["tp"] += int((flip & pf).sum()); flips["fn"] += int((flip & ~pf).sum())
            flips["fp"] += int((~flip & pf).sum())
    out = {}
    for (kn, fn), a in sorted(acc.items()):
        out[f"{kn}.{fn}"] = dict(model_mae=a[0] / max(a[2], 1), persist_mae=a[1] / max(a[2], 1),
                                 changed_model_mae=a[3] / max(a[5], 1), changed_persist_mae=a[4] / max(a[5], 1),
                                 n=a[2], n_changed=a[5])
    out["alive_flip"] = dict(recall=flips["tp"] / max(flips["tp"] + flips["fn"], 1), **flips)
    return out


@torch.no_grad()
def eval_rollout(model, fz, store, stats, kind, dev, gen, n=16, burn=8, H=64, marks=(1, 4, 16, 64)):
    """Open loop: burn-in on real states, then feed predictions back for H steps."""
    B = n
    ent, act, tick, cidx = store.sample(B, burn + H, gen)
    h = torch.zeros(B, S.N_ENTITIES, model.d, device=dev)
    for t in range(burn):
        f = fz(ent[:, t], tick[:, t], cidx[:, t])
        _, h = model.step(f, act[:, t], h)
    cur, ct, cc = ent[:, burn].clone(), tick[:, burn].clone(), cidx[:, burn].clone()
    start = cur.clone()
    res = {}
    for k in range(1, H + 1):
        f = fz(cur, ct, cc)
        pred, h = model.step(f, act[:, burn + k - 1], h)
        cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
        if k in marks:
            truth = ent[:, burn + k]
            res[k] = {name: (metric(cur, truth), metric(start, truth)) for name, metric in METRICS.items()}
    return res


def _hero_pos(s, t):
    return ((s[:, :10, F["x"]] - t[:, :10, F["x"]]).abs() + (s[:, :10, F["y"]] - t[:, :10, F["y"]]).abs()).mean().item()


def _hero_hp(s, t):
    return (s[:, :10, F["health"]] - t[:, :10, F["health"]]).abs().mean().item()


def _creep_pos(s, t):
    m = (s[:, 10:110, F["alive"]] > 0) & (t[:, 10:110, F["alive"]] > 0)
    d = (s[:, 10:110, F["x"]] - t[:, 10:110, F["x"]]).abs() + (s[:, 10:110, F["y"]] - t[:, 10:110, F["y"]]).abs()
    return d[m].mean().item() if m.any() else float("nan")


def _creep_alive(s, t):
    return ((s[:, 10:110, F["alive"]] > 0) == (t[:, 10:110, F["alive"]] > 0)).float().mean().item()


def _tower_hp(s, t):
    return (s[:, 182:, F["health"]] - t[:, 182:, F["health"]]).abs().mean().item()


METRICS = {"hero_pos_L1": _hero_pos, "hero_hp": _hero_hp, "creep_pos_L1": _creep_pos,
           "creep_alive_acc": _creep_alive, "tower_hp": _tower_hp}


def log_report(run, report):
    """End-of-run evaluation as wandb tables and summary numbers."""
    import wandb
    he, sh = report["onestep_heldout"], report["onestep_heldout_shuffled_actions"]
    rows = []
    for f, v in he.items():
        if f == "alive_flip" or v["n_changed"] < 20:
            continue
        kn, fname = f.split(".", 1)
        rows.append([kn, fname, EFFECT_OF.get(fname, "?"), v["model_mae"], v["persist_mae"], sh[f]["model_mae"],
                     100 * (1 - v["model_mae"] / max(v["persist_mae"], 1e-9)),
                     100 * (sh[f]["model_mae"] / max(v["model_mae"], 1e-9) - 1),
                     v["changed_model_mae"], v["changed_persist_mae"], v["n_changed"]])
    run.log({"final/onestep_heldout": wandb.Table(
        columns=["unit", "field", "effect", "model_err", "copy_err", "wrong_actions_err", "better_than_copy_pct",
                 "action_dependence_pct", "changed_model_err", "changed_copy_err", "n_changed"], data=rows)})
    for split in ("rollout_heldout", "rollout_train"):
        rrows = [[int(h), name, m[0], m[1]] for h, ms in report[split].items() for name, m in ms.items()]
        run.log({f"final/{split}": wandb.Table(columns=["horizon", "metric", "model", "copy"], data=rrows)})
    run.summary.update({"final/loss_last_100": report["loss_last_100"],
                        "final/slope_per_1k_it": report["log_loss_slope_per_1k_it_last_quarter"],
                        "final/deaths_recall": he["alive_flip"]["recall"]})


def backfill(run_dir: Path, wandb_id: str):
    """Attach an older run's report.json to its wandb run (tables + summary)."""
    import wandb
    report = json.loads((run_dir / "report.json").read_text())
    run = wandb.init(project="moba-wm", id=wandb_id, resume="must")
    log_report(run, report)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/moba/mix100/dense")
    ap.add_argument("--train-games", default="0-25")
    ap.add_argument("--held-games", default="91-99")
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--burn", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/token_wm/smoke")
    ap.add_argument("--gate", action="store_true", help="copy by default: per-field change gate")
    ap.add_argument("--effect-weights", default="", help="semantic loss, e.g. abilities=3,health=3,deaths=3")
    ap.add_argument("--event-frac", type=float, default=0.0,
                    help="fraction of each batch drawn from windows with a cast, damage or death")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--backfill", help="run dir: attach its report.json to --wandb-id and exit")
    ap.add_argument("--wandb-id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.backfill:
        return backfill(Path(args.backfill), args.wandb_id)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    egen = torch.Generator(device=dev); egen.manual_seed(1)

    files = sorted(Path(args.data).glob("games.*.mbs"))
    tr_idx, he_idx = parse_games(args.train_games), parse_games(args.held_games)
    assert not tr_idx & he_idx, "train and held-out games overlap"
    t_load = time.time()
    print(f"loading from {args.data} ({len(files)} files) ...", flush=True)
    tr_store = load_store(files, tr_idx, dev)
    he_store = load_store(files, he_idx, dev)
    t_f = time.time()
    stats = fit_stats(tr_store)
    print(f"decompress + store {t_f - t_load:.1f}s, fit_stats {time.time() - t_f:.1f}s", flush=True)
    print(f"train {len(tr_store.spans)} games / {tr_store.n_steps} steps ({tr_store.nbytes / 1e9:.2f} GB on {dev}); "
          f"held-out {len(he_store.spans)} games / {he_store.n_steps} steps; loaded in {time.time() - t_load:.0f}s",
          flush=True)
    fz = Featurizer(dev)
    kind = torch.from_numpy(KIND).to(dev)
    global BF16
    BF16 = args.bf16
    EFFECT_WEIGHT.update({k: float(v) for k, v in (kv.split("=") for kv in args.effect_weights.split(",") if kv)})
    model = TokenWM(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate).to(dev)
    model.set_change_prior(stats.change_rate)
    # Everything needed to load and roll out this model elsewhere (e.g. the viewer).
    torch.save(dict(stats=vars(stats), config=dict(d=args.d, n_heads=args.heads, n_layers=args.layers, gate=args.gate),
                    train_games=args.train_games, held_games=args.held_games), out / "stats.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="moba-wm", name=out.name, config=vars(args) | dict(params=n_params,
                         train_steps=tr_store.n_steps, held_steps=he_store.n_steps))

    logf = open(out / "log.jsonl", "w")

    def emit(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if run:
            run.log({k: v for k, v in rec.items() if k != "it"}, step=rec["it"])

    def run_val(it):
        vl, vp = validate(model, fz, he_store, stats, kind, dev, args)
        sk = skill_eval(model, fz, he_store, stats, kind, dev, args)
        emit(dict(it=it, **{"val/loss": vl}, **{f"val/{k}": v for k, v in vp.items()},
                  **{f"skill/{k}": v for k, v in sk.items()}))
        kinds = "  ".join(f"{n} {vp[n + '/total']:.3f}" for n in KIND_NAMES)
        print(f"VAL it {it:6d}  loss {vl:.4f}  | {kinds}", flush=True)
        hero = "  ".join(f"{e} {sk.get(f'hero/{e}/better_than_copy_pct', float('nan')):+.0f}%/"
                         f"{sk.get(f'hero/{e}/action_dependence_pct', float('nan')):+.0f}%"
                         for e in ("walking", "attack", "q", "w", "e", "mana", "health"))
        print(f"    hero better-than-copy / action-dependence: {hero}  | deaths recall {sk['deaths/recall_pct']:.0f}%", flush=True)
        return vl

    # Step 0: the zero-initialized model predicts "no change", so this is the persistence baseline.
    base = run_val(0)
    if run:
        run.summary["val/persistence_loss"] = base
    t0 = time.time()
    it, losses, nan_count = 0, [], 0
    part_acc, part_n = {}, 0
    while time.time() - t0 < args.minutes * 60:
        lr = args.lr * min(1.0, (it + 1) / 100)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = tr_store.sample(args.batch, args.window, gen, event_frac=args.event_frac, burn=args.burn)
        loss, parts, _, _ = run_window(model, fz, batch, stats, kind, args.burn, dev)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if not math.isfinite(loss.item()) or not math.isfinite(gn):
            nan_count += 1
            continue
        opt.step()
        it += 1
        losses.append(loss.item())
        for k, v in parts.items():
            part_acc[k] = part_acc.get(k, 0.0) + float(v.detach())
        part_n += 1
        if it % args.log_every == 0:
            tp = summarize_parts({k: v / part_n for k, v in part_acc.items()})
            part_acc, part_n = {}, 0
            rec = dict(it=it, **{"train/loss": float(np.mean(losses[-args.log_every:]))},
                       **{f"train/{k}": v for k, v in tp.items()}, grad_norm=gn,
                       elapsed_s=round(time.time() - t0, 1), nan=nan_count,
                       it_per_s=round(it / (time.time() - t0), 2))
            emit(rec)
            kinds = "  ".join(f"{n} {tp[n + '/total']:.3f}" for n in KIND_NAMES)
            print(f"it {it:6d}  loss {rec['train/loss']:.4f}  | {kinds}  | grad {gn:.2f}  "
                  f"{rec['it_per_s']} it/s  {rec['elapsed_s']:.0f}s", flush=True)
        if it % args.val_every == 0:
            run_val(it)
            torch.save(model.state_dict(), out / "model.pt")
    run_val(it)
    torch.save(model.state_dict(), out / "model.pt")

    # loss slope over the last quarter (per 1000 iterations, in log-loss)
    n = len(losses)
    q = np.array(losses[3 * n // 4:])
    slope = float(np.polyfit(np.arange(len(q)), np.log(q), 1)[0] * 1000) if len(q) > 10 else float("nan")

    print("evaluating ...", flush=True)
    model.eval()
    ev = dict(n_batches=args.eval_batches)
    report = dict(
        args=vars(args), params=n_params, iterations=it, nan_skipped=nan_count,
        train_steps=tr_store.n_steps, held_steps=he_store.n_steps,
        loss_first_100=float(np.mean(losses[:100])), loss_last_100=float(np.mean(losses[-100:])),
        log_loss_slope_per_1k_it_last_quarter=slope,
        onestep_train=eval_onestep(model, fz, tr_store, stats, kind, dev, egen, **ev),
        onestep_heldout=eval_onestep(model, fz, he_store, stats, kind, dev, egen, **ev),
        onestep_heldout_shuffled_actions=eval_onestep(model, fz, he_store, stats, kind, dev, egen, shuffle=True, **ev),
        rollout_train=eval_rollout(model, fz, tr_store, stats, kind, dev, egen, n=args.rollouts),
        rollout_heldout=eval_rollout(model, fz, he_store, stats, kind, dev, egen, n=args.rollouts),
    )
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if run:
        log_report(run, report)
        run.finish()
    print(f"wrote {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
