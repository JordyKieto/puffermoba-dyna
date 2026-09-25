"""Side-by-side viewer: the real MOBA game vs the token world model's open-loop dream.

    .venv/bin/python tools/moba_viz.py --run runs/token_wm/smoke2 \
        --dense data/moba/mix100/dense/games.0000091.mbs --game 0 --start 500 --horizon 300 --burn 8

    # headless: render one frame to a PNG and exit
    .venv/bin/python tools/moba_viz.py --run runs/token_wm/local_check --screenshot out.png --frame 16

Left panel is the recorded game from record `start` on. Right panel starts
from the same real state and then only ever sees its own predictions, driven
by the REAL recorded actions of all 10 heroes (as eval_rollout in
mobawm/wm/train.py). Before `start`, `burn` real states warm up the model's
per-entity GRU memory. Everything is precomputed, so scrubbing is instant.

Controls:
    SPACE           play / pause
    RIGHT / LEFT    step +1 / -1 frame      (SHIFT: +/-10)
    HOME / END      jump to first / last frame
    UP / DOWN, + -  playback speed
    G               ghost: true positions as outlines on the model panel,
                    with a line from each predicted hero to its true position
    Q / ESC         quit

Symbols: heroes are circles with class initial (S support, A assassin, B burst,
T tank, C carry) and level; creeps small squares; neutrals grey diamonds;
towers squares, the two ancients larger with a white outline. Cyan = Radiant,
red = Dire.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobawm import shards as S  # noqa: E402
from mobawm.wm.data import (Featurizer, Stats, Store, apply_prediction, fit_stats,  # noqa: E402
                            load_games, load_walls)
from mobawm.wm.model import TokenWM  # noqa: E402

F = S.F
CELL = 5                       # px per map cell: two 640 px panels fit a ~1400 px window
PANEL = 128 * CELL
PAD, TITLE_H, HUD_H = 16, 28, 150
W, H = 2 * PANEL + 3 * PAD, TITLE_H + PANEL + HUD_H + PAD
SPEEDS = (1, 2, 5, 10, 20, 30, 60)   # frames per second
CLASSES = "SABTC"                    # hero index % 5: support, assassin, burst, tank, carry

BG, GRID, WALL, TEXT, DIM = (18, 18, 36), (30, 30, 58), (58, 58, 84), (230, 230, 240), (140, 140, 165)
TEAM = {0: (0, 220, 255), 1: (255, 70, 50), 2: (150, 150, 150)}
CREEP = {0: (0, 150, 190), 1: (200, 90, 40), 2: (150, 150, 150)}
HP_BG, HP_FG = (120, 20, 20), (40, 220, 60)


# --- model + stats ---------------------------------------------------------

def parse_games(spec) -> set[int]:
    out = set()
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def load_model(run: Path, dense: Path, dev):
    """TokenWM + Stats for a run dir. Newer runs ship stats.pt; for older ones
    the stats are refit here, because rollouts need the same scales and spawn
    lookups the model was trained with (approximately: refit on few games)."""
    report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
    args = report.get("args", {})
    train_games = args.get("train_games")
    if (run / "stats.pt").exists():
        d = torch.load(run / "stats.pt", map_location="cpu", weights_only=False)
        # stats.pt from before the change gate has no change_rate; it is only used by gated models
        d["stats"].setdefault("change_rate", torch.zeros_like(d["stats"]["reg_scale"]))
        stats, config = Stats(**d["stats"]), d["config"]
        train_games = d.get("train_games", train_games)
    else:
        config = dict(d=args.get("d", 128), n_heads=args.get("heads", 4), n_layers=args.get("layers", 2))
        # the run's own training games that live in the first file, else games 0-12
        idx = (parse_games(train_games) if train_games else set()) & set(range(13)) or set(range(13))
        src = dense.parent / "games.0000000.mbs"
        print(f"WARNING: {run}/stats.pt missing; refitting stats locally on games {sorted(idx)} of {src}",
              flush=True)
        stats = fit_stats(Store(load_games([src], idx, log=lambda *_: None), torch.device("cpu")))
    model = TokenWM(**config)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))   # includes change_prior if gated
    print(f"model {config}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    return model.to(dev).eval(), stats, (parse_games(train_games) if train_games else set())


@torch.no_grad()
def rollout(model, stats, recs, start, H, burn, dev):
    """Open loop from real record `start`: (H+1, 206, 26), frame 0 = the real start state."""
    fz = Featurizer(dev)
    st = recs["state"]
    ent = torch.from_numpy(np.ascontiguousarray(st["entity"], np.float32)).to(dev)
    tick = torch.from_numpy(st["tick"].astype(np.int64)).to(dev)
    cidx = torch.from_numpy(st["creep_idx"].astype(np.int64)).to(dev)
    act = torch.from_numpy(recs["actions"].astype(np.int64)).to(dev)
    h = torch.zeros(1, S.N_ENTITIES, model.d, device=dev)
    for t in range(start - burn, start):
        _, h = model.step(fz(ent[t:t + 1], tick[t:t + 1], cidx[t:t + 1]), act[t:t + 1], h)
    cur, ct, cc = ent[start:start + 1].clone(), tick[start:start + 1].clone(), cidx[start:start + 1].clone()
    out = [cur[0].cpu().numpy()]
    t0 = time.time()
    for k in range(1, H + 1):
        pred, h = model.step(fz(cur, ct, cc), act[start + k - 1:start + k], h)
        cur, ct, cc = apply_prediction(cur, ct, cc, pred, stats, fz)
        out.append(cur[0].cpu().numpy())
        if k % 50 == 0 or k == H:
            print(f"  rollout {k}/{H} ({time.time() - t0:.1f}s)", flush=True)
    return np.stack(out)


def metrics(s, t):
    """Divergence of state(s) s from truth t, (N,206,26) -> dict of (N,) arrays."""
    pos = (np.abs(s[:, :10, F["x"]] - t[:, :10, F["x"]]) + np.abs(s[:, :10, F["y"]] - t[:, :10, F["y"]])).mean(1)
    hp = np.abs(s[:, :10, F["health"]] - t[:, :10, F["health"]]).mean(1)
    alive = ((s[:, 10:110, F["alive"]] > 0) == (t[:, 10:110, F["alive"]] > 0)).mean(1) * 100
    tower = np.abs(s[:, 182:, F["health"]] - t[:, 182:, F["health"]]).mean(1)
    return {"hero pos L1 (cells)": pos, "hero hp MAE": hp, "creep alive agree %": alive, "tower hp MAE": tower}


# --- drawing ----------------------------------------------------------------

def map_surface(pg, walls):
    surf = pg.Surface((PANEL, PANEL))
    surf.fill(BG)
    for i in range(0, 129, 8):   # a light 8-cell grid for reading distances
        pg.draw.line(surf, GRID, (i * CELL, 0), (i * CELL, PANEL))
        pg.draw.line(surf, GRID, (0, i * CELL), (PANEL, i * CELL))
    for y, x in zip(*np.nonzero(walls)):     # walls[row=y, col=x]
        surf.fill(WALL, (x * CELL, y * CELL, CELL, CELL))
    return surf


def hp_bar(pg, surf, cx, top, w, frac):
    frac = float(np.clip(frac, 0, 1))
    pg.draw.rect(surf, HP_BG, (cx - w // 2, top, w, 3))
    pg.draw.rect(surf, HP_FG, (cx - w // 2, top, round(w * frac), 3))


def team_of(e, i):
    return int(np.clip(e[i, F["team"]], 0, 2))


def draw_entities(pg, surf, e, fonts):
    px = lambda i: (int(e[i, F["x"]] * CELL), int(e[i, F["y"]] * CELL))
    alive = e[:, F["alive"]] > 0
    frac = e[:, F["health"]] / np.maximum(e[:, F["max_health"]], 1)
    for i in np.nonzero(alive[110:182])[0] + 110:
        x, y = px(i)
        pg.draw.polygon(surf, TEAM[2], [(x, y - 3), (x + 3, y), (x, y + 3), (x - 3, y)])
    for i in np.nonzero(alive[10:110])[0] + 10:
        x, y = px(i)
        pg.draw.rect(surf, CREEP[team_of(e, i)], (x - 2, y - 2, 4, 4))
    for i in np.nonzero(alive[182:206])[0] + 182:
        x, y = px(i)
        r = 9 if i >= 204 else 6
        pg.draw.rect(surf, TEAM[team_of(e, i)], (x - r, y - r, 2 * r, 2 * r))
        if i >= 204:
            pg.draw.rect(surf, (255, 255, 255), (x - r - 2, y - r - 2, 2 * r + 4, 2 * r + 4), 2)
        hp_bar(pg, surf, x, y - r - 6, 2 * r + 4, frac[i])
    for i in np.nonzero(alive[:10])[0]:
        x, y = px(i)
        pg.draw.circle(surf, TEAM[team_of(e, i)], (x, y), 7)
        pg.draw.circle(surf, (255, 255, 255), (x, y), 7, 1)
        lab = fonts["hero"].render(CLASSES[i % 5], True, (0, 0, 0))
        surf.blit(lab, lab.get_rect(center=(x, y + 1)))
        lv = fonts["tiny"].render(str(int(e[i, F["level"]])), True, TEXT)
        surf.blit(lv, (x + 8, y - 2))
        hp_bar(pg, surf, x, y - 12, 16, frac[i])


def draw_ghost(pg, surf, pred, truth):
    """True positions as outlines over the prediction, and each hero's error vector."""
    alive = truth[:, F["alive"]] > 0
    for i in np.nonzero(alive)[0]:
        x, y = int(truth[i, F["x"]] * CELL), int(truth[i, F["y"]] * CELL)
        c = TEAM[team_of(truth, i)]
        if i < 10:
            pg.draw.circle(surf, c, (x, y), 9, 1)
            if pred[i, F["alive"]] > 0:
                pg.draw.line(surf, (255, 255, 120),
                             (int(pred[i, F["x"]] * CELL), int(pred[i, F["y"]] * CELL)), (x, y))
        elif i < 110:
            pg.draw.rect(surf, c, (x - 3, y - 3, 6, 6), 1)


# --- app --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/token_wm/smoke2")
    ap.add_argument("--dense", default="data/moba/mix100/dense/games.0000091.mbs")
    ap.add_argument("--game", type=int, default=0, help="index of the game within the dense file")
    ap.add_argument("--start", type=int, default=500, help="record (tick index) the dream starts from")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--burn", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--screenshot", help="render --frame to this PNG headless and exit")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--ghost", action="store_true", help="start with the ghost overlay on")
    args = ap.parse_args()
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg

    dev = torch.device(args.device)
    run, dense = Path(args.run), Path(args.dense)
    model, stats, train_games = load_model(run, dense, dev)
    d = S.load_dense(dense)
    recs = d.load_game(args.game)
    gi = int(recs[0]["game_index"])
    ctrl = f'{S.CONTROLLERS[int(recs[0]["radiant_ctrl"])]} vs {S.CONTROLLERS[int(recs[0]["dire_ctrl"])]}'
    n_steps = len(recs) - 1                         # the last record is the terminal state
    start = int(np.clip(args.start, 0, n_steps - 1))
    burn = min(args.burn, start)
    Hz = max(1, min(args.horizon, n_steps - start))
    if (start, burn, Hz) != (args.start, args.burn, args.horizon):
        print(f"clamped to start {start}, burn {burn}, horizon {Hz} (game has {n_steps} steps)", flush=True)
    split = "TRAIN game" if gi in train_games else "held-out"
    print(f"game {gi} ({ctrl}, {split}), {n_steps} steps; rolling out {Hz} from record {start} "
          f"after {burn} burn-in ...", flush=True)

    truth = np.ascontiguousarray(recs["state"]["entity"][start:start + Hz + 1], np.float32)
    ticks = recs["state"]["tick"][start:start + Hz + 1]
    pred = rollout(model, stats, recs, start, Hz, burn, dev)
    persist = np.broadcast_to(truth[:1], truth.shape)
    m_model, m_persist = metrics(pred, truth), metrics(persist, truth)
    for k in sorted({1, 4, 16, 64, Hz}):
        if k <= Hz:
            print(f"  k={k:4d}  " + "  ".join(f"{n}: {m_model[n][k]:.2f} / {m_persist[n][k]:.2f}" for n in m_model),
                  "(model / persistence)", flush=True)
    assert np.array_equal(pred[0], truth[0])

    pg.init()
    screen = pg.display.set_mode((W, H)) if not args.screenshot else pg.Surface((W, H))
    if not args.screenshot:
        pg.display.set_caption(f"MOBA world model: {run.name}, game {gi}")
    fonts = dict(hero=pg.font.Font(None, 15), tiny=pg.font.Font(None, 13),
                 ui=pg.font.Font(None, 20), title=pg.font.Font(None, 24))
    base = map_surface(pg, load_walls())
    state = dict(k=int(np.clip(args.frame, 0, Hz)), playing=False, speed=3, ghost=args.ghost, acc=0.0)

    def text(s, pos, font="ui", color=TEXT):
        screen.blit(fonts[font].render(s, True, color), pos)

    def render():
        k = state["k"]
        screen.fill((10, 10, 22))
        for p, (title, frame) in enumerate((("TRUTH", truth[k]),
                                            (f"MODEL (open loop from tick {int(ticks[0])})", pred[k]))):
            x0 = PAD + p * (PANEL + PAD)
            panel = base.copy()
            draw_entities(pg, panel, frame, fonts)
            if p == 1 and state["ghost"]:
                draw_ghost(pg, panel, pred[k], truth[k])
            screen.blit(panel, (x0, TITLE_H))
            text(title, (x0, 6), "title")
        y = TITLE_H + PANEL + 10
        status = "PLAYING" if state["playing"] else "PAUSED"
        text(f"game {gi} ({ctrl}, {split})   run {run.name}   tick {int(ticks[k])}   step {k}/{Hz}   "
             f"{status}   {SPEEDS[state['speed']]} fps   ghost {'on' if state['ghost'] else 'off'}   "
             f"burn-in {burn}", (PAD, y))
        y += 26
        text("divergence from truth at this step", (PAD, y), color=DIM)
        text("model", (PAD + 260, y), color=DIM)
        text("persistence (start state)", (PAD + 360, y), color=DIM)
        for n in m_model:
            y += 20
            better = m_model[n][k] >= m_persist[n][k] if "%" in n else m_model[n][k] <= m_persist[n][k]
            text(n, (PAD, y))
            text(f"{m_model[n][k]:8.2f}", (PAD + 250, y), color=(120, 230, 120) if better else (240, 130, 110))
            text(f"{m_persist[n][k]:8.2f}", (PAD + 380, y))
        text("SPACE play/pause   LEFT/RIGHT step (SHIFT x10)   HOME/END   UP/DOWN speed   "
             "G ghost   Q quit", (PAD, H - PAD - 8), color=DIM)

    if args.screenshot:
        render()
        pg.image.save(screen, args.screenshot)
        print(f"wrote {args.screenshot} (frame {state['k']})")
        return

    clock = pg.time.Clock()
    while True:
        dt = clock.tick(60) / 1000
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return
            if ev.type != pg.KEYDOWN:
                continue
            step = 10 if ev.mod & pg.KMOD_SHIFT else 1
            key = ev.key
            if key in (pg.K_q, pg.K_ESCAPE):
                return
            elif key == pg.K_SPACE:
                state["playing"] = not state["playing"]
                if state["playing"] and state["k"] == Hz:   # replay from the start
                    state["k"] = 0
            elif key == pg.K_RIGHT:
                state["k"] = min(Hz, state["k"] + step)
            elif key == pg.K_LEFT:
                state["k"] = max(0, state["k"] - step)
            elif key == pg.K_HOME:
                state["k"] = 0
            elif key == pg.K_END:
                state["k"] = Hz
            elif key in (pg.K_UP, pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                state["speed"] = min(len(SPEEDS) - 1, state["speed"] + 1)
            elif key in (pg.K_DOWN, pg.K_MINUS, pg.K_KP_MINUS):
                state["speed"] = max(0, state["speed"] - 1)
            elif key == pg.K_g:
                state["ghost"] = not state["ghost"]
        if state["playing"]:
            state["acc"] += dt * SPEEDS[state["speed"]]
            adv, state["acc"] = int(state["acc"]), state["acc"] % 1
            state["k"] = min(Hz, state["k"] + adv)
            if state["k"] == Hz:
                state["playing"] = False
        render()
        pg.display.flip()


if __name__ == "__main__":
    main()
