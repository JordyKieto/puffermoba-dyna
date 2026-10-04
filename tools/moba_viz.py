"""Side-by-side viewer: the real MOBA game vs the token world model's open-loop dream.

    .venv/bin/python tools/moba_viz.py --run runs/token_wm/e4_baseline_v2_1h \
        --dense data/moba/mix1k/dense/games.0000900.mbs --game 0 --start 500 --horizon 300 --burn 8

    # headless (no window): query or render, see docs/moba/VIEWER.md
    .venv/bin/python tools/moba_viz.py --frame 41 --entity 1     # real vs model, every field
    .venv/bin/python tools/moba_viz.py --casts                   # Q/W/E casts, real vs model
    .venv/bin/python tools/moba_viz.py --frame 41 --screenshot out.png --mouse 354 331

    # from Python: Replay is the data, Viewer only draws it
    rp = Replay(run, dense, game=0, start=500, horizon=300); rp.entity(1, 41); rp.cast_events()

Left panel is the recorded game from record `start` on. Right panel starts
from the same real state and then only ever sees its own predictions, driven
by the REAL recorded actions of all 10 heroes (as eval_rollout in
mobawm/wm/train.py). Before `start`, `burn` real states warm up the model's
per-entity GRU memory. Everything is precomputed, so scrubbing is instant.

Controls:
    SPACE           play / pause
    RIGHT / LEFT    step +1 / -1 frame      (SHIFT: +/-10)
    R, HOME / END   reset to the first frame / jump to the last
    N / P           next / previous game in the file (re-runs the model)
    mouse           buttons below the panels; click the timeline to jump;
                    hover an entity while paused: real vs model stats
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
import dataclasses
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
SLOT = {"Q": (255, 220, 0), "W": (0, 255, 150), "E": (255, 60, 255)}   # ability flash colours
CAST_JUMP = 5                        # min cooldown increase that counts as a cast
FLASH = 3                            # frames a cast ring stays visible
DEATH_FLASH = 12                     # frames a death marker flashes


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
        # stats.pt from a newer or experimental working tree may carry fields this Stats lacks
        known = {f.name for f in dataclasses.fields(Stats)}
        extra = {k: v for k, v in d["stats"].items() if k not in known}
        if any(bool(v) for v in extra.values() if not torch.is_tensor(v)):
            print(f"WARNING: {run}/stats.pt sets {extra}, which this code does not implement", flush=True)
        stats, config = Stats(**{k: v for k, v in d["stats"].items() if k in known}), d["config"]
        train_games = d.get("train_games", train_games)
    else:
        config = dict(d=args.get("d", 128), n_heads=args.get("heads", 4), n_layers=args.get("layers", 2))
        # the run's own training games that live in the first file, else games 0-12
        idx = (parse_games(train_games) if train_games else set()) & set(range(13)) or set(range(13))
        src = dense.parent / "games.0000000.mbs"
        print(f"WARNING: {run}/stats.pt missing; refitting stats locally on games {sorted(idx)} of {src}",
              flush=True)
        stats = fit_stats(Store(load_games([src], idx, log=lambda *_: None), torch.device("cpu")))
    if config.get("list_state"):                 # the stored target lists are part of the state
        import mobawm.wm.data as _wdl
        _wdl.enable_list_state()
    if config.get("attack_mode"):                  # the featurizer must build the same attack-graph inputs
        import mobawm.wm.data as _wd
        _wd.ATTACK_MODE = config["attack_mode"]
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


def casts(frames):
    """(N, 10, 3) bool: hero i cast Q/W/E between frame k-1 and k.

    A cast resets that ability's cooldown, so its timer jumps up by more than
    CAST_JUMP (the model's predicted timers wobble by a few ticks; a real reset is
    a full cooldown). The tank's Q
    has no cooldown; it is detected by its mana cost (5, net of the +2 regen).
    A hero that died that tick (teleport > 12 cells) is not counted: respawn
    resets timers down, not up."""
    out = np.zeros((len(frames), 10, 3), bool)
    if len(frames) < 2:
        return out
    a, b = frames[:-1, :10], frames[1:, :10]
    died = (np.abs(b[..., F["x"]] - a[..., F["x"]]) + np.abs(b[..., F["y"]] - a[..., F["y"]])) > 12
    for j, f in enumerate(("q_timer", "w_timer", "e_timer")):
        out[1:, :, j] = (b[..., F[f]] - a[..., F[f]] > CAST_JUMP) & ~died
    tank = np.arange(10) % 5 == 3
    spent = (b[..., F["mana"]] - a[..., F["mana"]]) <= -2.5
    out[1:, tank, 0] |= (spent & ~out[1:, :, 1] & ~out[1:, :, 2] & ~died)[:, tank]
    return out


def deaths(frames):
    """(N, 206) bool: entity i died between frame k-1 and k. Heroes and towers only.

    A hero at 0 health respawns in the same tick (moba.h: spawn_player right in
    the damage code), so a hero death is a jump of more than 12 cells with health
    reset to full (the assassin's blink jumps but keeps its health). A tower
    death is alive going 1 -> 0."""
    out = np.zeros(frames.shape[:2], bool)
    if len(frames) < 2:
        return out
    a, b = frames[:-1], frames[1:]
    jump = (np.abs(b[:, :10, F["x"]] - a[:, :10, F["x"]]) + np.abs(b[:, :10, F["y"]] - a[:, :10, F["y"]])) > 12
    reset = (b[:, :10, F["health"]] >= b[:, :10, F["max_health"]] - 0.5) & \
            (a[:, :10, F["health"]] < a[:, :10, F["max_health"]] - 0.5)
    out[1:, :10] = jump & reset
    out[1:, 182:] = (a[:, 182:, F["alive"]] > 0) & (b[:, 182:, F["alive"]] <= 0)
    return out


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


def draw_casts(pg, surf, frames, cast, k, fonts):
    """A ring per cast, in the ability's colour, fading over FLASH frames."""
    for age in range(FLASH):
        kk = k - age
        if kk < 1:
            break
        e = frames[k]
        for i, j in zip(*np.nonzero(cast[kk])):
            key = "QWE"[j]
            x, y = int(e[i, F["x"]] * CELL), int(e[i, F["y"]] * CELL)
            c = tuple(int(v * (1 - 0.3 * age)) for v in SLOT[key])
            pg.draw.circle(surf, c, (x, y), 11 + 3 * age, 3 - age if age < 3 else 1)
            if age == 0:
                lab = fonts["cast"].render(key, True, c)
                surf.blit(lab, lab.get_rect(center=(x + 14, y - 14)))


def draw_deaths(pg, surf, frames, died, k, fonts):
    """A flashing red/white X where the entity died (its last position), for DEATH_FLASH frames."""
    for age in range(DEATH_FLASH):
        kk = k - age
        if kk < 1:
            break
        for i in np.nonzero(died[kk])[0]:
            x, y = int(frames[kk - 1][i, F["x"]] * CELL), int(frames[kk - 1][i, F["y"]] * CELL)
            c = (255, 40, 40) if (k // 2) % 2 == 0 else (255, 255, 255)   # flash every 2 frames
            r = 16 - age // 2
            pg.draw.circle(surf, c, (x, y), r, 3)
            pg.draw.line(surf, c, (x - r + 4, y - r + 4), (x + r - 4, y + r - 4), 4)
            pg.draw.line(surf, c, (x - r + 4, y + r - 4), (x + r - 4, y - r + 4), 4)
            lab = fonts["cast"].render("DEATH", True, c)
            surf.blit(lab, lab.get_rect(center=(x, y - r - 8)))


def draw_presses(pg, surf, e, act, fonts):
    """Which ability keys each hero pressed this tick (same on both panels)."""
    if act is None:
        return
    for i in range(10):
        if e[i, F["alive"]] <= 0:
            continue
        x, y = int(e[i, F["x"]] * CELL), int(e[i, F["y"]] * CELL)
        for j, key in enumerate("QWE"):
            if act[i, 3 + j]:
                lab = fonts["tiny"].render(key, True, SLOT[key])
                surf.blit(lab, (x - 9 + 7 * j, y + 9))


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

# --- headless API -------------------------------------------------------------

HIDDEN = ("team", "entity_type", "grid_id", "lane")     # constant per entity, left out of comparisons


def entity_name(e, i):
    team = ("Radiant", "Dire", "neutral")[team_of(e, i)]
    if i < 10:
        return f"hero {i}  {('support', 'assassin', 'burst', 'tank', 'carry')[i % 5]}  {team}"
    if i < 110:
        return f"creep {i}  {team}"
    if i < 182:
        return f"neutral {i}"
    return f"{'ancient' if i >= 204 else 'tower'} {i}  {team}"


class Replay:
    """One game rolled out by one model: everything the viewer shows, queryable without a window.

        rp = Replay("runs/token_wm/e4_baseline_v2_1h", "data/moba/mix1k/dense/games.0000900.mbs")
        rp.entity(1, 41)          # every field of hero 1 at step 41: real, model, diff
        rp.cast_events()          # every Q/W/E cast, real and/or predicted
        rp.death_events()         # every hero / tower death, real and/or predicted
        rp.divergence(64)         # the HUD's divergence table at step 64
        rp.load(3)                # another game in the same file, same model

    Step k is frame k of the window: step 0 is the real state at record `start`,
    and step k is the model's k-th open-loop prediction, next to the real record start+k.
    Arrays: truth, pred (Hz+1, 206, 26); acts (Hz+1, 10, 6), the action taken from step k;
    cast_truth, cast_pred (Hz+1, 10, 3) bool; death_truth, death_pred (Hz+1, 206) bool.
    """

    def __init__(self, run, dense, game=0, start=500, horizon=300, burn=8, device="cpu"):
        self.run, self.dense, self.dev = Path(run), Path(dense), torch.device(device)
        self.model, self.stats, self.train_games = load_model(self.run, self.dense, self.dev)
        self.file = S.load_dense(self.dense)
        self.n_games = len(self.file.games)
        self.want = (start, horizon, burn)
        self.load(game)

    def load(self, game):
        recs = self.file.load_game(game)
        want_start, want_h, want_burn = self.want
        self.game, self.gi = game, int(recs[0]["game_index"])
        self.ctrl = (f'{S.CONTROLLERS[int(recs[0]["radiant_ctrl"])]} vs '
                     f'{S.CONTROLLERS[int(recs[0]["dire_ctrl"])]}')
        self.split = "TRAIN game" if self.gi in self.train_games else "held-out"
        self.n_steps = len(recs) - 1                     # the last record is the terminal state
        self.start = int(np.clip(want_start, 0, self.n_steps - 1))
        self.burn = min(want_burn, self.start)
        self.Hz = max(1, min(want_h, self.n_steps - self.start))
        if (self.start, self.Hz, self.burn) != self.want:
            print(f"clamped to start {self.start}, burn {self.burn}, horizon {self.Hz} "
                  f"(game has {self.n_steps} steps)", flush=True)
        print(f"game {self.gi} ({self.ctrl}, {self.split}), {self.n_steps} steps; rolling out {self.Hz} "
              f"from record {self.start} after {self.burn} burn-in ...", flush=True)
        sl = slice(self.start, self.start + self.Hz + 1)
        self.truth = np.ascontiguousarray(recs["state"]["entity"][sl], np.float32)
        self.pred = rollout(self.model, self.stats, recs, self.start, self.Hz, self.burn, self.dev)
        assert np.array_equal(self.pred[0], self.truth[0])
        self.ticks = recs["state"]["tick"][sl]
        self.last_tick = int(recs["state"]["tick"][-1])
        self.acts = recs["actions"][sl]
        self.cast_truth, self.cast_pred = casts(self.truth), casts(self.pred)
        self.death_truth, self.death_pred = deaths(self.truth), deaths(self.pred)
        self.m_model = metrics(self.pred, self.truth)
        self.m_persist = metrics(np.broadcast_to(self.truth[:1], self.truth.shape), self.truth)
        return self

    def divergence(self, k):
        """{metric: (model, persistence)} at step k: the HUD table."""
        return {n: (float(self.m_model[n][k]), float(self.m_persist[n][k])) for n in self.m_model}

    def entity(self, i, k):
        """Every field of entity i at step k, real vs model: the hover tooltip."""
        t, m = self.truth[k], self.pred[k]
        return dict(entity=i, name=entity_name(t, i), step=k, tick=int(self.ticks[k]),
                    fields={n: (float(t[i, F[n]]), float(m[i, F[n]]), float(m[i, F[n]] - t[i, F[n]]))
                            for n in F if n not in HIDDEN},
                    pos_err=float(abs(m[i, F["x"]] - t[i, F["x"]]) + abs(m[i, F["y"]] - t[i, F["y"]])),
                    pressed="".join(key for j, key in enumerate("QWE") if i < 10 and k < self.Hz and self.acts[k, i, 3 + j]))

    def entity_at(self, panel, x, y, k, radius=2.0):
        """The living entity nearest map cell (x, y) at step k on panel 0 (real) or 1 (model), or None."""
        e = (self.truth, self.pred)[panel][k]
        live = np.nonzero(e[:, F["alive"]] > 0)[0]
        if not len(live):
            return None
        dist = np.hypot(e[live, F["x"]] - x, e[live, F["y"]] - y) + (live >= 10) * 0.8   # prefer heroes
        j = int(np.argmin(dist))
        return int(live[j]) if dist[j] <= radius else None

    def cast_events(self):
        """[{step, tick, hero, ability, real, model}] for every cast in either panel."""
        out = []
        for k, i, j in zip(*np.nonzero(self.cast_truth | self.cast_pred)):
            out.append(dict(step=int(k), tick=int(self.ticks[k]), hero=int(i), ability="QWE"[j],
                            real=bool(self.cast_truth[k, i, j]), model=bool(self.cast_pred[k, i, j])))
        return out

    def death_events(self):
        """[{step, tick, entity, name, real, model}] for every hero or tower death in either panel."""
        out = []
        for k, i in zip(*np.nonzero(self.death_truth | self.death_pred)):
            out.append(dict(step=int(k), tick=int(self.ticks[k]), entity=int(i),
                            name=entity_name(self.truth[k - 1], i),
                            real=bool(self.death_truth[k, i]), model=bool(self.death_pred[k, i])))
        return out

    def entity_table(self, i, k):
        """entity() as the text the tooltip shows."""
        d = self.entity(i, k)
        lines = [f"{d['name']}   step {k}  tick {d['tick']}" + (f"   pressed {d['pressed']}" if d["pressed"] else ""),
                 f"{'field':<18}{'real':>9}{'model':>9}{'diff':>9}"]
        lines += [f"{n:<18}{a:>9.1f}{b:>9.1f}{c:>+9.1f}" for n, (a, b, c) in d["fields"].items()]
        lines.append(f"position error (L1 cells): {d['pos_err']:.1f}")
        return "\n".join(lines)


# --- window -------------------------------------------------------------------

class Viewer:
    """The pygame window over a Replay. headless=True draws to an off-screen surface (screenshot())."""

    BUTTONS = [("|< reset", "reset"), ("< step", "back"), ("play/pause", "play"), ("step >", "fwd"),
               ("< game", "prev_game"), ("game >", "next_game")]

    def __init__(self, rp: Replay, headless=False, ghost=False, frame=0):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame as pg
        self.pg, self.rp, self.headless = pg, rp, headless
        pg.init()
        self.screen = pg.Surface((W, H)) if headless else pg.display.set_mode((W, H))
        self.fonts = dict(hero=pg.font.Font(None, 15), tiny=pg.font.Font(None, 13), cast=pg.font.Font(None, 18),
                          ui=pg.font.Font(None, 20), title=pg.font.Font(None, 24), clock=pg.font.Font(None, 30),
                          mono=pg.font.SysFont("menlo,dejavusansmono,monospace", 12))
        self.base = map_surface(pg, load_walls())
        self.state = dict(k=int(np.clip(frame, 0, rp.Hz)), playing=False, speed=3, ghost=ghost, acc=0.0, mouse=None)
        self.hud_y = TITLE_H + PANEL
        self.bar = pg.Rect(PAD + 620, self.hud_y + 118, W - 2 * PAD - 620, 10)   # whole-game timeline
        self.rects, x = {}, PAD + 620
        for label, act in self.BUTTONS:
            w = self.fonts["ui"].size(label)[0] + 16
            self.rects[act] = pg.Rect(x, self.hud_y + 84, w, 24)
            x += w + 8
        self.caption()

    def caption(self):
        if not self.headless:
            self.pg.display.set_caption(f"MOBA world model: {self.rp.run.name}, game {self.rp.gi}")

    def text(self, s, pos, font="ui", color=TEXT):
        self.screen.blit(self.fonts[font].render(s, True, color), pos)

    def hovered(self):
        """(panel, entity) under the mouse, or None."""
        if self.state["mouse"] is None:
            return None
        mx, my = self.state["mouse"]
        for p in (0, 1):
            x0 = PAD + p * (PANEL + PAD)
            if x0 <= mx < x0 + PANEL and TITLE_H <= my < TITLE_H + PANEL:
                i = self.rp.entity_at(p, (mx - x0) / CELL, (my - TITLE_H) / CELL, self.state["k"])
                return None if i is None else (p, i)
        return None

    def draw_tooltip(self, i):
        pg, screen, font = self.pg, self.screen, self.fonts["mono"]
        d = self.rp.entity(i, self.state["k"])
        lines = [(d["name"], TEXT), (f"{'field':<18}{'real':>9}{'model':>9}{'diff':>9}", DIM)]
        for n, (a, b, diff) in d["fields"].items():
            c = TEXT if abs(diff) < 0.5 else (255, 200, 90) if abs(diff) < 5 else (240, 110, 100)
            lines.append((f"{n:<18}{a:>9.1f}{b:>9.1f}{diff:>+9.1f}", c))
        lines.append((f"position error (L1 cells): {d['pos_err']:.1f}",
                      (120, 230, 120) if d["pos_err"] < 2 else (240, 130, 110)))
        lh = 14
        w, h = max(font.size(s)[0] for s, _ in lines) + 16, lh * len(lines) + 12
        mx, my = self.state["mouse"]
        x, y = min(mx + 18, W - w - 4), min(my + 12, H - h - 4)
        box = pg.Surface((w, h), pg.SRCALPHA)
        box.fill((8, 8, 20, 235))
        screen.blit(box, (x, y))
        pg.draw.rect(screen, DIM, (x, y, w, h), 1)
        for n, (s, c) in enumerate(lines):
            screen.blit(font.render(s, True, c), (x + 8, y + 6 + n * lh))

    def render(self):
        pg, rp, st, text, hud_y = self.pg, self.rp, self.state, self.text, self.hud_y
        k, Hz = st["k"], rp.Hz
        hover = self.hovered() if not st["playing"] else None
        self.screen.fill((10, 10, 22))
        for p, (title, frames, cast) in enumerate((("TRUTH", rp.truth, rp.cast_truth),
                                                   (f"MODEL (open loop from tick {int(rp.ticks[0])})",
                                                    rp.pred, rp.cast_pred))):
            frame = frames[k]
            x0 = PAD + p * (PANEL + PAD)
            panel = self.base.copy()
            draw_entities(pg, panel, frame, self.fonts)
            draw_casts(pg, panel, frames, cast, k, self.fonts)
            draw_deaths(pg, panel, frames, (rp.death_truth, rp.death_pred)[p], k, self.fonts)
            draw_presses(pg, panel, frame, rp.acts[k] if k < Hz else None, self.fonts)
            if p == 1 and st["ghost"]:
                draw_ghost(pg, panel, rp.pred[k], rp.truth[k])
            if hover and frame[hover[1], F["alive"]] > 0:    # mark the hovered entity on both panels
                i = hover[1]
                pg.draw.circle(panel, (255, 255, 255), (int(frame[i, F["x"]] * CELL), int(frame[i, F["y"]] * CELL)),
                               13, 2)
            self.screen.blit(panel, (x0, TITLE_H))
            text(title, (x0, 6), "title")
        clock_s = f"GAME {rp.gi} ({rp.game + 1}/{rp.n_games})    TICK {int(rp.ticks[k])}  /  {rp.last_tick}"
        text(clock_s, (W - PAD - self.fonts["clock"].size(clock_s)[0], 3), "clock", (255, 230, 120))
        y = hud_y + 10
        text(f"game {rp.gi} ({rp.ctrl}, {rp.split})   run {rp.run.name}   step {k}/{Hz}   "
             f"{'PLAYING' if st['playing'] else 'PAUSED'}   {SPEEDS[st['speed']]} fps   "
             f"ghost {'on' if st['ghost'] else 'off'}   burn-in {rp.burn}", (PAD, y))
        y += 26
        text("divergence from truth at this step", (PAD, y), color=DIM)
        text("model", (PAD + 260, y), color=DIM)
        text("persistence (start state)", (PAD + 360, y), color=DIM)
        for n, (mm, mp) in rp.divergence(k).items():
            y += 20
            better = mm >= mp if "%" in n else mm <= mp
            text(n, (PAD, y))
            text(f"{mm:8.2f}", (PAD + 250, y), color=(120, 230, 120) if better else (240, 130, 110))
            text(f"{mp:8.2f}", (PAD + 380, y))
        n_t, n_p = int(rp.cast_truth[: k + 1].sum()), int(rp.cast_pred[: k + 1].sum())
        d_t, d_p = int(rp.death_truth[: k + 1].sum()), int(rp.death_pred[: k + 1].sum())
        text(f"casts so far: truth {n_t}   model {n_p}      deaths so far: truth {d_t}   model {d_p}",
             (PAD + 620, hud_y + 36))
        lx = PAD + 620
        for key, c in SLOT.items():
            pg.draw.circle(self.screen, c, (lx + 6, hud_y + 66), 6, 2)
            text(key, (lx + 16, hud_y + 58), color=c)
            lx += 40
        text("ring = cast fired   small letters under a hero = key pressed this tick", (PAD + 750, hud_y + 58), color=DIM)
        for label, act in self.BUTTONS:
            r = self.rects[act]
            pg.draw.rect(self.screen, (40, 40, 70), r, border_radius=4)
            pg.draw.rect(self.screen, DIM, r, 1, border_radius=4)
            lab = self.fonts["ui"].render(label, True, TEXT)
            self.screen.blit(lab, lab.get_rect(center=r.center))
        # timeline: the whole game, the rolled-out window highlighted, the cursor at the current step
        bar, n = self.bar, rp.n_steps
        pg.draw.rect(self.screen, (40, 40, 70), bar)
        pg.draw.rect(self.screen, (70, 90, 150), (bar.x + bar.w * rp.start / n, bar.y, max(2, bar.w * Hz / n), bar.h))
        pg.draw.rect(self.screen, (255, 230, 120), (bar.x + bar.w * (rp.start + k) / n - 1, bar.y - 3, 3, bar.h + 6))
        text("game timeline (blue = model window, click to jump)", (bar.x, bar.y + 12), "tiny", DIM)
        text("SPACE play/pause   LEFT/RIGHT step (SHIFT x10)   R/HOME reset   END   UP/DOWN speed   "
             "N/P next/prev game   G ghost   hover (paused) = stats   Q quit", (PAD, H - PAD - 8), color=DIM)
        if hover:
            self.draw_tooltip(hover[1])

    def screenshot(self, path, frame=None, mouse=None):
        """Render one frame (optionally with the mouse at `mouse`, showing the tooltip) to a PNG."""
        if frame is not None:
            self.state["k"] = int(np.clip(frame, 0, self.rp.Hz))
        self.state["mouse"], self.state["playing"] = mouse, False
        self.render()
        self.pg.image.save(self.screen, str(path))
        return path

    def do(self, act):
        st, rp = self.state, self.rp
        if act == "play":
            st["playing"] = not st["playing"]
            if st["playing"] and st["k"] == rp.Hz:        # replay from the start
                st["k"] = 0
        elif act == "fwd":
            st["playing"], st["k"] = False, min(rp.Hz, st["k"] + 1)
        elif act == "back":
            st["playing"], st["k"] = False, max(0, st["k"] - 1)
        elif act == "reset":
            st["playing"], st["k"] = False, 0
        elif act in ("next_game", "prev_game"):
            g = (rp.game + (1 if act == "next_game" else -1)) % rp.n_games
            if not self.headless:
                box = self.fonts["clock"].render(f"loading game {g + 1}/{rp.n_games}: rolling out the model ...",
                                                 True, (255, 230, 120))
                self.pg.draw.rect(self.screen, (10, 10, 22), box.get_rect(center=(W // 2, H // 2)).inflate(30, 20))
                self.screen.blit(box, box.get_rect(center=(W // 2, H // 2)))
                self.pg.display.flip()
            rp.load(g)
            st["playing"], st["k"] = False, 0
            self.caption()

    def loop(self):
        pg, st = self.pg, self.state
        clock = pg.time.Clock()
        while True:
            dt = clock.tick(60) / 1000
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return
                if ev.type == pg.MOUSEMOTION:
                    st["mouse"] = ev.pos
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    for act, r in self.rects.items():
                        if r.collidepoint(ev.pos):
                            self.do(act)
                    if self.bar.inflate(0, 10).collidepoint(ev.pos):   # jump inside the model window
                        rec = (ev.pos[0] - self.bar.x) / self.bar.w * self.rp.n_steps
                        st["k"] = int(np.clip(round(rec - self.rp.start), 0, self.rp.Hz))
                if ev.type != pg.KEYDOWN:
                    continue
                step = 10 if ev.mod & pg.KMOD_SHIFT else 1
                key = ev.key
                if key in (pg.K_q, pg.K_ESCAPE):
                    return
                elif key == pg.K_SPACE:
                    self.do("play")
                elif key == pg.K_RIGHT:
                    st["k"] = min(self.rp.Hz, st["k"] + step)
                elif key == pg.K_LEFT:
                    st["k"] = max(0, st["k"] - step)
                elif key in (pg.K_HOME, pg.K_r):
                    self.do("reset")
                elif key == pg.K_END:
                    st["k"] = self.rp.Hz
                elif key in (pg.K_n, pg.K_RIGHTBRACKET):
                    self.do("next_game")
                elif key in (pg.K_p, pg.K_LEFTBRACKET):
                    self.do("prev_game")
                elif key in (pg.K_UP, pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    st["speed"] = min(len(SPEEDS) - 1, st["speed"] + 1)
                elif key in (pg.K_DOWN, pg.K_MINUS, pg.K_KP_MINUS):
                    st["speed"] = max(0, st["speed"] - 1)
                elif key == pg.K_g:
                    st["ghost"] = not st["ghost"]
            if st["playing"]:
                st["acc"] += dt * SPEEDS[st["speed"]]
                adv, st["acc"] = int(st["acc"]), st["acc"] % 1
                st["k"] = min(self.rp.Hz, st["k"] + adv)
                if st["k"] == self.rp.Hz:
                    st["playing"] = False
            self.render()
            pg.display.flip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/token_wm/e4_baseline_v2_1h")
    ap.add_argument("--dense", default="data/moba/mix1k/dense/games.0000900.mbs")
    ap.add_argument("--game", type=int, default=0, help="index of the game within the dense file")
    ap.add_argument("--start", type=int, default=500, help="record (tick index) the dream starts from")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--burn", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ghost", action="store_true", help="start with the ghost overlay on")
    ap.add_argument("--frame", type=int, default=0, help="step to open at / render / query")
    ap.add_argument("--screenshot", help="headless: render --frame to this PNG and exit")
    ap.add_argument("--mouse", type=int, nargs=2, metavar=("X", "Y"),
                    help="with --screenshot: pretend the mouse is at this window pixel (shows the tooltip)")
    ap.add_argument("--entity", type=int, help="headless: print entity ENTITY at --frame, real vs model")
    ap.add_argument("--casts", action="store_true", help="headless: print every Q/W/E cast, real vs model")
    ap.add_argument("--deaths", action="store_true", help="headless: print every hero / tower death, real vs model")
    ap.add_argument("--json", action="store_true", help="with --entity / --casts / --deaths: print JSON instead of text")
    args = ap.parse_args()

    rp = Replay(args.run, args.dense, args.game, args.start, args.horizon, args.burn, args.device)
    for k in sorted({1, 4, 16, 64, rp.Hz}):
        if k <= rp.Hz:
            print(f"  k={k:4d}  " + "  ".join(f"{n}: {a:.2f} / {b:.2f}" for n, (a, b) in rp.divergence(k).items()),
                  "(model / persistence)", flush=True)
    k = int(np.clip(args.frame, 0, rp.Hz))
    if args.entity is not None:
        print(json.dumps(rp.entity(args.entity, k)) if args.json else rp.entity_table(args.entity, k))
    if args.casts:
        ev = rp.cast_events()
        if args.json:
            print(json.dumps(ev))
        else:
            print(f"{len(ev)} casts: real {int(rp.cast_truth.sum())}, model {int(rp.cast_pred.sum())}, "
                  f"both {sum(e['real'] and e['model'] for e in ev)}")
            for e in ev:
                who = "both" if e["real"] and e["model"] else "real only" if e["real"] else "model only"
                print(f"  step {e['step']:4d}  tick {e['tick']:5d}  hero {e['hero']}  {e['ability']}  {who}")
    if args.deaths:
        ev = rp.death_events()
        if args.json:
            print(json.dumps(ev))
        else:
            print(f"{len(ev)} deaths: real {int(rp.death_truth.sum())}, model {int(rp.death_pred.sum())}, "
                  f"both {sum(e['real'] and e['model'] for e in ev)}")
            for e in ev:
                who = "both" if e["real"] and e["model"] else "real only" if e["real"] else "model only"
                print(f"  step {e['step']:4d}  tick {e['tick']:5d}  {e['name']}  {who}")
    if args.screenshot:
        v = Viewer(rp, headless=True, ghost=args.ghost, frame=k)
        print(f"wrote {v.screenshot(args.screenshot, mouse=tuple(args.mouse) if args.mouse else None)} (frame {k})")
    if args.entity is None and not args.casts and not args.deaths and not args.screenshot:
        Viewer(rp, ghost=args.ghost, frame=k).loop()


if __name__ == "__main__":
    main()
