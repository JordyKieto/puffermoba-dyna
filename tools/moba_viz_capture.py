"""Record the replay viewer (tools/moba_viz.py) headless: videos and PNG snapshots.

The viewer is the microscope of this work: the real game on the left, the world
model on the right, both driven by the same recorded actions. This script drives
it without a window and records, per model:

  <label>_rollout.mp4        the whole window (--horizon ticks) played through
  <label>_rollout_ghost.mp4  the same with the ghost overlay (true positions on the model panel)
  <label>_hover_tour.mp4     paused at a few moments, a drawn cursor glides over heroes, a tower
                             and a creep on both panels, dwelling so each stats tooltip
                             (real vs model, every field) can be read
  <label>_*.png              snapshots: key frames, tooltips on a hero / tower / creep, the first
                             real cast and death, the ghost overlay

Media is written under media/moba/<timestamp>/ (gitignored): it is for looking at, not
for source control. A MANIFEST.json there lists every file with its md5 and what it shows.

    .venv/bin/python tools/moba_viz_capture.py \\
        --model E5_u20500=runs/token_wm/e5_rollout --model E6b_u3500=runs/token_wm/e6b_cf_effect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ["SDL_VIDEODRIVER"] = "dummy"

import moba_viz as V  # noqa: E402

F = V.F


def panel_px(panel, x, y):
    """Window pixel of map cell (x, y) on panel 0 (real) or 1 (model)."""
    return int(V.PAD + panel * (V.PANEL + V.PAD) + x * V.CELL), int(V.TITLE_H + y * V.CELL)


def draw_cursor(pg, screen, pos):
    x, y = pos
    pts = [(x, y), (x, y + 16), (x + 4, y + 12), (x + 7, y + 19), (x + 10, y + 18), (x + 7, y + 11), (x + 12, y + 11)]
    pg.draw.polygon(screen, (255, 255, 255), pts)
    pg.draw.polygon(screen, (0, 0, 0), pts, 1)


class Recorder:
    def __init__(self, path, fps):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{V.W}x{V.H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
            stdin=subprocess.PIPE)
        self.n = 0

    def add(self, pg, screen):
        self.p.stdin.write(pg.image.tobytes(screen, "RGB"))
        self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def frame(v, k, mouse=None, playing=False, ghost=None):
    v.state["k"], v.state["mouse"], v.state["playing"] = k, mouse, playing
    if ghost is not None:
        v.state["ghost"] = ghost
    v.render()
    if mouse is not None:
        draw_cursor(v.pg, v.screen, mouse)


def rollout_video(v, path, fps, ghost):
    rec = Recorder(path, fps)
    for k in range(v.rp.Hz + 1):
        frame(v, k, playing=True, ghost=ghost)
        rec.add(v.pg, v.screen)
    rec.close()
    return rec.n


def tour_targets(rp, k):
    """Where the cursor goes at step k: heroes on both panels, the most damaged tower, a creep."""
    t, m = rp.truth[k], rp.pred[k]
    targets = []
    for i in (0, 3, 5, 8):                                   # a support, a tank, and the same on the other team
        for p, e in ((0, t), (1, m)):
            if e[i, F["alive"]] > 0:
                targets.append(panel_px(p, e[i, F["x"]], e[i, F["y"]]))
    towers = [i for i in range(182, 206) if t[i, F["alive"]] > 0]
    if towers:
        i = min(towers, key=lambda i: t[i, F["health"]] / max(t[i, F["max_health"]], 1))
        targets += [panel_px(0, t[i, F["x"]], t[i, F["y"]]), panel_px(1, m[i, F["x"]], m[i, F["y"]])]
    creeps = [i for i in range(10, 110) if t[i, F["alive"]] > 0 and m[i, F["alive"]] > 0]
    if creeps:
        i = creeps[len(creeps) // 2]
        targets += [panel_px(0, t[i, F["x"]], t[i, F["y"]]), panel_px(1, m[i, F["x"]], m[i, F["y"]])]
    return targets


def hover_tour(v, path, fps, steps, glide=14, dwell=45):
    rec = Recorder(path, fps)
    pos = (V.W // 2, V.H // 2)
    for k in steps:
        for tgt in tour_targets(v.rp, k):
            for s in range(1, glide + 1):                    # glide
                a = s / glide
                m = (int(pos[0] + (tgt[0] - pos[0]) * a), int(pos[1] + (tgt[1] - pos[1]) * a))
                frame(v, k, mouse=m)
                rec.add(v.pg, v.screen)
            for _ in range(dwell):                           # dwell: the tooltip is up
                frame(v, k, mouse=tgt)
                rec.add(v.pg, v.screen)
            pos = tgt
    rec.close()
    return rec.n


def snap(v, path, k, mouse=None, ghost=False):
    frame(v, k, mouse=mouse, ghost=ghost)
    v.pg.image.save(v.screen, str(path))
    v.state["ghost"] = False


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="LABEL=run_dir (repeatable)")
    ap.add_argument("--dense", default="data/moba/mix1k/dense/games.0000900.mbs")
    ap.add_argument("--game", type=int, default=0)
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "media" / "moba" / time.strftime("%Y%m%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def add(path, what, label, **kw):
        manifest.append(dict(file=path.name, what=what, model=label, md5=md5(path), bytes=path.stat().st_size, **kw))
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)  {what}", flush=True)

    for spec in args.model:
        label, run = spec.split("=", 1)
        print(f"== {label}: {run}", flush=True)
        rp = V.Replay(run, args.dense, args.game, args.start, args.horizon)
        v = V.Viewer(rp, headless=True)
        meta = dict(run=run, dense=args.dense, game=args.game, start=args.start, horizon=rp.Hz,
                    game_index=rp.gi, split=rp.split)
        n = rollout_video(v, out / f"{label}_rollout.mp4", args.fps, ghost=False)
        add(out / f"{label}_rollout.mp4", f"rollout, {n} frames at {args.fps} fps (1 frame = 1 tick)", label, **meta)
        n = rollout_video(v, out / f"{label}_rollout_ghost.mp4", args.fps, ghost=True)
        add(out / f"{label}_rollout_ghost.mp4", "rollout with the ghost overlay (true positions on the model panel)",
            label, **meta)
        steps = [s for s in (64, 300, rp.Hz) if s <= rp.Hz]
        n = hover_tour(v, out / f"{label}_hover_tour.mp4", args.fps, steps)
        add(out / f"{label}_hover_tour.mp4", f"hover tour at steps {steps}: heroes, weakest tower, a creep, both panels",
            label, **meta)
        for k in sorted({0, 16, 64, 300, rp.Hz} & set(range(rp.Hz + 1))):
            p = out / f"{label}_step{k:04d}.png"
            snap(v, p, k)
            add(p, f"step {k} (tick {int(rp.ticks[k])})", label, **meta)
        k = min(300, rp.Hz)
        tg = tour_targets(rp, k)
        for name, mouse in (("hero_real", tg[0]), ("hero_model", tg[1]), ("tower_real", tg[-4]),
                            ("tower_model", tg[-3]), ("creep_model", tg[-1])):
            p = out / f"{label}_tooltip_{name}_step{k:04d}.png"
            snap(v, p, k, mouse=mouse)
            add(p, f"stats tooltip ({name.replace('_', ' ')}) at step {k}", label, **meta)
        p = out / f"{label}_ghost_step{k:04d}.png"
        snap(v, p, k, ghost=True)
        add(p, f"ghost overlay at step {k}", label, **meta)
        ce = [e for e in rp.cast_events() if e["real"]]
        if ce:
            p = out / f"{label}_first_real_cast.png"
            snap(v, p, ce[0]["step"])
            add(p, f"first real cast: hero {ce[0]['hero']} {ce[0]['ability']} at step {ce[0]['step']}", label, **meta)
        de = [e for e in rp.death_events() if e["real"]]
        if de:
            p = out / f"{label}_first_real_death.png"
            snap(v, p, min(de[0]["step"] + 1, rp.Hz))
            add(p, f"first real death: {de[0]['name']} at step {de[0]['step']}", label, **meta)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {out}  ({len(manifest)} files)")


if __name__ == "__main__":
    main()
