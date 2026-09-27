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


# Findings the recordings illustrate, by experiment. Each points at claim ids in
# docs/moba/paper/CLAIMS.md; the per-frame numbers come from the Replay, not from here.
FINDINGS = {
    "wm/E4": ("Teacher-forced (one-step) training: open-loop rollouts fall apart. Heroes drift to the map "
              "edges within a few hundred ticks; the ghost lines show how far they are from the real heroes.",
              ["W1", "W2"]),
    "wm/E5_rollout_u3500": ("Early autoregressive rollout training (the starting point of E6b): stable over long "
                            "horizons, one-step jitter.", ["W3", "W7"]),
    "wm/E5_rollout_u20500": ("Autoregressive rollout training, best long-horizon model: heroes stay a few cells "
                             "from the real ones over 1,000 ticks; heroes jitter and tower health drifts.",
                             ["W3", "W5", "W7", "W14"]),
    "wm/E6b": ("Rollout training + real-sim counterfactual siblings + effect losses: smoother one-step motion and "
               "tower health closer to real; long horizons drift more than E5.", ["W8", "W9", "W14"]),
}
FILE_FINDINGS = {   # (artifact-name prefix, file suffix) -> a finding specific to one file
    ("wm/E4", "_ghost_step0300.png"): ("THE open-loop failure in one frame (paper Figure 1 candidate): at step 300 "
                                       "the teacher-forced model's heroes have drifted toward the map edges (seen, Q) "
                                       "and its hero error is no better than copying the start state (metrics, M); "
                                       "ghost lines reach back to the true positions.", ["W1"]),
    ("wm/E4", "_rollout_ghost.mp4"): ("Watch the heroes leave the lanes and the ghost lines lengthen over "
                                      "1,000 ticks.", ["W1"]),
}
RULES_CAVEAT = ("The dreamed states include hand-coded rules R1-R10 (spawns, creep stats, level and derived stats, "
                "counters, targets, clamps); see docs/moba/WM_ARCHITECTURE.md 'Rules vs learned'.")


def git_state():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return head, dirty


def model_provenance(label, run, rp, dense):
    """Everything needed to know which model made a recording, without guessing."""
    ck = Path(run) / "model.pt"
    ck_md5 = md5(ck)
    art = json.loads((ROOT / "artifacts/moba/MANIFEST.json").read_text())["artifacts"]
    match = next((a for a in art if a.get("md5") == ck_md5), None)
    name = match["name"] if match else None
    fam = next((k for k in FINDINGS if name and name.startswith(k)), None)
    return dict(
        label=label, checkpoint=str(ck), checkpoint_md5=ck_md5, stats_md5=md5(Path(run) / "stats.pt"),
        artifact=name, artifact_description=match["description"] if match else "NOT IN artifacts/moba/MANIFEST.json",
        wandb_run=match.get("wandb_run") if match else None,
        wandb_export=f"results/moba/{match['wandb_run']}/" if match and match.get("wandb_run") else None,
        s3=match.get("s3") if match else None,
        data=dict(dense=dense, dense_md5=md5(dense), game_in_file=rp.game, game_index=rp.gi, split=rp.split,
                  start_record=rp.start, start_tick=int(rp.ticks[0]), horizon_ticks=rp.Hz, burn_in=rp.burn,
                  controllers=rp.ctrl),
        finding=FINDINGS[fam][0] if fam else None, claims=FINDINGS[fam][1] if fam else [],
        rules_caveat=RULES_CAVEAT)


def metrics_at(rp, k):
    d = rp.divergence(k)
    return dict(step=k, tick=int(rp.ticks[k]),
                **{n.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"):
                   dict(model=round(a, 3), copy_start_state=round(b, 3)) for n, (a, b) in d.items()},
                casts_so_far=dict(real=int(rp.cast_truth[:k + 1].sum()), model=int(rp.cast_pred[:k + 1].sum())),
                deaths_so_far=dict(real=int(rp.death_truth[:k + 1].sum()), model=int(rp.death_pred[:k + 1].sum())))


def file_entry(path, what, label, prov, rp, steps):
    e = dict(file=path.name, md5=md5(path), bytes=path.stat().st_size, model=label, artifact=prov["artifact"],
             checkpoint_md5=prov["checkpoint_md5"], wandb_run=prov["wandb_run"], what=what,
             metrics=[metrics_at(rp, k) for k in steps])
    for (pref, suf), (text, claims) in FILE_FINDINGS.items():
        if prov["artifact"] and prov["artifact"].startswith(pref) and path.name.endswith(suf):
            e["finding"], e["claims"] = text, claims
    return e


def write_manifest(out, capture, models, files):
    (out / "MANIFEST.json").write_text(json.dumps(dict(
        schema="moba viewer recording manifest v2: capture (how/when), models (which checkpoint, from which run, "
               "what it shows), files (md5, what, metrics at the shown steps, file-specific findings)",
        capture=capture, models=models, files=files), indent=1))
    lines = [f"# Viewer recordings {out.name}", "",
             f"Recorded by `tools/moba_viz_capture.py` at git `{capture['git_head']}`"
             f"{' (working tree dirty)' if capture['git_dirty'] else ''}. Local only (gitignored). "
             "Full provenance: `MANIFEST.json`.", "", RULES_CAVEAT, "", "## Models", "",
             "| label | checkpoint (md5) | artifact | W&B run | finding | claims |", "|---|---|---|---|---|---|"]
    for m in models.values():
        lines.append(f"| {m['label']} | `{m['checkpoint']}` (`{m['checkpoint_md5'][:12]}`) | {m['artifact']} | "
                     f"`{m['wandb_run']}` | {m['finding']} | {', '.join(m['claims'])} |")
    lines += ["", "## Files worth calling out", ""]
    for f in files:
        if f.get("finding"):
            lines.append(f"- `{f['file']}`: {f['finding']} ({', '.join(f['claims'])})")
    lines += ["", "## All files", "", "| file | what | hero pos error model / copy (cells) at the shown step |", "|---|---|---|"]
    for f in files:
        m = f["metrics"][-1]
        hp = m.get("hero_pos_L1_cells", {})
        lines.append(f"| `{f['file']}` | {f['what']} | step {m['step']}: {hp.get('model')} / {hp.get('copy_start_state')} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def parse_steps(name, what, horizon):
    """The steps a file shows, from its name / description (for --backfill)."""
    import re
    if name.endswith(".mp4"):
        if "hover_tour" in name:
            return [int(x) for x in re.findall(r"\d+", what.split("[")[1].split("]")[0])]
        return [s for s in (16, 64, 300, horizon) if s <= horizon]
    m = re.search(r"step(\d{4})", name)
    if m:
        return [int(m.group(1))]
    m = re.search(r"at step (\d+)", what)
    step = int(m.group(1)) if m else 0
    return [min(step + 1, horizon)] if "death" in name else [step]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="LABEL=run_dir (repeatable)")
    ap.add_argument("--dense", default="data/moba/mix1k/dense/games.0000900.mbs")
    ap.add_argument("--game", type=int, default=0)
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--backfill", help="rebuild MANIFEST.json + README.md of an existing capture dir (no re-recording)")
    ap.add_argument("--git-head", help="--backfill: the git HEAD the capture was recorded at, if known")
    args = ap.parse_args()
    head, dirty = git_state()

    if args.backfill:
        out = Path(args.backfill)
        old = json.loads((out / "MANIFEST.json").read_text())
        if isinstance(old, dict):                    # v2: run/data live under "models"
            ms = old["models"]
            old = [dict(f, run=str(Path(ms[f["model"]]["checkpoint"]).parent), dense=ms[f["model"]]["data"]["dense"],
                        game=ms[f["model"]]["data"]["game_in_file"], start=ms[f["model"]]["data"]["start_record"],
                        horizon=ms[f["model"]]["data"]["horizon_ticks"]) for f in old["files"]]
        first = old[0]
        capture = dict(recorded=time.strftime("%Y-%m-%d %H:%M", time.localtime((out / first["file"]).stat().st_mtime)),
                       git_head=args.git_head or head, git_dirty=True, fps=args.fps,
                       viewer_md5=md5(ROOT / "tools/moba_viz.py"), recorder_md5=md5(Path(__file__)),
                       manifest_backfilled=time.strftime("%Y-%m-%d %H:%M"))
        models, files = {}, []
        for label in dict.fromkeys(e["model"] for e in old):
            es = [e for e in old if e["model"] == label]
            rp = V.Replay(es[0]["run"], es[0]["dense"], es[0]["game"], es[0]["start"], es[0]["horizon"])
            models[label] = model_provenance(label, es[0]["run"], rp, es[0]["dense"])
            for e in es:
                files.append(file_entry(out / e["file"], e["what"], label, models[label], rp,
                                        parse_steps(e["file"], e["what"], rp.Hz)))
        write_manifest(out, capture, models, files)
        print(f"-> {out}/MANIFEST.json + README.md ({len(files)} files, {len(models)} models)")
        return

    out = Path(args.out or ROOT / "media" / "moba" / time.strftime("%Y%m%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    capture = dict(recorded=time.strftime("%Y-%m-%d %H:%M"), git_head=head, git_dirty=dirty, fps=args.fps,
                   viewer_md5=md5(ROOT / "tools/moba_viz.py"), recorder_md5=md5(Path(__file__)))
    models, files = {}, []

    for spec in args.model:
        label, run = spec.split("=", 1)
        print(f"== {label}: {run}", flush=True)
        rp = V.Replay(run, args.dense, args.game, args.start, args.horizon)
        v = V.Viewer(rp, headless=True)
        prov = models[label] = model_provenance(label, run, rp, args.dense)

        def add(path, what, steps):
            files.append(file_entry(path, what, label, prov, rp, steps))
            print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)  {what}", flush=True)

        vid_steps = [s for s in (16, 64, 300, rp.Hz) if s <= rp.Hz]
        n = rollout_video(v, out / f"{label}_rollout.mp4", args.fps, ghost=False)
        add(out / f"{label}_rollout.mp4", f"rollout, {n} frames at {args.fps} fps (1 frame = 1 tick)", vid_steps)
        n = rollout_video(v, out / f"{label}_rollout_ghost.mp4", args.fps, ghost=True)
        add(out / f"{label}_rollout_ghost.mp4", "rollout with the ghost overlay (true positions on the model panel)",
            vid_steps)
        steps = [s for s in (64, 300, rp.Hz) if s <= rp.Hz]
        hover_tour(v, out / f"{label}_hover_tour.mp4", args.fps, steps)
        add(out / f"{label}_hover_tour.mp4", f"hover tour at steps {steps}: heroes, weakest tower, a creep, both panels",
            steps)
        for k in sorted({0, 16, 64, 300, rp.Hz} & set(range(rp.Hz + 1))):
            p = out / f"{label}_step{k:04d}.png"
            snap(v, p, k)
            add(p, f"step {k} (tick {int(rp.ticks[k])})", [k])
        k = min(300, rp.Hz)
        tg = tour_targets(rp, k)
        for name, mouse in (("hero_real", tg[0]), ("hero_model", tg[1]), ("tower_real", tg[-4]),
                            ("tower_model", tg[-3]), ("creep_model", tg[-1])):
            p = out / f"{label}_tooltip_{name}_step{k:04d}.png"
            snap(v, p, k, mouse=mouse)
            add(p, f"stats tooltip ({name.replace('_', ' ')}) at step {k}", [k])
        p = out / f"{label}_ghost_step{k:04d}.png"
        snap(v, p, k, ghost=True)
        add(p, f"ghost overlay at step {k}", [k])
        ce = [e for e in rp.cast_events() if e["real"]]
        if ce:
            p = out / f"{label}_first_real_cast.png"
            snap(v, p, ce[0]["step"])
            add(p, f"first real cast: hero {ce[0]['hero']} {ce[0]['ability']} at step {ce[0]['step']}", [ce[0]["step"]])
        de = [e for e in rp.death_events() if e["real"]]
        if de:
            p = out / f"{label}_first_real_death.png"
            snap(v, p, min(de[0]["step"] + 1, rp.Hz))
            add(p, f"first real death: {de[0]['name']} at step {de[0]['step']}", [min(de[0]["step"] + 1, rp.Hz)])
    write_manifest(out, capture, models, files)
    print(f"-> {out}  ({len(files)} files)")


if __name__ == "__main__":
    main()
