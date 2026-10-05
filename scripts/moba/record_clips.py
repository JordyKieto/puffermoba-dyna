"""Record ~1-minute video clips of a dream-PPO policy playing REAL games against the shipped net, drawn by the actual
PufferLib MOBA client (moba/moba_render.c -> upstream c_render's drawing code), with a camera that frames the action.

Per seed:
  1. scan: play the game headless exactly like the real-game evaluation (mobawm/rl/arena.py: same library, same
     controllers, same sampling: torch.Generator(seed)), recording heroes and towers every tick, then find
       ancient     the final push: the last CLIP ticks before the enemy ancient falls (only if we won)
       tower_dive  an enemy tower our team destroys while the tower was shooting our heroes (else tower_take)
       teamfight   the window with the most hero deaths + damage where both teams' heroes are within FIGHT_R cells
  2. render: replay the same seed (deterministic; checked against the scan at every clip start) and draw only the
     chosen windows: 12 interpolated frames per tick at 60 fps (the client's own pace: 5 ticks/s, so 300 ticks = 60 s).
     Camera: snaps to the event at the clip's first frame, then eases (exponential, per frame) toward the centroid of
     the event's entities (the fight's heroes; the tower or ancient plus our heroes near it).
  3. write <out>/<clip_id>.mp4 (H.264, crf 18) + <clip_id>.json (policy, seed, ticks, events in the clip, camera,
     renderer, git commit, md5) and <out>/index.json.

usage: python scripts/moba/record_clips.py --policy results/moba/dream/ppol4_policies/policy_u0049.pt --name ppol4_u49 \
           --seeds 70000000-70000009 [--per-kind 2] [--out media/moba/clips/<date>_<name>]"""
import argparse, ctypes, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mobawm.rl.dream_ppo import Policy  # noqa: E402

PUFFER = ROOT / "third_party/pufferlib"
LIB = ROOT / "build/libmobarender.dylib"
W, H, FPS, FRAMES = 1280, 720, 60, 12            # FRAMES: the client's interpolated frames per game tick
TOWER_OFFSET, N_TOWERS = 10 + 100 + 72, 24
ANCIENT = {"radiant": TOWER_OFFSET + 23, "dire": TOWER_OFFSET + 22}
TOP, BOTTOM = 34, 100                               # caption bar / client HUD bands (px): the action is framed between them
CLASSES = ("support", "assassin", "burst", "tank", "carry")
FIGHT_R = 10.0
FRAME_R, ANCHOR_R, MARGIN = 12.0, 20.0, 3.0         # camera: fighting heroes within FRAME_R cells of a fight (ANCHOR_R of a
                                                    # tower/ancient: the push around it), border MARGIN
P = ctypes.c_void_p


def lib():
    L = ctypes.CDLL(str(LIB))
    L.ma_init.argtypes = [ctypes.c_char_p]; L.ma_start.argtypes = [ctypes.c_uint64]
    L.ma_obs.argtypes = [P]; L.ma_ctrl_actions.argtypes = [ctypes.c_int, P]; L.ma_step.argtypes = [P]
    L.mr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]; L.mr_entities.argtypes = [P]
    L.mr_set_bloom.argtypes = [ctypes.c_char_p]
    L.mr_frame.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_char_p, P]
    assert L.ma_init(str(PUFFER / "resources/moba/moba_weights.bin").encode()) == 0
    return L


class Game:
    """One real game: ours (the PPO policy) on `side`, the shipped net on the other, stepped one tick at a time
    with the same calls and the same sampling as mobawm/rl/arena.py play()."""

    def __init__(self, L, pol_path, seed, side, max_ticks=6000):
        self.L, self.seed, self.side, self.max_ticks = L, seed, side, max_ticks
        self.pol = Policy.load(pol_path).eval()
        self.pol.reset_state(5, torch.device("cpu"))
        self.ours = slice(0, 5) if side == "radiant" else slice(5, 10)
        self.obs = np.zeros((10, 510), np.uint8); self.act = np.zeros((10, 6), np.float32)
        self.E = np.zeros((L.mr_num_entities(), L.mr_fields()), np.float32)
        self.g = torch.Generator().manual_seed(seed)
        L.ma_start(seed)
        self.t, self.result = 0, 0

    def step(self):
        L = self.L
        L.ma_ctrl_actions(0, self.act.ctypes.data)                # the shipped net (CTRL_NET = 0) for everyone
        L.ma_obs(self.obs.ctypes.data)
        with torch.no_grad():
            logits, _ = self.pol.step(torch.from_numpy(self.obs[self.ours].copy()))
            a = torch.stack([torch.multinomial(torch.softmax(lg, -1), 1, generator=self.g).squeeze(-1) for lg in logits], -1)
        self.act[self.ours] = a.numpy().astype(np.float32)
        self.result = L.ma_step(self.act.ctypes.data)
        self.t += 1
        return self.done

    @property
    def done(self):
        return bool(self.result) or self.t >= self.max_ticks

    def entities(self):
        self.L.mr_entities(self.E.ctypes.data)
        return self.E


def scan(L, pol_path, seed, side):
    """Play headless; heroes + towers every tick (alive, x, y, health, target)."""
    g = Game(L, pol_path, seed, side)
    keep = list(range(10)) + list(range(TOWER_OFFSET, TOWER_OFFSET + N_TOWERS))
    rec = [g.entities()[keep, :10].copy()]
    while not g.done:
        g.step()
        rec.append(g.entities()[keep, :10].copy())
    winner = {1: "radiant", 2: "dire", 3: "both"}.get(g.result, "none")
    return np.stack(rec), winner, g.t


def events(R, side):
    """Hero deaths and tower falls from the scan (R: ticks x 34 x 10; rows 0-9 heroes, 10-33 towers)."""
    team_of = lambda r: "radiant" if R[0, r, 2] == 0 else "dire"
    ev = []
    for t in range(1, len(R)):
        for r in range(10):                                  # a dead hero respawns at its spawn on the same tick
            # (spawn +- 7 cells, full health); blinks also jump but keep their health
            jump = (np.hypot(*(R[t, r, 3:5] - R[t - 1, r, 3:5])) > 5 and np.hypot(*(R[t, r, 3:5] - R[0, r, 3:5])) <= 11
                    and R[t, r, 5] >= R[t, r, 6] and R[t - 1, r, 5] < R[t - 1, r, 6])
            if (R[t - 1, r, 0] and not R[t, r, 0]) or jump:
                ev.append(dict(tick=t, kind="hero_death", pid=r, team=team_of(r), cls=CLASSES[r % 5],
                               x=float(R[t - 1, r, 3]), y=float(R[t - 1, r, 4])))
        for k in range(N_TOWERS):
            r = 10 + k
            if R[t - 1, r, 0] and not R[t, r, 0]:
                ev.append(dict(tick=t, kind="ancient_fall" if k >= 22 else "tower_fall", tower=k, tier=int(R[0, r, 8]),
                               team=team_of(r), x=float(R[0, r, 3]), y=float(R[0, r, 4])))
    return ev


def candidates(R, ev, side, T, clip):
    ours = range(0, 5) if side == "radiant" else range(5, 10)
    theirs = [r for r in range(10) if r not in ours]
    enemy = "dire" if side == "radiant" else "radiant"
    out = []
    for e in ev:
        if e["kind"] == "ancient_fall" and e["team"] == enemy:
            t1 = min(T, e["tick"] + 20)
            out.append(dict(kind="ancient", score=1e9, t0=max(0, t1 - clip), t1=t1, peak=e["tick"], anchor=(e["x"], e["y"]),
                            focus=list(ours), title="the final push: destroying the enemy ancient"))
        if e["kind"] == "tower_fall" and e["team"] == enemy:
            r, t = 10 + e["tower"], e["tick"]
            t0 = max(0, t - clip + 60)
            shot = [R[s, r, 7] for s in range(t0, t) if R[s, r, 0]]          # whom the tower was shooting
            dive = any(int(p) in ours for p in shot)
            # heroes at the fall (last 40 ticks): a tower pushed down by creeps alone makes a poor clip
            near = lambda rs: max(sum(R[s, q, 0] > 0 and np.hypot(R[s, q, 3] - e["x"], R[s, q, 4] - e["y"]) < 8 for q in rs)
                                  for s in range(max(0, t - 40), t))
            n_ours, n_theirs = near(ours), near(theirs)
            if n_ours == 0:
                continue
            out.append(dict(kind="tower_dive" if dive else "tower_take", t0=t0, t1=min(T, t + 60),
                            score=100 * e["tier"] + 30 * n_ours + 20 * n_theirs + 50 * dive, heroes_at_fall=[int(n_ours), int(n_theirs)],
                            peak=t, anchor=(e["x"], e["y"]), focus=list(ours), tower=e["tower"], tier=e["tier"],
                            title=f"{'tower dive' if dive else 'taking a tower'}: tier-{e['tier']} enemy tower"))
    # teamfights: score windows by hero deaths + hero damage while both teams' heroes are close
    hp = R[:, :10, 5]
    dmg = np.zeros(len(R))
    for t in range(1, len(R)):
        alive = (R[t, :10, 0] > 0) & (R[t - 1, :10, 0] > 0)
        xy = R[t, :10, 3:5]
        close = np.zeros(10, bool)
        for a in ours:
            for b in theirs:
                if alive[a] and alive[b] and np.hypot(*(xy[a] - xy[b])) < FIGHT_R:
                    close[a] = close[b] = True
        dmg[t] = np.clip(hp[t - 1] - hp[t], 0, None)[close & alive].sum()
    deaths = np.zeros(len(R))
    for e in ev:
        if e["kind"] == "hero_death":
            deaths[e["tick"]] += 1 if e["team"] == side else 2          # enemy heroes down count double
    w = 120                                               # the fight's core, scored; the clip adds lead-in and aftermath
    score = np.convolve(dmg / 1000 + 3 * deaths, np.ones(w), "valid")
    used = []
    for _ in range(6):
        if not len(score) or score.max() <= 0:
            break
        s = int(score.argmax()); peak = s + w // 2
        if not any(abs(peak - u) < clip for u in used):
            used.append(peak)
            d = [e for e in ev if e["kind"] == "hero_death" and s <= e["tick"] < s + w]
            out.append(dict(kind="teamfight", score=float(score[s]), t0=max(0, peak - clip // 2 - 30), t1=min(T, peak + clip // 2 - 30),
                            peak=peak, anchor=None, focus=list(range(10)), deaths=len(d),
                            title=f"teamfight: {sum(e['team'] != side for e in d)} enemy / {sum(e['team'] == side for e in d)} of ours down"))
        score[max(0, s - w):s + w] = 0
    return out


def camera_target(E, c, state, max_cell, min_cell):
    """(center, cell px) that frames the action: the clip's tower/ancient (if any) plus every live hero, of either
    team, within FRAME_R cells of the action's center, with a MARGIN-cell border; zooms out down to min_cell px per
    cell when that box does not fit at max_cell. A teamfight's center follows the fight (the heroes near the
    previous center)."""
    live = [r for r in range(10) if E[r, 0]]
    xy = {r: E[r, 3:5].astype(np.float64) for r in live}
    if c["anchor"] is not None:
        center = np.array(c["anchor"], np.float64)
    else:
        center = state.get("fight")
        if center is None:                                   # the fight at the clip start: heroes of both teams that are close
            pairs = [(a, b) for a in live for b in live if (a < 5) != (b < 5) and np.hypot(*(xy[a] - xy[b])) < FIGHT_R]
            pts = [xy[r] for p in pairs for r in p] or [xy[r] for r in live]
            center = np.mean(pts, 0) if pts else np.array([64.0, 64.0])
    # heroes in combat: targeting a hero or tower, or targeted by one (creep-farming heroes walking by do not count)
    tgt = E[:, 7].astype(int)
    fighters = set(range(10)) | set(range(TOWER_OFFSET, TOWER_OFFSET + N_TOWERS))
    hot = {r for r in live if tgt[r] in fighters}
    hot |= {int(tgt[q]) for q in fighters if E[q, 0] and 0 <= tgt[q] < 10}
    rad = ANCHOR_R if c["anchor"] is not None else FRAME_R
    near = [xy[r] for r in live if r in hot and np.hypot(*(xy[r] - center)) < rad]
    if not near:                                             # nobody fighting yet: the closest heroes to the action
        near = [xy[r] for r in live if np.hypot(*(xy[r] - center)) < FRAME_R / 2]
    pts = np.array(near + ([center] if c["anchor"] is not None else []) or [center])
    if c["anchor"] is None and near:
        state["fight"] = np.mean(near, 0)                    # follow the fight as it moves
    lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
    cell = float(np.clip(min(W / max(hi[0] - lo[0], 1), (H - TOP - BOTTOM) / max(hi[1] - lo[1], 1)), min_cell, max_cell))
    center = (lo + hi) / 2
    center[1] += (H / 2 - (TOP + (H - BOTTOM)) / 2) / cell  # the box's center lands mid-way between the caption and the HUD
    return center, cell


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="70000000-70000009"); ap.add_argument("--side", default="radiant")
    ap.add_argument("--per-kind", type=int, default=2, help="clips per kind per game (ancient: 1)")
    ap.add_argument("--takes", type=int, default=1, help="tower takes (no dive) per game")
    ap.add_argument("--max-clips", type=int, default=12); ap.add_argument("--clip-ticks", type=int, default=300)
    ap.add_argument("--cell", type=float, default=32, help="max pixels per map cell (32 = the client's default)")
    ap.add_argument("--min-cell", type=float, default=22, help="the camera zooms out to this when the action is wide")
    ap.add_argument("--wins-only", action="store_true", default=True); ap.add_argument("--out", default="")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
    out = Path(a.out or f"media/moba/clips/{time.strftime('%Y-%m-%d')}_{a.name}").resolve(); out.mkdir(parents=True, exist_ok=True)
    rel_policy = a.policy
    a.policy = str(Path(a.policy).resolve())             # mr_init chdirs into the client's resources
    L = lib()
    assert L.mr_init(W, H, str(PUFFER).encode()) == 0
    bloom = (PUFFER / "resources/moba/bloom_shader_330.fs").read_text()
    bloom = bloom.replace("varying ", "in ").replace("texture2D(", "texture(").replace("gl_FragColor", "finalColor") \
                 .replace("void main()", "out vec4 finalColor;\nvoid main()", 1)
    assert L.mr_set_bloom(bloom.encode()) == 0
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    pol_md5 = md5(a.policy)
    index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else []
    n_clips = len(index)
    for seed in range(lo, hi + 1):
        if n_clips >= a.max_clips:
            break
        t0s = time.time()
        R, winner, T = scan(L, a.policy, seed, a.side)
        ev = events(R, a.side)
        print(f"seed {seed}: {winner} wins at tick {T} ({time.time() - t0s:.0f}s scan), {len(ev)} events", flush=True)
        if a.wins_only and winner != a.side:
            continue
        cands = candidates(R, ev, a.side, T, a.clip_ticks)
        pick = []
        for kind, k in (("ancient", 1), ("tower_dive", a.per_kind), ("tower_take", a.takes), ("teamfight", a.per_kind)):
            pick += sorted([c for c in cands if c["kind"] == kind], key=lambda c: -c["score"])[:k]
        pick = sorted(pick, key=lambda c: c["t0"])[: a.max_clips - n_clips]
        if not pick:
            continue
        # render pass: replay the same game, draw only the picked windows
        g = Game(L, a.policy, seed, a.side)
        buf = np.zeros((H, W, 3), np.uint8)
        for c in pick:
            if g.t > c["t0"]:                                # overlapping windows: replay the game again from tick 0
                g = Game(L, a.policy, seed, a.side)
            while g.t < c["t0"]:
                g.step()
            E = g.entities()
            ref = R[g.t, :10, 3:5]
            assert np.allclose(E[:10, 3:5], ref), f"seed {seed}: replay diverged from the scan at tick {g.t}"
            clip_id = f"{a.name}_s{seed}_{c['kind']}_t{c['t0']:04d}"
            path = out / f"{clip_id}.mp4"
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            state = {}
            cam, cell = camera_target(E, c, state, a.cell, a.min_cell)       # snap at frame 0
            hud = min((r for r in range(10) if (r < 5) == (a.side == "radiant")),
                      key=lambda r: np.hypot(*(E[r, 3:5] - cam)) if E[r, 0] else 1e9)
            frames, cam_track = 0, []
            ours_k = theirs_k = 0
            for t in range(c["t0"], c["t1"]):
                E = g.entities()
                tgt, tcell = camera_target(E, c, state, a.cell, a.min_cell)
                deaths_so_far = [e for e in ev if e["kind"] == "hero_death" and e["tick"] <= g.t]
                ours_k = sum(e["team"] != a.side for e in deaths_so_far); theirs_k = sum(e["team"] == a.side for e in deaths_so_far)
                tw = [e for e in ev if e["kind"] == "tower_fall" and e["tick"] <= g.t]
                tw_ours = sum(e["team"] != a.side for e in tw); tw_theirs = sum(e["team"] == a.side for e in tw)
                cap = c["title"]
                cap_r = f"PPO {a.name} ({a.side}) vs shipped net   tick {g.t}/{T}   kills {ours_k}-{theirs_k}   towers {tw_ours}-{tw_theirs}"
                for f in range(FRAMES):
                    cam += (tgt - cam) * 0.05
                    cell += (tcell - cell) * 0.03
                    halfw, halfh = W / cell / 2, H / cell / 2
                    cam[0] = np.clip(cam[0], halfw, 128 - halfw); cam[1] = np.clip(cam[1], halfh, 128 - halfh)
                    L.mr_frame(float(cam[0]), float(cam[1]), float(cell), f / FRAMES, frames / FPS, hud, cap.encode(),
                               cap_r.encode(), buf.ctypes.data)
                    ff.stdin.write(buf.tobytes()); frames += 1
                cam_track.append([round(float(cam[0]), 2), round(float(cam[1]), 2), round(float(cell), 2)])
                if g.done:
                    break
                g.step()
            ff.stdin.close(); ff.wait()
            in_clip = [e for e in ev if c["t0"] <= e["tick"] <= c["t1"]]
            meta = dict(
                clip_id=clip_id, file=path.name, title=c["title"], kind=c["kind"],
                policy=dict(name=a.name, path=rel_policy, md5=pol_md5, side=a.side,
                            note="dream-PPO policy (E39 continuous Dyna loop), plays all 5 heroes of its side"),
                opponent=dict(name="shipped PufferNet", weights="third_party/pufferlib/resources/moba/moba_weights.bin",
                              md5=md5(PUFFER / "resources/moba/moba_weights.bin")),
                seed=seed, game=dict(winner=winner, ticks=T, our_side=a.side),
                clip=dict(tick_start=c["t0"], tick_end=c["t1"], event_tick=c["peak"], seconds=round(frames / FPS, 2),
                          frames=frames, fps=FPS, ticks_per_second=FPS / FRAMES, resolution=[W, H], cell_px_range=[a.min_cell, a.cell]),
                events=in_clip, score_at_end=dict(kills=[ours_k, theirs_k], towers=[tw_ours, tw_theirs]),
                camera=dict(mode=f"frame the box around the event's tower/ancient + every fighting hero within {ANCHOR_R} cells of it "
                                 f"(teamfights: within {FRAME_R} cells of the moving fight) "
                                 f"(+{MARGIN} cells border); snap at frame 0, then ease 5%/frame (center) and 3%/frame "
                                 f"(zoom, {a.min_cell}-{a.cell} px per cell; the client uses 32)",
                            anchor=c["anchor"], per_tick_center_x_y_cellpx=cam_track),
                hud_hero=dict(pid=int(hud), cls=CLASSES[hud % 5]),
                renderer="actual PufferLib MOBA client drawing (third_party/pufferlib/ocean/moba/moba.h c_render), via "
                         "moba/moba_render.c: given camera, offscreen 1280x720, shader clock = clip time, caption bar; "
                         "bloom shader keywords translated to GLSL 330 core (macOS)",
                deterministic_replay="the render pass replays the scanned game; hero positions checked equal at the clip start",
                code_commit=commit, created=time.strftime("%Y-%m-%d %H:%M:%S"), md5=md5(path))
            (out / f"{clip_id}.json").write_text(json.dumps(meta, indent=1))
            index.append({k: meta[k] for k in ("clip_id", "file", "title", "kind", "seed")} | dict(seconds=meta["clip"]["seconds"]))
            (out / "index.json").write_text(json.dumps(index, indent=1))
            n_clips += 1
            print(f"  {clip_id}: {frames} frames ({frames / FPS:.0f}s), {len(in_clip)} events, {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
