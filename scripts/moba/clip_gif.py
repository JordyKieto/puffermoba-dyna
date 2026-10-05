"""A short GIF from a recorded clip (scripts/moba/record_clips.py), for READMEs and the paper: SECONDS seconds ending
TAIL seconds after the clip's event, scaled to WIDTH px, FPS frames/s, one optimised palette.
usage: python scripts/moba/clip_gif.py <clip.mp4> <out.gif> [--seconds 8] [--tail 1.5] [--width 640] [--fps 12]"""
import argparse, json, subprocess
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=8); ap.add_argument("--tail", type=float, default=1.5)
ap.add_argument("--width", type=int, default=640); ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()
m = json.loads(Path(a.clip).with_suffix(".json").read_text())
ev = (m["clip"]["event_tick"] - m["clip"]["tick_start"]) / m["clip"]["ticks_per_second"]
end = min(m["clip"]["seconds"], ev + a.tail); start = max(0.0, end - a.seconds)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{a.seconds:.2f}", "-i", a.clip, "-vf",
                f"fps={a.fps},scale={a.width}:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
                a.out], check=True)
print(a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB", f"t={start:.1f}-{end:.1f}s of {m['clip_id']}")
