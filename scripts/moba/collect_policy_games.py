"""Dyna data: real games played by dream-trained policies, recorded in the corpus's own format.

Each game: a dream-PPO policy (mobawm.rl.dream_ppo.Policy) plays one side; the other side
is the shipped PufferNet, uniform-random actions, or a dream-PPO policy (--matchups), in the
real C game (build/libmobaarena, many games per process).
Every game's seed and all 10 heroes' action codes are written as standard .mbl logs
(controller id 2 = dream_ppo on the policy's side), with the collector's own FNV-1a state
hash, so `build/moba_collect replay` regenerates the dense states AND verifies them
bit-for-bit, exactly like mix1k.

    .venv/bin/python scripts/moba/collect_policy_games.py \\
        --policy runs/dream_ppo/p3/selfplay_rerun/policy.pt --policy runs/dream_ppo/p3/vs_net/policy.pt \\
        --games-per-policy 50 --first-game 5000 --out data/moba/dyna1
    # dyna2: policy x random, policy x net, policy x policy
    .venv/bin/python scripts/moba/collect_policy_games.py --policy ... --policy ... \
        --matchups policy-random,policy-net,policy-policy --games-per-matchup 270 \
        --first-game 6000 --seed-base 81000000 --max-ticks 3000 --out data/moba/dyna2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mobawm import shards as S  # noqa: E402
from mobawm.rl.dream_ppo import Policy  # noqa: E402
from mobawm.rl.plan_mpc import Arena  # noqa: E402
from mobawm.wm.cf import pack_codes  # noqa: E402

CTRL_NET, CTRL_POLICY, CTRL_RANDOM = 0, 2, 3
NET, RANDOM = -1, -2                                # side controllers; >= 0 = policy index
ACTION_SIZES = (7, 7, 3, 2, 2, 2)


def side_jobs(matchup, n, n_pol):
    """(radiant, dire) controllers for n games of one matchup: the policy alternates sides,
    and with several policies each one plays each side (and each pairing) equally often."""
    out = []
    for i in range(n):
        pi = (i // 2) % n_pol
        if matchup == "policy-policy":
            a, b = i % n_pol, (i // n_pol) % n_pol
            out.append((a, b))
            continue
        other = {"policy-net": NET, "policy-random": RANDOM}[matchup]
        out.append((pi, other) if i % 2 == 0 else (other, pi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--games-per-policy", type=int, default=50, help="without --matchups: policy vs net games per policy")
    ap.add_argument("--matchups", default="", help="comma list of policy-net, policy-random, policy-policy")
    ap.add_argument("--games-per-matchup", type=int, default=270)
    ap.add_argument("--slice", default="", help="lo:hi -- play only games lo..hi-1 of the job list (game index "
                    "first-game+lo..., seed seed-base+lo...): split a corpus into memory-sized jobs (~270 MB/game)")
    ap.add_argument("--first-game", type=int, default=5000, help="global game index of the first game")
    ap.add_argument("--seed-base", type=int, default=80_000_000)
    ap.add_argument("--per-file", type=int, default=10)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--out", default="data/moba/dyna1")
    ap.add_argument("--template-log", default="", help="a mix1k log to copy header constants from")
    ap.add_argument("--replay-jobs", type=int, default=3, help="replays at once (memory: ~2.2 GB each)")
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "dense").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pols = []
    for p in a.policy:
        pol = Policy()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        pol.load_state_dict(ck["policy"] if "policy" in ck else ck)
        pols.append(pol.eval())
    sides = []                                      # (radiant, dire) controller per game
    if a.matchups:
        for m in a.matchups.split(","):
            sides += side_jobs(m, a.games_per_matchup, len(pols))
    else:
        for pi in range(len(pols)):
            sides += [(pi, NET) if i % 2 == 0 else (NET, pi) for i in range(a.games_per_policy)]
    lo, hi = (int(x) for x in a.slice.split(":")) if a.slice else (0, len(sides))
    jobs = [(a.first_game + g, a.seed_base + g, sides[g]) for g in range(lo, hi)]   # (index, seed, sides)
    G = len(jobs)
    A = Arena(G)
    for g, (_, seed, _) in enumerate(jobs):
        A.start(g, seed)
    ctrl = np.array([[r] * 5 + [d] * 5 for _, _, (r, d) in jobs])      # (G, 10) per-hero controller
    rng = np.random.default_rng(a.seed_base)
    codes = [[] for _ in range(G)]
    running, result = np.ones(G, bool), np.zeros(G, int)
    gen = torch.Generator().manual_seed(0)
    o = np.zeros((10, 510), np.uint8)
    t0, t = time.time(), 0
    while running.any() and t < a.max_ticks:
        net = A.net_actions()                        # the net for everyone; ours overwritten below
        obs = np.zeros((G, 10, 510), np.uint8)
        for g in range(G):
            A.L.ma_multi_obs(g, o.ctypes.data)
            obs[g] = o
        act = net.copy()
        live = running[:, None]
        with torch.no_grad():
            for pi, pol in enumerate(pols):
                m = (ctrl == pi) & live
                if not m.any():
                    continue
                logits, _ = pol(torch.from_numpy(obs[m].reshape(-1, 510)))
                sm = torch.stack([torch.multinomial(torch.softmax(l, -1), 1, generator=gen).squeeze(-1) for l in logits], -1)
                act[m] = sm.numpy().astype(act.dtype)
        m = (ctrl == RANDOM) & live
        if m.any():
            act[m] = np.stack([rng.integers(0, n, m.sum()) for n in ACTION_SIZES], -1).astype(act.dtype)
        for g in np.nonzero(running)[0]:
            codes[g].append(pack_codes(act[g].astype(np.int64)))
            res = A.step(g, act[g])
            if res:
                running[g], result[g] = False, res
        t += 1
        if t % 1000 == 0:
            print(f"  tick {t}: {running.sum()}/{G} running ({time.time() - t0:.0f}s)", flush=True)
    A.L.ma_multi_final_hash.restype = __import__("ctypes").c_uint64
    A.L.ma_multi_final_hash.argtypes = [__import__("ctypes").c_int]
    tmpl = Path(a.template_log) if a.template_log else next((ROOT / "data/moba/mix1k/logs").glob("games.*.mbl"))
    th = np.frombuffer(tmpl.read_bytes(), S.LOG_HEADER_DTYPE, count=1)[0].copy()
    meta = []
    for f0 in range(0, G, a.per_file):
        chunk = list(range(f0, min(G, f0 + a.per_file)))
        hdr = th.copy()
        hdr["n_games"], hdr["max_ticks"], hdr["first_game"] = len(chunk), a.max_ticks, jobs[chunk[0]][0]
        hdr["base_seed"], hdr["epsilon"] = a.seed_base, 0.0
        parts = [hdr.tobytes()]
        code = lambda c: CTRL_NET if c == NET else CTRL_RANDOM if c == RANDOM else CTRL_POLICY
        name_ = lambda c: "net" if c == NET else "random" if c == RANDOM else a.policy[c]
        for g in chunk:
            gi, seed, (rc, dc) = jobs[g]
            c = np.ascontiguousarray(np.stack(codes[g]), dtype="<u2")
            z = zlib.compress(c.tobytes(), 6)
            gh = np.zeros(1, S.GAME_HEADER_DTYPE)[0]
            gh["game_seed"], gh["game_index"], gh["n_ticks"] = seed, gi, len(codes[g])
            gh["winner"], gh["truncated"] = int(result[g]), int(result[g] == 0)
            gh["radiant_ctrl"], gh["dire_ctrl"] = code(rc), code(dc)
            gh["zbytes"], gh["state_hash"] = len(z), A.L.ma_multi_final_hash(g)
            parts += [gh.tobytes(), z]
            meta.append(dict(game_index=gi, seed=seed, radiant=name_(rc), dire=name_(dc), ticks=len(codes[g]),
                             winner=int(result[g])))
        name = f"games.{jobs[chunk[0]][0]:07d}"
        (out / "logs" / f"{name}.mbl").write_bytes(b"".join(parts))
    (out / "games.json").write_text(json.dumps(meta, indent=1))
    print(f"{G} games, {sum(m['ticks'] for m in meta)} ticks recorded in {time.time() - t0:.0f}s; replaying to dense ...",
          flush=True)
    logs = sorted((out / "logs").glob("*.mbl"))
    bad = 0
    for i in range(0, len(logs), a.replay_jobs):     # each replay holds ~2.2 GB of raw states per 10 games
        procs = [subprocess.Popen([str(ROOT / "build/moba_collect"), "replay", str(p), "--dense",
                                   str(out / "dense" / (p.stem + ".mbs"))], stdout=subprocess.PIPE, text=True)
                 for p in logs[i:i + a.replay_jobs]]
        for p in procs:
            o_, _ = p.communicate()
            line = [l for l in o_.splitlines() if "replayed" in l]
            print("  " + (line[0] if line else o_.strip()[-200:]), flush=True)
            bad += p.returncode != 0
    print(f"replay: {len(logs)} files, {bad} with hash/obs mismatches -> {out / 'dense'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
