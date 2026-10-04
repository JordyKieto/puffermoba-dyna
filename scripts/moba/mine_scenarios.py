"""Scenario mining: tower-death slices with variation, simulated by the real game.

For every tower/ancient death in the given training games, the real C game (moba_fork.c
ms_run) restores the state 96 ticks before the death and simulates 128-tick VARIANTS
(new random stream, small state nudges, random action stretches for some heroes) until
the event has `--quota` slices where the tower still dies and `--quota` where it survives.
Only the recorded logs are needed (the game is replayed to find the events).

Shard files (npz): per slice the zlib-compressed perturbed start state (MobaState bytes),
the 128 x 10 action codes played, its length, event tick, tower (0-23), outcome
(died), death step, variation strength, game index. `ms_replay` regenerates any slice's
full states exactly (see --verify).

    .venv/bin/python scripts/moba/mine_scenarios.py --logs data/moba/mix1k/logs --games 0-899 \\
        --quota 36 --out data/moba/scenarios_tower --workers 30
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mobawm.wm.cf import load_log_games  # noqa: E402

MS_LEN, NE, NF = 128, 206, 26
_L = None


def lib():
    global _L
    if _L is None:
        p = ROOT / "build" / ("libmobafork.so" if sys.platform.startswith("linux") else "libmobafork.dylib")
        L = ctypes.CDLL(str(p))
        P, I, I64, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64
        L.ms_run.restype = I64
        L.ms_run.argtypes = [U64, I, P, I, I, I, U64, I64, P, P, P, P, P, P, P, P, P]
        L.ms_replay.restype = I
        L.ms_replay.argtypes = [P, P, I, I, P, P, P]
        if hasattr(L, "ms_replay_scan"):
            L.ms_replay_scan.restype = I
            L.ms_replay_scan.argtypes = [P, P, I, I, P, P, P, P]
        L.mf_sizes.argtypes = [P, P, P]
        ne, nf, sb = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        L.mf_sizes(ctypes.byref(ne), ctypes.byref(nf), ctypes.byref(sb))
        L.state_bytes = sb.value
        _L = L
    return _L


def mine_game(args):
    gi, seed, codes, (q_die, q_live), max_att, var_seed = args
    quota = max(q_die, q_live)
    L = lib()
    SB = L.state_bytes
    cap = 20 * 2 * quota                                    # events per game are ~2-15 (mix1k max 14)
    st = np.zeros((cap, SB), np.uint8)
    cd = np.zeros((cap, MS_LEN, 10), np.uint16)
    ln, te, tw, dk = (np.zeros(cap, np.int32) for _ in range(4))
    died, strength, ev = np.zeros(cap, np.int8), np.zeros(cap, np.float32), ctypes.c_int32()
    codes = np.ascontiguousarray(codes, np.uint16)
    n = L.ms_run(seed, len(codes), codes.ctypes.data, q_die, q_live, max_att, var_seed, cap,
                 st.ctypes.data, cd.ctypes.data, ln.ctypes.data, te.ctypes.data, tw.ctypes.data,
                 died.ctypes.data, dk.ctypes.data, strength.ctypes.data, ctypes.byref(ev))
    blobs = [zlib.compress(st[i].tobytes(), 1) for i in range(n)]
    return dict(game=gi, events=ev.value, blobs=blobs, codes=cd[:n], len=ln[:n], te=te[:n], tower=tw[:n],
                died=died[:n], death_k=dk[:n], strength=strength[:n])


def write_shard(path, parts):
    blobs = [b for p in parts for b in p["blobs"]]
    off = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    game = np.concatenate([np.full(len(p["blobs"]), p["game"], np.int32) for p in parts])
    np.savez(path, state_blob=np.frombuffer(b"".join(blobs), np.uint8), state_off=off, codes=cat("codes"),
             len=cat("len"), te=cat("te"), tower=cat("tower"), died=cat("died"), death_k=cat("death_k"),
             strength=cat("strength"), game=game)


def verify(path, k=20):
    """Regenerate k slices of a shard with ms_replay; the outcome must match the stored one."""
    L = lib()
    z = np.load(path)
    idx = np.random.default_rng(0).choice(len(z["len"]), min(k, len(z["len"])), replace=False)
    bad = 0
    for i in idx:
        st = np.frombuffer(zlib.decompress(z["state_blob"][z["state_off"][i]:z["state_off"][i + 1]].tobytes()), np.uint8)
        n = int(z["len"][i])
        ent = np.zeros((n + 1, NE, NF), np.float32); tk = np.zeros(n + 1, np.int32); cc = np.zeros(n + 1, np.int32)
        codes = np.ascontiguousarray(z["codes"][i])
        dkk = L.ms_replay(st.ctypes.data, codes.ctypes.data, n, int(z["tower"][i]), ent.ctypes.data, tk.ctypes.data, cc.ctypes.data)
        ok = (dkk >= 0) == bool(z["died"][i]) and (dkk == int(z["death_k"][i]) or not z["died"][i])
        bad += not ok
    return len(idx), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="log dirs (games.*.mbl)")
    ap.add_argument("--games", default="", help="game indices to use (training games only), e.g. 0-899,1000-1139")
    ap.add_argument("--quota", type=int, default=36, help="slices per event per outcome (died / survived)")
    ap.add_argument("--quota-die", type=int, default=None, help="override --quota for slices where the tower dies")
    ap.add_argument("--quota-live", type=int, default=None, help="override --quota for slices where it survives")
    ap.add_argument("--max-attempts", type=int, default=600)
    ap.add_argument("--var-seed", type=int, default=0)
    ap.add_argument("--games-per-shard", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--persist", action="store_true", help="upload each shard (mobawm.s3persist; needs a policy file)")
    a = ap.parse_args()
    from mobawm.wm.train import parse_games
    want = parse_games(a.games) if a.games else None
    paths = sorted(p for d in a.logs for p in Path(d).glob("games.*.mbl"))
    games = [g for g in load_log_games(paths, want)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(games)} games from {len(paths)} log files; quota {a.quota}+{a.quota} per event; {a.workers} workers", flush=True)
    if a.persist:
        from mobawm.s3persist import flush, persist
    t0 = time.time()
    tot = dict(slices=0, died=0, events=0, games=0)
    pending, shard_i = [], 0
    q = (a.quota if a.quota_die is None else a.quota_die, a.quota if a.quota_live is None else a.quota_live)
    jobs = [(gi, seed, codes, q, a.max_attempts, a.var_seed) for gi, seed, codes in games]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(mine_game, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            pending.append(r)
            tot["games"] += 1; tot["events"] += r["events"]; tot["slices"] += len(r["blobs"]); tot["died"] += int(r["died"].sum())
            if len(pending) >= a.games_per_shard:
                p = out / f"shard_{shard_i:05d}.npz"
                write_shard(p, pending); pending, shard_i = [], shard_i + 1
                if a.persist:
                    persist(p)
            if tot["games"] % 20 == 0 or tot["games"] == len(jobs):
                el = time.time() - t0
                print(f"{tot['games']}/{len(jobs)} games | {tot['events']} events | {tot['slices']} slices "
                      f"({tot['died']} died / {tot['slices'] - tot['died']} survived) | {el:.0f}s "
                      f"| {tot['slices'] / max(el, 1):.0f} slices/s", flush=True)
    if pending:
        p = out / f"shard_{shard_i:05d}.npz"; write_shard(p, pending)
        if a.persist:
            persist(p)
    (out / f"summary_{a.var_seed}.json").write_text(json.dumps(dict(args=vars(a), **tot, seconds=time.time() - t0), indent=1))
    if a.persist:
        persist(out / f"summary_{a.var_seed}.json"); flush()
    n, bad = verify(out / "shard_00000.npz")
    print(f"verify: {n} slices regenerated with ms_replay, {bad} outcome mismatches", flush=True)
    print("MINING DONE", tot, flush=True)


if __name__ == "__main__":
    main()
