"""Counterfactual amplification of a recorded corpus with the real sim.

build/libmobafork (moba/moba_fork.c) replays recorded games from their logs
(seed + action codes, a few KB per game) and, at fork ticks, restores the real
state into a scratch game and steps it with K edited actions: one hero, one
action head, a new value. So every fork point is one real transition plus K
counterfactual ones from the SAME state -- the corpus never grows, it is
amplified, and nothing has to be stored: (game, tick, edit seed) regenerates it.

Parallelism is by process (the C side keeps the game's random stream in a
process-global): `CFStream` runs N worker processes, each writing its chunks
into its own shared-memory slab; the trainer copies finished chunks to the GPU.

    from mobawm.wm.cf import load_log_games, generate
    games = load_log_games(["data/moba/mix100/logs/games.0000000.mbl"])
    out = generate(games[:1], stride=8, K=7, check=True)
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from .. import shards as S

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "build" / ("libmobafork.so" if sys.platform.startswith("linux") else "libmobafork.dylib")
NE, NF = S.N_ENTITIES, len(S.F)

_lib = None


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise SystemExit(f"{LIB} missing; run: tools/build.sh moba-fork")
        L = ctypes.CDLL(str(LIB))
        P = ctypes.c_void_p
        L.mf_run.restype = ctypes.c_int64
        L.mf_run.argtypes = [ctypes.c_int, P, P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_uint64, ctypes.c_int, ctypes.c_int64, ctypes.c_int] + [P] * 15
        ne, nf, sb = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        L.mf_sizes(ctypes.byref(ne), ctypes.byref(nf), ctypes.byref(sb))
        assert (ne.value, nf.value) == (NE, NF), (ne.value, nf.value)
        _lib = L
    return _lib


from .data import usable_cpus  # noqa: E402  (one cgroup-aware CPU count for the whole package)


def pack_codes(actions: np.ndarray) -> np.ndarray:
    """(..., 6) head indices -> (...,) uint16 codes (moba_pack_action)."""
    code = np.zeros(actions.shape[:-1], np.int64)
    for h, n in enumerate(S.ACTION_SIZES):
        code = code * n + actions[..., h].astype(np.int64)
    return code.astype(np.uint16)


def load_log_games(paths, indices=None):
    """[(game_index, seed, codes (n_ticks, 10) uint16)] from .mbl logs."""
    out = []
    for p in paths:
        _, it = S.read_log(p)
        for g in it:
            if indices is None or g.index in indices:
                out.append((g.index, g.seed, np.ascontiguousarray(pack_codes(g.actions))))
    return out


def alloc(cap, K, burn=4):
    """Output buffers for up to `cap` fork points with K counterfactuals each."""
    Sb = K + 1
    return dict(
        hist=np.empty((cap, burn, NE, NF), np.float32), hist_tick=np.empty((cap, burn), np.int32),
        hist_cidx=np.empty((cap, burn), np.int32), hist_codes=np.empty((cap, burn, 10), np.uint16),
        ent_in=np.empty((cap, NE, NF), np.float32), tick_in=np.empty(cap, np.int32), cidx_in=np.empty(cap, np.int32),
        codes=np.empty((cap, Sb, 10), np.uint16), edit_hero=np.empty((cap, Sb), np.int8),
        edit_head=np.empty((cap, Sb), np.int8),
        ent_next=np.empty((cap, Sb, NE, NF), np.float32), tick_next=np.empty((cap, Sb), np.int32),
        cidx_next=np.empty((cap, Sb), np.int32), game=np.empty(cap, np.int32), t=np.empty(cap, np.int32))


ORDER = ("hist", "hist_tick", "hist_cidx", "hist_codes", "ent_in", "tick_in", "cidx_in", "codes", "edit_hero", "edit_head", "ent_next", "tick_next",
         "cidx_next", "game", "t")


def generate(games, stride=8, phase=0, K=7, edit_seed=0, check=False, cap=None, out=None, burn=4):
    """Fork points of `games` (see load_log_games) into numpy arrays (or `out`)."""
    seeds = np.array([g[1] for g in games], np.uint64)
    n_ticks = np.array([len(g[2]) for g in games], np.int32)
    off = np.zeros(len(games), np.int64)
    off[1:] = np.cumsum(n_ticks[:-1].astype(np.int64) * 10)
    codes = np.ascontiguousarray(np.concatenate([g[2].reshape(-1) for g in games]))
    if cap is None:
        cap = int(sum((n - 1 - phase + stride - 1) // stride for n in n_ticks)) + len(games)
    out = out or alloc(cap, K, burn)
    assert out["hist"].shape[1] == burn
    n = lib().mf_run(len(games), seeds.ctypes.data, n_ticks.ctypes.data, off.ctypes.data, codes.ctypes.data,
                     stride, phase, K, edit_seed, int(check), cap, burn, *[out[k].ctypes.data for k in ORDER])
    if n < 0:
        raise RuntimeError("mf_run check failed: forking the recorded action did not reproduce the real step")
    return {k: v[:n] for k, v in out.items()}, n


# --- streaming: worker processes -> shared memory -> trainer ---------------------

def _worker(wid, games, slab_names, cap, K, stride, q_free, q_full, seed, burn):
    lib()
    slabs = [shared_memory.SharedMemory(name=n) for n in slab_names]
    rng = np.random.default_rng(seed + wid)
    views = [_views(s.buf, cap, K, burn) for s in slabs]
    epoch = 0
    while True:
        order = rng.permutation(len(games))
        for j in order:                               # one game per chunk: <= 4225/stride + 1 fork points
            chunk = [games[j]]
            slot = q_free.get()
            if slot is None:
                return
            phase = int(rng.integers(stride))
            _, n = generate(chunk, stride=stride, phase=phase, K=K, edit_seed=int(rng.integers(2 ** 63)),
                            cap=cap, out=views[slot], burn=burn)
            q_full.put((wid, slot, n))
        epoch += 1


def _views(buf, cap, K, burn):
    """Numpy views laid out back to back in one shared buffer (same layout as alloc)."""
    proto = alloc(1, K, burn)
    out, off = {}, 0
    for k in ORDER:
        shape = (cap,) + proto[k].shape[1:]
        a = np.ndarray(shape, proto[k].dtype, buffer=buf, offset=off)
        out[k] = a
        off += a.nbytes
        off = (off + 63) // 64 * 64
    return out


def _slab_bytes(cap, K, burn):
    proto, total = alloc(1, K, burn), 0
    for k in ORDER:
        total += proto[k].nbytes * cap
        total = (total + 63) // 64 * 64
    return total


class CFStream:
    """N processes generating counterfactual chunks forever (games reshuffled each pass).

    get() returns (worker, slot, n, views): the first n fork points of that slot;
    release(worker, slot) hands the slot back. Each worker owns `slots` slabs of
    `cap` fork points, so generation runs ahead of the trainer.
    """

    def __init__(self, games, workers=None, K=7, stride=8, cap=None, slots=3, seed=0, burn=4):
        cap = cap or (4300 // stride + 2)             # one game per chunk
        self.K, self.cap, self.burn = K, cap, burn
        workers = workers or max(1, usable_cpus() - 2)
        ctx = mp.get_context("spawn")
        self.q_full = ctx.Queue()
        self.shms, self.q_free, self.procs, self.views = [], [], [], []
        per = [games[i::workers] for i in range(workers)]
        nb = _slab_bytes(cap, K, burn)
        for w in range(workers):
            shms = [shared_memory.SharedMemory(create=True, size=nb) for _ in range(slots)]
            self.shms.append(shms)
            self.views.append([_views(s.buf, cap, K, burn) for s in shms])
            qf = ctx.Queue()
            for s in range(slots):
                qf.put(s)
            self.q_free.append(qf)
            p = ctx.Process(target=_worker, args=(w, per[w], [s.name for s in shms], cap, K, stride, qf,
                                                  self.q_full, seed, burn), daemon=True)
            p.start()
            self.procs.append(p)
        self.workers = workers
        print(f"CFStream: {workers} workers x {slots} slabs x {cap} fork points x {K + 1} siblings "
              f"({workers * slots * nb / 1e9:.1f} GB shared)", flush=True)

    def get(self):
        w, s, n = self.q_full.get()
        return w, s, n, self.views[w][s]

    def release(self, w, s):
        self.q_free[w].put(s)

    def close(self):
        for q in self.q_free:
            for _ in range(8):
                q.put(None)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for shms in self.shms:
            for s in shms:
                s.close()
                s.unlink()


if __name__ == "__main__":
    # correctness + speed: python -m mobawm.wm.cf data/moba/mix100/logs/games.0000000.mbl [dense.mbs]
    log = sys.argv[1]
    games = load_log_games([log])[:2]
    t0 = time.time()
    out, n = generate(games, stride=8, K=7, check=True)
    dt = time.time() - t0
    ticks = sum(len(g[2]) for g in games)
    print(f"{n} fork points x 8 siblings from {len(games)} games ({ticks} ticks) in {dt:.2f}s: "
          f"{n * 7 / dt:.0f} counterfactual steps/s + {ticks / dt:.0f} replay ticks/s on one core; check passed")
    if len(sys.argv) > 2:
        d = S.load_dense(sys.argv[2])
        rec = d.load_game(0)
        m = out["game"] == 0
        ts = out["t"][m]
        same_in = np.array_equal(out["ent_in"][m], rec["state"]["entity"][ts])
        same_next = np.array_equal(out["ent_next"][m][:, 0], rec["state"]["entity"][ts + 1])
        same_hist = np.array_equal(out["hist"][m], rec["state"]["entity"][ts[:, None] + np.arange(-4, 0)])
        print(f"vs dense corpus: fork inputs identical {same_in}, real next states identical {same_next}, "
              f"history identical {same_hist}")
    diff = np.abs(out["ent_next"][:, 1:] - out["ent_next"][:, :1]).sum((2, 3)) > 0
    print(f"counterfactuals that change the next state: {100 * diff.mean():.1f}%  "
          f"(by edited head: " + ", ".join(f"{h} {100 * diff[out['edit_head'][:, 1:] == i].mean():.0f}%"
                                           for i, h in enumerate(("move_y", "move_x", "target", "q", "w", "e"))) + ")")
