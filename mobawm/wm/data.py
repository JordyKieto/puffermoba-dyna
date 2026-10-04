"""Data for the token world model: games in memory, windows, featurization,
targets, and the game rules used to fill derived fields in rollouts.

Featurization runs in torch on raw MobaState entity arrays, so training and
open-loop rollouts use exactly the same code path (no train/rollout skew).
See docs/moba/TOKEN_WM_DESIGN.md.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import shards as S

F = S.F
N_ENT = S.N_ENTITIES
ROOT = Path(__file__).resolve().parents[2]

# Entity kind per slot: 0 hero, 1 creep, 2 neutral, 3 tower.
KIND = np.zeros(N_ENT, np.int64)
KIND[S.CREEPS] = 1
KIND[S.NEUTRALS] = 2
KIND[S.TOWERS] = 3
N_KIND = 4
HEROES = slice(0, 10)

# Fields predicted by delta regression (see design doc: Targets).
REG_FIELDS = ("x", "y", "health", "mana", "xp", "q_timer", "w_timer", "e_timer",
              "move_modifier", "stun_timer", "move_timer", "basic_attack_timer", "waypoint")
REG_IDX = [F[f] for f in REG_FIELDS]
INT_FIELDS = {"xp", "q_timer", "w_timer", "e_timer", "stun_timer", "move_timer",
              "basic_attack_timer", "waypoint"}

# Upstream XP_FOR_LEVEL (moba.h) -- level is recomputed from XP on every XP gain.
XP_FOR_LEVEL = np.array([0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200,
                         9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000,
                         28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900], np.float32)


def load_walls() -> np.ndarray:
    """Static map: 1 = wall. From the compiled-in upstream game_map.h."""
    src = (ROOT / "third_party/pufferlib/ocean/moba/game_map.h").read_text().split("{", 1)[1]
    vals = np.array([int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", src)], np.uint8)
    return (vals.reshape(128, 128) == 1).astype(np.float32)


@dataclass
class Game:
    index: int
    ctrl: tuple            # (radiant, dire) controller names
    ent: np.ndarray        # (n+1, 206, 26) float32, record n is the terminal state
    tick: np.ndarray       # (n+1,) int
    creep_idx: np.ndarray  # (n+1,) int
    actions: np.ndarray    # (n, 10, 6) uint8

    @property
    def n_steps(self) -> int:
        return len(self.actions)


def load_games(dense_files, game_indices=None, log=print, workers=32) -> list[Game]:
    """Decompress games in parallel threads (zlib releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for path in dense_files:
        d = S.load_dense(path)
        jobs += [(d, k) for k, (gi, _, _, _) in enumerate(d.games)
                 if game_indices is None or gi in game_indices]

    def one(job):
        d, k = job
        r = d.load_game(k)
        st = r["state"]
        g = Game(index=int(r[0]["game_index"]),
                 ctrl=(S.CONTROLLERS[int(r[0]["radiant_ctrl"])], S.CONTROLLERS[int(r[0]["dire_ctrl"])]),
                 ent=np.ascontiguousarray(st["entity"], np.float32),
                 tick=st["tick"].astype(np.int64),
                 creep_idx=st["creep_idx"].astype(np.int64),
                 actions=np.ascontiguousarray(r["actions"][:-1]))
        log(f"  loaded game {g.index} ({g.ctrl[0]} vs {g.ctrl[1]}, {g.n_steps} steps)")
        return g

    with ThreadPoolExecutor(workers) as ex:
        return sorted(ex.map(one, jobs), key=lambda g: g.index)


def sample_windows(games, batch, K, rng, starts=None):
    """B windows of K steps: states (B,K+1,206,26), actions (B,K,10,6), ticks, creep_idx."""
    ents, acts, ticks, cidx = [], [], [], []
    for b in range(batch):
        if starts is not None:
            g, s0 = starts[b]
        else:
            g = games[rng.integers(len(games))]
            s0 = int(rng.integers(0, g.n_steps - K + 1))
        ents.append(g.ent[s0:s0 + K + 1])
        acts.append(g.actions[s0:s0 + K])
        ticks.append(g.tick[s0:s0 + K + 1])
        cidx.append(g.creep_idx[s0:s0 + K + 1])
    return (torch.from_numpy(np.stack(ents)), torch.from_numpy(np.stack(acts).astype(np.int64)),
            torch.from_numpy(np.stack(ticks)), torch.from_numpy(np.stack(cidx)))


# --- parallel loading: every core decompresses games into shared memory ------

_SHM = {}


def _attach(names, shapes, dtypes):
    from multiprocessing import shared_memory
    for key in names:
        shm = shared_memory.SharedMemory(name=names[key])
        _SHM[key] = (shm, np.ndarray(shapes[key], dtype=dtypes[key], buffer=shm.buf))


def _load_one(job):
    path, k, off = job
    r = S.load_dense(path).load_game(k)
    n = len(r)
    arr = _SHM["ent"][1]
    arr[off:off + n, :, :N_BASE] = r["state"]["entity"]
    if arr.shape[-1] > N_BASE:                      # LIST_STATE: the stored target lists as membership bits
        arr[off:off + n, :, N_BASE:N_BASE + LIST_NF] = pack_lists_np(r["state"]["scan"])
    _SHM["act"][1][off:off + n - 1] = r["actions"][:-1]
    _SHM["tick"][1][off:off + n] = r["state"]["tick"]
    _SHM["cidx"][1][off:off + n] = r["state"]["creep_idx"]
    return n


def usable_cpus() -> int:
    """CPUs this process may actually use: the container's cgroup quota (v2 cpu.max or v1
    cfs quota/period) if there is one, else the core count. A RunPod container reports the
    host's 128-144 cores but may be limited to ~14-16: starting a process per host core
    oversubscribes it (measured 2026-09-27: full mix1k decode 134 s with 144 workers vs 55 s
    with 15)."""
    import os
    for q_path, p_path in (("/sys/fs/cgroup/cpu.max", None),
                           ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if p_path is None:
                q, period = Path(q_path).read_text().split()
            else:
                q, period = Path(q_path).read_text().strip(), Path(p_path).read_text().strip()
            if q not in ("max", "-1"):
                return max(1, int(int(q) / int(period)))
        except (OSError, ValueError):
            pass
    return os.cpu_count() or 2


def load_store(dense_files, game_indices, device, workers=None, packed=None) -> "Store":
    """Decompress the selected games with one process per core straight into
    shared memory, then copy to the device in chunks. No per-game Python
    objects, no temporary files.

    packed: a directory for a raw (uncompressed) copy of exactly this store, keyed by the files, games and
    field count. If it exists the arrays are memory-mapped (seconds instead of minutes of decompression);
    otherwise the store is built as usual and written there once (CPU stores only)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context, shared_memory
    jobs, spans, off = [], [], 0
    for path in dense_files:
        d = S.load_dense(path)
        for k, (gi, n_rec, _, _) in enumerate(d.games):
            if game_indices is None or gi in game_indices:
                jobs.append((str(path), k, off))
                spans.append((off, n_rec - 1))
                off += n_rec
    shapes = dict(ent=(off, N_ENT, len(F)), act=(off, 10, 6), tick=(off,), cidx=(off,))
    dtypes = dict(ent=np.float32, act=np.uint8, tick=np.int32, cidx=np.int32)
    pdir = None
    if packed:
        import hashlib, json
        key = hashlib.sha1(json.dumps([[Path(j[0]).name, j[1]] for j in jobs] + [len(F)]).encode()).hexdigest()[:16]
        pdir = Path(packed) / key
        if (pdir / "done").exists():
            import time as _t
            t0 = _t.time()
            # copy into ordinary RAM with many threads (parallel reads): training straight off a copy-on-write file
            # mapping was CPU-bound (0.16 vs 0.45 updates/s), and a single-threaded np.load reads ~100 MB/s
            from concurrent.futures import ThreadPoolExecutor
            arrays = {}
            for k in shapes:
                m = np.load(pdir / f"{k}.npy", mmap_mode="r")
                out = np.empty(m.shape, m.dtype)
                step = max(1, len(m) // 256)
                with ThreadPoolExecutor(32) as ex:
                    list(ex.map(lambda i: out.__setitem__(slice(i, i + step), m[i:i + step]), range(0, len(m), step)))
                arrays[k] = out
            print(f"packed store {pdir}: read {arrays['ent'].nbytes / 1e9:.1f} GB in {_t.time() - t0:.0f}s", flush=True)
            return Store.from_arrays(arrays, spans, device, share=device.type == "cpu")
    if os.name == "posix" and os.uname().sysname == "Linux":
        st = _load_store_anon(jobs, spans, shapes, dtypes, device, workers)
        import shutil
        need = st.ent.numel() * 4 * 1.05
        if pdir is not None and os.environ.get("RANK", "0") != "0":
            pdir = None                                   # multi-GPU: only rank 0 may write the packed copy
        if pdir is not None and device.type == "cpu" and shutil.disk_usage(Path(packed).parent if not Path(packed).exists() else packed).free < need:
            print(f"packed store skipped: {need / 1e9:.0f} GB needed, not enough free disk", flush=True)
            pdir = None
        if pdir is not None and device.type == "cpu":
            import time as _t
            t0 = _t.time()
            pdir.mkdir(parents=True, exist_ok=True)
            np.save(pdir / "ent.npy", st.ent.numpy())
            np.save(pdir / "act.npy", st.act.cpu().numpy().astype(np.uint8))
            np.save(pdir / "tick.npy", st.tick.cpu().numpy().astype(np.int32))
            np.save(pdir / "cidx.npy", st.cidx.cpu().numpy().astype(np.int32))
            (pdir / "done").write_text("ok")
            print(f"packed store written to {pdir} in {_t.time() - t0:.0f}s", flush=True)
        return st
    shms = {k: shared_memory.SharedMemory(create=True, size=max(1, int(np.prod(shapes[k])) * np.dtype(dtypes[k]).itemsize))
            for k in shapes}
    names = {k: v.name for k, v in shms.items()}
    try:
        arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=shms[k].buf) for k in shapes}
        arrays["act"][:] = 0
        workers = workers or min(len(jobs), usable_cpus(), int(os.environ.get("LOAD_WORKERS", "1000")))   # small-RAM pods
        ctx = get_context("fork") if os.name == "posix" and os.uname().sysname == "Linux" else get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_attach,
                                 initargs=(names, shapes, dtypes)) as ex:
            list(ex.map(_load_one, jobs, chunksize=1))
        return Store.from_arrays(arrays, spans, device)
    finally:
        for v in shms.values():
            v.close()
            v.unlink()


def _load_store_anon(jobs, spans, shapes, dtypes, device, workers):
    """Linux: decompress into anonymous shared mappings made before the workers fork.
    Unlike /dev/shm (a tmpfs capped at e.g. 117 GB on RunPod), these are limited only by
    the container's memory, and a CPU store uses them in place: no second copy."""
    import mmap
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    mms = {k: mmap.mmap(-1, max(1, int(np.prod(shapes[k])) * np.dtype(dtypes[k]).itemsize)) for k in shapes}
    arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=mms[k]) for k in shapes}
    _SHM.clear()
    _SHM.update({k: (mms[k], arrays[k]) for k in shapes})          # inherited by the forked workers
    try:
        workers = workers or min(len(jobs), usable_cpus(), int(os.environ.get("LOAD_WORKERS", "1000")))   # small-RAM pods
        with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
            list(ex.map(_load_one, jobs, chunksize=1))
    finally:
        _SHM.clear()
    st = Store.from_arrays(arrays, spans, device, share=device.type == "cpu")
    if device.type == "cpu":
        st._mms = mms                   # the store's ent tensor lives in these mappings
    else:
        del arrays
        for m in mms.values():
            m.close()
    return st


class Store:
    """Every game concatenated into device tensors; windows are gathered by
    index arithmetic on the device, so training never touches the CPU.

    States of game g occupy rows off_g .. off_g + n_g (n_g + 1 states, the
    last terminal); the action row of the terminal state is zero and never
    sampled.
    """

    def __init__(self, games, device):
        self.device = device
        n_rows = sum(g.n_steps + 1 for g in games)
        self.ent = torch.empty(n_rows, N_ENT, len(F), dtype=torch.float32, device=device)
        self.act = torch.zeros(n_rows, 10, 6, dtype=torch.uint8, device=device)
        self.tick = torch.empty(n_rows, dtype=torch.long, device=device)
        self.cidx = torch.empty(n_rows, dtype=torch.long, device=device)
        self.spans = []                       # (offset, n_steps) per game
        off = 0
        for g in games:
            n = g.n_steps
            self.ent[off:off + n + 1] = torch.from_numpy(g.ent).to(device)
            self.act[off:off + n] = torch.from_numpy(g.actions).to(device)
            self.tick[off:off + n + 1] = torch.from_numpy(g.tick).to(device)
            self.cidx[off:off + n + 1] = torch.from_numpy(g.creep_idx).to(device)
            self.spans.append((off, n))
            off += n + 1
        self._starts = {}

    @classmethod
    def from_arrays(cls, a, spans, device, chunk=8192, share=False):
        """Build from contiguous host arrays (see load_store), copying in chunks; with
        share (CPU store) the entity tensor uses the host array's memory in place."""
        st = cls.__new__(cls)
        st.device, st.spans, st._starts = device, spans, {}
        n = len(a["ent"])
        if share:
            st.ent = torch.from_numpy(a["ent"])
        else:
            st.ent = torch.empty(n, N_ENT, len(F), dtype=torch.float32, device=device)
            for i in range(0, n, chunk):
                st.ent[i:i + chunk] = torch.from_numpy(a["ent"][i:i + chunk]).to(device)
        st.act = torch.from_numpy(a["act"].copy()).to(device)   # copy: the shared memory is freed after
        st.tick = torch.from_numpy(a["tick"].astype(np.int64)).to(device)
        st.cidx = torch.from_numpy(a["cidx"].astype(np.int64)).to(device)
        return st

    @property
    def n_steps(self) -> int:
        return sum(n for _, n in self.spans)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.ent, self.act, self.tick, self.cidx))

    def starts(self, K):
        if K not in self._starts:
            s = [torch.arange(off, off + n - K + 1) for off, n in self.spans if n >= K]
            self._starts[K] = torch.cat(s).to(self.device)
        return self._starts[K]

    def event_rows(self):
        """Rows t whose step t -> t+1 contains a rare event the model must learn:
        a hero casts Q/W/E (a cooldown jumps up), a hero or tower takes damage,
        or a unit dies."""
        if not hasattr(self, "_events"):
            ev = torch.zeros(len(self.ent), dtype=torch.bool, device=self.device)
            for i in range(0, len(self.ent) - 1, 16384):
                e0, e1 = self.ent[i:i + 16384], self.ent[i + 1:i + 16385]
                n = len(e1)
                e0 = e0[:n]
                cast = ((e1[:, :10, [F["q_timer"], F["w_timer"], F["e_timer"]]]
                         > e0[:, :10, [F["q_timer"], F["w_timer"], F["e_timer"]]])).any(-1).any(-1)
                hit = (e1[:, :10, F["health"]] < e0[:, :10, F["health"]]).any(-1) | \
                      (e1[:, 182:, F["health"]] < e0[:, 182:, F["health"]]).any(-1)
                died = ((e0[..., F["alive"]] > 0) & (e1[..., F["alive"]] == 0)).any(-1)
                ev[i:i + n] = cast | hit | died
            for off, n in self.spans:           # the terminal row has no step
                ev[off + n] = False
            self._events = ev
        return self._events

    def event_starts(self, K, burn):
        """Window starts whose loss steps (burn .. K-1) contain an event."""
        key = ("ev", K, burn)
        if key not in self._starts:
            ev = self.event_rows().long()
            csum = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), ev.cumsum(0)])
            st = self.starts(K)
            has = csum[st + K] - csum[st + burn] > 0
            self._starts[key] = st[has]
        return self._starts[key]

    def sample(self, B, K, gen, event_frac=0.0, burn=0):
        st = self.starts(K)
        n_ev = int(round(B * event_frac))
        i = st[torch.randint(len(st), (B - n_ev,), device=self.device, generator=gen)]
        if n_ev:
            es = self.event_starts(K, burn)
            i = torch.cat([i, es[torch.randint(len(es), (n_ev,), device=self.device, generator=gen)]])
        idx = i[:, None] + torch.arange(K + 1, device=self.device)
        out = self.ent[idx], self.act[idx[:, :K]].long(), self.tick[idx], self.cidx[idx]
        dev = getattr(self, "out_dev", None)            # store in CPU RAM, consumer on the GPU (--store-cpu)
        return tuple(t.to(dev) for t in out) if dev is not None else out


# ---------------------------------------------------------------------------
# Statistics fitted on training games: loss scales, supervision masks, and the
# lookups that fill derived fields in rollouts.
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    reg_scale: torch.Tensor     # (4 kinds, n_reg) RMS of non-zero deltas
    reg_mask: torch.Tensor      # (4, n_reg) bool: this field changes for this kind
    change_rate: torch.Tensor   # (4, n_reg) fraction of supervised steps where the field changes
    alive_mask: torch.Tensor    # (4,) bool: alive can flip for this kind
    hero_stats: torch.Tensor    # (10 slots, 31 levels, 3): max_health, max_mana, damage; NaN = unseen
    creep_const: dict           # field -> value for alive creeps (constants set by spawn_creep)
    creep_spawn: np.ndarray     # (6 lanes, 2) mean (x, y) of a freshly spawned creep
    neutral_spawn: np.ndarray   # (72, 2) mean (x, y) of a freshly spawned neutral
    regen_rule: bool = False    # hero +2/tick health & mana regen applied by rule, not predicted


def regen_part(ent):
    """(..., 206, n_reg) the change the game's regeneration makes this tick, in
    REG_FIELDS columns: heroes below their maximum regain +2 health and +2 mana
    every tick (step_players), capped at the maximum. Zero elsewhere."""
    out = torch.zeros(*ent.shape[:-1], len(REG_FIELDS), dtype=ent.dtype, device=ent.device)
    h = ent[..., :10, :]
    out[..., :10, REG_FIELDS.index("health")] = (h[..., F["max_health"]] - h[..., F["health"]]).clamp(0, 2)
    out[..., :10, REG_FIELDS.index("mana")] = (h[..., F["max_mana"]] - h[..., F["mana"]]).clamp(0, 2)
    return out


@torch.no_grad()
def fit_stats(store: "Store", chunk: int = 16384, regen_rule: bool = False) -> Stats:
    """Fit loss scales, supervision masks and rollout lookups on the training
    Store, on its device, vectorized in chunks of consecutive (s_t, s_{t+1})
    pairs. Wave respawns are excluded from delta statistics (they are rules)."""
    dev = store.device
    kind = torch.from_numpy(KIND).to(dev)
    onehot = torch.nn.functional.one_hot(kind, N_KIND).double()          # (206, 4)
    rows = torch.cat([torch.arange(off, off + n, device=dev) for off, n in store.spans])
    nreg = len(REG_IDX)
    sumsq = torch.zeros(N_KIND, nreg, dtype=torch.float64, device=dev)
    cnt = torch.zeros(N_KIND, nreg, dtype=torch.float64, device=dev)
    total = torch.zeros(N_KIND, dtype=torch.float64, device=dev)
    flips = torch.zeros(N_KIND, dtype=torch.float64, device=dev)
    hero = torch.full((10, 31, 3), float("nan"), device=dev)
    c_sum = torch.zeros(6, 2, dtype=torch.float64, device=dev); c_n = torch.zeros(6, dtype=torch.float64, device=dev)
    n_sum = torch.zeros(72, 2, dtype=torch.float64, device=dev); n_n = torch.zeros(72, dtype=torch.float64, device=dev)
    creep_const = None
    reg_idx = torch.tensor(REG_IDX, device=dev)
    slot = torch.arange(10, device=dev)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        e0, e1 = store.ent[r], store.ent[r + 1]
        tick, cidx = store.tick[r], store.cidx[r]
        a0, a1 = e0[..., F["alive"]] > 0, e1[..., F["alive"]] > 0
        wc, wlane, wn = wave_respawns(tick, cidx)
        rule = torch.zeros_like(a0)
        rule[:, 10:110] = wc
        rule[:, 110:182] = wn.unsqueeze(-1)
        d = (e1[..., reg_idx] - e0[..., reg_idx]).double()                 # (T, 206, nreg)
        if regen_rule:
            d = d - regen_part(e0).double()
        valid = ((a0 & a1 & ~rule).unsqueeze(-1) & (d != 0)).double()
        sumsq += torch.einsum("tej,ek->kj", d * d * valid, onehot)
        cnt += torch.einsum("tej,ek->kj", valid, onehot)
        total += torch.einsum("te,ek->k", (a0 & a1 & ~rule).double(), onehot)
        flips += torch.einsum("te,ek->k", (a0 != a1).double(), onehot)
        # hero max stats by (slot, level): constant per key, so any row will do
        lv = e0[:, :10, F["level"]].long().clamp(0, 30)
        vals = torch.stack([e0[:, :10, F[f]] for f in ("max_health", "max_mana", "damage")], -1)
        hero[slot.expand_as(lv).reshape(-1), lv.reshape(-1)] = vals.reshape(-1, 3)
        # spawn points: creeps that appear (from dead, or reused by a wave), neutrals likewise
        sc = (~a0[:, 10:110] & a1[:, 10:110]) | wc
        lane1 = e1[:, 10:110, F["lane"]].long().clamp(0, 5)
        xy1 = torch.stack([e1[..., F["x"]], e1[..., F["y"]]], -1).double()
        c_sum.index_add_(0, lane1[sc], xy1[:, 10:110][sc]); c_n.index_add_(0, lane1[sc], torch.ones_like(lane1[sc], dtype=torch.float64))
        sn = (~a0[:, 110:182] & a1[:, 110:182]) | wn.unsqueeze(-1)
        n_sum += (xy1[:, 110:182] * sn.unsqueeze(-1)).sum(0); n_n += sn.sum(0)
        if creep_const is None and a1[:, 10:110].any():
            alive_rows = e1[:, 10:110][a1[:, 10:110]]
            creep_const = {f: float(alive_rows[:, F[f]].median())
                           for f in ("entity_type", "basic_attack_cd", "xp_on_kill", "damage", "max_health")}
    mask = cnt > 10
    scale = torch.where(mask, (sumsq / cnt.clamp(min=1)).sqrt(), torch.ones_like(sumsq)).clamp(min=1e-3)
    rate = (cnt / total.clamp(min=1).unsqueeze(-1)).float()
    return Stats(scale.float().cpu(), mask.cpu(), rate.cpu(), (flips > 0).cpu(), hero.cpu(), creep_const,
                 (c_sum / c_n.clamp(min=1).unsqueeze(-1)).float().cpu().numpy(),
                 (n_sum / n_n.clamp(min=1).unsqueeze(-1)).float().cpu().numpy(), regen_rule)


# ---------------------------------------------------------------------------
# Featurization (torch): raw entity array -> typed inputs for the tokenizer.
# ---------------------------------------------------------------------------

# --- who attacks whom (the attack graph) --------------------------------------
# ATTACK_MODE None: no attack inputs (E1-E13). "derived": each unit's target from the game's
# own targeting rule on positions (moba.h: heroes nearest enemy in their 11x11 box, towers
# nearest non-neutral enemy within 5, creeps within 7 and hitting at L1 < 2, neutrals heroes
# within 3). "real": the recorded target_pid where the state has any (real states), else derived
# (model-generated states, whose target_pid rule R9 blanks).
ATTACK_MODE = None
ATTACK_DIM = 12   # what I attack (has target, target kind x4, distance) + who attacks me (count, damage, by hero/creep/neutral/tower)
_KIND_T = torch.from_numpy(KIND)
_BOX = torch.tensor([5.0, 7.0, 3.0, 5.0])                    # per kind: hero, creep, neutral, tower


def derived_targets(ent):
    """(..., 206, F) -> (..., 206) long: the unit each one would attack (-1 none)."""
    kind = _KIND_T.to(ent.device)
    x, y, team = ent[..., F["x"]], ent[..., F["y"]], ent[..., F["team"]]
    alive = ent[..., F["alive"]] > 0
    dx = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs()
    dy = (y.unsqueeze(-1) - y.unsqueeze(-2)).abs()
    box = _BOX.to(ent.device)[kind]                                            # (206,)
    ok = (torch.maximum(dx, dy) <= box.unsqueeze(-1)) & alive.unsqueeze(-1) & alive.unsqueeze(-2)
    ok &= team.unsqueeze(-1) != team.unsqueeze(-2)
    tk = kind.view(*([1] * (ent.dim() - 2)), 1, -1)
    ak = kind.view(*([1] * (ent.dim() - 2)), -1, 1)
    ok &= ~(((ak == 1) | (ak == 3)) & (tk == 2))                              # creeps, towers: no neutrals
    ok &= ~((ak == 2) & (tk != 0))                                             # neutrals: heroes only
    l1 = torch.where(ok, dx + dy, torch.full_like(dx, 1e9))
    d, j = l1.min(-1)
    return torch.where(d < 1e9, j, torch.full_like(j, -1))


def attack_inputs(ent, tgt):
    """(..., 206, F), (..., 206) targets -> (..., 206, ATTACK_DIM): what I attack, who attacks me."""
    kind = _KIND_T.to(ent.device)
    has = tgt >= 0
    t = tgt.clamp(min=0)
    tkind = kind[t] * has + (-1) * (~has)
    gx = torch.gather(ent[..., F["x"]], -1, t)
    gy = torch.gather(ent[..., F["y"]], -1, t)
    dist = ((ent[..., F["x"]] - gx).abs() + (ent[..., F["y"]] - gy).abs()) * has
    oh = torch.stack([(tkind == k) for k in range(4)], -1).float()
    src = has.float()
    n_att = torch.zeros_like(src).scatter_add_(-1, t, src)
    dmg = torch.zeros_like(src).scatter_add_(-1, t, src * ent[..., F["damage"]])
    by = [torch.zeros_like(src).scatter_add_(-1, t, src * (kind == k).float()) for k in range(4)]
    return torch.stack([src, *oh.unbind(-1), (dist / 8).clamp(max=4), n_att / 5, dmg / 300, by[0] / 5,
                        by[1] / 5, by[2] / 5, by[3].clamp(max=1)], -1)


def rebuild_kill_edges(ent, chunk=1024):
    """In place on consecutive states of ONE game (T, 206, F): the game records no target_pid for a
    killing blow (moba.h attack(): the kill branch returns before setting it). Rebuild it: a unit whose
    basic-attack timer was <= 1 and is reset to its full cooldown at t+1, with no recorded target, hit
    the nearest enemy that died on t -> t+1 (alive 1 -> 0, or a hero respawn jump) within 12 cells
    (82% of such resets match one, ~4% ambiguous; 3 held-out games). Returns the number of edges added."""
    kind = _KIND_T.to(ent.device)
    added = 0
    for lo in range(0, len(ent) - 1, chunk):
        hi = min(len(ent) - 1, lo + chunk)
        a, b = ent[lo:hi], ent[lo + 1:hi + 1]
        al0, al1 = a[..., F["alive"]] > 0, b[..., F["alive"]] > 0
        jump = (b[..., F["x"]] - a[..., F["x"]]).abs() + (b[..., F["y"]] - a[..., F["y"]]).abs()
        respawn = (kind == 0) & (jump > 12) & (b[..., F["health"]] >= b[..., F["max_health"]] - 0.5)
        died = (al0 & ~al1) | (al0 & respawn)
        fired = al0 & al1 & (b[..., F["target_pid"]] < 0) & (a[..., F["basic_attack_timer"]] <= 1) & \
            (b[..., F["basic_attack_timer"]] == b[..., F["basic_attack_cd"]])
        rows = torch.nonzero(fired.any(-1) & died.any(-1)).flatten()
        if len(rows) == 0:
            continue
        a_, d_, f_ = a[rows], died[rows], fired[rows]
        x, y, team = a_[..., F["x"]], a_[..., F["y"]], a_[..., F["team"]]
        l1 = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs() + (y.unsqueeze(-1) - y.unsqueeze(-2)).abs()
        ok = d_.unsqueeze(-2) & (team.unsqueeze(-1) != team.unsqueeze(-2)) & (l1 <= 12)
        dist, j = torch.where(ok, l1, torch.full_like(l1, 1e9)).min(-1)
        new = f_ & (dist < 1e9)
        tp = b[rows][..., F["target_pid"]]
        b[rows, :, F["target_pid"]] = torch.where(new, j.to(tp.dtype), tp)
        added += int(new.sum())
    return added


def rebuild_kill_edges_store(store, dev=None):
    """rebuild_kill_edges over every game of a Store, computed on `dev` (default: the store's device)."""
    n = 0
    for off, steps in store.spans:
        g = store.ent[off:off + steps + 1]
        x = g.to(dev) if dev is not None else g
        n += rebuild_kill_edges(x)
        if x is not g:
            g[..., F["target_pid"]] = x[..., F["target_pid"]].to(g.device)
    return n


# --- stored target lists (LIST_STATE) -------------------------------------------
# The game keeps, per unit, a stored target list (env->scanned_targets) and attacks the nearest unit in it.
# Towers rescan every 3 ticks, creeps every 5; in between the list is stale. The dense files record it
# (state['scan']), the 26 entity fields don't. With LIST_STATE the lists of creeps and towers (OWNERS) over
# heroes, creeps and towers (MEMBERS) become 6 extra entity fields (24 membership bits each), so every data
# path and the dream carry them; the model predicts next-tick membership (no rule), and the featurizer
# derives aggro inputs from them (who each unit will hit, whether it is ready, incoming damage).
LIST_STATE = False
N_BASE = len(F)                                               # 26 recorded entity fields
LIST_NF = 6
OWNERS = np.r_[10:110, 182:206]                               # creeps, towers (124)
MEMBERS = np.r_[0:110, 182:206]                               # heroes, creeps, towers (134)
_MIDX = np.full(256, len(MEMBERS), np.int64)                  # pid -> member column (134 = not a member)
_MIDX[MEMBERS] = np.arange(len(MEMBERS))
AGGRO_DIM = 13


def enable_list_state():
    """Add the list fields to the shared field map (call before any store or model is built)."""
    global LIST_STATE
    for k in range(LIST_NF):
        F.setdefault(f"lst{k}", N_BASE + k)
    LIST_STATE = True


def pack_lists_np(scan):
    """(..., 206, 64) uint8 MOBA_NONE-terminated lists -> (..., 206, LIST_NF) float32 membership bits."""
    sc = scan[..., OWNERS, :]
    valid = np.cumprod(sc != 255, -1).astype(bool)
    col = np.where(valid, _MIDX[sc], len(MEMBERS))
    bits = np.zeros(sc.shape[:-1] + (len(MEMBERS) + 1,), bool)
    np.put_along_axis(bits, col, True, -1)
    bits = np.pad(bits[..., :len(MEMBERS)], [(0, 0)] * (bits.ndim - 1) + [(0, 24 * LIST_NF - len(MEMBERS))])
    w = (2 ** np.arange(24)).astype(np.float64)
    packed = (bits.reshape(*bits.shape[:-1], LIST_NF, 24) * w).sum(-1).astype(np.float32)
    out = np.zeros(scan.shape[:-1] + (LIST_NF,), np.float32)
    out[..., OWNERS, :] = packed
    return out


def unpack_lists(ent):
    """(..., 206, F) -> (..., 124, 134) bool membership."""
    v = ent[..., torch.as_tensor(OWNERS, device=ent.device), N_BASE:N_BASE + LIST_NF].long()
    bits = (v.unsqueeze(-1) >> torch.arange(24, device=ent.device)) & 1
    return bits.flatten(-2)[..., :len(MEMBERS)].bool()


def pack_lists(bits):
    """(..., 124, 134) bool -> (..., 124, LIST_NF) float."""
    b = Fn_pad(bits.long(), (0, 24 * LIST_NF - len(MEMBERS)))
    w = 1 << torch.arange(24, device=bits.device)
    return (b.unflatten(-1, (LIST_NF, 24)) * w).sum(-1).float()                # < 2^24: exact in float32


def Fn_pad(x, pad):
    return torch.nn.functional.pad(x, pad)


def aggro_inputs(ent, tick):
    """(..., 206, F), (...) -> (..., 206, AGGRO_DIM): whom each unit will hit (creeps/towers: nearest alive unit
    in its stored list; heroes/neutrals: nearest enemy in their box, as derived_targets), whether it is ready
    (attack timer <= 1, in range), incoming damage from ready attackers, how many / whether a tower aim at me,
    list size, AoE threat (towers: levels of enemy heroes within 2 cells with an ability ready), rescan next."""
    dev = ent.device
    kind = _KIND_T.to(dev)
    own, mem = torch.as_tensor(OWNERS, device=dev), torch.as_tensor(MEMBERS, device=dev)
    x, y, alive = ent[..., F["x"]], ent[..., F["y"]], ent[..., F["alive"]] > 0
    L = unpack_lists(ent)
    d = (x[..., own].unsqueeze(-1) - x[..., mem].unsqueeze(-2)).abs() + (y[..., own].unsqueeze(-1) - y[..., mem].unsqueeze(-2)).abs()
    d = torch.where(L & alive[..., mem].unsqueeze(-2), d, torch.full_like(d, 1e9))
    dm, j = d.min(-1)
    tgt = derived_targets(ent)
    tgt[..., own] = torch.where(dm < 1e9, mem[j], torch.full_like(j, -1))
    has = tgt >= 0
    t = tgt.clamp(min=0)
    gx, gy = torch.gather(x, -1, t), torch.gather(y, -1, t)
    dist = ((x - gx).abs() + (y - gy).abs()) * has
    melee = (kind == 1) | (kind == 2)
    ready = alive & has & (ent[..., F["basic_attack_timer"]] <= 1) & torch.where(melee, dist < 2, torch.ones_like(has))
    src = ready.float()
    inc = torch.zeros_like(src).scatter_add_(-1, t, src * ent[..., F["damage"]])
    on = (has & alive).float()
    n_on = torch.zeros_like(src).scatter_add_(-1, t, on)
    tw_on = torch.zeros_like(src).scatter_add_(-1, t, on * (kind == 3).float()).clamp(max=1)
    oh = torch.stack([(kind[t] == k) & has for k in range(4)], -1).float()
    size = torch.zeros_like(src)
    size[..., own] = L.sum(-1).float()
    # AoE threat on towers
    hx, hy, hteam = x[..., :10], y[..., :10], ent[..., :10, F["team"]]
    tx, ty, tteam = x[..., 182:], y[..., 182:], ent[..., 182:, F["team"]]
    near = (torch.maximum((hx.unsqueeze(-2) - tx.unsqueeze(-1)).abs(), (hy.unsqueeze(-2) - ty.unsqueeze(-1)).abs()) <= 2) & \
        (hteam.unsqueeze(-2) != tteam.unsqueeze(-1)) & alive[..., :10].unsqueeze(-2)
    abil = (ent[..., :10, F["q_timer"]] <= 0) | (ent[..., :10, F["w_timer"]] <= 0) | (ent[..., :10, F["e_timer"]] <= 0)
    aoe = torch.zeros_like(src)
    aoe[..., 182:] = (near * (abil * ent[..., :10, F["level"]]).unsqueeze(-2)).sum(-1)
    tk = tick.unsqueeze(-1)
    rescan = torch.where(kind == 3, (tk + 1) % 3 == 0, torch.where(melee, (tk + 1) % 5 == 0, torch.ones_like(has))).float()
    return torch.stack([has.float(), *oh.unbind(-1), (dist / 8).clamp(max=4), ready.float(), inc / 300, n_on / 5, tw_on,
                        size / 10, aoe / 10, rescan], -1)


class Featurizer:
    def __init__(self, device):
        self.device = device
        walls = torch.from_numpy(load_walls())
        self.walls = torch.nn.functional.pad(walls, (4, 4, 4, 4), value=1.0).to(device)
        r = torch.arange(-4, 5, device=device)
        self.dy, self.dx = torch.meshgrid(r, r, indexing="ij")
        self.kind = torch.from_numpy(KIND).to(device)

    def __call__(self, ent, tick, cidx):
        """ent (...,206,26), tick (...), cidx (...) -> dict of feature tensors."""
        f = lambda name: ent[..., F[name]]
        hp_frac = f("health") / f("max_health").clamp(min=1)
        shared = torch.stack([
            f("alive"), hp_frac, f("health") / 1000, f("damage") / 100, f("move_modifier") / 2,
            f("stun_timer") / 20, f("move_timer") / 30, f("basic_attack_timer") / 10, f("is_hit"),
        ], -1)
        pos = torch.stack([f("x") / 128, f("y") / 128], -1)
        atk = None
        if ATTACK_MODE:
            tgt = derived_targets(ent)
            if ATTACK_MODE == "state":                                          # the state's own targets: real or predicted
                tgt = ent[..., F["target_pid"]].long()
            elif ATTACK_MODE == "real":
                real = ent[..., F["target_pid"]].long()
                has_real = (real >= 0).any(-1, keepdim=True)                   # a real state records targets
                tgt = torch.where(has_real, real, tgt)
            atk = attack_inputs(ent, tgt)
        h = ent[..., HEROES, :]
        g = lambda name: h[..., F[name]]
        hero = torch.stack([
            g("mana") / g("max_mana").clamp(min=1), g("level") / 30, g("xp") / 5000,
            g("max_health") / 2000, g("q_timer") / 100, g("w_timer") / 100, g("e_timer") / 100,
        ], -1)
        creep = (ent[..., S.CREEPS, F["waypoint"]] / 12).unsqueeze(-1)
        tower = (ent[..., S.TOWERS, F["max_health"]] / 4500).unsqueeze(-1)
        team = f("team").long().clamp(0, 2)
        lane = ent[..., S.CREEPS, F["lane"]].long().clamp(0, 5)
        # 9x9 wall patch around each entity (dead entities sit at (0,0): walls)
        yi = f("y").long().clamp(0, 127) + 4
        xi = f("x").long().clamp(0, 127) + 4
        patch = self.walls[(yi[..., None, None] + self.dy), (xi[..., None, None] + self.dx)]
        patch = patch.flatten(-2)
        t = tick.float()
        glob = torch.stack([
            torch.sin(2 * np.pi * t / 150), torch.cos(2 * np.pi * t / 150),
            torch.sin(2 * np.pi * t / 600), torch.cos(2 * np.pi * t / 600),
            *[(tick % 3 == k).float() for k in range(3)],
            *[(tick % 5 == k).float() for k in range(5)],
            cidx.float() / 100, t / 5000,
        ], -1)
        # For action tokens: what a hero's ability press can actually do this tick.
        # Targets are enemies in the 11x11 box around the hero (int cells, as upstream
        # scan_aoe); target mode 2 keeps only heroes and towers.
        ix, iy = f("x").long(), f("y").long()
        hx, hy = ix[..., :10], iy[..., :10]
        inbox = ((ix.unsqueeze(-2) - hx.unsqueeze(-1)).abs() <= 5) & \
                ((iy.unsqueeze(-2) - hy.unsqueeze(-1)).abs() <= 5)            # (..., 10, 206)
        enemy = (team.unsqueeze(-2) != team[..., :10].unsqueeze(-1)) & (f("alive") > 0).unsqueeze(-2)
        heroes_towers = (self.kind == 0) | (self.kind == 3)
        tgt_any = (inbox & enemy).any(-1)
        tgt_ht = (inbox & enemy & heroes_towers).any(-1)
        # area abilities need an enemy within their own radius (a box, as scan_aoe)
        dxy = torch.maximum((ix.unsqueeze(-2) - hx.unsqueeze(-1)).abs(), (iy.unsqueeze(-2) - hy.unsqueeze(-1)).abs())
        near = torch.stack([((dxy <= r) & enemy).any(-1) for r in (2, 3, 4)], -1)   # (..., 10, 3)
        # the automatic target is the nearest enemy in the box (L1); assassin Q needs it to be a creep/neutral
        l1 = (f("x").unsqueeze(-2) - f("x")[..., :10].unsqueeze(-1)).abs() + \
             (f("y").unsqueeze(-2) - f("y")[..., :10].unsqueeze(-1)).abs()
        l1 = torch.where(inbox & enemy, l1, torch.full_like(l1, 1e9))
        nearest_kind = self.kind[l1.argmin(-1)]
        tgt_minion = (nearest_kind == 1) | (nearest_kind == 2)
        hero_raw = torch.stack([g("mana"), g("max_mana"), g("q_timer"), g("w_timer"), g("e_timer"),
                                g("stun_timer")], -1)
        out = dict(shared=shared, pos=pos, hero=hero, creep=creep, tower=tower,
                   team=team, lane=lane, patch=patch, glob=glob, alive=f("alive") > 0,
                   xy=torch.stack([f("x"), f("y")], -1), hero_raw=hero_raw,
                   tgt_any=tgt_any, tgt_ht=tgt_ht, near=near, tgt_minion=tgt_minion)
        if atk is not None:
            out["atk"] = atk
        if LIST_STATE:
            out["aggro"] = aggro_inputs(ent, tick)
            out["lists"] = unpack_lists(ent)
            out["tick"] = tick
        return out


# ---------------------------------------------------------------------------
# Rollout: apply predictions to a raw state, filling derived fields by rule.
# ---------------------------------------------------------------------------

def level_from_xp(xp: torch.Tensor) -> torch.Tensor:
    """upstream calc_level: first i with xp < XP_FOR_LEVEL[i] gives level i + 1."""
    table = torch.from_numpy(XP_FOR_LEVEL).to(xp.device)
    below = xp.unsqueeze(-1) < table
    first = torch.where(below.any(-1), below.float().argmax(-1), torch.full_like(xp, 29).long())
    return (first + 1).float()


def wave_respawns(tick, cidx):
    """Units that upstream respawns *in place* during the step from tick t.

    Every 150 ticks step_creeps kills and respawns 30 creep slots, lane by
    lane: slot (creep_idx + 5*lane + i) % 100 gets `lane`, i = 0..4, whether
    or not the old creep there was still alive. Every 600 ticks every neutral
    is respawned at its camp. Both are fixed functions of (tick, creep_idx),
    so they are applied by rule, not predicted.

    Returns creep (..., 100) bool, creep_lane (..., 100) long, neutral (...,) bool.
    """
    dev = tick.device
    wave = (tick % 150 == 0)
    k = torch.arange(30, device=dev)
    slots = (cidx.unsqueeze(-1) + k) % 100                          # (..., 30)
    creep = torch.zeros(*tick.shape, 100, dtype=torch.bool, device=dev)
    creep.scatter_(-1, slots, wave.unsqueeze(-1).expand_as(slots))
    lane = torch.zeros(*tick.shape, 100, dtype=torch.long, device=dev)
    lane.scatter_(-1, slots, (k // 5).expand_as(slots))
    return creep, lane, (tick % 600 == 0)


TOWER_DEATH_RULE = False   # smoke-test rule: False = off; a number x = tower dies when predicted health <= x * max (0 = at 0)
XP_MONOTONE = True   # rule R7 "XP never decreases" in the dream's state update (--no-xp-clamp turns it off)


def apply_prediction(ent, tick, cidx, pred, stats: Stats, featurizer: Featurizer):
    """Next raw state from the current one and the model's predictions (no grad)."""
    dev = ent.device
    kind = featurizer.kind
    new = ent.clone()
    scale = stats.reg_scale.to(dev)[kind]                       # (206, n_reg)
    mask = stats.reg_mask.to(dev)[kind]
    delta = pred["reg"] * scale * mask
    if stats.regen_rule:
        delta = delta + regen_part(ent)
    for j, name in enumerate(REG_FIELDS):
        v = ent[..., F[name]] + delta[..., j]
        if name in INT_FIELDS:
            v = v.round()
        new[..., F[name]] = v
    for name in ("q_timer", "w_timer", "e_timer", "stun_timer", "move_timer", "basic_attack_timer",
                 "waypoint", "move_modifier", "mana"):
        new[..., F[name]] = new[..., F[name]].clamp(min=0)
    if XP_MONOTONE:
        new[..., F["xp"]] = torch.maximum(new[..., F["xp"]], ent[..., F["xp"]])
    new[..., F["mana"]] = torch.minimum(new[..., F["mana"]], new[..., F["max_mana"]])
    new[..., F["health"]] = new[..., F["health"]].clamp(min=0)
    new[..., F["x"]] = new[..., F["x"]].clamp(0, 127.99)
    new[..., F["y"]] = new[..., F["y"]].clamp(0, 127.99)
    new[..., F["is_hit"]] = (torch.sigmoid(pred["is_hit"]) > 0.5).float()

    # creep lane / team
    c = slice(10, 110)
    new[..., c, F["lane"]] = pred["lane"].argmax(-1).float()
    new[..., c, F["team"]] = pred["team"].argmax(-1).float()

    # alive flips
    can_flip = stats.alive_mask.to(dev)[kind]
    flip = (torch.sigmoid(pred["alive_flip"]) > 0.5) & can_flip
    alive0 = ent[..., F["alive"]] > 0
    alive1 = alive0 ^ flip
    if TOWER_DEATH_RULE:
        # RULE (smoke test, author-approved 2026-09-29): a tower is alive iff its predicted health > 0;
        # the learned alive flip is ignored for towers, and a dead tower never comes back
        t = slice(182, 206)
        alive1 = alive1.clone()
        frac = new[..., t, F["health"]] / new[..., t, F["max_health"]].clamp(min=1)
        alive1[..., t] = alive0[..., t] & (frac > float(TOWER_DEATH_RULE))      # threshold: fraction of max health
    new[..., F["alive"]] = alive1.float()
    died = alive0 & ~alive1
    spawned = ~alive0 & alive1
    for name in ("x", "y", "health"):
        new[..., F[name]] = torch.where(died, torch.zeros_like(new[..., F[name]]), new[..., F[name]])
    if "list_logits" in pred:            # predicted stored target lists (LIST_STATE); dead owners keep none
        bits = (pred["list_logits"] > 0) & alive1[..., torch.as_tensor(OWNERS, device=dev)].unsqueeze(-1)
        new[..., torch.as_tensor(OWNERS, device=dev), N_BASE:N_BASE + LIST_NF] = pack_lists(bits)
    if "target_logits" in pred:          # predicted attack graph: who each unit attacks this tick
        tg = pred["target_logits"].argmax(-1)
        tg = torch.where((tg >= N_ENT) | ~alive1, torch.full_like(tg, -1), tg)
        new[..., F["target_pid"]] = tg.to(new.dtype)
    else:
        new[..., F["target_pid"]] = -1   # rule R9 (no attack graph in the dream)

    # wave respawns (rule): reused creep slots and all neutrals come back fresh, alive
    wc, wlane, wn = wave_respawns(tick, cidx)
    new[..., c, F["lane"]] = torch.where(wc, wlane.float(), new[..., c, F["lane"]])
    new[..., c, F["team"]] = torch.where(wc, (wlane >= 3).float(), new[..., c, F["team"]])
    alive1 = alive1.clone()
    alive1[..., c] |= wc
    alive1[..., 110:182] |= wn.unsqueeze(-1)
    new[..., F["alive"]] = alive1.float()
    spawned = (~alive0 & alive1)
    spawned[..., c] |= wc
    spawned[..., 110:182] |= wn.unsqueeze(-1)

    # creep spawn: position by lane, constants set by spawn_creep
    cs = spawned[..., c]
    spawn_xy = torch.from_numpy(stats.creep_spawn).to(dev)[new[..., c, F["lane"]].long()]
    new[..., c, F["x"]] = torch.where(cs, spawn_xy[..., 0], new[..., c, F["x"]])
    new[..., c, F["y"]] = torch.where(cs, spawn_xy[..., 1], new[..., c, F["y"]])
    alive_c = alive1[..., c]
    for name, val in stats.creep_const.items():
        new[..., c, F[name]] = torch.where(alive_c, torch.full_like(new[..., c, F[name]], val),
                                           new[..., c, F[name]])
    new[..., c, F["health"]] = torch.where(cs, new[..., c, F["max_health"]], new[..., c, F["health"]])
    new[..., c, F["grid_id"]] = torch.where(alive_c, 3 + new[..., c, F["team"]], new[..., c, F["grid_id"]])
    # neutral respawn at its camp
    n = slice(110, 182)
    ns = spawned[..., n]
    nxy = torch.from_numpy(stats.neutral_spawn).to(dev)
    new[..., n, F["x"]] = torch.where(ns, nxy[:, 0], new[..., n, F["x"]])
    new[..., n, F["y"]] = torch.where(ns, nxy[:, 1], new[..., n, F["y"]])
    new[..., n, F["health"]] = torch.where(ns, new[..., n, F["max_health"]], new[..., n, F["health"]])

    # heroes: level from XP on XP gain; max stats from (slot, level)
    h = slice(0, 10)
    gained = new[..., h, F["xp"]] > ent[..., h, F["xp"]]
    lvl = torch.where(gained, level_from_xp(new[..., h, F["xp"]]), ent[..., h, F["level"]])
    new[..., h, F["level"]] = lvl
    table = stats.hero_stats.to(dev)[torch.arange(10, device=dev), lvl.long().clamp(0, 30)]  # (...,10,3)
    for j, name in enumerate(("max_health", "max_mana", "damage")):
        v = table[..., j]
        new[..., h, F[name]] = torch.where(torch.isnan(v), new[..., h, F[name]], v)
    new[..., F["health"]] = torch.minimum(new[..., F["health"]], new[..., F["max_health"]])

    new_tick = tick + 1
    new_cidx = torch.where(tick % 150 == 0, (cidx + 30) % 100, cidx)
    return new, new_tick, new_cidx
