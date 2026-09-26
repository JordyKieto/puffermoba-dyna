"""Data for the token world model: games in memory, windows, featurization,
targets, and the game rules used to fill derived fields in rollouts.

Featurization runs in torch on raw MobaState entity arrays, so training and
open-loop rollouts use exactly the same code path (no train/rollout skew).
See docs/moba/TOKEN_WM_DESIGN.md.
"""
from __future__ import annotations

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
    _SHM["ent"][1][off:off + n] = r["state"]["entity"]
    _SHM["act"][1][off:off + n - 1] = r["actions"][:-1]
    _SHM["tick"][1][off:off + n] = r["state"]["tick"]
    _SHM["cidx"][1][off:off + n] = r["state"]["creep_idx"]
    return n


def load_store(dense_files, game_indices, device, workers=None) -> "Store":
    """Decompress the selected games with one process per core straight into
    shared memory, then copy to the device in chunks. No per-game Python
    objects, no temporary files."""
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
    shms = {k: shared_memory.SharedMemory(create=True, size=max(1, int(np.prod(shapes[k])) * np.dtype(dtypes[k]).itemsize))
            for k in shapes}
    names = {k: v.name for k, v in shms.items()}
    try:
        arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=shms[k].buf) for k in shapes}
        arrays["act"][:] = 0
        workers = workers or min(len(jobs), os.cpu_count() or 1, 160)
        ctx = get_context("fork") if os.name == "posix" and os.uname().sysname == "Linux" else get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_attach,
                                 initargs=(names, shapes, dtypes)) as ex:
            list(ex.map(_load_one, jobs, chunksize=1))
        return Store.from_arrays(arrays, spans, device)
    finally:
        for v in shms.values():
            v.close()
            v.unlink()


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
    def from_arrays(cls, a, spans, device, chunk=8192):
        """Build from contiguous host arrays (see load_store), copying in chunks."""
        st = cls.__new__(cls)
        st.device, st.spans, st._starts = device, spans, {}
        n = len(a["ent"])
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
        return self.ent[idx], self.act[idx[:, :K]].long(), self.tick[idx], self.cidx[idx]


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
        return dict(shared=shared, pos=pos, hero=hero, creep=creep, tower=tower,
                    team=team, lane=lane, patch=patch, glob=glob, alive=f("alive") > 0,
                    xy=torch.stack([f("x"), f("y")], -1), hero_raw=hero_raw,
                    tgt_any=tgt_any, tgt_ht=tgt_ht, near=near, tgt_minion=tgt_minion)


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
    new[..., F["alive"]] = alive1.float()
    died = alive0 & ~alive1
    spawned = ~alive0 & alive1
    for name in ("x", "y", "health"):
        new[..., F[name]] = torch.where(died, torch.zeros_like(new[..., F[name]]), new[..., F[name]])
    new[..., F["target_pid"]] = -1

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
