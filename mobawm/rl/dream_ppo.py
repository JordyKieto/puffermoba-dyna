"""PPO inside the world model's dream: all 10 heroes played by ONE shared policy.

The dream is a frozen TokenWM (default: E5's final checkpoint) stepping full
game states from its own predictions. Each hero sees the byte-exact ego
observation the real game would give it (mobawm.obs.ego_obs, with the tile
grid and cell->entity map rebuilt from the dreamed entities: checked byte-exact
on real states), acts through the game's 6 action heads, and is rewarded with
the game's own per-hero reward, computed from dreamed state changes:

    -0.164  when the hero dies  (a >12-cell jump with health reset to full)
    +0.0067 per XP gained
    +0.642  when a tower falls, to the nearest enemy hero within 6 cells
            (the dream has no kill credit; the last hitter is usually adjacent)

Radiant and dire are the same network, so it plays against itself and each
hero simply maximises its own reward.

Episodes start from real states of training games (8 real ticks warm the
model's memory, with the recorded actions), run `--episode` dream ticks, and
end early if an ancient falls. Resets happen only between rollouts, so a
truncated episode is bootstrapped with the critic.

    .venv/bin/python -m mobawm.rl.dream_ppo --steps 3000000 --name first
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from .. import shards as S  # noqa: E402
from ..obs import VISION, ego_obs  # noqa: E402
from ..wm.data import Featurizer, apply_prediction, load_store, load_walls  # noqa: E402

F = S.F
HEADS = S.ACTION_SIZES                    # 7, 7, 3, 2, 2, 2
R_DEATH, R_XP, R_TOWER = -0.163764, 0.00665677, 0.642119
WALLS = load_walls().reshape(-1).astype(np.uint8)


# --- dream env ----------------------------------------------------------------

class DreamEnv:
    def __init__(self, run, dense_files, games, n_envs, episode, dev, burn=8, seed=0):
        import moba_viz as V
        self.model, self.stats, _ = V.load_model(Path(run), Path(dense_files[0]), dev)
        self.model.requires_grad_(False)
        self.fz, self.dev, self.B, self.episode, self.burn = Featurizer(dev), dev, n_envs, episode, burn
        self.store = load_store(dense_files, games, torch.device("cpu"))
        self.statics = S.load_dense(dense_files[0]).statics
        self.starts = self.store.starts(burn + 1)
        self.gen = torch.Generator().manual_seed(seed)
        self.reset_all()

    @torch.no_grad()
    def reset_all(self):
        """New episodes for every env: real start state, memory warmed on `burn` real ticks."""
        B, st = self.B, self.store
        i = self.starts[torch.randint(len(self.starts), (B,), generator=self.gen)]
        rows = i[:, None] + torch.arange(self.burn + 1)
        ent, tick, cidx = st.ent[rows].to(self.dev), st.tick[rows].to(self.dev), st.cidx[rows].to(self.dev)
        act = st.act[rows[:, :self.burn]].long().to(self.dev)
        self.h = torch.zeros(B, S.N_ENTITIES, self.model.d, device=self.dev)
        for t in range(self.burn):
            _, self.h = self.model.step(self.fz(ent[:, t], tick[:, t], cidx[:, t]), act[:, t], self.h)
        self.ent, self.tick, self.cidx = ent[:, -1].clone(), tick[:, -1].clone(), cidx[:, -1].clone()
        self.last_reward = torch.zeros(B, 10, 4, device=self.dev)
        self.alive = torch.ones(B, dtype=torch.bool, device=self.dev)       # episode still running
        self.age = 0
        self.repairs = getattr(self, "repairs", 0)

    RANGES = {"team": (0, 2), "lane": (0, 5), "entity_type": (0, 3), "grid_id": (0, 15)}

    def sanitize(self, prev, nxt):
        """Guard the dream against invalid states: non-finite values fall back to the
        previous state's, categorical fields are clamped to their valid range. Counted in
        self.repairs (entries repaired), logged per update, so an unstable dream is visible."""
        bad = ~torch.isfinite(nxt)
        nxt = torch.where(bad, prev, nxt)
        n = int(bad.sum())
        for name, (lo, hi) in self.RANGES.items():
            v = nxt[..., F[name]]
            out = (v < lo) | (v > hi)
            if out.any():
                n += int(out.sum())
                nxt[..., F[name]] = v.clamp(lo, hi)
        self.repairs += n
        return nxt

    def obs_gpu(self):
        """obs() on the GPU (mobawm/rl/gpu_net.py, byte-identical): (B, 10, 510) uint8 on the env's device."""
        from .gpu_net import ego_obs_gpu
        if getattr(self, "_walls_t", None) is None:
            self._walls_t = torch.from_numpy(WALLS).to(self.dev)
        return ego_obs_gpu(self.ent, self.last_reward, self._walls_t, self.statics)

    def obs(self):
        """(B, 10, 510) uint8 ego observations of the current dreamed states."""
        e = self.ent.cpu().numpy()
        B = len(e)
        e = e.copy()
        for c in ("x", "y"):                       # keep view windows inside the map buffer
            e[:, :10, F[c]] = np.clip(e[:, :10, F[c]], VISION, 127 - VISION)
        grid = np.tile(WALLS, (B, 1))
        pids = np.full((B, 128 * 128), 255, np.uint8)
        alive = e[..., F["alive"]] > 0
        b, i = np.nonzero(alive)
        cell = e[b, i, F["y"]].astype(np.int64) * 128 + e[b, i, F["x"]].astype(np.int64)
        cell = np.clip(cell, 0, 128 * 128 - 1)
        grid[b, cell] = e[b, i, F["grid_id"]].astype(np.uint8)
        pids[b, cell] = i
        return ego_obs(e, self.last_reward.cpu().numpy(), grid, pids, self.statics)

    @torch.no_grad()
    def step(self, actions):
        """actions (B,10,6) long -> reward (B,10), done (B,) (an ancient fell this tick)."""
        pred, self.h = self.model.step(self.fz(self.ent, self.tick, self.cidx), actions, self.h)
        prev = self.ent
        nxt, self.tick, self.cidx = apply_prediction(self.ent, self.tick, self.cidx, pred, self.stats, self.fz)
        self.ent = self.sanitize(prev, nxt)
        r, comp = reward(prev, self.ent)
        self.last_reward = comp
        done = (prev[:, 204:206, F["alive"]] > 0).any(-1) & (self.ent[:, 204:206, F["alive"]] <= 0).any(-1)
        r = r * self.alive[:, None]
        done = done & self.alive
        self.alive = self.alive & ~done
        self.age += 1
        return r, done


XP_SIGNED = False    # True: the XP term is the signed XP change (noise cancels) instead of increases only
# Dream-only reward shaping (--r-tower/--tower-team/--r-win). The game itself pays only death,
# XP and +0.642 to the tower's killer; nothing for winning. These bonuses change the TOTAL the
# policy optimises; the per-component rewards (which the observation shows) stay the game's.
SHAPE = dict(r_tower=None, tower_team=False, r_win=0.0,
             dmg_k=0.0, dmg_boost=0.0, dmg_radius=8.0, fall_bonus=0.0, fall_radius=8.0, def_scale=0.0,
             # reward-structure sweep (dream only): scale the game's death / XP terms; push = progress toward the enemy
             # ancient per cell; siege = per tick within siege_radius of a living enemy tower; anc_k = enemy ancient
             # health lost (fraction) to the whole attacking team; team_share = every hero gets its team's mean;
             # zero_sum = minus the enemy team's mean (relative reward)
             xp_scale=1.0, death_scale=1.0, push_k=0.0, siege_k=0.0, siege_radius=8.0, anc_k=0.0,
             team_share=False, zero_sum=False, kill_k=0.0, kill_radius=8.0)


def reward(prev, nxt):
    """Per-hero reward (B,10) and its components (B,10,4: death, xp, distance, tower)."""
    B = prev.shape[0]
    p, n = prev[:, :10], nxt[:, :10]
    jump = (n[..., F["x"]] - p[..., F["x"]]).abs() + (n[..., F["y"]] - p[..., F["y"]]).abs()
    died = (jump > 12) & (n[..., F["health"]] >= n[..., F["max_health"]] - 0.5) & \
        (p[..., F["health"]] < p[..., F["max_health"]] - 0.5)
    xp = n[..., F["xp"]] - p[..., F["xp"]]
    if not XP_SIGNED:
        xp = xp.clamp(min=0)
    comp = torch.zeros(B, 10, 4, device=prev.device)
    comp[..., 0] = died.float() * R_DEATH
    comp[..., 1] = xp * R_XP
    tfell = (prev[:, 182:, F["alive"]] > 0) & (nxt[:, 182:, F["alive"]] <= 0)          # (B,24)
    if tfell.any():
        tx, ty, tteam = prev[:, 182:, F["x"]], prev[:, 182:, F["y"]], prev[:, 182:, F["team"]]
        dist = (p[..., F["x"]][:, :, None] - tx[:, None]).abs() + (p[..., F["y"]][:, :, None] - ty[:, None]).abs()
        enemy = (p[..., F["team"]][:, :, None] != tteam[:, None]) & (p[..., F["alive"]][:, :, None] > 0)
        dist = torch.where(enemy & (dist <= 6), dist, torch.full_like(dist, 1e9))      # (B,10,24)
        best = dist.argmin(1)                                                            # (B,24)
        ok = tfell & (dist.min(1).values < 1e9)
        credit = torch.zeros(B, 10, 24, device=prev.device)
        credit.scatter_(1, best[:, None], ok[:, None].float())
        comp[..., 3] = credit.sum(-1) * R_TOWER
    total = comp[..., 0] * SHAPE["death_scale"] + comp[..., 1] * SHAPE["xp_scale"] + comp[..., 2] + comp[..., 3]
    if SHAPE["r_tower"] is not None or SHAPE["r_win"] or SHAPE["dmg_k"] or SHAPE["fall_bonus"] or SHAPE["def_scale"]:
        total = total + shaping(prev, nxt)
    total = total + structure(prev, nxt)
    if SHAPE["team_share"] or SHAPE["zero_sum"]:
        rad, dire = total[:, :5].mean(-1, keepdim=True), total[:, 5:].mean(-1, keepdim=True)
        own = torch.cat([rad.expand(-1, 5), dire.expand(-1, 5)], -1)
        other = torch.cat([dire.expand(-1, 5), rad.expand(-1, 5)], -1)
        if SHAPE["team_share"]:
            total = own
        if SHAPE["zero_sum"]:
            total = total - other
    return total, comp


def structure(prev, nxt):
    """(B,10) reward-structure terms: push toward the enemy ancient, siege presence, enemy ancient damage."""
    B = prev.shape[0]
    out = torch.zeros(B, 10, device=prev.device)
    if not (SHAPE["push_k"] or SHAPE["siege_k"] or SHAPE["anc_k"] or SHAPE["kill_k"]):
        return out
    p, n = prev[:, :10], nxt[:, :10]
    team = p[..., F["team"]]                                                         # (B,10) 0 radiant / 1 dire
    alive = (p[..., F["alive"]] > 0) & (n[..., F["alive"]] > 0)
    anc = prev[:, 204:206]                                                           # 204 dire ancient, 205 radiant
    ateam = anc[..., F["team"]]
    # the enemy ancient of each hero
    ei = (ateam[:, None, :] != team[:, :, None]).float().argmax(-1)                  # (B,10) index 0/1 into anc
    ex = torch.gather(anc[..., F["x"]], 1, ei); ey = torch.gather(anc[..., F["y"]], 1, ei)
    if SHAPE["push_k"]:
        d0 = (p[..., F["x"]] - ex).abs() + (p[..., F["y"]] - ey).abs()
        d1 = (n[..., F["x"]] - ex).abs() + (n[..., F["y"]] - ey).abs()
        step = d0 - d1
        out += SHAPE["push_k"] * torch.where(alive & (step.abs() <= 5), step, torch.zeros_like(step))
    if SHAPE["siege_k"]:
        tw = prev[:, 182:]
        dist = (p[..., F["x"]][:, :, None] - tw[..., F["x"]][:, None]).abs() + (p[..., F["y"]][:, :, None] - tw[..., F["y"]][:, None]).abs()
        enemy = (tw[..., F["team"]][:, None] != team[:, :, None]) & (tw[..., F["alive"]][:, None] > 0)
        out += SHAPE["siege_k"] * (alive & (enemy & (dist <= SHAPE["siege_radius"])).any(-1)).float()
    if SHAPE["kill_k"]:                                                      # champion kills: an enemy hero dies ->
        jump = (n[..., F["x"]] - p[..., F["x"]]).abs() + (n[..., F["y"]] - p[..., F["y"]]).abs()     # the nearest
        died = (jump > 12) & (n[..., F["health"]] >= n[..., F["max_health"]] - 0.5) & \
            (p[..., F["health"]] < p[..., F["max_health"]] - 0.5)                                       # living enemy
        if died.any():                                                                                  # hero in range
            d = (p[..., F["x"]][:, :, None] - p[..., F["x"]][:, None]).abs() + (p[..., F["y"]][:, :, None] - p[..., F["y"]][:, None]).abs()
            ok = (team[:, :, None] != team[:, None, :]) & alive[:, :, None] & died[:, None, :] & (d <= SHAPE["kill_radius"])
            dd = torch.where(ok, d, torch.full_like(d, 1e9))                                             # (B, killer, victim)
            best = dd.argmin(1)                                                                          # (B, victim)
            got = (dd.min(1).values < 1e9).float()
            out += SHAPE["kill_k"] * torch.zeros_like(out).scatter_add_(1, best, got)
    if SHAPE["anc_k"]:
        an = nxt[:, 204:206]
        maxh = anc[..., F["max_health"]].clamp(min=1.0)
        hp1 = torch.where(an[..., F["alive"]] > 0, an[..., F["health"]], torch.zeros_like(maxh))
        lost = ((anc[..., F["health"]] - hp1).clamp(min=0) / maxh) * (anc[..., F["alive"]] > 0)        # (B,2)
        out += SHAPE["anc_k"] * torch.gather(lost, 1, ei)
    return out


def shaping(prev, nxt):
    """(B,10) extra dream reward: tower kills (to the attacking team or the nearest hero) and wins."""
    B = prev.shape[0]
    out = torch.zeros(B, 10, device=prev.device)
    hero_team = prev[:, :10, F["team"]]                                                  # (B,10)
    tfell = (prev[:, 182:, F["alive"]] > 0) & (nxt[:, 182:, F["alive"]] <= 0)          # (B,24)
    tteam = prev[:, 182:, F["team"]]
    if SHAPE["r_tower"] is not None and tfell.any():
        extra = SHAPE["r_tower"] - R_TOWER                                               # on top of the game's credit
        if SHAPE["tower_team"]:                                                          # the whole attacking team
            per = (tfell[:, None, :] & (hero_team[:, :, None] != tteam[:, None, :])).float().sum(-1)
            out += per * SHAPE["r_tower"]
        else:
            out += comp_tower_count(prev, nxt) * extra
    if SHAPE["dmg_k"] or SHAPE["fall_bonus"]:
        out += tower_push(prev, nxt)
    if SHAPE["r_win"]:
        anc = tfell[:, 22:24]                                                            # entities 204 (dire), 205 (radiant)
        if anc.any():
            fallen_team = torch.where(anc[:, 0], tteam[:, 22], tteam[:, 23])             # (B,)
            won = anc.any(-1)[:, None] & (hero_team != fallen_team[:, None])
            lost = anc.any(-1)[:, None] & (hero_team == fallen_team[:, None])
            out += SHAPE["r_win"] * (won.float() - lost.float())
    return out


def tower_push(prev, nxt):
    """(B,10) dense tower shaping. Damage: each tick a tower loses health, every enemy hero within
    dmg_radius gets dmg_k * (lost / max_health) * (1 + dmg_boost * (1 - health left / max_health)),
    doubled for the ancients (entities 204, 205). Fall: when a tower falls, every enemy hero
    within fall_radius gets fall_bonus * (1 - distance / fall_radius)."""
    tp, tn = prev[:, 182:], nxt[:, 182:]                                                  # (B,24,F)
    alive = tp[..., F["alive"]] > 0
    maxh = tp[..., F["max_health"]].clamp(min=1.0)
    hp_next = torch.where(tn[..., F["alive"]] > 0, tn[..., F["health"]], torch.zeros_like(maxh))
    lost = (tp[..., F["health"]] - hp_next).clamp(min=0) * alive
    left = (hp_next / maxh).clamp(0, 1)
    w = lost / maxh * (1 + SHAPE["dmg_boost"] * (1 - left))
    w[:, 22:24] *= 2                                                                      # ancients
    fell = alive & (tn[..., F["alive"]] <= 0)
    h = prev[:, :10]
    dist = (h[..., F["x"]][:, :, None] - tp[..., F["x"]][:, None]).abs() + \
           (h[..., F["y"]][:, :, None] - tp[..., F["y"]][:, None]).abs()                 # (B,10,24)
    enemy = (h[..., F["team"]][:, :, None] != tp[..., F["team"]][:, None]) & (h[..., F["alive"]][:, :, None] > 0)
    out = SHAPE["dmg_k"] * ((enemy & (dist <= SHAPE["dmg_radius"])).float() * w[:, None]).sum(-1)
    if SHAPE["fall_bonus"]:
        prox = (1 - dist / SHAPE["fall_radius"]).clamp(min=0) * enemy * fell[:, None]
        out = out + SHAPE["fall_bonus"] * prox.sum(-1)
    if SHAPE["def_scale"]:
        # the defending team pays for its towers: -def_scale x (damage signal + fall bonus), team-wide
        own = (h[..., F["team"]][:, :, None] == tp[..., F["team"]][:, None]) & (h[..., F["alive"]][:, :, None] > 0)
        loss_t = SHAPE["dmg_k"] * w + SHAPE["fall_bonus"] * fell.float()                        # (B,24)
        out = out - SHAPE["def_scale"] * (own.float() * loss_t[:, None]).sum(-1)
    return out


def comp_tower_count(prev, nxt):
    """(B,10) number of towers credited to each hero by the game-faithful rule in reward()."""
    B = prev.shape[0]
    p = prev[:, :10]
    tfell = (prev[:, 182:, F["alive"]] > 0) & (nxt[:, 182:, F["alive"]] <= 0)
    tx, ty, tteam = prev[:, 182:, F["x"]], prev[:, 182:, F["y"]], prev[:, 182:, F["team"]]
    dist = (p[..., F["x"]][:, :, None] - tx[:, None]).abs() + (p[..., F["y"]][:, :, None] - ty[:, None]).abs()
    enemy = (p[..., F["team"]][:, :, None] != tteam[:, None]) & (p[..., F["alive"]][:, :, None] > 0)
    dist = torch.where(enemy & (dist <= 6), dist, torch.full_like(dist, 1e9))
    best = dist.argmin(1)
    ok = tfell & (dist.min(1).values < 1e9)
    credit = torch.zeros(B, 10, 24, device=prev.device)
    credit.scatter_(1, best[:, None], ok[:, None].float())
    return credit.sum(-1)


class NetPool:
    """The shipped PufferNet (the corpus's "net" controller) acting in the dream for n games:
    build/libmobaarena ma_pool_*, verified bit-identical to the collector's net_act logits."""

    def __init__(self, n_rows, seed=0):
        import ctypes
        lib = ROOT / "build" / ("libmobaarena.so" if sys.platform.startswith("linux") else "libmobaarena.dylib")
        L = ctypes.CDLL(str(lib))
        P = ctypes.c_void_p
        L.ma_pool_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64]
        L.ma_pool_act.argtypes = [P, P]
        L.ma_pool_reset_rows.argtypes = [ctypes.c_int, ctypes.c_int]
        w = ROOT / "third_party/pufferlib/resources/moba/moba_weights.bin"
        if L.ma_pool_init(str(w).encode(), n_rows, seed) != 0:
            raise SystemExit(f"could not load {w}")
        self.L, self.n = L, n_rows
        self.out = np.zeros((n_rows, 6), np.float32)

    def reset(self):
        self.L.ma_pool_reset_rows(0, self.n)

    def act(self, obs):                          # obs (n, 510) uint8 -> (n, 6) long
        obs = np.ascontiguousarray(obs)
        self.L.ma_pool_act(obs.ctypes.data, self.out.ctypes.data)
        return self.out.astype(np.int64)


# --- policy -------------------------------------------------------------------

class Policy(nn.Module):
    """Shared actor-critic over one hero's 510-byte ego observation.

    Bytes 0..123 are tile ids (embedded), 124..483 are always zero (dropped),
    484..509 are the hero's own scalars (scaled)."""

    N_ROLES = 5                                 # hero type one-hot at obs bytes 484+17 .. 484+21
    recurrent = False

    def __init__(self, hidden=256, role_heads=False):
        super().__init__()
        self.role_heads = role_heads
        self.tile = nn.Embedding(256, 8)
        extra = 16 if role_heads else 0
        if role_heads:                          # the role, unscaled, as its own embedding
            self.role_emb = nn.Embedding(self.N_ROLES, extra)
        self.trunk = nn.Sequential(nn.Linear(124 * 8 + 26 + extra, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        k = self.N_ROLES if role_heads else 1   # one action head and one value head per role
        self.pi = nn.Linear(hidden, k * sum(HEADS))
        self.v = nn.Linear(hidden, k)
        nn.init.zeros_(self.pi.weight); nn.init.zeros_(self.pi.bias)   # uniform actions at start

    @staticmethod
    def role(obs):
        return obs[:, 484 + 17:484 + 22].long().argmax(-1)

    def forward(self, obs):                     # obs (N, 510) uint8
        o = obs.long()
        parts = [self.tile(o[:, :124]).flatten(1), o[:, 484:].float() / 32.0]
        if self.role_heads:
            r = self.role(obs)
            parts.append(self.role_emb(r))
        z = self.trunk(torch.cat(parts, -1))
        pi, v = self.pi(z), self.v(z)
        if self.role_heads:
            n = len(o)
            pi = pi.view(n, self.N_ROLES, -1)[torch.arange(n, device=o.device), r]
            v = v[torch.arange(n, device=o.device), r][:, None]
        return pi.split(HEADS, -1), v.squeeze(-1)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """A saved policy (either variant: role heads are detected from the weights)."""
        ck = torch.load(path, map_location=map_location, weights_only=False)
        sd = ck["policy"] if "policy" in ck else ck
        if any(k.startswith("net.") for k in sd):
            pol = PufferPolicy()
            pol.load_state_dict(sd)
            return pol
        pol = cls(role_heads="role_emb.weight" in sd)
        pol.load_state_dict(sd)
        return pol

    def act(self, obs):
        logits, v = self(obs)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        a = torch.stack([d.sample() for d in dists], -1)
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(dists))
        return a, logp, v

    def evaluate(self, obs, a):
        logits, v = self(obs)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(dists))
        ent = sum(d.entropy() for d in dists)
        return logp, ent, v


def _puffer_models():
    """PufferLib's models.py (vendored; identical to the old ocean-wm repo's), loaded as a
    file so importing the pufferlib package (and its compiled parts) is not needed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("puffer_models", ROOT / "third_party/pufferlib/pufferlib/models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class PufferPolicy(nn.Module):
    """PufferLib's MOBA policy, as in the old ocean-wm repo (tools/train_ppo_in_world_model.py):
    Linear(510 -> 64) on the raw observation bytes, a 6-layer MinGRU (recurrent), a linear
    decoder to the 6 action heads + value. The memory is carried per hero during rollouts
    (reset_state at every episode start) and replayed over whole sequences in the update."""
    recurrent = True

    def __init__(self, hidden=64, layers=6):
        super().__init__()
        M = _puffer_models()
        self.net = M.Policy(M.DefaultEncoder(510, hidden), M.DefaultDecoder(HEADS, hidden),
                            M.MinGRU(hidden, num_layers=layers))
        self.state = None

    def reset_state(self, n, dev):
        self.state = self.net.initial_state(n, dev)

    def step(self, obs):                        # one tick for n heroes, advances the memory
        logits, v, self.state = self.net.forward_eval(obs.float(), self.state)
        return logits, v.squeeze(-1)

    def peek_value(self, obs):                  # value of the next state, memory unchanged
        _, v, _ = self.net.forward_eval(obs.float(), tuple(x.clone() for x in self.state))
        return v.squeeze(-1)

    def act(self, obs):
        logits, v = self.step(obs)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        a = torch.stack([d.sample() for d in dists], -1)
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(dists))
        return a, logp, v

    def evaluate_seq(self, obs, a):             # obs (n, T, 510), a (n, T, 6) -> logp, ent, v: (n*T,)
        logits, v = self.net(obs.float())
        self.last_logits = logits                # for a KL penalty toward an anchor policy
        a = a.reshape(-1, 6)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(dists))
        ent = sum(d.entropy() for d in dists)
        return logp, ent, v.reshape(-1)


# --- PPO ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="runs/token_wm/p1_at1000", help="world model run dir (model.pt, stats.pt); default P1 @1000 (E12 @4000 + 1,000 updates at lr 1e-4): matches or beats E12 on every tower/event/movement metric")
    ap.add_argument("--dense", default="data/moba/mix100/dense", help="real start states (training games)")
    ap.add_argument("--games", default="0-19")
    ap.add_argument("--steps", type=int, default=3_000_000, help="agent steps (10 per env tick)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = only --steps)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=64, help="env ticks per PPO update")
    ap.add_argument("--episode", type=int, default=512, help="dream ticks before an env is reset")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--name", default="dream_ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=25, help="keep policy_<agent_steps>.pt every N updates")
    ap.add_argument("--role-heads", action="store_true", help="role embedding + one action/value head per role")
    ap.add_argument("--arch", default="mlp", choices=("mlp", "puffer"),
                    help="mlp: our feed-forward policy; puffer: PufferLib's Encoder + 6-layer MinGRU (recurrent)")
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-5)
    ap.add_argument("--vf-clip", type=float, default=0.0, help="value clipping (recurrent update; 0 = off)")
    ap.add_argument("--r-tower", type=float, default=None, help="dream-only: reward per tower kill (default: the game's 0.642 to the nearest hero)")
    ap.add_argument("--tower-team", action="store_true", help="dream-only: tower reward to every hero of the attacking team")
    ap.add_argument("--dmg-k", type=float, default=0.0, help="dream-only: tower-damage reward scale (see tower_push)")
    ap.add_argument("--dmg-boost", type=float, default=0.0, help="dream-only: extra weight as the tower's health runs out")
    ap.add_argument("--dmg-radius", type=float, default=8.0)
    ap.add_argument("--fall-bonus", type=float, default=0.0, help="dream-only: falling-tower bonus, largest for the closest enemy hero")
    ap.add_argument("--fall-radius", type=float, default=8.0)
    ap.add_argument("--def-scale", type=float, default=0.0, help="dream-only: defenders pay def_scale x the tower damage/fall reward")
    # Distributed dream PPO (zelda-RL pattern): every pod runs the whole PPO loop on its own dream games (own seed);
    # after every --fleet-every updates the pods average their policy weights through Upstash (mobawm/fleet_link.py)
    ap.add_argument("--fleet-exp", default=""); ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world", type=int, default=1); ap.add_argument("--fleet-launch", default="0")
    ap.add_argument("--fleet-every", type=int, default=1)
    ap.add_argument("--kl-anchor", default="", help="a saved policy: add --kl-coef x KL(anchor || policy) to the PPO loss")
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--anchor-watch", default="", help="continuous Dyna: dir of anchor_<U>.pt; at update >= U the KL anchor becomes it")
    ap.add_argument("--wm-watch", default="", help="continuous Dyna: dir of wm_<U>_<it>.pt; at update >= U the dream takes those weights")
    ap.add_argument("--reset-watch", default="", help="continuous Dyna: dir of policy_<U>.pt; at update >= U the policy restarts from it")
    ap.add_argument("--gpu-opp", action="store_true", help="build observations and run the shipped net on the GPU (exact; ~5x+ faster)")
    ap.add_argument("--side", default="both", choices=("both", "radiant", "dire"), help="with --opponent net: which side ours plays")
    ap.add_argument("--xp-scale", type=float, default=1.0, help="dream-only: scale of the game's XP reward (0 = off)")
    ap.add_argument("--death-scale", type=float, default=1.0, help="dream-only: scale of the game's death penalty")
    ap.add_argument("--push-k", type=float, default=0.0, help="dream-only: reward per cell of progress toward the enemy ancient")
    ap.add_argument("--siege-k", type=float, default=0.0, help="dream-only: reward per tick within --siege-radius of a living enemy tower")
    ap.add_argument("--siege-radius", type=float, default=8.0)
    ap.add_argument("--anc-k", type=float, default=0.0, help="dream-only: enemy ancient health lost (fraction) to the attacking team")
    ap.add_argument("--kill-k", type=float, default=0.0, help="dream-only: reward to the nearest enemy hero (within 8 cells) when a hero dies")
    ap.add_argument("--team-share", action="store_true", help="dream-only: every hero gets its team's mean reward")
    ap.add_argument("--zero-sum", action="store_true", help="dream-only: reward minus the enemy team's mean")
    ap.add_argument("--init-policy", default="", help="start from this saved policy (fine-tune)")
    ap.add_argument("--r-win", type=float, default=0.0, help="dream-only: +W to the winning team, -W to the losing team when an ancient falls")
    ap.add_argument("--net-frac", type=float, default=0.5, help="--opponent mix: share of games against the net")
    ap.add_argument("--opponent", default="self", choices=("self", "net", "mix"),
                    help="self: one policy plays all 10 heroes; net: ours plays one random side per game, the "
                         "shipped PufferNet the other (only our heroes are trained and scored)")
    ap.add_argument("--wandb", action="store_true", help="log to W&B drift-ai-models/moba-dream-rl")
    ap.add_argument("--group", default="", help="--wandb group (experiment)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    out = ROOT / "runs" / "dream_ppo" / args.name
    out.mkdir(parents=True, exist_ok=True)
    from ..wm.train import parse_games
    files = sorted(Path(args.dense).glob("games.*.mbs"))
    env = DreamEnv(args.wm, files, parse_games(args.games), args.envs, args.episode, dev, seed=args.seed)
    SHAPE.update(r_tower=args.r_tower, tower_team=args.tower_team, r_win=args.r_win, dmg_k=args.dmg_k,
                 dmg_boost=args.dmg_boost, dmg_radius=args.dmg_radius, fall_bonus=args.fall_bonus,
                 fall_radius=args.fall_radius, def_scale=args.def_scale, xp_scale=args.xp_scale,
                 death_scale=args.death_scale, push_k=args.push_k, siege_k=args.siege_k, siege_radius=args.siege_radius,
                 anc_k=args.anc_k, team_share=args.team_share, zero_sum=args.zero_sum, kill_k=args.kill_k)
    pol = (PufferPolicy() if args.arch == "puffer" else Policy(role_heads=args.role_heads)).to(dev)
    anchor = None
    if args.kl_anchor:
        anchor = Policy.load(args.kl_anchor).to(dev).eval().requires_grad_(False)
        print(f"KL anchor {args.kl_anchor}, coef {args.kl_coef}", flush=True)
    if args.init_policy:
        src = Policy.load(args.init_policy)
        pol.load_state_dict(src.state_dict())
        print(f"policy initialised from {args.init_policy}", flush=True)
    if pol.recurrent and args.episode != args.rollout:
        raise SystemExit("--arch puffer needs --episode == --rollout (each rollout = one episode from a fresh memory)")
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    B, T, N = args.envs, args.rollout, args.envs * 10
    logf = open(out / "log.jsonl", "w")
    from ..s3persist import flush as persist_flush, persist   # no-op without a policy file above `out`
    if args.opponent in ("net", "mix") and args.gpu_opp:      # the shipped net in torch on the GPU (gpu_net.py, exact)
        from .gpu_net import ShippedNet
        class _GpuPool:
            def __init__(s):
                s.net = ShippedNet(N, dev); s.net.gen.manual_seed(args.seed)
            def reset(s):
                s.net.state.zero_()
            def act(s, o):
                return s.net.act(o)
        opp = _GpuPool()
    else:
        opp = NetPool(N, seed=args.seed) if args.opponent in ("net", "mix") else None
    side_gen = torch.Generator().manual_seed(args.seed + 1)

    def draw_sides():
        """(B, 10) bool: which hero slots OUR policy controls in each game."""
        if opp is None:
            return torch.ones(B, 10, dtype=torch.bool, device=dev)
        radiant = torch.rand(B, generator=side_gen) < 0.5
        if args.side != "both":                     # train one side only (e.g. dire, which no policy has learned)
            radiant = torch.full((B,), args.side == "radiant")
        m = torch.zeros(B, 10, dtype=torch.bool)
        m[radiant, :5] = True
        m[~radiant, 5:] = True
        if args.opponent == "mix":                  # curriculum mix: the rest of the games are self-play
            m[torch.rand(B, generator=side_gen) >= args.net_frac] = True
        opp.reset()
        return m.to(dev)

    ours = draw_sides()
    wb = None
    if args.wandb:
        from mobawm import secrets
        secrets.export_to_env("WANDB_API_KEY")
        import wandb
        wb = wandb.init(entity="drift-ai-models", project="moba-dream-rl", group=args.group or None, name=args.name,
                        config=vars(args), tags=["dream-ppo", f"opponent-{args.opponent}"])
    print(f"dream PPO: wm {args.wm}, {B} envs x 10 heroes, rollout {T}, episode {args.episode}, "
          f"{args.steps / 1e6:.1f}M agent steps on {dev}", flush=True)

    link = None
    if args.fleet_exp:
        from ..fleet_link import FleetLink
        link = FleetLink(args.fleet_exp, args.fleet_rank, args.fleet_world, args.fleet_launch)
        print(f"PPO fleet {args.fleet_exp}: rank {args.fleet_rank}/{args.fleet_world}, average every {args.fleet_every} update(s)", flush=True)
    steps, update, t0 = 0, 0, time.time()
    ep_ret = torch.zeros(B, 10, device=dev)          # running per-hero return of the current episodes
    finished = []                                    # mean per-hero return of episodes finished this update
    CLS = ("support", "assassin", "burst", "tank", "carry")
    wm_now, reset_now, anchor_now = "start", "", ""
    def due(d, pat):                                 # newest file in d whose target update (wm_<U>_..., policy_<U>) has come
        fs = [f for f in Path(d).glob(pat) if int(f.stem.split("_")[1]) <= update] if d and Path(d).exists() else []
        return max(fs, key=lambda f: f.name) if fs else None
    while steps < args.steps and not (args.minutes and time.time() - t0 > args.minutes * 60):
        f = due(args.wm_watch, "wm_*.pt")
        if f is not None and f.name != wm_now:       # every rank swaps at the same update U (episodes restart each update)
            env.model.load_state_dict(torch.load(f, map_location=dev)); wm_now = f.name
            print(f"  update {update}: dream world model -> {f.name}", flush=True)
        f = due(args.anchor_watch, "anchor_*.pt")
        if anchor is not None and f is not None and f.name != anchor_now:
            anchor.load_state_dict(torch.load(f, map_location=dev, weights_only=False)["policy"]); anchor_now = f.name
            print(f"  update {update}: KL anchor -> {f.name}", flush=True)
        f = due(args.reset_watch, "policy_*.pt")
        if f is not None and f.name != reset_now:
            pol.load_state_dict(torch.load(f, map_location=dev, weights_only=False)["policy"]); opt.state.clear(); reset_now = f.name
            print(f"  update {update}: policy reset -> {f.name}", flush=True)
        if env.age >= args.episode or not env.alive.any():
            finished.append(float(ep_ret[ours].mean()))
            ep_ret.zero_()
            env.reset_all()
            ours = draw_sides()
        obs_b = torch.zeros(T, N, 510, dtype=torch.uint8, device=dev)
        if pol.recurrent:                            # every rollout is one fresh episode (episode == rollout)
            pol.reset_state(N, dev)
        act_b = torch.zeros(T, N, 6, dtype=torch.long, device=dev)
        logp_b, val_b, rew_b = (torch.zeros(T, N, device=dev) for _ in range(3))
        done_b, valid_b = torch.zeros(T, N, device=dev), torch.zeros(T, N, device=dev)
        comp_sum = torch.zeros(4, device=dev)
        comp_opp = torch.zeros(4, device=dev)      # the opponent's (net's) components: who takes towers in the dream
        ours_r = opp_r = 0.0
        comp_hero = torch.zeros(10, 4, device=dev)   # per hero slot: 0-4 radiant, 5-9 dire; class = slot % 5
        t_env = t_pol = 0.0
        for t in range(T):
            ta = time.time()
            if args.gpu_opp:                        # observations built on the GPU: no host round trip
                o = env.obs_gpu().reshape(N, 510)
            else:
                o_np = env.obs().reshape(N, 510)
                o = torch.from_numpy(o_np).to(dev)
            with torch.no_grad():
                a, logp, v = pol.act(o)
            a_env = a
            if opp is not None:                      # the shipped net plays the other side
                net_a = opp.act(o) if args.gpu_opp else torch.from_numpy(opp.act(o_np)).to(dev)
                a_env = torch.where(ours.reshape(N, 1), a, net_a)
            tb = time.time()
            valid = (env.alive[:, None] & ours).reshape(N).float()
            r, done = env.step(a_env.view(B, 10, 6))
            ours_r += float((r * ours).sum())
            opp_r += float((r * ~ours).sum())
            r = r * ours
            t_env += time.time() - tb
            t_pol += tb - ta
            obs_b[t], act_b[t], logp_b[t], val_b[t] = o, a, logp, v
            rew_b[t], done_b[t], valid_b[t] = r.reshape(N), done.repeat_interleave(10).float(), valid
            comp_sum += (env.last_reward * ours[..., None]).sum((0, 1))
            comp_opp += (env.last_reward * ~ours[..., None]).sum((0, 1))
            comp_hero += (env.last_reward * ours[..., None]).sum(0)
            ep_ret += r
        with torch.no_grad():
            o_last = env.obs_gpu().reshape(N, 510) if args.gpu_opp else torch.from_numpy(env.obs().reshape(N, 510)).to(dev)
            v_last = pol.peek_value(o_last) if pol.recurrent else pol(o_last)[1]
        # GAE over the rollout; a finished episode (ancient fell) cuts the bootstrap
        adv = torch.zeros(T, N, device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + args.gamma * nv * nonterm - val_b[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + val_b
        m = valid_b.reshape(-1) > 0
        flat = lambda x: x.reshape(T * N, *x.shape[2:])[m]
        ret_f = flat(ret)
        if pol.recurrent:
            # whole hero sequences (T ticks from a fresh memory), minibatched over heroes, loss masked to
            # our valid ticks; advantages normalised over the valid ticks
            vm = valid_b > 0
            advn = (adv - adv[vm].mean()) / (adv[vm].std() + 1e-8)
            seq = lambda x: x.transpose(0, 1)                       # (T, N, ...) -> (N, T, ...)
            n_h, mb = N, max(1, N // args.minibatches)
            tc = time.time()
            kls, clips, kl_anchor = [], [], []
            for _ in range(args.epochs):
                perm = torch.randperm(n_h, device=dev)
                for s0 in range(0, n_h, mb):
                    idx = perm[s0:s0 + mb]
                    logp, ent, v = pol.evaluate_seq(seq(obs_b[:, idx]), seq(act_b[:, idx]))
                    k = seq(valid_b[:, idx]).reshape(-1)
                    kn = k.sum().clamp(min=1)
                    old = seq(logp_b[:, idx]).reshape(-1)
                    a_ = seq(advn[:, idx]).reshape(-1)
                    r_ = seq(ret[:, idx]).reshape(-1)
                    v0 = seq(val_b[:, idx]).reshape(-1)
                    ratio = (logp - old).exp()
                    with torch.no_grad():
                        kls.append(float((((ratio - 1) - (logp - old)) * k).sum() / kn))
                        clips.append(float((((ratio - 1).abs() > args.clip).float() * k).sum() / kn))
                    pg = (-torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_) * k).sum() / kn
                    if args.vf_clip > 0:
                        vc = v0 + (v - v0).clamp(-args.vf_clip, args.vf_clip)
                        vl = (0.5 * torch.maximum((v - r_) ** 2, (vc - r_) ** 2) * k).sum() / kn
                    else:
                        vl = (((v - r_) ** 2) * k).sum() / kn
                    loss = pg + args.vf * vl - args.ent * (ent * k).sum() / kn
                    if anchor is not None:
                        with torch.no_grad():
                            al, _ = anchor.net(seq(obs_b[:, idx]).float())
                        kl_a = sum((Fn.softmax(x, -1) * (Fn.log_softmax(x, -1) - Fn.log_softmax(y, -1))).sum(-1)
                                   for x, y in zip(al, pol.last_logits))
                        kl_a = (kl_a * k).sum() / kn
                        kl_anchor.append(float(kl_a))
                        loss = loss + args.kl_coef * kl_a
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), args.max_grad_norm)
                    opt.step()
        else:
            m = valid_b.reshape(-1) > 0
            flat = lambda x: x.reshape(T * N, *x.shape[2:])[m]
            o_f, a_f, lp_f, adv_f, ret_f = flat(obs_b), flat(act_b), flat(logp_b), flat(adv), flat(ret)
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
            n, mb = len(o_f), max(1, len(o_f) // args.minibatches)
            tc = time.time()
            kls, clips, kl_anchor = [], [], []
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n, mb):
                    idx = perm[s:s + mb]
                    logp, ent, v = pol.evaluate(o_f[idx], a_f[idx])
                    ratio = (logp - lp_f[idx]).exp()
                    with torch.no_grad():
                        kls.append(float(((ratio - 1) - (logp - lp_f[idx])).mean()))
                        clips.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                    pg = -torch.min(ratio * adv_f[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv_f[idx]).mean()
                    vl = Fn.mse_loss(v, ret_f[idx])
                    loss = pg + args.vf * vl - args.ent * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), args.max_grad_norm)
                    opt.step()
        t_upd = time.time() - tc
        steps += int(m.sum())
        update += 1
        if link is not None and update % args.fleet_every == 0:
            fi = link.average_weights(pol, update // args.fleet_every, timeout=600)
            link.heartbeat(dict(update=update, steps=steps, sps=steps / (time.time() - t0),
                                ours_towers=float(comp_sum[3] / R_TOWER) / max(B, 1),
                                net_towers=float(comp_opp[3] / R_TOWER) / max(B, 1), **fi))
            if update <= 3 or update % 10 == 0:
                print(f"  fleet round {update // args.fleet_every}: averaged {fi['sync_ranks']} pods, wait {fi['sync_wait_s']:.1f}s, hash {fi['weight_hash']}", flush=True)
        n_valid = max(int(m.sum()), 1)
        per1k = comp_sum / n_valid * 1000
        hero1k = comp_hero / (n_valid / 10) * 1000                  # per hero slot, per 1k of ITS steps
        ev = float(1 - (ret_f - flat(val_b)).var() / ret_f.var().clamp(min=1e-8))
        detail = {}
        for side, sl in (("radiant", slice(0, 5)), ("dire", slice(5, 10))):
            h = hero1k[sl]
            detail[f"{side}/reward_per_1k"] = float(h.sum(-1).mean())
            detail[f"{side}/xp_per_1k"] = float(h[:, 1].mean() / R_XP)
            detail[f"{side}/deaths_per_1k"] = float(h[:, 0].mean() / R_DEATH)
            detail[f"{side}/tower_kills_per_1k"] = float(h[:, 3].mean() / R_TOWER)
        for c, name in enumerate(CLS):
            h = hero1k[[c, c + 5]]
            detail[f"class/{name}/reward_per_1k"] = float(h.sum(-1).mean())
            detail[f"class/{name}/deaths_per_1k"] = float(h[:, 0].mean() / R_DEATH)
            detail[f"class/{name}/xp_per_1k"] = float(h[:, 1].mean() / R_XP)
        rec = dict(update=update, agent_steps=steps, sps=round(steps / (time.time() - t0)),
                   episode_return_per_hero=float(np.mean(finished)) if finished else None,
                   episodes_finished=len(finished), approx_kl=float(np.mean(kls)), clip_frac=float(np.mean(clips)),
                   explained_var=ev, kl_anchor=float(np.mean(kl_anchor)) if kl_anchor else None, **detail,
                   reward_per_1k=float(rew_b.sum() / max(int(m.sum()), 1) * 1000),
                   deaths_per_1k=float(per1k[0] / R_DEATH), xp_per_1k=float(per1k[1] / R_XP),
                   tower_kills_per_1k=float(per1k[3] / R_TOWER), entropy=float(ent.mean()),
                   ours_towers_per_episode=float(comp_sum[3] / R_TOWER) / max(B, 1), opp_towers_per_episode=float(comp_opp[3] / R_TOWER) / max(B, 1),
                   value_loss=float(vl), time_env=round(t_env, 1), time_policy=round(t_pol, 1),
                   time_update=round(t_upd, 1), state_repairs=env.repairs,
                   opponent=args.opponent, wm=wm_now, ours_reward_per_1k=ours_r / n_valid * 1000,
                   opp_reward_per_1k=(opp_r / max(float((env.alive[:, None] & ~ours).sum()) * T, 1) * 1000
                                      if opp is not None else None))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if wb:
            wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float)) and k != "update"}, step=update)
        finished = []
        print(f"upd {update:4d}  {steps / 1e6:5.2f}M steps  {rec['sps']} sps  | reward/1k {rec['reward_per_1k']:+.2f}  "
              f"xp/1k {rec['xp_per_1k']:.0f}  deaths/1k {rec['deaths_per_1k']:.2f}  towers/1k "
              f"{rec['tower_kills_per_1k']:.3f} (towers/episode ours {rec['ours_towers_per_episode']:.3f} vs net {rec['opp_towers_per_episode']:.3f})  | ent {rec['entropy']:.2f}  vloss {rec['value_loss']:.4f}  "
              f"(env {t_env:.0f}s pol {t_pol:.0f}s upd {t_upd:.0f}s)", flush=True)
        print(f"          radiant r/1k {detail['radiant/reward_per_1k']:+.2f}  dire r/1k {detail['dire/reward_per_1k']:+.2f}  "
              f"| by class r/1k: " + "  ".join(f"{c} {detail[f'class/{c}/reward_per_1k']:+.2f}" for c in CLS)
              + f"  | kl {rec['approx_kl']:.4f}  clip {rec['clip_frac']:.2f}  ev {ev:.2f}"
              + (f"  | episode return/hero {rec['episode_return_per_hero']:+.2f}" if finished else ""), flush=True)
        ck = dict(policy=pol.state_dict(), optimizer=opt.state_dict(), update=update, agent_steps=steps,
                  args=vars(args), wm=wm_now, reset=reset_now, anchor=anchor_now)
        torch.save(ck, out / "policy.pt")                             # latest, every update
        persist(out / "policy.pt")
        persist(out / "log.jsonl")
        if update % args.ckpt_every == 0:
            torch.save(ck, out / f"policy_{steps:09d}.pt")
            persist(out / f"policy_{steps:09d}.pt")
    torch.save(dict(policy=pol.state_dict(), optimizer=opt.state_dict(), update=update, agent_steps=steps,
                    args=vars(args)), out / "policy.pt")
    persist(out / "policy.pt")
    persist(out / "log.jsonl")
    persist_flush()
    if wb:
        wb.finish()
    print(f"done: {steps} agent steps in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
