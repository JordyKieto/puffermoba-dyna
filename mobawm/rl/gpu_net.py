"""GPU versions of the two CPU-bound parts of dream PPO (they were ~83% of an update on an A40):

  ShippedNet  the shipped PufferLib MOBA policy (moba_weights.bin) in torch: Linear(510->64, no bias) -> 5 MinGRU
              layers (Linear 64->192 each: hidden / gate / highway) -> Linear(64 -> 23 logits + value), sampled per
              head from the softmax. Same weights, same maths as puffernet.h (make_puffernet / mingru / linear),
              verified against the C pool's logits.
  ego_obs_gpu the 510-byte ego observations (mobawm/obs.py, itself a port of moba_ego_obs) from dream states on the
              GPU, including the map grid / pid buffers DreamEnv.obs() builds on the CPU. Verified byte-identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..shards import F, MAP, N_HEROES, NONE, OBS_DIM
from ..obs import VISION, WIN, EXTRA

ACTION_SIZES = (7, 7, 3, 2, 2, 2)
WEIGHTS = Path(__file__).resolve().parents[2] / "third_party/pufferlib/resources/moba/moba_weights.bin"


class ShippedNet(torch.nn.Module):
    def __init__(self, n_rows, device, weights=WEIGHTS, hidden=64, layers=5, obs_dim=OBS_DIM):
        super().__init__()
        w = np.fromfile(weights, dtype=np.float32)
        idx = 0

        def take(n):                                   # get_weights_aligned: next tensor starts 8-float aligned
            nonlocal idx
            t = torch.from_numpy(w[idx:idx + n].copy())
            idx = (idx + n + 7) & ~7
            return t
        n_out = sum(ACTION_SIZES) + 1
        self.enc = take(hidden * obs_dim).view(hidden, obs_dim).to(device)          # make_puffernet order: encoder,
        self.dec = take(n_out * hidden).view(n_out, hidden).to(device)              # decoder, then the MinGRU layers
        self.proj = [take(3 * hidden * hidden).view(3 * hidden, hidden).to(device) for _ in range(layers)]
        assert idx <= len(w) + 7, (idx, len(w))
        self.H, self.L, self.n = hidden, layers, n_rows
        self.state = torch.zeros(layers, n_rows, hidden, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0)

    def reset_rows(self, mask):
        """Zero the recurrent state of rows where mask (n_rows,) bool is True (new episodes)."""
        self.state[:, mask] = 0

    @torch.no_grad()
    def logits(self, obs):
        """obs (n, 510) uint8 -> (n, 24) decoder output (23 logits + value); advances the recurrent state."""
        x = obs.float() @ self.enc.T
        H = self.H
        for l in range(self.L):
            c = x @ self.proj[l].T
            hidden, gate, hw = c[:, :H], c[:, H:2 * H], c[:, 2 * H:]
            s = self.state[l]
            h_tilde = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
            out = s + torch.sigmoid(gate) * (h_tilde - s)
            g = torch.sigmoid(hw)
            self.state[l] = out
            x = g * out + (1 - g) * x
        return x @ self.dec.T

    @torch.no_grad()
    def act(self, obs):
        """obs (n, 510) uint8 cuda -> (n, 6) long actions, sampled per head as ma_pool_act."""
        lg = self.logits(obs)
        out, a = [], 0
        for k in ACTION_SIZES:
            p = torch.softmax(lg[:, a:a + k], -1)
            out.append(torch.multinomial(p, 1, generator=self.gen).squeeze(-1))
            a += k
        return torch.stack(out, -1)


_OFF = None


def ego_obs_gpu(ent, reward, walls, statics):
    """ent (B,206,F) float cuda, reward (B,10,4) cuda, walls (16384,) uint8 cuda, statics -> (B,10,510) uint8 cuda.
    Same bytes as DreamEnv.obs() + mobawm.obs.ego_obs (view windows clamped inside the map buffer)."""
    global _OFF
    dev = ent.device
    if _OFF is None or _OFF.device != dev:
        r = torch.arange(-VISION, VISION + 1, device=dev)
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        _OFF = (dy * MAP + dx).reshape(-1)
    B, N = ent.shape[0], ent.shape[1]
    e = ent.clone()
    e[:, :N_HEROES, F["x"]] = e[:, :N_HEROES, F["x"]].clamp(VISION, 127 - VISION)
    e[:, :N_HEROES, F["y"]] = e[:, :N_HEROES, F["y"]].clamp(VISION, 127 - VISION)
    alive = e[..., F["alive"]] > 0
    cell = (e[..., F["y"]].long() * MAP + e[..., F["x"]].long()).clamp(0, MAP * MAP - 1)
    # numpy fancy assignment: for several entities in one cell, the highest entity index wins
    ids = torch.arange(N, device=dev).expand(B, N)
    top = torch.full((B, MAP * MAP), -1, dtype=torch.long, device=dev)
    top.scatter_reduce_(1, torch.where(alive, cell, torch.zeros_like(cell)), torch.where(alive, ids, torch.full_like(ids, -1)),
                        reduce="amax", include_self=True)
    has = top >= 0
    gid = torch.gather(e[..., F["grid_id"]], 1, top.clamp(min=0)).to(torch.uint8)
    grid = torch.where(has, gid, walls.expand(B, -1))
    pids = torch.where(has, top, torch.full_like(top, NONE))
    out = torch.zeros(B, N_HEROES, OBS_DIM, dtype=torch.uint8, device=dev)
    p = e[:, :N_HEROES]
    u8 = lambda v: torch.trunc(v).long().to(torch.uint8)
    ex = out[..., EXTRA:]
    ex[..., 0] = p[..., F["x"]].long().to(torch.uint8)
    ex[..., 1] = p[..., F["y"]].long().to(torch.uint8)
    ex[..., 2] = p[..., F["level"]].long().to(torch.uint8)
    ex[..., 3] = u8(10.0 * p[..., F["health"]] / p[..., F["max_health"]])
    ex[..., 4] = u8(10.0 * p[..., F["mana"]] / p[..., F["max_mana"]])
    ex[..., 5] = u8(p[..., F["damage"]].double() / 50.0)
    ex[..., 6] = u8(torch.as_tensor(statics["move_speed"][:N_HEROES], dtype=torch.float32, device=dev)).expand(B, -1)
    ex[..., 7] = u8(p[..., F["move_modifier"]])
    for i, name in enumerate(("stun_timer", "move_timer", "q_timer", "w_timer", "e_timer", "basic_attack_timer"), start=8):
        ex[..., i] = p[..., F[name]].long().to(torch.uint8)
    ex[..., 14] = p[..., F["basic_attack_cd"]].long().to(torch.uint8)
    ex[..., 15] = p[..., F["is_hit"]].long().to(torch.uint8)
    ex[..., 16] = p[..., F["team"]].long().to(torch.uint8)
    hero = torch.as_tensor(statics["hero_type"][:N_HEROES].astype(np.int64), device=dev)
    ex[:, torch.arange(N_HEROES, device=dev), 17 + hero] = 1
    ex[..., 22:26] = (reward[:, :N_HEROES] != 0).to(torch.uint8)
    xy = p[..., F["y"]].long() * MAP + p[..., F["x"]].long()
    adr = xy[..., None] + _OFF                                                     # (B,10,121)
    out[..., :WIN * WIN] = torch.gather(grid, 1, adr.reshape(B, -1)).reshape(B, N_HEROES, -1)
    cells = torch.gather(pids, 1, adr[..., -3:].reshape(B, -1)).reshape(B, N_HEROES, 3)
    for j, c in enumerate(range(WIN * WIN - 3, WIN * WIN)):
        idx = cells[..., j]
        hs = idx != NONE
        tgt = torch.gather(e, 1, torch.where(hs, idx, torch.zeros_like(idx))[..., None].expand(-1, -1, e.shape[-1]))
        hp = torch.nan_to_num(10.0 * tgt[..., F["health"]] / tgt[..., F["max_health"]])
        mp = torch.nan_to_num(10.0 * tgt[..., F["mana"]] / tgt[..., F["max_mana"]])
        for k, val, m in ((1, u8(hp), hs), (2, u8(mp), hs & (tgt[..., F["max_mana"]] > 0)), (3, u8(tgt[..., F["level"]]), hs)):
            pos = c + k
            if pos >= WIN * WIN:
                out[..., pos] = torch.where(m, val, out[..., pos])
    return out
