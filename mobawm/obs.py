"""Ego observations as a pure, batched function of the full game state.

numpy port of moba_ego_obs() in moba/moba_state.h, which is itself a copy of
upstream compute_observations(). Given states that no env ever held -- e.g.
predicted by a transition model inside a dream -- this produces exactly the
bytes the pretrained policy would have seen in the real game. Byte-exactness
against the env is checked by ``python -m mobawm.obs DENSE.mbs``.

Upstream quirks are reproduced on purpose (see moba_state.h):
bytes 0..120 are tile ids, 121..123 are leftovers from the last window cells,
124..483 are always zero, and windows wrap across rows at the map edge.
"""
from __future__ import annotations

import numpy as np

from .shards import F, MAP, N_HEROES, NONE, OBS_DIM

VISION = 5
WIN = 2 * VISION + 1          # 11
EXTRA = WIN * WIN * 4         # 484: start of the per-agent scalars

_dy, _dx = np.meshgrid(np.arange(-VISION, VISION + 1), np.arange(-VISION, VISION + 1), indexing="ij")
_OFFSETS = (_dy * MAP + _dx).reshape(-1)   # flat-index offsets, upstream loop order


def _u8(v: np.ndarray) -> np.ndarray:
    """C's float -> unsigned char: truncate toward zero (values are in range)."""
    return np.trunc(v).astype(np.int64).astype(np.uint8)


def ego_obs(entity: np.ndarray, reward: np.ndarray, grid: np.ndarray, pids: np.ndarray,
            statics: np.ndarray) -> np.ndarray:
    """(B,206,F) f32, (B,10,4) f32, (B,16384) u8, (B,16384) u8, (10,) statics
    -> (B, 10, 510) uint8: one observation per hero, radiant 0-4 then dire 5-9."""
    entity = np.asarray(entity, np.float32)
    B = entity.shape[0]
    out = np.zeros((B, N_HEROES, OBS_DIM), np.uint8)
    p = entity[:, :N_HEROES]                                   # (B,10,20)
    f10 = np.float32(10)

    x = p[..., F["x"]].astype(np.int64)
    y = p[..., F["y"]].astype(np.int64)
    ex = out[..., EXTRA:]
    ex[..., 0] = x
    ex[..., 1] = y
    ex[..., 2] = p[..., F["level"]].astype(np.int64)
    ex[..., 3] = _u8(f10 * p[..., F["health"]] / p[..., F["max_health"]])
    ex[..., 4] = _u8(f10 * p[..., F["mana"]] / p[..., F["max_mana"]])
    ex[..., 5] = _u8(p[..., F["damage"]].astype(np.float64) / 50.0)
    ex[..., 6] = _u8(statics["move_speed"][:N_HEROES])
    ex[..., 7] = _u8(p[..., F["move_modifier"]])
    for i, name in enumerate(("stun_timer", "move_timer", "q_timer", "w_timer",
                              "e_timer", "basic_attack_timer"), start=8):
        ex[..., i] = p[..., F[name]].astype(np.int64)
    ex[..., 14] = p[..., F["basic_attack_cd"]].astype(np.int64)
    ex[..., 15] = p[..., F["is_hit"]].astype(np.int64)
    ex[..., 16] = p[..., F["team"]].astype(np.int64)
    hero = statics["hero_type"][:N_HEROES].astype(np.int64)
    ex[:, np.arange(N_HEROES), 17 + hero] = 1
    ex[..., 22:26] = np.asarray(reward)[:, :N_HEROES] != 0

    adr = (y * MAP + x)[..., None] + _OFFSETS                  # (B,10,121)
    if adr.min() < 0 or adr.max() >= MAP * MAP:
        raise ValueError("ego window leaves the map buffer (upstream reads out of bounds here)")
    bidx = np.arange(B)[:, None, None]
    out[..., :WIN * WIN] = np.asarray(grid)[bidx, adr]

    # Only the last three window cells can leave anything behind in 121..123.
    cells = np.asarray(pids)[bidx, adr[..., -3:]].astype(np.int64)   # (B,10,3)
    for j, c in enumerate(range(WIN * WIN - 3, WIN * WIN)):
        e = cells[..., j]
        has = e != NONE
        tgt = entity[np.arange(B)[:, None], np.where(has, e, 0)]     # (B,10,20)
        with np.errstate(divide="ignore", invalid="ignore"):
            chans = (
                (1, _u8(np.nan_to_num(f10 * tgt[..., F["health"]] / tgt[..., F["max_health"]])), has),
                (2, _u8(np.nan_to_num(f10 * tgt[..., F["mana"]] / tgt[..., F["max_mana"]])),
                 has & (tgt[..., F["max_mana"]] > 0)),
                (3, _u8(tgt[..., F["level"]]), has),
            )
        for k, val, mask in chans:
            pos = c + k
            if pos >= WIN * WIN:                               # lower bytes are overwritten by tiles
                out[..., pos] = np.where(mask, val, out[..., pos])
    return out


def ego_obs_from_states(states: np.ndarray, statics: np.ndarray) -> np.ndarray:
    """Convenience: states is a STATE_DTYPE array of shape (B,)."""
    return ego_obs(states["entity"], states["reward"], states["grid"], states["pids"], statics)


if __name__ == "__main__":
    import sys

    from .shards import load_dense

    d = load_dense(sys.argv[1])
    bad = n = 0
    for recs in d.iter_games():
        for lo in range(0, len(recs), 256):
            r = recs[lo:lo + 256]
            mine = ego_obs_from_states(r["state"], d.statics)
            bad += int((mine != r["obs"]).any(axis=(1, 2)).sum())
            n += len(r)
    print(f"{d.path}: {n} records, records where mobawm.obs.ego_obs != env obs: {bad}")
    sys.exit(1 if bad else 0)
