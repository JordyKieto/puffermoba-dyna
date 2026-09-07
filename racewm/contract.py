"""The Race state contract.

The observation IS the state. Upstream whisker_racer observed 3 floats
(two whiskers plus score/100) while the transition depended on heading,
position, the nearest-centerline index used to window the whisker raycast,
and sector bookkeeping that determines every unit of reward. See docs/AUDIT.md.

This module is the single source of truth on the Python side. It is checked
against race/race.h by tests/test_contract.py, so the two cannot drift.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_VERSION = 1
MAX_SECTORS = 16
OBS_SIZE = 14 + MAX_SECTORS  # 30


@dataclass(frozen=True)
class Field:
    name: str
    index: int
    kind: str      # "unit" | "signed" | "count" | "binary"
    scale: str     # how to invert back to raw env units
    note: str = ""


def _fields() -> list[Field]:
    f = [
        Field("flw_length", 0, "unit", "x", "whisker 0, geometric RIGHT (see race_bots.h)"),
        Field("frw_length", 1, "unit", "x", "whisker 1, geometric LEFT"),
        Field("sin_ang", 2, "signed", "atan2(sin, cos)", "heading; sin/cos because ang wraps at 2pi"),
        Field("cos_ang", 3, "signed", "atan2(sin, cos)", ""),
        Field("px_norm", 4, "unit", "x * width", ""),
        Field("py_norm", 5, "unit", "x * height", ""),
        Field("v_norm", 6, "unit", "x * maxv", "constant maxv upstream; no throttle exists"),
        Field("sin_sector", 7, "signed", "cyclic(S)", "current_sector, cyclic index"),
        Field("cos_sector", 8, "signed", "cyclic(S)", ""),
        Field("sin_near_point", 9, "signed", "cyclic(total_points)",
              "near_point_idx; the whisker raycast only scans +-5 segments around it, "
              "so it is load-bearing dynamics state, not a convenience field"),
        Field("cos_near_point", 10, "signed", "cyclic(total_points)", ""),
        Field("sectors_crossed_norm", 11, "count", "x * S", "distinct sectors visited this episode"),
        Field("tick_norm", 12, "unit", "x * max_episode_steps",
              "the episode timer; upstream had no time limit and no clock in obs"),
        Field("score_norm", 13, "count", "x * 100", "upstream's obs[2] scale, kept"),
    ]
    f += [
        Field(f"sector_completed_{i}", 14 + i, "binary", "x",
              "gates total_sectors_crossed only; reward is identical either way")
        for i in range(MAX_SECTORS)
    ]
    return f


FIELDS: tuple[Field, ...] = tuple(_fields())
NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)
INDEX = {f.name: f.index for f in FIELDS}

assert len(FIELDS) == OBS_SIZE, (len(FIELDS), OBS_SIZE)
assert [f.index for f in FIELDS] == list(range(OBS_SIZE))

# Dimensions that are angles encoded as a (sin, cos) pair. A world model must
# not regress these independently -- renormalize the pair after prediction.
ANGLE_PAIRS: tuple[tuple[int, int], ...] = (
    (INDEX["sin_ang"], INDEX["cos_ang"]),
    (INDEX["sin_sector"], INDEX["cos_sector"]),
    (INDEX["sin_near_point"], INDEX["cos_near_point"]),
)

# Dimensions that only ever move by +0 or +1 step (monotone within an episode).
MONOTONE = (INDEX["sectors_crossed_norm"], INDEX["tick_norm"], INDEX["score_norm"])

BINARY = tuple(INDEX[f"sector_completed_{i}"] for i in range(MAX_SECTORS))

NUM_ACTIONS = 3
ACTION_NAMES = ("LEFT", "NOOP", "RIGHT")


def schema_hash() -> str:
    """Stable hash of the layout. Embedded in every dataset shard manifest."""
    payload = "|".join(f"{f.index}:{f.name}:{f.kind}" for f in FIELDS)
    payload = f"v{STATE_VERSION}|S{MAX_SECTORS}|A{NUM_ACTIONS}|{payload}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def parse_header(header_path: str | Path) -> dict[str, int]:
    """Read the constants race.h actually compiles with."""
    src = Path(header_path).read_text()
    out = {}
    for key in ("RACE_STATE_VERSION", "RACE_MAX_SECTORS"):
        m = re.search(rf"^#define\s+{key}\s+(\d+)\s*$", src, re.M)
        if not m:
            raise ValueError(f"{key} not found in {header_path}")
        out[key] = int(m.group(1))
    m = re.search(r"^#define\s+RACE_OBS_SIZE\s+\((\d+)\s*\+\s*RACE_MAX_SECTORS\)\s*$", src, re.M)
    if not m:
        raise ValueError(f"RACE_OBS_SIZE not found in {header_path}")
    out["RACE_OBS_SIZE"] = int(m.group(1)) + out["RACE_MAX_SECTORS"]
    return out


def check_against_header(header_path: str | Path) -> None:
    """Raise if race.h and this module have drifted."""
    h = parse_header(header_path)
    problems = []
    if h["RACE_STATE_VERSION"] != STATE_VERSION:
        problems.append(f"STATE_VERSION: race.h={h['RACE_STATE_VERSION']} contract.py={STATE_VERSION}")
    if h["RACE_MAX_SECTORS"] != MAX_SECTORS:
        problems.append(f"MAX_SECTORS: race.h={h['RACE_MAX_SECTORS']} contract.py={MAX_SECTORS}")
    if h["RACE_OBS_SIZE"] != OBS_SIZE:
        problems.append(f"OBS_SIZE: race.h={h['RACE_OBS_SIZE']} contract.py={OBS_SIZE}")
    if problems:
        raise AssertionError("race.h / contract.py drift:\n  " + "\n  ".join(problems))


def decode_angles(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Recover the raw cyclic quantities from their sin/cos encodings."""
    o = np.asarray(obs)
    return {
        "ang": np.arctan2(o[..., INDEX["sin_ang"]], o[..., INDEX["cos_ang"]]),
        "sector_phase": np.arctan2(o[..., INDEX["sin_sector"]], o[..., INDEX["cos_sector"]]),
        "near_point_phase": np.arctan2(o[..., INDEX["sin_near_point"]], o[..., INDEX["cos_near_point"]]),
    }



def denorm(obs: np.ndarray, hdr: dict) -> dict[str, np.ndarray]:
    """Recover the raw env quantities the track lookup needs.

    `hdr` is a shard header (racewm.shards.read_header); the normalization
    constants travel with the data so this can never be applied against the
    wrong config.
    """
    o = np.asarray(obs)
    p = int(hdr["total_points"])
    phase = np.arctan2(o[..., INDEX["sin_near_point"]], o[..., INDEX["cos_near_point"]])
    return {
        "px": o[..., INDEX["px_norm"]] * hdr["width"],
        "py": o[..., INDEX["py_norm"]] * hdr["height"],
        "ang": np.arctan2(o[..., INDEX["sin_ang"]], o[..., INDEX["cos_ang"]]),
        # near_point_idx is an integer index encoded as a phase; round back.
        "near_point_idx": (np.rint((phase % (2 * np.pi)) / (2 * np.pi) * p).astype(np.int64) % p),
    }

if __name__ == "__main__":
    print(f"Race state contract v{STATE_VERSION}")
    print(f"  obs_size    {OBS_SIZE}")
    print(f"  schema_hash {schema_hash()}")
    for f in FIELDS:
        note = f"  # {f.note}" if f.note else ""
        print(f"  [{f.index:2d}] {f.name:<24} {f.kind:<7}{note}")
