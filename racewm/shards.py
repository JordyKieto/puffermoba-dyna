"""Reader for the binary transition shards written by build/race_eval --dump.

Layout mirrors ShardHeader / Record in race/race_eval.c. Each record holds the
state BEFORE the step, so s_{t+1} is the next record in the same episode --
which is why episode boundaries have to be honored when building pairs.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contract

SHARD_MAGIC = 0x52434531  # "RCE1"

HEADER_DTYPE = np.dtype([
    ("magic", "<u4"), ("state_version", "<u4"), ("obs_dim", "<u4"),
    ("record_bytes", "<u4"), ("seed", "<u4"), ("n_records", "<u4"),
    ("num_radial_sectors", "<u4"), ("max_episode_steps", "<u4"),
    ("width", "<f4"), ("height", "<f4"), ("maxv", "<f4"),
    ("frameskip", "<u4"), ("total_points", "<u4"), ("reserved", "<u4", (3,)),
])

RECORD_DTYPE = np.dtype([
    ("env_idx", "<i4"), ("episode_id", "<i4"), ("t", "<i4"), ("action", "<i4"),
    ("reward", "<f4"), ("terminated", "u1"), ("truncated", "u1"),
    ("bot_kind", "u1"), ("pad", "u1"), ("obs", "<f4", (contract.OBS_SIZE,)),
])


@dataclass
class Transitions:
    """(s, a, r, s') pairs with episode boundaries already removed."""
    obs: np.ndarray        # (N, OBS_SIZE) float32  -- s_t
    action: np.ndarray     # (N,) int64
    reward: np.ndarray     # (N,) float32
    next_obs: np.ndarray   # (N, OBS_SIZE) float32  -- s_{t+1}
    done: np.ndarray       # (N,) bool  (terminated OR truncated at s_t -> s_{t+1})
    env_idx: np.ndarray    # (N,) int32
    bot_kind: np.ndarray   # (N,) uint8
    schema_hash: str

    def __len__(self) -> int:
        return len(self.obs)

    def split(self, frac: float = 0.9, seed: int = 0):
        """Split by env_idx, not by row: adjacent rows are near-duplicates, so a
        random row split leaks the test set into training and makes every model
        look good."""
        rng = np.random.default_rng(seed)
        envs = np.unique(self.env_idx)
        rng.shuffle(envs)
        n_train = max(1, int(len(envs) * frac))
        train_envs = set(envs[:n_train].tolist())
        mask = np.array([e in train_envs for e in self.env_idx])
        return self._subset(mask), self._subset(~mask)

    def _subset(self, mask: np.ndarray) -> "Transitions":
        return Transitions(
            obs=self.obs[mask], action=self.action[mask], reward=self.reward[mask],
            next_obs=self.next_obs[mask], done=self.done[mask],
            env_idx=self.env_idx[mask], bot_kind=self.bot_kind[mask],
            schema_hash=self.schema_hash,
        )


def read_header(path: str | Path) -> dict:
    raw = Path(path).read_bytes()[:HEADER_DTYPE.itemsize]
    h = np.frombuffer(raw, dtype=HEADER_DTYPE)[0]
    return {k: h[k].item() for k in HEADER_DTYPE.names if k != "reserved"}


def read_shard(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    hdr = np.frombuffer(raw[:HEADER_DTYPE.itemsize], dtype=HEADER_DTYPE)[0]
    if int(hdr["magic"]) != SHARD_MAGIC:
        raise ValueError(f"{path}: bad magic {hdr['magic']:#x}")
    if int(hdr["state_version"]) != contract.STATE_VERSION:
        raise ValueError(
            f"{path}: shard is state v{hdr['state_version']}, contract.py is "
            f"v{contract.STATE_VERSION}. Regenerate the dataset.")
    if int(hdr["obs_dim"]) != contract.OBS_SIZE:
        raise ValueError(f"{path}: obs_dim {hdr['obs_dim']} != {contract.OBS_SIZE}")
    if int(hdr["record_bytes"]) != RECORD_DTYPE.itemsize:
        raise ValueError(
            f"{path}: record is {hdr['record_bytes']}B, reader expects "
            f"{RECORD_DTYPE.itemsize}B -- struct layout drift")
    body = raw[HEADER_DTYPE.itemsize:]
    n = len(body) // RECORD_DTYPE.itemsize
    return np.frombuffer(body[:n * RECORD_DTYPE.itemsize], dtype=RECORD_DTYPE)


def _shard_paths(pattern: str) -> list[str]:
    """Transition shards only. PREFIX.tracks.bin sits next to PREFIX.NNN.bin and
    matches the same glob, but it is a different format."""
    paths = [p for p in sorted(glob.glob(pattern)) if not p.endswith(".tracks.bin")]
    if not paths:
        raise FileNotFoundError(f"no transition shards matched {pattern!r}")
    return paths


def load(pattern: str) -> Transitions:
    paths = _shard_paths(pattern)
    recs = np.concatenate([read_shard(p) for p in paths])

    # A record is a valid (s, a, r, s') pair only when the NEXT record continues
    # the same episode on the same env. c_step resets in place on termination,
    # so the record after a terminal already holds the new episode's first obs.
    same_env = recs["env_idx"][:-1] == recs["env_idx"][1:]
    same_ep = recs["episode_id"][:-1] == recs["episode_id"][1:]
    not_done = (recs["terminated"][:-1] == 0) & (recs["truncated"][:-1] == 0)
    valid = same_env & same_ep & not_done

    idx = np.flatnonzero(valid)
    return Transitions(
        obs=recs["obs"][idx].astype(np.float32),
        action=recs["action"][idx].astype(np.int64),
        reward=recs["reward"][idx].astype(np.float32),
        next_obs=recs["obs"][idx + 1].astype(np.float32),
        done=np.zeros(len(idx), dtype=bool),
        env_idx=recs["env_idx"][idx].astype(np.int32),
        bot_kind=recs["bot_kind"][idx].astype(np.uint8),
        schema_hash=contract.schema_hash(),
    )


def load_sequences(pattern: str, length: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous within-episode windows, for history and rollout evaluation.

    Returns (obs[N, length, OBS_SIZE], actions[N, length], env_idx[N]).

    env_idx comes back because the transition function depends on the track,
    and the track is NOT part of the state. Mixing tracks in one regression
    makes the dynamics look non-Markovian for a reason that has nothing to do
    with the state contract. See docs/CONTRACT.md.
    """
    paths = _shard_paths(pattern)
    recs = np.concatenate([read_shard(p) for p in paths])

    # Episode id restarts per env, so key on the pair.
    key = recs["env_idx"].astype(np.int64) * (2**20) + recs["episode_id"].astype(np.int64)
    boundaries = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(recs)]])

    obs_out, act_out, env_out = [], [], []
    for s, e in zip(starts, ends):
        # Drop the final step of each episode: its successor is a fresh episode.
        n = e - s - 1
        if n < length:
            continue
        for start in range(s, s + n - length + 1):
            obs_out.append(recs["obs"][start:start + length])
            act_out.append(recs["action"][start:start + length])
            env_out.append(recs["env_idx"][start])
    if not obs_out:
        raise ValueError(f"no episode long enough for length={length}")
    return (np.stack(obs_out).astype(np.float32),
            np.stack(act_out).astype(np.int64),
            np.asarray(env_out, dtype=np.int32))


TRACK_MAGIC = 0x52434B31  # "RCK1"


@dataclass
class Tracks:
    """Episode-constant track geometry, keyed by env_idx.

    The dynamics read this every step -- compute_whiskers intersects the edge
    polylines, update_nearest_point walks the centerline -- so it is part of
    the Markov state. It just happens to be the part that is constant within an
    episode, which is why it lives here and not in the per-step observation.
    """
    centerline: dict[int, np.ndarray]   # env_idx -> (P, 2)
    inner: dict[int, np.ndarray]
    outer: dict[int, np.ndarray]

    def local_window(self, env_idx: int, near_idx: np.ndarray, px: np.ndarray,
                     py: np.ndarray, ang: np.ndarray, half_width: int = 5) -> np.ndarray:
        """Edge points around near_point_idx, in the car's ego frame.

        This is exactly the geometry compute_whiskers can see: it scans
        offsets -5..+5 around near_point_idx (window_size 10) against both
        edges. Ego-framing it makes the feature translation- and
        rotation-invariant, so a model can generalize across tracks instead of
        memorizing absolute pixel coordinates.
        """
        inner = self.inner[env_idx]
        outer = self.outer[env_idx]
        p = len(inner)
        offs = np.arange(-half_width, half_width + 1)
        idx = (near_idx[:, None] + offs[None, :]) % p        # (N, W)

        cos_a, sin_a = np.cos(-ang), np.sin(-ang)
        feats = []
        for edge in (inner, outer):
            pts = edge[idx]                                   # (N, W, 2)
            dx = pts[..., 0] - px[:, None]
            dy = pts[..., 1] - py[:, None]
            ex = dx * cos_a[:, None] - dy * sin_a[:, None]
            ey = dx * sin_a[:, None] + dy * cos_a[:, None]
            feats += [ex, ey]
        return np.concatenate(feats, axis=1).astype(np.float32)   # (N, 4W+4)


def load_tracks(path: str | Path) -> Tracks:
    raw = Path(path).read_bytes()
    hdr = np.frombuffer(raw[:16], dtype=np.dtype([
        ("magic", "<u4"), ("version", "<u4"), ("n_tracks", "<u4"), ("seed", "<u4")]))[0]
    if int(hdr["magic"]) != TRACK_MAGIC:
        raise ValueError(f"{path}: bad track magic {hdr['magic']:#x}")
    if int(hdr["version"]) != contract.STATE_VERSION:
        raise ValueError(f"{path}: track file is v{hdr['version']}, "
                         f"contract.py is v{contract.STATE_VERSION}")

    centerline, inner, outer = {}, {}, {}
    off = 16
    for _ in range(int(hdr["n_tracks"])):
        env_idx, total = np.frombuffer(raw[off:off + 8], dtype="<i4")
        off += 8
        arrs = []
        for _ in range(3):
            n = int(total) * 2 * 4
            arrs.append(np.frombuffer(raw[off:off + n], dtype="<f4").reshape(-1, 2).copy())
            off += n
        centerline[int(env_idx)], inner[int(env_idx)], outer[int(env_idx)] = arrs
    return Tracks(centerline=centerline, inner=inner, outer=outer)
