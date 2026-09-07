"""Corpus assembly for behaviour cloning and world-model training.

The corpus is written as four slices (see tools/make_dataset.sh):

    expert   centerline, no noise    strong, clean
    whisker  whisker,    no noise    sensor-limited, crash-prone
    noisy    centerline, eps 0.15    world-model coverage only
    random   uniform actions         world-model coverage only

BC and the world model want different mixes, so the slice is a first-class
selector rather than something baked into one merged file. BC should not clone
a deliberately-degraded policy; the world model should still see the actions a
good policy never takes.

Train/validation is split by **track**, never by row. Consecutive rows are
near-duplicates, so a random row split leaks the validation set into training
and makes every model look excellent.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import contract, shards

SLICES = ("expert", "whisker", "noisy", "random")
BC_SLICES = ("expert", "whisker")            # clean policies only
WM_SLICES = ("expert", "whisker", "noisy", "random")

GEOM_DIM = 44        # 11 window offsets x 2 edges x (ex, ey)
GEOM_SCALE = 100.0   # max_whisker_length; keeps features O(1)


@dataclass
class Corpus:
    obs: np.ndarray        # (N, 30) float32  s_t
    action: np.ndarray     # (N,)   int64
    reward: np.ndarray     # (N,)   float32
    next_obs: np.ndarray   # (N, 30) float32  s_{t+1}
    env_idx: np.ndarray    # (N,)   int32   track id
    slice_id: np.ndarray   # (N,)   int8    index into SLICES
    geom: np.ndarray | None  # (N, 44) float32 ego-frame local track geometry
    header: dict
    schema_hash: str
    split: str = "?"

    def __len__(self) -> int:
        return len(self.obs)

    def counts(self) -> dict[str, int]:
        return {SLICES[i]: int((self.slice_id == i).sum())
                for i in range(len(SLICES)) if (self.slice_id == i).any()}

    def select(self, slices) -> "Corpus":
        want = {SLICES.index(s) for s in slices}
        m = np.isin(self.slice_id, list(want))
        return self._subset(m)

    def split_by_track(self, val_frac: float = 0.2, seed: int = 0):
        """Sub-split an already-loaded split by track.

        Not the primary mechanism: train/val/holdout are separate files over
        disjoint track ranges (see load_split). This exists for ad-hoc
        analysis inside one split.
        """
        tracks = np.unique(self.env_idx)
        rng = np.random.default_rng(seed)
        rng.shuffle(tracks)
        n_val = max(1, int(round(len(tracks) * val_frac)))
        val_tracks = set(tracks[:n_val].tolist())
        m = np.array([e in val_tracks for e in self.env_idx])
        return self._subset(~m), self._subset(m)

    def _subset(self, m: np.ndarray) -> "Corpus":
        return Corpus(
            obs=self.obs[m], action=self.action[m], reward=self.reward[m],
            next_obs=self.next_obs[m], env_idx=self.env_idx[m],
            slice_id=self.slice_id[m],
            geom=None if self.geom is None else self.geom[m],
            header=self.header, schema_hash=self.schema_hash, split=self.split)


def _local_geometry(obs, env_idx, tracks, hdr) -> np.ndarray:
    """The ±5 edge window compute_whiskers reads, in the car's frame."""
    d = contract.denorm(obs, hdr)
    out = np.zeros((len(obs), GEOM_DIM), dtype=np.float32)
    for e in np.unique(env_idx):
        m = env_idx == e
        out[m] = tracks.local_window(int(e), d["near_point_idx"][m],
                                     d["px"][m], d["py"][m], d["ang"][m])
    return out / GEOM_SCALE


def load_split(corpus_dir: str | Path, split: str, slices=SLICES,
               with_geometry: bool = True, max_rows: int | None = None,
               seed: int = 0) -> Corpus:
    """Load one split. Splits live in separate files over disjoint track
    ranges (see tools/make_dataset.sh), so this cannot accidentally mix them."""
    d = Path(corpus_dir)
    if split not in ("train", "val", "holdout"):
        raise ValueError(f"unknown split {split!r}")

    parts, ids = [], []
    for i, name in enumerate(SLICES):
        if name not in slices:
            continue
        try:
            tr = shards.load(str(d / f"{split}_{name}.*.bin"))
        except FileNotFoundError:
            continue
        parts.append(tr)
        ids.append(np.full(len(tr), i, dtype=np.int8))
    if not parts:
        raise FileNotFoundError(f"no slices for split {split!r} in {d}")

    obs = np.concatenate([p.obs for p in parts])
    action = np.concatenate([p.action for p in parts])
    reward = np.concatenate([p.reward for p in parts])
    next_obs = np.concatenate([p.next_obs for p in parts])
    env_idx = np.concatenate([p.env_idx for p in parts])
    slice_id = np.concatenate(ids)

    if max_rows is not None and len(obs) > max_rows:
        keep = np.random.default_rng(seed).choice(len(obs), max_rows, replace=False)
        keep.sort()
        obs, action, reward, next_obs, env_idx, slice_id = (
            obs[keep], action[keep], reward[keep], next_obs[keep],
            env_idx[keep], slice_id[keep])

    hdr = shards.read_header(shards._shard_paths(str(d / f"{split}_*.bin"))[0])
    geom = None
    if with_geometry:
        tracks = shards.load_tracks(str(d / f"{split}.tracks.bin"))
        geom = _local_geometry(obs, env_idx, tracks, hdr)

    return Corpus(obs=obs, action=action, reward=reward, next_obs=next_obs,
                  env_idx=env_idx, slice_id=slice_id, geom=geom, header=hdr,
                  schema_hash=contract.schema_hash(), split=split)


def load_manifest(corpus_dir: str | Path) -> dict:
    return json.loads((Path(corpus_dir) / "manifest.json").read_text())


def assert_disjoint(corpus_dir: str | Path) -> dict[str, list[int]]:
    """Prove the splits share no track. Cheap, and the failure it guards
    against -- a holdout number quietly measured on training tracks -- is
    invisible in every downstream metric."""
    d = Path(corpus_dir)
    seen = {}
    for split in ("train", "val", "holdout"):
        ids = set()
        for path in shards._shard_paths(str(d / f"{split}_*.bin")):
            ids |= set(np.unique(shards.read_shard(path)["env_idx"]).tolist())
        seen[split] = sorted(ids)
    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        overlap = set(seen[a]) & set(seen[b])
        if overlap:
            raise AssertionError(
                f"track leakage: {a} and {b} share env_idx {sorted(overlap)[:10]}")
    return seen


# ------------------------------------------------------------------ features
def wm_inputs(c: Corpus, use_geometry: bool = True) -> np.ndarray:
    """World-model input: state, action one-hot, and the local track geometry.

    Geometry is included because the transition function reads it every step;
    without it the problem is not Markov (measured: +19% history gain without,
    +4% with). See docs/AUDIT.md.
    """
    onehot = np.zeros((len(c), contract.NUM_ACTIONS), dtype=np.float32)
    onehot[np.arange(len(c)), c.action] = 1.0
    parts = [c.obs, onehot]
    if use_geometry:
        if c.geom is None:
            raise ValueError("corpus loaded without geometry")
        parts.append(c.geom)
    return np.concatenate(parts, axis=1).astype(np.float32)


def wm_targets(c: Corpus) -> np.ndarray:
    """Predict the delta, not the absolute next state. Most dimensions barely
    move per step, so absolute regression spends its capacity re-encoding the
    input and hides whether the dynamics were learned at all."""
    return (c.next_obs - c.obs).astype(np.float32)


def bc_inputs(c: Corpus, privileged: bool = False) -> np.ndarray:
    """BC policy input. By default only the agent's observation.

    `privileged=True` adds the track geometry, which is what the centerline bot
    actually used. Comparing the two measures how much of the expert's skill is
    simply unavailable to a policy restricted to whiskers.
    """
    if not privileged:
        return c.obs.astype(np.float32)
    if c.geom is None:
        raise ValueError("corpus loaded without geometry")
    return np.concatenate([c.obs, c.geom], axis=1).astype(np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="data/corpus")
    a = ap.parse_args()

    man = load_manifest(a.corpus)
    print(f"corpus {a.corpus}  seed={man['seed']}  frameskip={man['frameskip']}  "
          f"episodes/track={man['episodes_per_track']}")
    print("  track ranges:")
    for k, v in man["splits"].items():
        lo = v["track_start"]; hi = lo + v["n_tracks"] - 1
        print(f"    {k:<8} env_idx {lo:>3}..{hi:<3} ({v['n_tracks']} tracks)")

    seen = assert_disjoint(a.corpus)
    print("  disjoint: OK  " + "  ".join(
        f"{k}={min(v)}..{max(v)}" for k, v in seen.items()))

    total = 0
    for split in ("train", "val", "holdout"):
        c = load_split(a.corpus, split, with_geometry=False)
        total += len(c)
        print(f"  {split:<8} {len(c):>9,} transitions  "
              f"{len(np.unique(c.env_idx)):>3} tracks  {c.counts()}")
    print(f"  {'total':<8} {total:>9,}")