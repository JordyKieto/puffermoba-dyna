"""Torch-native track geometry lookup.

The numpy path in racewm/shards.py is fine for building a dataset once. Rolling
the world model forward inside a training loop needs the same gather every
step, from the model's own predicted pose, so it has to run on-device and stay
in the autograd graph.

Gradient flow is deliberately partial. The *index* into the polyline comes from
rounding a decoded angle and is not differentiable -- there is no gradient path
through "which wall segment am I near". The *coordinates* are: the ego-frame
transform uses px, py and ang directly, so the model still learns how its
predicted pose changes what it will see. That is the right split; making the
index soft would blur across track segments that are genuinely discrete.
"""
from __future__ import annotations

import numpy as np
import torch

from . import contract

HALF_WINDOW = 5          # compute_whiskers scans offsets -5..+5
GEOM_DIM = 4 * (2 * HALF_WINDOW + 1)     # 2 edges x 2 coords x 11 offsets = 44
GEOM_SCALE = 100.0       # max_whisker_length


class TrackBank:
    """All track polylines as tensors, indexed by a dense slot per env_idx."""

    def __init__(self, tracks, header: dict, device="cpu"):
        env_ids = sorted(tracks.inner.keys())
        self.slot_of = {e: i for i, e in enumerate(env_ids)}
        self.n_points = int(header["total_points"])
        self.width = float(header["width"])
        self.height = float(header["height"])

        inner = np.stack([tracks.inner[e] for e in env_ids])   # (T, P, 2)
        outer = np.stack([tracks.outer[e] for e in env_ids])
        self.inner = torch.from_numpy(inner.astype(np.float32)).to(device)
        self.outer = torch.from_numpy(outer.astype(np.float32)).to(device)
        self.device = device

        offs = torch.arange(-HALF_WINDOW, HALF_WINDOW + 1, device=device)
        self.offsets = offs                                     # (W,)

    def slots(self, env_idx: np.ndarray) -> torch.Tensor:
        return torch.tensor([self.slot_of[int(e)] for e in env_idx],
                            dtype=torch.long, device=self.device)

    def local_window(self, state: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """(B, 44) ego-frame edge points around the state's nearest segment."""
        i = contract.INDEX
        px = state[:, i["px_norm"]] * self.width
        py = state[:, i["py_norm"]] * self.height
        ang = torch.atan2(state[:, i["sin_ang"]], state[:, i["cos_ang"]])

        phase = torch.atan2(state[:, i["sin_near_point"]], state[:, i["cos_near_point"]])
        P = self.n_points
        near = torch.round((phase % (2 * torch.pi)) / (2 * torch.pi) * P).long() % P

        idx = (near[:, None] + self.offsets[None, :]) % P        # (B, W)
        batch = slots[:, None].expand_as(idx)

        cos_a, sin_a = torch.cos(-ang), torch.sin(-ang)
        feats = []
        for edge in (self.inner, self.outer):
            pts = edge[batch, idx]                               # (B, W, 2)
            dx = pts[..., 0] - px[:, None]
            dy = pts[..., 1] - py[:, None]
            ex = dx * cos_a[:, None] - dy * sin_a[:, None]
            ey = dx * sin_a[:, None] + dy * cos_a[:, None]
            feats += [ex, ey]
        return torch.cat(feats, dim=1) / GEOM_SCALE


def check_against_numpy(bank: TrackBank, tracks, obs: np.ndarray,
                        env_idx: np.ndarray, header: dict, n: int = 256) -> float:
    """Max absolute disagreement with the numpy implementation.

    Two implementations of the same gather is exactly where a silent
    off-by-one lives, and it would show up only as a slightly worse model.
    """
    from . import dataset
    k = min(n, len(obs))
    ref = dataset._local_geometry(obs[:k], env_idx[:k], tracks, header)
    got = bank.local_window(torch.from_numpy(obs[:k]),
                            bank.slots(env_idx[:k])).cpu().numpy()
    return float(np.abs(ref - got).max())
