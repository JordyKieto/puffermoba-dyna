"""World model over the structured Race state, plus the baselines it has to beat.

The model predicts a **delta**, not the next state. Most of the 30 dimensions
barely move in one step, so an absolute-state model spends its capacity
re-encoding its own input and one-step error looks impressive without the
dynamics having been learned at all.

Multi-step rollout is the metric that matters. One-step error is easy and
almost uninformative here: predicting `s_{t+1} = s_t` is already close.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import contract

UNIT_DIMS = [contract.INDEX[n] for n in
             ("flw_length", "frw_length", "px_norm", "py_norm", "v_norm", "tick_norm")]
BINARY_DIMS = list(contract.BINARY)


class DeltaMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = contract.OBS_SIZE,
                 hidden: int = 512, layers: int = 4, dropout: float = 0.0):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout:
                mods.append(nn.Dropout(dropout))
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*mods)
        self.d_in, self.d_out = d_in, d_out

    def forward(self, x):
        return self.net(x)


_CLAMP_CACHE: dict = {}


def _clamp_mask(device, dtype):
    """Boolean row mask marking dimensions that live in [0, 1]."""
    key = (str(device), dtype)
    if key not in _CLAMP_CACHE:
        m = torch.zeros(contract.OBS_SIZE, dtype=torch.bool, device=device)
        for i in UNIT_DIMS + BINARY_DIMS:
            m[i] = True
        _CLAMP_CACHE[key] = m
    return _CLAMP_CACHE[key]


def project_state(s: torch.Tensor) -> torch.Tensor:
    """Push a predicted state back onto the manifold the contract defines.

    Without this, rollouts leave the valid set within a few steps: the sin/cos
    pairs stop having unit norm, so the model's own encoding of heading becomes
    meaningless and every later step compounds it.

    Written without in-place writes. The obvious version (`s = s.clone()` then
    assigning column slices in a loop) works for a single forward pass but
    breaks autograd when the model is unrolled, because the repeated slice
    assignments bump the tensor's version counter between forward and backward.
    """
    scale = torch.ones_like(s)
    for i, j in contract.ANGLE_PAIRS:
        n = torch.sqrt(s[:, i] ** 2 + s[:, j] ** 2).clamp_min(1e-6)
        idx = torch.tensor([i, j], device=s.device)
        scale = scale.index_copy(1, idx, torch.stack([n, n], dim=1))
    out = s / scale
    m = _clamp_mask(s.device, s.dtype)
    return torch.where(m, out.clamp(0.0, 1.0), out)


# ------------------------------------------------------------------ baselines
class Persistence:
    """s_{t+1} = s_t. The floor any world model must clear."""
    name = "persistence"

    def delta(self, s, a_onehot, geom):
        return torch.zeros_like(s)


class MeanDelta:
    """s_{t+1} = s_t + E[delta], the average step from training data. Beats
    persistence wherever motion is consistent, and shows how much of a model's
    apparent skill is just knowing the typical step."""
    name = "mean_delta"

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def delta(self, s, a_onehot, geom):
        return self.mean.expand_as(s)


class ModelWrapper:
    name = "world_model"

    def __init__(self, model, x_mu, x_sd, y_mu, y_sd, use_geometry=True):
        self.model, self.use_geometry = model, use_geometry
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd

    def delta(self, s, a_onehot, geom):
        parts = [s, a_onehot] + ([geom] if self.use_geometry else [])
        x = torch.cat(parts, dim=1)
        x = (x - self.x_mu) / self.x_sd
        return self.model(x) * self.y_sd + self.y_mu


# ------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(predictor, s0: torch.Tensor, actions: torch.Tensor,
            tracks, env_idx: np.ndarray, header: dict,
            horizon: int, geom_scale: float = 100.0) -> torch.Tensor:
    """Roll the predictor forward `horizon` steps from s0 under `actions`.

    Closed loop: the track geometry fed in at step t is gathered from the
    model's OWN predicted position and heading, not from the ground truth. The
    track polyline is given -- it is episode-constant context the model is
    entitled to -- but where the car thinks it is on that track is the model's
    problem. Feeding true geometry would leak the answer and make rollout error
    look far better than it is.

    Returns (N, horizon, OBS_SIZE) of predicted states.
    """
    s = s0.clone()
    out = []
    n = len(s)
    for t in range(horizon):
        a = torch.zeros(n, contract.NUM_ACTIONS, device=s.device)
        a[torch.arange(n), actions[:, t]] = 1.0

        sn = s.detach().cpu().numpy()
        d = contract.denorm(sn, header)
        geom = np.zeros((n, 44), dtype=np.float32)
        for e in np.unique(env_idx):
            m = env_idx == e
            geom[m] = tracks.local_window(int(e), d["near_point_idx"][m],
                                          d["px"][m], d["py"][m], d["ang"][m])
        g = torch.from_numpy(geom / geom_scale).to(s.device)

        s = project_state(s + predictor.delta(s, a, g))
        out.append(s.clone())
    return torch.stack(out, dim=1)


def rollout_error(pred: torch.Tensor, true: torch.Tensor,
                  dim_sd: torch.Tensor) -> np.ndarray:
    """Per-horizon normalized error. Each dimension is scaled by its own
    standard deviation so a 30-dim mean is not dominated by whichever
    dimension happens to have the largest units."""
    err = ((pred - true) / dim_sd) ** 2
    return err.mean(dim=(0, 2)).cpu().numpy()
