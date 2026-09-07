"""Export a torch MLP policy to the flat format race_eval reads.

Input normalization is written INTO the file. A policy handed unnormalized
inputs still emits plausible-looking actions, so a mismatch here does not
crash -- it just quietly produces a worse policy and a wrong benchmark.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract

MAGIC = 0x52504F31  # "RPO1"
VERSION = 1
ACT_SILU, ACT_RELU, ACT_TANH = 0, 1, 2


def export(model: nn.Module, mu, sd, path: str | Path,
           activation: int = ACT_SILU, obs_dim: int = contract.OBS_SIZE,
           act_dim: int = contract.NUM_ACTIONS) -> Path:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("no nn.Linear layers found")
    if linears[0].in_features != obs_dim:
        raise ValueError(f"first layer takes {linears[0].in_features}, expected {obs_dim}")
    if linears[-1].out_features != act_dim:
        raise ValueError(f"last layer emits {linears[-1].out_features}, expected {act_dim}")

    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sd = np.asarray(sd, dtype=np.float32).reshape(-1)
    if mu.shape != (obs_dim,) or sd.shape != (obs_dim,):
        raise ValueError("mu/sd must be obs_dim long")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, VERSION, len(linears), obs_dim,
                            act_dim, activation))
        f.write(mu.tobytes())
        f.write(np.maximum(sd, 1e-6).tobytes())
        for L in linears:
            w = L.weight.detach().cpu().numpy().astype(np.float32)   # [out, in]
            b = (L.bias.detach().cpu().numpy().astype(np.float32)
                 if L.bias is not None else np.zeros(L.out_features, np.float32))
            f.write(struct.pack("<2I", L.in_features, L.out_features))
            f.write(np.ascontiguousarray(w).tobytes())
            f.write(b.tobytes())
    return path


def verify(model: nn.Module, mu, sd, path: str | Path, n: int = 256,
           seed: int = 0) -> float:
    """Re-implement the C forward pass in numpy and compare argmax agreement.

    Catches layout mistakes (row/column major, missing bias, wrong activation)
    that a shape check cannot.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, contract.OBS_SIZE)).astype(np.float32)

    with torch.no_grad():
        xt = (torch.from_numpy(x) - torch.as_tensor(mu)) / torch.as_tensor(sd).clamp_min(1e-6)
        want = model(xt.float()).argmax(1).numpy()

    raw = Path(path).read_bytes()
    magic, ver, n_layers, obs_dim, act_dim, act_kind = struct.unpack_from("<6I", raw, 0)
    assert magic == MAGIC, "bad magic"
    off = 24
    fmu = np.frombuffer(raw, np.float32, obs_dim, off); off += 4 * obs_dim
    fsd = np.frombuffer(raw, np.float32, obs_dim, off); off += 4 * obs_dim

    h = (x - fmu) / fsd
    for li in range(n_layers):
        i_dim, o_dim = struct.unpack_from("<2I", raw, off); off += 8
        w = np.frombuffer(raw, np.float32, i_dim * o_dim, off).reshape(o_dim, i_dim)
        off += 4 * i_dim * o_dim
        b = np.frombuffer(raw, np.float32, o_dim, off); off += 4 * o_dim
        h = h @ w.T + b
        if li + 1 < n_layers:
            h = h / (1.0 + np.exp(-h)) if act_kind == ACT_SILU else \
                np.maximum(h, 0) if act_kind == ACT_RELU else np.tanh(h)
    got = h.argmax(1)
    return float((got == want).mean())
