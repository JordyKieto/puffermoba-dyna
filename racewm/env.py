"""ctypes wrapper over the real Race env, batched.

For agents that cannot be exported to a .rpo file -- planners, anything that
needs the world model at inference. Policies that CAN be exported should still
go through race_eval: it is the same env, in C, without an interpreter in the
loop.

Observations are a numpy view straight onto the C buffer, so a step costs no
copy.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from . import contract

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        root = Path(__file__).resolve().parent.parent
        for name in ("libracecapi.dylib", "libracecapi.so"):
            p = root / "build" / name
            if p.exists():
                _LIB = ctypes.CDLL(str(p))
                break
        else:
            raise FileNotFoundError("build it first: tools/build.sh capi")
        L = _LIB
        L.race_pool_create.restype = ctypes.c_void_p
        # n, seed | track_start..bezier_resolution (10 ints) | 6 floats | method
        L.race_pool_create.argtypes = [ctypes.c_int, ctypes.c_uint] + \
            [ctypes.c_int] * 10 + [ctypes.c_float] * 6 + [ctypes.c_int]
        for fn, rt in (("race_pool_obs", ctypes.POINTER(ctypes.c_float)),
                       ("race_pool_rewards", ctypes.POINTER(ctypes.c_float)),
                       ("race_pool_terminated", ctypes.POINTER(ctypes.c_int)),
                       ("race_pool_truncated", ctypes.POINTER(ctypes.c_int)),
                       ("race_pool_episode_id", ctypes.POINTER(ctypes.c_int)),
                       ("race_pool_env_idx", ctypes.POINTER(ctypes.c_int))):
            getattr(L, fn).restype = rt
            getattr(L, fn).argtypes = [ctypes.c_void_p]
        L.race_pool_reset.argtypes = [ctypes.c_void_p]
        L.race_pool_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        L.race_pool_free.argtypes = [ctypes.c_void_p]
        L.race_pool_total_points.argtypes = [ctypes.c_void_p]
        L.race_pool_total_points.restype = ctypes.c_int
        L.race_pool_track.argtypes = [ctypes.c_void_p, ctypes.c_int] + \
            [ctypes.POINTER(ctypes.c_float)] * 3
    return _LIB


class RacePool:
    def __init__(self, n, seed=42, track_start=112, n_tracks=16, frameskip=4,
                 max_episode_steps=2000, width=1080, height=720, track_width=75,
                 num_radial_sectors=16, num_points=16, bezier_resolution=4,
                 turn_pi_frac=40.0, maxv=5.0, w_ang=0.777,
                 max_whisker_length=100.0, reward_yellow=0.2,
                 corner_thresh=0.5, method=2):
        L = _lib()
        self.n = n
        self._p = L.race_pool_create(
            n, seed, track_start, n_tracks, frameskip, max_episode_steps,
            width, height, track_width, num_radial_sectors, num_points,
            bezier_resolution, turn_pi_frac, maxv, w_ang, max_whisker_length,
            reward_yellow, corner_thresh, method)
        self._L = L
        D = contract.OBS_SIZE
        self.obs = np.ctypeslib.as_array(L.race_pool_obs(self._p), (n, D))
        self.rewards = np.ctypeslib.as_array(L.race_pool_rewards(self._p), (n,))
        self.terminated = np.ctypeslib.as_array(L.race_pool_terminated(self._p), (n,))
        self.truncated = np.ctypeslib.as_array(L.race_pool_truncated(self._p), (n,))
        self.episode_id = np.ctypeslib.as_array(L.race_pool_episode_id(self._p), (n,))
        self.env_idx = np.ctypeslib.as_array(L.race_pool_env_idx(self._p), (n,)).copy()
        self._act = (ctypes.c_int * n)()

    def reset(self):
        self._L.race_pool_reset(self._p)
        return self.obs

    def step(self, actions):
        a = np.ascontiguousarray(actions, dtype=np.int32)
        ctypes.memmove(self._act, a.ctypes.data, a.nbytes)
        self._L.race_pool_step(self._p, self._act)
        return self.obs, self.rewards, self.terminated, self.truncated

    def tracks(self):
        """Per-env polylines, keyed the way shards.Tracks is."""
        P = self._L.race_pool_total_points(self._p)
        buf = lambda: (ctypes.c_float * (2 * P))()
        centerline, inner, outer = {}, {}, {}
        for i in range(self.n):
            e = int(self.env_idx[i])
            if e in centerline:
                continue
            c, a, b = buf(), buf(), buf()
            self._L.race_pool_track(self._p, i, c, a, b)
            # as_array ignores the shape argument for an already-sized ctypes
            # array, so reshape explicitly rather than trusting it.
            centerline[e] = np.ctypeslib.as_array(c).reshape(P, 2).copy()
            inner[e] = np.ctypeslib.as_array(a).reshape(P, 2).copy()
            outer[e] = np.ctypeslib.as_array(b).reshape(P, 2).copy()
        from .shards import Tracks
        return Tracks(centerline=centerline, inner=inner, outer=outer)

    def close(self):
        if getattr(self, "_p", None):
            self._L.race_pool_free(self._p)
            self._p = None

    def __del__(self):
        self.close()
