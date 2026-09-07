"""A vectorized imagined environment backed by the world model.

Presents the same interface as the real env so PPO cannot tell the difference,
which is the point: the identical training code should run against either, and
the comparison is then about the model rather than the plumbing.

Three decisions worth stating, all of them consequences of earlier experiments:

**Resets are anchored to REAL states.** Every episode starts from a state drawn
from a buffer of real observations, never from an imagined one. This is
Dreamer's trick and it is what keeps rollouts near the data manifold. Starting
from imagined states compounds error into the initial condition.

**Termination is sampled from the CALIBRATED head.** The raw head over-predicts
terminations 3.05x on holdout (it is trained with pos_weight 65 against a 1.5%
positive rate), so uncalibrated it ends imagined episodes about three times too
early and teaches an over-cautious policy. After temperature+bias calibration
the predicted count matches actual to 1.01x. Sampling rather than thresholding
gives the right expected episode length and injects the stochasticity a
deterministic model otherwise lacks.

**Rollouts are capped by `horizon`.** Longer imagined episodes give PPO more
signal and give it more room to exploit the model. That trade-off is the
experiment, not a detail: E8/E10 measured this model being exploited by a
planner searching only 5 steps deep.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import contract, geometry, model as M


class DreamEnv:
    def __init__(self, wm, term_head, bank, real_obs, real_slots,
                 x_mu, x_sd, y_mu, y_sd, term_mu, term_sd,
                 n_envs=2048, horizon=200, temperature=1.0, bias=0.0,
                 device="cpu", seed=0):
        self.wm, self.term, self.bank = wm, term_head, bank
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd
        self.term_mu, self.term_sd = term_mu, term_sd
        self.T, self.b = temperature, bias
        self.real_obs, self.real_slots = real_obs, real_slots
        self.n, self.horizon, self.device = n_envs, horizon, device
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.imagined_steps = 0

    def _sample_real(self, k):
        idx = torch.randint(0, len(self.real_obs), (k,), generator=self.g)
        return self.real_obs[idx].to(self.device), self.real_slots[idx].to(self.device)

    def reset(self):
        self.obs, self.slots = self._sample_real(self.n)
        self.t = torch.zeros(self.n, dtype=torch.long, device=self.device)
        return self.obs

    @torch.no_grad()
    def step(self, actions):
        a = torch.zeros(self.n, contract.NUM_ACTIONS, device=self.device)
        a[torch.arange(self.n, device=self.device), actions] = 1.0
        g = self.bank.local_window(self.obs, self.slots)
        x = torch.cat([self.obs, a, g], dim=1)

        delta = self.wm((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu
        nxt = M.project_state(self.obs + delta)

        i = contract.INDEX
        # Reward is exactly what the env pays: reward_yellow per forward sector
        # crossing, which score_norm accumulates. Audited on holdout: MAE
        # 0.0108 against a 0.2 unit, and 1.02x total-reward bias.
        reward = (nxt[:, i["score_norm"]] - self.obs[:, i["score_norm"]]) * 100.0

        logit = self.term((x - self.term_mu) / self.term_sd) / self.T + self.b
        p_term = torch.sigmoid(logit)
        crashed = (torch.rand(self.n, generator=self.g).to(self.device) < p_term)

        self.t += 1
        truncated = self.t >= self.horizon
        done = crashed | truncated

        self.obs = nxt
        if done.any():
            k = int(done.sum())
            o, s = self._sample_real(k)
            self.obs = self.obs.clone()
            self.obs[done] = o
            self.slots = self.slots.clone()
            self.slots[done] = s
            self.t = torch.where(done, torch.zeros_like(self.t), self.t)

        self.imagined_steps += self.n
        # crashed and truncated are returned SEPARATELY and must stay that way.
        # A time limit is not the end of the world: the value function has to
        # bootstrap through a truncation and stop only at a real termination
        # (Pardo et al. 2018). Collapsing them teaches the policy that reality
        # ends at `horizon`, which makes it myopic in proportion to how often
        # the limit fires -- every 15 steps at horizon 15 against every 200 at
        # horizon 200, so it silently penalizes short-horizon arms most.
        return self.obs, reward, crashed, truncated


def build(corpus, wm_path, term_path, calib_path=None, real_budget=200_000,
          n_envs=2048, horizon=200, device="cpu", seed=0):
    """Assemble a DreamEnv from checkpoints, returning it plus normalization."""
    from . import dataset, shards
    from .train_termination import TerminationHead

    real = dataset.load_split(corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, max_rows=real_budget, seed=seed)
    x = torch.from_numpy(dataset.wm_inputs(real))
    y = torch.from_numpy(dataset.wm_targets(real))
    x_mu, x_sd = x.mean(0, keepdim=True).to(device), x.std(0, keepdim=True).clamp_min(1e-6).to(device)
    y_mu, y_sd = y.mean(0, keepdim=True).to(device), y.std(0, keepdim=True).clamp_min(1e-8).to(device)
    obs_t = torch.from_numpy(real.obs)
    obs_mu = obs_t.mean(0, keepdim=True).to(device)
    obs_sd = obs_t.std(0, keepdim=True).clamp_min(1e-6).to(device)

    ck = torch.load(wm_path, map_location=device, weights_only=False)
    c = ck["config"]
    wm = M.DeltaMLP(c["d_in"], contract.OBS_SIZE, c["hidden"], c["layers"]).to(device)
    wm.load_state_dict(ck["model"]); wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)

    tk = torch.load(term_path, map_location=device, weights_only=False)
    tc = tk["config"]
    head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"]).to(device)
    head.load_state_dict(tk["model"]); head.eval()
    for q in head.parameters():
        q.requires_grad_(False)
    ex = tk.get("extra") or {}
    t_mu, t_sd = ex["mu"].to(device), ex["sd"].to(device)

    T, b = 1.0, 0.0
    if calib_path and Path(calib_path).exists():
        cal = torch.load(calib_path, map_location=device, weights_only=False)
        T, b = cal["temperature"], cal["bias"]

    bank = geometry.TrackBank(shards.load_tracks(f"{corpus}/train.tracks.bin"),
                              real.header, device)
    env = DreamEnv(wm, head, bank, obs_t, bank.slots(real.env_idx),
                   x_mu, x_sd, y_mu, y_sd, t_mu, t_sd,
                   n_envs, horizon, T, b, device, seed)
    return env, dict(obs_mu=obs_mu, obs_sd=obs_sd, n_real=len(real),
                     calibrated=(T != 1.0 or b != 0.0), temperature=T, bias=b)
