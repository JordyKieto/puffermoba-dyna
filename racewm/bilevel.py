"""Bilevel search for a TEACHER world, evaluated by a frozen REFERENCE world.

    M_T* = argmax_{M_T}  J_{M_R}( PPO(M_T) )

Two neural worlds. **M_R** is trained conventionally to approximate the target
environment and then frozen — a cheap neural examiner standing in for reality.
**M_T** is a mutable world a policy is trained inside. M_T is under no
obligation to be realistic: it is scored only by how well the policy it
produces performs in M_R. If teaching a car to fly makes it drive better, that
is a valid solution.

**Why derivative-free.** The outer gradient dJ_{M_R}(PPO(M_T))/dM_T requires
differentiating through an entire PPO run. Truncated backprop through training
is possible but memory-bound and noisy, and it constrains the inner loop to be
short and differentiable. CEM over a low-dimensional teacher needs none of
that, parallelizes trivially, and — the reason it is the right smoke test —
produces an *interpretable* answer: we can read off exactly what distortion of
physics made a better teacher.

**Teacher parameterization (32 dims).** Perturbing all 843k dynamics weights is
hopeless for a few generations of search, so M_T is M_R composed with a learned
modulation:

    delta_scale (30)  per-state-dimension gain on the predicted transition
    term_temp   (1)   sharpness of the crash probability
    term_bias   (1)   crash frequency

The identity setting (`delta_scale = 1`, calibrated temperature and bias) makes
M_T exactly M_R, so the search starts from "teach in an accurate world" and any
improvement is evidence that inaccuracy teaches better. A gain of 1.4 on the
heading dimensions means "make steering feel stronger than it is"; a raised
term_bias means "make the world deadlier than it is".

**Three levels of evaluation, deliberately.** PPO optimizes inside M_T. The
outer search scores in M_R. And the *real environment* audits whether gains
measured by the neural examiner transfer at all — because a teacher that
exploits M_R's errors would look excellent right up to the point it is tested.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract, dream, export_policy, model as M
from .dream_ppo import ActorCritic, gae

N_PARAMS = contract.OBS_SIZE + 2


def identity_theta(cal_T, cal_b):
    return np.concatenate([np.ones(contract.OBS_SIZE), [cal_T], [cal_b]])


class Teacher:
    """M_R composed with a learned modulation. theta = identity gives M_R."""

    def __init__(self, base_env, theta):
        self.e = base_env
        self.theta = theta
        self.scale = torch.tensor(theta[:contract.OBS_SIZE], dtype=torch.float32)
        self.T = float(max(theta[contract.OBS_SIZE], 0.05))
        self.b = float(theta[contract.OBS_SIZE + 1])
        self.n = base_env.n
        self.imagined_steps = 0

    def reset(self):
        return self.e.reset()

    @torch.no_grad()
    def step(self, actions):
        e = self.e
        a = torch.zeros(e.n, contract.NUM_ACTIONS)
        a[torch.arange(e.n), actions] = 1.0
        g = e.bank.local_window(e.obs, e.slots)
        x = torch.cat([e.obs, a, g], dim=1)

        delta = e.wm((x - e.x_mu) / e.x_sd) * e.y_sd + e.y_mu
        delta = delta * self.scale                      # <-- the teacher's physics
        nxt = M.project_state(e.obs + delta)

        i = contract.INDEX
        reward = (nxt[:, i["score_norm"]] - e.obs[:, i["score_norm"]]) * 100.0
        p = torch.sigmoid(e.term((x - e.term_mu) / e.term_sd) / self.T + self.b)
        crashed = torch.rand(e.n, generator=e.g) < p

        e.t += 1
        trunc = e.t >= e.horizon
        done = crashed | trunc
        e.obs = nxt
        if bool(done.any()):
            k = int(done.sum())
            o, s = e._sample_real(k)
            e.obs = e.obs.clone(); e.obs[done] = o
            e.slots = e.slots.clone(); e.slots[done] = s
            e.t = torch.where(done, torch.zeros_like(e.t), e.t)
        self.imagined_steps += e.n
        return e.obs, reward, crashed, trunc


def train_ppo(env, obs_mu, obs_sd, iters, envs, rollout, seed=0, lr=3e-4,
              hidden=256, layers=2):
    torch.manual_seed(seed)
    ac = ActorCritic(hidden, layers)
    opt = torch.optim.AdamW(ac.parameters(), lr=lr, eps=1e-5)
    obs = env.reset()
    for _ in range(iters):
        O = torch.zeros(rollout, envs, contract.OBS_SIZE)
        A = torch.zeros(rollout, envs, dtype=torch.long)
        LP = torch.zeros(rollout, envs); R = torch.zeros(rollout, envs)
        C = torch.zeros(rollout, envs, dtype=torch.bool)
        TR = torch.zeros(rollout, envs, dtype=torch.bool)
        V = torch.zeros(rollout, envs)
        with torch.no_grad():
            for t in range(rollout):
                lg, v = ac((obs - obs_mu) / obs_sd)
                d = torch.distributions.Categorical(logits=lg)
                a = d.sample()
                O[t], A[t], LP[t], V[t] = obs, a, d.log_prob(a), v
                obs, r, c, tr = env.step(a)
                R[t], C[t], TR[t] = r, c, tr
            _, lastv = ac((obs - obs_mu) / obs_sd)
        adv, ret = gae(R, V, C, TR, lastv)
        bo = ((O - obs_mu) / obs_sd).reshape(-1, contract.OBS_SIZE)
        ba, blp = A.reshape(-1), LP.reshape(-1)
        badv = ((adv - adv.mean()) / adv.std().clamp_min(1e-8)).reshape(-1)
        bret = ret.reshape(-1)
        n = len(bo); mb = n // 4
        for _ in range(3):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i+mb]
                lg, v = ac(bo[idx])
                d = torch.distributions.Categorical(logits=lg)
                ratio = (d.log_prob(ba[idx]) - blp[idx]).exp()
                pg = -torch.min(ratio*badv[idx],
                                torch.clamp(ratio,0.8,1.2)*badv[idx]).mean()
                loss = pg + 0.5*((v-bret[idx])**2).mean() - 0.01*d.entropy().mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
    return ac


@torch.no_grad()
def score_in_reference(ac, ref_env, obs_mu, obs_sd, steps=600):
    """J_{M_R}: mean per-episode return of the policy inside the frozen M_R."""
    o = ref_env.reset()
    ret = torch.zeros(ref_env.n); ln = torch.zeros(ref_env.n); done_ret = []
    for _ in range(steps):
        a = ac.actor((o - obs_mu) / obs_sd).argmax(1)
        o, r, c, tr = ref_env.step(a)
        ret += r; ln += 1
        fin = c | tr
        if bool(fin.any()):
            for k in torch.nonzero(fin).flatten().tolist():
                done_ret.append(float(ret[k])); ret[k] = 0; ln[k] = 0
    return float(np.mean(done_ret)) if done_ret else 0.0


def audit_real(ac, obs_mu, obs_sd, tag, man, episodes=60):
    """The authoritative check: does an M_R-measured gain exist in reality?"""
    rpo = f"/tmp/bilevel_{tag}.rpo"
    export_policy.export(ac.actor, obs_mu.numpy(), obs_sd.numpy(), rpo)
    out = {}
    for name, (st, n) in (("hold", (man["splits"]["holdout"]["track_start"],
                                    man["splits"]["holdout"]["n_tracks"])),
                          ("train", (man["splits"]["train"]["track_start"],
                                     man["splits"]["train"]["n_tracks"]))):
        r = subprocess.run(["./build/race_eval","--bots","policy","--policy",rpo,
                            "--tracks",str(n),"--track-start",str(st),
                            "--episodes",str(episodes),"--threads","10"],
                           capture_output=True, text=True).stdout
        row = [l for l in r.splitlines() if l.strip().startswith("policy")]
        out[name] = float(row[0].split()[2]) if row else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus_big")
    ap.add_argument("--wm", default="checkpoints/wmbig_k20/best.pt")
    ap.add_argument("--termination", default="checkpoints/term_big/best.pt")
    ap.add_argument("--calibration", default="checkpoints/term_big/calibration.pt")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--inner-iters", type=int, default=40)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--out", default="data/bilevel.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from . import dataset
    man = dataset.load_manifest(args.corpus)
    rng = np.random.default_rng(args.seed)

    def make_env(seed):
        e, meta = dream.build(args.corpus, args.wm, args.termination,
                              args.calibration, 400_000, args.envs,
                              args.horizon, "cpu", seed)
        return e, meta

    base, meta = make_env(args.seed)
    obs_mu, obs_sd = meta["obs_mu"], meta["obs_sd"]
    ref_env, _ = make_env(args.seed + 999)          # frozen examiner M_R
    theta0 = identity_theta(meta["temperature"], meta["bias"])

    print(f"M_R = {args.wm} (frozen).  M_T = M_R + 32-dim modulation.")
    print(f"identity theta reproduces M_R exactly, so generation 0 is the "
          f"'teach in an accurate world' baseline.")
    print(f"CEM: {args.generations} gens x {args.population} pop, "
          f"inner PPO {args.inner_iters} iters x {args.envs} envs\n")

    mu, sigma = theta0.copy(), np.full(N_PARAMS, args.sigma)
    sigma[contract.OBS_SIZE:] = [0.15, 0.4]
    history = []

    print(f"{'gen':>4}{'cand':>6}{'J_MR':>9}{'note':>28}")
    t0 = time.time()
    for gen in range(args.generations + 1):
        if gen == 0:
            cands = [theta0.copy()]                 # identity = M_R itself
        else:
            cands = [mu + sigma * rng.standard_normal(N_PARAMS)
                     for _ in range(args.population)]
        scored = []
        for ci, th in enumerate(cands):
            env, _ = make_env(args.seed + 1000 * gen + ci)
            teacher = Teacher(env, th)
            ac = train_ppo(teacher, obs_mu, obs_sd, args.inner_iters,
                           args.envs, args.rollout, seed=args.seed)
            j = score_in_reference(ac, ref_env, obs_mu, obs_sd)
            scored.append((j, th, ac))
            tag = "IDENTITY (= M_R)" if gen == 0 else ""
            print(f"{gen:>4}{ci:>6}{j:>9.3f}{tag:>28}", flush=True)

        scored.sort(key=lambda x: -x[0])
        best_j, best_th, best_ac = scored[0]
        if gen == 0:
            baseline_j, baseline_ac = best_j, best_ac
        else:
            elite = np.stack([s[1] for s in scored[:args.elite]])
            mu, sigma = elite.mean(0), elite.std(0) + 1e-3

        real = audit_real(best_ac, obs_mu, obs_sd, f"g{gen}", man)
        history.append(dict(gen=gen, J_MR=best_j, real=real,
                            theta=best_th.tolist()))
        print(f"     best J_MR={best_j:.3f}  |  REAL hold={real['hold']:.3f} "
              f"train={real['train']:.3f}   ({time.time()-t0:.0f}s)\n", flush=True)

    print(f"{'':=<64}")
    print(f"baseline (M_T = M_R):        J_MR {baseline_j:.3f}")
    print(f"best learned teacher:        J_MR {history[-1]['J_MR']:.3f}  "
          f"({100*(history[-1]['J_MR']-baseline_j)/max(baseline_j,1e-9):+.1f}%)")
    print(f"\nreal-env audit (the number that decides it):")
    for h in history:
        print(f"  gen {h['gen']}  hold {h['real']['hold']:.3f}  "
              f"train {h['real']['train']:.3f}")
    th = np.array(history[-1]["theta"])
    print(f"\nwhat the best teacher changed (1.0 = accurate):")
    d = th[:contract.OBS_SIZE]
    for i in np.argsort(-np.abs(d - 1.0))[:6]:
        print(f"  {contract.NAMES[i]:<22} delta gain {d[i]:+.3f}")
    print(f"  termination temp {th[-2]:.3f} (M_R {meta['temperature']:.3f}), "
          f"bias {th[-1]:+.3f} (M_R {meta['bias']:+.3f})")
    Path(args.out).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
