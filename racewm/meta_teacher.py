"""Differentiable bilevel search for a teacher world.

    M_T* = argmax_{M_T}  J_{M_R}( student_update(M_T) )

**M_R** is the reference world: parameters frozen, but differentiable with
respect to its inputs, so gradients flow through it to whatever produced those
inputs. **M_T** is the mutable teacher whose parameters we optimize. It is
under no obligation to be realistic — it is scored only by how good a policy it
teaches, as judged by M_R.

The gradient we need is

    dJ_R/dM_T  =  (dJ_R/dtheta')  (dtheta'/dM_T),
    theta' = theta + alpha * grad_theta J_T(theta, M_T)

so dtheta'/dM_T is a mixed second derivative and the inner update must be built
with `create_graph=True` and applied functionally — an in-place optimizer step
severs the graph and silently yields a zero meta-gradient.

**Actions are marginalized, not sampled.** With three actions the expected
reward and expected next state can be computed exactly at every step. That
removes the score-function estimator entirely: no REINFORCE variance, no
Gumbel-softmax bias, and a clean gradient to differentiate twice through. It
costs 3 world-model evaluations per step, which is the right trade at short
horizons. The state carried forward is the action-marginal, which blurs the
dynamics slightly — acceptable at H<=10 and the reason to start short.

Termination enters as a differentiable survival weight rather than a sampled
event, for the same reason.

Start short and verify the teacher actually moves before worrying about long
unrolls: `--verify` checks that a non-zero meta-gradient reaches M_T at all,
which is the failure mode that makes everything downstream meaningless.

    python3 -m racewm.meta_teacher --verify
    python3 -m racewm.meta_teacher --meta-steps 200
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from . import contract, dataset, dream, geometry, model as M, shards

A = contract.NUM_ACTIONS


class Policy(nn.Module):
    """Student. Kept small: the inner update is differentiated twice."""

    def __init__(self, hidden=128, layers=2):
        super().__init__()
        mods, d = [], contract.OBS_SIZE
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, A))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def soft_rollout(policy, params, wm, term, bank, s0, slots, H, norms,
                 gamma=0.99, term_T=1.0, term_b=0.0):
    """Differentiable expected return of `params` inside world `wm`.

    Every action is marginalized rather than sampled, so this is an exact
    expectation under the policy and is twice-differentiable.
    """
    x_mu, x_sd, y_mu, y_sd, obs_mu, obs_sd, t_mu, t_sd = norms
    s = s0
    surv = torch.ones(len(s0))
    total = torch.zeros(len(s0))
    i = contract.INDEX
    ar = torch.arange(len(s0))

    for t in range(H):
        logits = functional_call(policy, params, ((s - obs_mu) / obs_sd,))
        pi = torch.softmax(logits, dim=-1)                      # (B, A)

        g = bank.local_window(s, slots)                          # depends on s
        nxt, rew, pterm = [], [], []
        for a in range(A):
            oh = torch.zeros(len(s), A)
            oh[ar, a] = 1.0
            x = torch.cat([s, oh, g], dim=1)
            delta = wm((x - x_mu) / x_sd) * y_sd + y_mu
            s_a = M.project_state(s + delta)
            nxt.append(s_a)
            rew.append((s_a[:, i["score_norm"]] - s[:, i["score_norm"]]) * 100.0)
            pterm.append(torch.sigmoid(term((x - t_mu) / t_sd) / term_T + term_b))

        R = torch.stack(rew, dim=1)                              # (B, A)
        S = torch.stack(nxt, dim=1)                              # (B, A, D)
        P = torch.stack(pterm, dim=1)                            # (B, A)

        total = total + (gamma ** t) * surv * (pi * R).sum(1)
        surv = surv * (1.0 - (pi * P).sum(1))                    # differentiable
        s = (pi.unsqueeze(-1) * S).sum(1)                        # action-marginal
    return total.mean()


def build_worlds(corpus, wm_path, term_path, calib_path, envs, horizon, seed):
    env, meta = dream.build(corpus, wm_path, term_path, calib_path,
                            400_000, envs, horizon, "cpu", seed)
    ref = env.wm                                    # frozen params
    for p in ref.parameters():
        p.requires_grad_(False)
    teacher = copy.deepcopy(env.wm)                 # mutable, starts AS M_R
    for p in teacher.parameters():
        p.requires_grad_(True)
    for p in env.term.parameters():
        p.requires_grad_(False)
    norms = (env.x_mu, env.x_sd, env.y_mu, env.y_sd,
             meta["obs_mu"], meta["obs_sd"], env.term_mu, env.term_sd)
    return env, ref, teacher, norms, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus_big")
    ap.add_argument("--wm", default="checkpoints/wmbig_k20/best.pt")
    ap.add_argument("--termination", default="checkpoints/term_big/best.pt")
    ap.add_argument("--calibration", default="checkpoints/term_big/calibration.pt")
    ap.add_argument("--inner-steps", type=int, default=1,
                    help="differentiable student updates per meta step")
    ap.add_argument("--inner-lr", type=float, default=0.1)
    ap.add_argument("--h-inner", type=int, default=5)
    ap.add_argument("--h-outer", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--meta-steps", type=int, default=200)
    ap.add_argument("--meta-lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--verify", action="store_true",
                    help="check a non-zero meta-gradient reaches the teacher")
    ap.add_argument("--out", default="data/meta_teacher.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env, ref, teacher, norms, meta = build_worlds(
        args.corpus, args.wm, args.termination, args.calibration,
        args.batch, 500, args.seed)
    T_, b_ = meta["temperature"], meta["bias"]
    bank, term = env.bank, env.term

    policy = Policy(args.hidden, 2)
    theta = {k: v.detach().clone().requires_grad_(True)
             for k, v in policy.named_parameters()}
    meta_opt = torch.optim.Adam(teacher.parameters(), lr=args.meta_lr)

    def sample_states():
        env.reset()
        return env.obs.detach().clone(), env.slots.clone()

    def inner_then_outer(theta_in):
        """One differentiable student update in M_T, scored in M_R."""
        s0, slots = sample_states()
        p = {k: v for k, v in theta_in.items()}
        for _ in range(args.inner_steps):
            J_T = soft_rollout(policy, p, teacher, term, bank, s0, slots,
                               args.h_inner, norms, term_T=T_, term_b=b_)
            # create_graph keeps dtheta'/dM_T alive. An optimizer step here
            # would detach it and the meta-gradient would be exactly zero.
            grads = torch.autograd.grad(J_T, list(p.values()), create_graph=True)
            p = {k: v + args.inner_lr * g for (k, v), g in zip(p.items(), grads)}
        s0r, slotsr = sample_states()
        J_R = soft_rollout(policy, p, ref, term, bank, s0r, slotsr,
                           args.h_outer, norms, term_T=T_, term_b=b_)
        return J_T, J_R, p

    if args.verify:
        print("VERIFY: does a meta-gradient reach the teacher at all?\n")
        J_T, J_R, _ = inner_then_outer(theta)
        g = torch.autograd.grad(J_R, list(teacher.parameters()),
                                retain_graph=True, allow_unused=True)
        norm = sum(float((x ** 2).sum()) for x in g if x is not None) ** 0.5
        n_none = sum(1 for x in g if x is None)
        print(f"  J_T (teacher world)   {float(J_T):.5f}")
        print(f"  J_R (reference world) {float(J_R):.5f}")
        print(f"  ||dJ_R/dM_T||         {norm:.3e}   "
              f"({n_none} of {len(g)} tensors unused)")
        # Control: without create_graph the path is severed and this must be 0.
        s0, slots = sample_states()
        p = {k: v for k, v in theta.items()}
        J_T2 = soft_rollout(policy, p, teacher, term, bank, s0, slots,
                            args.h_inner, norms, term_T=T_, term_b=b_)
        gr = torch.autograd.grad(J_T2, list(p.values()), create_graph=False)
        p2 = {k: (v + args.inner_lr * gg).detach().requires_grad_(True)
              for (k, v), gg in zip(p.items(), gr)}
        s0r, slotsr = sample_states()
        J_R2 = soft_rollout(policy, p2, ref, term, bank, s0r, slotsr,
                            args.h_outer, norms, term_T=T_, term_b=b_)
        g2 = torch.autograd.grad(J_R2, list(teacher.parameters()),
                                 allow_unused=True)
        n2 = sum(float((x ** 2).sum()) for x in g2 if x is not None) ** 0.5
        print(f"  control, graph severed: {n2:.3e}  (must be 0)")
        print(f"\n  {'PASS' if norm > 0 and n2 == 0 else 'FAIL'}: the second-order "
              f"path is the only thing carrying signal to M_T")
        return

    print(f"M_R frozen = {args.wm};  M_T initialized AS M_R "
          f"({sum(p.numel() for p in teacher.parameters()):,} params, trainable)")
    print(f"inner: {args.inner_steps} step(s), lr {args.inner_lr}, H={args.h_inner}"
          f"   outer: H={args.h_outer}   meta-lr {args.meta_lr}\n")
    print(f"{'step':>6}{'J_T':>10}{'J_R':>10}{'|dJR/dMT|':>12}{'MT-MR drift':>14}{'sec':>7}")

    ref_flat = torch.cat([p.detach().flatten() for p in ref.parameters()])
    hist, t0 = [], time.time()
    for step in range(args.meta_steps):
        J_T, J_R, _ = inner_then_outer(theta)
        meta_opt.zero_grad(set_to_none=True)
        (-J_R).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(teacher.parameters(), 10.0))
        meta_opt.step()

        if step % 10 == 0 or step == args.meta_steps - 1:
            with torch.no_grad():
                t_flat = torch.cat([p.flatten() for p in teacher.parameters()])
                drift = float((t_flat - ref_flat).norm() / ref_flat.norm())
            hist.append(dict(step=step, J_T=float(J_T), J_R=float(J_R),
                             grad=gn, drift=drift))
            print(f"{step:>6}{float(J_T):>10.4f}{float(J_R):>10.4f}"
                  f"{gn:>12.3e}{drift:>14.5f}{time.time()-t0:>7.0f}", flush=True)

    Path(args.out).write_text(json.dumps(hist, indent=2))
    print(f"\nJ_R: {hist[0]['J_R']:.4f} -> {hist[-1]['J_R']:.4f}"
          f"   teacher drifted {hist[-1]['drift']*100:.3f}% from M_R")


if __name__ == "__main__":
    main()
