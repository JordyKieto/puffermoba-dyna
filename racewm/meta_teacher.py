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
import hashlib
import pathlib
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
    ap.add_argument("--warm-steps", type=int, default=0,
                    help="student updates taken WITHOUT a graph before the "
                         "differentiable window. Truncated backprop: the "
                         "student can become competent over many steps while "
                         "second-order memory stays O(inner-steps). Pure "
                         "first-order (FOMAML) is not an option here -- "
                         "severing the graph gives EXACTLY zero gradient to "
                         "M_T, so the second-order path is the only signal.")
    ap.add_argument("--inner-steps", type=int, default=1,
                    help="differentiable student updates (the meta-gradient window)")
    ap.add_argument("--inner-lr", type=float, default=0.1)
    ap.add_argument("--h-inner", type=int, default=5)
    ap.add_argument("--h-outer", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--meta-steps", type=int, default=200)
    ap.add_argument("--meta-lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--normalize-inner-grad", action="store_true",
                    help="rescale the inner gradient to unit norm before the "
                         "student step. The teacher can otherwise win by "
                         "inflating ||grad J_T||, which scales how far theta' "
                         "travels, instead of shaping WHERE it travels -- "
                         "measured as J_T rising 22x while J_R stayed flat.")
    ap.add_argument("--jt-penalty", type=float, default=0.0,
                    help="penalty on |J_T| in the meta-objective; a blunt guard "
                         "on the degenerate mode")
    ap.add_argument("--jt-coef", type=float, default=0.0,
                    help="weight on J_T in the meta-objective, so the teacher is "
                         "rewarded for the student doing well in BOTH worlds "
                         "rather than only in the reference")
    ap.add_argument("--align-coef", type=float, default=0.0,
                    help="penalty on (J_T - J_R)^2. Pressure toward the two "
                         "worlds AGREEING about how good the student is. This "
                         "attacks the failure directly: unconstrained, J_T ran "
                         "to 2.99 and then to -19885 while J_R stayed flat, "
                         "because nothing tied the teacher's reward scale to "
                         "anything real. Agreement keeps the teacher free to "
                         "change its dynamics while stopping it from simply "
                         "rescaling its own reward.")
    ap.add_argument("--reset-student", action="store_true",
                    help="resample the student init each meta-step, so the "
                         "teacher must teach a distribution of students rather "
                         "than overfit one initialization")
    ap.add_argument("--share-encoder", type=int, default=0,
                    help="freeze the first N Linear layers of M_T, so teacher and "
                         "reference share a representation and differ only in the "
                         "dynamics head. A trust region in REPRESENTATION space "
                         "rather than in parameter norm or step size: M_T may "
                         "invent any physics that is a function of M_R's latent, "
                         "but cannot redefine what a state means. Motivated by "
                         "the collapse at 42%% parameter distance, where J_E fell "
                         "to 0.076 (below random) while J_R only fell to 0.166 -- "
                         "the signature of a student whose features stopped "
                         "meaning anything outside M_T. Note this RESTRICTS the "
                         "unrestricted-teacher hypothesis: it forbids exactly the "
                         "worlds that represent the task differently.")
    ap.add_argument("--freeze-teacher", action="store_true",
                    help="CONTROL ARM: run everything identically -- same warm-up, "
                         "same differentiable window, same student distribution, "
                         "same evaluation schedule -- but never update M_T. Any "
                         "movement in the real-env number is then pure "
                         "student-training variance, which is the noise band the "
                         "meta-optimized arm has to beat.")
    ap.add_argument("--probe", type=int, default=0,
                    help="diagnostic: run N student updates and report real-env "
                         "competence at checkpoints, to find where the student "
                         "clears a random policy")
    ap.add_argument("--verify", action="store_true",
                    help="check a non-zero meta-gradient reaches the teacher")
    ap.add_argument("--online-every", type=int, default=0,
                    help="audit the student in the REAL env every N meta-steps. "
                         "J_R is measured inside a frozen neural examiner, so a "
                         "teacher that exploits M_R's errors looks excellent "
                         "until the policy touches the actual game. This is the "
                         "only check that catches that.")
    ap.add_argument("--online-episodes", type=int, default=40)
    ap.add_argument("--bank-size", type=int, default=0,
                    help="Use a FIXED bank of N pre-warmed students instead of "
                         "re-warming one inside every meta-step. The warm-up is "
                         "create_graph=False, so it contributes exactly zero to "
                         "the meta-gradient -- warming it in M_R instead of M_T "
                         "therefore changes nothing the outer gradient measures, "
                         "while (a) paying the warm-up once instead of every "
                         "meta-step and (b) making the student distribution "
                         "teacher-independent, so arms become paired.")
    ap.add_argument("--bank-file", default="data/student_bank",
                    help="cache path; the student regime is hashed into the "
                         "filename so a changed regime cannot silently reuse a "
                         "stale bank, and identical regimes share one file")
    ap.add_argument("--out", default="data/meta_teacher.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env, ref, teacher, norms, meta = build_worlds(
        args.corpus, args.wm, args.termination, args.calibration,
        args.batch, 500, args.seed)
    T_, b_ = meta["temperature"], meta["bias"]
    bank, term = env.bank, env.term

    if args.share_encoder:
        lins = [m for m in teacher.net if isinstance(m, nn.Linear)]
        for m in lins[:args.share_encoder]:
            for q in m.parameters():
                q.requires_grad_(False)
        frozen = sum(q.numel() for m in lins[:args.share_encoder]
                     for q in m.parameters())
        print(f"shared encoder: first {args.share_encoder} of {len(lins)} Linear "
              f"layers frozen ({frozen:,} params shared with M_R)")


    policy = Policy(args.hidden, 2)
    theta = {k: v.detach().clone().requires_grad_(True)
             for k, v in policy.named_parameters()}
    meta_opt = torch.optim.Adam(
        [q for q in teacher.parameters() if q.requires_grad], lr=args.meta_lr)

    def sample_states():
        env.reset()
        return env.obs.detach().clone(), env.slots.clone()

    def warm(p, world, k):
        """Detached student updates in `world`.

        create_graph=False here, so this contributes EXACTLY ZERO to the
        meta-gradient: the teacher receives no credit for the warm-up either
        way. That is what makes the student bank legitimate -- warming in M_R
        rather than M_T does not change what the outer gradient measures, it
        only fixes the distribution of students the teacher is scored on.
        """
        for _ in range(k):
            s0, slots = sample_states()
            J = soft_rollout(policy, p, world, term, bank, s0, slots,
                             args.h_inner, norms, term_T=T_, term_b=b_)
            g = torch.autograd.grad(J, list(p.values()), create_graph=False)
            if args.normalize_inner_grad:
                gn2 = torch.sqrt(sum((x ** 2).sum() for x in g)).clamp_min(1e-8)
                g = [x / gn2 for x in g]
            p = {k2: (v + args.inner_lr * x).detach().requires_grad_(True)
                 for (k2, v), x in zip(p.items(), g)}
        return p

    def inner_then_outer(theta_in):
        """K differentiable student updates in M_T, scored in M_R.

        Memory grows linearly in K: every inner step keeps its graph alive so
        the mixed second derivative can be taken through all of them.
        """
        p = {k: v for k, v in theta_in.items()}

        # Warm-up: detached, so memory is flat in the number of steps. This is
        # what lets the student reach competence before the teacher is asked to
        # improve it -- E17 failed because a 3-step student scored below random
        # in the real env, leaving nothing to teach.
        p = warm(p, teacher, 0 if args.bank_size else args.warm_steps)

        s0, slots = sample_states()
        for _ in range(args.inner_steps):
            J_T = soft_rollout(policy, p, teacher, term, bank, s0, slots,
                               args.h_inner, norms, term_T=T_, term_b=b_)
            # create_graph keeps dtheta'/dM_T alive. An optimizer step here
            # would detach it and the meta-gradient would be exactly zero.
            grads = torch.autograd.grad(J_T, list(p.values()), create_graph=True)
            if args.normalize_inner_grad:
                # Direction, not magnitude. Keeps the norm in the graph so the
                # teacher still shapes the direction, but removes the trivial
                # win of simply making the step bigger.
                gn2 = torch.sqrt(sum((g ** 2).sum() for g in grads)).clamp_min(1e-8)
                grads = [g / gn2 for g in grads]
            p = {k: v + args.inner_lr * g for (k, v), g in zip(p.items(), grads)}
        s0r, slotsr = sample_states()
        J_R = soft_rollout(policy, p, ref, term, bank, s0r, slotsr,
                           args.h_outer, norms, term_T=T_, term_b=b_)
        return J_T, J_R, p

    from . import export_policy as _ep
    import subprocess as _sp
    _man = dataset.load_manifest(args.corpus)

    def audit_online(p_theta, tag):
        """Full behavioural readout in the REAL game, not just return.

        Return alone hides how a policy is failing. race_eval already reports
        laps, crash rate and its causes, track coverage, control smoothness and
        safety margin -- discarding them was throwing away the diagnostic half
        of every audit.
        """
        net = Policy(args.hidden, 2)
        with torch.no_grad():
            for k, v in net.named_parameters():
                v.copy_(p_theta[k].detach())
        rpo = f"/tmp/meta_{tag}.rpo"
        _ep.export(net.net, norms[4].numpy(), norms[5].numpy(), rpo)
        out = {}
        for name, key in (("hold", "holdout"), ("train", "train")):
            sp = _man["splits"][key]
            r = _sp.run(["./build/race_eval","--bots","policy","--policy",rpo,
                "--tracks",str(sp["n_tracks"]),"--track-start",str(sp["track_start"]),
                "--episodes","40","--threads","6"],
                capture_output=True, text=True).stdout
            rows = [l.split() for l in r.splitlines() if l.strip().startswith("policy")]
            # race_eval prints one `policy` row per metric group, in order:
            # OUTCOME / FAILURE / PROGRESS / CONTROL / SAFETY
            d = {}
            if len(rows) >= 5:
                d = dict(ret=float(rows[0][2]), length=float(rows[0][3]),
                         surv=float(rows[0][4]),
                         crash=float(rows[1][1]), wall=float(rows[1][2]),
                         steps2crash=float(rows[1][4]),
                         laps=float(rows[2][1]), coverage=float(rows[2][5]),
                         idle=float(rows[3][1]), reversals=float(rows[3][4]),
                         dist=float(rows[4][1]), minwhisk=float(rows[4][3]))
            elif rows:
                d = dict(ret=float(rows[0][2]))
            out[name] = d
        return out

    if args.probe:
        # How many student updates does it take to clear a random policy in the
        # REAL environment? Everything downstream is meaningless below that line.
        print("student competence vs inner-update count (real env)\n")
        print(f"  reference: random 0.177, straight 0.221, whisker 2.64, PPO 10.8\n")
        print(f"{'updates':>9}{'J_R':>10}{'REAL hold':>12}{'REAL train':>12}")
        p = {k: v.detach().clone().requires_grad_(True)
             for k, v in policy.named_parameters()}
        for n in range(1, args.probe + 1):
            s0, slots = sample_states()
            J = soft_rollout(policy, p, teacher, term, bank, s0, slots,
                             args.h_inner, norms, term_T=T_, term_b=b_)
            g = torch.autograd.grad(J, list(p.values()))
            gn2 = torch.sqrt(sum((x ** 2).sum() for x in g)).clamp_min(1e-8)
            p = {k: (v + args.inner_lr * x / gn2).detach().requires_grad_(True)
                 for (k, v), x in zip(p.items(), g)}
            if n in (1, 5, 10, 25, 50, 100, 200, 400, args.probe):
                s0r, sr = sample_states()
                with torch.no_grad():
                    jr = float(soft_rollout(policy, p, ref, term, bank, s0r, sr,
                                            args.h_outer, norms, term_T=T_, term_b=b_))
                o = audit_online(p, f"probe{n}")
                print(f"{n:>9}{jr:>10.4f}{o['hold']:>12.3f}{o['train']:>12.3f}",
                      flush=True)
        return

    if args.verify:
        print("VERIFY: does a meta-gradient reach the teacher at all?\n")
        J_T, J_R, _ = inner_then_outer(theta)
        g = torch.autograd.grad(J_R, [q for q in teacher.parameters()
                                       if q.requires_grad],
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
        g2 = torch.autograd.grad(J_R2, [q for q in teacher.parameters()
                                        if q.requires_grad], allow_unused=True)
        n2 = sum(float((x ** 2).sum()) for x in g2 if x is not None) ** 0.5
        print(f"  control, graph severed: {n2:.3e}  (must be 0)")
        print(f"\n  {'PASS' if norm > 0 and n2 == 0 else 'FAIL'}: the second-order "
              f"path is the only thing carrying signal to M_T")
        return

    trainable = [q for q in teacher.parameters() if q.requires_grad]
    print(f"M_R frozen = {args.wm};  M_T initialized AS M_R "
          f"({sum(q.numel() for q in trainable):,} trainable of "
          f"{sum(q.numel() for q in teacher.parameters()):,})")
    print(f"inner: {args.inner_steps} step(s), lr {args.inner_lr}, H={args.h_inner}"
          f"   outer: H={args.h_outer}   meta-lr {args.meta_lr}\n")
    print(f"{'step':>6}{'J_T':>10}{'J_R':>10}{'|dJR/dMT|':>12}{'MT-MR drift':>14}{'sec':>7}")

    from . import export_policy
    import subprocess
    man = dataset.load_manifest(args.corpus)

    ref_flat = torch.cat([p.detach().flatten() for p in ref.parameters()])

    student_bank = None
    if args.bank_size:
        # The regime that defines a bank. Hashed into the filename so that
        # every arm with the same student regime shares one byte-identical
        # bank, and a changed regime can never silently reuse a stale one.
        regime = (f"{args.wm}|{args.hidden}|{args.warm_steps}|{args.h_inner}"
                  f"|{args.inner_lr}|{int(args.normalize_inner_grad)}"
                  f"|{args.batch}|{args.bank_size}|{args.seed}")
        tag = hashlib.sha256(regime.encode()).hexdigest()[:12]
        bf = pathlib.Path(f"{args.bank_file}_{tag}.pt")
        if bf.exists():
            student_bank = torch.load(bf, weights_only=True)
            print(f"student bank: loaded {len(student_bank)} students from {bf}")
        else:
            tb = time.time()
            student_bank = []
            for i in range(args.bank_size):
                torch.manual_seed(args.seed * 100_003 + i)
                env.g.manual_seed(args.seed * 100_003 + i)
                fresh = Policy(args.hidden, 2)
                q = {k: v.detach().clone().requires_grad_(True)
                     for k, v in fresh.named_parameters()}
                q = warm(q, ref, args.warm_steps)   # warmed in M_R, not M_T
                student_bank.append({k: v.detach().clone()
                                     for k, v in q.items()})
            bf.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student_bank, bf)
            print(f"student bank: built {args.bank_size} students warmed "
                  f"{args.warm_steps} steps in M_R ({time.time()-tb:.0f}s) -> {bf}")

    hist, t0 = [], time.time()
    for step in range(args.meta_steps):
        if student_bank is not None:
            # Common random numbers: meta-step t sees the same student and the
            # same initial states in every arm, so arms differ only by the
            # teacher update and can be compared pairwise.
            env.g.manual_seed(args.seed * 7_919 + step)
            th = {k: v.clone().requires_grad_(True)
                  for k, v in student_bank[step % len(student_bank)].items()}
        else:
            th = theta
            if args.reset_student:
                fresh = Policy(args.hidden, 2)
                th = {k: v.detach().clone().requires_grad_(True)
                      for k, v in fresh.named_parameters()}
        J_T, J_R, p_out = inner_then_outer(th)
        meta_opt.zero_grad(set_to_none=True)
        # Maximize the student's return in BOTH worlds, with pressure toward
        # the two worlds agreeing. The teacher may still invent any dynamics it
        # likes -- it just cannot win by inflating or deflating its own reward
        # scale relative to the reference.
        meta_loss = (-(J_R + args.jt_coef * J_T)
                     + args.align_coef * (J_T - J_R) ** 2
                     + args.jt_penalty * J_T.abs())
        meta_loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            [q for q in teacher.parameters() if q.requires_grad], 10.0))
        if not args.freeze_teacher:
            meta_opt.step()

        if step % 10 == 0 or step == args.meta_steps - 1:
            with torch.no_grad():
                t_flat = torch.cat([p.flatten() for p in teacher.parameters()])
                drift = float((t_flat - ref_flat).norm() / ref_flat.norm())
            online = {}
            if args.online_every and (step % args.online_every == 0
                                      or step == args.meta_steps - 1):
                online = audit_online(p_out, f"s{step}")
            hist.append(dict(step=step, J_T=float(J_T), J_R=float(J_R),
                             grad=gn, drift=drift, online=online))
            if online and online.get("hold", {}).get("laps") is not None:
                h = online["hold"]
                extra = (f"  J_E {h['ret']:.2f} laps {h['laps']:.2f} "
                         f"crash {h['crash']:.0f}% cover {h['coverage']:.2f} "
                         f"idle {h['idle']:.0f}% revrs {h['reversals']:.2f}")
            elif online:
                extra = f"  J_E {online['hold'].get('ret', float('nan')):.3f}"
            else:
                extra = ""
            extra += f"  |JT-JR| {abs(float(J_T)-float(J_R)):.4f}"
            print(f"{step:>6}{float(J_T):>10.4f}{float(J_R):>10.4f}"
                  f"{gn:>12.3e}{drift:>14.5f}{time.time()-t0:>7.0f}{extra}", flush=True)

    Path(args.out).write_text(json.dumps(hist, indent=2))
    print(f"\nJ_R: {hist[0]['J_R']:.4f} -> {hist[-1]['J_R']:.4f}"
          f"   teacher drifted {hist[-1]['drift']*100:.3f}% from M_R")


if __name__ == "__main__":
    main()
