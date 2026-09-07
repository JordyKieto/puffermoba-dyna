"""Planning with the learned world model: exhaustive search vs Gumbel MCTS.

Both plan over the SAME model and the same horizon, so the comparison isolates
the search procedure rather than the model.

Why these two:

  Exhaustive to depth d. With |A| = 3 this is 3^d leaves -- 243 at d=5, 2187 at
  d=7. Batched on a GPU-shaped tensor that is nothing. If the tree is small
  enough to enumerate, MCTS is strictly extra machinery, and the honest thing
  is to check whether it buys anything before adopting it.

  Gumbel MuZero (Danihelka et al., ICLR 2022). Vanilla MCTS selects by visit
  counts, which has no policy-improvement guarantee at small simulation budgets
  -- exactly the regime worth running here. Gumbel top-k sampling plus
  sequential halving restores the guarantee at n = 2..16 simulations.

Both truncate at the horizon with a learned value, following the whole modern
line (MuZero, TD-MPC2, Dreamer): plan short, bootstrap the rest. Our own
measurement says the model is trustworthy to ~20 steps, so planning deeper than
that would be planning through error.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import contract, geometry, model as M

A = contract.NUM_ACTIONS


class WorldModelSim:
    """Batched simulator over the learned model: (states, actions) -> next states.

    Termination comes from a learned head, not from a threshold on the
    predicted state. The crash state is never observable -- c_step zeroes the
    whiskers and resets before computing observations -- so the corpus contains
    no state with a whisker below 0.05, and any threshold read off a predicted
    state is measuring the model's extrapolation rather than a crash. See
    racewm/train_termination.py.
    """

    def __init__(self, net, bank, x_mu, x_sd, y_mu, y_sd,
                 term_head=None, term_mu=None, term_sd=None):
        self.net, self.bank = net, bank
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd
        self.term_head = term_head
        self.term_mu = term_mu if term_mu is not None else x_mu
        self.term_sd = term_sd if term_sd is not None else x_sd

    @torch.no_grad()
    def step(self, s, slots, a_idx):
        """Returns (next_state, reward, p_terminate)."""
        n = len(s)
        a = torch.zeros(n, A, device=s.device)
        a[torch.arange(n, device=s.device), a_idx] = 1.0
        g = self.bank.local_window(s, slots)
        x = torch.cat([s, a, g], dim=1)

        delta = self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu
        s2 = M.project_state(s + delta)

        if self.term_head is not None:
            p_term = torch.sigmoid(self.term_head((x - self.term_mu) / self.term_sd))
        else:
            p_term = torch.zeros(n, device=s.device)

        # Exactly what the env pays: reward_yellow per forward sector
        # crossing, which score_norm accumulates. The old version subtracted a
        # hand-set 5.0 * p_term, dwarfing a max step reward of 0.2 and
        # paralysing planners into pure crash-avoidance. Termination now lives
        # where it belongs -- forfeited future return, carried by the critic
        # and applied as a survival weight on everything downstream.
        i = contract.INDEX
        reward = (s2[:, i["score_norm"]] - s[:, i["score_norm"]]) * 100.0
        return s2, reward, p_term


def load_sim(wm_path, term_path, bank, x_mu, x_sd, y_mu, y_sd, device="cpu"):
    """Assemble a simulator from a world-model and termination checkpoint."""
    import torch as _t
    from . import geometry as _geo
    from .train_termination import TerminationHead

    head = t_mu = t_sd = None
    if term_path:
        tk = _t.load(term_path, map_location=device, weights_only=False)
        tc = tk["config"]
        head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"]).to(device)
        head.load_state_dict(tk["model"])
        head.eval()
        for q in head.parameters():
            q.requires_grad_(False)
        extra = tk.get("extra") or {}
        t_mu = extra.get("mu", x_mu).to(device) if hasattr(extra.get("mu", x_mu), "to") else x_mu
        t_sd = extra.get("sd", x_sd).to(device) if hasattr(extra.get("sd", x_sd), "to") else x_sd
    return WorldModelSim(None, bank, x_mu, x_sd, y_mu, y_sd, head, t_mu, t_sd)


@torch.no_grad()
def plan_exhaustive(sim, s, slots, depth, value_fn=None, gamma=0.97):
    """Enumerate every action sequence to `depth`; return the best first action.

    Batched over B x A^depth: the whole tree is one wide tensor, so cost is
    depth forward passes, not A^depth of them.
    """
    B = len(s)
    n_leaves = A ** depth
    states = s.repeat_interleave(n_leaves, 0)
    sl = slots.repeat_interleave(n_leaves, 0)
    ret = torch.zeros(B * n_leaves, device=s.device)
    survive = torch.ones(B * n_leaves, device=s.device)

    for d in range(depth):
        block = A ** (depth - d - 1)
        acts = (torch.arange(n_leaves, device=s.device) // block) % A
        acts = acts.repeat(B)
        states, r, p_term = sim.step(states, sl, acts)
        # Reward past a likely crash is not collectable, so weight it by the
        # probability of still being alive to collect it.
        ret += (gamma ** d) * survive * r
        survive = survive * (1 - p_term)

    if value_fn is not None:
        ret += (gamma ** depth) * survive * value_fn(states, sl)

    ret = ret.view(B, A, n_leaves // A).max(dim=2).values     # best per first action
    return ret.argmax(dim=1), ret


@torch.no_grad()
def plan_beam(sim, s, slots, depth, beam=6, value_fn=None, gamma=0.97):
    """Beam search: expand every action, keep the best `beam` partials.

    Exhaustive enumeration is A^depth leaves -- 243 at depth 5, and each of the
    depth passes runs over B*A^depth states, which measured at 4.6ms per
    decision. Beam keeps the same depth for a fraction of the width: depth
    passes over B*beam*A states. With beam >= 4 and |A| = 3 it recovers
    essentially all of exhaustive's value here, because the value differences
    between sibling branches are small compared to the difference between
    turning and not turning.
    """
    B = len(s)
    dev = s.device
    states = s
    sl = slots
    ret = torch.zeros(B, device=dev)
    survive = torch.ones(B, device=dev)
    first = torch.full((B,), -1, dtype=torch.long, device=dev)
    width = 1

    for d in range(depth):
        n = B * width
        rep = states.repeat_interleave(A, 0)
        rep_sl = sl.repeat_interleave(A, 0) if d == 0 else sl.repeat_interleave(A, 0)
        acts = torch.arange(A, device=dev).repeat(n)
        nxt, r, p_term = sim.step(rep, rep_sl, acts)
        surv = survive.repeat_interleave(A, 0)
        cand_ret = ret.repeat_interleave(A, 0) + (gamma ** d) * surv * r
        cand_surv = surv * (1 - p_term)
        cand_first = (acts if d == 0 else first.repeat_interleave(A, 0))

        k = min(beam, width * A)
        score = cand_ret.view(B, width * A)
        if value_fn is not None and d == depth - 1:
            v = (gamma ** (d + 1)) * cand_surv * value_fn(nxt, rep_sl)
            score = score + v.view(B, width * A)
        top = score.topk(k, dim=1).indices                       # (B, k)

        flat = (torch.arange(B, device=dev)[:, None] * (width * A) + top).reshape(-1)
        states = nxt[flat]
        ret = cand_ret[flat]
        survive = cand_surv[flat]
        first = cand_first[flat]
        sl = slots.repeat_interleave(k, 0)
        width = k

    best = ret.view(B, width).argmax(dim=1)
    return first.view(B, width)[torch.arange(B, device=dev), best], ret


@torch.no_grad()
def plan_gumbel(sim, s, slots, depth, n_sims=16, value_fn=None, gamma=0.97,
                prior_logits=None, seed=None, c_visit=50.0, c_scale=1.0):
    """Gumbel action selection with sequential halving.

    Root actions are scored by Gumbel-perturbed logits plus their simulated
    value; the candidate set is halved each phase and the surviving actions get
    the freed simulations. With |A|=3 this reduces to a careful budget split,
    which is the point -- it shows what MCTS machinery is actually worth here.
    """
    B = len(s)
    dev = s.device
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    logits = (torch.zeros(B, A, device=dev) if prior_logits is None
              else prior_logits.to(dev))
    gumbel = -torch.log(-torch.log(
        torch.rand(B, A, generator=g).clamp_min(1e-20))).clamp_min(1e-20).to(dev)

    q = torch.zeros(B, A, device=dev)
    counts = torch.zeros(B, A, device=dev)

    candidates = torch.arange(A, device=dev).expand(B, A).clone()
    m = A
    phases = max(1, int(math.ceil(math.log2(A))))
    per_phase = max(1, n_sims // phases)

    while m > 1:
        sims_each = max(1, per_phase // m)
        for _ in range(sims_each):
            for c in range(m):
                a0 = candidates[:, c]
                s1, r, pt = sim.step(s, slots, a0)
                ret = r.clone()
                surv = 1 - pt
                st = s1
                for d in range(1, depth):
                    # Random continuation past the first action: with |A|=3 and
                    # a short horizon this is a cheap unbiased rollout, and it
                    # keeps the comparison against exhaustive honest.
                    ra = torch.randint(0, A, (B,), generator=g).to(dev)
                    st, rr, pt2 = sim.step(st, slots, ra)
                    ret += (gamma ** d) * surv * rr
                    surv = surv * (1 - pt2)
                if value_fn is not None:
                    ret += (gamma ** depth) * surv * value_fn(st, slots)
                idx = (torch.arange(B, device=dev), a0)
                counts[idx] += 1
                q[idx] += (ret - q[idx]) / counts[idx]

        m = max(1, m // 2)
        sigma = (c_visit + counts.max(dim=1, keepdim=True).values) * c_scale * q
        score = logits + gumbel + sigma
        score = score.scatter(1, candidates, score.gather(1, candidates))
        candidates = score.topk(m, dim=1).indices

    return candidates[:, 0], q


class GreedyPolicyValue:
    """Value head standing in for a learned critic: how much progress remains.

    Real experiments should swap in a trained critic. This exists so the search
    comparison can run before the Dreamer critic is available, and it is
    deliberately simple so nothing subtle hides in it.
    """

    def __init__(self, weight: float = 20.0):
        self.weight = weight

    def __call__(self, s, slots=None):
        i = contract.INDEX
        alive = ((s[:, i["flw_length"]] > 1e-3) | (s[:, i["frw_length"]] > 1e-3)).float()
        clearance = 0.5 * (s[:, i["flw_length"]] + s[:, i["frw_length"]])
        return self.weight * alive * clearance


class LearnedValue:
    """V(s) from racewm/train_critic.py, fitted by TD(0) on REAL transitions.

    Real data on purpose: a critic fitted inside the world model inherits the
    model's optimistic errors, and a planner then searches straight for them.
    Sees the track geometry as well as the state, which more than doubled the
    separation between healthy and about-to-crash states (0.165 -> 0.441).
    """

    def __init__(self, path, bank, device="cpu"):
        from .train_critic import Critic
        ck = torch.load(path, map_location=device, weights_only=False)
        ex = ck.get("extra") or {}
        cfg = ck["config"]
        self.use_geom = ex.get("use_geometry", cfg.get("use_geometry", True))
        self.net = Critic(cfg["d_in"], cfg["hidden"], cfg["layers"]).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.mu, self.sd = ex["mu"].to(device), ex["sd"].to(device)
        self.bank = bank

    @torch.no_grad()
    def __call__(self, s, slots=None):
        x = s
        if self.use_geom:
            if slots is None:
                raise ValueError("LearnedValue needs slots to gather geometry")
            x = torch.cat([s, self.bank.local_window(s, slots)], dim=1)
        return self.net((x - self.mu) / self.sd)
