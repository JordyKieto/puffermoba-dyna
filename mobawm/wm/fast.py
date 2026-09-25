"""Faster training step for the token world model.

Two pieces, both numerically the same objective as train.run_window/step_loss:

  VecLoss      step_loss without host syncs or Python loops over fields:
               per-kind sums use the fixed entity slices (heroes, creeps,
               neutrals, towers), boolean-index means become masked means,
               and the named parts come back as ONE tensor (names in .names).
               A part that the eager code omits when it is empty (got_hit of a
               kind with nobody alive, creep lane/team with no creep alive)
               is 0 here, which is what it contributes to the eager total.
  WindowLoss   featurize + K model steps + loss for a whole window, in one
               function, so torch.compile sees (and fuses) the whole forward
               and backward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from .. import shards as S
from .data import F, KIND, N_KIND, REG_FIELDS, REG_IDX, wave_respawns
from .train import EFFECT_OF, KIND_NAMES, featurize_window

KIND_SLICES = ((0, 10), (10, 110), (110, 182), (182, 206))
assert all((KIND[a:b] == k).all() for k, (a, b) in enumerate(KIND_SLICES))


class VecLoss:
    def __init__(self, stats, effect_weight, dev, gated):
        kind = torch.from_numpy(KIND).to(dev)
        self.gated = gated
        self.scale = stats.reg_scale.to(dev)[kind]                  # (206, nreg)
        self.reg_mask = stats.reg_mask.to(dev)[kind]                # (206, nreg) bool
        self.reg_idx = torch.tensor(REG_IDX, device=dev)
        rm = stats.reg_mask
        names, wmat = [], []
        for k, name in enumerate(KIND_NAMES):
            for j, fname in enumerate(REG_FIELDS):
                if rm[k, j]:
                    eff = EFFECT_OF[fname]
                    key = f"{name}/{eff}"
                    if key not in names:
                        names.append(key)
                        wmat.append(torch.zeros(N_KIND, len(REG_FIELDS)))
                    wmat[names.index(key)][k, j] = effect_weight.get(eff, 1.0)
        self.w_eff = torch.stack(wmat).to(dev)                      # (n_eff, 4, nreg)
        self.flip_kinds = [k for k in range(N_KIND) if stats.alive_mask[k]]
        self.w_death = effect_weight.get("deaths", 1.0)
        names += [f"{KIND_NAMES[k]}/deaths" for k in self.flip_kinds]
        names += [f"{n}/got_hit" for n in KIND_NAMES]
        names += ["creep/lane", "creep/team"]
        self.names = names

    @staticmethod
    def _per_kind(x):
        """(B, 206, ...) -> (4, ...) sums over batch and the entities of each kind."""
        return torch.stack([x[:, a:b].sum((0, 1)) for a, b in KIND_SLICES])

    def __call__(self, pred, e0, e1, tick, cidx):
        alive0, alive1 = e0[..., F["alive"]] > 0, e1[..., F["alive"]] > 0
        target = (e1[..., self.reg_idx] - e0[..., self.reg_idx]) / self.scale
        wc, _, wn = wave_respawns(tick, cidx)
        rule = torch.cat([torch.zeros_like(alive0[:, :10]), wc, wn.unsqueeze(-1).expand(-1, 72),
                          torch.zeros_like(alive0[:, 182:])], 1)
        m = ((alive0 & alive1 & ~rule).unsqueeze(-1) & self.reg_mask).float()
        if self.gated:
            changed = (target != 0).float()
            gate_err = Fn.binary_cross_entropy_with_logits(pred["change"], changed, reduction="none")
            err = Fn.huber_loss(pred["value"], target, reduction="none", delta=1.0)
            mc = m * changed
            per_field = (self._per_kind(gate_err * m) / self._per_kind(m).clamp(min=1)
                         + self._per_kind(err * mc) / self._per_kind(mc).clamp(min=1))
        else:
            err = Fn.huber_loss(pred["reg"], target, reduction="none", delta=1.0)
            per_field = self._per_kind(err * m) / self._per_kind(m).clamp(min=1)
        eff = (self.w_eff * per_field).sum((1, 2))
        B = e0.shape[0]
        flip = (alive0 != alive1).float()
        alive_bce = Fn.binary_cross_entropy_with_logits(pred["alive_flip"], flip, reduction="none")
        deaths = [self.w_death * alive_bce[:, a:b].mean() for a, b in (KIND_SLICES[k] for k in self.flip_kinds)]
        a1 = alive1.float()
        hit_bce = Fn.binary_cross_entropy_with_logits(pred["is_hit"], e1[..., F["is_hit"]], reduction="none")
        got_hit = self._per_kind(hit_bce * a1) / self._per_kind(a1).clamp(min=1)
        c = slice(10, 110)
        ca = a1[:, c]
        n_ca = ca.sum().clamp(min=1)
        lane_ce = Fn.cross_entropy(pred["lane"].flatten(0, 1), e1[:, c, F["lane"]].long().clamp(0, 5).flatten(),
                                   reduction="none").view(B, 100)
        team_ce = Fn.cross_entropy(pred["team"].flatten(0, 1), e1[:, c, F["team"]].long().clamp(0, 2).flatten(),
                                   reduction="none").view(B, 100)
        parts = torch.cat([eff, torch.stack(deaths), got_hit,
                           ((lane_ce * ca).sum() / n_ca).view(1), ((team_ce * ca).sum() / n_ca).view(1)])
        return parts.sum(), parts


class WindowLoss:
    """loss, parts = WindowLoss(...)(ent, act, tick, cidx): the training
    objective of train.run_window for one sampled batch."""

    def __init__(self, model, fz, loss_fn, burn, bf16):
        self.model, self.fz, self.loss_fn, self.burn, self.bf16 = model, fz, loss_fn, burn, bf16

    def __call__(self, ent, act, tick, cidx):
        feats = featurize_window(self.fz, ent, tick, cidx)
        B, K = act.shape[:2]
        h = torch.zeros(B, S.N_ENTITIES, self.model.d, device=ent.device)
        total, parts = 0.0, 0.0
        for t in range(K):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                pred, h = self.model.step(feats[t], act[:, t], h)
            pred = {k: v.float() for k, v in pred.items()}
            h = h.float()
            if t >= self.burn:
                l, p = self.loss_fn(pred, ent[:, t], ent[:, t + 1], tick[:, t], cidx[:, t])
                total, parts = total + l, parts + p
        n = K - self.burn
        return total / n, parts / n
