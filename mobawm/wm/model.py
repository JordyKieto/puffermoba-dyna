"""Token world model for Ocean MOBA. See docs/moba/TOKEN_WM_DESIGN.md.

One step: tokenize s_t (unified + class-specific parts), add each hero's
action and every entity's GRU memory, run 2 attention blocks with a learned
distance / team bias, update the per-entity GRU, and decode typed deltas.
Output layers are zero-initialized, so an untrained model predicts "nothing
changes" (the persistence baseline).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import shards as S
from .data import KIND, N_KIND, REG_FIELDS

N_ENT = S.N_ENTITIES


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Fourier(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi)

    def forward(self, p):                       # p in [0, 1]
        a = p.unsqueeze(-1) * self.freq
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class Tokenizer(nn.Module):
    """token = unified part (shared by every entity) + class-specific part."""

    def __init__(self, d):
        super().__init__()
        # unified
        self.entity = nn.Parameter(torch.zeros(d))
        self.kind_emb = nn.Embedding(N_KIND, d)
        self.team_emb = nn.Embedding(3, d)
        self.shared = mlp(9, d, d)
        self.fourier = Fourier(8)
        self.pos = mlp(2 * 2 * 8, d, d)
        self.terrain = mlp(81, d, d)
        # class-specific
        # class id: heroes 0-9 (hero class x team), creep 10, neutral 11, tower 12
        cls = torch.tensor([*range(10), *[10] * 100, *[11] * 72, *[12] * 24])
        self.register_buffer("cls", cls)
        self.class_emb = nn.Embedding(13, d)
        self.hero = mlp(7, d, d)
        self.creep = mlp(1, d, d)
        self.lane_emb = nn.Embedding(6, d)
        self.tower = mlp(1, d, d)
        self.register_buffer("kind", torch.from_numpy(KIND))
        self.glob = mlp(14, d, d)

    def forward(self, f):
        x = (self.entity + self.kind_emb(self.kind) + self.team_emb(f["team"])
             + self.shared(f["shared"]) + self.pos(self.fourier(f["pos"])) + self.terrain(f["patch"])
             + self.class_emb(self.cls))
        extra = torch.zeros_like(x)
        extra[..., 0:10, :] = self.hero(f["hero"])
        extra[..., 10:110, :] = self.creep(f["creep"]) + self.lane_emb(f["lane"])
        extra[..., 182:206, :] = self.tower(f["tower"])
        g = self.glob(f["glob"]).unsqueeze(-2)
        return torch.cat([x + extra, g], -2)    # (..., 207, d); last token is global


class ActionEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleList([nn.Embedding(n, d) for n in S.ACTION_SIZES])

    def forward(self, a):                       # (..., 10, 6) int
        return sum(h(a[..., i]) for i, h in enumerate(self.heads))


# The 15 abilities (hero class x slot), from docs/moba/GAME_MECHANICS.md.
# Columns: mana cost, cooldown, radius, needs target, damage, heal, stun, slow,
# moves self, moves target, self buff, area, creep/neutral targets only.
ABILITY_DESC = torch.tensor([
    [100, 15, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # support Q: hook
    [100, 50, 5, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],   # support W: area heal
    [75, 60, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],    # support E: stun
    [100, 40, 3, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],   # assassin Q: clear (creeps/neutrals)
    [150, 60, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],   # assassin W: blink strike
    [100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # assassin E: speed boost
    [200, 70, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # burst Q: nuke
    [100, 40, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],   # burst W: blast
    [75, 50, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],    # burst E: area stun
    [5, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],      # tank Q: burn (no cooldown)
    [100, 70, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   # tank W: self heal
    [50, 40, 4, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],    # tank E: engage
    [25, 40, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],    # carry Q: knockback + slow
    [150, 40, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],   # carry W: crippling shot
    [100, 40, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],   # carry E: blast
], dtype=torch.float32) / torch.tensor([200, 100, 5] + [1] * 10, dtype=torch.float32)


def cast_context(f, a, desc):
    """Per hero and ability (B,10,3), upstream step_players' own cast rule, as far
    as the state can say: this tick's regen and cooldown tick happen first, a
    stunned hero does nothing, and at most one of Q > W > E fires. Not
    captured: 'the hit lands' and class-specific conditions (e.g. assassin Q
    needs a creep/neutral target), and targets moving earlier in the tick."""
    raw = f["hero_raw"]                                           # mana, max_mana, q, w, e, stun
    mana = torch.minimum(raw[..., 0] + 2, raw[..., 1]).unsqueeze(-1)
    cd = raw[..., 2:5]
    free = (raw[..., 5] <= 1).unsqueeze(-1).float()               # stun expires this tick
    cost, needs = desc[..., 0] * 200, desc[..., 3] > 0
    pressed = a[..., 3:6].float()
    ready = (cd <= 1).float()
    mana_ok = (mana >= cost).float()
    tgt = torch.where(a[..., 2:3] == 2, f["tgt_ht"].unsqueeze(-1), f["tgt_any"].unsqueeze(-1)).float()
    target = tgt.expand_as(cd)
    able = free * pressed * ready * mana_ok * torch.where(needs, target, torch.ones_like(target))
    # area damage/stun abilities need an enemy within their radius (2, 3 or 4 cells)
    radius = (desc[..., 2] * 5).round()
    area_hit = (desc[..., 11] > 0) & ((desc[..., 4] > 0) | (desc[..., 6] > 0))
    near = f["near"].float()                                      # (B,10,3): within 2, 3, 4
    in_r = torch.where(radius <= 2, near[..., 0:1], torch.where(radius <= 3, near[..., 1:2], near[..., 2:3]))
    able = able * torch.where(area_hit, in_r, torch.ones_like(in_r))
    # creep/neutral-only abilities (assassin Q) need the automatic target to be one
    able = able * torch.where(desc[..., 12] > 0, f["tgt_minion"].unsqueeze(-1).float(), torch.ones_like(able))
    q, w, e = able.unbind(-1)
    fires = torch.stack([q, w * (1 - q), e * (1 - q) * (1 - w)], -1)    # priority Q > W > E
    return dict(pressed=pressed, ready=ready, cd=cd, mana_ok=mana_ok, mana=mana.expand_as(cd),
                target=target, fires=fires)


class Block(nn.Module):
    """Pre-LN attention + MLP with a learned per-head distance and team bias."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = mlp(d, 4 * d, d)
        self.dist_w = nn.Parameter(torch.zeros(n_heads))         # softplus(0)=0.69 per 16 cells
        self.team_bias = nn.Parameter(torch.zeros(n_heads, 3))   # same / enemy / involves neutral

    def forward(self, x, bias_parts):
        dist, rel, key_ok = bias_parts           # (B,N,N), (B,N,N,3) one-hot, (B,N) bool
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = (-Fn.softplus(self.dist_w).view(1, -1, 1, 1) * dist.unsqueeze(1)
                + torch.einsum("bnmr,hr->bhnm", rel, self.team_bias))   # one-hot: cheap backward
        bias = bias.masked_fill(~key_ok[:, None, None, :], -1e4)  # finite; the global key is always valid
        att = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(att.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.ln2(x))


class TokenWM(nn.Module):
    def __init__(self, d=128, n_heads=4, n_layers=2, gate=False, gate_fields=None, cast_features=False):
        super().__init__()
        self.d = d
        # cast_features: per hero and Q/W/E, whether the pressed ability can fire
        # this tick under the game's own cast rule (cast_context), added to the hero token
        self.cast_features = cast_features
        if cast_features:
            self.cast_in = mlp(21, d, d)
            self.register_buffer("cast_desc", ABILITY_DESC[(torch.arange(10) % 5).unsqueeze(-1) * 3 + torch.arange(3)])
        # gate=True: copy by default. Per delta field, a "does this field change
        # this tick" logit; the value head is only used where it says yes.
        # gate_fields limits it to those fields (None = all); the others are
        # predicted directly, as without the gate.
        self.gated = gate
        self.gate_fields = list(gate_fields) if gate_fields else list(REG_FIELDS)
        self.tok = Tokenizer(d)
        self.act = ActionEmbed(d)
        self.mem_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.gru = nn.GRUCell(d, d)
        self.reg = nn.Linear(d, len(REG_FIELDS))
        self.change = nn.Linear(d, len(REG_FIELDS)) if gate else None
        self.alive_flip = nn.Linear(d, 1)
        self.is_hit = nn.Linear(d, 1)
        self.lane = nn.Linear(d, 6)
        self.team = nn.Linear(d, 3)
        for head in (self.reg, self.alive_flip, self.lane, self.team):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.alive_flip.bias, -6.0)    # start at "no flip"
        if gate:
            nn.init.zeros_(self.change.weight)
            nn.init.zeros_(self.change.bias)
            # per-(entity, field) prior logit of changing, set from data by set_change_prior;
            # starting at "nothing changes" everywhere left hero movement (60% of steps) stuck
            self.register_buffer("change_prior", torch.full((N_ENT, len(REG_FIELDS)), -4.0))
            self.register_buffer("gate_mask", torch.tensor([f in self.gate_fields for f in REG_FIELDS]))

    def set_change_prior(self, change_rate):
        """Open each gate at its field's base rate. The value head is still zero,
        so the untrained model is still exactly persistence."""
        if self.gated:
            p = change_rate.clamp(1e-4, 1 - 1e-4).to(self.change_prior.device)
            self.change_prior.copy_(torch.log(p / (1 - p))[torch.from_numpy(KIND).to(p.device)])

    def bias_parts(self, f):
        xy = f["xy"]                                      # (B,206,2) cells
        B = xy.shape[0]
        dist = torch.zeros(B, N_ENT + 1, N_ENT + 1, device=xy.device)
        dist[:, :N_ENT, :N_ENT] = torch.cdist(xy, xy) / 16.0
        team = torch.cat([f["team"], torch.full((B, 1), 2, device=xy.device, dtype=torch.long)], 1)
        same = team.unsqueeze(2) == team.unsqueeze(1)
        neutral = (team.unsqueeze(2) == 2) | (team.unsqueeze(1) == 2)
        rel = Fn.one_hot(torch.where(neutral, 2, torch.where(same, 0, 1)), 3).float()
        key_ok = torch.cat([f["alive"], torch.ones(B, 1, dtype=torch.bool, device=xy.device)], 1)
        return dist, rel, key_ok

    def step(self, f, actions, h):
        """f: features of s_t (batch B), actions (B,10,6), h (B,206,d) -> preds, new h."""
        x = self.tok(f)
        a = torch.zeros_like(x)
        a[:, :10] = self.act(actions)
        if self.cast_features:
            c = cast_context(f, actions, self.cast_desc)
            ctx = torch.stack([c["pressed"], c["ready"], c["cd"] / 100, c["mana_ok"], c["mana"] / 1000,
                               c["target"], c["fires"]], -1)                        # (B,10,3,7)
            a[:, :10] = a[:, :10] + self.cast_in(ctx.flatten(-2))
        x = x + a
        x[:, :N_ENT] = x[:, :N_ENT] + self.mem_in(h)
        parts = self.bias_parts(f)
        for blk in self.blocks:
            x = blk(x, parts)
        e = self.ln(x[:, :N_ENT])
        B = e.shape[0]
        h = self.gru(e.reshape(-1, self.d), h.reshape(-1, self.d)).view(B, N_ENT, self.d)
        c = slice(10, 110)
        lane_cur = Fn.one_hot(f["lane"], 6).float()
        team_cur = Fn.one_hot(f["team"][:, c], 3).float()
        value = self.reg(h)
        out = {}
        if self.gated:
            change = self.change(h) + self.change_prior
            out = dict(change=change, value=value, gate_mask=self.gate_mask)
            closed = self.gate_mask & (change <= 0)                          # hard gate at p = 0.5
            value = torch.where(closed, torch.zeros_like(value), value)
        return dict(
            **out,
            reg=value,
            alive_flip=self.alive_flip(h).squeeze(-1),
            is_hit=self.is_hit(h).squeeze(-1),
            lane=self.lane(h[:, c]) + 5 * lane_cur,
            team=self.team(h[:, c]) + 5 * team_cur,
        ), h
