"""Token world model for Ocean MOBA. See docs/moba/TOKEN_WM_DESIGN.md.

One step: tokenize s_t (unified + class-specific parts), add each hero's
action and every entity's GRU memory, run 2 attention blocks with a learned
distance / team bias, update the per-entity GRU, and decode typed deltas.
Output layers are zero-initialized, so an untrained model predicts "nothing
changes" (the persistence baseline).
"""
from __future__ import annotations

import math

import numpy as np
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
                + sum(rel[..., r].unsqueeze(1) * self.team_bias[:, r].view(1, -1, 1, 1) for r in range(3)))
        # ^ ally/enemy/neutral bias as 3 masked adds, not einsum("bnmr,hr->bhnm"): that einsum's weight gradient
        #   is a 3 x (B*207*207) x heads matmul, which ran at ~40 ms per call and was 88% of an update's GPU time
        bias = bias.masked_fill(~key_ok[:, None, None, :], -1e4)  # finite; the global key is always valid
        att = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(att.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.ln2(x))


class TokenWM(nn.Module):
    def __init__(self, d=128, n_heads=4, n_layers=2, gate=False, gate_fields=None, cast_features=False,
                 attack_mode=None, predict_targets=False, target_dist=False, list_state=False, direct_damage=False, grid_cnn=False):
        super().__init__()
        self.d = d
        # attack_mode: "derived" / "real": the attack graph (data.attack_inputs: what each unit attacks,
        # who attacks it) added to every unit token through a zero-initialised projection, so a
        # checkpoint trained without it loads and starts out unchanged. The featurizer must be run
        # with data.ATTACK_MODE set to the same mode.
        self.attack_mode = attack_mode
        # predict_targets: who each unit attacks next tick (the recorded target_pid), a pointer over the
        # 206 units + "nobody", so a dream state carries a predicted attack graph (replaces rule R9).
        self.predict_targets = predict_targets
        if predict_targets:
            self.tq, self.tk = nn.Linear(d, 64), nn.Linear(d, 64)
            self.t_none = nn.Parameter(torch.zeros(64))
            self.t_none_bias = nn.Parameter(torch.tensor(4.0))           # start at "attacks nobody"
        # target_dist: a learned bias on the target scores per (attacker kind, enemy/ally/neutral, L1
        # distance 0..16+ cells), zero-initialised: real targets sit 1-3 cells away, which a dot product
        # of memories can't resolve (a frozen-trunk probe: right unit 58% -> 82% with it)
        self.target_dist = target_dist and predict_targets
        if self.target_dist:
            self.t_dist = nn.Parameter(torch.zeros(N_KIND, 3, 17))
            self.register_buffer("t_kind", torch.from_numpy(KIND).long(), persistent=False)
        # list_state: the stored target lists (data.LIST_STATE). Inputs: data.aggro_inputs through a zero-initialised
        # projection (a checkpoint without it starts unchanged). Output: next-tick list membership per (owner, member),
        # a dot product of memories plus a learned table over (owner kind, |dx|, |dy|, currently a member, tick phase,
        # member kind), initialised to "keep the list as it is".
        self.list_state = list_state
        if list_state:
            from .data import AGGRO_DIM, MEMBERS, OWNERS
            self.agg_in = nn.Linear(AGGRO_DIM, d)
            nn.init.zeros_(self.agg_in.weight)
            nn.init.zeros_(self.agg_in.bias)
            self.lq, self.lk = nn.Linear(d, 32), nn.Linear(d, 32)
            nn.init.zeros_(self.lq.weight)                  # q starts at 0 (output = the table); k random so q gets a gradient
            nn.init.zeros_(self.lq.bias)
            tab = torch.full((2, 10, 10, 2, 15, N_KIND), -2.0)
            tab[:, :, :, 1] = 2.0                           # start at "keep the list", +-2 so Adam at lr 1e-3 can move it
            self.l_tab = nn.Parameter(tab.flatten())
            self.register_buffer("l_own", torch.from_numpy(OWNERS).long(), persistent=False)
            self.register_buffer("l_mem", torch.from_numpy(MEMBERS).long(), persistent=False)
            self.register_buffer("l_mkind", torch.from_numpy(KIND[MEMBERS]).long(), persistent=False)
            self.register_buffer("l_okind", torch.from_numpy((KIND[OWNERS] == 3).astype(np.int64)), persistent=False)
        # direct_damage (needs list_state): incoming_damage (aggro input 7, HP) feeds the health prediction directly,
        # not only through the trunk: value[health] += dd_val[kind] * (-incoming / health scale) and the health gate
        # logit += dd_gate[kind] * (incoming > 0). Both start at 0 (model unchanged); dd_val = 1 means "health drops
        # by exactly the incoming damage". Learned, no rule.
        self.direct_damage = direct_damage and list_state
        if self.direct_damage:
            self.dd_val = nn.Parameter(torch.zeros(N_KIND))
            self.dd_gate = nn.Parameter(torch.zeros(N_KIND))
            self.register_buffer("h_scale", torch.ones(N_ENT))
            self.register_buffer("dd_kind", torch.from_numpy(KIND).long(), persistent=False)
        # grid_cnn: a global 64x64 map (2 cells per pixel) drawn from the state every step: walls, heroes / creeps /
        # attack-ready damage / tower health per team, neutrals. A dilated 3-layer CNN reads it; its features at each
        # unit's pixel enter that unit's token through a zero-initialised projection (a checkpoint without it starts
        # unchanged). In the dream the map is redrawn from the predicted state, so it is predicted with it.
        self.grid_cnn = grid_cnn
        if grid_cnn:
            from .data import load_walls
            w = torch.from_numpy(load_walls()).float()
            self.register_buffer("g_walls", Fn.avg_pool2d(w[None, None], 2)[0, 0], persistent=False)
            self.g_cnn = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU())
            self.g_out = nn.Linear(32, d)
            nn.init.zeros_(self.g_out.weight)
            nn.init.zeros_(self.g_out.bias)
            self.register_buffer("g_kind", torch.from_numpy(KIND).long(), persistent=False)
        if attack_mode:
            from .data import ATTACK_DIM
            self.atk_in = nn.Linear(ATTACK_DIM, d)
            nn.init.zeros_(self.atk_in.weight)
            nn.init.zeros_(self.atk_in.bias)
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

    def grid(self, f):
        """(B, 10, 64, 64) map of the state and each unit's flat pixel index (B, 206)."""
        G = 64
        xy, team, alive, sh = f["xy"], f["team"].clamp(0, 2), f["alive"], f["shared"]
        B = xy.shape[0]
        pix = (xy[..., 1] / 2).long().clamp(0, G - 1) * G + (xy[..., 0] / 2).long().clamp(0, G - 1)
        k = self.g_kind.view(1, -1).expand(B, -1)
        a = alive.float()
        ready = a * (sh[..., 7] * 10 <= 1).float() * ((k == 0) | (k == 1)).float()
        chans = []
        for tm in (0, 1):
            t = (team == tm).float() * a
            chans += [(1 + tm, t * (k == 0).float()), (3 + tm, t * (k == 1).float()),
                      (5 + tm, (team == tm).float() * ready * sh[..., 3]), (7 + tm, t * (k == 3).float() * sh[..., 1])]
        chans.append((9, a * (k == 2).float()))
        flat = torch.zeros(B, 10 * G * G, device=xy.device, dtype=xy.dtype)
        for ch, val in chans:
            flat.scatter_add_(1, pix + ch * G * G, val.to(flat.dtype))
        g = flat.view(B, 10, G, G)
        g[:, 0] = self.g_walls
        return g, pix

    def set_health_scale(self, stats):
        """direct_damage: the health regression scale per entity (value units = delta / scale)."""
        if self.direct_damage:
            j = REG_FIELDS.index("health")
            self.h_scale.copy_(stats.reg_scale[torch.from_numpy(KIND), j].to(self.h_scale.device).float())

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
        if self.attack_mode:
            x[:, :N_ENT] = x[:, :N_ENT] + self.atk_in(f["atk"])
        if self.list_state:
            x[:, :N_ENT] = x[:, :N_ENT] + self.agg_in(f["aggro"])
        if self.grid_cnn:
            g, pix = self.grid(f)
            fm = self.g_cnn(g).flatten(2)                                                     # (B, 32, 4096)
            at = torch.gather(fm, 2, pix.unsqueeze(1).expand(-1, fm.shape[1], -1)).transpose(1, 2)   # (B, 206, 32)
            x[:, :N_ENT] = x[:, :N_ENT] + self.g_out(at)
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
        if self.direct_damage:
            jh = REG_FIELDS.index("health")
            inc = f["aggro"][..., 7] * 300                                              # HP from ready attackers
            add = torch.zeros_like(value)
            add[..., jh] = self.dd_val[self.dd_kind] * (-inc / self.h_scale)
            value = value + add
        if self.gated:
            change = self.change(h) + self.change_prior
            if self.direct_damage:
                addc = torch.zeros_like(change)
                addc[..., jh] = self.dd_gate[self.dd_kind] * (inc > 0).float()
                change = change + addc
            out = dict(change=change, value=value, gate_mask=self.gate_mask)
            closed = self.gate_mask & (change <= 0)                          # hard gate at p = 0.5
            value = torch.where(closed, torch.zeros_like(value), value)
        if self.list_state:
            with torch.no_grad():
                xy = f["xy"]
                o, m = xy[:, self.l_own], xy[:, self.l_mem]                                       # (B,124,2), (B,134,2)
                dxy = (o.unsqueeze(2) - m.unsqueeze(1)).abs().clamp(max=9).to(torch.int32)        # (B,124,134,2)
                ph = (f["tick"] % 15).to(torch.int32).view(-1, 1, 1)
                idx = (((((self.l_okind.view(1, -1, 1).to(torch.int32) * 10 + dxy[..., 0]) * 10 + dxy[..., 1]) * 2
                         + f["lists"].to(torch.int32)) * 15 + ph) * N_KIND + self.l_mkind.view(1, 1, -1).to(torch.int32))
            lg = torch.einsum("bod,bmd->bom", self.lq(h[:, self.l_own]), self.lk(h[:, self.l_mem])) / math.sqrt(32)
            out["list_logits"] = lg + Fn.embedding(idx, self.l_tab.view(-1, 1)).squeeze(-1)        # (B,124,134)
        if self.predict_targets:
            q, k = self.tq(h), self.tk(h)                                         # (B,206,64)
            lg = torch.einsum("bnd,bmd->bnm", q, k) / 8.0
            if self.target_dist:
                with torch.no_grad():                                                    # flat table index per pair
                    xy, tm = f["xy"], f["team"]
                    l1 = (xy.unsqueeze(2) - xy.unsqueeze(1)).abs().sum(-1).clamp(max=16).to(torch.int32)   # (B,206,206)
                    rel = torch.where((tm.unsqueeze(2) == 2) | (tm.unsqueeze(1) == 2), 2,
                                      (tm.unsqueeze(2) != tm.unsqueeze(1)).to(torch.int32))           # 0 ally 1 enemy 2 neutral
                    idx = (self.t_kind.view(1, -1, 1).to(torch.int32) * 3 + rel) * 17 + l1
                # an embedding lookup, not advanced indexing: its backward sorts instead of 49M atomic adds into
                # 204 cells (H200: 41 vs 883 ms per step)
                lg = lg + Fn.embedding(idx, self.t_dist.view(-1, 1)).squeeze(-1)
            # dead units can't be targets (soft mask, bounded loss); self-targets are real (self-cast, ~4%)
            lg = lg.masked_fill(~f["alive"].unsqueeze(1), -30.0)
            none = (q @ self.t_none / 8.0 + self.t_none_bias).unsqueeze(-1)
            out["target_logits"] = torch.cat([lg, none], -1)                      # (B,206,207); 206 = nobody
        return dict(
            **out,
            reg=value,
            alive_flip=self.alive_flip(h).squeeze(-1),
            is_hit=self.is_hit(h).squeeze(-1),
            lane=self.lane(h[:, c]) + 5 * lane_cur,
            team=self.team(h[:, c]) + 5 * team_cur,
        ), h
