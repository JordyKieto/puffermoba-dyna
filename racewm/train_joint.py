"""Joint world-model + actor-critic training, with three coupling modes.

    --mode frozen   world model fixed; actor-critic trains against it.
                    What we ran before.
    --mode joint    world model keeps training on REAL data alongside the
                    actor. This is what Dreamer actually does -- "unfrozen"
                    means the model keeps learning from data, NOT that actor
                    gradients reach it. The two objectives stay separate.
    --mode full     true end-to-end: the actor's loss also backpropagates into
                    the world model.

`full` is the one to be suspicious of. If the actor can change the model, the
cheapest way to raise imagined return is to make the model predict more reward
-- the model learns to hallucinate rather than to be right. Dreamer, MuZero and
TD-MPC2 all avoid this: MuZero trains dynamics through reward/value/policy
prediction, never through the actor's objective.

We measured model exploitation already: with a learned critic, the planner that
searched hardest scored worst online (beam6 0.411 vs greedy 0.753). `full`
gives the actor a direct lever on the model, so it should make that worse. Run
it to find out rather than to confirm.

Every mode reports real and imagined transitions separately.

    python3 -m racewm.train_joint --mode joint --real-budget 200000 --eval
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt_mod
from . import contract, dataset, export_policy, geometry, model as M
from . import planning, shards, tracking
from .dreamer import ActorCritic, lambda_returns
from .train_termination import TerminationHead



def _load_actor_weights(actor, state_dict):
    """Load BC weights into the actor trunk.

    BC's PolicyMLP wraps its Sequential in `self.net`, so its keys are
    "net.0.weight"; ActorCritic.actor IS the Sequential, so its keys are
    "0.weight". The shapes match and only the prefix differs. This used to be a
    try/except that printed a warning -- which meant a failed warm start looked
    like a successful run that had simply learned nothing from BC, and the
    frozen-vs-joint comparison it was supposed to support was meaningless.
    Mismatched keys are now a hard error.
    """
    fixed = {k[len("net."):] if k.startswith("net.") else k: v
             for k, v in state_dict.items()}
    actor.load_state_dict(fixed)   # raises on any real mismatch

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "joint", "full"], default="joint")
    ap.add_argument("--wm", default="checkpoints/wm_multistep_warm/best.pt")
    ap.add_argument("--termination", default="checkpoints/termination/best.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--real-budget", type=int, default=200_000)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--wm-batch", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wm-lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--bc-init", default="checkpoints/bc_expert_only/best.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev, H = args.device, args.horizon
    run_name = args.run_name or f"joint_{args.mode}_real{args.real_budget//1000}k"
    out = args.out or f"checkpoints/{run_name}/policy.rpo"

    dataset.assert_disjoint(args.corpus)
    man = dataset.load_manifest(args.corpus)
    real = dataset.load_split(args.corpus, "train", slices=dataset.WM_SLICES,
                              with_geometry=True, include_terminal=False,
                              max_rows=args.real_budget, seed=args.seed)
    print(f"mode={args.mode}  real budget {len(real):,} transitions")

    x = torch.from_numpy(dataset.wm_inputs(real))
    y = torch.from_numpy(dataset.wm_targets(real))
    x_mu, x_sd = x.mean(0, keepdim=True).to(dev), x.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    y_mu, y_sd = y.mean(0, keepdim=True).to(dev), y.std(0, keepdim=True).clamp_min(1e-8).to(dev)
    xn = ((x - x_mu.cpu()) / x_sd.cpu()).to(dev)
    yn = ((y - y_mu.cpu()) / y_sd.cpu()).to(dev)

    obs_t = torch.from_numpy(real.obs).to(dev)
    obs_mu = obs_t.mean(0, keepdim=True)
    obs_sd = obs_t.std(0, keepdim=True).clamp_min(1e-6)

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    wcfg = ck["config"]
    wm = M.DeltaMLP(wcfg["d_in"], contract.OBS_SIZE, wcfg["hidden"], wcfg["layers"]).to(dev)
    wm.load_state_dict(ck["model"])
    trainable = args.mode != "frozen"
    for p in wm.parameters():
        p.requires_grad_(trainable)
    wm.train(trainable)
    wm_opt = torch.optim.AdamW(wm.parameters(), lr=args.wm_lr) if trainable else None

    head = None
    if Path(args.termination).exists():
        tk = torch.load(args.termination, map_location=dev, weights_only=False)
        tc = tk["config"]
        head = TerminationHead(tc["d_in"], tc["hidden"], tc["layers"]).to(dev)
        head.load_state_dict(tk["model"]); head.eval()
        for q in head.parameters():
            q.requires_grad_(False)
        ex = tk.get("extra") or {}
        t_mu, t_sd = ex.get("mu", x_mu).to(dev), ex.get("sd", x_sd).to(dev)

    tracks = shards.load_tracks(f"{args.corpus}/train.tracks.bin")
    bank = geometry.TrackBank(tracks, real.header, dev)
    slots_all = bank.slots(real.env_idx)

    ac = ActorCritic(hidden=args.hidden, layers=args.layers).to(dev)
    if args.bc_init and Path(args.bc_init).exists():
        _load_actor_weights(ac.actor, torch.load(args.bc_init, map_location=dev,
                                                 weights_only=False)["model"])
        print(f"actor warm started from {args.bc_init}")
    opt = torch.optim.AdamW(ac.parameters(), lr=args.lr, weight_decay=1e-5)

    cfg = dict(vars(args), n_real=len(real), corpus_manifest=man)
    cp = ckpt_mod.Checkpointer("checkpoints", run_name, config=cfg,
                               metric_name="imagined_return", metric_mode="max")
    trk = tracking.RunTracker(run_name, config=cfg,
                              mode="wandb" if args.wandb else "off",
                              tags=["phase2", "joint", args.mode])
    trk.watch_params(ac)

    g = torch.Generator().manual_seed(args.seed)
    imagined_total = 0
    t0 = time.time()
    ar = torch.arange(args.batch, device=dev)

    def imagine_differentiable(s0, slots):
        """Rollout that keeps the model in the graph when mode == full."""
        s = s0
        logps, rewards, ents, alives, states = [], [], [], [], []
        live = torch.ones(len(s0), device=dev)
        for _ in range(H):
            logits, _ = ac((s - obs_mu) / obs_sd)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            oh = torch.zeros(len(s), contract.NUM_ACTIONS, device=dev)
            oh[ar[:len(s)], a] = 1.0
            gm = bank.local_window(s, slots)
            xi = torch.cat([s, oh, gm], dim=1)
            delta = wm((xi - x_mu) / x_sd) * y_sd + y_mu
            s2 = M.project_state(s + delta)
            with torch.no_grad():
                p_term = (torch.sigmoid(head((xi - t_mu) / t_sd)) if head is not None
                          else torch.zeros(len(s), device=dev))
            i = contract.INDEX
            r = (s2[:, i["score_norm"]] - s[:, i["score_norm"]]) * 100.0
            states.append(s); logps.append(dist.log_prob(a))
            ents.append(dist.entropy()); rewards.append(r * live)
            live = live * (1 - p_term)
            alives.append(live.clone())
            s = s2
        return (torch.stack(states, 1), torch.stack(logps, 1),
                torch.stack(rewards, 1), torch.stack(ents, 1),
                torch.stack(alives, 1), s)

    for it in range(args.updates):
        # --- world-model update on REAL data (joint and full) ---------------
        wm_loss_val = float("nan")
        if trainable:
            wi = torch.randint(0, len(xn), (args.wm_batch,), generator=g).to(dev)
            wm_loss = nn.functional.mse_loss(wm(xn[wi]), yn[wi])
            wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            wm_opt.step()
            wm_loss_val = float(wm_loss.detach())

        # --- actor-critic update in imagination -----------------------------
        idx = torch.randint(0, len(obs_t), (args.batch,), generator=g).to(dev)
        s0, slots = obs_t[idx], slots_all[idx]

        if args.mode == "full":
            states, logps, rewards, ents, alive, s_end = imagine_differentiable(s0, slots)
        else:
            with torch.no_grad():
                sim = planning.WorldModelSim(wm, bank, x_mu, x_sd, y_mu, y_sd,
                                             head, t_mu if head is not None else None,
                                             t_sd if head is not None else None)
            from .dreamer import imagine
            states, logps, rewards, ents, alive, s_end = imagine(
                sim, ac, s0, slots, H, obs_mu, obs_sd)
        imagined_total += args.batch * H

        z = (states.reshape(-1, contract.OBS_SIZE) - obs_mu) / obs_sd
        _, v_flat = ac(z)
        values = v_flat.view(args.batch, H)
        with torch.no_grad():
            _, boot = ac((s_end.detach() - obs_mu) / obs_sd)
            ret = lambda_returns(rewards.detach(), values.detach(), boot,
                                 args.gamma, args.lam)
            adv = (ret - values.detach())
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

        actor_loss = -((logps * adv * alive.detach()).mean() + args.ent_coef * ents.mean())
        critic_loss = (((values - ret) ** 2) * alive.detach()).mean()
        loss = actor_loss + 0.5 * critic_loss
        # In `full`, this backward also reaches the world model -- the actor can
        # improve its objective by changing what the model predicts.
        opt.zero_grad(set_to_none=True)
        if trainable and args.mode == "full":
            wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        opt.step()
        if trainable and args.mode == "full":
            nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            wm_opt.step()

        if (it + 1) % 50 == 0 or it == 0:
            imag_ret = float((rewards * alive).sum(1).mean())
            p = trk.log(it + 1, samples=0, actor_loss=float(actor_loss.detach()),
                        critic_loss=float(critic_loss.detach()),
                        wm_loss=wm_loss_val, imagined_return=imag_ret,
                        survival=float(alive[:, -1].mean()),
                        entropy=float(ents.mean().detach()),
                        real_transitions=len(real),
                        imagined_transitions=imagined_total)
            print(f"  {it+1:4d}/{args.updates}  imag_ret={imag_ret:7.3f}  "
                  f"surv={float(alive[:,-1].mean()):.3f}  H={float(ents.mean()):.3f}  "
                  f"wm={wm_loss_val:.5f}  imagined={imagined_total:,}  "
                  f"({p['wall_clock_s']:.0f}s)")
            cp.save(step=it + 1, model=ac, optimizer=opt,
                    metrics={"imagined_return": imag_ret})

    secs = time.time() - t0
    print(f"\nmode={args.mode}  real {len(real):,}  imagined {imagined_total:,}  "
          f"ratio {imagined_total/len(real):.1f}:1  ({secs:.0f}s)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_policy.export(ac.actor, obs_mu.cpu().numpy(), obs_sd.cpu().numpy(), out)
    print(f"exported {out}")
    trk.summary(train_seconds=secs, real_transitions=len(real),
                imagined_transitions=imagined_total, mode=args.mode)

    if args.eval:
        h = man["splits"]["holdout"]
        subprocess.run(["./build/race_eval", "--bots", "policy,centerline,whisker",
                        "--policy", out, "--tracks", str(h["n_tracks"]),
                        "--track-start", str(h["track_start"]),
                        "--episodes", str(args.eval_episodes), "--threads", "10",
                        "--frameskip", str(man["frameskip"]),
                        "--max-steps", str(man["max_episode_steps"])], check=False)
    trk.finish()


if __name__ == "__main__":
    main()
