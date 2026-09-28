"""Regenerate every MOBA figure and table from committed files only.

Reads results/moba/<run_id>/history.csv (W&B exports, scripts/moba/export_wandb.py)
and results/moba/dream/*.jsonl. Writes results/moba/figures/*.png, each with the
data table behind it in results/moba/tables/*.md (the table is the accessible view
and the source for any number quoted in a doc). No number in a doc is typed by hand
from a chart: if a table here disagrees with a doc, the doc is stale.

    .venv/bin/python results/moba/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parent
FIG, TAB = R / "figures", R / "tables"
H = [1, 4, 16, 64, 256, 1024, 4096]
# validated categorical palette (dataviz reference, light), fixed order, never cycled
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def history(run):
    with open(R / run / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items() if v not in ("", None) and _num(v)} for r in rows]


def _num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def horizon_row(run, step):
    for r in history(run):
        if int(r.get("_step", -1)) == step and "horizon/loss@1" in r:
            return r
    raise KeyError(f"{run} has no horizon eval at step {step}")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def md_table(path, header, rows, note):
    lines = [f"<!-- regenerate: .venv/bin/python results/moba/make_figures.py -->", "", note, "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n")


def fig_horizon():
    """Held-out loss by open-loop horizon for the key world models."""
    models = [("E4 final (teacher-forced)", "93m7sktq", 0),          # E5's update-0 eval = E4 final
              ("E5 @20,500 (rollout, kept)", "93m7sktq", 20500),
              ("E5 continued +1,500 (weights lost)", "soeosr9h", 1500),
              ("E6b @3,500 (counterfactual + effect)", "d9xnnj0a", 3500)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    rows = []
    ends = []
    for (name, run, step), col in zip(models, C):
        r = horizon_row(run, step)
        y = [r[f"horizon/loss@{h}"] for h in H]
        ax.plot(H, y, color=col, linewidth=2, marker="o", markersize=5, label=name)
        ends.append((y[-1], name.split(" (")[0]))
        rows.append([name, f"`results/moba/{run}/` step {step}"] + [f"{v:.2f}" for v in y])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(H, [str(h) for h in H])
    # direct labels at the right edge, pushed apart so they never overlap (>= 11 pt vertically)
    placed = []
    for yv, label in sorted(ends):
        ypx = ax.transData.transform((H[-1], yv))[1]
        while any(abs(ypx - p) < 11 * fig.dpi / 72 for p in placed):
            ypx += 2
        placed.append(ypx)
        ax.annotate(label, (H[-1], yv), xytext=(6, (ypx - ax.transData.transform((H[-1], yv))[1]) * 72 / fig.dpi),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    style(ax, "Held-out loss by open-loop horizon", "horizon h (ticks the model runs on its own predictions)",
          "held-out loss@h (log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "horizon_loss.png")
    plt.close(fig)
    md_table(TAB / "horizon_loss.md", ["model", "source"] + [f"h{h}" for h in H], rows,
             "Held-out loss@h (games 900-999, same 32 windows; h4096 from the few long games only).")


def fig_training():
    """h1 and h1024 over training, E5-equivalent update axis (E6b forked at 3,500; the continuation at 20,500)."""
    series = [("E5", "93m7sktq", 0), ("E5 continued", "soeosr9h", 20500), ("E6b", "d9xnnj0a", 3500)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    rows = []
    for (name, run, off), col in zip(series, C):
        hs = [r for r in history(run) if "horizon/loss@1" in r]
        for ax, h in zip(axes, (1, 1024)):
            ax.plot([off + r["_step"] for r in hs], [r[f"horizon/loss@{h}"] for r in hs], color=col,
                    linewidth=2, marker="o", markersize=3, label=name)
        rows += [[name, int(off + r["_step"]), f"{r['horizon/loss@1']:.2f}", f"{r['horizon/loss@1024']:.1f}",
                  f"{r.get('val/loss', float('nan')):.2f}"] for r in hs]
    for ax, h in zip(axes, (1, 1024)):
        style(ax, f"held-out loss@{h}", "E5-equivalent update", "loss")
        ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "horizon_over_training.png")
    plt.close(fig)
    md_table(TAB / "horizon_over_training.md", ["run", "update (E5-equivalent)", "h1", "h1024", "val loss"], rows,
             "E6b starts from E5 @3,500; the continuation from E5 @20,500. Its update-0 row is the same checkpoint "
             "as E5 @20,500.")


def fig_dream():
    """Dream-PPO sweep on the H200: dream reward per 1k hero-steps vs agent steps."""
    cfgs = ["lr3e-4_mb2", "lr3e-3_mb10", "lr1e-3_mb10", "lr3e-4_mb10", "lr1e-3_mb2", "lr3e-3_mb2"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    rows = []
    for c, col in zip(cfgs, C):
        rs = [json.loads(l) for l in open(R / "dream" / f"sweep1_gpu_{c}.log.jsonl")]
        x, y = [r["agent_steps"] / 1e6 for r in rs], [r["reward_per_1k"] for r in rs]
        ax.plot(x, y, color=col, linewidth=2, label=c)
        rows.append([c, len(rs), f"{rs[-1]['agent_steps'] / 1e6:.2f}M", f"{y[0]:.2f}",
                     f"{sum(y[-3:]) / 3:.2f}", f"{sum(r['deaths_per_1k'] for r in rs[-3:]) / 3:.2f}"])
    best = [json.loads(l) for l in open(R / "dream" / "sweep1_gpu_lr3e-4_mb2.log.jsonl")][-1]
    ax.annotate("lr3e-4_mb2 (evaluated in the real game)", (best["agent_steps"] / 1e6, best["reward_per_1k"]),
                xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    style(ax, "Dream reward during PPO (inside the world model, not the real game)", "agent steps (millions)",
          "dream reward per 1k hero-steps")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "dream_ppo_sweep.png")
    plt.close(fig)
    md_table(TAB / "dream_ppo_sweep.md", ["config (lr, minibatches)", "updates", "agent steps", "reward/1k first",
                                          "reward/1k last 3", "deaths/1k last 3"], rows,
             "One seed per config, 4.5 min each, all six in parallel on one H200. Dream reward only.")


def fig_arena():
    """Real-game evaluation: mean return per hero by team and opponent."""
    games = [json.loads(l) for l in open(R / "dream" / "arena_real_game_lr3e-4_mb2.jsonl")]
    teams = [("ppo", "dream PPO"), ("random", "untrained (random)"), ("net", "shipped net (reference)")]
    opps = ["net", "script"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    w = 0.26
    rows = []
    for i, ((t, label), col) in enumerate(zip(teams, C)):
        vals = []
        for j, o in enumerate(opps):
            g = [x for x in games if x["team"] == t and x["opp"] == o]
            ret = sum(x["return_ours"] for x in g) / len(g)
            vals.append(ret)
            rows.append([label, f"vs {o}", len(g), f"{100 * sum(x['won'] for x in g) / len(g):.0f}",
                         f"{100 * sum(x['lost'] for x in g) / len(g):.0f}", f"{ret:.2f}",
                         f"{sum(x['ours_deaths'] for x in g) / len(g):.2f}", f"{sum(x['ours_level'] for x in g) / len(g):.2f}"])
        xs = [j + (i - 1) * w for j in range(len(opps))]
        bars = ax.bar(xs, vals, width=w - 0.03, color=col, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(opps)), [f"vs {o}" for o in opps])
    style(ax, "Real C game: mean return per hero (20 games per bar)", "opponent", "return per hero")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "arena_real_game.png")
    plt.close(fig)
    md_table(TAB / "arena_real_game.md", ["our team", "opponent", "games", "win %", "loss %", "return/hero",
                                          "deaths/hero", "level"], rows,
             "10 games per side x 2 sides, seeds 50000000+, max 6,000 ticks; same seeds for every team.")


def tab_census():
    c = json.load(open(R / "data" / "controller_census.json"))
    rows = []
    for corpus, d in c.items():
        for k, v in d["matchups"].items():
            rows.append([corpus, k, v["games"], v["ticks"]])
        rows.append([corpus, "hero-tick share", "", ", ".join(f"{k} {v}%" for k, v in d["hero_tick_share_pct"].items())])
        rows.append([corpus, "epsilon (log headers)", "", str(d["epsilon"])])
    md_table(TAB / "data_controller_census.md", ["corpus", "matchup (radiant vs dire)", "games", "ticks"], rows,
             "From the .mbl log headers (results/moba/data/controller_census.json).")


def _audit_rows(label, path):
    a = json.load(open(path))
    rows = []
    for team, r in a["teams"].items():
        for k, lab in (("real_per_1k", "real game"), ("teacher_forced_dream_per_1k", "dream, teacher-forced"),
                       ("openloop_dream_per_1k", "dream, open loop 128")):
            d, x, _, tw = r[k]
            rows.append([label, team, lab, f"{d:.2f}", f"{x:.0f}", f"{tw:.3f}"])
    return rows


def tab_xp():
    H = ["world model", "our team's actions", "measure", "deaths/1k", "XP/1k", "tower kills/1k"]
    rows = _audit_rows("E5 @20,500", R / "dream" / "dream_vs_real_reward.json")
    md_table(TAB / "xp_audit_e5.md", H, rows,
             "Our team (radiant) vs the shipped net, 8 real games x 900 ticks, seeds 70000000+; identical actions "
             "fed to the WM. scripts/moba/dream_vs_real_reward.py")
    rows = []
    for label, f in (("E5 @20,500, clamp off (eval only)", "xpclamp_eval_noclamp.json"),
                     ("E5 @20,500, clamp off + signed XP reward", "xpclamp_eval_noclamp_signed.json"),
                     ("+~30 upd, XP loss x1 (control)", "xpsmoke_xp1.json"),
                     ("+~30 upd, XP loss x100", "xpsmoke_xp100.json"),
                     ("+~30 upd, no XP clamp", "xpsmoke_noclamp.json")):
        rows += [r for r in _audit_rows(label, R / "dream" / f) if r[2] != "real game"]
    md_table(TAB / "xp_audit_fixes.md", H, rows, "Same audit (real rows as in xp_audit_e5.md). Fine-tunes: local CPU, "
             "batch 32, ~30 updates, single seed: smoke tests, not results.")


def tab_dyna():
    H = ["world model", "our team's actions", "measure", "deaths/1k", "XP/1k", "tower kills/1k"]
    rows = []
    for label, f in (("control: mix1k only", "dyna1_control_xp_audit.json"),
                     ("Dyna: mix1k + 100 policy games", "dyna1_policy_games_xp_audit.json")):
        rows += _audit_rows(label, R / "dyna" / f)
    md_table(TAB / "dyna1_xp_audit.md", H, rows, "Both from E5 @20,500, ~125 updates, batch 384, H100 NVL, single seed; "
             "audit seeds 70000000+ (the Dyna games used 80000000+).")


def tab_planner():
    P = [json.loads(l) for l in open(R / "planner" / "realtime_vs_net.jsonl")]
    B = [json.loads(l) for l in open(R / "planner" / "baselines_vs_net.jsonl")]
    import numpy as np
    def row(name, g):
        m = lambda k: np.mean([x[k] for x in g])
        return [name, len(g), f"{100 * m('won'):.0f}", f"{100 * m('lost'):.0f}",
                f"{100 * np.mean([x['winner'] == 'none' for x in g]):.0f}", f"{m('return_ours'):.2f}",
                f"{m('ours_deaths'):.2f}", f"{m('ours_level'):.2f}", f"{m('ours_creeps_killed'):.1f}"]
    rows = [row("WM planner (60 fps budget)", P)] + [row(f"planner as {s}", [x for x in P if x["side"] == s])
                                                    for s in ("radiant", "dire")]
    rows += [row("shipped net (mirror)", [x for x in B if x["team"] == "net"]),
             row("untrained (random)", [x for x in B if x["team"] == "random"])]
    ch = np.sum([x["branch_chosen"] for x in P], 0)
    md_table(TAB / "planner_vs_net.md", ["our team (vs shipped net)", "games", "win %", "loss %", "timeout %",
                                         "return/hero", "deaths", "level", "creeps"], rows,
             f"Seeds 50000000-50000059 per side, max 6,000 ticks. Branch chosen (net's action / keep plan / new plan): "
             f"{ch.tolist()} = net's action {100 * ch[0] / ch.sum():.0f}%.")


def tab_resume_horizons(run, out, title):
    """Held-out loss by open-loop horizon at every eval of a resumed run, x = vs its step 0."""
    H = [1, 4, 16, 64, 256, 1024, 4096]
    rows = []
    for pre, name in (("random/horizon", "random held-out"), ("horizon", "mix1k held-out 900-999")):
        ev = [r for r in history(run) if f"{pre}/loss@1" in r]
        if not ev:
            continue
        base = ev[0]
        hs = [h for h in H if f"{pre}/loss@{h}" in base]
        for r in ev:
            cells = [f"{r[f'{pre}/loss@{h}']:.2f}" + ("" if r is base else f" ({r[f'{pre}/loss@{h}'] / base[f'{pre}/loss@{h}']:.2f}x)")
                     for h in H if h in hs] + ["-"] * (len(H) - len(hs))
            rows.append([name, int(r["_step"]), *cells, f"{r[f'{pre}/hero_pos_L1@1024']:.1f}"])
    md_table(TAB / out, ["held-out set", "update", *[f"h{h}" for h in H], "hero cells @1024"], rows, title)


def tab_arena_e11():
    import numpy as np
    D = R / "dream"
    arms = [("P3 self-play (E5 dream, 4.5 min)", "arena_real_game_p3_selfplay_rerun.jsonl", "ppo"),
            ("P3 vs net (E5 dream, 4.5 min)", "arena_real_game_p3_vs_net.jsonl", "ppo"),
            ("P4 mix, expert starts (E11 @1500, 4.5 min)", "arena_real_game_p4_mix_e11.jsonl", "ppo"),
            ("P4b mix, expert + random starts (E11 @1500, 4.5 min)", "arena_real_game_p4_mix_e11_mixstarts.jsonl", "ppo"),
            ("P5 = P4b recipe @81.3M steps (policy_081342665)", "arena_real_game_p5_policy_081342665.jsonl", "ppo"),
            ("random actions", "arena_real_game_p4_mix_e11.jsonl", "random"),
            ("shipped net", "arena_real_game_p4_mix_e11.jsonl", "net")]
    rows = []
    for label, f, team in arms:
        g = [x for x in map(json.loads, open(D / f)) if x["team"] == team and x["opp"] == "net"]
        m = lambda k: np.mean([float(x[k]) for x in g])
        rows.append([label, len(g), f"{100 * m('won'):.0f}", f"{m('return_ours'):.2f}", f"{m('ours_deaths'):.2f}",
                     f"{m('ours_level'):.2f}", f"{m('ours_creeps_killed'):.2f}", f"{m('ours_towers_killed'):.2f}"])
    md_table(TAB / "arena_e11_ppo_vs_net.md", ["our team (vs shipped net)", "games", "win %", "return/hero", "deaths",
                                               "level", "creeps", "towers"], rows,
             "Real game, seeds 50000000+, 10 games per side, max 6,000 ticks (mobawm/rl/arena.py, the P2/P3 harness).")


def tab_dvr_e11():
    W = (-0.163764, 0.00665677, 0.0, 0.642119)
    ret = lambda v: sum(a * b for a, b in zip(v, W))
    rows = []
    for team in ("ppo", "random", "net"):
        for f, lab in (("dvr_p4policy_e5_rollout.json", "E5 @20,500"), ("dvr_p4policy_e11_at1500.json", "E11 @1500")):
            r = json.load(open(R / "dream" / f))["teams"][team]
            Rl, TF, OR, OL = (r[k] for k in ("real_per_1k", "teacher_forced_dream_per_1k", "openloop_real_per_1k",
                                             "openloop_dream_per_1k"))
            rows.append([{"ppo": "P4 policy"}.get(team, team), lab, f"{ret(Rl):.2f}", f"{ret(TF):.2f}",
                         f"{ret(OL):.2f} (real {ret(OR):.2f})", f"{Rl[1]:.0f}", f"{TF[1]:.0f}", f"{OL[1]:.0f} (real {OR[1]:.0f})"])
    md_table(TAB / "dream_vs_real_e11.md", ["our team", "world model", "real reward/1k", "dream, teacher-forced",
                                            "dream, open loop 128", "real XP/1k", "dream XP TF", "dream XP OL"], rows,
             "scripts/moba/dream_vs_real_reward.py: 8 real games x 900 ticks vs the shipped net, seeds 70000000+; "
             "reward = death + XP + tower terms per 1k hero-steps of our team.")


def tab_arena_sides(out, arms, note):
    """Real game vs the shipped net, split by side: arms = [(label, file in results/moba/dream, team)]."""
    import numpy as np
    rows = []
    for label, f, team in arms:
        g0 = [x for x in map(json.loads, open(R / "dream" / f)) if x["team"] == team and x["opp"] == "net"]
        for side in ("radiant", "dire"):
            g = [x for x in g0 if x["side"] == side]
            m = lambda k: np.mean([float(x[k]) for x in g])
            se = np.std([x["return_ours"] for x in g], ddof=1) / len(g) ** 0.5
            rows.append([label, side, len(g), f"{100 * m('won'):.0f}", f"{100 * m('lost'):.0f}",
                         f"{m('return_ours'):.2f} ± {se:.2f}", f"{m('ours_creeps_killed'):.1f}",
                         f"{m('ours_level'):.2f}", f"{m('ours_towers_killed'):.2f}"])
    md_table(TAB / out, ["our team (vs shipped net)", "side", "games", "win %", "loss %", "return/hero ± s.e.",
                         "creeps", "level", "towers"], rows, note)


def tab_tower_audit(out, arms, note):
    """Tower-death prediction (scripts/moba/tower_event_audit.py): precision/recall, one step and open loop 128."""
    rows = []
    for label, f in arms:
        r = json.load(open(R / "dream" / f))
        a, o = r["teacher_forced_alarms"], r["open_loop"]["128"]
        rows.append([label, r["games"], a["real"], f"{100 * a['recall']:.0f}%", f"{100 * a['precision_exact']:.0f}%",
                     f"{100 * a['precision_incl_early']:.0f}%", a["false"], o["real"], o["predicted"], o["hits"],
                     f"{100 * o['precision']:.0f}%", f"{100 * o['recall']:.0f}%"])
        if "512" in r["open_loop"]:
            o5 = r["open_loop"]["512"]
            rows.append([label + " (open loop 512)", r["games"], "", "", "", "", "", o5["real"], o5["predicted"],
                         o5["hits"], f"{100 * o5['precision']:.0f}%", f"{100 * o5['recall']:.0f}%"])
    md_table(TAB / out, ["world model", "games", "real falls", "one step recall", "one step precision (exact)",
                         "one step precision (incl. early)", "false alarms", "open-loop real", "open-loop predicted",
                         "hits", "open-loop precision", "open-loop recall"], rows, note)


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    fig_horizon()
    fig_training()
    fig_dream()
    fig_arena()
    tab_census()
    tab_xp()
    tab_dyna()
    tab_planner()
    tab_resume_horizons("mlbh5d3l", "e11_horizons.md", "E11 (W&B mlbh5d3l): update 0 = E5 @20,500. "
                        "32 fixed windows per horizon; x = loss relative to update 0.")
    tab_resume_horizons("xfldazic", "e12_horizons.md", "E12 (W&B xfldazic): update 0 = E11 @2000. "
                        "32 fixed windows per horizon; x = loss relative to update 0.")
    tab_arena_e11()
    tab_dvr_e11()
    tab_arena_sides("arena_e14_policy_arms.md", [
        ("P6 base (MLP, = P4b)", "arena_real_game_p6_base.jsonl", "ppo"),
        ("P6 MLP + role heads", "arena_real_game_p6_roleheads.jsonl", "ppo"),
        ("P7 B: PufferLib MinGRU + old-repo PPO", "arena_real_game_p7_puffer_tuned.jsonl", "ppo"),
        ("P7 C: PufferLib MinGRU + our PPO", "arena_real_game_p7_puffer_ours.jsonl", "ppo"),
        ("P8 C, entropy 0.003", "arena_real_game_p8_ent0.003.jsonl", "ppo"),
        ("P8 C, entropy 0.03", "arena_real_game_p8_ent0.03.jsonl", "ppo"),
        ("P8 C, entropy 0.05", "arena_real_game_p8_ent0.05.jsonl", "ppo"),
        ("P8 C, entropy 0.1", "arena_real_game_p8_ent0.1.jsonl", "ppo"),
        ("P9 C at 2x steps (3.42M)", "arena_real_game_p9_puffer_ours_2x.jsonl", "ppo"),
        ("P10 C in the E12 @4000 dream", "arena_real_game_p10_puffer_ours_e12.jsonl", "ppo"),
        ("random actions", "arena_real_game_p4_mix_e11.jsonl", "random"),
        ("shipped net (mirror)", "arena_real_game_p4_mix_e11.jsonl", "net")],
        "E14 arms, 1.70M dream steps each unless stated; 10 games per side, seeds 50000000+ (mobawm/rl/arena.py).")
    tab_arena_sides("arena_e14_armC_120.md", [
        ("arm C (P7 C)", "arena_real_game_p7_armC_120.jsonl", "ppo"),
        ("shipped net (mirror)", "arena_real_game_p7_armC_120.jsonl", "net"),
        ("random actions", "arena_real_game_p7_armC_120.jsonl", "random")],
        "Arm C over 120 games: 60 per side, seeds 50000000+.")
    tab_tower_audit("tower_audit_e11.md", [("E11 @1500", "tower_event_audit.json")],
                    "Held-out games 900-919. Open loop: 128/512-tick windows from real states, real actions; a hit = the "
                    "same tower falls within +-32 ticks. One step: alarms = runs of consecutive predicted falls.")
    tab_tower_audit("tower_audit_e15.md", [
        ("E11 @1500 (no fine-tune)", "tower_audit_e15_baseline.json"),
        ("control: fine-tune only", "tower_audit_e15_control.json"),
        ("real attack graph", "tower_audit_e15_real.json"),
        ("derived attack graph", "tower_audit_e15_derived.json"),
        ("consistency loss", "tower_audit_e15_consist.json"),
        ("tower-event sampling", "tower_audit_e15_events.json"),
        ("tower health x10", "tower_audit_e15_tower10.json")],
        "E15 arms: E11 @1500 fine-tuned 5 minutes each (~60-100 updates, batch 384); held-out games 900-909.")
    tab_arena_sides("arena_e14_tower_shaping.md", [
        ("arm C: game reward", "arena_real_game_p7_puffer_ours.jsonl", "ppo"),
        ("P12 tower damage (k2, boost 2)", "arena_real_game_p12_dmg.jsonl", "ppo"),
        ("P12 falling-tower bonus 5", "arena_real_game_p12_fall.jsonl", "ppo"),
        ("P12 damage + fall", "arena_real_game_p12_dmg_fall.jsonl", "ppo"),
        ("P12 damage + fall, big", "arena_real_game_p12_dmg_fall_big.jsonl", "ppo"),
        ("P13 fall bonus 15", "arena_real_game_p13_fall15.jsonl", "ppo"),
        ("P13 fall bonus 30", "arena_real_game_p13_fall30.jsonl", "ppo"),
        ("P13 fall bonus 60", "arena_real_game_p13_fall60.jsonl", "ppo"),
        ("P13 fall bonus 120", "arena_real_game_p13_fall120.jsonl", "ppo"),
        ("shipped net (mirror)", "arena_real_game_p4_mix_e11.jsonl", "net")],
        "Arm C + dream-only tower shaping, 1.70M steps each, single seed; 10 games per side vs the shipped net.")
    tab_arena_sides("arena_e14_reward_sweep.md", [
        ("arm C: game reward", "arena_real_game_p7_puffer_ours.jsonl", "ppo"),
        ("P11 tower 3 to team", "arena_real_game_p11_t3team.jsonl", "ppo"),
        ("P11 win +-10", "arena_real_game_p11_win10.jsonl", "ppo"),
        ("P11 tower 3 team + win +-10", "arena_real_game_p11_t3team_win10.jsonl", "ppo"),
        ("P11 tower 10 team + win +-30", "arena_real_game_p11_t10team_win30.jsonl", "ppo"),
        ("shipped net (mirror)", "arena_real_game_p4_mix_e11.jsonl", "net")],
        "P11: arm C + dream-only reward shaping, 1.70M steps each; 10 games per side vs the shipped net.")
    for f in sorted(FIG.iterdir()):
        print(f.relative_to(R.parent.parent))


if __name__ == "__main__":
    main()
