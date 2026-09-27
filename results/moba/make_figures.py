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


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    fig_horizon()
    fig_training()
    fig_dream()
    fig_arena()
    for f in sorted(FIG.iterdir()):
        print(f.relative_to(R.parent.parent))


if __name__ == "__main__":
    main()
