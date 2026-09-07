"""The real-vs-imagined sample budget study.

The claim worth testing is real-sample efficiency, not total compute, so real
transitions and imagined transitions are tracked as INDEPENDENT axes and every
arm is scored the same way: online, on holdout tracks, by behaviour.

    A   BC on N real, no imagination              the baseline
    B   BC on N real + imagination at 1:1         does dreaming help at parity
    C   BC on N real + imagination at 10:1        does more dreaming keep helping
    D   BC on 10N real, no imagination            what 10x real data buys

If B approaches D, imagination bought a 10x reduction in real samples. If C
barely beats B, dreaming past 1:1 is mostly compute. If C is worse than B,
imagination actively hurts past parity -- which is the outcome that gets hidden
when only totals are reported.

    python3 -m racewm.sample_budget --wm checkpoints/wm_multistep_warm/best.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, label):
    print(f"\n{'='*72}\n {label}\n{'='*72}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r


def read_metrics(path):
    d = json.loads(Path(path).read_text())
    b = d["bots"]["policy"]
    return {
        "return": b["outcome"]["return"],
        "laps": b["progress"]["laps"],
        "crash": b["failure"]["crash_rate"],
        "lap_pct": b["progress"]["lap_completion_rate"],
        "coverage": b["progress"]["track_coverage"],
        "survive": b["outcome"]["survival_rate"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="checkpoints/wm_multistep_warm/best.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--n", type=int, default=50_000, help="the small real budget N")
    ap.add_argument("--mult", type=int, default=10, help="the large budget multiplier")
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--out", default="data/sample_budget.json")
    args = ap.parse_args()

    N, big = args.n, args.n * args.mult
    py = sys.executable
    tmp = Path("data/_budget"); tmp.mkdir(parents=True, exist_ok=True)
    results, budgets = {}, {}

    # --- A: BC on N real -----------------------------------------------------
    run([py, "-m", "racewm.train_bc", "--corpus", args.corpus,
         "--slices", "expert", "--real-budget", str(N), "--epochs", str(args.bc_epochs),
         "--out", f"{tmp}/A.rpo", "--run-name", f"budget_A_{N}", "--eval",
         "--eval-episodes", str(args.eval_episodes), "--json-out", f"{tmp}/A.json"],
        f"A: BC on {N:,} real")
    results["A_bc_N"] = read_metrics(f"{tmp}/A.json")
    budgets["A_bc_N"] = {"real": N, "imagined": 0}

    # --- D: BC on 10N real ---------------------------------------------------
    run([py, "-m", "racewm.train_bc", "--corpus", args.corpus,
         "--slices", "expert", "--real-budget", str(big), "--epochs", str(args.bc_epochs),
         "--out", f"{tmp}/D.rpo", "--run-name", f"budget_D_{big}", "--eval",
         "--eval-episodes", str(args.eval_episodes), "--json-out", f"{tmp}/D.json"],
        f"D: BC on {big:,} real ({args.mult}x)")
    results["D_bc_10N"] = read_metrics(f"{tmp}/D.json")
    budgets["D_bc_10N"] = {"real": big, "imagined": 0}

    # --- B, C: imagination on top of the SAME N real -------------------------
    # updates * batch * horizon = imagined transitions.
    for tag, target_ratio in (("B_dream_1x", 1), ("C_dream_10x", args.mult)):
        batch, H = 512, 15
        updates = max(1, (N * target_ratio) // (batch * H))
        run([py, "-m", "racewm.dreamer", "--wm", args.wm, "--corpus", args.corpus,
             "--real-budget", str(N), "--updates", str(updates), "--batch", str(batch),
             "--horizon", str(H), "--bc-init", f"checkpoints/budget_A_{N}/best.pt",
             "--out", f"{tmp}/{tag}.rpo", "--run-name", f"budget_{tag}_{N}",
             "--eval", "--eval-episodes", str(args.eval_episodes)],
            f"{tag}: BC({N:,} real) + imagination at {target_ratio}:1")
        # Dreamer's own eval prints; re-run to capture json for the table.
        man = json.loads(Path(args.corpus, "manifest.json").read_text())
        h = man["splits"]["holdout"]
        subprocess.run(["./build/race_eval", "--bots", "policy",
                        "--policy", f"{tmp}/{tag}.rpo",
                        "--tracks", str(h["n_tracks"]),
                        "--track-start", str(h["track_start"]),
                        "--episodes", str(args.eval_episodes), "--threads", "10",
                        "--frameskip", str(man["frameskip"]),
                        "--max-steps", str(man["max_episode_steps"]),
                        "--json", f"{tmp}/{tag}.json"], capture_output=True)
        results[tag] = read_metrics(f"{tmp}/{tag}.json")
        budgets[tag] = {"real": N, "imagined": updates * batch * H}

    # --- table ---------------------------------------------------------------
    print(f"\n{'='*88}\n SAMPLE BUDGET -- online on holdout tracks\n{'='*88}")
    print(f"{'arm':<14}{'real':>10}{'imagined':>12}{'ratio':>8}"
          f"{'return':>9}{'laps':>8}{'crash%':>8}{'lap%':>8}{'cover':>8}")
    for k in ("A_bc_N", "B_dream_1x", "C_dream_10x", "D_bc_10N"):
        if k not in results:
            continue
        m, b = results[k], budgets[k]
        ratio = b["imagined"] / b["real"] if b["real"] else 0
        print(f"{k:<14}{b['real']:>10,}{b['imagined']:>12,}{ratio:>7.1f}x"
              f"{m['return']:>9.3f}{m['laps']:>8.3f}{100*m['crash']:>8.1f}"
              f"{100*m['lap_pct']:>8.1f}{m['coverage']:>8.3f}")

    Path(args.out).write_text(json.dumps(
        {"results": results, "budgets": budgets}, indent=2))
    print(f"\nwrote {args.out}")

    a, d = results["A_bc_N"]["return"], results["D_bc_10N"]["return"]
    if "B_dream_1x" in results:
        b = results["B_dream_1x"]["return"]
        gap = (b - a) / max(d - a, 1e-9)
        print(f"\n  imagination at 1:1 closed {100*gap:.0f}% of the gap between "
              f"{args.n:,} and {args.mult}x real data")


if __name__ == "__main__":
    main()
