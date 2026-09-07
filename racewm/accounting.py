"""Sample accounting audit — run this BEFORE reporting a headline.

Written because I got it wrong. A dream-PPO run was reported as reaching "91%
of BC with 200k real transitions"; the `--real-budget 200000` flag capped only
the normalization statistics and the reset-anchor buffer, while the world model
it depended on had been trained on the full 817,669-transition split. The true
footprint was 1.8x *larger* than BC's, not 2.3x smaller, which inverted the
claim.

Nothing in the code was wrong. The accounting was, and no test could fail
because no test existed. This is that test.

The rule it enforces: **a method's real-sample cost is the union over every
component it depends on, not the budget flag of the last script you ran.**
Pretraining, world model, termination head, critic, calibration rollouts and
reset anchors all count. So does whether those samples needed action labels,
because labelled and unlabelled data are different currencies.

    python3 -m racewm.accounting --run ppo2_h200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# component -> (what it consumes, whether it needs EXPERT action labels)
LABELLED = {
    "bc": True,             # (state, expert_action) pairs
    "policy": True,
}


def read_config(run: str, root="checkpoints") -> dict | None:
    p = Path(root) / run / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("config", {})


def component_samples(cfg: dict) -> int:
    """Real transitions a component consumed, from its own recorded config."""
    for k in ("n_train", "n_real", "n_train_seqs", "real_budget"):
        v = cfg.get(k)
        if v:
            return int(v)
    return 0


def trace(run: str, root="checkpoints") -> list[tuple[str, int, bool, str]]:
    """Walk a run's dependency chain, accumulating real-sample cost."""
    seen, rows = set(), []
    stack = [run]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        cfg = read_config(name, root)
        if cfg is None:
            rows.append((name, 0, False, "NO MANIFEST — cost unknown"))
            continue
        n = component_samples(cfg)
        labelled = any(k in name for k in LABELLED)
        note = ""
        if cfg.get("real_budget") and cfg.get("real_budget") != n:
            note = f"real_budget={cfg['real_budget']} but n_train={n}"
        rows.append((name, n, labelled, note))
        # Follow dependencies recorded in the config.
        for key in ("wm", "init_from", "bc_init", "termination", "critic_anchor",
                    "encoder_path", "policy"):
            dep = cfg.get(key)
            if isinstance(dep, str) and dep:
                parts = Path(dep).parts
                if "checkpoints" in parts:
                    stack.append(parts[parts.index("checkpoints") + 1])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--imagined", type=int, default=None)
    ap.add_argument("--claim", type=int, default=None,
                    help="the real-sample figure you were about to publish")
    ap.add_argument("--corpus-size", type=int, default=None,
                    help="unique transitions available; components drawing from "
                         "one corpus overlap, so unique cost is capped by this")
    ap.add_argument("--root", default="checkpoints")
    args = ap.parse_args()

    rows = trace(args.run, args.root)
    print(f"sample accounting for `{args.run}`\n")
    print(f"{'component':<26}{'real samples':>14}{'labelled':>10}  note")
    tot = lab = 0
    for name, n, l, note in rows:
        print(f"{name:<26}{n:>14,}{'yes' if l else 'no':>10}  {note}")
        tot += n
        lab += n if l else 0
    print(f"{'-'*70}")
    print(f"{'SUM over components':<26}{tot:>14,}")
    print(f"{'  of which expert-labelled':<26}{lab:>14,}")
    print(f"{'  unlabelled (s,a,s)':<26}{tot-lab:>14,}")
    unique = tot
    if args.corpus_size:
        # Components drawing from the same corpus reuse the same transitions,
        # so the sum double-counts. The honest figure for "how much real data
        # did this method need" is the unique count, capped by what exists.
        unique = min(tot, args.corpus_size)
        print(f"{'UNIQUE (capped by corpus)':<26}{unique:>14,}"
              f"   sum double-counts shared data")
    tot = unique
    if args.imagined:
        print(f"{'imagined transitions':<26}{args.imagined:>14,}"
              f"   ratio {args.imagined/max(tot,1):.1f}:1")

    if args.claim is not None:
        off = tot / max(args.claim, 1)
        verdict = "MATCHES" if 0.95 <= off <= 1.05 else f"OFF BY {off:.1f}x"
        print(f"\n  claimed {args.claim:,} real samples -> {verdict}")
        if verdict != "MATCHES":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
