"""Write samples/moba/: small committed excerpts WITH their field names, so the data
is legible without the loaders.

  state_excerpt.{npz,json}           16 ticks of held-out game 900 from tick 500: full state + all 10 actions
  counterfactual_excerpt.{npz,json}  one fork point (game 0 of mix100, a training game): the real state,
                                     8 action sets (recorded + 7 single edits) and the real sim's next state
                                     for each (moba/moba_fork.c)

    .venv/bin/python scripts/moba/make_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mobawm import shards as S  # noqa: E402

OUT = ROOT / "samples" / "moba"
ENTITY_SLOTS = {"heroes": [0, 9], "creeps": [10, 109], "neutrals": [110, 181], "towers": [182, 203],
                "ancients": [204, 205]}
HEADS = ["move_y (0..6, 3 = still)", "move_x (0..6, 3 = still)", "target mode (0..2)", "Q (0/1)", "W (0/1)", "E (0/1)"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = S.load_dense(ROOT / "data/moba/mix1k/dense/games.0000900.mbs")
    r = d.load_game(0)
    sl = slice(500, 516)
    np.savez_compressed(OUT / "state_excerpt.npz", entity=r["state"]["entity"][sl].astype(np.float32),
                        actions=r["actions"][sl], tick=r["state"]["tick"][sl], creep_idx=r["state"]["creep_idx"][sl])
    (OUT / "state_excerpt.json").write_text(json.dumps({
        "source": "data/moba/mix1k/dense/games.0000900.mbs, game 0 (global game 900, held out from every mix1k run), "
                  "records 500..515; regenerate: .venv/bin/python scripts/moba/make_samples.py",
        "arrays": {"entity": "(16, 206, 26) float32: [tick, entity slot, field]",
                   "actions": "(16, 10, 6) uint8: [tick, hero, head] -- the action taken FROM that tick",
                   "tick": "(16,) game tick", "creep_idx": "(16,) next creep slot a wave reuses"},
        "entity_fields": list(S.F), "entity_slots": ENTITY_SLOTS, "action_heads": HEADS,
        "hero_classes_by_slot": "slot % 5: 0 support, 1 assassin, 2 burst, 3 tank, 4 carry; slots 0-4 radiant, 5-9 dire",
    }, indent=1))

    from mobawm.wm import cf
    games = cf.load_log_games([ROOT / "data/moba/mix100/logs/games.0000000.mbl"])[:1]
    out, n = cf.generate(games, stride=97, K=7, check=True, edit_seed=1)
    k = min(3, n - 1)
    np.savez_compressed(OUT / "counterfactual_excerpt.npz", ent_in=out["ent_in"][k], ent_next=out["ent_next"][k],
                        codes=out["codes"][k], actions=S.unpack_actions(out["codes"][k]),
                        edit_hero=out["edit_hero"][k], edit_head=out["edit_head"][k],
                        tick_in=out["tick_in"][k], tick_next=out["tick_next"][k])
    changed = np.abs(out["ent_next"][k][1:] - out["ent_next"][k][:1]).sum((1, 2)) > 0
    (OUT / "counterfactual_excerpt.json").write_text(json.dumps({
        "source": f"moba/moba_fork.c via mobawm/wm/cf.py: mix100 log game 0 (a TRAINING game), fork at record "
                  f"{int(out['t'][k])}, K=7, edit_seed=1, check=1 (sibling 0 through the fork path reproduced the "
                  "real step byte-for-byte)",
        "arrays": {"ent_in": "(206, 26) the real state S_t",
                   "ent_next": "(8, 206, 26) next state per sibling; sibling 0 = the recorded actions (the real game)",
                   "actions": "(8, 10, 6) action heads per sibling", "codes": "(8, 10) packed action codes",
                   "edit_hero": "(8,) hero whose action was edited (-1 for sibling 0)",
                   "edit_head": "(8,) edited head 0..5 (-1 for sibling 0)"},
        "siblings_whose_next_state_differs_from_real": [bool(x) for x in changed],
        "entity_fields": list(S.F), "entity_slots": ENTITY_SLOTS, "action_heads": HEADS,
    }, indent=1))
    for f in sorted(OUT.iterdir()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
