# Counterfactual amplification with the real sim

The recorded corpus shows one outcome per state: what happened under the
actions that were actually taken. The world model has to learn what each action
*causes*, but a recording never shows what would have happened instead.

We have the real simulator, so we can ask. From any recorded state we fork the
game and replay the same tick with a different action. The corpus never grows;
it is **amplified**. Every recorded state becomes one real transition plus K
counterfactual ones, each labelled by the real game rules.

## How it works

```
recorded log (seed + actions, a few KB per game)
   │  replayed in C from the start (bit-exact with the corpus)
   ▼
at every fork tick t (t % stride == phase):
   capture S_t, plus the `burn` real states before it (to warm the WM's memory)
   sibling 0: the recorded actions      -> real S_{t+1}
   sibling k: ONE edit (one hero, one action head, a new value)
              restore S_t into a scratch game, step once -> counterfactual S'_{t+1}
```

- **Code:**
  - `moba/moba_fork.c`: the generator, built with `tools/build.sh moba-fork` as
    `build/libmobafork.{so,dylib}`.
  - `mobawm/wm/cf.py`: the ctypes wrapper and the multi-process stream.
  - `CFBatcher` in `mobawm/wm/train.py` (`--cf`): the trainer side.
- **Edits:** the action head is chosen with weights, move y/x 2 : 2, attack
  mode 2, Q/W/E 3 each. Then the hero is chosen. For an ability edit it's a
  living hero whose ability is off cooldown, since an edit on a cooling-down
  ability changes nothing. The new value is uniform over the head's other
  values.
- **Nothing is stored.** The sim is deterministic, so (game, tick, edit seed)
  regenerates any sample. Workers replay games forever, reshuffled each pass,
  with a random fork phase each time, so every recorded tick is eventually a
  fork point.
- **Parallelism is by process.** The game's random stream is a process-global
  in `moba/moba_game.h`, so threads would share it. Each worker process writes
  into its own shared-memory slabs, and the trainer copies finished slabs to
  the GPU.

## Correctness (checked, 2026-09-25)

`python -m mobawm.wm.cf <log> <dense>`:

| check | result |
|---|---|
| sibling 0 through the fork path (restore + step) equals the real step, byte for byte | passed, every fork point |
| fork inputs equal the recorded dense corpus | identical |
| real next states equal the corpus | identical |
| burn-in history equals the corpus | identical |

The fork path skips the ego observations, which nothing in a step reads. The
byte-for-byte check above is what shows that this is safe.

## Throughput (one core, Apple M-series, warm)

| stride | K | samples/s per core |
|---|---|---|
| 8 | 7 | ~33k |
| 2 | 7 | ~50k |

An H200 pod has about 12–16 cores, so roughly 0.4–0.6M counterfactual samples/s
in total. That is more than the GPU can train on, so the model, not the sim, is
the bottleneck, and batches can be as large as VRAM allows.

## What the counterfactuals contain (one game, K=7)

About 32% of counterfactual siblings have a next state that differs from the
real one. By edited head:

| edited head | move y | move x | attack mode | Q | W | E |
|---|---|---|---|---|---|---|
| next state changes | 84% | 86% | 1% | 18% | 19% | 13% |

- **Siblings with no change are kept.** "Pressing Q here does nothing" is
  information too.
- **Attack-mode edits almost never change the next tick.** Their effect is
  delayed; multi-tick branches would capture it, and are listed under next
  steps.

## Training (`--cf`)

For each fork point:
1. The model warms its memory on the 4 real states before S_t.
2. It steps S_t under all K+1 action sets at once: the siblings share the state
   and the memory.
3. Each sibling is scored with the normal `step_loss` against the sim's true
   next state.

This loss is added to the regular training loss (rollout lanes with
`--rollout`), weighted by `--cf-weight`.

| flag | default | meaning |
|---|---|---|
| `--cf` | off | enable |
| `--cf-forks` | 1024 | fork points per update; samples = forks × (K+1) |
| `--cf-k` | 7 | edits per fork point |
| `--cf-stride` | 8 | fork every N ticks of a replayed game |
| `--cf-workers` | cores − 2 | sim processes |
| `--cf-weight` | 1.0 | weight of the counterfactual loss |
| `--cf-logs` | `<data>/../logs` | the recorded logs to replay (training games only) |

Only the training games are ever forked. The held-out games are never touched.

## Second pod

`POD=<name>` selects a separate pod state file, so several pods can run at
once. Without it, `tools/pod.py` works on the original pod exactly as before.

```sh
POD=cf .venv/bin/python tools/pod.py create --confirm --name moba-cf
POD=cf .venv/bin/python tools/pod.py bootstrap
POD=cf .venv/bin/python tools/pod.py data --corpus mix1k    # S3-only corpora are pulled directly
POD=cf .venv/bin/python tools/pod.py ssh -- "cd /workspace/puffertank && tools/build.sh moba-fork"
POD=cf .venv/bin/python tools/pod.py train <name> -- ... --cf
```

## Next steps

- **Multi-tick branches:** the edit at t, then the recorded actions for a few
  more ticks. This captures delayed effects: damage landing, attacks, deaths,
  tower hits.
- **A difference loss** between siblings: (predicted CF − predicted real) vs
  (true CF − true real).
- **Error-driven fork sampling:** more forks where the model is currently
  wrong.
- **Labelling the model's own drifted states with the sim** (DAgger). This
  needs a predicted state snapped to a valid game state before it can be
  restored.
