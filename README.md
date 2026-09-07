# Race World Model

Building a world model over PufferLib Ocean's **Race** environment
(`ocean/whisker_racer`), starting from a Markovian, deterministic, replayable
state contract.

This repo is also the original [PufferTank](https://puffer.ai) dev container —
`puffertank.dockerfile`, `docker.sh`, `install.sh` are unchanged from upstream
and still work as before.

---

## Where things stand

**Phase 0 — audit, lock down, pin the contract.** Complete. All gates pass.

| gate | what it proves |
|---|---|
| contract drift | `race/race.h` and `racewm/contract.py` cannot disagree |
| determinism | thread count is irrelevant; reruns are bit-identical; N seeds give N distinct tracks |
| replay | client and server reproduce the same trajectory bit-exactly; tampering is detected |
| checkpoint | resume reproduces an uninterrupted run exactly, optimizer and scheduler included |
| Markov | with track context, history buys **+4.0%** — the state is sufficient. Control confirms the test has power. |

**Phase 1 — MLP world model.** Not started.
**Phase 2 — BC corpus, Dreamer, JEPA.** Not started.

## Quick start

```bash
git submodule update --init --recursive
tools/build.sh eval          # headless evaluator
tools/build.sh play          # replay client
tests/run_all.sh             # fast gates (~1 min); --full adds the Markov gate
```

### Evaluate bots at scale

```bash
build/race_eval --bots whisker,centerline,random,straight \
                --tracks 32 --episodes 40 --threads 10
```

Headless, no Python, no window. 231k transitions in ~0.5s on an M-series Mac.
Reports behavioral metrics, not just return: laps, crash rate and cause,
track coverage, centerline deviation, steering reversal rate, minimum whisker
clearance, steps-to-crash. `--json PATH` for machine-readable output.

```
bot              eps   return   length    laps   crash    cover     dist    revrs   minwhk
whisker         1280    8.468    286.4   2.646   0.875    0.741    11.30    0.462    0.111
centerline      1280   17.945    567.2   5.608   0.512    0.835    12.44    0.120    0.132
random          1280    0.203      7.1   0.063   1.000    0.123    13.95    0.262    0.117
```

### Generate a dataset

```bash
build/race_eval --bots whisker,centerline,random --tracks 32 --episodes 40 \
                --epsilon 0.08 --dump data/train
```

Writes sharded transitions plus `data/train.tracks.bin` (per-track geometry).
Read them with `racewm.shards.load()`. Shards are self-describing — the schema
hash, resolution, and frameskip travel with the data.

### Record and replay

```bash
build/race_eval --tracks 8 --record replays/r      # ~1 byte per step
build/race_play replays/r.whisker.001.rr           # watch
build/race_play replays/r.whisker.001.rr --verify  # headless hash check
build/race_play replays/r.whisker.001.rr --mp4 out.mp4
build/race_play replays/r.whisker.001.rr --rgb-out out/run1
```

One format, client and server. Frame *t* is step *t*; capture resolution is
pinned to the env's logical size regardless of HiDPI. `--rgb-out` writes PNGs,
an index-aligned `states.bin`, and a `capture.json` carrying the verified hash.

## Layout

```
race/           the env (fork of ocean/whisker_racer) + eval, replay, capture
  race.h          dynamics, determinism, the Markov observation
  race_eval.c     headless parallel evaluator
  race_play.c     replay client: render, mp4, RGB capture
  race_replay.h   the one replay format
  race_hash.h     trajectory hashing, shared by both sides
  race_shard.h    the one dataset record format
  race_bots.h     scripted baselines
racewm/         contract, shard IO, Markov gate, checkpoints, tracking
config/race.ini
tools/build.sh  overlays race/ into the pinned submodule and builds
tests/          the phase-0 gates
docs/           AUDIT.md, CONTRACT.md
third_party/pufferlib   pinned submodule, never edited
```

`race/` is an **overlay**: `tools/build.sh` symlinks it into
`third_party/pufferlib/ocean/race` so upstream stays byte-identical and rebasing
is trivial.

## Read next

- **[docs/AUDIT.md](docs/AUDIT.md)** — what was wrong with upstream and what we
  changed. Includes `tests/upstream_probe.c`, which demonstrates against
  unmodified upstream that all envs generate one identical track and that reset
  ordering swaps episode start states.
- **[docs/CONTRACT.md](docs/CONTRACT.md)** — the 30-dim state, why track
  geometry is part of it, episode boundaries, replay, capture.

## Known limitations

- `puffer train race` needs `tools/build.sh module`, which requires OpenMP.
  Apple clang has none, so PPO is Linux/GPU-box only.
- Headless rendering on a remote box needs Xvfb in a container. Deferred:
  raylib publishes no arm64 Linux binary, so that image must build raylib from
  source. Local capture is correct and resolution-stable today.
- Upstream's pretrained `puffer_whisker_racer_weights.bin` expects `obs_size=3`
  and no longer loads. Expected — the observation is now the full state.
