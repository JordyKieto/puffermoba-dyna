#!/bin/bash
# Build the Race corpus as three PHYSICALLY SEPARATE splits over disjoint
# track ranges.
#
#   TRACK SPACE (deterministic, seed 42)
#     train    env_idx   0.. 95   96 tracks   -> data/<name>/train_*
#     val      env_idx  96..111   16 tracks   -> data/<name>/val_*
#     holdout  env_idx 112..127   16 tracks   -> data/<name>/holdout_*
#
# Separate files, not a runtime filter. A split done in the loader is one
# refactor away from leaking; disjoint --track-start ranges cannot leak, and
# a shard's own env_idx values prove which split it belongs to.
#
# Track identity determines the dynamics (geometry, start poses), so splitting
# by track is the only split that measures generalization. Splitting by row
# would put step t in train and step t+1 in val.
#
#   train    fit the world model / BC policy
#   val      model selection, early stopping, hyperparameters
#   holdout  touched ONLY for the final number. Used for both offline rollout
#            eval and online eval (running a policy on unseen tracks).
#
# Each split contains four slices:
#   expert   centerline, no noise    strong, clean          -> BC + WM
#   whisker  whisker,    no noise    sensor-limited         -> BC + WM
#   noisy    centerline, eps 0.15    action coverage        -> WM only
#   random   uniform actions         action coverage        -> WM only
#
# Noise is quarantined rather than mixed into the BC slices: at frameskip 4 one
# random action swings the heading 18 degrees, usually fatally. Measured, eps
# 0.02 took centerline from 5.6 laps to 1.9; eps 0.08 took whisker from 2.6 to
# 0.55. Cloning that would mean cloning a policy far worse than the one we have.
#
#   tools/make_dataset.sh [NAME] [EPISODES] [TRAIN_N] [VAL_N] [HOLDOUT_N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-corpus}"
EPISODES="${2:-20}"
DIR="$ROOT/data/$NAME"
SEED="${SEED:-42}"
FRAMESKIP="${FRAMESKIP:-4}"
MAXSTEPS="${MAXSTEPS:-2000}"
THREADS="${THREADS:-10}"

# Track counts are arguments now. The first corpus used 96/16/16 and the
# 16-track holdout turned out to be 1.4x EASIER than train for every agent
# (centerline 7.49 vs 10.22, whisker 4.03 vs 5.80) -- a small-sample artifact
# that inflated every absolute number in the project. More holdout tracks
# shrinks that bias; tools/check_split_difficulty.sh measures what remains.
# $1 NAME, $2 EPISODES, then the track counts.
TRAIN_N="${3:-480}"
VAL_N="${4:-80}"
HOLDOUT_N="${5:-80}"
TRAIN_START=0
VAL_START=$TRAIN_N
HOLDOUT_START=$((TRAIN_N + VAL_N))

BIN="$ROOT/build/race_eval"
[ -x "$BIN" ] || { echo "build first: tools/build.sh eval" >&2; exit 1; }
rm -rf "$DIR"; mkdir -p "$DIR"

gen() {  # split start n episodes
    local split=$1 start=$2 n=$3 eps=$4
    echo "== $split : env_idx $start..$((start + n - 1)) ($n tracks) =="
    local common=(--seed "$SEED" --tracks "$n" --track-start "$start"
                  --frameskip "$FRAMESKIP" --max-steps "$MAXSTEPS" --threads "$THREADS")
    "$BIN" "${common[@]}" --bots centerline --episodes "$eps"          --epsilon 0.0  --dump "$DIR/${split}_expert"  | tail -1
    "$BIN" "${common[@]}" --bots whisker    --episodes "$eps"          --epsilon 0.0  --dump "$DIR/${split}_whisker" | tail -1
    "$BIN" "${common[@]}" --bots centerline --episodes $((eps / 2 + 1)) --epsilon 0.15 --dump "$DIR/${split}_noisy"   | tail -1
    "$BIN" "${common[@]}" --bots random     --episodes $((eps * 4))     --epsilon 0.0  --dump "$DIR/${split}_random"  | tail -1
    # Geometry is identical across slices (same seed, same env_idx). Keep one.
    mv "$DIR/${split}_expert.tracks.bin" "$DIR/${split}.tracks.bin"
    rm -f "$DIR/${split}"_*.tracks.bin
}

gen train   $TRAIN_START   $TRAIN_N   "$EPISODES"
gen val     $VAL_START     $VAL_N     "$EPISODES"
gen holdout $HOLDOUT_START $HOLDOUT_N "$EPISODES"

cat > "$DIR/manifest.json" <<JSON
{
  "name": "$NAME",
  "seed": $SEED,
  "frameskip": $FRAMESKIP,
  "max_episode_steps": $MAXSTEPS,
  "episodes_per_track": $EPISODES,
  "splits": {
    "train":   {"track_start": $TRAIN_START,   "n_tracks": $TRAIN_N},
    "val":     {"track_start": $VAL_START,     "n_tracks": $VAL_N},
    "holdout": {"track_start": $HOLDOUT_START, "n_tracks": $HOLDOUT_N}
  },
  "slices": {
    "expert":  {"bot": "centerline", "epsilon": 0.0,  "use": "bc+wm"},
    "whisker": {"bot": "whisker",    "epsilon": 0.0,  "use": "bc+wm"},
    "noisy":   {"bot": "centerline", "epsilon": 0.15, "use": "wm"},
    "random":  {"bot": "random",     "epsilon": 0.0,  "use": "wm"}
  },
  "generated_by": "tools/make_dataset.sh"
}
JSON

echo; echo "manifest: $DIR/manifest.json"; du -sh "$DIR"
