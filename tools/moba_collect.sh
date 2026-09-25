#!/bin/bash
# Collect MOBA games in parallel: one process per file, one env per process
# (MOBA's randomness is the process-global rand(), so envs cannot share one).
#
#   tools/moba_collect.sh TOTAL_GAMES [GAMES_PER_FILE] [PROCS] [OUT_DIR]
#   DENSE=1 tools/moba_collect.sh 100 13 8 data/moba/mix100
#
# Every game is --mix: radiant/dire = net/net, net/script, script/net by
# global game index. Logs go to OUT_DIR/logs. With DENSE=1 each finished log
# is replayed (verifying state hash and ego obs on every tick) into
# OUT_DIR/dense with the full state of every tick.
#
# Resumable: files are written as .tmp and renamed when complete, and complete
# files are skipped, so rerunning the same command continues where it stopped.
# Game i always gets the same seed (base seed 0), so the corpus is identical
# however it is split or resumed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL=${1:?usage: tools/moba_collect.sh TOTAL_GAMES [GAMES_PER_FILE] [PROCS] [OUT_DIR]}
PER_FILE=${2:-1000}
PROCS=${3:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
OUT=${4:-$ROOT/data/moba/corpus}
DENSE=${DENSE:-0}

mkdir -p "$OUT/logs"
[ "$DENSE" = 1 ] && mkdir -p "$OUT/dense"
cd "$ROOT"
tools/build.sh moba-collect > /dev/null
rm -f "$OUT"/logs/*.tmp "$OUT"/dense/*.tmp 2>/dev/null || true

NFILES=$(( (TOTAL + PER_FILE - 1) / PER_FILE ))
for ((i = 0; i < NFILES; i++)); do
    first=$(( i * PER_FILE ))
    n=$(( TOTAL - first < PER_FILE ? TOTAL - first : PER_FILE ))
    name=$(printf "games.%07d" "$first")
    if [ ! -f "$OUT/logs/$name.mbl" ] || { [ "$DENSE" = 1 ] && [ ! -f "$OUT/dense/$name.mbs" ]; }; then
        echo "$first $n $name"
    fi
done | xargs -P "$PROCS" -L 1 sh -c '
    log="'"$OUT"'/logs/$2.mbl"
    [ -f "$log" ] || build/moba_collect collect --mix --first-game "$0" --games "$1" --out "$log" >> "'"$OUT"'/collect.log"
    if [ "'"$DENSE"'" = 1 ]; then
        build/moba_collect replay "$log" --dense "'"$OUT"'/dense/$2.mbs" >> "'"$OUT"'/replay.log"
    fi'
echo "done: $(ls "$OUT"/logs/games.*.mbl | wc -l | tr -d ' ') logs in $OUT/logs"
[ "$DENSE" = 1 ] && echo "      $(ls "$OUT"/dense/games.*.mbs | wc -l | tr -d ' ') dense in $OUT/dense"
exit 0
