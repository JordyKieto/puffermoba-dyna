#!/bin/bash
# Phase 0 gates. Everything the contract claims, checked.
#
#   tests/run_all.sh            # fast gates (~1 min)
#   tests/run_all.sh --full     # adds the Markov gate (~7 min, trains 16 MLPs)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

results=()
run() {
    local name="$1"; shift
    echo
    echo "───────────────────────────────────────────────────────────"
    echo " $name"
    echo "───────────────────────────────────────────────────────────"
    if "$@"; then results+=("PASS  $name"); else results+=("FAIL  $name"); fi
}

echo "Building..."
bash tools/build.sh eval >/dev/null || { echo "build failed"; exit 1; }
bash tools/build.sh play >/dev/null || { echo "build failed"; exit 1; }
echo "  build/race_eval, build/race_play"

run "contract drift (race.h vs contract.py)" python3 tests/test_contract.py
run "determinism"                            bash tests/test_determinism.sh
run "replay round trip"                      bash tests/test_replay.sh
run "checkpoint save/load/resume"            python3 tests/test_checkpoint.py

if [ "$FULL" -eq 1 ]; then
    echo
    echo "Generating dataset for the Markov gate..."
    mkdir -p data
    ./build/race_eval --bots whisker,centerline,random --tracks 32 --episodes 40 \
        --frameskip 4 --max-steps 2000 --epsilon 0.08 --threads 10 \
        --dump data/gate >/dev/null
    run "markov gate" python3 -m racewm.markov \
        --shards 'data/gate.*.bin' --tracks-file data/gate.tracks.bin \
        --epochs 50 --max-windows 120000 --json data/markov.json
else
    echo
    echo "(skipping the Markov gate; use --full to include it)"
fi

echo
echo "═══════════════════════════════════════════════════════════"
fail=0
for r in "${results[@]}"; do
    echo "  $r"
    [[ "$r" == FAIL* ]] && fail=1
done
echo "═══════════════════════════════════════════════════════════"
[ "$fail" -eq 0 ] && echo "phase 0: ALL GATES PASS" || echo "phase 0: FAILURES"
exit $fail
