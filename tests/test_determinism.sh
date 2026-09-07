#!/bin/bash
# Phase 0 determinism gate.
#
# Upstream whisker_racer seeded the process-global libc RNG (srand(rng + i))
# and drew every episode start from the shared rand() stream, so trajectories
# depended on env construction order and on thread interleaving. Race gives
# each env two private PCG32 streams keyed on (seed, env_idx).
#
# What must hold:
#   1. thread count never changes results
#   2. reruns are bit-identical
#   3. a different seed gives different trajectories
#   4. different env_idx values give genuinely different tracks
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/race_eval"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -x "$BIN" ] || { echo "build first: tools/build.sh eval" >&2; exit 1; }

ARGS="--bots whisker,centerline,random --tracks 24 --episodes 3 --max-steps 1500"
fail=0
check() {
    if [ "$1" = "$2" ]; then echo "  PASS  $3"; else echo "  FAIL  $3"; fail=1; fi
}
check_ne() {
    if [ "$1" != "$2" ]; then echo "  PASS  $3"; else echo "  FAIL  $3"; fail=1; fi
}

echo "== 1. thread count does not affect results =="
$BIN $ARGS --seed 42 --threads 1  --hash > "$TMP/t1"
$BIN $ARGS --seed 42 --threads 4  --hash > "$TMP/t4"
$BIN $ARGS --seed 42 --threads 16 --hash > "$TMP/t16"
h1=$(shasum "$TMP/t1"  | cut -d' ' -f1)
h4=$(shasum "$TMP/t4"  | cut -d' ' -f1)
h16=$(shasum "$TMP/t16" | cut -d' ' -f1)
check "$h1" "$h4"  "1 thread == 4 threads"
check "$h1" "$h16" "1 thread == 16 threads"

echo "== 2. reruns are bit-identical =="
$BIN $ARGS --seed 42 --threads 8 --hash > "$TMP/r1"
$BIN $ARGS --seed 42 --threads 8 --hash > "$TMP/r2"
check "$(shasum "$TMP/r1" | cut -d' ' -f1)" "$(shasum "$TMP/r2" | cut -d' ' -f1)" "rerun reproducible"

echo "== 3. a different seed changes trajectories =="
$BIN $ARGS --seed 43 --threads 8 --hash > "$TMP/s43"
check_ne "$(shasum "$TMP/r1" | cut -d' ' -f1)" "$(shasum "$TMP/s43" | cut -d' ' -f1)" "seed 42 != seed 43"

echo "== 4. env_idx gives distinct tracks =="
# Upstream overwrote env->rng from config for every env, so all envs shared one
# track. Distinct per-track hashes are the direct regression test for that.
uniq_tracks=$(grep '^whisker ' "$TMP/r1" | awk '{print $3}' | sort -u | wc -l | tr -d ' ')
check "$uniq_tracks" "24" "24 tracks produce 24 distinct trajectories"

echo "== 5. a learned policy is thread-invariant too =="
# Regression test for a real bug: policy activations were scratch buffers
# inside the shared Policy struct, so workers clobbered each other's forward
# pass. Nothing crashed -- the policy just silently scored 20x worse on 8
# threads than on 1, and no existing gate noticed because none ran a policy.
POL="$ROOT/checkpoints/bc/whisker_only.rpo"
if [ -f "$POL" ]; then
    PARGS="--bots policy --policy $POL --tracks 8 --track-start 112 --episodes 5"
    $BIN $PARGS --threads 1 --hash > "$TMP/p1"
    $BIN $PARGS --threads 8 --hash > "$TMP/p8"
    check "$(shasum "$TMP/p1" | cut -d' ' -f1)" "$(shasum "$TMP/p8" | cut -d' ' -f1)" \
          "policy: 1 thread == 8 threads"
else
    echo "  SKIP  no policy at $POL (train one: python3 -m racewm.train_bc)"
fi

echo
if [ "$fail" -eq 0 ]; then echo "determinism: ALL PASS"; else echo "determinism: FAILURES"; fi
exit $fail
