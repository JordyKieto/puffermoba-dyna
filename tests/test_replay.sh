#!/bin/bash
# Replay must round-trip bit-exactly between the headless server and the client.
#
# The client and server share race_hash.h and race_replay.h, so this checks the
# thing that actually matters: that stepping under the renderer produces the
# same trajectory as stepping headlessly. Upstream could not have offered this
# at all -- episode starts came from a global rand() stream, so the same
# actions replayed into a different state.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0

for b in build/race_eval build/race_play; do
    [ -x "$ROOT/$b" ] || { echo "build first: tools/build.sh ${b##*race_}" >&2; exit 1; }
done

echo "== record =="
"$ROOT/build/race_eval" --bots whisker,centerline,random --tracks 6 --episodes 3 \
    --frameskip 4 --max-steps 1500 --threads 4 --record "$TMP/r" --record-episodes 2 >/dev/null
n=$(ls "$TMP"/*.rr 2>/dev/null | wc -l | tr -d ' ')
echo "  wrote $n replays"
[ "$n" -eq 18 ] && echo "  PASS  one replay per (bot, track)" \
                || { echo "  FAIL  expected 18 replays, got $n"; fail=1; }

echo "== verify each replay reproduces its recorded hash =="
bad=0
for f in "$TMP"/*.rr; do
    if ! "$ROOT/build/race_play" "$f" --verify >/dev/null 2>&1; then
        echo "  FAIL  $(basename "$f")"; bad=$((bad+1))
    fi
done
[ "$bad" -eq 0 ] && echo "  PASS  all $n replays match" \
                 || { echo "  FAIL  $bad replay(s) diverged"; fail=1; }

echo "== replays are compact =="
# One action byte per step plus a fixed header; anything larger means state
# leaked into the format.
big=$(find "$TMP" -name '*.rr' -size +8k | wc -l | tr -d ' ')
[ "$big" -eq 0 ] && echo "  PASS  every replay under 8KB" \
                 || { echo "  FAIL  $big replay(s) over 8KB"; fail=1; }

echo "== a corrupted replay is rejected, not silently played =="
cp "$(ls "$TMP"/*.rr | head -1)" "$TMP/corrupt.rr"
# Flip an action byte well past the header.
python3 - "$TMP/corrupt.rr" <<'PY'
import sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
b[-5] = (b[-5] + 1) % 3
open(p,'wb').write(b)
PY
if "$ROOT/build/race_play" "$TMP/corrupt.rr" --verify >/dev/null 2>&1; then
    echo "  FAIL  tampered replay reported a match"; fail=1
else
    echo "  PASS  tampered replay reports MISMATCH"
fi

echo
[ "$fail" -eq 0 ] && echo "replay: ALL PASS" || echo "replay: FAILURES"
exit $fail
