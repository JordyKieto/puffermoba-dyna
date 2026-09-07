#!/bin/bash
# Is the holdout split difficulty-matched to train?
#
# It was not, in the first corpus: holdout scored 1.4x higher than train for
# every agent, so every absolute holdout number in the project was inflated.
# Agent-vs-agent comparisons stay valid (same tracks for all), but a holdout
# figure read on its own overstates skill. Scripted bots are the right probe --
# they are fixed, so any difference is the tracks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="${1:-$ROOT/data/corpus}"
python3 - "$CORPUS" <<'PY' > /tmp/_split.txt
import json,sys
m=json.load(open(sys.argv[1]+"/manifest.json"))["splits"]
for k in ("train","val","holdout"):
    print(k, m[k]["track_start"], m[k]["n_tracks"])
PY
printf "%-12s %-10s %10s %10s\n" split bot return "surv%"
while read -r name start n; do
  for bot in centerline whisker; do
    printf "%-12s %-10s " "$name" "$bot"
    "$ROOT/build/race_eval" --bots "$bot" --tracks "$n" --track-start "$start" \
      --episodes 40 --threads 10 2>/dev/null \
      | awk -v b="$bot" '$1==b{printf "%10s %10s\n",$3,$5; exit}'
  done
done < /tmp/_split.txt
echo
echo "train and holdout should be close. A large gap means the holdout is a"
echo "lucky or unlucky sample and absolute numbers on it do not transfer."
