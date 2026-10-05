#!/bin/bash
cd /Users/jordy/Documents/dev/puffertank
while pgrep -f paper_evals.sh >/dev/null; do sleep 20; done
E="env POD=evc5 .venv/bin/python -u scripts/moba/seed_eval.py"
$E --label shipped_net --team net --set holdout --note "the shipped net against itself: the radiant/dire base rate"
$E --label random --team random --set holdout --note "untrained policy (random init PufferPolicy)"
echo QUEUE2 DONE
