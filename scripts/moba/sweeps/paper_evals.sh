#!/bin/bash
# Paper evaluation queue on evc5 (one arena at a time). Appends to results/moba/paper/seed_evals.jsonl.
cd /Users/jordy/Documents/dev/puffertank
E="env POD=evc5 .venv/bin/python -u scripts/moba/seed_eval.py"
$E --label ppol4_u69 --policy results/moba/dream/ppol4_policies/policy_u0069.pt --set holdout --note "anchor since 13:42 (fresh 223/300)"
$E --label shipped_net --team net --set holdout --note "the shipped net against itself: the radiant/dire base rate"
$E --label random --team random --set holdout --note "untrained policy (random init PufferPolicy)"
$E --label e38_u0017 --policy results/moba/dream/g4_policies/all_handcrafted/policy_u0017.pt --set holdout --note "start of the Dyna loop (E38 best)"
$E --label ppol_u71 --policy results/moba/dream/ppol_policies/policy_u0071.pt --set holdout
$E --label ppol4_u10 --policy results/moba/dream/ppol4_policies/policy_u0010.pt --set holdout
$E --label ppol4_u49 --policy results/moba/dream/ppol4_policies/policy_u0049.pt --set holdout
$E --label ppol4_u69 --policy results/moba/dream/ppol4_policies/policy_u0069.pt --set holdout2 --note "second untouched seed set"
echo QUEUE DONE
