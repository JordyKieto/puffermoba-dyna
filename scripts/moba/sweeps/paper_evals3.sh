#!/bin/bash
cd /Users/jordy/Documents/dev/puffertank
while pgrep -f "paper_evals[2]*.sh" >/dev/null; do sleep 20; done
E="env POD=evc5 .venv/bin/python -u scripts/moba/seed_eval.py"
$E --label e14_armC --policy runs/dream_ppo/p7/puffer_ours/policy.pt --set holdout --note "E14 arm C: recurrent PufferPolicy, dream PPO with the game's reward (short episodes)"
$E --label e36_J_envs128_4x_s2 --policy results/moba/dream/g3/J_envs128_4x_s2/policy.pt --set holdout --note "E36 best: + long episodes + win reward, calibrated dream"
$E --label e37_pf1_u7 --policy results/moba/dream/pf1_policies/policy_u0007.pt --set holdout --note "E37: + 5-pod PPO fleet, best real checkpoint"
echo QUEUE3 DONE
