#!/bin/bash
# E40 endpoint: every frozen-WM arm's final policy (update 40) on the holdout seeds, after queues 1-3.
cd /Users/jordy/Documents/dev/puffertank
while pgrep -f "paper_evals[23]*.sh" >/dev/null; do sleep 20; done
for arm in fz_e38 fz_e38s1 fz_dyna fz_dynas1; do
  until .venv/bin/python - "$arm" <<'PY'
import sys, torch; sys.path.insert(0, "tools"); import pod
arm = sys.argv[1]; c, b = pod.s3()
p = f"results/moba/dream/{arm}_policies/final.pt"
c.download_file(b, f"moba/runs/{arm}_r0/policy.pt", p)
u = torch.load(p, map_location="cpu", weights_only=False)["update"]
print(arm, "update", u); sys.exit(0 if u >= 40 else 1)
PY
  do sleep 60; done
  env POD=evc5 .venv/bin/python -u scripts/moba/seed_eval.py --label E40_${arm}_u40 --policy results/moba/dream/${arm}_policies/final.pt --set holdout --note "E40 frozen-WM control, final policy (40 updates)"
done
echo QUEUE4 DONE
