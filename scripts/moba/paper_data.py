"""Collect every source the MOBA paper uses into results/moba/paper/ and compute every reported number.
Re-run after any new evaluation: the notebook (docs/moba/paper/figures.ipynb) and the paper read only these files.

  results/moba/paper/
    online_fixed.csv      every real-game evaluation after a PPO update (fixed seeds 50000000+, 60 per side)
    dream.csv             rank 0's dream-side log per PPO update (reward, XP, deaths, towers, KL, entropy, ...)
    wm_train.csv          world-model rank 0: train loss, live loss (the policy's newest real games), val loss
    wm_tower.csv          world-model 100-game tower evaluation per checkpoint (held-out corpus games)
    dyna_tower.csv        each world-model version on the CURRENT policy's real games (none trained on them)
    collector.csv         every real game the collector played (policy update, side, winner, length)
    seed_evals.jsonl      fresh / holdout evaluations (scripts/moba/seed_eval.py)  [written by that script]
    numbers.json          the headline and every number the paper quotes, with 95% Wilson intervals

usage: python scripts/moba/paper_data.py [--no-remote]   (--no-remote: skip S3/pod downloads, recompute only)"""
import argparse, csv, json, math, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
D = ROOT / "results/moba/dream"
OUT = ROOT / "results/moba/paper"
RAW = OUT / "raw"
PPO_RUNS = {"ppol": "lr 3e-4, death penalty on, no KL", "ppol2": "lr 5e-5, death penalty on, no KL",
            "ppol3": "lr 5e-5, death 0, KL 1.0 to anchor", "ppol4": "lr 5e-5, death 0, KL 0.3 to anchor"}
ABL = {"abl_ent0": "entropy bonus 0", "abl_death0": "death penalty 0", "abl_kl": "KL 1.0 to u71",
       "fz_e38": "frozen E38 dream (no policy games), seed 0", "fz_e38s1": "frozen E38 dream, seed 1",
       "fz_dyna": "frozen Dyna WM (wml2 @3400), seed 0", "fz_dynas1": "frozen Dyna WM, seed 1"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def diff_ci(k1, n1, k2, n2, z=1.96):
    """Newcombe's hybrid score interval for p1 - p2."""
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_csv(name, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "update", "it"), k))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    return len(rows)


def remote():
    import pod
    c, b = pod.s3()
    RAW.mkdir(parents=True, exist_ok=True)
    for run in list(PPO_RUNS) + list(ABL):
        try:
            c.download_file(b, f"moba/runs/{run}_r0/log.jsonl", str(RAW / f"{run}_r0_log.jsonl"))
        except Exception:
            pass
    for run in ("wml_r0", "wml2_r0"):
        for o in [o for pg in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=f"moba/runs/{run}/tower_big_")
                  for o in pg.get("Contents", [])]:
            f = RAW / f"{run}_{o['Key'].rsplit('/', 1)[1]}"
            if not f.exists():
                c.download_file(b, o["Key"], str(f))
    up = {o["Key"].rsplit("/", 1)[1]: o["LastModified"].isoformat() for pg in c.get_paginator("list_objects_v2").paginate(
          Bucket=b, Prefix="moba/runs/dyna_live/meta/") for o in pg.get("Contents", [])}
    (RAW / "collector_upload_times.json").write_text(json.dumps(up, indent=0))
    # the per-update world-model logs live on rank 0 (S3 has them only every 1,000 updates)
    env = dict(os.environ, POD="dl0")
    for run in ("wml_r0", "wml2_r0"):
        subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, 'tools'); import pod; "
                        f"pod.rsync_from(pod.R_REPO + '/runs/token_wm/{run}/log.jsonl', '{RAW}/{run}_log.jsonl')"], env=env, cwd=ROOT)
    # only the small result files (the pod's dyna_eval/ also holds 2 GB of dense games)
    (RAW / "dyna_eval").mkdir(parents=True, exist_ok=True)
    code = ("import sys; sys.path.insert(0, 'tools'); import pod\n"
            "D = pod.R_REPO + '/data/moba/dyna_eval'\n"
            "for f in pod.run(f'cd {D} && ls *.json *.csv 2>/dev/null', capture=True, check=False).split():\n"
            f"    pod.rsync_from(D + '/' + f, '{RAW}/dyna_eval/' + f)\n")
    subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-remote", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_remote:
        remote()
    N = {}
    live_state = ROOT / "runs/pod/dyna_live_state.json"                # the controller's live state, else the committed copy
    if live_state.exists():
        (D / "dyna_live_state.json").write_text(live_state.read_text())
    st = json.loads((D / "dyna_live_state.json").read_text()) if (D / "dyna_live_state.json").exists() else {}

    # online: fixed seeds after every update
    rows = []
    for run, setting in {**PPO_RUNS, **ABL}.items():
        for r in jl(D / f"{run}_eval.jsonl"):
            if "radiant_wins" not in r:
                continue
            rows.append(dict(run=run, setting=setting, update=r["update"], wm=r.get("wm"), radiant_wins=r["radiant_wins"],
                             dire_wins=r.get("dire_wins", 0), radiant_towers=r.get("radiant_towers_ours"),
                             radiant_kills=r.get("radiant_kills"), radiant_deaths=r.get("radiant_deaths"),
                             radiant_ticks=r.get("radiant_ticks"), dire_towers=r.get("dire_towers_ours"), t=r.get("t")))
    N["online_rows"] = write_csv("online_fixed.csv", rows)

    # dream side
    rows = []
    for run in list(PPO_RUNS) + list(ABL):
        for r in jl(RAW / f"{run}_r0_log.jsonl"):
            rows.append(dict(run=run, update=r["update"], agent_steps=r["agent_steps"], sps=r.get("sps"),
                             reward_per_1k=r.get("reward_per_1k"), xp_per_1k=r.get("xp_per_1k"), deaths_per_1k=r.get("deaths_per_1k"),
                             ours_towers_per_episode=r.get("ours_towers_per_episode"), opp_towers_per_episode=r.get("opp_towers_per_episode"),
                             approx_kl=r.get("approx_kl"), clip_frac=r.get("clip_frac"), entropy=r.get("entropy"),
                             kl_anchor=r.get("kl_anchor"), explained_var=r.get("explained_var"), wm=r.get("wm")))
    N["dream_rows"] = write_csv("dream.csv", rows)

    # drift diagnosis (ppol2, updates 27+): real behaviour per update (per-game arena records; their files were
    # overwritten per update number by later runs, so the ppol2 ones are picked by time once, into drift.csv) + dream metrics
    dl = {r["update"]: r for r in jl(D / "drift/ppol2_r0_log.jsonl")}
    if not (OUT / "drift.csv").exists() and (D / "drift/eval_live").exists():
        import time as _t
        cut = _t.mktime(_t.strptime("2026-10-05 10:54", "%Y-%m-%d %H:%M"))
        rows = []
        for f in sorted((D / "drift/eval_live").glob("arena_u*.jsonl")):
            u = int(f.stem.split("_u")[1])
            if os.path.getmtime(f) < cut or u < 27 or u not in dl:
                continue
            g = [x for x in jl(f) if x["side"] == "radiant"]
            m = lambda k: sum(x[k] for x in g) / len(g)
            rows.append(dict(run="ppol2", update=u, real_wins=sum(x["won"] for x in g), real_kills=m("ours_kills"), real_deaths=m("ours_deaths"),
                             real_creeps=m("ours_creeps_killed"), real_ticks=m("ticks"), real_towers=5 * m("ours_towers_killed"),
                             net_level=m("theirs_level"), **{f"dream_{k}": dl[u].get(k) for k in (
                                 "reward_per_1k", "xp_per_1k", "deaths_per_1k", "ours_towers_per_episode", "opp_towers_per_episode",
                                 "entropy", "approx_kl")}))
        N["drift_rows"] = write_csv("drift.csv", rows)

    # world model: train / live / val
    rows = []
    for run in ("wml_r0", "wml2_r0"):
        for r in jl(RAW / f"{run}_log.jsonl"):
            row = dict(run=run, it=r.get("it"), train_loss=r.get("train/loss"), live_loss=r.get("train/live/loss"),
                       val_loss=r.get("val/loss"), elapsed_s=r.get("elapsed_s"))
            if any(v is not None for k, v in row.items() if k not in ("run", "it")):
                rows.append(row)
    N["wm_rows"] = write_csv("wm_train.csv", rows)
    rows = []
    for f in sorted(RAW.glob("wml*_r0_tower_big_*.json")):
        d = json.loads(f.read_text()); run = f.name.split("_tower_big_")[0]; it = int(f.stem.split("_")[-1])
        rows.append(dict(run=run, it=it, precision=d["precision"], recall=d["recall"], siege_ratio=d["siege_ratio"],
                         pred=d["pred"], real=d["real"], games=d["games"]))
    N["wm_tower_rows"] = write_csv("wm_tower.csv", rows)
    rows = []
    for f in sorted((RAW / "dyna_eval").glob("tower_*.json")):
        d = json.loads(f.read_text())
        rows.append(dict(wm=f.stem[len("tower_"):], precision=d["precision"], recall=d["recall"], siege_ratio=d["siege_ratio"],
                         pred=d["pred"], real=d["real"], games=d["games"], windows=d["windows"],
                         precision_ci=d.get("precision_ci"), recall_ci=d.get("recall_ci")))
    N["dyna_tower_rows"] = write_csv("dyna_tower.csv", rows)

    # world-model mechanics (scripts/moba/wm_mechanics_eval.py, one file per model on the pod)
    mech = []
    for f in sorted((RAW / "dyna_eval").glob("wm_mechanics_*.csv")):
        if not f.stem.endswith("_windows"):
            mech += list(csv.DictReader(open(f)))
    if mech:
        N["wm_mechanics_rows"] = write_csv("wm_mechanics.csv", mech)

    # collector: every real game of the current policy. The games record the policy's update number but not its run;
    # the run is the one whose launch precedes the game's upload (launch times from runs/pod/<run>_launch.log, PDT)
    import datetime as dt
    launches = []
    for run in PPO_RUNS:
        lf = ROOT / f"runs/pod/{run}_launch.log"
        if lf.exists():
            ts = lf.read_text().split("launch ", 1)[1][:14]
            launches.append((dt.datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone(dt.timedelta(hours=-7))), run))
    launches.sort()
    up = {}
    if (RAW / "collector_upload_times.json").exists():
        up = json.loads((RAW / "collector_upload_times.json").read_text())
    rows = []
    for f in sorted((D / "dyna_live_meta").glob("*.json")):
        t = up.get(f.name)
        run = None
        if t:
            tt = dt.datetime.fromisoformat(t)
            run = ([r for l, r in launches if l <= tt] or [None])[-1]
        for g in json.loads(f.read_text()):
            ours = "radiant" if g["radiant"] != "net" else "dire"
            rows.append(dict(run=run, uploaded=t, game_index=g["game_index"], policy_update=g.get("policy_update"), policy_wm=g.get("policy_wm"),
                             our_side=ours, won=int(g["winner"] == (1 if ours == "radiant" else 2)),
                             lost=int(g["winner"] == (2 if ours == "radiant" else 1)), ticks=g["ticks"], file=f.name))
    N["collector_games"] = write_csv("collector.csv", rows)
    for side in ("radiant", "dire"):
        x = [r for r in rows if r["our_side"] == side]
        N[f"collector_{side}"] = dict(games=len(x), wins=sum(r["won"] for r in x), losses=sum(r["lost"] for r in x))

    # seed-set evaluations
    ev = jl(OUT / "seed_evals.jsonl")
    best = {}
    for r in ev:
        k, n = r["radiant_wins"], r["radiant_games"]
        p, lo, hi = wilson(k, n)
        r["radiant_rate"], r["radiant_ci95"] = p, [lo, hi]
        r["dire_rate"], r["dire_ci95"] = wilson(r["dire_wins"], r["dire_games"])[0], list(wilson(r["dire_wins"], r["dire_games"])[1:])
        best[(r["label"], r["set"])] = r                                   # the latest row per (label, set)
    N["seed_evals"] = list(best.values())
    hold = {l: r for (l, s), r in best.items() if s == "holdout"}
    ppo_hold = {l: r for l, r in hold.items() if r["team"] == "ppo"}
    if ppo_hold:
        top = max(ppo_hold.values(), key=lambda r: r["radiant_rate"])
        N["headline"] = dict(policy=top["label"], set="holdout", radiant_wins=top["radiant_wins"], radiant_games=top["radiant_games"],
                             radiant_rate=top["radiant_rate"], radiant_ci95=top["radiant_ci95"], dire_wins=top["dire_wins"],
                             dire_games=top["dire_games"], dire_ci95=top["dire_ci95"])
        h2 = best.get((top["label"], "holdout2"))
        if h2:                                                              # pooled over both untouched sets
            k, n = top["radiant_wins"] + h2["radiant_wins"], top["radiant_games"] + h2["radiant_games"]
            N["headline"]["pooled_untouched"] = dict(radiant_wins=k, radiant_games=n, rate=wilson(k, n)[0], ci95=list(wilson(k, n)[1:]))
        N["deltas_vs_headline"] = {}
        for l, r in hold.items():
            if l != top["label"]:
                d, lo, hi = diff_ci(top["radiant_wins"], top["radiant_games"], r["radiant_wins"], r["radiant_games"])
                N["deltas_vs_headline"][l] = dict(diff=d, ci95=[lo, hi], other_rate=r["radiant_rate"])
    # how much the REAL game was used (it never gives gradients to the policy, but it selects, gates and feeds the WM)
    online_rows = jl(D / "ppol_eval.jsonl") + jl(D / "ppol2_eval.jsonl") + jl(D / "ppol3_eval.jsonl") + jl(D / "ppol4_eval.jsonl")
    fresh_rows = jl(D / "ppol3_fresh.jsonl") + jl(D / "ppol4_fresh.jsonl")
    dream_final = {run: max((r["agent_steps"] for r in jl(RAW / f"{run}_r0_log.jsonl")), default=0) for run in PPO_RUNS}
    pods = {"ppol": 10, "ppol2": 8, "ppol3": 8, "ppol4": 8}      # PPO ranks per run (ppol launched 10 ranks, 8 ran: see E39)
    N["real_game_budget"] = dict(
        per_update_eval_games=120 * sum(1 for r in online_rows if "radiant_wins" in r),
        anchor_check_games=600 * len(fresh_rows) + 600,               # + the first fresh check of u71
        collector_games=len(rows),
        holdout_games=sum(r["radiant_games"] + r["dire_games"] for r in best.values()),
        dream_agent_steps_rank0=dream_final,
        dream_agent_steps_total_approx=sum(dream_final[r] * (8 if r != "ppol" else 8) for r in dream_final),
        note="the real game never provides policy gradients; it trains the world model (collector games) and selects "
             "checkpoints / anchors / resets (evaluation games)")
    N["anchor"] = st.get("anchor")
    N["resets"] = {k: len(v) for k, v in st.items() if k.startswith("resets")}
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print(json.dumps({k: v for k, v in N.items() if k not in ("seed_evals",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
