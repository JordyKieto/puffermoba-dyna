"""Whole-project compute, cost and data accounting for the MOBA track -> results/moba/paper/compute.json (+ .md).

Cost: RunPod's billing API (per pod per day: amount, billed time) for the whole account. A pod is attributed to this
repository when its id appears in a pod state file (runs/pod/state_*.json*) or in a launch log's "created <id> (<gpu>)"
line; the rest of the account's spend in the same window is reported as UNATTRIBUTED (early pods of this repository
created before creation logging, and possibly other projects on the account). The billing API starts on 2026-09-22:
spend before it (the first MOBA world-model runs, PROVENANCE §3.4, ~$100, I) is listed from the earlier estimate.
Purpose of a pod comes from its name (pool prefixes); GPU type from the state file or the creation line.

Data: every MOBA corpus in s3://puffertank/moba/ (objects, bytes), with game and tick counts and the controllers that
played them (from PROVENANCE §2 and the collector's game records).
usage: python scripts/moba/compute_accounting.py"""
import collections, glob, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import pod  # noqa: E402

PURPOSE = [  # (name regex, purpose): first match wins
    (r"^(fz_)", "E40 frozen-WM control (PPO)"),
    (r"^(evc|evalcpu|cpu)", "CPU: real-game evaluation + collection"),
    (r"^dl(1[2-9])$", "Dyna loop: PPO fleet"),
    (r"^dl", "Dyna loop: world-model fleet"),
    (r"^(rw|gb|g3|g4|gen|pf|ppo|p1[67]|armc|p\d)", "dream PPO (sweeps, fleets)"),
    (r"^(cf|ft)", "world-model training / ablations"),
    (r"^(wm|fleet|e\d|g\d|s\d|sa|sb|lr|a\d|a_|c_|k\d|fp|sweep|e28|p1_|grid)", "world-model training / ablations"),
]


def price_gpu(usd_hr):
    """The hardware, inferred from the pod's hourly price when the state file does not name it (labelled as inferred)."""
    if usd_hr is None:
        return None
    for lo, hi, g in ((0.30, 0.60, "A40 (inferred)"), (0.60, 0.90, "A6000/L40S (inferred)"), (0.90, 1.10, "CPU 32 vCPU (inferred)"),
                      (1.10, 2.00, "A100 80GB (inferred)"), (2.00, 3.40, "H100 (inferred)"), (3.40, 5.00, "H200 (inferred)")):
        if lo <= usd_hr < hi:
            return g
    return f"${usd_hr}/hr"


def purpose(name):
    for rx, p in PURPOSE:
        if re.search(rx, name or ""):
            return p
    return "other (this repository)"


def main():
    out = ROOT / "results/moba/paper"
    B = pod.api("GET", "/billing/pods")
    vols = pod.api("GET", "/billing/networkvolumes")
    ids = {}                                                           # pod id -> (name, gpu)
    for f in glob.glob(str(ROOT / "runs/pod/state_*.json*")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ids[d["id"]] = (Path(f).name.split("state_")[1].split(".")[0], d.get("gpu") or price_gpu(d.get("cost_per_hr")))
    txt = subprocess.run("grep -rHoE 'created [a-z0-9]{14} \\([^)]*\\)' runs/ logs/ 2>/dev/null", shell=True, cwd=ROOT,
                         capture_output=True, text=True).stdout
    for line in txt.splitlines():
        if ":" not in line:
            continue
        fpath, m = line.split(":", 1)
        g = re.match(r"created ([a-z0-9]{14}) \(([^)]*)\)", m)
        if g and g.group(1) not in ids:
            name = Path(fpath).stem.replace("_r", "").replace("_launch", "")
            ids[g.group(1)] = (name, g.group(2))
    rows = []
    for b in B:
        name, gpu = ids.get(b["podId"], (None, None))
        rows.append(dict(day=b["time"][:10], pod=b["podId"], name=name, gpu=(gpu or "unknown").replace(" x1", ""),
                         attributed=name is not None, purpose=purpose(name) if name else "UNATTRIBUTED",
                         usd=b["amount"], hours=b["timeBilledMs"] / 3.6e6))
    agg = lambda key, rs: {k: dict(usd=round(sum(r["usd"] for r in v), 2), hours=round(sum(r["hours"] for r in v), 1), pods=len({r["pod"] for r in v}))
                           for k, v in sorted(collections.defaultdict(list, {kk: [r for r in rs if r[key] == kk] for kk in {r[key] for r in rs}}).items())}
    att = [r for r in rows if r["attributed"]]
    C = dict(
        billing_window=[min(r["day"] for r in rows), max(r["day"] for r in rows)],
        account_total_usd=round(sum(r["usd"] for r in rows), 2),
        attributed_usd=round(sum(r["usd"] for r in att), 2), unattributed_usd=round(sum(r["usd"] for r in rows if not r["attributed"]), 2),
        network_volumes_usd=round(sum(v.get("amount", 0) + v.get("highPerformanceStorageAmount", 0) for v in vols), 2),
        pre_billing_estimate=dict(usd=100, note="E1-E6 + E5-continued on H200/H100 NVL before 2026-09-22 (PROVENANCE §3.4, I)"),
        attributed_pod_hours=round(sum(r["hours"] for r in att), 1),
        attributed_gpu_hours=round(sum(r["hours"] for r in att if "cpu" not in r["gpu"].lower()), 1),
        attributed_cpu_pod_hours=round(sum(r["hours"] for r in att if "cpu" in r["gpu"].lower()), 1),
        by_purpose=agg("purpose", rows), by_gpu=agg("gpu", att), by_day=agg("day", rows),
        note="RunPod billing per pod per day; 'today' is partial. CPU pods are 32-vCPU.")
    # data inventory
    c, bk = pod.s3()
    size, objs = collections.Counter(), collections.Counter()
    for pg in c.get_paginator("list_objects_v2").paginate(Bucket=bk, Prefix="moba/"):
        for o in pg.get("Contents", []):
            p = o["Key"].split("/"); k = "/".join(p[1:3]) if p[1] == "runs" else p[1]
            size[k] += o["Size"]; objs[k] += 1
    col = list(__import__("csv").DictReader(open(out / "collector.csv"))) if (out / "collector.csv").exists() else []
    live_ticks = sum(int(r["ticks"]) for r in col)
    C["data"] = dict(
        s3_total_gb=round(sum(size.values()) / 1e9, 1),
        corpora={
            "mix1k": dict(games=1000, ticks=2663317, controllers="shipped net / scripted bots (net-net, net-script, script-net)", gb=round(size["mix1k"] / 1e9, 1)),
            "mixrand1k": dict(games=1000, ticks=2608592, controllers="random / net / scripted mixes, 3,000-tick cap", gb=round(size["mixrand1k"] / 1e9, 1)),
            "dyna1": dict(games=100, controllers="P3 dream policies vs the net", gb=round(size["dyna1"] / 1e9, 1)),
            "dyna2": dict(games=810, controllers="P3 dream policies vs random / net / each other", gb=round(size["dyna2"] / 1e9, 1)),
            "dyna3_r1": dict(games=300, controllers="E38 policies (u0017, u21, u29) vs the net", gb=round(size["runs/dyna3_r1"] / 1e9, 1)),
            "dyna_live": dict(games=len(col), ticks=live_ticks, controllers=f"{len({(r['run'], r['policy_update']) for r in col})} distinct "
                              "PPO checkpoints (ppol, ppol2, ppol4) vs the net, alternating sides", gb=round(size["runs/dyna_live"] / 1e9, 1)),
            "scenarios_tower_v1": dict(slices=2014178, note="tower-death slices from 12,794 events of training games (E20)",
                                       gb=round(size["runs/scenarios_tower_v1"] / 1e9, 1)),
        },
        counterfactual_stream="not stored: regenerated on demand by moba_fork.c (~240M single-edit transitions available, I)")
    real_games = sum(v.get("games", 0) for v in C["data"]["corpora"].values())
    C["data"]["total_real_games"] = real_games
    (out / "compute.json").write_text(json.dumps(C, indent=1))
    print(json.dumps({k: v for k, v in C.items() if k not in ("by_day",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
