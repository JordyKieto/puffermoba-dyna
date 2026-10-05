"""Continuous Dyna loop: four workers that never wait for each other, plus the relays between them (this process).

  WM fleet    pods dl0..dl{NW-1}: FedAvg world-model training that never stops (--live-data): a LIVE_FRAC share of
              every batch comes from the newest LIVE_WINDOW files of the policy's real games; rank 0 checkpoints to S3
              every WM_CKPT updates.
  PPO fleet   pods dl{PPO_OFF}..dl{PPO_OFF+NP-1}: FedAvg dream PPO that never stops; it swaps in each new WM checkpoint at a
              scheduled update (--wm-watch) and restarts from a policy when told to (--reset-watch).
  collectors  CPU pods: workers that forever play 10 real games of the newest PPO policy vs the shipped net, record +
              verify them and upload to s3://.../moba/runs/dyna_live/ (scripts/moba/dyna_live_collect.py).
  evaluator   CPU pod EVAL: 60 real games per side for every new PPO policy -> results/moba/dream/<PPO>_eval.jsonl.

Relays (threads here): new PPO policy -> collectors + evaluator; new real games -> WM pods (data/moba/live, hard-linked
in only after the size check); new WM checkpoint -> PPO pods (wm_<U>_<it>.pt, all ranks swap at update U).
Drift guard: if the last 3 evaluated policies each won <= half the best's real wins (best >= 10), every PPO rank restarts
from the best policy at the same update (logged as a reset).
Headstart: WM from wm_r1 (fleet20e x1.5 + 270 games of dyna3_r1); PPO + collectors from all_handcrafted u0017;
the dyna3_r1 games are the first live files.

usage: python runs/pod/dyna_live.py launch   (start every worker)   |   python runs/pod/dyna_live.py relay   (relays)"""
import json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import pod

POOL, NW, NP, PPO_OFF = "dl", 7, 8, 12                        # WM pods dl0-6 (dl0 = A100), PPO pods dl12-19 (dl7-11 never got capacity)
WM, PPO = "wml2", "ppol4"                                     # ppol4: lr 5e-5, death penalty 0, KL 0.3 toward the anchor
COLLECT = {"evc4": 6}                                         # CPU pod -> collector workers
EVAL = "evc1"
INIT_WM = "runs/token_wm/wm_r1"
INIT_POL = "results/moba/dream/g4_policies/all_handcrafted/policy_u0017.pt"
EXTRA_CORPORA = "runs/dyna3_r1:7000-7269"
LIVE_FRAC, LIVE_WINDOW, WM_CKPT = 0.25, 10, 200
FRESH_MARGIN = 9                                              # ~1 SE of 300 games: a new anchor must win >= 9 more fresh radiant games
WM_UPDATES, PPO_UPDATES, ENVS = 20000, 1000, 128             # long upper bounds: the loop runs until stopped
REWARD = "--r-win 100 --dmg-k 5 --dmg-boost 2 --fall-bonus 10 --fall-radius 8 --kill-k 2 --siege-k 0.02 --anc-k 50 --xp-scale 0.5"
LIVE = "moba/runs/dyna_live"
R, D = pod.R_REPO, f"{pod.R_REPO}/data/moba"
EVF = Path(f"results/moba/dream/{PPO}_eval.jsonl")
POLS = Path(f"results/moba/dream/{PPO}_policies")
STATE = Path("runs/pod/dyna_live_state.json")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
c, bucket = pod.s3()


def on(p, cmd, check=False, capture=False):
    """Run a shell command on pod p (POD is per process in tools/pod.py: go through a subprocess)."""
    code = f"import sys; sys.path.insert(0, 'tools'); import pod; print(pod.run({cmd!r}, check={check}, capture=True) or '')"
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, POD=p), capture_output=True, text=True)
    return r.stdout if capture else r.returncode == 0 or r.stderr[-300:]


def py_on(p, code):
    """Run a tools/pod.py snippet with POD=p (rsync, s3_sync_to_pod, ...)."""
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'tools'); sys.path.insert(0, '.'); import pod\n" + code],
                       env=dict(os.environ, POD=p), capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def keys(prefix):
    return [o for pg in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix) for o in pg.get("Contents", [])]


# ---------------------------------------------------------------------------------------------------- launch
def launch():
    if not Path(INIT_WM, "model.pt").exists():
        k = sorted(o["Key"] for o in keys("moba/runs/wm_r1_r0/model_"))[-1]
        Path(INIT_WM).mkdir(parents=True, exist_ok=True)
        c.download_file(bucket, k, f"{INIT_WM}/model.pt"); c.download_file(bucket, "moba/runs/wm_r1_r0/stats.pt", f"{INIT_WM}/stats.pt")
        log(f"init WM: {k}")
    subprocess.run("pkill -f 'tools/fleet_rank.py wm_r1'", shell=True)
    stop = "tmux kill-server 2>/dev/null; pkill -f [m]obawm.wm.train; pkill -f [e]val_watch; pkill -f [t]ower_eval_big; true"
    with ThreadPoolExecutor(NW) as ex:                         # the WM pods: stop wm_r1, create the live dirs
        list(ex.map(lambda i: on(f"{POOL}{i}", f"{stop}; mkdir -p {D}/live {D}/live_in"), range(NW)))
    log("WM pods ready; launching the WM fleet")
    env = dict(os.environ, FLEET_PODS=POOL, FLEET_SHARDS="20", SKIP_SETUP="1", EXTRA_CORPORA=EXTRA_CORPORA,
               RANK0_GPU="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe")
    subprocess.run([sys.executable, "tools/fleet.py", WM, str(NW), "NVIDIA A40", "384", str(WM_UPDATES), INIT_WM,
                    "--lr", "3e-4", "--tower-fp-weight", "1.5", "--compile", "--fleet-every", "50", "--ckpt-every", str(WM_CKPT),
                    "--live-data", "data/moba/live", "--live-frac", str(LIVE_FRAC), "--live-window", str(LIVE_WINDOW),
                    "--live-every", "50"], env=env, check=True)
    log("launching the PPO fleet")
    env = dict(os.environ, POD_OFFSET=str(PPO_OFF), FLEET_SHARDS="20", EXTRA_CORPORA=EXTRA_CORPORA)
    subprocess.Popen([sys.executable, "runs/pod/ppo_pool.py", PPO, POOL, str(NP), INIT_WM, INIT_POL, str(PPO_UPDATES), str(ENVS),
                      *REWARD.split()], env=env, stdout=open(f"runs/pod/{PPO}_launch.log", "w"), stderr=subprocess.STDOUT)
    log("launching the collectors")

    def collector(i_p):
        i, p = i_p
        code = f'''
import json
out = f"{{pod.R_REPO}}/dyna/dyna_live"
pod.run("pkill -f [d]yna_live_collect; tmux kill-session -t col 2>/dev/null; true", check=False)
pod.push_code(); pod.run(f"mkdir -p {{out}}/pols")
pod.rsync_to(["data/moba/mix1k/logs/games.0000900.mbl"], f"{{out}}/")
pod.rsync_to(["{INIT_POL}"], f"{{out}}/pols/.incoming.pt"); pod.run(f"[ -e {{out}}/pols/current.pt ] || mv {{out}}/pols/.incoming.pt {{out}}/pols/current.pt")
pol = pod.REPO / "runs/pod/.s3_policy_live_{i}.json"; pol.write_text(json.dumps(pod.s3_upload_policy("dyna_live", days=3))); pol.chmod(0o600)
pod.rsync_to([pol], f"{{out}}/.s3_policy.json"); pol.unlink(); pod.run(f"chmod 600 {{out}}/.s3_policy.json")
ws = " & ".join(f"python -u scripts/moba/dyna_live_collect.py --dir {{out}} --worker {{w}} --base {{1_000_000 + {i} * 100_000 + w * 10_000}} > {{out}}/w{{w}}.log 2>&1" for w in range({COLLECT[p]}))
pod.run(f"cat > {{out}}/col.sh <<'EOT'\\ncd {{pod.R_REPO}}\\n{{ws}} &\\nwait\\nEOT")
pod.run(f"tmux new-session -d -s col 'bash {{out}}/col.sh'")
print("COLLECTING")
'''
        return p, py_on(p, code)
    with ThreadPoolExecutor(len(COLLECT)) as ex:
        for p, (ok, out) in ex.map(collector, enumerate(COLLECT)):
            log(f"collector {p}: {'ok' if ok and 'COLLECTING' in out else 'FAILED ' + out[-400:]}")
    ok, out = py_on(EVAL, "pod.push_code()")
    log(f"evaluator {EVAL}: code pushed {ok}")


# ---------------------------------------------------------------------------------------------------- relays
st = json.loads(STATE.read_text()) if STATE.exists() else dict(live_synced=[], wm_sent=None, resets=[], best=None)
lock = threading.Lock()
save = lambda: STATE.write_text(json.dumps(st, indent=1))


def push_all(pods, src, dest_dir, name):
    """rsync src to every pod as dest_dir/.<name>.tmp, then rename to dest_dir/<name> (atomic for the reader)."""
    code = (f"pod.rsync_to([{src!r}], '{dest_dir}/.{name}.tmp'); "
            f"pod.run('mv -f {dest_dir}/.{name}.tmp {dest_dir}/{name}')")
    with ThreadPoolExecutor(len(pods)) as ex:
        return [ok for ok, _ in ex.map(lambda p: py_on(p, code), pods)]


def policy_relay():
    """New rank-0 policy.pt -> keep a copy, hand it to the collectors, queue it for the evaluator."""
    key, last = f"moba/runs/{PPO}_r0/policy.pt", None
    POLS.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            h = c.head_object(Bucket=bucket, Key=key)
            if h["LastModified"] != last:
                last = h["LastModified"]
                tmp = POLS / "latest.pt"; c.download_file(bucket, key, str(tmp))
                import torch
                ck = torch.load(tmp, map_location="cpu", weights_only=False)
                u = ck["update"]; keep = POLS / f"policy_u{u:04d}.pt"; tmp.rename(keep)
                with lock:
                    st["ppo_update"], st["pending_eval"] = u, str(keep); save()
                oks = push_all(list(COLLECT), str(keep), f"{R}/dyna/dyna_live/pols", "current.pt")
                log(f"policy u{u} (wm {ck.get('wm')}, reset {ck.get('reset') or '-'}): to collectors {sum(oks)}/{len(oks)}")
        except Exception as e:  # noqa: BLE001
            if "404" not in str(e):                     # 404: rank 0 has not finished its first update yet
                log(f"policy relay: {e!r}")
        time.sleep(20)


def evaluator():
    """60 real games per side for the newest unevaluated policy; best-so-far; drift guard."""
    py_on(EVAL, "pod.run(f'mkdir -p {pod.R_REPO}/eval_live')")
    while True:
        with lock:
            p = st.pop("pending_eval", None)
        if not p:
            time.sleep(10); continue
        p = Path(p); u = int(p.stem.split("_u")[1]); t0 = time.time()
        code = (f"pod.rsync_to(['{p}'], f'{{pod.R_REPO}}/eval_live/'); "
                f"pod.run(f'cd {{pod.R_REPO}} && timeout 600 python -u -m mobawm.rl.arena --policy eval_live/{p.name} --games 60 --workers 32 "
                f"--teams ppo --opponents net --out eval_live/arena_u{u:04d}.jsonl > eval_live/arena_u{u:04d}.log 2>&1', check=False); "
                f"print('@@' + pod.run(f'cat {{pod.R_REPO}}/eval_live/arena_u{u:04d}.jsonl', capture=True))")
        py_on(EVAL, "pod.run(" + repr("ps -eo pid,ppid,cmd | awk '$2==1 && /spawn_main/ {print $1}' | xargs -r kill") + ", check=False)")   # orphans of a killed eval only
        ok, out = py_on(EVAL, code)
        games = [json.loads(l) for l in out.split("@@", 1)[-1].splitlines() if l.strip().startswith("{")]
        if not games:
            log(f"eval u{u}: FAILED {out[-300:]}"); continue
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        rec = dict(update=u, fleet_steps=ck.get("agent_steps"), wm=ck.get("wm"), reset=ck.get("reset"), eval_s=round(time.time() - t0),
                   t=time.strftime("%H:%M:%S"))
        for side in ("radiant", "dire"):
            g = [x for x in games if x["side"] == side]
            if g:
                rec[f"{side}_wins"] = sum(x["won"] for x in g); rec[f"{side}_games"] = len(g)
                rec[f"{side}_towers_ours"] = round(5 * sum(x["ours_towers_killed"] for x in g) / len(g), 2)
                rec[f"{side}_towers_theirs"] = round(5 * sum(x["theirs_towers_killed"] for x in g) / len(g), 2)
                rec[f"{side}_return"] = round(sum(x["return_ours"] for x in g) / len(g), 2)
        with EVF.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        score = lambda r: (r.get("radiant_wins", 0) + r.get("dire_wins", 0), r.get("radiant_towers_ours", 0))
        rows = [json.loads(l) for l in EVF.read_text().splitlines() if l.strip()]
        anc = st["anchor"]
        log(f"eval u{u}: radiant {rec.get('radiant_wins')}/60 towers {rec.get('radiant_towers_ours')} | dire {rec.get('dire_wins')}/60 "
            f"| wm {rec['wm']} | anchor u{anc['update']} ({anc['fixed']} fixed-seed wins, {anc['fresh']}/300 fresh)")
        if score(rec)[0] >= anc["fixed"]:                 # candidate: confirm on 300 fresh games per side before it becomes the anchor
            fr = fresh_check(p)
            with open(f"results/moba/dream/{PPO}_fresh.jsonl", "a") as f:
                f.write(json.dumps(dict(update=u, fixed=score(rec)[0], **fr, anchor=anc["update"], anchor_fresh=anc["fresh"])) + "\n")
            if fr["radiant_wins"] >= anc["fresh"] + FRESH_MARGIN:
                at = st.get("ppo_update", u) + 2
                oks = push_all([f"{POOL}{PPO_OFF + i}" for i in range(NP)], str(p), f"{R}/runs/dream_ppo/init_{PPO}/anchor", f"anchor_{at:05d}.pt")
                with lock:
                    st["anchor"] = dict(path=str(p), update=u, fixed=score(rec)[0], fresh=fr["radiant_wins"], at=at); save()
                log(f"NEW ANCHOR u{u}: fresh radiant {fr['radiant_wins']}/300 vs {anc['fresh']}: KL anchor + reset target from PPO update {at} "
                    f"({sum(oks)}/{NP} pods)")
            else:
                log(f"fresh check u{u}: radiant {fr['radiant_wins']}/300 (towers {fr['radiant_towers']}) vs anchor {anc['fresh']} + {FRESH_MARGIN}: kept u{anc['update']}")
            anc = st["anchor"]
        since = max(st["resets"][-1]["at"] if st["resets"] else -1, anc.get("at", -1))
        last3 = [r for r in rows if r["update"] > since][-3:]
        bw = anc["fixed"]
        if bw >= 10 and len(last3) == 3 and all(score(r)[0] <= bw / 2 for r in last3):
            at = st.get("ppo_update", u) + 2
            oks = push_all([f"{POOL}{PPO_OFF + i}" for i in range(NP)], anc["path"], f"{R}/runs/dream_ppo/init_{PPO}/reset", f"policy_{at:05d}.pt")
            with lock:
                st["resets"].append(dict(at=at, to=anc["update"], last3=[score(r)[0] for r in last3], pods=sum(oks))); save()
            log(f"DRIFT: last 3 policies won {[score(r)[0] for r in last3]} vs anchor {bw}: every PPO rank restarts from anchor u{anc['update']} "
                f"at update {at} ({sum(oks)}/{NP} pods)")


def fresh_check(p):
    """300 games per side on seeds never used for selection (5 x 60: one 300-game arena hangs)."""
    g = []
    for k in range(5):
        code = (f"pod.rsync_to(['{p}'], f'{{pod.R_REPO}}/eval_live/fresh.pt'); "
                f"pod.run(f'cd {{pod.R_REPO}} && timeout 300 python -u -m mobawm.rl.arena --policy eval_live/fresh.pt --games 60 --workers 32 "
                f"--seed-base {60_000_000 + 1000 * k} --teams ppo --opponents net --out eval_live/fresh_{k}.jsonl > eval_live/fresh_{k}.log 2>&1', check=False); "
                f"print('@@' + pod.run(f'cat {{pod.R_REPO}}/eval_live/fresh_{k}.jsonl', capture=True, check=False))")
        ok, out = py_on(EVAL, code)
        g += [json.loads(l) for l in out.split("@@", 1)[-1].splitlines() if l.strip().startswith("{")]
    rad, dire = [x for x in g if x["side"] == "radiant"], [x for x in g if x["side"] == "dire"]
    return dict(radiant_wins=sum(x["won"] for x in rad), radiant_games=len(rad), dire_wins=sum(x["won"] for x in dire),
                radiant_towers=round(5 * sum(x["ours_towers_killed"] for x in rad) / max(len(rad), 1), 2))


def data_relay():
    """New real games in S3 -> every WM pod's data/moba/live (synced into live_in, size-checked, then hard-linked)."""
    sources = ["moba/runs/dyna3_r1/dense/", f"{LIVE}/dense/"]
    while True:
        try:
            objs = sorted((o for src in sources for o in keys(src) if o["Key"].endswith(".mbs")), key=lambda o: o["LastModified"])
            want = objs[-LIVE_WINDOW - 2:]                    # the newest arrivals, a little more than the window
            new = [o["Key"] for o in want if o["Key"].rsplit("/", 1)[1] not in st["live_synced"]]
            if new:
                jobs = {src: {int(k.rsplit("/", 1)[1].split(".")[1]) + g for k in new if k.startswith(src) for g in range(10)} for src in sources}
                code = "".join(f"pod.s3_sync_to_pod({src!r}, '{D}/live_in', {g!r})\n" for src, g in jobs.items() if g)
                code += (f"pod.run('cd {D} && for f in live_in/*.mbs; do b=$(basename $f); [ -e live/$b ] || ln $f live/$b; done')\n"
                         "print('SYNCED')")
                with ThreadPoolExecutor(NW) as ex:
                    res = list(ex.map(lambda i: py_on(f"{POOL}{i}", code), range(NW)))
                good = sum(ok and "SYNCED" in o for ok, o in res)
                names = [k.rsplit("/", 1)[1] for k in new]
                if good == NW:
                    st["live_synced"] = sorted(set(st["live_synced"]) | set(names)); save()
                log(f"live data: {len(new)} new files ({names[0]} .. {names[-1]}) -> {good}/{NW} WM pods"
                    + ("" if good == NW else f"; first failure: {next(o for ok, o in res if not (ok and 'SYNCED' in o))[-200:]}"))
            meta = Path("results/moba/dream/dyna_live_meta"); meta.mkdir(parents=True, exist_ok=True)
            for o in keys(f"{LIVE}/meta/"):
                f = meta / o["Key"].rsplit("/", 1)[1]
                if not f.exists():
                    c.download_file(bucket, o["Key"], str(f))
        except Exception as e:  # noqa: BLE001
            log(f"data relay: {e!r}")
        time.sleep(60)


def wm_relay():
    """New WM rank-0 checkpoint -> every PPO pod, swapped in by every rank at the same update (now + 2)."""
    while True:
        try:
            ks = sorted(o["Key"] for o in keys(f"moba/runs/{WM}_r0/model_"))
            if ks and ks[-1] != st["wm_sent"]:
                k = ks[-1]; it = int(k.rsplit("_", 1)[1].split(".")[0]); at = st.get("ppo_update", 0) + 2
                stem = k.rsplit("/", 1)[1][:-3]
                dst = f"{R}/runs/token_wm/wm_{PPO}"
                code = (f"pod.s3_sync_to_pod('{k[:-3]}', '{dst}/incoming/{stem}')\n"
                        f"pod.run('ln -f {dst}/incoming/{stem}/.pt {dst}/ready/wm_{at:05d}_{it:06d}.pt')\nprint('SENT')")
                with ThreadPoolExecutor(NP) as ex:
                    res = list(ex.map(lambda i: py_on(f"{POOL}{PPO_OFF + i}", code), range(NP)))
                good = sum(ok and "SENT" in o for ok, o in res)
                if good < NP:
                    log(f"WM relay failure: {next(o for ok, o in res if not (ok and 'SENT' in o))[-200:]}")
                with lock:
                    st["wm_sent"] = k; save()
                log(f"WM {stem} -> {good}/{NP} PPO pods, swap at PPO update {at}")
        except Exception as e:  # noqa: BLE001
            log(f"wm relay: {e!r}")
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1] == "data":                       # only the real games -> WM relay (e.g. during a PPO ablation)
        data_relay()
    elif sys.argv[1] == "launch":
        launch()
    else:
        for f in (policy_relay, evaluator, data_relay, wm_relay):
            threading.Thread(target=f, daemon=True).start()
        while True:
            time.sleep(3600)
