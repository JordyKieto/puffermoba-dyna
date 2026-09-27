"""One GPU pod on RunPod for world-model training. Adapted from zelda-RL's fleet/.

    .venv/bin/python tools/pod.py template                 # register the RunPod template (idempotent)
    .venv/bin/python tools/pod.py create --confirm         # H200 (fallback H100 NVL), SSH-ready
    .venv/bin/python tools/pod.py bootstrap                # code, apt, build moba_collect
    .venv/bin/python tools/pod.py data --corpus mix100     # logs S3 -> pod, dense on pod, dense -> S3
    .venv/bin/python tools/pod.py push --corpus mix1k      # corpus collected ON the pod -> S3
    .venv/bin/python tools/pod.py train NAME -- --train-games 0-90 --held-games 91-99 --minutes 20
    .venv/bin/python tools/pod.py status [NAME]            # tmux, GPU, last log lines
    .venv/bin/python tools/pod.py sync                     # runs/token_wm/ -> ./runs/token_wm/
    .venv/bin/python tools/pod.py kill --confirm           # the pod bills until this runs

Design, from zelda-RL's measured RunPod runbook (docs/moba/RUNPOD.md):
  - The Mac's uplink is slow (~0.7 MB/s), so nothing big goes Mac -> pod.
    Corpora live in S3 and pods pull them over the datacenter link with
    presigned URLs; pods never hold AWS credentials.
  - Only the tiny game logs are the source of truth in S3; the full-state
    files are regenerated on the pod (bit-exact, replay-verified) and pushed
    back to S3 through presigned PUTs so later pods can pull them directly.
  - Nothing on the pod must survive it: checkpoints/logs go to S3 as they are written,
    data is pulled from S3 in about a minute (s5cmd + scoped read-only credentials).
  - Secrets reach the pod only as a mode-600 file, never on a command line.
"""
from __future__ import annotations

import argparse
import os
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from racewm import secrets  # noqa: E402

# POD=<name> selects a second (third, ...) pod with its own state file; unset = the
# original pod, runs/pod/state.json, exactly as before.
POD_NAME = os.environ.get("POD", "")
STATE = REPO / "runs" / "pod" / (f"state_{POD_NAME}.json" if POD_NAME else "state.json")
REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"   # CUDA torch preinstalled
TEMPLATE = "puffertank-moba-wm"
# Whole dataset in VRAM, with room to grow the corpus ~25x (fp32) before it no longer fits.
GPUS = ["NVIDIA H200", "NVIDIA H100 NVL"]
# The repo, data and runs live on the pod's LOCAL container disk: measured 2026-09-27,
# /workspace (network storage) wrote at ~5 MB/s and read at ~255 MB/s, the container disk
# reads at ~4.7 GB/s. Nothing there needs to survive the pod: checkpoints and logs are
# uploaded to S3 as they are written (mobawm/s3persist.py), data comes from S3 (cmd_data).
R_REPO = "/root/puffertank"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
S3_PREFIX = "moba"


def log(msg):
    print(f"[pod] {msg}", flush=True)


# --- credentials --------------------------------------------------------------

def runpod_key() -> str:
    key = secrets.get("RUNPOD", "RUNPOD_API_KEY")
    if key:
        return key
    # zelda-RL / sunnypilot convention: KEY=VALUE lines in sunnypilot's secrets.yaml
    p = Path.home() / "Documents/dev/sunnypilot/secrets.yaml"
    for line in p.read_text().splitlines() if p.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "RUNPOD" and v.strip():
            return v.strip()
    raise SystemExit("RunPod API key not found (Keychain race-wm:RUNPOD or sunnypilot secrets.yaml)")


def s3():
    secrets.export_to_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")
    import boto3
    return boto3.client("s3"), secrets.get("S3_BUCKET")


# --- RunPod API ---------------------------------------------------------------

def api(method, path, **kw):
    r = requests.request(method, f"{REST}{path}", timeout=60,
                         headers={"Authorization": f"Bearer {runpod_key()}"}, **kw)
    if not r.ok:
        raise SystemExit(f"RunPod {method} {path} -> {r.status_code}: {r.text[:400]}")
    return r.json() if r.content else {}


def gql(query):
    r = requests.post(GQL, json={"query": query}, timeout=30,
                      headers={"Authorization": f"Bearer {runpod_key()}"}).json()
    return r["data"]


def pod_spec() -> dict:
    return dict(imageName=IMAGE, containerDiskInGb=80, volumeInGb=300, volumeMountPath="/workspace",
                ports=["22/tcp", "6006/http"])


def cmd_template(a):
    existing = [t for t in api("GET", "/templates") if t.get("name") == TEMPLATE]
    body = dict(name=TEMPLATE, isServerless=False, **pod_spec(),
                readme="puffertank MOBA world-model training. See docs/moba/RUNPOD.md.")
    if existing:
        api("PATCH", f"/templates/{existing[0]['id']}", json={k: v for k, v in body.items() if k != "isServerless"})
        log(f"template {TEMPLATE} updated ({existing[0]['id']})")
    else:
        t = api("POST", "/templates", json=body)
        log(f"template {TEMPLATE} created ({t.get('id')})")


def load_state() -> dict:
    if not STATE.exists():
        raise SystemExit("no pod recorded; run: tools/pod.py create --confirm")
    return json.loads(STATE.read_text())


def cmd_create(a):
    gpus = a.gpu.split(",") if a.gpu else GPUS
    bal = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    log(f"plan: 1 GPU pod, priority {gpus}, secure cloud, image {IMAGE}, disk 80 GB + volume 300 GB")
    log(f"balance ${bal['clientBalance']:.2f}, current spend ${bal['currentSpendPerHr']:.2f}/hr")
    if not a.confirm:
        log("dry run: add --confirm to create (the pod bills by the hour until `kill --confirm`)")
        return
    pub = (SSH_KEY.with_suffix(".pub")).read_text().strip()
    body = dict(name=f"puffertank-{a.name}", computeType="GPU", gpuTypeIds=gpus, gpuCount=1,
                cloudType="SECURE", env={"PUBLIC_KEY": pub}, **pod_spec())
    for attempt in range(4):
        try:
            pod = api("POST", "/pods", json=body)
            break
        except SystemExit as e:                     # RunPod 500s transiently; retry (zelda-RL: 2 of 4 creates)
            log(f"create failed ({e}); retry in 20s")
            time.sleep(20)
    else:
        raise SystemExit("create failed 4 times")
    pid = pod["id"]
    log(f"created {pid} ({pod.get('machine', {}).get('gpuTypeId', gpus[0])}); waiting for SSH ...")
    t0 = time.time()
    while time.time() - t0 < 900:
        p = api("GET", f"/pods/{pid}")
        ip, port = p.get("publicIp"), (p.get("portMappings") or {}).get("22")
        if p.get("desiredStatus") == "RUNNING" and ip and port:
            break
        log(f"  status={p.get('desiredStatus')} ip={ip or '-'} ssh={port or '-'} ({time.time() - t0:.0f}s)")
        time.sleep(15)
    else:
        raise SystemExit(f"{pid} not SSH-ready after 900s")
    st = dict(id=pid, ip=ip, port=int(port), gpu=p.get("machine", {}).get("gpuTypeId") or p.get("gpuTypeId"),
              cost_per_hr=p.get("costPerHr"), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))
    log(f"SSH ready in {time.time() - t0:.0f}s: root@{ip} -p {port} | {st['gpu']} ${st['cost_per_hr']}/hr")


def cmd_kill(a):
    st = load_state()
    if not a.confirm:
        log(f"would delete pod {st['id']} ({st['gpu']}); add --confirm")
        return
    api("DELETE", f"/pods/{st['id']}")
    STATE.rename(STATE.with_suffix(".killed.json"))
    log(f"deleted {st['id']}; spend now ${gql('query { myself { currentSpendPerHr } }')['myself']['currentSpendPerHr']:.2f}/hr")


# --- SSH ----------------------------------------------------------------------

def ssh_base(st):
    return ["ssh", "-i", str(SSH_KEY), "-p", str(st["port"]), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", f"root@{st['ip']}"]


def run(cmd: str, capture=False, check=True):
    st = load_state()
    r = subprocess.run(ssh_base(st) + [cmd], text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise SystemExit(f"remote command failed ({r.returncode}): {cmd[:120]}\n{(r.stderr or '')[-800:]}")
    return r.stdout if capture else r.returncode


def rsync_to(paths, dest, excludes=(), includes=()):
    st = load_state()
    e = f"ssh -i {SSH_KEY} -p {st['port']} -o StrictHostKeyChecking=accept-new"
    cmd = ["rsync", "-rlptz", "--no-owner", "--no-group", "-e", e]
    for x in includes:
        cmd += ["--include", x]
    for x in excludes:
        cmd += ["--exclude", x]
    subprocess.run(cmd + [str(p) for p in paths] + [f"root@{st['ip']}:{dest}"], check=True)


def rsync_from(src, dest):
    st = load_state()
    e = f"ssh -i {SSH_KEY} -p {st['port']} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-rlptz", "-e", e, f"root@{st['ip']}:{src}", str(dest)], check=True)


CODE_EXCLUDES = [".git", ".venv", "data", "runs", "build", "checkpoints", "wandb", "logs", "out", "replays",
                 "__pycache__", "*.pt", "raylib-5.5_macos", "/third_party/pufferlib/resources/*",
                 "media", "results", "artifacts", "samples"]
CODE_INCLUDES = ["/third_party/pufferlib/resources/moba/***"]   # weights for `collect`; the rest is render art


def push_code():
    t0 = time.time()
    run(f"mkdir -p {R_REPO}")
    rsync_to([f"{REPO}/"], R_REPO + "/", CODE_EXCLUDES, CODE_INCLUDES)
    log(f"code pushed in {time.time() - t0:.0f}s")


def cmd_bootstrap(a):
    t0 = time.time()
    # apt first: the image has no rsync, which push_code needs
    run("export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq && apt-get install -y -qq clang zlib1g-dev "
        "tmux rsync curl libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev > /tmp/apt.log 2>&1) || (tail -20 /tmp/apt.log; exit 1)")
    push_code()
    run(f"cd {R_REPO} && pip install -q numpy requests wandb 2>&1 | tail -2; "
        f"python -c 'import torch; print(\"torch\", torch.__version__, torch.cuda.get_device_name(0))'")
    run(f"cd {R_REPO} && (tools/build.sh moba-collect > /tmp/build.log 2>&1 || (tail -20 /tmp/build.log; exit 1)) "
        f"&& tail -1 /tmp/build.log")
    out = run("nproc; free -g | awk '/Mem/{print $2\" GB RAM\"}'; nvidia-smi --query-gpu=name,memory.total "
              "--format=csv,noheader; df -h /workspace | tail -1", capture=True)
    log(f"bootstrap done in {time.time() - t0:.0f}s:\n{out.strip()}")


# --- data ---------------------------------------------------------------------

def cmd_data(a):
    """Logs (source of truth) S3 -> pod; dense regenerated on the pod; dense -> S3."""
    c, bucket = s3()
    lp = REPO / "data" / "moba" / a.corpus / "logs"
    key = lambda kind, name: f"{S3_PREFIX}/{a.corpus}/{kind}/{name}"
    logs = sorted(lp.glob("games.*.mbl"))
    if not logs:                                    # corpus generated on a pod: it lives only in S3
        return data_from_s3(a, c, bucket, key)
    for f in logs:                                  # tiny (KB); skip if already present with same size
        try:
            if c.head_object(Bucket=bucket, Key=key("logs", f.name))["ContentLength"] == f.stat().st_size:
                continue
        except Exception:
            pass
        c.upload_file(str(f), bucket, key("logs", f.name))
    log(f"s3://{bucket}/{S3_PREFIX}/{a.corpus}/logs: {len(logs)} logs "
        f"({sum(f.stat().st_size for f in logs) / 1e6:.1f} MB)")
    rdir = f"{R_REPO}/data/moba/{a.corpus}"
    lines = [f"set -e; mkdir -p {rdir}/logs {rdir}/dense"]
    for f in logs:
        g = c.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key("logs", f.name)},
                                     ExpiresIn=6 * 3600)
        lines.append(f"curl -sf -o {rdir}/logs/{f.name} {shlex.quote(g)}")
    dense_have = []
    for f in logs:                                  # pull dense from S3 if it is there already
        k = key("dense", f.name.replace(".mbl", ".mbs"))
        try:
            c.head_object(Bucket=bucket, Key=k)
            dense_have.append(f)
            g = c.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": k}, ExpiresIn=6 * 3600)
            lines.append(f"curl -sf -o {rdir}/dense/{f.name.replace('.mbl', '.mbs')} {shlex.quote(g)}")
        except Exception:
            pass
    todo = [f for f in logs if f not in dense_have]
    if todo:
        names = " ".join(f.stem for f in todo)
        lines.append(f"cd {R_REPO} && echo {names} | tr ' ' '\\n' | xargs -P $(nproc) -I{{}} sh -c "
                     f"'build/moba_collect replay {rdir}/logs/{{}}.mbl --dense {rdir}/dense/{{}}.mbs | tail -2'")
        for f in todo:
            n = f.name.replace(".mbl", ".mbs")
            put = c.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key("dense", n)},
                                           ExpiresIn=6 * 3600)
            lines.append(f"curl -sf -X PUT -T {rdir}/dense/{n} {shlex.quote(put)}")
    lines.append(f"du -sh {rdir}/logs {rdir}/dense")
    script = REPO / "runs" / "pod" / "data.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o600)
    rsync_to([script], "/workspace/data.sh")
    t0 = time.time()
    out = run("bash /workspace/data.sh && rm /workspace/data.sh", capture=True)
    script.unlink()
    log(f"data on pod in {time.time() - t0:.0f}s ({len(dense_have)} dense pulled, {len(todo)} regenerated "
        f"+ pushed to S3):\n{out.strip()}")


def s3_read_credentials(prefix: str, seconds: int = 3600) -> dict:
    """Temporary AWS credentials (STS federation token) that can only LIST and READ
    s3://<bucket>/<prefix>*, expiring after `seconds`. The standard way a pod reads S3:
    the long-term keys never leave the Mac (IAM: user datasets, inline policy
    pod-temporary-credentials allows sts:GetFederationToken on federated-user/pod-*)."""
    import boto3
    _, bucket = s3()
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{bucket}/{prefix}*"},
        {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
         "Condition": {"StringLike": {"s3:prefix": [f"{prefix}*", prefix.rstrip("/")]}}}]}
    cr = boto3.client("sts").get_federation_token(Name="pod-read", Policy=json.dumps(policy),
                                                  DurationSeconds=seconds)["Credentials"]
    return dict(AWS_ACCESS_KEY_ID=cr["AccessKeyId"], AWS_SECRET_ACCESS_KEY=cr["SecretAccessKey"],
                AWS_SESSION_TOKEN=cr["SessionToken"], AWS_DEFAULT_REGION=secrets.get("AWS_DEFAULT_REGION") or "us-east-1")


def s3_sync_to_pod(prefix: str, dest: str, games=None):
    """Copy s3://<bucket>/<prefix> to <dest> ON the pod with s5cmd (parallel), using
    temporary read-only credentials for that prefix, then verify every size against S3.
    games: optional set of game indices -> only the corpus files that hold them (file
    games.<first>.mbs/.mbl holds games first .. next file's first - 1).
    Measured 2026-09-27 (H100 NVL pod): all of mix1k, 15.3 GB, in ~60 s (the pod's ~2 Gb/s
    link is the limit); 20 games in ~2 s."""
    c, bucket = s3()
    objs = [o for p in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in p.get("Contents", []) if not o["Key"].endswith(".tmp")]
    if games is not None:
        firsts = sorted({int(o["Key"].rsplit(".", 2)[-2]) for o in objs if o["Key"].endswith(".mbs")})
        keep = {f for f, nxt in zip(firsts, firsts[1:] + [10 ** 9]) if any(f <= g < nxt for g in games)}
        objs = [o for o in objs if not o["Key"].endswith((".mbs", ".mbl")) or int(o["Key"].rsplit(".", 2)[-2]) in keep]
    env = s3_read_credentials(prefix)
    f = REPO / "runs" / "pod" / ".s3_read.env"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(f"export {k}='{v}'\n" for k, v in env.items()))
    f.chmod(0o600)
    cmds = REPO / "runs" / "pod" / "s5cmd.txt"
    cmds.write_text("".join(f"cp -s s3://{bucket}/{o['Key']} {dest}/{o['Key'][len(prefix):]}\n" for o in objs))
    rsync_to([f, cmds], "/root/")
    f.unlink()
    cmds.unlink()
    t0 = time.time()
    try:
        run("chmod 600 /root/.s3_read.env; command -v s5cmd >/dev/null || (cd /tmp && curl -sL "
            "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz s5cmd "
            f"&& mv s5cmd /usr/local/bin/); mkdir -p {dest}; . /root/.s3_read.env && "
            "s5cmd --numworkers 128 run /root/s5cmd.txt > /dev/null")
    finally:
        run("rm -f /root/.s3_read.env /root/s5cmd.txt", check=False)
    want = {o["Key"][len(prefix):]: o["Size"] for o in objs}
    have = dict(l.split(" ", 1)[::-1] for l in run(f"cd {dest} && find . -type f -printf '%s %P\\n'",
                                                    capture=True).splitlines() if l.strip())
    bad = [k for k, n in want.items() if int(have.get(k, -1)) != n]
    if bad:
        raise SystemExit(f"s3 copy incomplete: {len(bad)} files differ from S3, e.g. {bad[:3]}")
    log(f"copied and verified {len(want)} files ({sum(want.values()) / 1e9:.1f} GB) s3://{bucket}/{prefix} -> {dest} "
        f"in {time.time() - t0:.0f}s")


def data_from_s3(a, c, bucket, key):
    """A corpus that exists in S3 (logs + dense): `aws s3 sync` onto the pod (s3_sync_to_pod)."""
    games = None
    if getattr(a, "games", None):
        from mobawm.wm.train import parse_games
        games = parse_games(a.games)
    s3_sync_to_pod(f"{S3_PREFIX}/{a.corpus}/", f"{R_REPO}/data/moba/{a.corpus}", games)


def cmd_push(a):
    """A corpus generated ON the pod -> S3 (logs and dense), via presigned PUTs,
    then verified: every object's size in S3 must equal the file on the pod."""
    c, bucket = s3()
    rdir = f"{R_REPO}/data/moba/{a.corpus}"
    sizes = dict(line.split()[::-1] for line in run(
        f"cd {rdir} && stat -c '%s %n' logs/*.mbl dense/*.mbs", capture=True).strip().splitlines())
    lines = []
    for i, rel in enumerate(sorted(sizes)):
        key = f"{S3_PREFIX}/{a.corpus}/{rel}"
        put = c.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=6 * 3600)
        lines.append(f"curl -sf -X PUT -T {rdir}/{rel} {shlex.quote(put)} || echo FAIL {rel} &")
        if i % 32 == 31:
            lines.append("wait")      # a plain script, not xargs: xargs strips the URL's quoting
    lines.append("wait")
    script = REPO / "runs" / "pod" / "push.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o600)
    rsync_to([script], "/workspace/push.sh")
    script.unlink()
    t0 = time.time()
    out = run("bash /workspace/push.sh; rm -f /workspace/push.sh", capture=True)
    fails = [l for l in out.splitlines() if l.startswith("FAIL")]
    s3_sizes = {o["Key"].split(f"{a.corpus}/", 1)[1]: o["Size"]
                for p in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{S3_PREFIX}/{a.corpus}/")
                for o in p.get("Contents", [])}
    bad = [rel for rel, n in sizes.items() if s3_sizes.get(rel) != int(n)]
    total = sum(int(n) for n in sizes.values())
    if fails or bad:
        raise SystemExit(f"push incomplete: {len(fails)} curl failures, {len(bad)} size mismatches, e.g. {(fails + bad)[:3]}")
    log(f"pushed and verified {len(sizes)} files, {total / 1e9:.2f} GB -> s3://{bucket}/{S3_PREFIX}/{a.corpus}/ "
        f"in {time.time() - t0:.0f}s")


# --- training -----------------------------------------------------------------

def s3_upload_policy(run: str, days: float = 7.0) -> dict:
    """A presigned S3 POST policy that lets a pod upload anything under
    moba/runs/<run>/ (and nothing else) for `days`, without holding credentials."""
    c, bucket = s3()
    prefix = f"{S3_PREFIX}/runs/{run}/"
    secs = int(days * 86400)
    post = c.generate_presigned_post(bucket, prefix + "${filename}",
                                     Conditions=[["starts-with", "$key", prefix],
                                                 ["content-length-range", 0, 20 * 1024 ** 3]],
                                     ExpiresIn=secs)
    fields = {k: v for k, v in post["fields"].items() if k != "key"}
    return dict(url=post["url"], fields=fields, prefix=prefix, bucket=bucket, expires=time.time() + secs - 60)


def cmd_train(a):
    push_code()
    wandb_key = secrets.get("WANDB_API_KEY")
    if wandb_key:
        sec = REPO / "runs" / "pod" / ".pod_secrets"
        sec.write_text(f"export WANDB_API_KEY='{wandb_key}'\n")
        sec.chmod(0o600)
        rsync_to([sec], f"{R_REPO}/.pod_secrets")
        sec.unlink()
    out = f"runs/token_wm/{a.name}"
    # automatic off-pod persistence: the trainer uploads every checkpoint/log to S3 (mobawm/s3persist.py)
    pol = REPO / "runs" / "pod" / ".s3_policy.json"
    pol.write_text(json.dumps(s3_upload_policy(a.name)))
    pol.chmod(0o600)
    run(f"mkdir -p {R_REPO}/{out}")
    rsync_to([pol], f"{R_REPO}/{out}/.s3_policy.json")
    pol.unlink()
    run(f"chmod 600 {R_REPO}/{out}/.s3_policy.json")
    log(f"S3 persistence: every checkpoint/log -> s3://{secrets.get('S3_BUCKET')}/{S3_PREFIX}/runs/{a.name}/ "
        f"(upload policy valid 7 days)")
    args = " ".join(shlex.quote(x) for x in a.train_args)
    launch = (f"cd {R_REPO} && mkdir -p {out} && [ -f .pod_secrets ] && . ./.pod_secrets; "
              f"python -u -m mobawm.wm.train --out {out} {args} 2>&1 | tee {out}/train.log")
    run(f"mkdir -p {R_REPO}/{out} && cat > {R_REPO}/{out}/launch.sh <<'EOF'\n{launch}\nEOF")
    run(f"tmux kill-session -t {a.name} 2>/dev/null; true", check=False)
    run(f"tmux new-session -d -s {a.name} 'bash {R_REPO}/{out}/launch.sh'")
    time.sleep(3)
    alive = run(f"tmux has-session -t {a.name} 2>/dev/null && echo ALIVE || echo DEAD", capture=True).strip()
    log(f"train {a.name}: tmux {alive}; log {R_REPO}/{out}/train.log")


def cmd_status(a):
    name = a.name or "*"
    out = run(f"tmux ls 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
              f"--format=csv,noheader; for f in {R_REPO}/runs/token_wm/{name}/train.log; do "
              f"[ -f $f ] && echo \"== $f\" && tail -n {a.lines} $f; done", capture=True, check=False)
    print(out)


def cmd_sync(a):
    dest = REPO / "runs" / "token_wm"
    dest.mkdir(parents=True, exist_ok=True)
    rsync_from(f"{R_REPO}/runs/token_wm/", dest)
    log(f"synced -> {dest}")


def cmd_s3renew(a):
    """Push a fresh 7-day upload policy to a running run (the uploader re-reads it per file)."""
    pol = REPO / "runs" / "pod" / ".s3_policy.json"
    pol.write_text(json.dumps(s3_upload_policy(a.name)))
    pol.chmod(0o600)
    rsync_to([pol], f"{R_REPO}/runs/token_wm/{a.name}/.s3_policy.json")
    pol.unlink()
    log(f"renewed S3 upload policy for {a.name} (7 days)")


def cmd_s3ls(a):
    """What a run has persisted to S3 (key, size, time)."""
    c, bucket = s3()
    prefix = f"{S3_PREFIX}/runs/{a.name}/"
    rows = [o for p in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in p.get("Contents", [])]
    for o in sorted(rows, key=lambda o: o["Key"]):
        print(f"{o['LastModified']:%Y-%m-%d %H:%M}  {o['Size'] / 1e6:9.2f} MB  {o['Key']}")
    print(f"{len(rows)} objects under s3://{bucket}/{prefix}")


def cmd_s3pull(a):
    """Download a run's persisted files from S3 to runs/token_wm/<name>/."""
    c, bucket = s3()
    prefix = f"{S3_PREFIX}/runs/{a.name}/"
    dest = REPO / "runs" / "token_wm" / a.name
    n = 0
    for p in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in p.get("Contents", []):
            f = dest / o["Key"][len(prefix):]
            f.parent.mkdir(parents=True, exist_ok=True)
            c.download_file(bucket, o["Key"], str(f))
            n += 1
    log(f"{n} files -> {dest}")


def cmd_ssh(a):
    print(run(" ".join(a.command), capture=True, check=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    p = sub.add_parser("create"); p.add_argument("--confirm", action="store_true")
    p.add_argument("--gpu"); p.add_argument("--name", default="moba-wm")
    p = sub.add_parser("kill"); p.add_argument("--confirm", action="store_true")
    sub.add_parser("bootstrap")
    p = sub.add_parser("data"); p.add_argument("--corpus", default="mix100")
    p.add_argument("--games", help="only the files holding these games, e.g. 0-19,900-999 (default: all)")
    p = sub.add_parser("push"); p.add_argument("--corpus", required=True)
    p = sub.add_parser("train"); p.add_argument("name"); p.add_argument("train_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("status"); p.add_argument("name", nargs="?"); p.add_argument("--lines", type=int, default=15)
    sub.add_parser("sync")
    p = sub.add_parser("ssh"); p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("s3ls"); p.add_argument("name")
    p = sub.add_parser("s3renew"); p.add_argument("name")
    p = sub.add_parser("s3pull"); p.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "train" and a.train_args and a.train_args[0] == "--":
        a.train_args = a.train_args[1:]
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
