"""Automatic off-pod persistence: upload run files to S3 as they are written.

Pods never hold AWS credentials. When `tools/pod.py train` launches a run, the Mac
signs one S3 POST policy scoped to s3://<bucket>/moba/runs/<run>/ (valid 7 days)
and copies it to the pod as <run dir>/.s3_policy.json (mode 600). The trainer
calls persist(path) after writing each checkpoint, stats file and log; the file
is POSTed in a background thread and verified: S3's ETag (the MD5 of the object
for a single-part upload) must equal the local file's MD5. Every result is
printed ("S3 ok ..." / "S3 FAILED ...") so a missing upload is visible in the log.

Without a policy file (local runs) persist() does nothing.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_pending: list[threading.Thread] = []


def _policy(path: Path):
    for d in (path.parent, *path.parents):
        f = d / ".s3_policy.json"
        if f.exists():
            return json.loads(f.read_text()), d
    return None, None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload(path: Path, pol: dict, root: Path, tries: int = 5):
    import requests
    key = pol["prefix"] + path.relative_to(root).as_posix()
    data = path.read_bytes()                      # snapshot: the file may be rewritten later
    md5 = hashlib.md5(data).hexdigest()
    for attempt in range(tries):
        try:
            fields = dict(pol["fields"], key=key)
            r = requests.post(pol["url"], data=fields, files={"file": (path.name, data)}, timeout=300)
            etag = r.headers.get("ETag", "").strip('"')
            if r.status_code in (200, 201, 204) and etag == md5:
                print(f"S3 ok {key} ({len(data) / 1e6:.1f} MB, md5 {md5[:8]})", flush=True)
                return True
            err = f"HTTP {r.status_code}, etag {etag or '-'} vs md5 {md5}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001 -- report and retry
            err = repr(e)
        time.sleep(2 ** attempt)
    print(f"S3 FAILED {key}: {err}", flush=True)
    return False


def persist(path, wait: bool = False):
    """Upload `path` to the run's S3 prefix in the background (no-op without a policy)."""
    path = Path(path).resolve()
    pol, root = _policy(path)
    if pol is None:
        return
    if pol.get("expires", 1e18) < time.time():
        print(f"S3 FAILED {path.name}: upload policy expired", flush=True)
        return
    t = threading.Thread(target=_upload, args=(path, pol, root), daemon=False)
    with _lock:
        _pending.append(t)
    t.start()
    if wait:
        t.join()


def flush():
    """Wait for every pending upload (call at the end of a run)."""
    with _lock:
        ts = list(_pending)
    for t in ts:
        t.join()
