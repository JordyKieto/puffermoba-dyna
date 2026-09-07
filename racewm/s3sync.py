"""Push and pull corpora, checkpoints, and replays to S3.

    python3 -m racewm.s3sync push data/corpus
    python3 -m racewm.s3sync pull corpus --dest data/corpus
    python3 -m racewm.s3sync ls

Corpora are large but deterministically regenerable from
`tools/make_dataset.sh` plus the manifest, so this is for sharing with a GPU
box rather than for archival. The manifest travels with the data: without it
you cannot tell which track ranges a shard came from.

Credentials come from racewm.secrets (Keychain first). The IAM user is scoped
to the dataset buckets only -- see docs/SECRETS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import secrets


def _client():
    secrets.export_to_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                          "AWS_DEFAULT_REGION")
    import boto3
    return boto3.client("s3")


def _bucket() -> str:
    b = secrets.get("S3_BUCKET")
    if not b:
        raise SystemExit("S3_BUCKET is not set (python3 -m racewm.secrets)")
    return b


def _prefix() -> str:
    return (secrets.get("S3_PREFIX") or "race-wm").strip("/")


def _etag_matches(s3, bucket, key, path: Path) -> bool:
    """Skip re-uploading an unchanged file. Only valid for single-part uploads,
    where ETag is the MD5; multipart ETags carry a -N suffix and are skipped."""
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    etag = head["ETag"].strip('"')
    if "-" in etag or head["ContentLength"] != path.stat().st_size:
        return False
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == etag


def push(local: str | Path, name: str | None = None, workers: int = 8,
         dry_run: bool = False) -> int:
    local = Path(local)
    if not local.exists():
        raise SystemExit(f"{local} does not exist")
    name = name or local.name
    s3, bucket, root = _client(), _bucket(), f"{_prefix()}/{name}"

    files = sorted(p for p in local.rglob("*") if p.is_file()) if local.is_dir() else [local]
    total = sum(p.stat().st_size for p in files)
    print(f"push {local} -> s3://{bucket}/{root}/  ({len(files)} files, {total/1e6:.1f} MB)")

    done = skipped = 0

    def one(p: Path):
        nonlocal done, skipped
        rel = p.relative_to(local) if local.is_dir() else p.name
        key = f"{root}/{rel}"
        if _etag_matches(s3, bucket, key, p):
            skipped += 1
            return
        if not dry_run:
            s3.upload_file(str(p), bucket, key)
        done += 1
        print(f"  up   {rel} ({p.stat().st_size/1e6:.1f} MB)")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, files))
    print(f"uploaded {done}, unchanged {skipped}")
    return 0


def pull(name: str, dest: str | Path, workers: int = 8) -> int:
    s3, bucket, root = _client(), _bucket(), f"{_prefix()}/{name}"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    keys = []
    tok = None
    while True:
        kw = {"Bucket": bucket, "Prefix": root + "/"}
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        keys += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    if not keys:
        raise SystemExit(f"nothing at s3://{bucket}/{root}/")

    total = sum(s for _, s in keys)
    print(f"pull s3://{bucket}/{root}/ -> {dest}  ({len(keys)} files, {total/1e6:.1f} MB)")

    def one(item):
        key, size = item
        rel = key[len(root) + 1:]
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == size:
            return
        s3.download_file(bucket, key, str(out))
        print(f"  down {rel} ({size/1e6:.1f} MB)")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, keys))
    print("done")
    return 0


def ls(name: str | None = None) -> int:
    s3, bucket = _client(), _bucket()
    prefix = f"{_prefix()}/" + (f"{name}/" if name else "")
    r = s3.list_objects_v2(Bucket=bucket, Prefix=prefix,
                           Delimiter="" if name else "/")
    if name:
        items = r.get("Contents", [])
        print(f"s3://{bucket}/{prefix}  ({len(items)} objects)")
        for o in items:
            print(f"  {o['Size']/1e6:>9.1f} MB  {o['Key'][len(prefix):]}")
    else:
        pres = r.get("CommonPrefixes", [])
        print(f"s3://{bucket}/{_prefix()}/")
        for p in pres:
            print("  " + p["Prefix"][len(prefix):].rstrip("/"))
        if not pres:
            print("  (empty)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("local"); p.add_argument("--name")
    p.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("pull"); q.add_argument("name"); q.add_argument("--dest", required=True)
    l = sub.add_parser("ls"); l.add_argument("name", nargs="?")
    a = ap.parse_args()

    if a.cmd == "push":
        return push(a.local, a.name, dry_run=a.dry_run)
    if a.cmd == "pull":
        return pull(a.name, a.dest)
    return ls(a.name)


if __name__ == "__main__":
    sys.exit(main())
