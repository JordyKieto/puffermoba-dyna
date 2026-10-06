"""Credential resolution (moved from racewm.secrets when the race code left this repository; the Keychain
service names and file paths are unchanged, so existing stores keep working).

Precedence, first hit wins:

    1. the process environment            (CI, containers, one-off overrides)
    2. the macOS Keychain                 (primary store on this machine)
    3. ~/Documents/dev/secrets/race-wm.env   (offline backup / non-macOS)
    4. ~/Documents/dev/sunnypilot/secrets.yaml  (legacy, RunPod + Upstash)

Values are never logged or printed. `describe()` reports which keys exist and
how many characters they are, never what they contain.

Nothing here parses ~/.netrc or ~/.aws/credentials: wandb and boto3 read those
themselves, so copying secrets into this process would add risk and no benefit.

Keychain items are stored as generic passwords under the service name
`race-wm:<KEY>`, e.g.:

    security find-generic-password -a "$USER" -s "race-wm:S3_BUCKET" -w
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYCHAIN_PREFIX = "race-wm"

DEFAULT_PATHS = (
    Path(os.environ["RACE_SECRETS_FILE"]) if os.environ.get("RACE_SECRETS_FILE") else None,
    Path.home() / "Documents" / "dev" / "secrets" / "race-wm.env",
    Path.home() / "Documents" / "dev" / "sunnypilot" / "secrets.yaml",
)

WANDB_KEYS = ("WANDB_API_KEY", "WANDB_KEY", "WANDB_TOKEN")
WANDB_PROJECT_KEYS = ("WANDB_PROJECT",)
WANDB_ENTITY_KEYS = ("WANDB_ENTITY",)
S3_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
           "AWS_REGION", "S3_BUCKET", "S3_PREFIX", "S3_ENDPOINT_URL",
           "AWS_ENDPOINT_URL")

ALL_KEYS = WANDB_KEYS[:1] + WANDB_PROJECT_KEYS + WANDB_ENTITY_KEYS + S3_KEYS

# Standard stores, checked for existence only -- never parsed. See module docs.
NETRC_PATH = Path.home() / ".netrc"
AWS_CREDENTIALS_PATH = Path.home() / ".aws" / "credentials"
AWS_PROFILE = "race-wm"


# ------------------------------------------------------------------ keychain
def keychain_get(key: str) -> str | None:
    if os.uname().sysname != "Darwin":
        return None
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""),
             "-s", f"{KEYCHAIN_PREFIX}:{key}", "-w"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def keychain_set(key: str, value: str) -> bool:
    if os.uname().sysname != "Darwin":
        return False
    r = subprocess.run(
        ["security", "add-generic-password", "-a", os.environ.get("USER", ""),
         "-s", f"{KEYCHAIN_PREFIX}:{key}", "-w", value,
         "-j", "race-wm project credential", "-U"],
        capture_output=True, text=True)
    return r.returncode == 0


def keychain_delete(key: str) -> bool:
    if os.uname().sysname != "Darwin":
        return False
    r = subprocess.run(
        ["security", "delete-generic-password", "-a", os.environ.get("USER", ""),
         "-s", f"{KEYCHAIN_PREFIX}:{key}"],
        capture_output=True, text=True)
    return r.returncode == 0


# ---------------------------------------------------------------- flat files
def _parse(path: Path) -> dict[str, str]:
    """Plain KEY=value lines. sunnypilot's secrets.yaml is this format too,
    despite the extension."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_file() -> dict[str, str]:
    for p in DEFAULT_PATHS:
        if p and p.exists():
            return _parse(p)
    return {}


def secrets_file_path() -> str | None:
    return next((str(p) for p in DEFAULT_PATHS if p and p.exists()), None)


# ------------------------------------------------------------------ resolve
def get(*names: str, default: str | None = None) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    for n in names:
        v = keychain_get(n)
        if v:
            return v
    data = load_file()
    for n in names:
        if data.get(n):
            return data[n]
    return default


def export_to_env(*names: str) -> list[str]:
    """Push secrets into os.environ for libraries that only read env vars."""
    done = []
    for n in names:
        if not os.environ.get(n):
            v = get(n)
            if v:
                os.environ[n] = v
                done.append(n)
    return done


def wandb_key() -> str | None:
    return get(*WANDB_KEYS)


def wandb_project(default: str = "race-wm") -> str:
    return get(*WANDB_PROJECT_KEYS, default=default) or default


def wandb_entity() -> str | None:
    return get(*WANDB_ENTITY_KEYS)


def s3_config() -> dict[str, str]:
    return {k: v for k in S3_KEYS if (v := get(k))}


def wandb_available() -> bool:
    return wandb_key() is not None or NETRC_PATH.exists()


def s3_available() -> bool:
    cfg = s3_config()
    return ("AWS_ACCESS_KEY_ID" in cfg and "AWS_SECRET_ACCESS_KEY" in cfg) \
        or AWS_CREDENTIALS_PATH.exists()


def describe() -> dict[str, object]:
    """What is available and where it came from. Never any values."""
    where = {}
    for k in ALL_KEYS:
        if os.environ.get(k):
            src, n = "env", len(os.environ[k])
        elif (v := keychain_get(k)):
            src, n = "keychain", len(v)
        elif (v := load_file().get(k)):
            src, n = "file", len(v)
        else:
            continue
        where[k] = {"source": src, "chars": n}
    return {
        "keychain_prefix": KEYCHAIN_PREFIX,
        "backup_file": secrets_file_path(),
        "aws_cli_profile": AWS_PROFILE,
        "resolved": where,
        "wandb_usable": wandb_available(),
        "s3_usable": s3_available(),
    }


# ------------------------------------------------------------------- sync
def sync_backup(path: Path | None = None) -> Path:
    """Rewrite the offline backup file from the Keychain. Keychain is the
    source of truth; this exists so a rotated key survives a lost keychain."""
    path = path or (Path.home() / "Documents" / "dev" / "secrets" / "race-wm.env")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    lines = [
        "# Backup of the macOS Keychain entries under service race-wm:<KEY>.",
        "# Keychain is the source of truth; regenerate with:",
        "#     python3 -m racewm.secrets --sync-backup",
        "# AWS here is IAM user race-wm, scoped to s3://puffertank/* only.",
        "",
    ]
    for k in ALL_KEYS:
        v = keychain_get(k)
        if v:
            lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return path


def sync_aws_profile(profile: str = AWS_PROFILE) -> bool:
    """Point an aws-cli named profile at the Keychain credentials. Never
    touches [default]."""
    ak, sk = keychain_get("AWS_ACCESS_KEY_ID"), keychain_get("AWS_SECRET_ACCESS_KEY")
    region = keychain_get("AWS_DEFAULT_REGION") or "us-west-1"
    if not (ak and sk):
        return False
    for key, val in (("aws_access_key_id", ak), ("aws_secret_access_key", sk),
                     ("region", region), ("output", "json")):
        if subprocess.run(["aws", "configure", "set", key, val, "--profile", profile],
                          capture_output=True).returncode != 0:
            return False
    return True


if __name__ == "__main__":
    import argparse
    import getpass
    import json

    ap = argparse.ArgumentParser(description="Inspect and rotate race-wm credentials.")
    ap.add_argument("--set", metavar="KEY",
                    help="store a value in the Keychain (prompted, not echoed)")
    ap.add_argument("--delete", metavar="KEY", help="remove a Keychain entry")
    ap.add_argument("--sync-backup", action="store_true",
                    help="rewrite the offline backup file from the Keychain")
    ap.add_argument("--aws-profile", action="store_true",
                    help="rewrite the race-wm aws-cli profile from the Keychain")
    args = ap.parse_args()

    if args.set:
        value = getpass.getpass(f"value for {args.set} (input hidden): ").strip()
        if not value:
            raise SystemExit("empty value, nothing stored")
        if not keychain_set(args.set, value):
            raise SystemExit("keychain write failed")
        print(f"stored race-wm:{args.set} ({len(value)} chars)")
        print("backup ->", sync_backup())
        if args.set.startswith("AWS_"):
            print("aws profile updated:", sync_aws_profile())
    elif args.delete:
        print("deleted" if keychain_delete(args.delete) else "not found")
    elif args.sync_backup:
        print("backup ->", sync_backup())
    elif args.aws_profile:
        print("aws profile updated:", sync_aws_profile())
    else:
        print(json.dumps(describe(), indent=2))
