"""Credential loading.

Reads the same secrets.yaml the sunnypilot repo uses. Despite the extension it
is plain `KEY=value` lines, not YAML, so this parses it the same way
sunnypilot/runpod_fleet/secrets_loader.py does.

Values are never logged or printed. `describe()` reports which keys were found
and nothing about what they contain.

Resolution order, first hit wins:
    1. the process environment
    2. $RACE_SECRETS_FILE
    3. ~/Documents/dev/sunnypilot/secrets.yaml
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATHS = (
    Path(os.environ.get("RACE_SECRETS_FILE", "")) if os.environ.get("RACE_SECRETS_FILE") else None,
    Path.home() / "Documents" / "dev" / "sunnypilot" / "secrets.yaml",
)

WANDB_KEYS = ("WANDB_API_KEY", "WANDB_KEY", "WANDB_TOKEN")
S3_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
           "AWS_REGION", "S3_BUCKET", "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")


def _parse(path: Path) -> dict[str, str]:
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


def get(*names: str, default: str | None = None) -> str | None:
    """First of `names` present in the environment, then in the secrets file."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    data = load_file()
    for n in names:
        v = data.get(n)
        if v:
            return v
    return default


def wandb_key() -> str | None:
    return get(*WANDB_KEYS)


def s3_config() -> dict[str, str]:
    cfg = {}
    for k in S3_KEYS:
        v = get(k)
        if v:
            cfg[k] = v
    return cfg


def export_to_env(*names: str) -> list[str]:
    """Push secrets into os.environ for libraries that only read env vars.
    Returns the names actually set."""
    data = load_file()
    done = []
    for n in names:
        if not os.environ.get(n) and data.get(n):
            os.environ[n] = data[n]
            done.append(n)
    return done


# Standard credential stores. We deliberately do NOT parse these: `wandb reads
# ~/.netrc itself and boto3 reads ~/.aws/credentials itself, so reading them
# here would copy secrets into this process for no benefit. We only check that
# they exist, to give a useful message when they do not.
NETRC_PATH = Path.home() / ".netrc"
AWS_CREDENTIALS_PATH = Path.home() / ".aws" / "credentials"


def wandb_available() -> bool:
    """True if wandb can authenticate: an explicit key, or a ~/.netrc for it to
    read (written by `wandb login`)."""
    return wandb_key() is not None or NETRC_PATH.exists()


def s3_available() -> bool:
    """True if boto3 can authenticate: explicit env/file keys, or an AWS
    profile in the usual place."""
    cfg = s3_config()
    has_keys = "AWS_ACCESS_KEY_ID" in cfg and "AWS_SECRET_ACCESS_KEY" in cfg
    return has_keys or AWS_CREDENTIALS_PATH.exists()


def describe() -> dict[str, object]:
    """What is available. Names and presence only, never values."""
    path = next((str(p) for p in DEFAULT_PATHS if p and p.exists()), None)
    data = load_file()
    return {
        "secrets_file": path,
        "keys_in_secrets_file": sorted(data.keys()),
        "wandb": {
            "explicit_key": wandb_key() is not None,
            "netrc_present": NETRC_PATH.exists(),
            "usable": wandb_available(),
        },
        "s3": {
            "explicit_keys": sorted(s3_config().keys()),
            "aws_credentials_present": AWS_CREDENTIALS_PATH.exists(),
            "usable": s3_available(),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
