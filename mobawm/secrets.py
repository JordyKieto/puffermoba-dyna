"""Credential resolution. No plaintext credential files are read.

Lookup order, first hit wins:
    1. the process environment          (pods: sourced from a mode-600 .pod_secrets written at launch; CI)
    2. the macOS login Keychain, item "dev:<NAME>"      (the shared convention for every project on this Mac;
       manage with ~/Documents/dev/secrets/devsecrets.py: `list`, `set NAME`)
    3. the macOS login Keychain, item "race-wm:<NAME>"  (legacy items, read-only)

Values are never printed, logged or passed on a command line.

Standard names: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET, RUNPOD_API_KEY,
UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT.
"""
from __future__ import annotations

import os
import subprocess
import sys

PREFIXES = ("dev", "race-wm")
ALIASES = {"RUNPOD": "RUNPOD_API_KEY"}          # older call sites


def _keychain(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    for prefix in PREFIXES:
        try:
            r = subprocess.run(["security", "find-generic-password", "-a", os.environ.get("USER", ""),
                                "-s", f"{prefix}:{name}", "-w"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.rstrip("\n")
    return None


def get(*names: str, default: str | None = None) -> str | None:
    """The first of `names` that is set in the environment or the Keychain."""
    for n in names:
        n = ALIASES.get(n, n)
        v = os.environ.get(n) or _keychain(n)
        if v:
            return v
    return default


def export_to_env(*names: str) -> list[str]:
    """Copy credentials into os.environ for libraries that only read environment variables (boto3, wandb)."""
    done = []
    for n in names:
        v = get(n)
        if v:
            os.environ[ALIASES.get(n, n)] = v
            done.append(n)
    return done
