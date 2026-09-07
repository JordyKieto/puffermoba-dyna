"""Run tracking.

Loss curves alone do not answer "how long did this take and how much data did
it need", so every log call also carries wall-clock, samples consumed, and
throughput. Those are the numbers that make two runs comparable when the
architecture changed underneath them.

wandb authenticates through ~/.netrc (written by `wandb login`) or WANDB_API_KEY;
see racewm/secrets.py. If neither is available, or wandb is not installed, the
tracker degrades to console logging rather than failing a training run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import contract, secrets

PROJECT = "race-wm"


class RunTracker:
    def __init__(self, name: str, config: dict[str, Any] | None = None,
                 project: str = PROJECT, group: str | None = None,
                 tags: list[str] | None = None, mode: str = "auto",
                 log_dir: str | Path = "logs"):
        """mode: 'auto' (wandb if usable), 'wandb' (required), 'offline', 'off'."""
        self.name = name
        self.config = dict(config or {})
        # Pin the contract into every run: a loss curve is meaningless without
        # knowing which state layout produced it.
        self.config.update(
            schema_hash=contract.schema_hash(),
            state_version=contract.STATE_VERSION,
            obs_size=contract.OBS_SIZE,
        )
        self.start = time.time()
        self.samples_seen = 0
        self._last_t = self.start
        self._last_samples = 0
        self.run = None
        self.backend = "console"

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._jsonl = open(Path(log_dir) / f"{name}.jsonl", "a")

        if mode == "off":
            return
        try:
            import wandb
        except ImportError:
            if mode == "wandb":
                raise
            print("[tracking] wandb not installed; console + jsonl only")
            return

        if mode == "auto" and not secrets.wandb_available():
            print("[tracking] no wandb credentials; console + jsonl only "
                  "(run `wandb login` or set WANDB_API_KEY)")
            return

        secrets.export_to_env(*secrets.WANDB_KEYS)
        self.run = wandb.init(
            project=project, name=name, group=group, tags=tags,
            config=self.config, mode="offline" if mode == "offline" else "online",
            settings=wandb.Settings(console="off"),
        )
        self.backend = "wandb"

    # ------------------------------------------------------------------ log
    def log(self, step: int, samples: int | None = None, **metrics: float) -> dict:
        """Log metrics, automatically annotated with time and throughput."""
        now = time.time()
        if samples is not None:
            self.samples_seen += samples

        dt = max(now - self._last_t, 1e-9)
        payload = dict(metrics)
        payload.update(
            step=step,
            wall_clock_s=now - self.start,
            samples_seen=self.samples_seen,
            samples_per_sec=(self.samples_seen - self._last_samples) / dt,
        )
        self._last_t = now
        self._last_samples = self.samples_seen

        self._jsonl.write(json.dumps(payload) + "\n")
        self._jsonl.flush()
        if self.run is not None:
            self.run.log(payload, step=step)
        return payload

    def summary(self, **kv: Any) -> None:
        self._jsonl.write(json.dumps({"summary": kv}) + "\n")
        self._jsonl.flush()
        if self.run is not None:
            for k, v in kv.items():
                self.run.summary[k] = v

    def watch_params(self, model) -> int:
        n = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.summary(params_total=n, params_trainable=trainable)
        return n

    def finish(self) -> None:
        total = time.time() - self.start
        self.summary(
            total_wall_clock_s=total,
            total_samples=self.samples_seen,
            mean_samples_per_sec=self.samples_seen / total if total > 0 else 0.0,
        )
        self._jsonl.close()
        if self.run is not None:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False


def format_row(payload: dict, keys: tuple[str, ...] = ()) -> str:
    keys = keys or ("step", "train_loss", "val_nmse", "samples_seen",
                    "samples_per_sec", "wall_clock_s")
    parts = []
    for k in keys:
        if k not in payload:
            continue
        v = payload[k]
        parts.append(f"{k}={v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 1000
                     else f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)
