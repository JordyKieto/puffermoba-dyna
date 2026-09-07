"""Checkpoint save / load / resume.

Every checkpoint carries the state-contract hash. A model trained on contract
v1 loaded against a changed contract is silently wrong -- the input columns
mean different things -- so loading refuses rather than warning. That is the
whole reason contract.schema_hash() exists.

Layout, one directory per run:

    checkpoints/<run>/
        step_000500.pt      periodic
        latest.pt           symlink-free copy of the newest step
        best.pt             newest checkpoint that improved the tracked metric
        manifest.json       run metadata, metric history, file index
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch

from . import contract

FORMAT_VERSION = 1


@dataclass
class RunManifest:
    run_name: str
    schema_hash: str
    state_version: int
    format_version: int = FORMAT_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    config: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    best_step: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class Checkpointer:
    def __init__(self, root: str | Path, run_name: str, config: dict | None = None,
                 metric_name: str = "val_nmse", metric_mode: str = "min"):
        if metric_mode not in ("min", "max"):
            raise ValueError("metric_mode must be 'min' or 'max'")
        self.dir = Path(root) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        mpath = self.dir / "manifest.json"
        if mpath.exists():
            raw = json.loads(mpath.read_text())
            self.manifest = RunManifest(**raw)
            if self.manifest.schema_hash != contract.schema_hash():
                raise RuntimeError(
                    f"{self.dir} was written under contract {self.manifest.schema_hash}, "
                    f"current contract is {contract.schema_hash()}. Start a new run.")
        else:
            self.manifest = RunManifest(
                run_name=run_name,
                schema_hash=contract.schema_hash(),
                state_version=contract.STATE_VERSION,
                config=config or {},
            )
            self._flush()

    # ------------------------------------------------------------------ save
    def save(self, *, step: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer | None = None,
             scheduler: Any = None, metrics: dict[str, float] | None = None,
             extra: dict[str, Any] | None = None) -> Path:
        metrics = metrics or {}
        payload = {
            "format_version": FORMAT_VERSION,
            "schema_hash": contract.schema_hash(),
            "state_version": contract.STATE_VERSION,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": self.manifest.config,
            "extra": extra or {},
            "torch_version": torch.__version__,
            "saved_at": time.time(),
        }
        path = self.dir / f"step_{step:06d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        # Write then rename: a crash mid-write leaves the previous checkpoint
        # intact instead of a truncated file that torch.load will choke on.
        torch.save(payload, tmp)
        tmp.replace(path)
        shutil.copyfile(path, self.dir / "latest.pt")

        entry = {"step": step, "path": path.name, "metrics": metrics,
                 "saved_at": payload["saved_at"]}
        self.manifest.history.append(entry)

        value = metrics.get(self.metric_name)
        if value is not None:
            better = (self.manifest.best_metric is None
                      or (value < self.manifest.best_metric if self.metric_mode == "min"
                          else value > self.manifest.best_metric))
            if better:
                self.manifest.best_metric = float(value)
                self.manifest.best_step = step
                shutil.copyfile(path, self.dir / "best.pt")

        self.manifest.updated_at = time.time()
        self._flush()
        return path

    def _flush(self) -> None:
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(asdict(self.manifest), indent=2))
        tmp.replace(self.dir / "manifest.json")

    # ------------------------------------------------------------------ load
    def resume(self, model: torch.nn.Module,
               optimizer: torch.optim.Optimizer | None = None,
               scheduler: Any = None, which: str = "latest",
               map_location: str = "cpu") -> int:
        """Restore in place and return the step to continue from. 0 if absent."""
        path = self.dir / f"{which}.pt"
        if not path.exists():
            return 0
        step = load_into(path, model, optimizer, scheduler, map_location=map_location)
        return step


def load_into(path: str | Path, model: torch.nn.Module,
              optimizer: torch.optim.Optimizer | None = None,
              scheduler: Any = None, map_location: str = "cpu",
              strict_contract: bool = True) -> int:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if strict_contract and ckpt.get("schema_hash") != contract.schema_hash():
        raise RuntimeError(
            f"{path} was trained under contract {ckpt.get('schema_hash')}, current "
            f"contract is {contract.schema_hash()}. The observation columns no "
            f"longer mean the same thing; retrain or pin the old contract.")

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


def read_manifest(root: str | Path, run_name: str) -> RunManifest:
    return RunManifest(**json.loads((Path(root) / run_name / "manifest.json").read_text()))
