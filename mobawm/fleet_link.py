"""Pod <-> Mac link over Upstash Redis REST: heartbeats and cross-pod weight averaging (FedAvg) for world-model
training on many pods. Ported from zelda-RL (zelda_rl/fleet_link.py), which ran 20-pod fleets with it: RunPod pods
cannot reach each other, so they meet only at shared Upstash keys.

Credentials: UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN from the environment (pods: sourced from the mode-600
.pod_secrets), else sunnypilot's secrets.yaml (KEY=VALUE) on the Mac. Keys are namespaced `ptk:{experiment}:...`;
values are never logged.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time

import requests

CHUNK = 400_000          # base64 chars per value (Upstash request limit headroom)
HEARTBEAT_TTL = 300
SYNC_TTL = 600           # sync blobs are only needed for one update; keep the Upstash footprint small at 20 ranks
BLOB_TTL = 24 * 3600


class Upstash:
    def __init__(self):
        url, token = os.environ.get("UPSTASH_REDIS_REST_URL"), os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if not (url and token):
            from pathlib import Path
            p = Path.home() / "Documents/dev/sunnypilot/secrets.yaml"
            kv = dict(l.split("=", 1) for l in (p.read_text().splitlines() if p.exists() else []) if "=" in l)
            url, token = (kv.get("UPSTASH_REDIS_REST_URL") or "").strip(), (kv.get("UPSTASH_REDIS_REST_TOKEN") or "").strip()
        if not (url and token):
            raise KeyError("UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN not configured")
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}"})

    TIMEOUT = (5, 20)        # connect, read. A pod that can't reach Upstash must fail fast, not hang minutes (seen live).

    def cmd(self, *args, attempts: int = 3):
        for attempt in range(attempts):
            try:
                r = self.s.post(self.url, json=[str(a) for a in args], timeout=self.TIMEOUT)
                r.raise_for_status()
                return r.json().get("result")
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                time.sleep(1.0 * (attempt + 1))

    def pipeline(self, commands: list[list]):
        r = self.s.post(f"{self.url}/pipeline", json=[[str(a) for a in c] for c in commands], timeout=self.TIMEOUT)
        r.raise_for_status()
        return [x.get("result") for x in r.json()]

    # ---- blobs (chunked, base64) ----
    def put_blob(self, key: str, data: bytes, ttl: int = BLOB_TTL) -> dict:
        b64 = base64.b64encode(data).decode()
        chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
        for i, c in enumerate(chunks):
            self.cmd("SET", f"{key}:chunk{i}", c, "EX", ttl)
        meta = {"chunks": len(chunks), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.cmd("SET", f"{key}:meta", json.dumps(meta), "EX", ttl)
        return meta

    def get_blob(self, key: str) -> bytes | None:
        meta = self.cmd("GET", f"{key}:meta")
        if not meta:
            return None
        meta = json.loads(meta)
        parts = self.pipeline([["GET", f"{key}:chunk{i}"] for i in range(meta["chunks"])])
        if any(p is None for p in parts):
            return None
        data = base64.b64decode("".join(parts))
        return data if hashlib.sha256(data).hexdigest() == meta["sha256"] else None


def _state_bytes(obj) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _state_load(data: bytes):
    import torch
    return torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")


class FleetLink:
    """Used by mobawm.wm.train on each pod (rank) of a fleet experiment."""

    def __init__(self, experiment: str, rank: int, world_size: int, launch_id: str = "0"):
        self.exp, self.rank, self.world = experiment, rank, world_size
        self.launch = launch_id          # sync keys are per launch: a relaunch must never average with stale blobs
        self.u = Upstash()
        self.u.cmd("SADD", "ptk:experiments", experiment)

        import queue
        import threading
        self._q: "queue.Queue" = queue.Queue(maxsize=8)
        threading.Thread(target=self._worker, daemon=True, name="fleet-link").start()

    def _worker(self):
        while True:
            fn, args = self._q.get()
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 -- telemetry must never take training down
                print(f"[fleet-link] {fn.__name__} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)

    def _submit(self, fn, *args):
        try:
            self._q.put_nowait((fn, args))
        except Exception:  # noqa: BLE001 -- queue full (Upstash unreachable): drop, don't block the trainer
            pass

    def heartbeat(self, metrics: dict) -> None:
        """Non-blocking."""
        self._submit(self._heartbeat, {"rank": self.rank, "t": time.time(), **metrics})

    def _heartbeat(self, payload: dict) -> None:
        self.u.cmd("SET", f"ptk:{self.exp}:hb:{self.rank}", json.dumps(payload, default=float), "EX", HEARTBEAT_TTL)

    def publish_policy(self, model, config: dict, update: int, global_step: int, metrics: dict) -> None:
        """Non-blocking: snapshots the weights now, uploads in the background."""
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._submit(self._publish_policy, state, config, update, global_step, metrics)

    def live_peers(self, max_age: float = 150.0) -> set[int]:
        keys = self.u.cmd("KEYS", f"ptk:{self.exp}:hb:*") or []
        vals = self.u.pipeline([["GET", k] for k in keys]) if keys else []
        now = time.time()
        return {h["rank"] for h in (json.loads(v) for v in vals if v) if now - h["t"] < max_age and h["rank"] != self.rank}

    def _publish_policy(self, state, config: dict, update: int, global_step: int, metrics: dict) -> None:
        """Latest policy (weights + config) for Mac-side play/eval. One pointer per experiment: newest wins."""
        payload = {"model": state, "config": config, "update": update, "global_step": global_step,
                   "rank": self.rank, "best": metrics, "optimizer": None, "episode_count": metrics.get("episode_count", 0)}
        key = f"ptk:{self.exp}:policy:r{self.rank}:u{update}"
        cur = self.u.cmd("GET", f"ptk:{self.exp}:policy:latest")
        if cur and json.loads(cur).get("update", -1) > update:  # a slow rank must not replace a newer policy
            return
        meta = self.u.put_blob(key, _state_bytes(payload))
        self.u.cmd("SET", f"ptk:{self.exp}:policy:latest",
                   json.dumps({"key": key, "rank": self.rank, "update": update, "t": time.time(), **meta,
                               "metrics": metrics}, default=float), "EX", BLOB_TTL)

    def average_weights(self, model, update: int, timeout: float = 300.0, aggregator: int = 0,
                        straggler_wait: float = 30.0) -> dict:
        """Parameter averaging after PPO update `update`, aggregator pattern, straggler tolerant.

        Every rank uploads its weights. The aggregator waits at most `straggler_wait` for live ranks, averages what
        arrived (rank order, float64 -> bit-identical), publishes `avg` + a `latest` pointer. Followers load their
        update's average, or catch up to any NEWER average; a slow rank never holds the fleet back and never drifts
        (seen live: fast ranks timed out waiting on a slow aggregator and trained alone)."""
        import torch
        root = f"ptk:{self.exp}:sync:{self.launch}"
        base = f"{root}:u{update}"
        t0 = time.time()
        local = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self.u.put_blob(f"{base}:r{self.rank}", _state_bytes(local), ttl=SYNC_TTL)
        if self.rank == aggregator:
            live = self.live_peers()
            expected = live if (live or update > 2) else {r for r in range(self.world) if r != self.rank}
            peers = {self.rank: local}
            wait = straggler_wait if update > 2 else max(straggler_wait, 120.0)
            from concurrent.futures import ThreadPoolExecutor
            while time.time() - t0 < wait and not expected <= set(peers):
                todo = sorted(expected - set(peers))
                with ThreadPoolExecutor(min(16, len(todo))) as ex:           # 20 pods: fetch in parallel, not one by one
                    blobs = list(ex.map(lambda r: self.u.get_blob(f"{base}:r{r}"), todo))
                for r, blob in zip(todo, blobs):
                    if blob is not None:
                        peers[r] = _state_load(blob)
                if not expected <= set(peers):
                    time.sleep(1.0)
            states = [peers[r] for r in sorted(peers)]
            with torch.no_grad():
                avg = {k: (sum(st[k].double() for st in states) / len(states)).to(local[k].dtype) for k in local}
            self.u.put_blob(f"{base}:avg", _state_bytes({"state": avg, "ranks": sorted(peers), "update": update}), ttl=SYNC_TTL)
            self.u.cmd("SET", f"{root}:latest", str(update), "EX", SYNC_TTL)
            model.load_state_dict({k: v.to(model.state_dict()[k].device) for k, v in avg.items()})
            n, got, src = len(states), sorted(peers), update
            old = update - 5
            if old >= 1:
                self._submit(self._cleanup, [f"{root}:u{old}:avg"] + [f"{root}:u{old}:r{r}" for r in range(self.world)])
        else:
            payload, src = None, None
            while time.time() - t0 < timeout and payload is None:
                blob = self.u.get_blob(f"{base}:avg")
                if blob is not None:
                    payload, src = _state_load(blob), update
                    break
                latest = self.u.cmd("GET", f"{root}:latest")
                if latest is not None and int(latest) > update:           # aggregator is ahead: catch up now
                    blob = self.u.get_blob(f"{root}:u{int(latest)}:avg")
                    if blob is not None:
                        payload, src = _state_load(blob), int(latest)
                        break
                time.sleep(1.0)
            if payload is None:
                n, got = 1, [self.rank]
            else:
                model.load_state_dict({k: v.to(model.state_dict()[k].device) for k, v in payload["state"].items()})
                n, got = len(payload["ranks"]), payload["ranks"]
        h = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()[:12]
        return {"sync_ranks": n, "sync_wait_s": time.time() - t0, "weight_hash": h,
                "sync_included_self": self.rank in got, "sync_from_update": src if src is not None else -1}

    def _cleanup(self, keys: list[str]) -> None:
        dels = []
        for k in keys:
            dels.append(f"{k}:meta")
            dels += [f"{k}:chunk{i}" for i in range(8)]
        for i in range(0, len(dels), 200):
            self.u.cmd("DEL", *dels[i:i + 200])

def fetch_latest_policy(experiment: str, dest_dir) -> "tuple[str, dict] | None":
    """Mac side: download the newest published policy for an experiment into dest_dir. Returns (path, pointer)."""
    from pathlib import Path
    u = Upstash()
    ptr = u.cmd("GET", f"ptk:{experiment}:policy:latest")
    if not ptr:
        return None
    ptr = json.loads(ptr)
    dest = Path(dest_dir) / f"{experiment}_r{ptr['rank']}_u{ptr['update']:05d}.pt"
    if not dest.exists():
        blob = u.get_blob(ptr["key"])
        if blob is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    return str(dest), ptr


def clear_sync_keys(experiment: str) -> int:
    """Delete an experiment's sync blobs (call before relaunching a fleet)."""
    u = Upstash()
    keys = u.cmd("KEYS", f"ptk:{experiment}:sync:*") or []
    for i in range(0, len(keys), 100):
        u.cmd("DEL", *keys[i:i + 100])
    return len(keys)


def heartbeats(experiment: str) -> list[dict]:
    u = Upstash()
    keys = u.cmd("KEYS", f"ptk:{experiment}:hb:*") or []
    vals = u.pipeline([["GET", k] for k in keys]) if keys else []
    return sorted((json.loads(v) for v in vals if v), key=lambda h: h["rank"])
