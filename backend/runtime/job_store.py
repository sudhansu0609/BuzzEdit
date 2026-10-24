"""Durable job state for BuzzEdit's background jobs (STUDIO_DIRECTOR_PLAN.md
enhancement 2; Fable review row 7): the vibe reel (`routes/reels.py`) and
narration (`audio/narration.py`) keep their `_jobs` dict in memory only, so a
BuzzEdit restart loses the job the Studio was following -- its follower got a
404 and raised.

Every job write also lands as one small JSON file under
`DATA_DIR/jobs/<kind>/<job_id>.json` (gitignored, like `data/master_control.json`):
a GET for a job no longer in memory reads it from here instead of 404-ing, and
`sweep_stale(kind)` -- run once per kind at import, i.e. at startup -- marks
any job still "queued"/"running" on disk `failed` ("BuzzEdit restarted"):
nothing in a just-started process is running it, so it is lost work, not a
job to keep polling forever.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import DATA_DIR

logger = logging.getLogger("runtime.job_store")

JOBS_ROOT = DATA_DIR / "jobs"
RESTARTED_ERROR = "BuzzEdit restarted"
ACTIVE_STATES = ("queued", "running")

_lock = threading.Lock()


def _dir(kind: str) -> Path:
    path = JOBS_ROOT / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(kind: str, job_id: str) -> Path:
    return _dir(kind) / f"{job_id}.json"


def save(kind: str, job_id: str, data: Dict[str, Any]) -> None:
    """Write `data` for `kind`/`job_id`, atomically. Never raises -- a job
    still runs (and answers in memory) even when its disk copy could not be
    written."""
    path = _path(kind, job_id)
    with _lock:
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp, str(path))
        except Exception as exc:  # noqa: BLE001 - the in-memory job still holds
            logger.warning("job_store: could not persist %s/%s (%s)", kind, job_id, exc)


def load(kind: str, job_id: str) -> Optional[Dict[str, Any]]:
    """`data` for `kind`/`job_id` as last saved, or None (never made, or
    unreadable)."""
    try:
        with open(_path(kind, job_id), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def all_jobs(kind: str) -> Dict[str, Dict[str, Any]]:
    """Every job of `kind` on disk, keyed by job id. A file that fails to
    parse is skipped, not raised."""
    out: Dict[str, Dict[str, Any]] = {}
    for file in _dir(kind).glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            out[file.stem] = data
    return out


def sweep_stale(kind: str) -> int:
    """At startup: mark every `kind` job still "queued"/"running" on disk
    `failed` (`RESTARTED_ERROR`) -- this process just started, so nothing of
    it is actually running that job. Returns how many were marked."""
    marked = 0
    for job_id, data in all_jobs(kind).items():
        if data.get("state") in ACTIVE_STATES:
            data["state"] = "failed"
            data["error"] = RESTARTED_ERROR
            save(kind, job_id, data)
            marked += 1
    if marked:
        logger.warning("job_store: marked %d stale %r job(s) failed (%s) at startup.",
                       marked, kind, RESTARTED_ERROR)
    return marked
