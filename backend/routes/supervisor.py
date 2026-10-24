"""Re-parenting the headless BuzzEdit's own Electron watchdog
(STUDIO_DIRECTOR_PLAN.md enhancement 4): Electron's own `BUZZCAF_STUDIO_PID`
env var is fixed for the process's whole life, but the Studio that finds a
BuzzEdit already running headless on its port adopts it instead of spawning a
second one -- it has a different pid than the one that started this process.
`POST /api/supervisor/parent` is how it says so; the Electron watchdog polls
`GET /api/supervisor/parent` instead of the env var once this is set.

POST /api/supervisor/parent {pid}  -> {pid}
GET  /api/supervisor/parent        -> {pid}  (0 when none has been set)
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from config import DATA_DIR

logger = logging.getLogger("routes.supervisor")
router = APIRouter()

STATE_PATH = DATA_DIR / "supervisor_parent.json"


class ParentBody(BaseModel):
    pid: int


def _save(pid: int) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = f"{STATE_PATH}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": pid}, f)
        os.replace(tmp, str(STATE_PATH))
    except Exception as exc:  # noqa: BLE001 - the Studio's own call still answers
        logger.warning("supervisor: could not persist the parent pid (%s)", exc)


def _load() -> int:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("pid") or 0)
    except Exception:
        return 0


@router.post("/parent")
async def set_parent(body: ParentBody) -> Dict[str, Any]:
    _save(int(body.pid))
    logger.info("supervisor: the Electron watchdog now follows pid %s.", body.pid)
    return {"pid": int(body.pid)}


@router.get("/parent")
async def get_parent() -> Dict[str, Any]:
    return {"pid": _load()}
