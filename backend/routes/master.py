"""The BuzzEdit mirror of the Studio's master pause / resume
(docs/STUDIO_DIRECTOR_PLAN.md Part 1). The Studio calls these whenever its
own pause is switched; see `runtime/master_control.py` for what pausing and
resuming actually do.

GET  /api/master/state
POST /api/master/pause  {mode?: "soft"|"now", reason?: str, by?: str}
POST /api/master/resume
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from runtime import master_control

router = APIRouter()


class PauseBody(BaseModel):
    mode: Optional[str] = "soft"
    reason: Optional[str] = ""
    by: Optional[str] = ""


@router.get("/state")
async def get_state() -> Dict[str, Any]:
    return master_control.state()


@router.post("/pause")
async def post_pause(body: PauseBody) -> Dict[str, Any]:
    return master_control.pause(mode=body.mode or "soft", reason=body.reason or "", by=body.by or "")


@router.post("/resume")
async def post_resume() -> Dict[str, Any]:
    return master_control.resume()
