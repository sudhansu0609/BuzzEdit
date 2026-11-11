"""The Motion Gallery (MOTION_GRAPHICS_PLAN.md M11): renders every registered
motion composition with its own sample props into a dated, browsable folder.

POST /api/motion/gallery {themes?, width?, height?} -- starts the render in
    the background (refused with 409 while a production render -- a
    `rendering` job or a `presentation` pass -- is running); returns
    {state, folder, count, errors, total}. `total` is the planned render
    count, known immediately (the compositions are listed up front) so the
    UI can show "Rendering N effects..." without waiting on the first one.
GET  /api/motion/gallery -- {state, folder, count, errors}.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from presentation import motion_gallery

logger = logging.getLogger("routes.motion_gallery")

router = APIRouter()

_state: Dict[str, Any] = {"state": "idle", "folder": None, "count": 0, "errors": [], "total": 0,
                          "message": ""}


class GalleryRequest(BaseModel):
    themes: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _render_busy() -> Optional[str]:
    """Why a gallery render must wait, or None -- checked against the same
    job registries the UI already polls (`routes.rendering.jobs`, the
    presentation pass's own `_jobs`), so the gallery never fights a real
    production render for the GPU/CPU."""
    from routes import presentation as presentation_routes
    from routes import rendering as rendering_routes
    for job in rendering_routes.jobs.values():
        if job.get("status") == "running":
            return f"render job {job.get('id')}"
    for job in presentation_routes._jobs.values():
        if job.get("status") == "running":
            return f"presentation job {job.get('id')}"
    return None


@router.post("/gallery")
async def start_gallery(body: GalleryRequest):
    if _state.get("state") == "running":
        return _public_state()
    reason = _render_busy()
    if reason:
        raise HTTPException(
            status_code=409,
            detail=f"A production render is running ({reason}); try the motion gallery again once it finishes.")

    themes = body.themes or list(motion_gallery.DEFAULT_THEMES)
    width = int(body.width or motion_gallery.DEFAULT_WIDTH)
    height = int(body.height or motion_gallery.DEFAULT_HEIGHT)

    try:
        comps = await asyncio.to_thread(motion_gallery.dump_compositions)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not list motion compositions: {e}")

    total = len(comps) * len(themes)
    _state.update(state="running", folder=None, count=0, errors=[], total=total,
                  message=f"Rendering {total} effect(s)...")
    asyncio.create_task(_run(comps, themes, width, height))
    return _public_state()


async def _run(comps, themes: List[str], width: int, height: int) -> None:
    def cb(fraction: float, message: str = ""):
        if message:
            _state["message"] = message

    try:
        result = await asyncio.to_thread(motion_gallery.run_gallery, themes, width, height, comps, cb)
        _state.update(state="completed", folder=result.get("folder"), count=result.get("count", 0),
                      errors=result.get("errors", []), message="Motion gallery ready")
    except Exception as e:  # noqa: BLE001
        logger.exception("Motion gallery failed")
        _state.update(state="failed", errors=[{"id": "gallery", "reason": str(e)[:300]}],
                      message=f"Motion gallery failed: {e}")


def _public_state() -> Dict[str, Any]:
    return {"state": _state.get("state", "idle"), "folder": _state.get("folder"),
            "count": _state.get("count", 0), "errors": _state.get("errors", []),
            "total": _state.get("total", 0), "message": _state.get("message", "")}


@router.get("/gallery")
async def gallery_status():
    return _public_state()
