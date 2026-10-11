"""BuzzEdit -> BuzzAnimate routes, at `/api/buzzanimate` (step E4 of
`BuzzAnimate/docs/ECOSYSTEM_BRIDGE_PLAN.md`).

* `GET  /status[?start=true]` -- is BuzzAnimate up, where, what is open.
* `POST /call {tool, args, start?}` -- one BuzzAnimate tool call.
* `POST /open {buzz_path, start?}` -- open a `.buzz` in BuzzAnimate.
* `POST /render {buzz_path, project_id?, format?, start?}` -- open, export
  (MP4 by default, beside the `.buzz`), wait, and with `project_id` add the
  result to that project's media library (the `media/import` route's own
  path). A background job: poll `GET /render/{job_id}`.

When BuzzAnimate is not running the answer is a 503 that says so; `start:
true` launches its `.vbs` and waits (bounded) for its health route first.
"""

import asyncio
import logging
import re
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from integrations import buzzanimate_client as animate

logger = logging.getLogger("buzzanimate")


# **This machine only.** These routes reach BuzzAnimate, which can open, save
# and export files anywhere on disk, so a web page must not be able to call
# them — and this app's CORS policy is no protection here (BuzzEdit's is
# `allow_origins=["*"]`). A request from a backend (no `Origin`) or from a page
# served on this machine (`http://127.0.0.1:*`, `http://localhost:*`) is
# served; anything else, `null` included, is refused. `null` is what a
# sandboxed iframe on any website sends — and also what a `file://` page
# sends, so a front-end button for these routes must call them from the
# desktop shell's main process (no `Origin`), not from a `file://` page.
_LOCAL_ORIGIN = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", re.IGNORECASE)


def _this_machine_only(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin.strip() and not _LOCAL_ORIGIN.match(origin.strip()):
        raise HTTPException(
            status_code=403,
            detail=f"refused: requests from {origin} may not drive BuzzAnimate; only this machine may",
        )


router = APIRouter(dependencies=[Depends(_this_machine_only)])
jobs: Dict[str, Dict[str, Any]] = {}


class _StartOption(BaseModel):
    start: bool = False
    start_timeout: float = Field(animate.START_TIMEOUT_SECONDS, gt=0)


class CallRequest(_StartOption):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class OpenRequest(_StartOption):
    buzz_path: str


class RenderRequest(_StartOption):
    buzz_path: str
    project_id: Optional[str] = None
    format: str = "mp4"


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, animate.BuzzAnimateUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, animate.BuzzAnimateError):
        return HTTPException(status_code=exc.status, detail=f"BuzzAnimate: {exc.message}")
    return HTTPException(status_code=502, detail=f"BuzzAnimate: {exc}")


def _ensure(start: bool, timeout: float) -> None:
    try:
        animate.ensure(start, timeout)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/status")
def buzzanimate_status(start: bool = False, start_timeout: float = animate.START_TIMEOUT_SECONDS):
    if start:
        _ensure(True, start_timeout)
    return animate.status()


@router.post("/call")
def buzzanimate_call(body: CallRequest):
    _ensure(body.start, body.start_timeout)
    try:
        return animate.call(body.tool, body.args)
    except Exception as exc:
        raise _http_error(exc)


@router.post("/open")
def buzzanimate_open(body: OpenRequest):
    _ensure(body.start, body.start_timeout)
    try:
        return animate.open_document(body.buzz_path)
    except Exception as exc:
        raise _http_error(exc)


@router.post("/render")
async def buzzanimate_render(body: RenderRequest):
    from pathlib import Path

    from routes import projects as projects_route

    fmt = body.format.lower()
    if fmt not in animate.FORMATS:
        raise HTTPException(status_code=400, detail=f"format is one of {', '.join(animate.FORMATS)}")
    if not Path(body.buzz_path).is_file():
        raise HTTPException(status_code=404, detail=f"No file at {body.buzz_path}")
    if body.project_id and not projects_route.project_store.get_project(body.project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if not body.start:
        # Say so now rather than in a job that fails a moment later.
        await asyncio.to_thread(_ensure, False, body.start_timeout)

    job_id = f"buzzanimate_{uuid.uuid4().hex[:8]}"
    jobs[job_id] = {
        "id": job_id,
        "job_type": "buzzanimate_render",
        "project_id": body.project_id,
        "buzz_path": body.buzz_path,
        "format": fmt,
        "status": "running",
        "progress": 0.0,
        "message": "Starting BuzzAnimate..." if body.start else "Opening in BuzzAnimate...",
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_render(job_id, body, fmt))
    return {"status": "running", "job_id": job_id}


@router.get("/render/{job_id}")
async def buzzanimate_render_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


async def _run_render(job_id: str, body: RenderRequest, fmt: str) -> None:
    """Open -> export -> wait -> (import into the project), updating jobs[job_id]."""
    job = jobs[job_id]

    def on_progress(answer: Dict[str, Any]) -> None:
        fraction = ((answer.get("progress") or {}).get("fraction"))
        if isinstance(fraction, (int, float)):
            # 5% for opening, 85% for the export, the rest for the import.
            job["progress"] = round(0.05 + 0.85 * max(0.0, min(1.0, float(fraction))), 4)
        job["message"] = f"BuzzAnimate is exporting ({answer.get('state')})..."

    try:
        if body.start:
            await asyncio.to_thread(animate.ensure, True, body.start_timeout)
            job["message"] = "Opening in BuzzAnimate..."
        output = await asyncio.to_thread(animate.render_file, body.buzz_path, None, fmt, on_progress)
        result: Dict[str, Any] = {"output_path": output}
        if body.project_id:
            from routes import projects as projects_route

            job.update(progress=0.9, message="Adding the render to the project's media...")
            try:
                answer = await projects_route.import_media(
                    body.project_id, projects_route.MediaImportRequest(paths=[output])
                )
                result["media"] = answer.get("added")
            except HTTPException as exc:
                # A re-render overwrites the same file, which the library
                # already links: that is the render being updated, not a failure.
                if "already in library" not in str(exc.detail):
                    raise
                result["media"] = []
                result["note"] = "Already in the project's media; the file there was updated."
        job.update(status="completed", progress=1.0, message="Rendered.", result=result)
    except HTTPException as exc:
        job.update(status="failed", message=f"Render failed: {exc.detail}", error=str(exc.detail))
    except Exception as exc:
        logger.warning("BuzzAnimate render %s failed: %s", job_id, exc)
        job.update(status="failed", message=f"Render failed: {exc}", error=str(exc))
