"""Video stabilization for a handheld library clip.

Starting a job renders `<name>_stable.mp4` beside the source (vidstab, NVENC), then swaps it
in everywhere the original was used -- the library entry, the timeline's sources and the
project's own `source_video` -- like eye contact does. The swap is safe because the
stabilized file is the original frame for frame (timestamps passed through, audio copied).
The original path is kept on the entry under `stabilize`, so the UI can badge the clip and
the change can be reverted. The original file is never overwritten.

    GET  /available                                   {available, reason}
    GET  /settings, PUT /settings                     the saved defaults (DEFAULTS keys)
    POST /projects/{pid}/media/{mid}                  full run -> {job_id}
    POST /projects/{pid}/media/{mid}/analyse          detect pass only -> {job_id}; result
                                                      {mean_shift_px, max_shift_px, shaky, frames, seconds}
    POST /projects/{pid}/media/{mid}/preview          10 s before/after -> {job_id}; result adds
                                                      {before_path, after_path, start_s, duration_s}
    GET  /status/{job_id}, POST /cancel/{job_id}
    POST /projects/{pid}/media/{mid}/revert           original back everywhere -> {status, media}
"""
import asyncio
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import PROJECTS_DIR
from routes.eyecontact import _relink_keeping_name, swap_source
from runtime.gpu_broker import gpu_broker
from store import media_pool
from store.app_settings import app_settings
from store.project_store import ProjectStore

logger = logging.getLogger(__name__)
router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

jobs: Dict[str, Dict[str, Any]] = {}
VRAM_MB = 1000.0        # NVDEC + NVENC only; vidstab itself runs on the CPU
ENTRY_KEY = "stabilize"
SETTINGS_KEY = "stabilize"


def _defaults() -> Dict[str, Any]:
    from stabilize import Settings
    return asdict(Settings())


DEFAULTS = _defaults()


class StabilizeSetup(BaseModel):
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    crop: Optional[str] = Field(None, pattern="^(auto|keep|black)$")
    smoothing_s: Optional[float] = Field(None, ge=0.0, le=10.0)
    # an explicit null clears a saved fixed zoom (back to the crop mode's own zoom)
    zoom_pct: Optional[float] = Field(None, ge=-50.0, le=50.0)
    tripod: Optional[bool] = None
    quality: Optional[str] = Field(None, pattern="^(standard|high|max)$")


def saved_setup() -> Dict[str, Any]:
    stored = app_settings.get(SETTINGS_KEY) or {}
    return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}


def _merged(body: StabilizeSetup) -> Dict[str, Any]:
    given = body.model_dump(exclude_unset=True)
    return {**saved_setup(), **{k: v for k, v in given.items() if v is not None or k == "zoom_pct"}}


def _settings(setup: Dict[str, Any]):
    from stabilize import Settings
    try:
        return Settings(strength=float(setup["strength"]), crop=str(setup["crop"]),
                        smoothing_s=float(setup["smoothing_s"]),
                        zoom_pct=None if setup.get("zoom_pct") is None else float(setup["zoom_pct"]),
                        tripod=bool(setup.get("tripod")), quality=str(setup["quality"]))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Bad stabilization settings: {e}")


def _entry(p_data, media_id):
    entry = next((m for m in p_data.get("media_pool", []) if m.get("id") == media_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Media not found in library")
    return entry


def _source(project_id: str, media_id: str):
    """(project, entry, the file to stabilize) -- always the pre-stabilization file, even
    when re-running with new settings."""
    from stabilize import available
    ok, reason = available()
    if not ok:
        raise HTTPException(status_code=400, detail=f"Stabilization unavailable: {reason}")
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    entry = _entry(p_data, media_id)
    if entry.get("kind") != "video":
        raise HTTPException(status_code=400, detail="Stabilization needs a video clip")
    src = (entry.get(ENTRY_KEY) or {}).get("original_path") or entry.get("path")
    if not src or not os.path.isfile(src):
        raise HTTPException(status_code=400, detail=f"Source file not found: {src}")
    return p_data, entry, src


def _new_job(prefix: str, project_id: str, media_id: str, job_type: str, stage: str) -> str:
    job_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    jobs[job_id] = {"id": job_id, "project_id": project_id, "media_id": media_id, "job_type": job_type,
                    "status": "queued", "progress": 0.0, "stage": stage, "result": None,
                    "error": None, "cancel": False}
    return job_id


def _running(media_id: str, job_type: str) -> Optional[Dict[str, Any]]:
    return next((j for j in jobs.values() if j["media_id"] == media_id and j["job_type"] == job_type
                 and j["status"] in ("queued", "running")), None)


def _stage(f: float) -> str:
    from stabilize import DETECT_SHARE
    return "Measuring the shake" if f < DETECT_SHARE else "Steadying and encoding"


@router.get("/available")
async def stabilize_available():
    from stabilize import available
    ok, reason = available()
    return {"available": ok, "reason": reason}


@router.get("/settings")
async def get_stabilize_settings():
    return saved_setup()


@router.put("/settings")
async def put_stabilize_settings(body: StabilizeSetup):
    setup = _merged(body)
    _settings(setup)                 # 422 before saving anything unusable
    app_settings.set(SETTINGS_KEY, setup)
    return setup


@router.post("/projects/{project_id}/media/{media_id}")
async def start_stabilize(project_id: str, media_id: str, body: StabilizeSetup):
    from stabilize import default_output
    _p, _e, src = _source(project_id, media_id)
    setup = _merged(body)
    settings = _settings(setup)
    running = _running(media_id, "stabilize")
    if running:
        return {"job_id": running["id"]}
    job_id = _new_job("stabilize", project_id, media_id, "stabilize", "Waiting for the GPU")
    asyncio.create_task(_run(job_id, project_id, media_id, src, default_output(src), settings))
    return {"job_id": job_id}


async def _run(job_id: str, project_id: str, media_id: str, src: str, dst: str, settings):
    from stabilize import stabilize_video
    job = jobs[job_id]

    def progress(f: float):
        job["progress"] = round(f, 4)
        job["stage"] = _stage(f)

    try:
        await gpu_broker.acquire_lease("stabilize", required_vram_mb=VRAM_MB)
        try:
            job["status"] = "running"
            report = await asyncio.to_thread(stabilize_video, src, dst, settings, progress, lambda: job["cancel"])
        finally:
            await gpu_broker.release_lease("stabilize")

        # the user may have edited the project meanwhile: apply the swap to a fresh copy
        p_data = project_store.get_project(project_id)
        if not p_data:
            raise RuntimeError("Project disappeared while stabilizing")
        entry = _entry(p_data, media_id)
        swap_source(p_data, entry.get("path", ""), dst)
        _relink_keeping_name(p_data, entry, dst)
        entry[ENTRY_KEY] = {"original_path": src, "output_path": dst, "report": report}
        project_store.save_project(project_id, p_data)
        job.update(status="completed", progress=1.0, stage="Done", result=report)
    except InterruptedError:
        job.update(status="cancelled", stage="Cancelled")
    except Exception as e:
        logger.exception("Stabilization failed")
        job.update(status="failed", error=str(e), stage="Failed")


@router.post("/projects/{project_id}/media/{media_id}/analyse")
async def start_stabilize_analyse(project_id: str, media_id: str, body: StabilizeSetup):
    """The detect pass only: how shaky the clip is ({mean_shift_px, max_shift_px, shaky}),
    so the UI can suggest stabilizing. Cached, so a full run afterwards skips it."""
    _p, _e, src = _source(project_id, media_id)
    settings = _settings(_merged(body))
    running = _running(media_id, "stabilize_analyse")
    if running:
        return {"job_id": running["id"]}
    job_id = _new_job("stabanalyse", project_id, media_id, "stabilize_analyse", "Measuring the shake")
    asyncio.create_task(_run_analyse(job_id, src, settings))
    return {"job_id": job_id}


async def _run_analyse(job_id: str, src: str, settings):
    from stabilize import analyse
    job = jobs[job_id]

    def progress(f: float):
        job["progress"] = round(f, 4)

    try:
        job["status"] = "running"
        result = await asyncio.to_thread(analyse, src, settings, progress, lambda: job["cancel"])
        job.update(status="completed", progress=1.0, stage="Done", result=result)
    except InterruptedError:
        job.update(status="cancelled", stage="Cancelled")
    except Exception as e:
        logger.exception("Stabilization analysis failed")
        job.update(status="failed", error=str(e), stage="Failed")


class PreviewRequest(StabilizeSetup):
    start_s: float = Field(0.0, ge=0.0)
    duration_s: float = Field(10.0, gt=0.0, le=30.0)


@router.post("/projects/{project_id}/media/{media_id}/preview")
async def start_stabilize_preview(project_id: str, media_id: str, body: PreviewRequest):
    """Stabilize a short stretch so the settings can be judged before the full run.
    Nothing in the project changes; the before/after files live in the preview cache."""
    _p, _e, src = _source(project_id, media_id)
    setup = _merged(StabilizeSetup(**body.model_dump(exclude_unset=True, exclude={"start_s", "duration_s"})))
    settings = _settings(setup)
    # one preview per clip at a time: a newer request replaces an older one
    for j in jobs.values():
        if j["media_id"] == media_id and j["job_type"] == "stabilize_preview" and j["status"] in ("queued", "running"):
            j["cancel"] = True
    job_id = _new_job("stabpreview", project_id, media_id, "stabilize_preview", "Waiting for the GPU")
    asyncio.create_task(_run_preview(job_id, src, float(body.start_s), float(body.duration_s), settings))
    return {"job_id": job_id}


async def _run_preview(job_id: str, src: str, start_s: float, duration_s: float, settings):
    from stabilize import preview_clip, prune_previews, stabilize_video
    job = jobs[job_id]

    def progress(f: float):
        job["progress"] = round(0.1 + 0.9 * f, 4)
        job["stage"] = _stage(f)

    try:
        await gpu_broker.acquire_lease("stabilize", required_vram_mb=VRAM_MB)
        try:
            if job["cancel"]:
                raise InterruptedError("cancelled")
            job.update(status="running", stage=f"Cutting {duration_s:g} s from {start_s:.0f} s")
            before = await asyncio.to_thread(preview_clip, src, start_s, duration_s)
            # a distinct file per setup: the dialog may still be streaming the previous one
            dst = before.replace("_before.mp4", f"_after_{uuid.uuid4().hex[:6]}.mp4")
            report = await asyncio.to_thread(stabilize_video, before, dst, settings, progress,
                                             lambda: job["cancel"], "h264")
        finally:
            await gpu_broker.release_lease("stabilize")
        prune_previews()
        job.update(status="completed", progress=1.0, stage="Done",
                   result={**report, "before_path": before, "after_path": dst,
                           "start_s": start_s, "duration_s": duration_s})
    except InterruptedError:
        job.update(status="cancelled", stage="Cancelled")
    except Exception as e:
        logger.exception("Stabilization preview failed")
        err = getattr(e, "stderr", None)
        detail = err.decode(errors="replace").strip()[-400:] if isinstance(err, bytes) else ""
        job.update(status="failed", error=f"{e} {detail}".strip(), stage="Failed")


@router.get("/status/{job_id}")
async def stabilize_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k != "cancel"}


@router.post("/cancel/{job_id}")
async def cancel_stabilize(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel"] = True
    return {"status": "cancelling"}


@router.post("/projects/{project_id}/media/{media_id}/revert")
async def revert_stabilize(project_id: str, media_id: str):
    """Put the original file back everywhere. The stabilized file stays on disk."""
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    entry = _entry(p_data, media_id)
    original = (entry.get(ENTRY_KEY) or {}).get("original_path")
    if not original or not os.path.isfile(original):
        raise HTTPException(status_code=400, detail="No original file to revert to")
    swap_source(p_data, entry.get("path", ""), original)
    _relink_keeping_name(p_data, entry, original)
    entry.pop(ENTRY_KEY, None)
    project_store.save_project(project_id, p_data)
    return {"status": "success", "media": media_pool.decorate(project_id, p_data["media_pool"])}
