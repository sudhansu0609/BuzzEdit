"""Teleprompter eye-contact correction for a library clip.

Starting a job renders `<name>_eyecontact.mp4` beside the source on the GPU, then swaps
it in everywhere the original was used — the library entry, the timeline's sources and
the project's own `source_video`. The swap is safe because the corrected file is the
original frame for frame (same timestamps, same audio packets). The original path is
kept on the entry, so the change can be reverted.
"""
import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import PROJECTS_DIR
from runtime.gpu_broker import gpu_broker
from store import media_pool
from store.app_settings import app_settings
from store.project_store import ProjectStore

logger = logging.getLogger(__name__)
router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

jobs: Dict[str, Dict[str, Any]] = {}
VRAM_MB = 3000.0


# The speaker's prompter setup rarely changes between recordings: it is saved once and
# pre-filled for every run; a run may still override any of it.
SETTINGS_KEY = "eye_contact"
DEFAULTS = {"prompter_side": "left", "angle_deg": 0.0, "prompter_cm": 30.0, "camera_cm": 110.0,
            "steadiness": 0.7, "aim_deg": 0.0, "pitch_deg": 0.0, "quality": "high"}


class EyeContactSetup(BaseModel):
    prompter_side: Optional[str] = Field(None, pattern="^(left|right|auto)$")
    angle_deg: Optional[float] = Field(None, ge=0.0, le=45.0)
    prompter_cm: Optional[float] = Field(None, ge=0.0, le=200.0)
    camera_cm: Optional[float] = Field(None, ge=20.0, le=1000.0)
    steadiness: Optional[float] = Field(None, ge=0.0, le=1.0)
    aim_deg: Optional[float] = Field(None, ge=-15.0, le=15.0)
    # vertical re-aim, + = lower the gaze (down), - = raise it (eyecontact.plan.vertical_shift)
    pitch_deg: Optional[float] = Field(None, ge=-10.0, le=10.0)
    quality: Optional[str] = Field(None, pattern="^(standard|high|max)$")


def saved_setup() -> Dict[str, Any]:
    stored = app_settings.get(SETTINGS_KEY) or {}
    return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}


def _merged(body: EyeContactSetup) -> Dict[str, Any]:
    return {**saved_setup(), **{k: v for k, v in body.model_dump().items() if v is not None}}


def _same(a: str, b: str) -> bool:
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def swap_source(p_data: Dict[str, Any], old: str, new: str) -> int:
    """Point every reference to `old` at `new`. Returns how many references moved."""
    moved = 0
    for src in ((p_data.get("timeline") or {}).get("sources") or {}).values():
        if isinstance(src, dict) and _same(src.get("path", ""), old):
            src["path"] = new
            moved += 1
    if _same(p_data.get("source_video", ""), old):
        p_data["source_video"] = new
        moved += 1
    return moved


def _relink_keeping_name(p_data, entry, path):
    """relink() renames the entry after the file; keep whatever name the user sees."""
    name = entry.get("name")
    media_pool.relink(p_data["media_pool"], entry["id"], path)
    if name:
        entry["name"] = name


def _entry(p_data, media_id):
    entry = next((m for m in p_data.get("media_pool", []) if m.get("id") == media_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Media not found in library")
    return entry


@router.get("/available")
async def eye_contact_available():
    from eyecontact import available
    ok, reason = available()
    return {"available": ok, "reason": reason}


@router.get("/settings")
async def get_eye_contact_settings():
    return saved_setup()


@router.put("/settings")
async def put_eye_contact_settings(body: EyeContactSetup):
    setup = _merged(body)
    app_settings.set(SETTINGS_KEY, setup)
    return setup


@router.post("/projects/{project_id}/media/{media_id}")
async def start_eye_contact(project_id: str, media_id: str, body: EyeContactSetup):
    from eyecontact import available, default_output
    ok, reason = available()
    if not ok:
        raise HTTPException(status_code=400, detail=f"Eye contact correction unavailable: {reason}")
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    entry = _entry(p_data, media_id)
    if entry.get("kind") != "video":
        raise HTTPException(status_code=400, detail="Eye contact correction needs a video clip")
    # always work from the untouched recording, even when re-running with new settings
    src = entry.get("eye_contact", {}).get("original_path") or entry.get("path")
    if not src or not os.path.isfile(src):
        raise HTTPException(status_code=400, detail=f"Source file not found: {src}")
    running = next((j for j in jobs.values() if j["media_id"] == media_id and j["job_type"] == "eye_contact"
                    and j["status"] in ("queued", "running")), None)
    if running:
        return {"job_id": running["id"]}

    job_id = f"eyecontact_{uuid.uuid4().hex[:8]}"
    jobs[job_id] = {"id": job_id, "project_id": project_id, "media_id": media_id, "job_type": "eye_contact",
                    "status": "queued", "progress": 0.0, "stage": "Waiting for the GPU", "result": None,
                    "error": None, "cancel": False}
    asyncio.create_task(_run(job_id, project_id, media_id, src, default_output(src), _merged(body)))
    return {"job_id": job_id}


def _settings(setup: Dict[str, Any]):
    from eyecontact import Settings
    return Settings(prompter_side=setup["prompter_side"], angle_deg=float(setup["angle_deg"]),
                    prompter_cm=float(setup["prompter_cm"]), camera_cm=float(setup["camera_cm"]),
                    steadiness=float(setup["steadiness"]), aim_deg=float(setup["aim_deg"]),
                    pitch_deg=float(setup.get("pitch_deg") or 0.0))


def _cleanup(dst: str):
    for leftover in (dst + ".video.tmp", dst + ".mux.tmp.mp4"):
        try:
            os.remove(leftover)
        except OSError:
            pass


class PreviewRequest(EyeContactSetup):
    start_s: float = Field(0.0, ge=0.0)
    duration_s: float = Field(10.0, gt=0.0, le=30.0)


@router.post("/projects/{project_id}/media/{media_id}/preview")
async def start_eye_contact_preview(project_id: str, media_id: str, body: PreviewRequest):
    """Correct a short stretch of the clip so the setup can be judged before the full run.
    Nothing in the project changes; the before/after files live in the preview cache."""
    from eyecontact import available
    ok, reason = available()
    if not ok:
        raise HTTPException(status_code=400, detail=f"Eye contact correction unavailable: {reason}")
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    entry = _entry(p_data, media_id)
    if entry.get("kind") != "video":
        raise HTTPException(status_code=400, detail="Eye contact correction needs a video clip")
    src = entry.get("eye_contact", {}).get("original_path") or entry.get("path")
    if not src or not os.path.isfile(src):
        raise HTTPException(status_code=400, detail=f"Source file not found: {src}")
    # one preview per clip at a time: a newer request replaces an older one
    for j in jobs.values():
        if j["media_id"] == media_id and j["job_type"] == "eye_contact_preview" and j["status"] in ("queued", "running"):
            j["cancel"] = True

    setup = _merged(body)
    job_id = f"eyepreview_{uuid.uuid4().hex[:8]}"
    jobs[job_id] = {"id": job_id, "project_id": project_id, "media_id": media_id,
                    "job_type": "eye_contact_preview", "status": "queued", "progress": 0.0,
                    "stage": "Waiting for the GPU", "result": None, "error": None, "cancel": False}
    asyncio.create_task(_run_preview(job_id, src, float(body.start_s), float(body.duration_s), setup))
    return {"job_id": job_id}


async def _run_preview(job_id: str, src: str, start_s: float, duration_s: float, setup: Dict[str, Any]):
    from eyecontact import correct_eye_contact, lens_positions, preview_clip, prune_previews
    job = jobs[job_id]
    dst = ""

    def progress(f: float):
        job["progress"] = round(0.1 + 0.9 * f, 4)
        job["stage"] = "Finding your eyes" if f < 0.5 else "Re-aiming"

    try:
        await gpu_broker.acquire_lease("eye_contact", required_vram_mb=VRAM_MB)
        try:
            if job["cancel"]:
                raise InterruptedError("cancelled")
            job.update(status="running", stage=f"Cutting {duration_s:g} s from {start_s:.0f} s")
            before = await asyncio.to_thread(preview_clip, src, start_s, duration_s)
            # a distinct file per setup: the dialog may still be streaming the previous one
            dst = before.replace("_before.mp4", f"_after_{uuid.uuid4().hex[:6]}.mp4")
            # Each eye's own lens position from the whole recording: the preview clip alone is
            # too short to measure it, and would show the old over-pushed eyes.
            lens = await asyncio.to_thread(lens_positions, src, _settings(setup))
            report = await asyncio.to_thread(
                correct_eye_contact, before, dst, _settings(setup), "high", "h264", progress,
                lambda: job["cancel"], lens)
        finally:
            await gpu_broker.release_lease("eye_contact")
        prune_previews()
        job.update(status="completed", progress=1.0, stage="Done",
                   result={**report, "before_path": before, "after_path": dst,
                           "start_s": start_s, "duration_s": duration_s})
    except InterruptedError:
        job.update(status="cancelled", stage="Cancelled")
        if dst:
            _cleanup(dst)
    except Exception as e:
        logger.exception("Eye contact preview failed")
        err = getattr(e, "stderr", None)
        detail = err.decode(errors="replace").strip()[-400:] if isinstance(err, bytes) else ""
        job.update(status="failed", error=f"{e} {detail}".strip(), stage="Failed")


async def _run(job_id: str, project_id: str, media_id: str, src: str, dst: str, setup: Dict[str, Any]):
    from eyecontact import correct_eye_contact
    job = jobs[job_id]

    def progress(f: float):
        job["progress"] = round(f, 4)
        job["stage"] = "Finding your eyes" if f < 0.5 else "Re-aiming and encoding"

    try:
        await gpu_broker.acquire_lease("eye_contact", required_vram_mb=VRAM_MB)
        try:
            job["status"] = "running"
            report = await asyncio.to_thread(
                correct_eye_contact, src, dst, _settings(setup), setup["quality"], "hevc", progress,
                lambda: job["cancel"])
        finally:
            await gpu_broker.release_lease("eye_contact")

        # the user may have edited the project meanwhile: apply the swap to a fresh copy
        p_data = project_store.get_project(project_id)
        if not p_data:
            raise RuntimeError("Project disappeared while correcting")
        entry = _entry(p_data, media_id)
        swap_source(p_data, entry.get("path", ""), dst)
        _relink_keeping_name(p_data, entry, dst)
        entry["eye_contact"] = {"original_path": src, "output_path": dst, "report": report}
        project_store.save_project(project_id, p_data)
        job.update(status="completed", progress=1.0, stage="Done", result=report)
    except InterruptedError:
        job.update(status="cancelled", stage="Cancelled")
        _cleanup(dst)
    except Exception as e:
        logger.exception("Eye contact correction failed")
        job.update(status="failed", error=str(e), stage="Failed")


@router.get("/status/{job_id}")
async def eye_contact_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k != "cancel"}


@router.post("/cancel/{job_id}")
async def cancel_eye_contact(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel"] = True
    return {"status": "cancelling"}


@router.post("/projects/{project_id}/media/{media_id}/revert")
async def revert_eye_contact(project_id: str, media_id: str):
    """Put the original recording back everywhere. The corrected file stays on disk."""
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    entry = _entry(p_data, media_id)
    info = entry.get("eye_contact") or {}
    original = info.get("original_path")
    if not original or not os.path.isfile(original):
        raise HTTPException(status_code=400, detail="No original recording to revert to")
    swap_source(p_data, entry.get("path", ""), original)
    _relink_keeping_name(p_data, entry, original)
    entry.pop("eye_contact", None)
    project_store.save_project(project_id, p_data)
    return {"status": "success", "media": media_pool.decorate(project_id, p_data["media_pool"])}
