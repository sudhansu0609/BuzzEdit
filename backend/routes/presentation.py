"""Routes for the presentation pass and the workflow registry.

The run endpoint is here mainly for testing a pass in the foreground; the way
this is meant to be used is `POST /api/scheduler/enqueue` with
`job_type: "presentation"` and a `start_at`, so it happens overnight.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import FFMPEG_BIN, PROJECTS_DIR
from presentation import PresentationSettings, run_presentation_pass
from presentation import workflows as workflow_registry
from presentation import sound as sound_stage
from render.audio import build_voice_chain
from store.project_store import ProjectStore
from utils.proc import NO_WINDOW

logger = logging.getLogger("routes.presentation")

router = APIRouter()

# Background presentation jobs, polled by the UI (mirrors rendering.jobs).
_jobs: Dict[str, Dict[str, Any]] = {}


class RunRequest(BaseModel):
    settings: Dict[str, Any] = {}


def _needs_comfyui(settings: PresentationSettings) -> bool:
    return settings.broll or settings.graphics or settings.thumbnail


async def _require_comfyui(settings: PresentationSettings) -> None:
    """Refuse up front when the pass needs pictures and nothing can make them.

    Without this the job "succeeds" minutes later with zero images and only a
    log line to explain — the user finds out in the morning. A 503 with the fix
    in the message is worth far more than a degraded overnight run they did not
    ask for. The pass itself still degrades gracefully (the director records
    `comfyui_online`), because ComfyUI can also die mid-run.
    """
    if not _needs_comfyui(settings):
        return
    from comfyui_bridge import queue_manager
    online = await asyncio.to_thread(queue_manager.client.is_connected, 5.0)
    if not online:
        raise HTTPException(
            status_code=503,
            detail="ComfyUI is not reachable. Start ComfyUI, or turn off "
                   "B-roll, graphics and thumbnail to run without generated "
                   "pictures.")


@router.get("/workflows")
async def list_workflows():
    """Every ComfyUI workflow on disk, what it can do, and which role uses it.

    This is what the settings dropdowns read: a user drops a workflow file into
    `workflows/`, and it appears here with its detected bindings and the roles it
    can serve.
    """
    return {
        "workflows": workflow_registry.list_workflows(),
        "selected": workflow_registry.current_selection(),
        "roles": list(workflow_registry.ROLES),
        "settings_keys": workflow_registry.SETTINGS_KEY,
    }


@router.get("/workflows/{file_name}")
async def describe_workflow(file_name: str):
    detail = workflow_registry.describe(file_name)
    if not detail.get("valid"):
        raise HTTPException(status_code=404, detail=detail.get("error", "Not a workflow"))
    return detail


@router.post("/{project_id}/start")
async def start_pass(project_id: str, body: RunRequest):
    """Start the presentation pass as a background job and return its id.

    The pass generates B-roll through ComfyUI, cuts it in on an overlay track,
    and adds the auto zoom/punch-ins — minutes of work. Running it inline blocked
    the request with no feedback, so it now runs as a job the UI polls via GET
    /status/{job_id}, like the render and auto-edit.
    """
    try:
        settings = PresentationSettings(**(body.settings or {})).resolve_density()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bad settings: {e}")
    await _require_comfyui(settings)

    job_id = f"presentation_{project_id}"
    _jobs[job_id] = {
        "id": job_id,
        "project_id": project_id,
        "job_type": "presentation",
        "status": "running",
        "progress": 0.0,
        "message": "Starting presentation pass…",
    }
    asyncio.create_task(_run_presentation(project_id, job_id, settings))
    return {"status": "running", "job_id": job_id}


async def _run_presentation(project_id: str, job_id: str, settings: PresentationSettings):
    def cb(fraction: float, message: str = ""):
        _jobs[job_id]["progress"] = max(0.0, min(1.0, fraction))
        if message:
            _jobs[job_id]["message"] = message

    try:
        report = await run_presentation_pass(project_id, settings, progress_cb=cb)
        _jobs[job_id].update(status="completed", progress=1.0,
                             message="Presentation pass complete",
                             result=report.model_dump(exclude={"timings"}))
    except Exception as e:
        logger.exception("Presentation pass failed for %s", project_id)
        _jobs[job_id].update(status="failed", message=f"Presentation failed: {e}", error=str(e))


# --- the effects sample (presentation/effects_sample.py) -----------------------

_samples: Dict[str, Dict[str, Any]] = {}


class EffectsSampleRequest(BaseModel):
    settings: Dict[str, Any] = {}
    seconds: float = 20.0


@router.post("/{project_id}/effects_sample")
async def start_effects_sample(project_id: str, body: EffectsSampleRequest):
    """Render ~20 s of the cut with the effects placed on purpose, small and
    fast, so the look can be checked before the long pass. Poll GET on the
    same path; the clip is served at .../effects_sample/video."""
    current = _samples.get(project_id) or {}
    if current.get("status") == "running":
        return current
    try:
        from store.app_settings import apply_presentation_overrides
        merged, _sources = apply_presentation_overrides(dict(body.settings or {}))
        settings = PresentationSettings(**merged).resolve_density()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bad settings: {e}")
    _samples[project_id] = {"project_id": project_id, "status": "running", "progress": 0.0,
                            "message": "Starting the effects sample..."}
    asyncio.create_task(_run_effects_sample(project_id, settings, body.seconds))
    return _samples[project_id]


async def _run_effects_sample(project_id: str, settings: PresentationSettings, seconds: float):
    from presentation.effects_sample import render_effects_sample

    def cb(fraction: float, message: str = ""):
        _samples[project_id]["progress"] = max(0.0, min(1.0, fraction))
        if message:
            _samples[project_id]["message"] = message

    try:
        result = await render_effects_sample(project_id, settings, seconds, progress_cb=cb)
        _samples[project_id].update(status="completed", progress=1.0, message="Effects sample ready",
                                    result=result)
    except Exception as e:
        logger.exception("Effects sample failed for %s", project_id)
        _samples[project_id].update(status="failed", message=f"Effects sample failed: {e}", error=str(e))


@router.get("/{project_id}/effects_sample")
async def effects_sample_status(project_id: str):
    status = _samples.get(project_id)
    if status is None:
        from presentation.effects_sample import sample_path
        path = sample_path(project_id)
        return {"project_id": project_id, "status": "completed" if path.exists() else "none",
                "result": {"path": str(path)} if path.exists() else None}
    return status


@router.get("/{project_id}/effects_sample/video")
async def effects_sample_video(project_id: str):
    from presentation.effects_sample import sample_path
    path = sample_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No effects sample rendered yet")
    return FileResponse(str(path), media_type="video/mp4",
                        headers={"Cache-Control": "no-store"})


@router.get("/status/{job_id}")
async def pass_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.post("/{project_id}/run")
async def run_pass(project_id: str, body: RunRequest):
    """Run the whole pass now, in the foreground. Long — minutes to hours."""
    try:
        settings = PresentationSettings(**(body.settings or {})).resolve_density()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bad settings: {e}")
    await _require_comfyui(settings)

    try:
        report = await run_presentation_pass(project_id, settings)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Presentation pass failed for %s", project_id)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "completed", "report": report.model_dump()}


@router.get("/{project_id}/report")
async def get_report(project_id: str):
    import json
    from config import PROJECTS_DIR

    path = PROJECTS_DIR / project_id / "presentation_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No presentation report for this project")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{project_id}/shot_plan")
async def get_shot_plan(project_id: str):
    """The prompts the last B-roll run actually used, one per generated shot.

    Read from the persisted shot plan (already the multiplied, budgeted beats),
    so the UI can show exactly what text produced each image — the 'show the
    prompts used in a tab' view.
    """
    import json
    from config import PROJECTS_DIR

    path = PROJECTS_DIR / project_id / "shot_plan.json"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="No shot plan yet — run a presentation pass with B-roll on.")
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = [{
        "id": b.get("id"),
        "topic": b.get("topic"),
        "kind": b.get("kind"),
        "start_s": b.get("start_s"),
        "end_s": b.get("end_s"),
        "image_prompt": b.get("image_prompt"),
        "video_prompt": b.get("video_prompt"),
        "popup_text": b.get("popup_text"),
    } for b in (data.get("beats") or [])]
    return {
        "source": data.get("source"),
        "thumbnail_title": data.get("thumbnail_title"),
        "count": len(prompts),
        "prompts": prompts,
    }


class AudioPreviewRequest(BaseModel):
    preset: str = "studio_mic"
    start_s: float = 0.0
    seconds: float = 20.0


@router.post("/{project_id}/audio_preview")
async def audio_preview(project_id: str, body: AudioPreviewRequest):
    """A short WAV of the programme voice run through one named preset.

    Simple on purpose: it reuses the exact same `build_voice_chain` the real
    render uses, just applied straight to the project's own source audio for
    `seconds` starting at `start_s`, so a preset can be auditioned in a few
    seconds instead of waiting on a full render.
    """
    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    data = store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    source_video = data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=404, detail="No source video for this project")

    seconds = max(1.0, min(60.0, float(body.seconds or 20.0)))
    start_s = max(0.0, float(body.start_s or 0.0))

    master = sound_stage.master_for_preset(body.preset)
    if master is None:
        raise HTTPException(status_code=422, detail=f"Unknown or 'off' preset: {body.preset!r}")
    chain = build_voice_chain(master)

    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    safe_preset = "".join(c for c in body.preset if c.isalnum() or c in ("_", "-")) or "preset"
    output_path = project_dir / f"audio_preview_{safe_preset}.wav"

    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start_s:.3f}", "-t", f"{seconds:.3f}", "-i", str(source_video)]
    if chain:
        command.extend(["-af", ",".join(chain)])
    command.extend(["-vn", "-ar", "48000", "-ac", "2", str(output_path)])

    try:
        result = await asyncio.to_thread(
            subprocess.run, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Audio preview render timed out")
    if result.returncode != 0 or not output_path.exists():
        err = (result.stderr or b"").decode("utf-8", "replace")[-500:]
        raise HTTPException(status_code=500, detail=f"Preview render failed: {err}")

    return {"path": str(output_path), "preset": body.preset, "start_s": start_s,
            "seconds": seconds}
