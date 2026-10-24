"""The vibe reel (REELS_DESK_PLAN.md §6, VR1).

POST /api/reels/vibe        {out_dir, name, art_style, world, shots, lines,
                             credit?, music, ambience?, font?} -> {job_id}
GET  /api/reels/jobs/{id}   {state, progress, message, result, error}

A job runs as a background task: generation hands the GPU to ComfyUI
(`presentation.assets.generate_assets`), then to the music/SFX engines
(`presentation.music_gen` / `presentation.sfx_library`), and releases
everything when the job ends, win or lose (`presentation.vibe_reel`).

Only one vibe reel renders at a time -- the same rule every other generation
job on this card follows.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from presentation import vibe_reel
from runtime import job_store

logger = logging.getLogger("routes.reels")
router = APIRouter()

JOB_KIND = "reels"

_jobs: Dict[str, Dict[str, Any]] = {}

# A restart's own startup sweep (STUDIO_DIRECTOR_PLAN.md enhancement 2, Fable
# review row 7): a reel job still "queued"/"running" on disk belonged to the
# process before this one -- nothing here is rendering it, so it is marked
# failed rather than left for the Studio's follower to poll forever.
job_store.sweep_stale(JOB_KIND)


def _persist(job: Dict[str, Any]) -> None:
    job["updated_at"] = time.time()
    job_store.save(JOB_KIND, job["job_id"], job)


def _busy() -> bool:
    return any(j["state"] in ("queued", "running") for j in _jobs.values())


async def _run(job_id: str, req: vibe_reel.VibeReelRequest) -> None:
    job = _jobs[job_id]
    job["state"] = "running"
    _persist(job)

    def on_progress(fraction: float, message: str = "") -> None:
        job["progress"] = max(0.0, min(1.0, fraction))
        if message:
            job["message"] = message
        _persist(job)

    try:
        result = await vibe_reel.render_vibe_reel(req, progress_cb=on_progress)
        job.update(state="done", progress=1.0, result=result, error=None)
    except vibe_reel.VibeReelError as e:
        logger.warning("Vibe reel job %s failed: %s", job_id, e)
        job.update(state="failed", error=str(e))
    except Exception as e:
        logger.exception("Vibe reel job %s failed", job_id)
        job.update(state="failed", error=str(e)[:500])
    finally:
        _persist(job)


@router.post("/vibe")
async def vibe(req: vibe_reel.VibeReelRequest):
    if _busy():
        raise HTTPException(status_code=409, detail="A reel is already rendering.")
    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {"job_id": job_id, "state": "queued", "progress": 0.0, "message": "",
                           "result": None, "error": None, "started_at": time.time(),
                           "out_dir": req.out_dir, "request": req.model_dump()}
    _jobs[job_id] = job
    _persist(job)
    asyncio.create_task(_run(job_id, req))
    return {"job_id": job_id, "state": "queued"}


@router.get("/busy")
async def busy():
    """Whether a reel is rendering: the Studio waits for this before a reel's own model
    steps, so its LLM never loads while ComfyUI is generating (2026-10-23)."""
    running = [j["job_id"] for j in _jobs.values() if j["state"] in ("queued", "running")]
    return {"busy": bool(running), "job_id": running[0] if running else None}


@router.get("/jobs/{job_id}")
async def job(job_id: str):
    found = _jobs.get(job_id)
    if found is None:
        # The in-memory registry is empty after a restart; the job's own
        # file (STUDIO_DIRECTOR_PLAN.md enhancement 2) still answers for it,
        # already marked failed ("BuzzEdit restarted") by the startup sweep
        # if it was still "queued"/"running" when this process started.
        found = job_store.load(JOB_KIND, job_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _json_safe(dict(found))


def _json_safe(value):
    """A finished job's result can hold NaN/inf floats (a loudness reading) or Paths,
    which the JSON response refuses -- the job showed 500 while its video was done
    (2026-10-24, the illustrated test). NaN/inf become None, Paths strings."""
    import math
    from pathlib import Path
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
