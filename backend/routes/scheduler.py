from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from agents.scheduler import JobScheduler

router = APIRouter()
scheduler_instance = JobScheduler()


class EnqueueJobRequest(BaseModel):
    project_id: str
    job_type: str = "full_edit"        # "full_edit" | "presentation" | "presentation_render"
    priority: str = "normal"           # urgent, normal, low
    # "23:00" for the next occurrence of that time, or an ISO timestamp. Empty
    # means start as soon as the queue reaches it.
    start_at: Optional[str] = None
    # Free-form settings for the job type; PresentationSettings for a presentation.
    settings: Dict[str, Any] = {}
    generate_thumbnail: bool = True
    burn_captions: bool = True


# The scheduler is started/stopped by the application lifespan in main.py.
# A router-level @on_event("startup") is deprecated in FastAPI and only fired
# under the merged-lifespan shim.


@router.get("/jobs")
async def get_scheduler_status():
    return {
        "is_running": scheduler_instance.is_running,
        "is_paused": scheduler_instance.is_paused,
        "current_job_id": scheduler_instance.current_job_id,
        "jobs": scheduler_instance.jobs,
    }


@router.post("/enqueue")
async def enqueue_job(req: EnqueueJobRequest):
    # A presentation job is configured entirely by `settings`; the legacy
    # booleans belong to the older full_edit job and would be meaningless there.
    if req.job_type == "presentation":
        # BuzzEdit-side overrides beat whatever BuzzcafStudio sent — applied
        # again at execution time in agents/scheduler.py, so an override the
        # user changes after enqueuing but before a delayed job runs still
        # takes effect; this early merge just makes the queued job reflect it.
        from store.app_settings import apply_presentation_overrides
        settings, _sources = apply_presentation_overrides(dict(req.settings))
    else:
        settings = {
            "generate_thumbnail": req.generate_thumbnail,
            "burn_captions": req.burn_captions,
            **req.settings,
        }
    job = scheduler_instance.enqueue_job(
        project_id=req.project_id,
        job_type=req.job_type,
        priority=req.priority,
        settings=settings,
        start_at=req.start_at,
    )
    return {"status": "enqueued", "job": job}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    success = scheduler_instance.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}


@router.post("/clear_completed")
async def clear_completed():
    scheduler_instance.clear_completed()
    return {"status": "cleared"}


@router.post("/pause")
async def pause_scheduler():
    scheduler_instance.is_paused = True
    return {"status": "paused"}


@router.post("/resume")
async def resume_scheduler():
    scheduler_instance.is_paused = False
    return {"status": "resumed"}
