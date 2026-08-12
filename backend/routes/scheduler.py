from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from agents.scheduler import JobScheduler

router = APIRouter()
scheduler_instance = JobScheduler()


class EnqueueJobRequest(BaseModel):
    project_id: str
    job_type: str = "full_edit"
    priority: str = "normal" # urgent, normal, low
    generate_broll: bool = True
    generate_thumbnail: bool = True
    burn_captions: bool = True


@router.on_event("startup")
async def startup_scheduler():
    scheduler_instance.start()


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
    settings = {
        "generate_broll": req.generate_broll,
        "generate_thumbnail": req.generate_thumbnail,
        "burn_captions": req.burn_captions,
    }
    job = scheduler_instance.enqueue_job(
        project_id=req.project_id,
        job_type=req.job_type,
        priority=req.priority,
        settings=settings,
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
