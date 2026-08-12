from fastapi import APIRouter, HTTPException
from models import RenderJob, Project
from config import PROJECTS_DIR, OUTPUT_DIR
from store.project_store import ProjectStore
from timeline import Timeline, build_timeline_from_transcript
from render import render_timeline_async
from utils.ffmpeg_utils import get_video_info, extract_audio
from asr import whisper_engine, analyze_disfluencies

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}

@router.post("/render")
async def render_video(job: RenderJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video:
        raise HTTPException(status_code=400, detail="No source video uploaded for project")

    job_id = f"render_{job.project_id}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = job.output_path or str(OUTPUT_DIR / f"{job.project_id}_output.mp4")

    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "render",
        "status": "running",
        "progress": 0.0,
    }

    try:
        # Load or generate timeline
        if "timeline" in p_data:
            tl = Timeline.model_validate(p_data["timeline"])
        else:
            audio_path = extract_audio(source_video)
            v_info = get_video_info(source_video)
            words = await whisper_engine.transcribe_audio_async(audio_path)
            ann_words = analyze_disfluencies(words)
            tl = build_timeline_from_transcript(
                source_path=source_video,
                duration_seconds=v_info.get("duration", 0.0),
                transcript_words=ann_words,
                fps_num=int(round(v_info.get("fps", 30))),
                fps_den=1,
                width=v_info.get("width", 1920),
                height=v_info.get("height", 1080)
            )
            p_data["timeline"] = tl.model_dump()

        def update_progress(pct: float):
            jobs[job_id]["progress"] = pct / 100.0

        result_path = await render_timeline_async(
            tl,
            output_path,
            progress_callback=update_progress
        )

        p_data["output_path"] = result_path
        p_data["status"] = "rendered"
        project_store.save_project(job.project_id, p_data)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"output_path": result_path}

        return {
            "status": "completed",
            "job_id": job_id,
            "output_path": result_path,
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def render_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.post("/{project_id}/auto_edit")
async def auto_edit(project_id: str):
    render_job = RenderJob(project_id=project_id)
    return await render_video(render_job)
