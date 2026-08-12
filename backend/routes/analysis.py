from fastapi import APIRouter, HTTPException
from models import AnalyzeJob, DetectedSegment, ClipType
from config import PROJECTS_DIR
from store.project_store import ProjectStore
from asr import detect_speech_silence_intervals, analyze_disfluencies
from timeline import Timeline

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}

@router.post("/analyze")
async def analyze_video(job: AnalyzeJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    job_id = f"analyze_{job.project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "analyze",
        "status": "running",
        "progress": 0.0,
    }

    try:
        segments = []
        source_video = p_data.get("source_video")

        if source_video:
            silences = detect_speech_silence_intervals(source_video, min_silence_duration=1.2)
            for start, end in silences:
                segments.append(DetectedSegment(
                    start=start,
                    end=end,
                    segment_type=ClipType.SILENCE,
                    confidence=0.9,
                    label="silence",
                ))

        if "timeline" in p_data:
            tl = Timeline.model_validate(p_data["timeline"])
            fps = tl.fps_num / tl.fps_den
            for word in tl.words:
                if word.disfluency:
                    segments.append(DetectedSegment(
                        start=word.start_frame / fps,
                        end=word.end_frame / fps,
                        segment_type=ClipType.FUMBLE,
                        confidence=0.8,
                        label=f"disfluency: {word.text}",
                    ))

        segments.sort(key=lambda s: s.start)

        p_data["detected_segments"] = [s.model_dump() for s in segments]
        p_data["status"] = "analyzed"
        project_store.save_project(job.project_id, p_data)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "segments_count": len(segments),
            "silence_count": sum(1 for s in segments if s.segment_type == ClipType.SILENCE),
            "fumble_count": sum(1 for s in segments if s.segment_type == ClipType.FUMBLE),
        }

        return {
            "status": "completed",
            "job_id": job_id,
            "segments": len(segments),
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def analysis_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/{project_id}/segments")
async def get_segments(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"segments": p_data.get("detected_segments", [])}
