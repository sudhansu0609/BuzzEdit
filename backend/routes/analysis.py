from fastapi import APIRouter, HTTPException
from models import AnalyzeJob, DetectedSegment, ClipType, Project
from config import PROJECTS_DIR
import asyncio

router = APIRouter()
jobs: dict = {}


@router.post("/analyze")
async def analyze_video(job: AnalyzeJob):
    from routes.projects import projects

    if job.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[job.project_id]

    if not project.transcript:
        raise HTTPException(status_code=400, detail="Video must be transcribed first")

    job_id = f"analyze_{job.project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "analyze",
        "status": "running",
        "progress": 0.0,
    }

    try:
        segments = await asyncio.to_thread(
            run_analysis,
            project,
            job.remove_silence,
            job.remove_fumbles,
        )

        project.detected_segments = segments
        project.status = "analyzed"

        project_dir = PROJECTS_DIR / job.project_id
        (project_dir / "project.json").write_text(project.model_dump_json(indent=2))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "segments_count": len(segments),
            "silence_count": sum(1 for s in segments if s.segment_type == ClipType.SILENCE),
            "fumble_count": sum(1 for s in segments if s.segment_type == ClipType.FUMBLE),
            "speech_count": sum(1 for s in segments if s.segment_type == ClipType.SPEECH),
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


def run_analysis(project: Project, remove_silence: bool = True, remove_fumbles: bool = True) -> list[DetectedSegment]:
    from audio_pipeline.silence_detector import detect_silence_segments
    from audio_pipeline.fumble_detector import detect_fumbles

    segments = []

    silence_segments = detect_silence_segments(
        project.source_video,
        project.transcript.segments,
        min_duration=1.5,
    )

    for seg in silence_segments:
        if remove_silence:
            segments.append(DetectedSegment(
                start=seg["start"],
                end=seg["end"],
                segment_type=ClipType.SILENCE,
                confidence=seg.get("confidence", 0.9),
                label="silence",
            ))

    if remove_fumbles and project.transcript:
        fumble_segments = detect_fumbles(project.transcript.segments)
        for seg in fumble_segments:
            segments.append(DetectedSegment(
                start=seg["start"],
                end=seg["end"],
                segment_type=ClipType.FUMBLE,
                confidence=seg.get("confidence", 0.7),
                label=seg.get("reason", "fumble"),
            ))

    segments.sort(key=lambda s: s.start)

    return segments


@router.get("/status/{job_id}")
async def analysis_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/{project_id}/segments")
async def get_segments(project_id: str):
    from routes.projects import projects

    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    return {"segments": project.detected_segments}
