from fastapi import APIRouter, HTTPException
from models import TranscribeJob, TranscriptionResult, TranscriptSegment
from config import PROJECTS_DIR
import asyncio
import json

router = APIRouter()
jobs: dict = {}


@router.post("/transcribe")
async def transcribe_video(job: TranscribeJob):
    from routes.projects import projects

    if job.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[job.project_id]

    if not project.source_video:
        raise HTTPException(status_code=400, detail="No source video")

    job_id = f"trans_{job.project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "transcribe",
        "status": "running",
        "progress": 0.0,
    }

    try:
        result = await asyncio.to_thread(run_transcription, project.source_video, job.model, job.language)

        project.transcript = result
        project.status = "transcribed"

        project_dir = PROJECTS_DIR / job.project_id
        (project_dir / "project.json").write_text(project.model_dump_json(indent=2))
        (project_dir / "transcript.json").write_text(result.model_dump_json(indent=2))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"segments_count": len(result.segments)}

        return {
            "status": "completed",
            "job_id": job_id,
            "segments": len(result.segments),
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


def run_transcription(video_path: str, model_name: str = "large-v3", language: str = "en") -> TranscriptionResult:
    import whisper
    from utils.ffmpeg_utils import extract_audio

    audio_path = extract_audio(video_path)

    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False,
        word_timestamps=True,
    )

    segments = []
    for i, seg in enumerate(result["segments"]):
        words = []
        if "words" in seg:
            for w in seg["words"]:
                word_info = {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
                if "probability" in w:
                    word_info["confidence"] = w["probability"]
                words.append(word_info)

        segment = TranscriptSegment(
            id=i,
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
            words=words if words else None,
            confidence=float(seg.get("avg_logprob", 0) * -1) if "avg_logprob" in seg else 0.0,
        )
        segments.append(segment)

    duration = float(result["segments"][-1]["end"]) if result["segments"] else 0.0

    return TranscriptionResult(
        segments=segments,
        language=language,
        duration=duration,
    )


@router.get("/status/{job_id}")
async def transcription_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/{project_id}/transcript")
async def get_transcript(project_id: str):
    from routes.projects import projects

    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    if not project.transcript:
        project_dir = PROJECTS_DIR / project_id
        transcript_file = project_dir / "transcript.json"
        if transcript_file.exists():
            project.transcript = TranscriptionResult.model_validate_json(
                transcript_file.read_text()
            )

    if not project.transcript:
        raise HTTPException(status_code=404, detail="No transcript available")

    return project.transcript
