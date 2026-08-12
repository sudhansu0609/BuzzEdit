from pathlib import Path
from fastapi import APIRouter, HTTPException
from models import TranscribeJob, TranscriptionResult, TranscriptSegment
from config import PROJECTS_DIR
from store.project_store import ProjectStore
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine, analyze_disfluencies
from timeline import build_timeline_from_transcript

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}

@router.post("/transcribe")
async def transcribe_video(job: TranscribeJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")

    job_id = f"trans_{job.project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "transcribe",
        "status": "running",
        "progress": 0.0,
    }

    try:
        # 1. Extract WAV audio
        audio_path = extract_audio(source_video)

        # 2. Get Video info
        v_info = get_video_info(source_video)

        # 3. Transcribe via faster-whisper
        words = await whisper_engine.transcribe_audio_async(audio_path, language=job.language or "en")

        # 4. Disfluency analysis
        ann_words = analyze_disfluencies(words)

        # 5. Build EDL Timeline
        tl = build_timeline_from_transcript(
            source_path=source_video,
            duration_seconds=v_info.get("duration", 0.0),
            transcript_words=ann_words,
            fps_num=int(round(v_info.get("fps", 30))),
            fps_den=1,
            width=v_info.get("width", 1920),
            height=v_info.get("height", 1080)
        )

        # Convert to TranscriptSegment list for legacy UI components
        segments = []
        if ann_words:
            seg = TranscriptSegment(
                id=0,
                start=ann_words[0]["start"],
                end=ann_words[-1]["end"],
                text=" ".join([w["word"] for w in ann_words]),
                words=ann_words,
                confidence=1.0
            )
            segments.append(seg)

        res = TranscriptionResult(
            segments=segments,
            language=job.language or "en",
            duration=ann_words[-1]["end"] if ann_words else 0.0
        )

        p_data["transcript"] = res.model_dump()
        p_data["timeline"] = tl.model_dump()
        p_data["status"] = "transcribed"
        project_store.save_project(job.project_id, p_data)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"segments_count": len(segments)}

        return {
            "status": "completed",
            "job_id": job_id,
            "segments": len(segments),
            "timeline": tl
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def transcription_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/{project_id}/transcript")
async def get_transcript(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data or "transcript" not in p_data:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return p_data["transcript"]
