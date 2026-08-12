import asyncio
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from store.project_store import ProjectStore
from config import PROJECTS_DIR, OUTPUT_DIR
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine, detect_speech_silence_intervals, analyze_disfluencies
from timeline import build_timeline_from_transcript, toggle_word, Timeline
from render import render_timeline_async

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

class ToggleWordRequest(BaseModel):
    word_id: str
    enabled: bool

@router.post("/{project_id}/generate")
async def generate_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")

    # 1. Extract Audio
    audio_path = extract_audio(source_video)

    # 2. Get Video Info
    v_info = get_video_info(source_video)
    fps_num = int(round(v_info.get("fps", 30)))
    fps_den = 1

    # 3. ASR Transcription
    words = await whisper_engine.transcribe_audio_async(audio_path)

    # 4. Disfluency Analysis
    annotated_words = analyze_disfluencies(words)

    # 5. Build Timeline
    tl = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=v_info.get("duration", 0.0),
        transcript_words=annotated_words,
        fps_num=fps_num,
        fps_den=fps_den,
        width=v_info.get("width", 1920),
        height=v_info.get("height", 1080)
    )

    # Store timeline in project data
    p_data["timeline"] = tl.model_dump()
    p_data["status"] = "ready"
    project_store.save_project(project_id, p_data)

    return {"status": "success", "timeline": tl}

@router.get("/{project_id}")
async def get_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    tl_data = p_data.get("timeline")
    if not tl_data:
        raise HTTPException(status_code=404, detail="Timeline not yet generated for this project")
    return tl_data

@router.post("/{project_id}/toggle_word")
async def toggle_word_endpoint(project_id: str, body: ToggleWordRequest):
    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")

    tl = Timeline.model_validate(p_data["timeline"])
    primary_src_id = list(tl.sources.keys())[0]

    success = toggle_word(tl, body.word_id, body.enabled, primary_src_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Word {body.word_id} not found")

    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)

    return {"status": "success", "timeline": tl}

@router.post("/{project_id}/render")
async def render_project_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")

    tl = Timeline.model_validate(p_data["timeline"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUTPUT_DIR / f"{project_id}_rendered.mp4")

    rendered_path = await render_timeline_async(tl, out_path)
    p_data["rendered_video"] = rendered_path
    p_data["status"] = "rendered"
    project_store.save_project(project_id, p_data)

    return {"status": "success", "output_path": rendered_path}
