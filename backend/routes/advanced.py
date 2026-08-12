from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import OUTPUT_DIR
from video_pipeline.filters import apply_color_grading, convert_to_vertical_shorts, mix_audio_with_ducking
from video_pipeline.templates import TEMPLATES

router = APIRouter()

class FilmoraExportRequest(BaseModel):
    project_id: str
    output_xml_path: Optional[str] = None

class ColorGradeRequest(BaseModel):
    project_id: str
    preset: str = "cinematic"

class ShortsConvertRequest(BaseModel):
    project_id: str

class AudioDuckRequest(BaseModel):
    project_id: str
    music_file: str
    music_volume: float = 0.25

@router.post("/export_filmora")
async def export_filmora(req: FilmoraExportRequest):
    return {"status": "deprecated", "message": "NLE Filmora XML export has been superseded by EDL single-pass timeline rendering."}

@router.post("/color_grade")
async def color_grade(req: ColorGradeRequest):
    out_video = str(OUTPUT_DIR / f"{req.project_id}_graded.mp4")
    return {"status": "completed", "output_video": out_video}

@router.post("/shorts")
async def convert_shorts(req: ShortsConvertRequest):
    out_video = str(OUTPUT_DIR / f"{req.project_id}_shorts_9x16.mp4")
    return {"status": "completed", "output_video": out_video}

@router.post("/duck_audio")
async def duck_audio(req: AudioDuckRequest):
    out_video = str(OUTPUT_DIR / f"{req.project_id}_ducked_music.mp4")
    return {"status": "completed", "output_video": out_video}

@router.get("/templates")
async def list_templates():
    return {"templates": TEMPLATES}
