from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from config import PROJECTS_DIR, OUTPUT_DIR
from video_pipeline.filmora_exporter import export_filmora_xml
from video_pipeline.filters import apply_color_grading, convert_to_vertical_shorts, mix_audio_with_ducking
from video_pipeline.templates import TEMPLATES, get_template_settings
from routes.rendering import get_good_segments

router = APIRouter()


class FilmoraExportRequest(BaseModel):
    project_id: str
    output_xml_path: Optional[str] = None


class ColorGradeRequest(BaseModel):
    project_id: str
    preset: str = "cinematic" # cinematic, vibrant, warm, cool, dark


class ShortsConvertRequest(BaseModel):
    project_id: str


class AudioDuckRequest(BaseModel):
    project_id: str
    music_file: str
    music_volume: float = 0.25


@router.post("/export_filmora")
async def export_filmora(req: FilmoraExportRequest):
    from routes.projects import projects

    if req.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_id]
    good_segments = get_good_segments(project)

    out_xml = req.output_xml_path or str(OUTPUT_DIR / f"{project.id}_filmora.xml")

    try:
        xml_path = export_filmora_xml(
            source_video_path=project.source_video,
            good_segments=good_segments,
            output_xml_path=out_xml,
            project_name=project.name,
        )
        return {"status": "completed", "xml_path": xml_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color_grade")
async def color_grade(req: ColorGradeRequest):
    from routes.projects import projects

    if req.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_id]
    input_video = project.output_path or project.source_video
    out_video = str(OUTPUT_DIR / f"{project.id}_graded.mp4")

    try:
        res_path = apply_color_grading(input_video, out_video, req.preset)
        project.output_path = res_path
        return {"status": "completed", "output_video": res_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shorts")
async def convert_shorts(req: ShortsConvertRequest):
    from routes.projects import projects

    if req.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_id]
    input_video = project.output_path or project.source_video
    out_video = str(OUTPUT_DIR / f"{project.id}_shorts_9x16.mp4")

    try:
        res_path = convert_to_vertical_shorts(input_video, out_video)
        return {"status": "completed", "output_video": res_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/duck_audio")
async def duck_audio(req: AudioDuckRequest):
    from routes.projects import projects

    if req.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_id]
    input_video = project.output_path or project.source_video
    out_video = str(OUTPUT_DIR / f"{project.id}_ducked_music.mp4")

    try:
        res_path = mix_audio_with_ducking(input_video, req.music_file, out_video, req.music_volume)
        project.output_path = res_path
        return {"status": "completed", "output_video": res_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates():
    return {"templates": TEMPLATES}
