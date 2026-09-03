from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import PROJECTS_DIR, OUTPUT_DIR
from store.project_store import ProjectStore
from models import Project
from agents.edit_agent import EditAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.caption_agent import CaptionAgent

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
edit_agent = EditAgent()
thumbnail_agent = ThumbnailAgent()
caption_agent = CaptionAgent()

class AgentJobRequest(BaseModel):
    project_id: str
    generate_thumbnail: bool = True
    burn_captions: bool = True
    title: Optional[str] = None
    genre: Optional[str] = None


def _project_genre(req: AgentJobRequest, project: Project, project_dir) -> str:
    """The genre to style a standalone thumbnail with: the caller's choice,
    else what the last presentation pass detected, else the transcript's own
    words. Never raises — an unknown genre just styles nothing."""
    from presentation.genre import keyword_genre, normalise
    if req.genre:
        return normalise(req.genre)
    try:
        import json
        plan = json.loads((project_dir / "shot_plan.json").read_text(encoding="utf-8"))
        if plan.get("genre"):
            return normalise(plan["genre"])
    except Exception:
        pass
    try:
        segments = project.transcript.segments if project.transcript else []
        return keyword_genre(" ".join(s.text for s in segments))
    except Exception:
        return "general"

@router.post("/full_edit")
async def full_auto_edit(req: AgentJobRequest):
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(p_data)

    try:
        results = await edit_agent.execute_full_auto_edit(
            project=project,
            generate_thumbnail=req.generate_thumbnail,
            burn_captions=req.burn_captions,
        )
        return {"status": "completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/thumbnail")
async def generate_thumbnail(req: AgentJobRequest):
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(p_data)
    project_dir = PROJECTS_DIR / req.project_id

    try:
        thumb_path = await thumbnail_agent.generate_thumbnail(
            project, project_dir, title=req.title,
            genre=_project_genre(req, project, project_dir))
        return {"status": "completed", "thumbnail_path": thumb_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/captions")
async def generate_captions(req: AgentJobRequest):
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(p_data)
    if not project.transcript or not project.transcript.segments:
        raise HTTPException(status_code=400, detail="Project must be transcribed first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_video = str(OUTPUT_DIR / f"{project.id}_captioned.mp4")

    try:
        res_video = caption_agent.burn_captions(
            input_video=project.source_video,
            output_video=out_video,
            segments=project.transcript.segments,
        )
        return {"status": "completed", "output_video": res_video}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
