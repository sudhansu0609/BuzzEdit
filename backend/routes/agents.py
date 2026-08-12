from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import PROJECTS_DIR, OUTPUT_DIR
from store.project_store import ProjectStore
from models import Project
from agents.edit_agent import EditAgent
from agents.broll_agent import BRollAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.caption_agent import CaptionAgent

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
edit_agent = EditAgent()
broll_agent = BRollAgent()
thumbnail_agent = ThumbnailAgent()
caption_agent = CaptionAgent()

class AgentJobRequest(BaseModel):
    project_id: str
    generate_broll: bool = True
    generate_thumbnail: bool = True
    burn_captions: bool = True
    title: Optional[str] = None

@router.post("/full_edit")
async def full_auto_edit(req: AgentJobRequest):
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(p_data)

    try:
        results = await edit_agent.execute_full_auto_edit(
            project=project,
            generate_broll=req.generate_broll,
            generate_thumbnail=req.generate_thumbnail,
            burn_captions=req.burn_captions,
        )
        return {"status": "completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/broll")
async def generate_broll(req: AgentJobRequest):
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = PROJECTS_DIR / req.project_id

    try:
        if "timeline" in p_data:
            from timeline import Timeline
            tl = Timeline.model_validate(p_data["timeline"])
            added = await broll_agent.generate_and_apply_broll(tl, project_dir)
            p_data["timeline"] = tl.model_dump()
            project_store.save_project(req.project_id, p_data)
            return {"status": "completed", "broll_count": added}
        return {"status": "error", "detail": "Timeline not generated"}
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
        thumb_path = await thumbnail_agent.generate_thumbnail(project, project_dir, title=req.title)
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
