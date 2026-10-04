import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
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


# ─────────────────────── BuzzcafAI bridge: plan visuals ───────────────────────


class PlanVisualsRequest(BaseModel):
    project_id: str
    brand: str


def _transcript_plain_text(p_data: Dict[str, Any]) -> str:
    """Plain spoken text reconstructed from a project's transcript.

    `transcript` on the stored project dict is either a full
    `{segments, language, duration}` object (what this backend writes) or a
    bare list of segments (what the frontend sometimes persists -- see
    `models.Project._coerce_transcript`); accept either rather than 500ing.
    """
    transcript = p_data.get("transcript")
    if isinstance(transcript, dict):
        segments = transcript.get("segments") or []
    elif isinstance(transcript, list):
        segments = transcript
    else:
        segments = []
    return " ".join(str(s.get("text", "")) for s in segments if isinstance(s, dict)).strip()


@router.post("/plan_visuals")
async def plan_visuals(req: PlanVisualsRequest):
    """Ask BuzzcafAI to plan the visuals for this project's script or
    transcript, then apply the annotated script it hands back the same way
    `PUT /api/projects/{id}/script` does -- so the normal presentation pass
    can run immediately afterwards with no extra step from the caller.
    """
    p_data = project_store.get_project(req.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    transcript_text = ((p_data.get("script") or {}).get("text") or "").strip()
    if not transcript_text:
        transcript_text = _transcript_plain_text(p_data)
    if not transcript_text:
        raise HTTPException(status_code=400, detail="No transcript or script text yet -- transcribe first.")

    from integrations.buzzcaf_client import plan_visuals as buzzcaf_plan_visuals, BuzzcafUnavailable

    try:
        result = await asyncio.to_thread(buzzcaf_plan_visuals, transcript_text, req.brand)
    except BuzzcafUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BuzzcafAI could not plan visuals: {e}")

    annotated_script = result.get("annotated_script") or ""
    if not annotated_script.strip():
        raise HTTPException(status_code=502, detail="BuzzcafAI returned an empty annotated script.")

    if not p_data.get("timeline"):
        raise HTTPException(status_code=422, detail="Transcribe the recording before planning visuals.")

    from presentation.script import apply_project_script
    from timeline.schema import Timeline

    timeline = Timeline.model_validate(p_data["timeline"])
    record = apply_project_script(p_data, timeline, text=annotated_script)
    if record is None:
        raise HTTPException(status_code=422, detail="Nothing to align the annotated script to.")
    p_data["timeline"] = timeline.model_dump()
    project_store.save_project(req.project_id, p_data)

    return {
        "status": "applied",
        "settings": result.get("settings"),
        "visual_plan": result.get("visual_plan"),
        "skipped_beats": result.get("skipped_beats"),
        "script": {
            "text": record.get("text", ""),
            "directives": [{"kind": d["kind"], "arg": d["arg"], "at": d["at"]} for d in (record.get("directives") or [])],
            "paragraphs": len(record.get("paragraphs") or []),
        },
    }
