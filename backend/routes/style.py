"""Reference-video style matching endpoints.

Analysis is CPU-bound ffmpeg + numpy work measured in seconds, so it runs in a
worker thread rather than blocking the event loop.
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import PROJECTS_DIR
from store.project_store import ProjectStore
from style import ApplyOptions, analyze_video, apply_profile
from style import profile as profile_store
from style.frames import FrameSampleError
from timeline import Timeline

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


class AnalyzeRequest(BaseModel):
    path: str
    name: Optional[str] = None


class ApplyRequest(BaseModel):
    project_id: str
    profile_id: str
    look: bool = True
    motion: bool = True
    captions: bool = True
    transitions: bool = True


@router.get("/")
async def list_styles():
    """Every saved style profile, newest first. Profiles are shared across projects."""
    return {"profiles": [p.model_dump() for p in profile_store.list_profiles()]}


@router.get("/{profile_id}")
async def get_style(profile_id: str):
    found = profile_store.load(profile_id)
    if not found:
        raise HTTPException(status_code=404, detail="Style profile not found")
    return found


@router.post("/analyze")
async def analyze(body: AnalyzeRequest):
    """Measure a reference video and save the resulting profile."""
    if not Path(body.path).is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {body.path}")
    try:
        built = await asyncio.to_thread(analyze_video, body.path, body.name)
    except FrameSampleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # analysis touches ffmpeg, numpy and cv2
        raise HTTPException(status_code=500, detail=f"Style analysis failed: {exc}")

    profile_store.save(built)
    return {"status": "success", "profile": built}


@router.delete("/{profile_id}")
async def remove_style(profile_id: str):
    if not profile_store.delete(profile_id):
        raise HTTPException(status_code=404, detail="Style profile not found")
    return {"status": "deleted", "id": profile_id}


@router.post("/apply")
async def apply(body: ApplyRequest):
    """Apply the selected parts of a profile to a project's timeline."""
    found = profile_store.load(body.profile_id)
    if not found:
        raise HTTPException(status_code=404, detail="Style profile not found")

    p_data = project_store.get_project(body.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    if not p_data.get("timeline"):
        raise HTTPException(
            status_code=400,
            detail="This project has no timeline yet — transcribe it or drag media in first.")

    tl = Timeline.model_validate(p_data["timeline"])
    options = ApplyOptions(look=body.look, motion=body.motion,
                           captions=body.captions, transitions=body.transitions)

    report = await asyncio.to_thread(
        apply_profile, tl, found, p_data.get("source_video"), options)

    p_data["timeline"] = tl.model_dump()
    # Caption geometry is not applied here: captions are generated on demand, so
    # the overrides are stashed for the generator to pick up.
    caption_report = report.get("captions") or {}
    if caption_report.get("applied"):
        settings = p_data.get("settings") or {}
        settings["caption_style_overrides"] = caption_report["overrides"]
        p_data["settings"] = settings
    project_store.save_project(body.project_id, p_data)

    return {"status": "success", "report": report, "timeline": tl}
