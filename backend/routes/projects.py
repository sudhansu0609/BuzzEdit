import uuid
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from models import Project
from config import PROJECTS_DIR, TEMP_DIR
from utils.ffmpeg_utils import get_video_info, FFmpegError

router = APIRouter()
projects: Dict[str, Project] = {}


class PathImportRequest(BaseModel):
    path: str


@router.post("/import_path", response_model=Project)
async def import_video_path(body: PathImportRequest):
    src_path = Path(body.path)
    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"File not found at path: {body.path}")

    file_ext = src_path.suffix
    project_id = str(uuid.uuid4())[:8]
    dest_video_path = PROJECTS_DIR / f"{project_id}_source{file_ext}"

    try:
        shutil.copy(str(src_path), str(dest_video_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy video file: {str(e)}")

    try:
        video_info = get_video_info(str(dest_video_path))
    except Exception as e:
        dest_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")

    project = Project(
        id=project_id,
        name=src_path.name,
        source_video=str(dest_video_path),
        status="draft",
    )

    project.settings["source_width"] = video_info["width"]
    project.settings["source_height"] = video_info["height"]
    project.settings["source_duration"] = video_info["duration"]

    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(project.model_dump_json(indent=2))
    projects[project_id] = project

    return project


@router.post("/upload", response_model=Project)
async def upload_video(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix
    project_id = str(uuid.uuid4())[:8]
    video_path = PROJECTS_DIR / f"{project_id}_source{file_ext}"

    content = await file.read()
    video_path.write_bytes(content)

    try:
        video_info = get_video_info(str(video_path))
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")

    project = Project(
        id=project_id,
        name=file.filename,
        source_video=str(video_path),
        status="draft",
    )

    project.settings["source_width"] = video_info["width"]
    project.settings["source_height"] = video_info["height"]
    project.settings["source_duration"] = video_info["duration"]

    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(project.model_dump_json(indent=2))
    projects[project_id] = project

    return project


@router.get("/list")
async def list_projects():
    result = []
    for pid, proj in projects.items():
        result.append({
            "id": proj.id,
            "name": proj.name,
            "status": proj.status,
            "source_video": proj.source_video,
        })
    return {"projects": result}


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    if project_id in projects:
        return projects[project_id]

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    proj_file = project_dir / "project.json"
    if not proj_file.exists():
        raise HTTPException(status_code=404, detail="Project file not found")

    project = Project.model_validate_json(proj_file.read_text())
    projects[project_id] = project
    return project


@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, updates: Dict[str, Any]):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    updated = project.model_copy(update=updates)

    project_dir = PROJECTS_DIR / project_id
    (project_dir / "project.json").write_text(updated.model_dump_json(indent=2))
    projects[project_id] = updated

    return updated


@router.put("/{project_id}/settings", response_model=Project)
async def update_settings(project_id: str, settings: Dict[str, Any]):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    merged = {**project.settings, **settings}
    updated = project.model_copy(update={"settings": merged})

    project_dir = PROJECTS_DIR / project_id
    (project_dir / "project.json").write_text(updated.model_dump_json(indent=2))
    projects[project_id] = updated

    return updated


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    if project_id in projects:
        del projects[project_id]

    project_dir = PROJECTS_DIR / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)

    return {"status": "deleted", "id": project_id}


@router.get("/{project_id}/video_info")
async def get_video_info_endpoint(project_id: str):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    try:
        info = get_video_info(project.source_video)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
