import uuid
import shutil
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from models import Project
from config import PROJECTS_DIR
from utils.ffmpeg_utils import get_video_info
from store.project_store import ProjectStore

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

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

    project_store.save_project(project_id, project.model_dump())
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

    project_store.save_project(project_id, project.model_dump())
    return project

@router.get("/list")
async def list_projects():
    projs = project_store.list_projects()
    result = []
    for p in projs:
        result.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "status": p.get("status"),
            "source_video": p.get("source_video"),
            "sourceVideo": p.get("source_video"),
        })
    return {"projects": result}

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    data = project_store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project.model_validate(data)

@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, updates: Dict[str, Any]):
    data = project_store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(data)
    updated = project.model_copy(update=updates)
    project_store.save_project(project_id, updated.model_dump())
    return updated

@router.delete("/{project_id}")
async def delete_project(project_id: str):
    success = project_store.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "id": project_id}
