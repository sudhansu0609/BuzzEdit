from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from store.app_settings import app_settings
from store.project_store import ProjectStore
from config import PROJECTS_DIR

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


class SetLastProjectRequest(BaseModel):
    project_id: Optional[str] = None


class AutoSaveRequest(BaseModel):
    project_id: str
    project_data: Dict[str, Any]


@router.get("/")
async def get_app_settings():
    """Get all app settings including last project ID."""
    return app_settings.get_all()


@router.put("/")
async def update_app_settings(updates: Dict[str, Any]):
    """Update app settings."""
    app_settings.update(updates)
    return app_settings.get_all()


@router.get("/last_project")
async def get_last_project():
    """Get the last opened project ID and verify it still exists."""
    project_id = app_settings.get_last_project()
    if not project_id:
        return {"project_id": None, "exists": False}

    project_data = project_store.get_project(project_id)
    if project_data:
        return {
            "project_id": project_id,
            "exists": True,
            "project": project_data
        }
    else:
        app_settings.set_last_project(None)
        return {"project_id": None, "exists": False}


@router.post("/last_project")
async def set_last_project(body: SetLastProjectRequest):
    """Set the last opened project ID."""
    app_settings.set_last_project(body.project_id)
    return {"status": "ok", "project_id": body.project_id}


@router.post("/auto_save")
async def auto_save_project(body: AutoSaveRequest):
    """
    Auto-save endpoint for crash recovery.
    Saves project state atomically to disk.
    """
    project_id = body.project_id
    project_data = body.project_data

    existing = project_store.get_project(project_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Project not found")

    existing.update(project_data)
    project_store.save_project(project_id, existing)
    app_settings.mark_auto_save()
    app_settings.set_last_project(project_id)

    return {
        "status": "saved",
        "project_id": project_id,
        "timestamp": app_settings.get("last_save_timestamp")
    }


@router.get("/recovery_state")
async def get_recovery_state():
    """
    Get app recovery state on startup.
    Returns last project if available, otherwise returns list of recent projects.
    """
    last_project_id = app_settings.get_last_project()
    last_save = app_settings.get("last_save_timestamp")

    recovery_project = None
    if last_project_id:
        project_data = project_store.get_project(last_project_id)
        if project_data:
            recovery_project = project_data

    recent_projects = project_store.list_projects()[:5]

    return {
        "has_recovery": recovery_project is not None,
        "last_project_id": last_project_id if recovery_project else None,
        "last_save_timestamp": last_save,
        "recovery_project": recovery_project,
        "recent_projects": [
            {"id": p.get("id"), "name": p.get("name"), "status": p.get("status")}
            for p in recent_projects
        ]
    }
