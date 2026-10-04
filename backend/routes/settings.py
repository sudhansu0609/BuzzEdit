from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from store.app_settings import (
    app_settings, OVERRIDE_MODES, get_presentation_overrides_state,
    set_presentation_overrides,
)
from store.project_store import ProjectStore
from config import PROJECTS_DIR

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


class SetLastProjectRequest(BaseModel):
    project_id: Optional[str] = None


class PresentationOverridesRequest(BaseModel):
    presentation_overrides: Dict[str, Any] = {}
    override_mode: str = "off"


class AutoSaveRequest(BaseModel):
    project_id: str
    project_data: Dict[str, Any]


def _masked_all() -> Dict[str, Any]:
    from store.app_settings import _mask_key
    out = app_settings.get_all()
    editor = out.get("auto_cut_editor")
    if isinstance(editor, dict) and editor.get("api_key"):
        out["auto_cut_editor"] = {**editor, "api_key": _mask_key(editor["api_key"])}
    return out


@router.get("/")
async def get_app_settings():
    """Get all app settings including last project ID (secrets masked)."""
    return _masked_all()


@router.put("/")
async def update_app_settings(updates: Dict[str, Any]):
    """Update app settings."""
    # The editor's proxy key has its own route; a generic save round-tripping the
    # masked value must never overwrite the real key.
    updates = {k: v for k, v in updates.items() if k != "auto_cut_editor"}
    app_settings.update(updates)
    return _masked_all()


class AutoCutEditorRequest(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    planner: Optional[str] = None      # "editor" | "classic": the default for every project


@router.get("/auto_cut_editor")
async def get_auto_cut_editor():
    """Which planner cuts by default, and whether the AI editor has a proxy to call."""
    from asr.auto_edit import choose_planner
    from asr.editor import DEFAULT_BASE_URL, DEFAULT_EFFORT, DEFAULT_MODEL, proxy_config
    proxy = proxy_config()
    own = app_settings.get("auto_cut_editor") or {}
    return {"planner": choose_planner(None), "model": DEFAULT_MODEL, "effort": DEFAULT_EFFORT,
            "base_url": proxy["base_url"] or DEFAULT_BASE_URL, "key_configured": bool(proxy["api_key"]),
            "key_source": "app_settings" if own.get("api_key") else ("env" if proxy["api_key"] else None)}


@router.put("/auto_cut_editor")
async def put_auto_cut_editor(body: AutoCutEditorRequest):
    """Store the proxy the AI editor calls (key never echoed back) and the default planner."""
    own = dict(app_settings.get("auto_cut_editor") or {})
    if body.base_url is not None:
        own["base_url"] = body.base_url
    if body.api_key is not None:
        own["api_key"] = body.api_key
    app_settings.set("auto_cut_editor", own)
    if body.planner is not None:
        if body.planner not in ("editor", "classic"):
            raise HTTPException(status_code=400, detail="planner must be 'editor' or 'classic'")
        app_settings.set("auto_cut_planner", body.planner)
    return await get_auto_cut_editor()


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


@router.get("/presentation_overrides")
async def get_presentation_overrides():
    """BuzzEdit-side presentation defaults that win over whatever
    BuzzcafStudio sends for a presentation job — see
    store.app_settings.apply_presentation_overrides."""
    return get_presentation_overrides_state()


@router.put("/presentation_overrides")
async def put_presentation_overrides(body: PresentationOverridesRequest):
    if body.override_mode not in OVERRIDE_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"override_mode must be one of {OVERRIDE_MODES}")
    return set_presentation_overrides(body.presentation_overrides, body.override_mode)


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
