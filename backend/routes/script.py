"""The project's script: upload it, read it back, remove it.

Uploading aligns the script to the transcript straight away (the words'
spelling changes, so captions regenerated afterwards read as written) and
stores the paragraph/directive structure the presentation pass plans from.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import PROJECTS_DIR
from presentation.script import apply_project_script
from store.project_store import ProjectStore
from timeline.schema import Timeline

logger = logging.getLogger("routes.script")

router = APIRouter()


class ScriptBody(BaseModel):
    text: str


def _store() -> ProjectStore:
    return ProjectStore(base_dir=str(PROJECTS_DIR))


def _summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "token_count": record.get("token_count", 0),
        "aligned_words": record.get("aligned_words", 0),
        "spelling_fixed": record.get("spelling_fixed", 0),
        "paragraphs": len(record.get("paragraphs") or []),
        "directives": [{"kind": d["kind"], "arg": d["arg"], "at": d["at"]}
                       for d in (record.get("directives") or [])],
        "text": record.get("text", ""),
    }


@router.put("/{project_id}/script")
async def set_script(project_id: str, body: ScriptBody):
    store = _store()
    data = store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="The script is empty")

    if not data.get("timeline"):
        # No transcript yet: keep the text; the pass aligns it once one exists.
        data["script"] = {"text": body.text, "paragraphs": [], "directives": [],
                          "token_count": 0, "aligned_words": 0, "spelling_fixed": 0}
        store.save_project(project_id, data)
        return {"status": "stored", "aligned": False, **_summary(data["script"])}

    timeline = Timeline.model_validate(data["timeline"])
    record = apply_project_script(data, timeline, text=body.text)
    if record is None:
        raise HTTPException(status_code=422, detail="Nothing to align the script to yet")
    data["timeline"] = timeline.model_dump()
    store.save_project(project_id, data)
    return {"status": "aligned", "aligned": True, **_summary(record)}


@router.get("/{project_id}/script")
async def get_script(project_id: str):
    data = _store().get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    record = data.get("script")
    if not record:
        return {"status": "none", "text": ""}
    return {"status": "aligned" if record.get("aligned_words") else "stored",
            **_summary(record)}


@router.delete("/{project_id}/script")
async def delete_script(project_id: str):
    store = _store()
    data = store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    data.pop("script", None)
    store.save_project(project_id, data)
    return {"status": "removed"}
