from typing import Dict
from fastapi import APIRouter
from pydantic import BaseModel

from store import glossary_store

router = APIRouter()


class GlossaryMergeRequest(BaseModel):
    entries: Dict[str, str]


@router.get("/{key}")
async def get_glossary(key: str):
    """The channel's own glossary file (not merged with `_global`)."""
    return {"key": key, "entries": glossary_store.load_channel(key)}


@router.put("/{key}")
async def put_glossary(key: str, body: GlossaryMergeRequest):
    """Merge `entries` into the channel's glossary file. New keys are added,
    existing keys are overwritten; everything else already on file is kept."""
    entries = glossary_store.merge_entries(key, body.entries)
    return {"key": key, "entries": entries}
