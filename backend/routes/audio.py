"""Audio effect racks (AUDIO_SUITE_PLAN.md §2.6, brief A1a).

The effect catalogue, the user's presets, a project's racks (A1, MASTER, A2…
and one per clip) and a short rendered preview of one rack. Racks live on the
Timeline, so the render compiler reads them from the same place. The live
socket (`WS /{project}/live`) is `routes/audio_live.py`, on the same prefix.
Whole-rack presets (brief A5) are saved racks any scope can be set to; the
Studio applies a channel's one to the A1 rack before each render.
"""

import asyncio
import json
import logging
import os
import re
import secrets
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import DATA_DIR, FFMPEG_BIN, PROJECTS_DIR
from render import preview as frame_preview
from render.audio import build_voice_chain
from render.compiler import rack_filters
from store.project_store import ProjectStore
from timeline.schema import AudioEffect, SourceFile, Timeline, TimelineItem
from utils.proc import NO_WINDOW

logger = logging.getLogger("routes.audio")

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

# User presets, {effect: {name: params}}, beside the other data files.
PRESETS_PATH = DATA_DIR / "audio_presets.json"
_presets_lock = threading.Lock()

# Whole-rack presets, {name: {effects: [AudioEffect], created, note}}.
RACK_PRESETS_PATH = DATA_DIR / "audio_rack_presets.json"
_rack_presets_lock = threading.Lock()

_TRACK_SCOPE = re.compile(r"^A[1-9][0-9]*$")
# An effect id is also the ffmpeg instance stem (`volume@fx_7f3a_0`), so it
# must not carry anything the filtergraph parser reads as syntax.
_EFFECT_ID = re.compile(r"^[A-Za-z0-9_]+$")


# ------------------------------------------------------------------ catalogue

def _fallback_catalogue() -> Dict[str, Any]:
    """What GET /effects serves while `audio.catalogue` is absent: the one
    effect the compiler's fallback can compile."""
    return {
        "effects": [{
            "id": "amplify",
            "name": "Amplify",
            "category": "Amplitude and Compression",
            "engine": "ffmpeg",
            "description": "Raises or lowers the level by a fixed gain.",
            "params": [{"key": "gain_db", "label": "Gain", "type": "number",
                        "min": -40.0, "max": 40.0, "step": 0.1, "default": 0.0,
                        "unit": "dB", "live": True}],
            "presets": {},
            "windowable": True,
        }],
        "categories": ["Amplitude and Compression"],
    }


@router.get("/effects")
async def list_effects():
    """The effect catalogue the Audio panel generates its editors from."""
    try:
        from audio.catalogue import CATEGORIES
        from audio.compile import public_catalogue
    except ImportError:
        return _fallback_catalogue()
    catalogue = public_catalogue()
    if isinstance(catalogue.get("effects"), list):
        return {"effects": catalogue["effects"],
                "categories": list(catalogue.get("categories") or CATEGORIES)}
    return {"effects": list(catalogue.values()), "categories": list(CATEGORIES)}


# -------------------------------------------------------------------- presets

class PresetRequest(BaseModel):
    effect: str
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


def _read_presets() -> Dict[str, Dict[str, Dict[str, Any]]]:
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # Refused rather than treated as empty: the next save would overwrite
        # every preset the creator has.
        raise HTTPException(status_code=500,
                            detail=f"{PRESETS_PATH.name} is unreadable: {exc}")
    return data if isinstance(data, dict) else {}


def _write_presets(data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Atomic, like ProjectStore.save_project."""
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(PRESETS_PATH.parent), prefix="audio_presets_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, PRESETS_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@router.get("/presets")
async def list_presets():
    with _presets_lock:
        return {"presets": _read_presets()}


@router.post("/presets")
async def save_preset(body: PresetRequest):
    effect, name = body.effect.strip(), body.name.strip()
    if not effect or not name or "/" in effect or "/" in name:
        raise HTTPException(status_code=422,
                            detail="effect and name must be non-empty and contain no '/'")
    with _presets_lock:
        presets = _read_presets()
        presets.setdefault(effect, {})[name] = body.params
        _write_presets(presets)
    return {"presets": presets}


@router.delete("/presets/{effect}/{name}")
async def delete_preset(effect: str, name: str):
    with _presets_lock:
        presets = _read_presets()
        if name not in presets.get(effect, {}):
            raise HTTPException(status_code=404, detail=f"No preset {name!r} for {effect!r}")
        del presets[effect][name]
        if not presets[effect]:
            del presets[effect]
        _write_presets(presets)
    return {"presets": presets}


# ---------------------------------------------------------------------- racks

class RackRequest(BaseModel):
    scope: str
    effects: List[AudioEffect] = Field(default_factory=list)


class PreviewRequest(BaseModel):
    scope: str
    source_id: Optional[str] = None
    start_s: float
    seconds: float
    effects: Optional[List[AudioEffect]] = None


def _load(project_id: str, readonly: bool = False):
    """The project dict and its Timeline. `readonly` reads through the store's
    per-version cache; that dict must not be mutated."""
    getter = project_store.get_project_readonly if readonly else project_store.get_project
    p_data = getter(project_id)
    if not p_data or not p_data.get("timeline"):
        raise HTTPException(status_code=404, detail="Project or timeline not found")
    return p_data, Timeline.model_validate(p_data["timeline"])


def _walk_items(items: List[TimelineItem]) -> Iterator[TimelineItem]:
    for item in items:
        yield item
        yield from _walk_items(item.children)


def _find_item(tl: Timeline, item_id: str) -> Optional[TimelineItem]:
    return next((i for i in _walk_items(tl.items) if i.id == item_id), None)


def _parse_scope(scope: str) -> Optional[str]:
    """None for a track or MASTER rack, the item id for `item:<id>`."""
    if scope == "MASTER" or _TRACK_SCOPE.match(scope):
        return None
    if scope.startswith("item:") and len(scope) > len("item:"):
        return scope[len("item:"):]
    raise HTTPException(status_code=422,
                        detail=f"Unknown rack scope {scope!r}: use A1, A2…, MASTER or item:<id>")


def _check_id_format(effects: List[AudioEffect]) -> None:
    for effect in effects:
        if not _EFFECT_ID.match(effect.id):
            raise HTTPException(
                status_code=422,
                detail=f"Effect id {effect.id!r} may hold only letters, digits and '_' "
                       f"(it names an ffmpeg filter instance)")


def _check_ids(tl: Timeline, scope: str, effects: List[AudioEffect]) -> None:
    """Ids are unique in the project: they name the ffmpeg instances."""
    _check_id_format(effects)
    ids = [e.id for e in effects]
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        raise HTTPException(status_code=422, detail=f"Effect ids repeated in the rack: {twice}")
    taken = {e.id for key, rack in tl.audio_racks.items() if key != scope for e in rack}
    taken |= {e.id for item in _walk_items(tl.items) if f"item:{item.id}" != scope
              for e in item.audio_effects}
    clash = sorted(set(ids) & taken)
    if clash:
        raise HTTPException(status_code=422,
                            detail=f"Effect ids already used elsewhere in the project: {clash}")


@router.get("/{project_id}/racks")
async def get_racks(project_id: str):
    _p_data, tl = _load(project_id, readonly=True)
    racks: Dict[str, List[Dict[str, Any]]] = {"A1": [], "MASTER": []}
    racks.update({scope: [e.model_dump() for e in rack] for scope, rack in tl.audio_racks.items()})
    items = {item.id: [e.model_dump() for e in item.audio_effects]
             for item in _walk_items(tl.items) if item.audio_effects}
    sources = {sid: {"audio_override": src.audio_override, "history_len": len(src.audio_history)}
               for sid, src in tl.sources.items()}
    return {"racks": racks, "items": items, "sources": sources}


@router.put("/{project_id}/rack")
async def put_rack(project_id: str, body: RackRequest):
    """Replace one rack: A1, MASTER, A2… or `item:<id>` (an A2+ clip)."""
    scope = body.scope.strip()
    item_id = _parse_scope(scope)
    p_data, tl = _load(project_id)
    _check_ids(tl, scope, body.effects)
    if item_id is not None:
        item = _find_item(tl, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"No clip {item_id!r} in the timeline")
        track = item.track.upper()
        if not track.startswith("A") or track == "A1":
            # A1 clips are rebuilt from the transcript and the compiler ignores
            # their racks; programme effects go on the A1 rack, with a window.
            raise HTTPException(status_code=422,
                                detail=f"A clip rack belongs on an A2+ clip, not {item.track}; "
                                       f"use the A1 rack for the programme voice")
        item.audio_effects = list(body.effects)
    else:
        tl.audio_racks[scope] = list(body.effects)
    tl.revision += 1
    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)
    frame_preview.invalidate(project_id)
    return {"scope": scope, "effects": [e.model_dump() for e in body.effects],
            "revision": tl.revision}


# --------------------------------------------------------------- rack presets

class RackPresetRequest(BaseModel):
    name: str
    effects: List[AudioEffect] = Field(default_factory=list)
    note: Optional[str] = None


class ApplyRackPresetRequest(BaseModel):
    scope: str
    name: str


def _read_rack_presets() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(RACK_PRESETS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # Refused rather than treated as empty, as for the effect presets.
        raise HTTPException(status_code=500,
                            detail=f"{RACK_PRESETS_PATH.name} is unreadable: {exc}")
    return data if isinstance(data, dict) else {}


def _write_rack_presets(data: Dict[str, Dict[str, Any]]) -> None:
    """Atomic, like `_write_presets`."""
    RACK_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(RACK_PRESETS_PATH.parent), prefix="audio_rack_presets_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, RACK_PRESETS_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _fresh_effect_id(taken: Set[str]) -> str:
    """`fx_<8 hex>`, as the panel makes them, not in `taken` (and added to it)."""
    while True:
        candidate = f"fx_{secrets.token_hex(4)}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate


@router.get("/rack_presets")
async def list_rack_presets():
    with _rack_presets_lock:
        return {"presets": _read_rack_presets()}


@router.post("/rack_presets")
async def save_rack_preset(body: RackPresetRequest):
    """Save (or replace) a whole rack under `name`. The effects' ids are kept
    as sent; applying the preset gives every effect a fresh one."""
    name = body.name.strip()
    if not name or "/" in name:
        raise HTTPException(status_code=422, detail="name must be non-empty and contain no '/'")
    with _rack_presets_lock:
        presets = _read_rack_presets()
        presets[name] = {"effects": [e.model_dump() for e in body.effects],
                         "created": datetime.now().isoformat(timespec="seconds"),
                         "note": (body.note or "").strip()}
        _write_rack_presets(presets)
    return {"presets": presets}


@router.delete("/rack_presets/{name}")
async def delete_rack_preset(name: str):
    with _rack_presets_lock:
        presets = _read_rack_presets()
        if name not in presets:
            raise HTTPException(status_code=404, detail=f"No rack preset {name!r}")
        del presets[name]
        _write_rack_presets(presets)
    return {"presets": presets}


@router.post("/{project_id}/rack/apply_preset")
async def apply_rack_preset(project_id: str, body: ApplyRackPresetRequest):
    """Replace one rack (A1, MASTER, A2… or `item:<id>`) with a rack preset's
    effects, each under a fresh id unique in the project. Checked, saved and
    answered exactly as `PUT /{project}/rack`."""
    with _rack_presets_lock:
        preset = _read_rack_presets().get(body.name)
    if not isinstance(preset, dict):
        raise HTTPException(status_code=404, detail=f"No rack preset {body.name!r}")
    _p_data, tl = _load(project_id, readonly=True)
    taken = {e.id for rack in tl.audio_racks.values() for e in rack}
    taken |= {e.id for item in _walk_items(tl.items) for e in item.audio_effects}
    effects = [AudioEffect.model_validate({**raw, "id": _fresh_effect_id(taken)})
               for raw in preset.get("effects") or [] if isinstance(raw, dict)]
    return await put_rack(project_id, RackRequest(scope=body.scope, effects=effects))


# -------------------------------------------------------------------- preview

def _programme_source(tl: Timeline) -> Optional[SourceFile]:
    """The source of the first A1 clip: the programme voice."""
    a1 = sorted((i for i in tl.items if i.track == "A1" and i.source_id in tl.sources),
                key=lambda i: i.timeline_start_frame)
    return tl.sources[a1[0].source_id] if a1 else None


@router.post("/{project_id}/preview")
async def preview_rack(project_id: str, body: PreviewRequest):
    """A short WAV of one rack applied to the source audio.

    The approach of `routes/presentation.py::audio_preview`: `seconds` of the
    source from `start_s`, through the voice chain plus the A1/MASTER rack, or
    through a clip's own rack. `effects` previews an unsaved rack. A source
    with processed audio is read from its `audio_override`.
    """
    scope = body.scope.strip()
    _p_data, tl = _load(project_id, readonly=True)
    if body.source_id is not None and body.source_id not in tl.sources:
        raise HTTPException(status_code=404, detail=f"No source {body.source_id!r} in the timeline")

    if scope in ("A1", "MASTER"):
        effects = body.effects if body.effects is not None else tl.audio_racks.get(scope, [])
        _check_id_format(effects)
        chain = build_voice_chain(tl.audio_master) + rack_filters(effects, scope=scope)
        source = tl.sources[body.source_id] if body.source_id else _programme_source(tl)
    elif scope.startswith("item:") and len(scope) > len("item:"):
        item = _find_item(tl, scope[len("item:"):])
        if item is None:
            raise HTTPException(status_code=404, detail=f"No clip {scope[5:]!r} in the timeline")
        effects = body.effects if body.effects is not None else item.audio_effects
        _check_id_format(effects)
        chain = rack_filters(effects, scope=scope)
        source = tl.sources.get(body.source_id or item.source_id or "")
    else:
        raise HTTPException(status_code=422,
                            detail=f"Preview takes scope A1, MASTER or item:<id>, not {scope!r}")

    if source is None:
        raise HTTPException(status_code=404, detail="No source audio to preview")
    input_path = source.audio_override or source.path
    if not input_path or not Path(input_path).exists():
        raise HTTPException(status_code=404, detail=f"Source audio not found: {input_path}")

    seconds = max(1.0, min(60.0, float(body.seconds or 20.0)))
    start_s = max(0.0, float(body.start_s or 0.0))

    out_dir = PROJECTS_DIR / project_id / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = re.sub(r"[^A-Za-z0-9_-]", "_", scope) or "rack"
    output_path = out_dir / f"preview_{safe_scope}.wav"

    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start_s:.3f}", "-t", f"{seconds:.3f}", "-i", str(input_path)]
    if chain:
        # A complex graph, never -af: a rack chain may carry ";"-separated
        # segments with its own labels and extra sources (a reverb loads its
        # impulse response in-graph with amovie), which a simple filtergraph
        # rejects. The chain's head input and tail output are unlabeled by
        # contract (audio/compile.py), so [0:a] and [out] wrap it.
        command.extend(["-filter_complex", "[0:a]" + ",".join(chain) + "[out]",
                        "-map", "[out]"])
    command.extend(["-vn", "-ar", "48000", "-ac", "2", str(output_path)])

    try:
        result = await asyncio.to_thread(
            subprocess.run, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Audio preview render timed out")
    if result.returncode != 0 or not output_path.exists():
        err = (result.stderr or b"").decode("utf-8", "replace")[-500:]
        raise HTTPException(status_code=500, detail=f"Preview render failed: {err}")

    return {"path": str(output_path), "seconds": seconds}
