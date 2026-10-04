"""Background-music library and generation (presentation/music_gen.py).

GET  /api/music/styles           style presets, engines, builder vocabulary
GET  /api/music/library          every track, with licence status
POST /api/music/generate         {engine, style?, music_genres, instruments, moods, bpm,
                                  variations, seconds, extra_tags, genre} -> {job_id}
POST /api/music/preview          same body as /generate -> {tags, dropped}: the exact prompt
GET  /api/music/jobs/{job_id}    progress/result of a generation job, stage by stage
GET  /api/music/file/{path}      stream one library track (preview)
DELETE /api/music/file/{path}    delete one library track and its manifest entry
POST /api/music/upload           multipart {file, license, label?}: add your own track to
                                  uploads/, loudness-normalised, with its licence on record

Generation runs as a background task -- a few variations take minutes -- and,
per the GPU policy, hands the card to the engine (ACE-Step or Stable Audio
Open) first and offloads every model
when done.
"""

import asyncio
import copy
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from presentation import music_gen, workflows
from presentation.sound import AUDIO_EXTS, MUSIC_DIR

logger = logging.getLogger("routes.music")
router = APIRouter()

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = asyncio.Lock()   # one music job on the GPU at a time

# Relative share of a job's time, for the overall bar. Sampling dominates.
_WEIGHTS = {"gpu": 4, "load": 8, "sample": 70, "decode": 6, "master": 3, "offload": 3}


def _stage_plan(variations: int) -> List[Dict[str, Any]]:
    def stage(key, label, group=""):
        return {"key": key, "label": label, "group": group, "state": "pending",
                "progress": None, "started": None, "ended": None, "seconds": None}
    plan = [stage("gpu", "Free the GPU")]
    for n in range(1, variations + 1):
        group = f"Variation {n}" if variations > 1 else "Track"
        plan += [stage(f"v{n}:{k}", label, group) for k, label in music_gen.VARIATION_STAGES]
    plan.append(stage("offload", "Offload models"))
    return plan


def _close(stage: Dict[str, Any], state: str, now: float) -> None:
    stage["state"] = state
    if stage["started"] is not None:
        stage["ended"] = now
        stage["seconds"] = round(now - stage["started"], 1)
        if state == "done":
            stage["progress"] = 1.0


def _advance(job: Dict[str, Any], key: str, fraction: Optional[float]) -> None:
    """`key` is running now: earlier running stages finish, unreached ones skip."""
    stages = job["stages"]
    index = next((i for i, st in enumerate(stages) if st["key"] == key), None)
    if index is None:
        return
    now = time.time()
    for st in stages[:index]:
        if st["state"] == "running":
            _close(st, "done", now)
        elif st["state"] == "pending":
            st["state"] = "skipped"
    st = stages[index]
    if st["state"] == "pending":
        st.update(state="running", started=now)
    if fraction is not None:
        st["progress"] = max(0.0, min(1.0, fraction))


def _finish(job: Dict[str, Any], ok: bool) -> None:
    now = time.time()
    for st in job["stages"]:
        if st["state"] == "running":
            _close(st, "done" if ok else "failed", now)
        elif st["state"] == "pending":
            st["state"] = "skipped"
    job["ended_at"] = now


def _view(job: Dict[str, Any]) -> Dict[str, Any]:
    """The job with live durations and the weighted overall progress."""
    out = copy.deepcopy(job)
    now = time.time()
    total = done = 0.0
    for st in out["stages"]:
        weight = _WEIGHTS.get(st["key"].split(":")[-1], 1)
        if st["state"] == "skipped":
            continue
        total += weight
        if st["state"] in ("done", "failed"):
            done += weight
        elif st["state"] == "running":
            st["seconds"] = round(now - st["started"], 1)
            done += weight * (st["progress"] or 0.0)
    out["progress"] = 1.0 if out["state"] == "done" else (round(done / total, 3) if total else 0.0)
    out["elapsed"] = round((out.get("ended_at") or now) - out["started_at"], 1)
    return out


class GenerateRequest(BaseModel):
    style: Optional[str] = None
    engine: str = music_gen.DEFAULT_ENGINE
    music_genres: List[str] = []
    instruments: List[str] = []
    moods: List[str] = []
    bpm: Optional[int] = None
    variations: int = 2
    seconds: float = music_gen.DEFAULT_SECONDS
    extra_tags: str = ""
    genre: Optional[str] = None


@router.get("/styles")
def styles():
    return {
        "styles": [{"name": k, "label": v["label"], "tags": v["tags"],
                    "genres": v["genres"], "moods": v["moods"]}
                   for k, v in music_gen.MUSIC_STYLES.items()],
        "model_ready": workflows.resolve("music") is not None,
        "max_seconds": music_gen.MAX_SECONDS,
        "max_variations": music_gen.MAX_VARIATIONS,
        "default_engine": music_gen.DEFAULT_ENGINE,
        "engines": [{"name": k, "label": v["label"], "model": v["model"],
                     "license": v["model_license"], "note": v["note"],
                     "max_seconds": v["max_seconds"],
                     "ready": workflows.resolve(v["role"]) is not None}
                    for k, v in music_gen.ENGINES.items()],
        "music_genres": music_gen.MUSIC_GENRES,
        "instruments": music_gen.INSTRUMENTS,
        "moods": music_gen.MOODS,
        "bpm_range": [music_gen.MIN_BPM, music_gen.MAX_BPM],
        # The video genres a track can be filed for (the edit's library folders).
        "video_genres": sorted(music_gen.DEFAULT_STYLE_FOR_GENRE),
    }


@router.get("/library")
def library():
    return {"tracks": music_gen.library_tracks(),
            "allowed_licenses": sorted(music_gen.ALLOWED_LICENSES)}


async def _run(job_id: str, req: GenerateRequest) -> None:
    job = _jobs[job_id]
    async with _lock:
        job["state"] = "running"
        try:
            result = await music_gen.generate_variations(
                req.style, count=req.variations, seconds=req.seconds,
                extra_tags=req.extra_tags, genre=req.genre, engine=req.engine,
                music_genres=req.music_genres, instruments=req.instruments,
                moods=req.moods, bpm=req.bpm,
                on_stage=lambda key, fraction=None: _advance(job, key, fraction))
            ok = bool(result["made"])
            _finish(job, ok)
            job.update(state="done" if ok else "failed", made=result["made"], errors=result["errors"])
        except Exception as e:
            logger.exception("Music job %s failed", job_id)
            _finish(job, False)
            job.update(state="failed", errors=[str(e)[:300]])


@router.post("/generate")
async def generate(req: GenerateRequest):
    if req.style and req.style not in music_gen.MUSIC_STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown style {req.style!r}")
    engine = music_gen.ENGINES.get(req.engine)
    if engine is None:
        raise HTTPException(status_code=400, detail=f"Unknown engine {req.engine!r}")
    if not (req.style or req.music_genres or req.instruments or req.moods or req.extra_tags.strip()):
        raise HTTPException(status_code=400,
                            detail="Pick a style, or at least one genre, instrument or mood.")
    if workflows.resolve(engine["role"]) is None:
        raise HTTPException(status_code=503,
                            detail=f"No workflow configured for {engine['label']}.")
    job_id = uuid.uuid4().hex[:12]
    variations = max(1, min(music_gen.MAX_VARIATIONS, int(req.variations)))
    prompt = music_gen.compose_prompt(req.style, req.music_genres, req.instruments,
                                      req.moods, req.bpm, req.extra_tags)
    _jobs[job_id] = {"job_id": job_id, "state": "queued", "style": req.style or "custom",
                     "engine": req.engine, "variations": variations, "tags": prompt["tags"],
                     "made": [], "errors": [], "started_at": time.time(), "ended_at": None,
                     "stages": _stage_plan(variations)}
    asyncio.create_task(_run(job_id, req))
    return _view(_jobs[job_id])


@router.post("/preview")
def preview(req: GenerateRequest):
    if req.style and req.style not in music_gen.MUSIC_STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown style {req.style!r}")
    return music_gen.compose_prompt(req.style, req.music_genres, req.instruments,
                                    req.moods, req.bpm, req.extra_tags)


@router.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _view(_jobs[job_id])


@router.get("/file/{path:path}")
def file(path: str):
    target = (MUSIC_DIR / path).resolve()
    root = MUSIC_DIR.resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))


@router.delete("/file/{path:path}")
def delete_file(path: str):
    if not music_gen.delete_track(path):
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": path}


MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@router.post("/upload")
async def upload(file: UploadFile = File(...), license: str = Form(...), label: str = Form("")):
    import os
    import shutil
    import tempfile
    from pathlib import Path
    name = Path(file.filename or "track").name
    suffix = Path(name).suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(status_code=400,
                            detail=f"Not an audio file ({suffix or 'no extension'}); use one of "
                                   f"{', '.join(sorted(AUDIO_EXTS))}.")
    if license.strip().lower() not in music_gen.ALLOWED_LICENSES:
        raise HTTPException(status_code=400, detail=f"Licence {license!r} is not accepted.")
    handle, temp = tempfile.mkstemp(suffix=suffix, prefix="music_upload_")
    try:
        with os.fdopen(handle, "wb") as out:
            await asyncio.to_thread(shutil.copyfileobj, file.file, out, 1024 * 1024)
        if os.path.getsize(temp) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="That file is over 500 MB.")
        try:
            result = await asyncio.to_thread(music_gen.import_track, Path(temp), name, license, label)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.remove(temp)
        except OSError:
            pass
    return result
