"""The creator's sound-effects library (CLIPS_AND_SOUND_PLAN.md, brief S1):
read and change the library folders, re-index, resolve an effect against the
library, and generate one that is missing into the library.

GET  /api/sfx/library            folders, counts, top tags -- sfx_library.summary()
PUT  /api/sfx/library            {dirs: [...]}: replace the configured library folders
POST /api/sfx/library/refresh    re-index (probe every file again)
GET  /api/sfx/resolve            {text, kind} -> the library's best match; never generates
POST /api/sfx/generate           make a missing effect with Stable Audio and save it
GET  /api/sfx/jobs/{job_id}      progress/result of a generate job
GET  /api/sfx/file               stream one library or generated file

A generation job runs as a background task, the GPU-policy way: `prepare_for_phase`
hands the card to Stable Audio first, `offload_all` releases everything when the
job ends, win or lose.
"""

import asyncio
import logging
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import DATA_DIR
from presentation import director, sfx_library, workflows
from runtime import gpu_handover
from store.app_settings import app_settings

logger = logging.getLogger("routes.sfx")
router = APIRouter()

_jobs: Dict[str, Dict[str, Any]] = {}
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MIN_GEN_S = 0.5
_MAX_GEN_S = 47.0


class LibraryDirsRequest(BaseModel):
    dirs: List[str] = []


class GenerateRequest(BaseModel):
    text: str
    kind: str = "accent"
    seconds: Optional[float] = None
    seed: Optional[int] = None


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    slug = slug[:48].strip("-")
    return slug or "effect"


# --- the library folders -------------------------------------------------------

@router.get("/library")
async def get_library():
    return await asyncio.to_thread(sfx_library.summary)


@router.put("/library")
async def put_library(body: LibraryDirsRequest):
    dirs = [d.strip() for d in body.dirs if d and d.strip()]
    bad = [d for d in dirs if not Path(d).is_dir()]
    if bad:
        raise HTTPException(status_code=422, detail=f"Not a directory: {', '.join(bad)}")
    app_settings.set("sfx_library_dirs", dirs)
    return await asyncio.to_thread(sfx_library.summary, refresh=True)


@router.post("/library/refresh")
async def refresh_library():
    return await asyncio.to_thread(sfx_library.summary, refresh=True)


# --- resolving an effect --------------------------------------------------------

@router.get("/resolve")
async def resolve(text: str, kind: str = "accent"):
    kind = "bed" if kind == "bed" else "accent"
    path, score, source = await asyncio.to_thread(
        sfx_library.resolve_effect, text, random.Random(0), kind)
    duration_s = None
    if path is not None:
        duration_s = await asyncio.to_thread(sfx_library._probe, str(path))
    return {"text": text, "kind": kind, "path": str(path) if path is not None else None,
            "score": score, "source": source, "duration_s": duration_s}


# --- generating a missing effect ------------------------------------------------

async def _run_generate(job_id: str, text: str, kind: str, seconds: float, seed: int) -> None:
    job = _jobs[job_id]
    job["state"] = "running"
    try:
        job["stage"] = "gpu"
        prep = await gpu_handover.prepare_for_phase("sfx", need_vram_mb=4000, need_ram_mb=4000)
        if not prep.get("ready"):
            job["state"] = "failed"
            job["error"] = prep.get("reason") or "GPU not ready"
            return
        job["stage"] = "generate"
        made = await sfx_library.generated_effect(text, seconds, seed)
        if made is None:
            job["state"] = "failed"
            job["error"] = "nothing usable was generated (noise or ComfyUI error)"
            return
        job["stage"] = "save"
        tags = sorted(sfx_library.canonical_tags(sfx_library.words(text)))
        tag = tags[0] if tags else "misc"
        slug = _slugify(text)
        destination = sfx_library.LIBRARY_DIR / "generated" / tag / f"{slug}-{seed}.wav"
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, str(made), str(destination))
        entries = await asyncio.to_thread(sfx_library.index, None, True)
        duration_s = next((e.duration_s for e in entries if e.path == str(destination)), None)
        job["path"] = str(destination)
        job["duration_s"] = duration_s
        job["state"] = "done"
    except Exception as e:
        logger.exception("SFX generate job %s failed", job_id)
        job["state"] = "failed"
        job["error"] = str(e)[:300]
    finally:
        job["stage"] = "offload"
        try:
            await gpu_handover.offload_all("sfx generation")
        except Exception as e:
            logger.debug("SFX offload skipped: %s", e)
        job["stage"] = "done"
        job["ended_at"] = time.time()


@router.post("/generate")
async def generate(body: GenerateRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    kind = "bed" if body.kind == "bed" else "accent"
    if not director._comfyui_online() or workflows.resolve("audio") is None:
        raise HTTPException(status_code=503,
                            detail="ComfyUI is offline or no audio workflow is configured.")
    if any(j["state"] in ("queued", "running") for j in _jobs.values()):
        raise HTTPException(status_code=409, detail="A generate job is already running.")
    seconds = body.seconds if body.seconds is not None else sfx_library._gen_seconds(kind)
    seconds = max(_MIN_GEN_S, min(_MAX_GEN_S, float(seconds)))
    seed = body.seed if body.seed is not None else 0
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "state": "queued", "stage": "gpu", "text": text,
                     "kind": kind, "seconds": seconds, "path": None, "duration_s": None,
                     "error": None, "started_at": time.time(), "ended_at": None}
    asyncio.create_task(_run_generate(job_id, text, kind, seconds, seed))
    return dict(_jobs[job_id])


@router.get("/jobs/{job_id}")
async def job(job_id: str):
    found = _jobs.get(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return dict(found)


# --- serving a file --------------------------------------------------------------

@router.get("/file")
async def file(path: str):
    target = Path(path)
    allowed_roots = list(sfx_library.library_roots()) + [Path(DATA_DIR) / "audio_gen"]
    norm = sfx_library._norm(target)
    within = any(norm == sfx_library._norm(root)
                or norm.startswith(sfx_library._norm(root).rstrip(os.sep) + os.sep)
                for root in allowed_roots)
    if not within:
        raise HTTPException(status_code=403, detail="Not within an allowed library folder")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))
