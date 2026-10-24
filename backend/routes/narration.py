"""Narration: local text-to-speech in a voice (AUTO_STORY_PLAN.md brief N1, §3.1; audio/narration.py).
Voice delivery -- mood, spacing, pace, inline cues -- is VOICE_DELIVERY_PLAN.md.

POST /api/narration/synthesize   {text, language: "hi"|"en", voice: "sudhansu"|"<preset>"|"engine",
                                  engine?, style: {mood, exaggeration, cfg_weight, temperature,
                                  pace, word_gap_ms, phrase_pause_s, sentence_pause_s}, convert_to?,
                                  out_dir?}
                                  -> {job_id, state: running|done|failed, message, progress, result, error}
GET  /api/narration/jobs/{id}    the job; result {wav, seconds, duration_s, engine, converted, ...}
GET  /api/narration/busy         {busy, job_id}: whether a narration is running right now
                                  (STUDIO_DIRECTOR_PLAN.md enhancement 2 -- the Studio's idle-
                                  offload and code-reload holds; same shape as GET /api/reels/busy)
GET  /api/narration/voices       {"voices": [{id, kind: reference|kokoro, path?, language?}]}
GET  /api/narration/moods        {"moods": [{id, label, description, style: {...}}]}

400 for a bad style (an unknown mood, or a key outside its range), 409 while a render runs (the
GPU rule), 503 when the TTS engine is not installed.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from audio import narration

router = APIRouter()


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "hi"
    voice: str = "sudhansu"
    engine: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    convert_to: Optional[str] = None
    out_dir: Optional[str] = None


@router.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        return narration.start_job(req.text, language=req.language, voice=req.voice, engine=req.engine,
                                   style=req.style, convert_to=req.convert_to, out_dir=req.out_dir)
    except narration.NarrationBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except narration.NarrationUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}")
def job(job_id: str):
    found = narration.get_job(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown narration job")
    return found


@router.get("/busy")
def busy():
    running = [j["job_id"] for j in narration._jobs.values() if j.get("state") == "running"]
    return {"busy": narration.busy(), "job_id": running[0] if running else None}


@router.get("/voices")
def voices():
    return {"voices": narration.list_voices()}


@router.get("/moods")
def moods():
    return {"moods": narration.list_moods()}
