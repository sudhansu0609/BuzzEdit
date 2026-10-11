"""Voice conversion routes (VOICE_CHANGER_PLAN.md, briefs S1 and S2a, §2.7), on the `/api/audio`
prefix beside the audio suite's.

    GET    /voices                         the voices library
    POST   /voices                         multipart: file, name, kind, consent (3-60 s of audio)
    PUT    /voices/{name}                  {name}: rename
    DELETE /voices/{name}
    GET    /voice/presets                  the five presets: voice, engine, FX rack, parameter values
    GET    /{project}/speaker_spans        who speaks when, from the script and the kept words
    GET    /{project}/voice_map            the project setting {speaker: {voice, preset?, ...}} and
                                           `applied` {map_hash, version_path, applied_at} | null
    PUT    /{project}/voice_map            {voice_map}
    POST   /{project}/voice/apply_map      -> {job_id}: every mapped speaker's spans, one version;
                                           idempotent (result skipped / replaced / removed / warning)
    POST   /{project}/voice/convert        {source_id?, start_s, end_s, program_range?, voice?, preset?,
                                           params} -> {job_id}; program_range: programme seconds,
                                           mapped to source pieces (a cut = several, one version)
    GET    /{project}/voice/regions        {regions}
    PUT    /{project}/voice/regions        {regions}: replace (checked, overlaps trimmed) -> {regions}
    POST   /{project}/voice/regions/apply  {ids?} -> {job_id}; result {applied, failed, version}
    POST   /{project}/voice/regions/undo   undo the last apply -> {regions}
    POST   /{project}/voice/convert_file   {media_id? | path?, voice, preset?, params, name?} -> {job_id};
                                           result {media_id, path, seconds}
    POST   /{project}/voice/batch          {media_ids, voice, preset?, params} -> {job_id};
                                           result {items: [{media_id, out_media_id, path} | {media_id, error}]}
    GET    /voice/worker                   {running, device, idle_s}
    POST   /voice/worker/stop

A request's `voice` + `preset` + `params` (regions, convert_file, batch, and /voice/convert when it
names a voice or preset beside its params) = the preset's values, then `params`, then that voice.

Jobs run in the process lane's registry: poll `GET /api/audio/{project}/process/{job_id}`.
Every route answers 409 while a render job is running.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from audio import analysis, noise, processes, seedvc_env, voice_conversion
from audio.analysis import AudioToolError
from audio.noise import Conflict, NotFound


def _no_render_running() -> None:
    """409 while `routes/rendering.py` has a render job running."""
    try:
        from routes import rendering
    except Exception:
        return
    for job in list(rendering.jobs.values()):
        if isinstance(job, dict) and job.get("job_type") == "render" and job.get("status") == "running":
            raise HTTPException(status_code=409, detail="A render is running; voice conversion waits for it")


router = APIRouter(dependencies=[Depends(_no_render_running)])


class RenameRequest(BaseModel):
    name: str


class VoiceMapRequest(BaseModel):
    voice_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ConvertRequest(BaseModel):
    source_id: Optional[str] = None
    start_s: float
    end_s: float
    program_range: bool = False     # start_s/end_s are programme seconds (mapped to source pieces)
    voice: Optional[str] = None
    preset: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class RegionsRequest(BaseModel):
    regions: List[Dict[str, Any]]


class ApplyRegionsRequest(BaseModel):
    ids: Optional[List[str]] = None


class ConvertFileRequest(BaseModel):
    media_id: Optional[str] = None
    path: Optional[str] = None
    voice: Optional[str] = None
    preset: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None


class BatchRequest(BaseModel):
    media_ids: List[str]
    voice: Optional[str] = None
    preset: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (Conflict, seedvc_env.SeedVCBusy)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, seedvc_env.SeedVCUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (ValueError, AudioToolError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Run blocking work off the event loop; domain errors become HTTP codes."""
    try:
        return analysis.json_safe(await asyncio.to_thread(fn, *args))
    except (NotFound, Conflict, seedvc_env.SeedVCError, ValueError, AudioToolError) as exc:
        raise _http(exc)


def _engine_ready() -> None:
    env = seedvc_env.resolve()
    if env["source"] == "unavailable":
        raise seedvc_env.SeedVCUnavailable(f"seed-vc engine unavailable: {env.get('reason') or 'not found'}")


def _worker_view(status: Dict[str, Any]) -> Dict[str, Any]:
    return {"running": status.get("running"), "device": status.get("device"), "idle_s": status.get("idle_s")}


# -------------------------------------------------------------------- voices

@router.get("/voices")
async def list_voices():
    return await _call(lambda: {"voices": voice_conversion.list_voices()})


@router.post("/voices")
async def add_voice(file: UploadFile = File(...), name: str = Form(...), kind: str = Form("custom"),
                    consent: str = Form("")):
    filename = Path(file.filename or "upload").name
    staging = Path(tempfile.mkdtemp(prefix="voice_upload_"))
    try:
        staged = staging / (re.sub(r"[^A-Za-z0-9_.-]", "_", filename) or "upload")
        with staged.open("wb") as handle:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        voice = await _call(voice_conversion.add_voice, name, staged, kind, consent, filename)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"voice": voice}


@router.put("/voices/{name}")
async def rename_voice(name: str, body: RenameRequest):
    return {"voice": await _call(voice_conversion.rename_voice, name, body.name)}


@router.delete("/voices/{name}")
async def delete_voice(name: str):
    return await _call(voice_conversion.delete_voice, name)


@router.get("/voice/presets")
async def voice_presets():
    return await _call(voice_conversion.presets)


# -------------------------------------------------------------------- worker

@router.get("/voice/worker")
async def worker_status():
    return _worker_view(seedvc_env.WORKER.status())


@router.post("/voice/worker/stop")
async def worker_stop():
    return _worker_view(await _call(seedvc_env.WORKER.stop))


# ------------------------------------------------------------------- project

@router.get("/{project_id}/speaker_spans")
async def speaker_spans(project_id: str):
    return await _call(voice_conversion.project_speaker_spans, project_id)


@router.get("/{project_id}/voice_map")
async def get_voice_map(project_id: str):
    """The voice map and `applied`: what its last apply recorded, without the span lists (None
    when nothing is applied)."""
    def read() -> Dict[str, Any]:
        return {**voice_conversion.get_voice_map(project_id),
                "applied": voice_conversion.get_applied(project_id)}

    return await _call(read)


@router.put("/{project_id}/voice_map")
async def put_voice_map(project_id: str, body: VoiceMapRequest):
    return await _call(voice_conversion.set_voice_map, project_id, body.voice_map)


@router.post("/{project_id}/voice/apply_map")
async def apply_voice_map(project_id: str):
    """Queue `apply_voice_map`; progress 0..1 across the spans. Idempotent: the job result carries
    `skipped`, `replaced`, `removed` or `warning` (an emptied map takes the recorded apply back)."""
    def check() -> None:
        _engine_ready()
        processes.check_available(voice_conversion.EFFECT)
        voice_conversion.preflight_voice_map(project_id)

    await _call(check)
    return {"job_id": voice_conversion.start_apply_job(project_id)}


@router.post("/{project_id}/voice/convert")
async def convert_range(project_id: str, body: ConvertRequest):
    """Queue one range in a voice as a process-lane job (a new version, Undo pops it). With
    `program_range`, start_s/end_s are programme seconds: one source piece runs as that source
    range, several (a range across a cut) run as `ranges` into one version."""
    prepared: Dict[str, Any] = {"pieces": []}

    def check() -> None:
        if body.end_s <= body.start_s:
            raise ValueError(f"Empty range: {body.start_s}..{body.end_s} s")
        params = voice_conversion.request_params(body.params, body.voice, body.preset)
        values = processes.resolve_params(voice_conversion.EFFECT, params)
        processes.check_available(voice_conversion.EFFECT)
        voice, _params = voice_conversion.engine_params(values)
        voice_conversion.voice_path(voice)
        _pid, p_data, tl = noise.load_project(project_id)
        if body.program_range:
            pieces = voice_conversion.program_pieces(tl, body.start_s, body.end_s)
            noise.source_audio(tl.sources[voice_conversion.one_source(pieces)])
            prepared["pieces"] = pieces
        else:
            _sid, source = noise.resolve_source(tl, p_data, body.source_id)
            noise.source_audio(source)
        prepared["params"] = params
        _engine_ready()

    await _call(check)
    params, pieces = prepared["params"], prepared["pieces"]
    if not body.program_range:
        job_id = processes.start_job(project_id, body.source_id, voice_conversion.EFFECT, params,
                                     body.start_s, body.end_s)
    elif len(pieces) == 1:
        job_id = processes.start_job(project_id, pieces[0][0], voice_conversion.EFFECT, params,
                                     pieces[0][2], pieces[0][3])
    else:
        job_id = processes.start_job(project_id, pieces[0][0], voice_conversion.EFFECT, params,
                                     ranges=[(a, b) for _sid, _path, a, b in pieces])
    return {"job_id": job_id}


# ------------------------------------------------------------- voice regions

@router.get("/{project_id}/voice/regions")
async def get_regions(project_id: str):
    return await _call(voice_conversion.get_regions, project_id)


@router.put("/{project_id}/voice/regions")
async def put_regions(project_id: str, body: RegionsRequest):
    """Replace the regions (programme seconds): checked, clamped to the programme, overlaps
    trimmed like SkinWalker's manifest; revision + 1."""
    return await _call(voice_conversion.set_regions, project_id, body.regions)


@router.post("/{project_id}/voice/regions/apply")
async def apply_regions(project_id: str, body: Optional[ApplyRegionsRequest] = None):
    """Queue the pending regions (or the listed ones) as one job and one version."""
    ids = body.ids if body is not None else None

    def check() -> None:
        _engine_ready()
        processes.check_available(voice_conversion.EFFECT)
        voice_conversion.preflight_regions(project_id, ids)

    await _call(check)
    return {"job_id": voice_conversion.start_regions_job(project_id, ids)}


@router.post("/{project_id}/voice/regions/undo")
async def undo_regions(project_id: str):
    return await _call(voice_conversion.undo_regions, project_id)


# --------------------------------------------------------- whole files, batch

@router.post("/{project_id}/voice/convert_file")
async def convert_file(project_id: str, body: ConvertFileRequest):
    """Queue a whole media-pool item (or a file, or the project source) in a voice; the result is
    a new media-pool item."""
    def check() -> None:
        processes.check_available(voice_conversion.EFFECT)
        voice_conversion.resolved_values(voice_conversion.request_entry(body.voice, body.preset, body.params))
        voice_conversion.convert_input(project_id, body.media_id, body.path)
        _engine_ready()

    await _call(check)
    return {"job_id": voice_conversion.start_convert_file_job(project_id, body.media_id, body.path, body.voice,
                                                              body.preset, body.params, body.name)}


@router.post("/{project_id}/voice/batch")
async def batch(project_id: str, body: BatchRequest):
    """Queue several media-pool items in one voice; progress and the result are per item."""
    def check() -> None:
        processes.check_available(voice_conversion.EFFECT)
        voice_conversion.check_media_ids(body.media_ids)
        voice_conversion.resolved_values(voice_conversion.request_entry(body.voice, body.preset, body.params))
        noise.load_project(project_id)
        _engine_ready()

    await _call(check)
    return {"job_id": voice_conversion.start_batch_job(project_id, body.media_ids, body.voice, body.preset,
                                                       body.params)}
