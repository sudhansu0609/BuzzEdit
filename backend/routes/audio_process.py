"""Analysis, noise print and process-lane routes of the audio suite
(AUDIO_SUITE_PLAN.md §2.3 and §2.1, brief A2).

On the `/api/audio` prefix beside `routes/audio.py` (catalogue, presets,
racks, rack preview). `source_id` defaults to the project's programme source.
With `program_range: true`, analyze / process / process preview take start_s/end_s in
PROGRAMME time (the cut A1 concat, timeline/program_time.py) and map it to source pieces;
noise detect and the noise print stay source-based (pauses are cut out of the programme).
Every route answers 409 while a render job is running: processing reads and
writes the same audio a render is reading, on the same CPU and disk.
"""

import asyncio
import inspect
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from audio import analysis, deepfilter, noise, processes
from audio.analysis import AudioToolError
from audio.noise import Conflict, NotFound
from config import FFMPEG_BIN
from timeline.program_time import Piece, a1_segments, pieces_of
from timeline.schema import Timeline
from utils.proc import NO_WINDOW


def _no_render_running() -> None:
    """409 while `routes/rendering.py` has a render job running."""
    try:
        from routes import rendering
    except Exception:
        return
    for job in list(rendering.jobs.values()):
        if isinstance(job, dict) and job.get("job_type") == "render" and job.get("status") == "running":
            raise HTTPException(status_code=409,
                                detail="A render is running; audio analysis and processing wait for it")


router = APIRouter(dependencies=[Depends(_no_render_running)])


class RangeRequest(BaseModel):
    source_id: Optional[str] = None
    start_s: float
    end_s: float
    program_range: bool = False     # analyze only: start_s/end_s are programme time


class SourceRequest(BaseModel):
    source_id: Optional[str] = None


class ProcessRequest(BaseModel):
    source_id: Optional[str] = None
    effect: str
    params: Dict[str, Any] = Field(default_factory=dict)
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    program_range: bool = False


class ProcessPreviewRequest(BaseModel):
    source_id: Optional[str] = None
    effect: str
    params: Dict[str, Any] = Field(default_factory=dict)
    start_s: float
    seconds: float
    program_range: bool = False


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, Conflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, deepfilter.DeepFilterUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Run blocking work off the event loop; domain errors become HTTP codes."""
    try:
        return analysis.json_safe(await asyncio.to_thread(fn, *args))
    except (NotFound, Conflict, deepfilter.DeepFilterUnavailable, ValueError, AudioToolError) as exc:
        raise _http(exc)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text) or "source"


# ------------------------------------------------------------ programme time

SPANS_A_CUT = "range spans a cut — one segment at a time"


def _program_pieces(tl: Timeline, start_s: float, end_s: float) -> List[Piece]:
    """Source pieces of programme [start_s, end_s), split at every cut."""
    if end_s <= start_s:
        raise ValueError(f"Empty range: {start_s}..{end_s} s")
    segments = a1_segments(tl)
    if not segments:
        raise NotFound("The timeline has no A1 clips, so there is no programme to map")
    pieces = pieces_of(segments, start_s, end_s)
    if not pieces:
        raise ValueError(f"Programme {start_s}..{end_s} s holds no A1 audio")
    return pieces


def _one_source(pieces: List[Piece]) -> str:
    sources = {sid for sid, _path, _a, _b in pieces}
    if len(sources) > 1:
        raise Conflict("The range spans clips of more than one source — one source at a time")
    return pieces[0][0]


def _takes_ranges(fn: Callable[..., Any]) -> bool:
    try:
        return "ranges" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _render_pieces(pieces: List[Piece], out_wav: str) -> None:
    """The programme pieces back to back, as the render's A1 concat joins them."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"]
    for _sid, path, a, b in pieces:
        cmd += ["-ss", f"{a:.6f}", "-t", f"{b - a:.6f}", "-i", path]
    graph = "".join(f"[{k}:a]" for k in range(len(pieces))) + f"concat=n={len(pieces)}:v=0:a=1[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]", "-ar", "48000", "-c:a", "pcm_s16le", out_wav]
    done = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        raise AudioToolError(f"programme range render failed: {tail[0]}")


def _analyze_program(project_id: str, tl: Timeline, start_s: float, end_s: float) -> Dict[str, Any]:
    pieces = _program_pieces(tl, start_s, end_s)
    work_dir = noise.audio_dir(project_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="program_range_", suffix=".wav", dir=str(work_dir))
    os.close(fd)
    try:
        _render_pieces(pieces, tmp)
        seconds = sum(b - a for _sid, _path, a, b in pieces)
        result = analysis.analyze(tmp, 0.0, seconds, png_path=work_dir / "spectrum_program.png")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    result.update(path=None, program_range=True, start_s=start_s, end_s=end_s,
                  source_id=pieces[0][0],
                  pieces=[{"source_id": sid, "start_s": a, "end_s": b} for sid, _p, a, b in pieces])
    return result


# ------------------------------------------------------------------ analysis

@router.get("/processes")
async def list_processes():
    """Every process effect except those whose def says `available: false`."""
    return processes.listed_processes()


@router.post("/{project_id}/analyze")
async def analyze_range(project_id: str, body: RangeRequest):
    def work() -> Dict[str, Any]:
        _pid, p_data, tl = noise.load_project(project_id)
        if body.program_range:
            return _analyze_program(project_id, tl, body.start_s, body.end_s)
        sid, source = noise.resolve_source(tl, p_data, body.source_id)
        path = noise.source_audio(source)
        png = noise.audio_dir(project_id) / f"spectrum_{_safe_name(sid)}.png"
        result = analysis.analyze(path, body.start_s, body.end_s, png_path=png)
        result["source_id"] = sid
        return result

    return await _call(work)


# --------------------------------------------------------------------- noise

@router.post("/{project_id}/noise/detect")
async def detect_noise(project_id: str, body: Optional[SourceRequest] = None):
    return await _call(noise.detect_noise, project_id, body.source_id if body else None)


@router.post("/{project_id}/noise/print")
async def capture_print(project_id: str, body: RangeRequest):
    return await _call(noise.capture_print, project_id, body.source_id, body.start_s, body.end_s)


@router.get("/{project_id}/noise/print")
async def get_print(project_id: str):
    def work() -> Dict[str, Any]:
        noise.load_project(project_id)
        stored = noise.load_print(project_id)
        if stored is None:
            raise NotFound("No noise print captured for this project")
        return stored

    return await _call(work)


# ------------------------------------------------------------------- process

@router.post("/{project_id}/process")
async def start_process(project_id: str, body: ProcessRequest):
    """Queue a process effect as a job; poll `GET /{project}/process/{job_id}`.

    With `program_range`, the programme range becomes source pieces: one piece is processed
    as that source range; several need `ranges` on processes.start_job, else 409."""
    pieces: List[Piece] = []

    def check() -> None:
        processes.resolve_params(body.effect, body.params)
        processes.check_available(body.effect)          # 409 with the reason
        _pid, p_data, tl = noise.load_project(project_id)
        if body.program_range:
            if body.start_s is None or body.end_s is None:
                raise ValueError("program_range needs start_s and end_s")
            pieces.extend(_program_pieces(tl, body.start_s, body.end_s))
            noise.source_audio(tl.sources[_one_source(pieces)])
            if len(pieces) > 1 and not _takes_ranges(processes.start_job):
                raise Conflict(SPANS_A_CUT)
        else:
            _sid, source = noise.resolve_source(tl, p_data, body.source_id)
            noise.source_audio(source)
        if body.start_s is not None and body.end_s is not None and body.end_s <= body.start_s:
            raise ValueError(f"Empty range: {body.start_s}..{body.end_s} s")
        if body.effect == "noise_reduction" and noise.load_print(project_id) is None:
            raise Conflict("Capture a noise print first (Detect noise, then Capture Noise Print)")

    await _call(check)
    if not body.program_range:
        job_id = processes.start_job(project_id, body.source_id, body.effect, body.params,
                                     body.start_s, body.end_s)
        return {"job_id": job_id}
    sid = pieces[0][0]
    if len(pieces) == 1:
        job_id = processes.start_job(project_id, sid, body.effect, body.params,
                                     pieces[0][2], pieces[0][3])
    else:
        job_id = processes.start_job(project_id, sid, body.effect, body.params,
                                     ranges=[(a, b) for _s, _p, a, b in pieces])
    return {"job_id": job_id, "source_id": sid,
            "pieces": [{"start_s": a, "end_s": b} for _s, _p, a, b in pieces]}


@router.post("/{project_id}/process/preview")
async def preview_process(project_id: str, body: ProcessPreviewRequest):
    if not body.program_range:
        return await _call(processes.preview_process, project_id, body.source_id, body.effect,
                           body.params, body.start_s, body.seconds)

    def work() -> Dict[str, Any]:
        _pid, _p_data, tl = noise.load_project(project_id)
        pieces = _program_pieces(tl, body.start_s, body.start_s + body.seconds)
        sid = _one_source(pieces)
        if len(pieces) == 1:
            _s, _p, a, b = pieces[0]
            return processes.preview_process(project_id, sid, body.effect, body.params, a, b - a)
        if not _takes_ranges(processes.preview_process):
            raise Conflict(SPANS_A_CUT)
        return processes.preview_process(project_id, sid, body.effect, body.params,
                                         pieces[0][2], sum(b - a for _s, _p, a, b in pieces),
                                         ranges=[(a, b) for _s, _p, a, b in pieces])

    return await _call(work)


@router.get("/{project_id}/process/{job_id}")
async def process_status(project_id: str, job_id: str):
    job = processes.get_job(job_id)
    if job is None or job.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail=f"No audio process job {job_id!r}")
    return analysis.json_safe(job)


@router.post("/{project_id}/undo")
async def undo_process(project_id: str, body: Optional[SourceRequest] = None):
    return await _call(processes.undo, project_id, body.source_id if body else None)


@router.get("/{project_id}/versions")
async def list_versions(project_id: str, source_id: Optional[str] = None):
    return await _call(processes.versions, project_id, source_id)
