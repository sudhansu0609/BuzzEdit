import asyncio
import logging
import uuid
from fastapi import APIRouter, HTTPException
from models import RenderJob, Project
from config import PROJECTS_DIR, OUTPUT_DIR
from store.project_store import ProjectStore
from timeline import Timeline, build_timeline_from_transcript
from timeline.ops import apply_auto_zoom
from timeline.schema import Transition, WordItem, frame_to_time, time_to_frame
from render import render_timeline_async
from utils.ffmpeg_utils import get_video_info, extract_audio
from asr import whisper_engine, refine_disfluencies
from asr.auto_edit import (
    apply_report_to_timeline,
    extract_project_audio,
    plan_auto_edit,
    public_report,
    rebuild_and_check,
)

logger = logging.getLogger("rendering")

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}

@router.post("/render")
async def render_video(job: RenderJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video:
        raise HTTPException(status_code=400, detail="No source video uploaded for project")

    job_id = f"render_{job.project_id}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = job.output_path or str(OUTPUT_DIR / f"{job.project_id}_output.mp4")

    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "render",
        "status": "running",
        "progress": 0.0,
    }

    try:
        # Load or generate timeline
        if "timeline" in p_data:
            tl = Timeline.model_validate(p_data["timeline"])
        else:
            audio_path = extract_audio(source_video)
            v_info = get_video_info(source_video)
            settings0 = p_data.get("settings") or {}
            # Pin the language across runs — auto-detect flips code-switched
            # speech into an English paraphrase with useless timings.
            words, detected_language = await whisper_engine.transcribe_words_async(
                audio_path, language=settings0.get("language"))
            settings0["language"] = detected_language
            p_data["settings"] = settings0
            ann_words = await refine_disfluencies(words, aggressiveness=float(settings0.get("fumble_aggressiveness", 0.5)))
            tl = build_timeline_from_transcript(
                source_path=source_video,
                duration_seconds=v_info.get("duration", 0.0),
                transcript_words=ann_words,
                fps_num=v_info.get("fps_num", 30),
                fps_den=v_info.get("fps_den", 1),
                width=v_info.get("width", 1920),
                height=v_info.get("height", 1080),
                has_audio=v_info.get("audio_codec") not in (None, "none"),
            )
            p_data["timeline"] = tl.model_dump()

        def update_progress(pct: float):
            jobs[job_id]["progress"] = pct / 100.0

        # Honor the project's chosen output resolution, if set.
        settings = p_data.get("settings") or {}
        output_resolution = settings.get("resolution")

        result_path = await render_timeline_async(
            tl,
            output_path,
            progress_callback=update_progress,
            output_resolution=output_resolution
        )

        p_data["output_path"] = result_path
        p_data["status"] = "rendered"
        project_store.save_project(job.project_id, p_data)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"output_path": result_path}

        return {
            "status": "completed",
            "job_id": job_id,
            "output_path": result_path,
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def render_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.post("/{project_id}/auto_edit")
async def auto_edit(project_id: str):
    """Start an auto-edit and return a job id to poll.

    Auto-edit runs an LLM fluency pass and a full re-render — tens of seconds to
    minutes. Running it inline meant the request blocked with no feedback and the
    UI sat at 0% looking frozen. It now runs as a background job whose progress
    and phase message the client polls via GET /status/{job_id}, exactly like the
    plain render.
    """
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    job_id = f"auto_edit_{project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": project_id,
        "job_type": "auto_edit",
        "status": "running",
        "progress": 0.0,
        "message": "Starting auto-edit…",
    }
    # Fire-and-forget: the worker owns the job entry from here and records its own
    # failures, so a crash in it never takes down the request that started it.
    asyncio.create_task(_run_auto_edit(project_id, job_id))
    return {"status": "running", "job_id": job_id}


async def _run_auto_edit(project_id: str, job_id: str):
    """The actual auto-edit, updating jobs[job_id] as it moves through phases."""
    def phase(pct: float, msg: str):
        jobs[job_id]["progress"] = max(0.0, min(1.0, pct))
        jobs[job_id]["message"] = msg

    try:
        p_data = project_store.get_project(project_id)
        if not p_data:
            raise RuntimeError("Project not found")

        settings = p_data.get("settings") or {}
        tl_data = p_data.get("timeline")
        tl = Timeline.model_validate(tl_data) if tl_data else None

        if tl is None or not tl.words:
            # Nothing to re-cut yet — transcribe and scaffold a timeline, then
            # fall through into the SAME pipeline a re-cut takes. This used to
            # branch to a text-only path with no VAD, no filler detection and no
            # coverage audit, so a fresh project got a visibly worse first edit
            # than the second run of the same button.
            tl = await _transcribe_and_scaffold(p_data, project_id, phase)

        result = await _plan_cut_and_render(p_data, tl, project_id, phase)
        jobs[job_id].update(status="completed", progress=1.0,
                            message="Auto-edit complete", result=result)
    except Exception as e:
        logger.exception("auto_edit[%s] failed", project_id)
        jobs[job_id].update(status="failed", message=f"Auto-edit failed: {e}", error=str(e))


async def _plan_cut_and_render(p_data: dict, tl: Timeline, project_id: str, phase) -> dict:
    """The one auto-edit pipeline: audio analysis, fluency passes, rebuild,
    coverage audit, render. Every entry point ends up here so a fresh project
    and a re-cut get exactly the same treatment."""
    settings = p_data.get("settings") or {}
    aggressiveness = float(settings.get("fumble_aggressiveness", 0.5))
    tl.max_pause_seconds = float(settings.get("max_pause_seconds", tl.max_pause_seconds))
    tl.pause_padding_seconds = float(settings.get("pause_padding_seconds", tl.pause_padding_seconds))

    # The planner needs the audio, not just the transcript: the filler sounds
    # it cuts were never in the transcript, and the word timings need checking
    # against real speech before anything is cut on them.
    phase(0.18, "Extracting audio…")
    audio_path = await extract_project_audio(p_data.get("source_video"))

    word_dicts = [{
        "word": w.text,
        "start": frame_to_time(w.start_frame, tl.fps_num, tl.fps_den),
        "end": frame_to_time(w.end_frame, tl.fps_num, tl.fps_den),
        "probability": 1.0,
    } for w in tl.words]

    phase(0.25, "Cleaning fumbles with the LLM… (this is the slow part)")
    plan = await plan_auto_edit(word_dicts, audio_path, aggressiveness=aggressiveness)
    report = plan.report
    apply_report_to_timeline(tl, report)

    phase(0.6, "Rebuilding the edit…")
    # Rebuild the word list rather than zipping onto the old one: the planner
    # both repairs timings and inserts the filler sounds it found in the
    # audio, so the two lists no longer line up index for index.
    tl.words = [
        WordItem(
            id=f"w_{i}_{uuid.uuid4().hex[:4]}",
            text=str(rd.get("word", "")).strip() or "…",
            start_frame=time_to_frame(float(rd.get("start", 0.0)), tl.fps_num, tl.fps_den),
            end_frame=max(
                time_to_frame(float(rd.get("start", 0.0)), tl.fps_num, tl.fps_den) + 1,
                time_to_frame(float(rd.get("end", 0.0)), tl.fps_num, tl.fps_den),
            ),
            enabled=not bool(rd.get("disfluency", False)),
            disfluency=bool(rd.get("disfluency", False)),
            reason=rd.get("reason") or None,
            candidate=bool(rd.get("candidate", False)),
        )
        for i, rd in enumerate(plan.words)
    ]
    cut_count = sum(1 for w in tl.words if not w.enabled)
    logger.info(f"auto_edit[{project_id}]: removed {cut_count}/{len(tl.words)} words; "
                f"{plan.public_report}")

    # Soften the joins. An auto-edit of a talking head is one long series of
    # jump cuts — each a visible snap in the speaker's head position. A short
    # dissolve reads as the cut not being jarring. Only applied when the project
    # has no transition preference of its own. The default was 0.12s, which is
    # ~4 frames and barely perceptible on a jump cut; a quarter second is a
    # clearly visible cross-dissolve while still tight enough for talking head.
    if tl.default_transition is None and not settings.get("hard_cuts"):
        tl.default_transition = Transition(
            type="fade", duration=float(settings.get("transition_seconds", 0.25)))

    # Rebuild AI cut (V1/A1) from fresh decisions; manual V2+/A2+ clips are
    # preserved. The rebuild is also where the plan is checked against what
    # will actually be rendered.
    if tl.sources:
        primary_src_id = next(iter(tl.sources.keys()))
        rebuild_and_check(tl, primary_src_id, report)

    # Give the cut some life: a slow Ken Burns push-in / pull-back on each
    # segment. The rebuild above throws V1 items away and rebuilds them flat, so
    # this must run *after* it. On by default; a project can turn it off or tune
    # the depth through its settings.
    if settings.get("auto_zoom", True):
        moved = apply_auto_zoom(tl, depth=float(settings.get("zoom_depth", 0.10)))
        logger.info("auto_edit[%s]: applied Ken Burns zoom to %d segments",
                    project_id, moved)

    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)

    phase(0.7, "Rendering…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"{project_id}_output.mp4")

    def on_render(pct: float):
        phase(0.7 + (pct / 100.0) * 0.3, f"Rendering… {int(pct)}%")

    rendered = await render_timeline_async(
        tl, output_path,
        progress_callback=on_render,
        output_resolution=settings.get("resolution"),
    )

    p_data["output_path"] = rendered
    p_data["status"] = "rendered"
    project_store.save_project(project_id, p_data)

    return {
        "status": "completed",
        "output_path": rendered,
        "words_removed": cut_count,
        "words_total": len(tl.words),
        "report": public_report(report),
    }


async def _transcribe_and_scaffold(p_data: dict, project_id: str, phase) -> Timeline:
    """Fresh-project path: transcribe and build the initial timeline scaffold.

    Deliberately does no editing of its own — the caller hands the scaffold to
    `_plan_cut_and_render`, which owns every cut decision, so a first run and a
    re-run go through identical machinery.
    """
    source_video = p_data.get("source_video")
    if not source_video:
        raise RuntimeError("No source video uploaded for project")

    phase(0.02, "Extracting audio…")
    audio_path = await extract_project_audio(source_video)
    if not audio_path:
        raise RuntimeError("Could not extract audio from the source video")

    v_info = get_video_info(source_video)
    settings = p_data.get("settings") or {}

    # The first transcription of a session pays a one-off model load (the ~3GB
    # large-v3 weights onto the GPU) that dominates "why is it stuck at the
    # start". Load it as its own phase so the UI says what the wait is, instead
    # of a silent gap before "Transcribing".
    if whisper_engine.model is None:
        phase(0.05, "Loading transcription model (first run, ~30s)…")
        await whisper_engine.preload_async()

    phase(0.10, "Transcribing audio…")
    # Reuse the language settled on previously and persist what this run used:
    # auto-detection is unstable on code-switched speech, and a run that lands
    # on "en" paraphrases Hindi instead of transcribing it.
    words, detected_language = await whisper_engine.transcribe_words_async(
        audio_path, language=settings.get("language"))
    settings["language"] = detected_language
    p_data["settings"] = settings

    tl = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=v_info.get("duration", 0.0),
        transcript_words=words,
        fps_num=v_info.get("fps_num", 30),
        fps_den=v_info.get("fps_den", 1),
        width=v_info.get("width", 1920),
        height=v_info.get("height", 1080),
        has_audio=v_info.get("audio_codec") not in (None, "none"),
    )
    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)
    return tl
