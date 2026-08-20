"""The overnight pass, end to end.

The shape of this module is set by one requirement: **it runs while nobody is
watching**. That makes partial failure the normal case to design for, not an
edge case. Every stage is wrapped so that a stage which fails costs only what it
was going to add — the language model being unavailable loses the topic planning
but keeps the zooms and captions; ComfyUI being off loses the pictures but keeps
everything else. Only saving and rendering are allowed to fail the run, because
without them there is nothing to wake up to.

The stages also run strictly one after another. All three heavy tenants —
Whisper, LM Studio and ComfyUI — want the same graphics card, and running two of
them at once is how an overnight job turns into an out-of-memory error at 3am.
"""

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import OUTPUT_DIR, PROJECTS_DIR
from store.project_store import ProjectStore
from timeline import Timeline, Transition, generate_captions
from timeline.authoring import CAPTION_ORIGIN, clear_generated

from . import assets as assets_stage
from . import facezoom, placement
from .models import PresentationReport, PresentationSettings, StageTiming
from .program import build_program
from .shotplan import plan_shots, seed_for

logger = logging.getLogger("presentation.director")


class _Stages:
    """Runs stages, timing them and recording how each one went."""

    def __init__(self, report: PresentationReport):
        self.report = report

    async def run(self, name: str, coro, required: bool = False, default=None):
        started = time.time()
        try:
            result = await coro
            self.report.timings.append(
                StageTiming(stage=name, seconds=round(time.time() - started, 2), ok=True))
            return result
        except Exception as e:
            logger.exception("Presentation stage %r failed", name)
            self.report.timings.append(
                StageTiming(stage=name, seconds=round(time.time() - started, 2),
                            ok=False, note=str(e)[:300]))
            self.report.degraded.append(name)
            if required:
                raise
            return default


async def _llm_asker():
    """A JSON-asking callable bound to a model that is actually loaded, or None.

    Readiness has to come from `ensure_ready`, which loads the preferred model
    and returns the id that is really serving. Asking `/v1/models` instead lists
    everything downloaded and happily names a model that cannot load — which is
    how the LLM layer once spent weeks never running at all.
    """
    try:
        from llm.client import lm_studio_client
        from llm.lm_launcher import ensure_ready
        model = await ensure_ready(lm_studio_client.base_url, lm_studio_client.model_name)
        if not model:
            return None
        # Shot planning makes many model calls in windows over the transcript. On
        # a model too slow (or a thinking model that answers empty) that turns the
        # pass into an hours-long stall. If it can't answer a trivial prompt fast,
        # skip it — plan_shots then uses its keyword fallback, so B-roll still
        # gets made, just with generic prompts.
        if not await lm_studio_client.is_responsive():
            logger.warning("Language model is present but too slow for shot planning; "
                           "using the keyword fallback so the pass still runs.")
            return None

        async def ask(system_prompt: str, user_prompt: str, schema=None):
            return await lm_studio_client._chat_json(model, system_prompt,
                                                     user_prompt, schema=schema)

        return ask
    except Exception as e:
        logger.warning("Language model unavailable (%s); planning will fall back", e)
        return None


async def run_presentation_pass(
    project_id: str,
    settings: Optional[PresentationSettings] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    output_dir: Optional[Path] = None,
) -> PresentationReport:
    """Dress a finished cut into a presentable video. Never raises for a missing
    optional stage; raises only when the result could not be saved or rendered."""
    settings = settings or PresentationSettings()
    report_progress = progress_cb or (lambda fraction, message="": None)
    store = ProjectStore(base_dir=str(PROJECTS_DIR))

    report = PresentationReport(
        project_id=project_id,
        started_at=datetime.now().isoformat(),
        settings=settings.model_dump(),
    )
    stages = _Stages(report)
    seed = settings.seed if settings.seed is not None else seed_for(project_id)

    data = store.get_project(project_id)
    if not data:
        raise FileNotFoundError(f"Project {project_id} not found")
    source_video = data.get("source_video")

    # --- the cut itself ----------------------------------------------------
    report_progress(0.02, "Preparing the edit")
    timeline = await _ensure_timeline(data, store, project_id, stages)
    if timeline is None:
        raise RuntimeError("No timeline: the auto-edit has not run and could not be run")
    if timeline.default_transition is None:
        timeline.default_transition = Transition(type="fade", duration=0.25)

    # --- A. the programme --------------------------------------------------
    report_progress(0.10, "Reading the edit")
    program = build_program(timeline)
    report.program = {
        "duration_s": round(program.duration_s, 2),
        "word_count": len(program.words),
        "segment_count": len(program.segments),
        "has_energy": program.has_energy,
    }
    if not program.words:
        logger.warning("Presentation: the programme has no words; nothing to plan from")

    # --- B. the plan -------------------------------------------------------
    report_progress(0.15, "Working out the topics")
    ask = await _llm_asker()
    if ask is None:
        report.degraded.append("llm_unavailable")
    plan = await stages.run(
        "shotplan", plan_shots(program, settings, ask),
        default=None)
    if plan is None:
        from .models import ShotPlan
        plan = ShotPlan(beats=[], source="none")
    report.beats_planned = len(plan.beats)
    report.beats_dropped = plan.dropped
    report.plan_source = plan.source

    # --- everything the language model is for, done in one go --------------
    #
    # The thumbnail title is the one other thing that needs the model, and it
    # used to be asked for at the very end — after ComfyUI had already run. That
    # meant the LLM had to stay resident through generation. Ask for it now, while
    # the model is already up, so nothing downstream needs it.
    thumb_title: Optional[str] = None
    if settings.thumbnail and ask is not None:
        report_progress(0.20, "Writing the thumbnail title")
        thumb_title = await stages.run(
            "thumbnail_title", assets_stage.generate_thumbnail_title(program, ask),
            default=None)

    # Persist the plan (prompts + title) so generation reads from a file rather
    # than from a model that is about to be unloaded — and so a run is inspectable.
    _write_shot_plan(project_id, plan, thumb_title)

    # Eject the language model before ComfyUI starts. On a single card a resident
    # multi-GB LLM is exactly what makes image generation run out of memory, and
    # its work here is finished — everything below reads the plan, not the model.
    if ask is not None:
        report_progress(0.23, "Freeing the language model for generation")
        await stages.run(
            "eject_llm",
            _eject_llm(),
            default=None)

    # --- C. the assets -----------------------------------------------------
    # Record whether ComfyUI is actually there. The stages below degrade rather
    # than fail when it is not, so without this the report of a picture-less run
    # could not say *why* there are no pictures.
    if (settings.broll and plan.beats) or settings.thumbnail or settings.graphics:
        report.comfyui_online = await asyncio.to_thread(_comfyui_online)
        if not report.comfyui_online:
            report.degraded.append("comfyui_offline")

    generated: List = []
    if settings.broll and plan.beats:
        report_progress(0.25, "Generating pictures")
        result = await stages.run(
            "assets",
            assets_stage.generate_assets(
                plan.beats, PROJECTS_DIR / project_id, settings,
                progress_cb=lambda f, m="": report_progress(0.25 + 0.35 * f, m),
                seed_base=seed,
                canvas_size=(timeline.width, timeline.height)),
            default=([], []))
        generated, failures = result if result else ([], [])
        report.assets_generated = sum(1 for a in generated if not a.cache_hit)
        report.assets_cached = sum(1 for a in generated if a.cache_hit)
        report.assets_failed = failures

    # --- D. placement ------------------------------------------------------
    report_progress(0.62, "Cutting in the B-roll")
    rejected: List[Dict[str, str]] = []
    if settings.broll and generated:
        report.broll_placed = await stages.run(
            "place_broll",
            _sync(placement.place_broll, timeline, plan.beats, generated,
                  program, settings, seed, rejected),
            default=0) or 0
    report.beats_placement_rejected = rejected

    busy = placement.broll_windows(timeline)
    report.coverage_target = settings.target_coverage
    if program.duration_s > 0:
        report.coverage_achieved = round(
            sum(end - start for start, end in busy) / program.duration_s, 3)

    # Topic pop-ups are text call-outs, not a substitute for B-roll. When the user
    # asked for B-roll but ComfyUI was offline, the pass produces no images — and
    # dropping lone topic labels onto the plain video reads as "dummy markers"
    # standing in for the pictures that never generated. Suppress them in exactly
    # that case; when B-roll was not requested, or ComfyUI is up, they run as normal.
    broll_failed = settings.broll and report.comfyui_online is False
    if settings.popups and plan.beats and not broll_failed:
        report.popups_placed = await stages.run(
            "place_popups",
            _sync(placement.place_popups, timeline, plan.beats, program, settings, busy),
            default=0) or 0
    elif broll_failed:
        report.degraded.append("popups_skipped_no_broll")

    # --- E. the zooms ------------------------------------------------------
    if settings.face_zoom:
        report_progress(0.70, "Finding the speaker")
        anchors = {}
        if source_video:
            anchors = await stages.run(
                "face_detect",
                asyncio.to_thread(facezoom.detect_faces, source_video, program),
                default={}) or {}
        report.faces_detected_pct = (
            round(100.0 * len(anchors) / len(program.segments), 1)
            if program.segments else 0.0)

        report_progress(0.76, "Adding the zooms")
        zoom_stats: Dict[str, int] = {}
        planned = await stages.run(
            "plan_zooms",
            _sync(facezoom.plan_zooms, program, settings, busy, seed, zoom_stats),
            default=([], []))
        segment_zooms, window_zooms = planned if planned else ([], [])
        applied = await stages.run(
            "apply_zooms",
            _sync(facezoom.apply_zooms, timeline, segment_zooms, window_zooms, anchors),
            default=(0, 0))
        report.zooms_segment, report.zooms_windowed = applied if applied else (0, 0)
        report.zooms_suppressed_by_broll = int(zoom_stats.get("suppressed") or 0)

    # --- captions ----------------------------------------------------------
    if settings.captions:
        report_progress(0.80, "Writing the captions")
        items = await stages.run(
            "captions",
            _sync(_regenerate_captions, timeline, settings, data),
            default=[])
        report.captions = len(items or [])

    # --- save before rendering --------------------------------------------
    #
    # The render is a file; the timeline is the edit. Saving first means that
    # even if the render dies at 4am, the morning still has something to open.
    report_progress(0.86, "Saving the timeline")
    data["timeline"] = timeline.model_dump()
    data["status"] = "presented"
    store.save_project(project_id, data)

    # --- thumbnail ---------------------------------------------------------
    if settings.thumbnail:
        report_progress(0.88, "Making a thumbnail")
        # Title was written up front (before the LLM was ejected); this stage is
        # only the ComfyUI image now.
        report.thumbnail = await stages.run(
            "thumbnail", _make_thumbnail(project_id, data, thumb_title), default=None)

    # --- render ------------------------------------------------------------
    if settings.render:
        report_progress(0.90, "Rendering")
        output_path = str(OUTPUT_DIR / f"{project_id}_presented.mp4")
        rendered = await stages.run(
            "render",
            _render(timeline, output_path, data,
                    lambda pct: report_progress(0.90 + 0.09 * (pct / 100.0), "Rendering")),
            required=True)
        report.output_path = rendered
        data["output_path"] = rendered
        data["status"] = "rendered"
        store.save_project(project_id, data)

    # --- deliver -----------------------------------------------------------
    if output_dir:
        _collect(Path(output_dir), report.output_path, report.thumbnail)

    report.finished_at = datetime.now().isoformat()
    _write_report(project_id, report)
    report_progress(1.0, "Done")
    logger.info("Presentation pass complete for %s: %s", project_id,
                report.model_dump(exclude={"timings", "beats_dropped"}))
    return report


async def _sync(fn, *args, **kwargs):
    """Await a synchronous function so every stage has the same shape."""
    return fn(*args, **kwargs)


async def _ensure_timeline(data: Dict[str, Any], store: ProjectStore,
                           project_id: str, stages: _Stages) -> Optional[Timeline]:
    """The project's timeline, running the auto-edit first if there is not one."""
    if data.get("timeline"):
        return Timeline.model_validate(data["timeline"])

    source_video = data.get("source_video")
    if not source_video or not Path(source_video).exists():
        return None

    logger.info("Presentation: no timeline yet, running the auto-edit first")
    from asr import whisper_engine
    from asr.auto_edit import extract_project_audio, plan_auto_edit, record_cut_coverage
    from timeline import build_timeline_from_transcript
    from utils.ffmpeg_utils import get_video_info

    info = get_video_info(source_video)
    audio_path = await extract_project_audio(source_video)
    settings = data.get("settings") or {}
    # Pin the language across runs — auto-detect flips code-switched speech
    # into an English paraphrase with useless timings.
    words, detected_language = await whisper_engine.transcribe_words_async(
        audio_path or source_video, language=settings.get("language"))
    settings["language"] = detected_language
    data["settings"] = settings
    plan = await plan_auto_edit(
        words, audio_path,
        aggressiveness=float(settings.get("fumble_aggressiveness", 0.5)))
    timeline = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=info.get("duration", 0.0),
        transcript_words=plan.words,
        fps_num=info.get("fps_num", 30),
        fps_den=info.get("fps_den", 1),
        width=info.get("width", 1920),
        height=info.get("height", 1080),
        has_audio=info.get("audio_codec") not in (None, "none"),
        speech_regions=plan.report.get("speech"),
        energy_envelope=plan.report.get("energy"),
    )
    record_cut_coverage(timeline, plan.report)
    data["timeline"] = timeline.model_dump()
    store.save_project(project_id, data)
    return timeline


def _regenerate_captions(timeline: Timeline, settings: PresentationSettings,
                         data: Dict[str, Any]) -> List:
    clear_generated(timeline, CAPTION_ORIGIN)
    project_settings = data.get("settings") or {}
    # The chosen preset owns the caption band; a style profile's measured position
    # must not drag captions off it (its colour/box/size still carry). Matches the
    # /captions/generate route so both entry points place captions the same way.
    overrides = dict(project_settings.get("caption_style_overrides") or {})
    for geometry_key in ("pos_x", "pos_y"):
        overrides.pop(geometry_key, None)
    return generate_captions(
        timeline,
        preset=settings.caption_preset,
        style_overrides=overrides,
    )


async def _make_thumbnail(project_id: str, data: Dict[str, Any],
                          title: Optional[str]) -> Optional[str]:
    """The thumbnail image, from a title already written up front (no LLM here)."""
    from agents.thumbnail_agent import ThumbnailAgent
    from models import Project

    project = Project.model_validate(data)
    agent = ThumbnailAgent()
    return await agent.generate_thumbnail(project, PROJECTS_DIR / project_id, title=title)


def _comfyui_online() -> bool:
    """One probe, module-level so tests can stand it in."""
    from comfyui_bridge import queue_manager
    return queue_manager.client.is_connected()


async def _eject_llm() -> None:
    """Unload every LM Studio model so ComfyUI gets the whole GPU."""
    from llm.lm_launcher import unload_all_models
    from llm.client import lm_studio_client
    await unload_all_models(lm_studio_client.base_url)


def _write_shot_plan(project_id: str, plan, thumb_title: Optional[str]) -> None:
    """Save the one-shot LLM output — beat prompts and the thumbnail title — so
    generation reads it from disk, and so a run can be inspected afterwards."""
    try:
        path = PROJECTS_DIR / project_id / "shot_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": plan.source,
            "thumbnail_title": thumb_title,
            "beats": [b.model_dump() for b in plan.beats],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write the shot plan for %s: %s", project_id, e)


async def _render(timeline: Timeline, output_path: str, data: Dict[str, Any],
                  progress) -> str:
    from render.runner import render_timeline_async
    return await render_timeline_async(
        timeline, output_path,
        progress_callback=progress,
        output_resolution=(data.get("settings") or {}).get("resolution"),
    )


def _collect(output_dir: Path, *paths: Optional[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path:
            continue
        source = Path(path)
        if source.exists():
            try:
                shutil.copy2(source, output_dir / source.name)
            except Exception as e:
                logger.warning("Could not copy %s into %s: %s", source, output_dir, e)


def _write_report(project_id: str, report: PresentationReport) -> None:
    try:
        path = PROJECTS_DIR / project_id / "presentation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write the presentation report: %s", e)
