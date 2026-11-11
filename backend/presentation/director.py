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
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import OUTPUT_DIR, PROJECTS_DIR
from store.project_store import ProjectStore
from timeline import Timeline, Transition, clip_ops, generate_captions
from timeline.authoring import CAPTION_ORIGIN, clear_generated

from . import assets as assets_stage
from . import captions as captions_stage
from . import cards as cards_stage
from . import composite as composite_stage
from . import entities as entities_stage
from . import graphics as graphics_stage
from . import facezoom, look, placement
from . import mood as mood_stage
from . import script as script_stage
from . import shorts as shorts_stage
from . import sound as sound_stage
from . import structure as structure_stage
from . import verify as verify_stage
from .models import PresentationReport, PresentationSettings, StageTiming
from .program import build_program
from .shotplan import plan_shots, seed_for

logger = logging.getLogger("presentation.director")


class _Stages:
    """Runs stages, timing them and recording how each one went."""

    def __init__(self, report: PresentationReport):
        self.report = report

    async def run(self, name: str, coro, required: bool = False, default=None):
        from comfyui_bridge.errors import ComfyUIUnavailableError
        started = time.time()
        try:
            result = await coro
            self.report.timings.append(
                StageTiming(stage=name, seconds=round(time.time() - started, 2), ok=True))
            return result
        except ComfyUIUnavailableError:
            # A B-roll run that could not get its pictures must fail the whole
            # pass -- the scheduler job ends "failed" and is retried later --
            # rather than "degrade" and deliver a video with none. See the
            # 2026-10-20 incident. Recorded like any other failed stage, then
            # let through regardless of `required`.
            logger.exception("Presentation stage %r failed: ComfyUI never came back", name)
            self.report.timings.append(
                StageTiming(stage=name, seconds=round(time.time() - started, 2),
                            ok=False, note="ComfyUI did not come back"))
            self.report.degraded.append(name)
            raise
        except Exception as e:
            logger.exception("Presentation stage %r failed", name)
            self.report.timings.append(
                StageTiming(stage=name, seconds=round(time.time() - started, 2),
                            ok=False, note=str(e)[:300]))
            self.report.degraded.append(name)
            if required:
                raise
            return default


# --- shot-plan checkpoint ----------------------------------------------------
#
# The shot plan is the pass's longest model work (~10 minutes of calls on a
# long video), and the picture prompts it writes are the image cache's keys. A
# retry after a failed render used to plan again from nothing: new wording,
# so every picture already made missed the cache and was generated again. The
# plan is saved beside the project and reused while its inputs are unchanged.
SHOTPLAN_CHECKPOINT = "shotplan_checkpoint.json"
# Settings that do not shape the plan: which model answered, and the image seed.
_PLAN_IRRELEVANT = {"llm_api_key", "llm_base_url", "llm_model", "llm_provider", "seed"}


def _shotplan_key(program, settings: PresentationSettings, genre: str,
                  script_topics, extra_beats) -> str:
    def dump(x):
        if isinstance(x, (list, tuple)):
            return [dump(i) for i in x]
        return x.model_dump(mode="json") if hasattr(x, "model_dump") else x
    blob = json.dumps({
        "v": 1,
        "program": dump(program),
        "settings": settings.model_dump(mode="json", exclude=_PLAN_IRRELEVANT),
        "genre": genre,
        "script_topics": dump(script_topics),
        "extra_beats": dump(extra_beats),
    }, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _load_shotplan(project_dir: Path, key: str):
    """(ShotPlan, hook) saved for these exact inputs, or None."""
    from .models import ShotPlan
    from .structure import Hook
    path = project_dir / SHOTPLAN_CHECKPOINT
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("key") != key:
            return None
        hook = saved.get("hook")
        return (ShotPlan.model_validate(saved["plan"]),
                Hook.model_validate(hook) if hook else None)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Ignoring an unreadable shot-plan checkpoint (%s)", e)
        return None


def _save_shotplan(project_dir: Path, key: str, plan, hook) -> None:
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / SHOTPLAN_CHECKPOINT).write_text(json.dumps({
            "key": key,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "plan": plan.model_dump(mode="json"),
            "hook": hook.model_dump(mode="json") if hook is not None else None,
        }, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save the shot-plan checkpoint (%s)", e)


def _remote_client_for(settings: Optional[PresentationSettings]):
    """The Studio-supplied remote model for this job, or None for LM Studio."""
    if settings is None or getattr(settings, "llm_provider", "auto") != "openai_compat":
        return None
    base_url = (getattr(settings, "llm_base_url", "") or "").strip()
    if not base_url:
        return None
    from llm.client import RemoteChatClient
    return RemoteChatClient(base_url, getattr(settings, "llm_api_key", "") or "",
                            getattr(settings, "llm_model", "") or "")


async def _llm_asker(settings: Optional[PresentationSettings] = None):
    """A JSON-asking callable bound to a model that is actually loaded, or None.

    A job that names a remote endpoint (`settings.llm_provider ==
    "openai_compat"`, sent by BuzzcafStudio for its Claude proxy) uses that
    when it answers a probe -- no loading, no ejecting, and the slow
    responsiveness gate below is skipped because it exists for local models
    that can be present yet too slow to plan with. Otherwise, or when the
    endpoint is down, the local LM Studio path applies.

    Readiness has to come from `ensure_ready`, which loads the preferred model
    and returns the id that is really serving. Asking `/v1/models` instead lists
    everything downloaded and happily names a model that cannot load — which is
    how the LLM layer once spent weeks never running at all.
    """
    remote = _remote_client_for(settings)
    if remote is not None:
        model = await remote.ensure_ready()
        if model:
            logger.info("Planning passes on the remote model %r at %s", model, remote.base_url)

            async def ask_remote(system_prompt: str, user_prompt: str, schema=None):
                return await remote._chat_json(model, system_prompt, user_prompt, schema=schema)

            return ask_remote
        logger.warning("Remote model at %s not usable; falling back to LM Studio.", remote.base_url)
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
    settings_sources: Optional[Dict[str, str]] = None,
) -> PresentationReport:
    """See `_run_presentation_pass`. Whatever happens, every model the pass
    loaded (LLM, ComfyUI checkpoints, ASR) is offloaded when it ends -- the
    GPU policy in runtime/gpu_handover.py."""
    from runtime import gpu_handover
    try:
        return await _run_presentation_pass(project_id, settings, progress_cb,
                                            output_dir, settings_sources)
    finally:
        await gpu_handover.offload_all(f"presentation pass {project_id} finished")


async def _run_presentation_pass(
    project_id: str,
    settings: Optional[PresentationSettings] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    output_dir: Optional[Path] = None,
    settings_sources: Optional[Dict[str, str]] = None,
) -> PresentationReport:
    """Dress a finished cut into a presentable video. Never raises for a missing
    optional stage; raises only when the result could not be saved or rendered.

    `settings_sources`, when given, is recorded on the report as-is — it comes
    from wherever `settings` itself was assembled (Studio's dict, BuzzEdit's
    own overrides, the density preset, or a plain default); see
    presentation.models.settings_sources_for.
    """
    settings = settings or PresentationSettings()
    report_progress = progress_cb or (lambda fraction, message="": None)
    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    # The mid-production change (presentation/live.py): numbers the creator
    # set while an earlier pass of this project was running. A pass that
    # starts (or restarts) now plans with them rather than its job's snapshot.
    from . import live as live_mod
    live_at_start = live_mod.read(PROJECTS_DIR / project_id)
    if live_at_start:
        settings = live_mod.apply(settings, live_at_start)
        logger.info("Presentation: live settings applied at start (%s)", live_mod.describe(live_at_start))

    report = PresentationReport(
        project_id=project_id,
        started_at=datetime.now().isoformat(),
        settings=settings.model_dump(exclude={"llm_api_key"}),
        settings_sources=dict(settings_sources or {}),
    )
    stages = _Stages(report)
    seed = settings.seed if settings.seed is not None else seed_for(project_id)

    data = store.get_project(project_id)
    if not data:
        raise FileNotFoundError(f"Project {project_id} not found")
    source_video = data.get("source_video")

    # --- the cut itself ----------------------------------------------------
    report_progress(0.02, "Preparing the edit")
    # If the auto-edit has to run here, its pacing follows the genre (horror
    # keeps its pauses), and a genre named for this pass is the one to use.
    project_settings = data.get("settings") or {}
    data["settings"] = project_settings
    if settings.genre and not project_settings.get("genre"):
        project_settings["genre"] = settings.genre
    timeline = await _ensure_timeline(data, store, project_id, stages)
    if timeline is None:
        raise RuntimeError("No timeline: the auto-edit has not run and could not be run")
    if timeline.default_transition is None:
        timeline.default_transition = Transition(type="fade", duration=0.25)
    if settings.shorts_full and not shorts_stage.is_portrait(timeline):
        # The whole programme is wanted as a Short: plan it on the 9:16 canvas
        # from the start (portrait pictures and clips, close framing), rather
        # than dressing it landscape and cropping at the end.
        timeline = shorts_stage.plan_portrait(timeline)
        logger.info("Presentation: planning on the 9:16 canvas (the programme is a Short)")

    # --- the script, when there is one ---------------------------------------
    #
    # Aligned before anything reads the words: it fixes the caption spelling
    # and gives the planner the paragraph boundaries and stage directions.
    script_ctx = script_stage.ScriptContext()
    if (data.get("script") or {}).get("text"):
        record = await stages.run(
            "script", _sync(script_stage.apply_project_script, data, timeline),
            default=None)
        if record:
            script_ctx = script_stage.context_from(data, timeline)
            report.script_aligned_words = int(record.get("aligned_words", 0))
            report.script_spelling_fixed = int(record.get("spelling_fixed", 0))
            report.script_directives = len(script_ctx.directives)

    # --- A. the programme --------------------------------------------------
    report_progress(0.10, "Reading the edit")
    program = build_program(timeline)
    directives: Dict[str, Any] = {"beats": [], "sfx": [], "title": None, "moods": {},
                                  "music_mood": None, "music_cues": [], "fx": []}
    script_topics: List = []
    if script_ctx.present:
        script_topics = script_stage.paragraph_topics(script_ctx, program)
        directives = script_stage.directive_beats(script_ctx, program, settings)
    report.program = {
        "duration_s": round(program.duration_s, 2),
        "word_count": len(program.words),
        "segment_count": len(program.segments),
        "has_energy": program.has_energy,
    }
    if not program.words:
        logger.warning("Presentation: the programme has no words; nothing to plan from")

    # --- B. the plan -------------------------------------------------------
    report_progress(0.13, "Working out what kind of video this is")
    ask = await _llm_asker(settings)
    if ask is None:
        report.degraded.append("llm_unavailable")
    # The genre styles every generated prompt — a horror video gets horror
    # B-roll and a horror thumbnail. The user's setting wins; otherwise it is
    # read from the transcript, and a failure is just a plain ("general") video.
    from . import genre as genre_mod
    genre = settings.genre
    if not genre:
        genre = await stages.run(
            "genre", genre_mod.detect_genre(program, ask), default="general")
    genre = genre_mod.normalise(genre)
    report.genre = genre
    # The secondary genre is resolved again inside plan_shots (it needs the
    # blend for the beats it writes); the report just needs the same name.
    genre_secondary = (genre_mod.normalise(settings.genre_secondary)
                       if settings.genre_secondary else None)
    if genre_secondary in (None, "general", genre):
        genre_secondary = None
    report.genre_secondary = genre_secondary or ""

    report_progress(0.15, "Working out the topics")
    hook_box: Dict[str, Any] = {"hook": None}

    async def enrich(topics):
        # Acts tag the topics (so the budget follows the story) and the same
        # call picks the hook sentence for the cold open.
        if settings.structure:
            hook_box["hook"] = await structure_stage.tag_structure(topics, program, ask, genre)

    plan_key = _shotplan_key(program, settings, genre, script_topics, directives["beats"])
    checkpoint = _load_shotplan(PROJECTS_DIR / project_id, plan_key)
    if checkpoint is not None:
        plan, hook_box["hook"] = checkpoint
        logger.info("Reusing the saved shot plan (%d beats); the inputs are unchanged",
                    len(plan.beats))
        report.timings.append(StageTiming(stage="shotplan", seconds=0.0, ok=True,
                                          note="reused the saved plan"))
    else:
        plan = await stages.run(
            "shotplan", plan_shots(program, settings, ask, genre,
                                   script_topics=script_topics,
                                   extra_beats=directives["beats"],
                                   enrich=enrich if settings.structure else None),
            default=None)
        # Only a real model plan is worth keeping: a keyword fallback (model
        # down) should be replaced by a proper plan on the next run.
        if plan is not None and plan.source == "llm" and plan.beats:
            _save_shotplan(PROJECTS_DIR / project_id, plan_key, plan, hook_box["hook"])
    if plan is None:
        from .models import ShotPlan
        plan = ShotPlan(beats=[], source="none", genre=genre)
    # --- what the speaker names: cards and maps -----------------------------
    #
    # The model passes that do not read each other's answers -- moods,
    # the caption translation, the thumbnail title -- start now and run
    # beside the entity pass instead of one after another (~70 s saved on a
    # remote model). Each is awaited where its result is first needed.
    mood_task = None
    if settings.moods and plan.topics:
        mood_task = asyncio.create_task(stages.run(
            "moods",
            mood_stage.tag_moods(plan.topics, program, ask, genre,
                                 overrides=directives.get("moods") or {}),
            default=None))
    translate_task = None
    if settings.captions and settings.captions_bilingual.lower() not in ("off", "none", ""):
        from timeline.authoring import caption_script_for
        if (settings.captions_bilingual.lower() == "on"
                or caption_script_for(data.get("settings") or {}) == "native"):
            translate_task = asyncio.create_task(stages.run(
                "translate", captions_stage.translate_lines(program, ask), default=[]))
    thumb_title: Optional[str] = directives.get("title")
    thumb_scene: Optional[str] = None
    thumb_words: Optional[str] = directives.get("thumbnail_words")
    thumb_label: Optional[str] = directives.get("thumbnail_label")
    thumb_task = None
    if (settings.thumbnail or (settings.title and thumb_title is None)) and ask is not None:
        thumb_task = asyncio.create_task(stages.run(
            "thumbnail_title",
            assets_stage.generate_thumbnail_concept(program, ask, plan.topics),
            default=None))

    # One extraction over the topics (the model when it is up, patterns when
    # not) feeds every text card and every map. Validated like the model's
    # own beats, then merged into the plan so placement sees one list.
    if (settings.cards or settings.maps) and plan.topics:
        report_progress(0.18, "Reading names, places and numbers")
        records = await stages.run(
            "entities",
            entities_stage.extract_entities(plan.topics, program, ask, genre),
            default=[]) or []
        report.entities_found = sum(
            len(r[k]) for r in records
            for k in ("people", "places", "dates", "numbers", "terms", "quotes", "sources"))
        extra: List = []
        if records:
            extra.extend(entities_stage.entity_beats(records, plan.topics, program,
                                                     settings, genre))
        if settings.cards:
            extra.extend(entities_stage.structure_beats(plan.topics, program, settings))
        if not settings.cards:
            extra = [b for b in extra if not b.is_text]
        if extra:
            from .shotplan import sanitize_beats
            clean, dropped = sanitize_beats(extra, program, settings)
            plan.beats = sorted(plan.beats + clean, key=lambda b: b.start_s)
            plan.dropped = plan.dropped + dropped
            for index, beat in enumerate(plan.beats):
                if not beat.id:
                    beat.id = f"e{index:02d}"

    # Some claim/stat/quote beats become newspaper cutaways instead of text
    # cards, in genres whose blended text-fx palette wants a highlighter
    # sweep — a no-op everywhere else (shotplan.convert_claims_to_newspapers).
    if plan.beats:
        from .shotplan import convert_claims_to_newspapers
        plan.beats = convert_claims_to_newspapers(plan.beats, program, settings, genre,
                                                   genre_secondary)

    if settings.structure and plan.topics and not any(t.act for t in plan.topics):
        hook_box["hook"] = await stages.run(
            "structure", structure_stage.tag_structure(plan.topics, program, ask, genre),
            default=None)
    if settings.structure and hook_box["hook"] is None and settings.cold_open:
        hook_box["hook"] = structure_stage.fallback_hook(program)
    report.acts = [t.act or "" for t in sorted(plan.topics, key=lambda t: t.start_s)]

    # --- the emotional shape --------------------------------------------------
    if mood_task is not None:
        report_progress(0.19, "Reading the mood of each part")
        await mood_task
        report.moods = [t.mood or "" for t in sorted(plan.topics, key=lambda t: t.start_s)]

    report.beats_planned = len(plan.beats)
    report.beats_dropped = plan.dropped
    report.plan_source = plan.source

    # A split beat is two pictures; expand it before anything counts assets.
    plan.beats = composite_stage.expand_splits(plan.beats)
    # A paper-collage beat (MOTION_GRAPHICS_PLAN.md M5) carries 2-4 prompts;
    # expand it into generated-but-not-placed broll_image sub-beats the same
    # way, right above.
    from . import motion_collage as motion_collage_stage
    plan.beats = motion_collage_stage.expand_collages(plan.beats, genre, genre_secondary,
                                                       settings.art_style)
    # A stick_scene beat whose `bg` names a scene (STICK_WORLD_PLAN.md SW2)
    # needs a background picture of its own before the Remotion render can
    # run at all; expand it into a generated-but-not-placed broll_image
    # sub-beat the same way, right above.
    from . import motion_stick as motion_stick_stage
    plan.beats = motion_stick_stage.expand_stick_backgrounds(plan.beats, genre, genre_secondary,
                                                              settings.art_style)
    # A callout beat (MOTION_GRAPHICS_PLAN.md M10) carries one image prompt;
    # expand it into its own generated-but-not-placed broll_image sub-beat
    # the same way, right above.
    from . import motion_ui as motion_ui_stage
    plan.beats = motion_ui_stage.expand_callouts(plan.beats, genre, genre_secondary,
                                                 settings.art_style)

    # --- the bilingual line ------------------------------------------------------
    #
    # Captions are generated after the model is ejected, so the translation is
    # asked for now, sentence by sentence, and attached to the cards later.
    translations: List[Tuple[float, float, str]] = []
    if translate_task is not None:
        report_progress(0.195, "Translating the captions")
        translations = await translate_task or []
        if not translations:
            translations = captions_stage.translations_from_transcript(data, program, timeline)

    # --- everything the language model is for, done in one go --------------
    #
    # The thumbnail title is the one other thing that needs the model, and it
    # used to be asked for at the very end — after ComfyUI had already run. That
    # meant the LLM had to stay resident through generation. Ask for it now, while
    # the model is already up, so nothing downstream needs it.
    if thumb_task is not None:
        report_progress(0.20, "Writing the thumbnail title")
        concept = await thumb_task or {}
        thumb_title = thumb_title or concept.get("title")
        thumb_scene = concept.get("scene")
        thumb_words = thumb_words or concept.get("thumb")
        thumb_label = thumb_label if thumb_label is not None else concept.get("label")

    # The listing — titles, description with chapters, tags — while the model
    # is still resident; chapters shift by the cold open's length once it is cut.
    metadata_task = None
    if settings.metadata:
        report_progress(0.21, "Writing the listing")
        metadata_task = await stages.run(
            "metadata",
            structure_stage.write_metadata(PROJECTS_DIR / project_id, program, plan.topics,
                                           ask, genre, thumb_title),
            default=None)
        report.metadata_written = metadata_task is not None

    # Horror zooms creep: a clip asked to zoom is generated locked off and the
    # move goes back on the timeline, slow (presentation.camera).
    from . import camera as camera_mod
    steadied = camera_mod.steady_for_genre(plan.beats, genre)
    if steadied:
        logger.info("Presentation: %d %s clips will zoom gradually on the timeline", steadied, genre)

    # Persist the plan (prompts + title) so generation reads from a file rather
    # than from a model that is about to be unloaded — and so a run is inspectable.
    _write_shot_plan(project_id, plan, thumb_title, thumb_scene)

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
        generation_stats: Dict[str, Any] = {}
        result = await stages.run(
            "assets",
            assets_stage.generate_assets(
                plan.beats, PROJECTS_DIR / project_id, settings,
                progress_cb=lambda f, m="": report_progress(0.25 + 0.35 * f, m),
                seed_base=seed,
                canvas_size=(timeline.width, timeline.height),
                stats=generation_stats,
                character=plan.character),
            default=([], []))
        report.generation = generation_stats
        video_phase = generation_stats.get("video_phase") or {}
        if video_phase.get("skipped_reason"):
            report.degraded.append(f"video_broll: {video_phase['skipped_reason']}")
        # One subject flooding the pictures (Raat3Baje ep2: "tiger" in 91% of the
        # prompts, from the world line) is a prompt problem the report must name.
        drift = generation_stats.get("prompt_drift") or {}
        if isinstance(drift, dict) and drift:
            report.prompt_drift = {str(k): float(v) for k, v in drift.items()}
            worst = max(drift.items(), key=lambda kv: kv[1])
            report.degraded.append(
                f"prompt_drift: '{worst[0]}' is in {round(100 * float(worst[1]))}% of the prompts; "
                "check the world/look line")
        generated, failures = result if result else ([], [])
        # A paper-collage beat's sub-pictures just generated (MOTION_GRAPHICS_
        # PLAN.md M5): stash their files on the parent beat so build_paper_
        # collage and its fallback can find them without needing `generated`
        # threaded through MotionContext.
        motion_collage_stage.annotate_collage_assets(plan.beats, generated)
        # A stick_scene beat's own background picture just generated
        # (STICK_WORLD_PLAN.md SW2): stash its file and measured luma on the
        # parent beat so build_stick_scene and its fallback can find them
        # without needing `generated` threaded through MotionContext.
        motion_stick_stage.annotate_stick_backgrounds(plan.beats, generated)
        # A callout beat's own still just generated (MOTION_GRAPHICS_PLAN.md
        # M10): stash its file on the parent beat so build_callout and its
        # fallback can find it the same way.
        motion_ui_stage.annotate_callout_assets(plan.beats, generated)
        comfyui_made = [a for a in generated if getattr(a, "source", "generated") == "generated"]
        report.assets_generated = sum(1 for a in comfyui_made if not a.cache_hit)
        report.assets_cached = sum(1 for a in comfyui_made if a.cache_hit)
        report.assets_failed = failures
        # Beats ComfyUI could not make that free stock (Pexels/Pixabay) filled
        # instead — see presentation.stock. Empty unless allow_free_stock and a
        # key were both configured for this project.
        report.stock_used = [
            {"beat_id": a.beat_id, "source": a.source, "url": a.stock_url or "",
             "author": a.stock_author or ""}
            for a in generated if getattr(a, "source", "generated") != "generated"
        ]

    # --- C2. the pictures drawn here: maps and charts -------------------------
    #
    # No ComfyUI involved, so these run whether or not it is up, and they take
    # the same placement path as the generated B-roll.
    local_assets: List = []
    designed = any(b.kind in graphics_stage.DRAWN_CUTAWAY_KINDS for b in plan.beats)
    from . import motion as motion_stage
    # Not gated on settings.motion_graphics: a beat from a `[kinetic: ...]` the
    # user wrote still needs its static fallback drawn when the engine is off.
    has_motion = any(b.kind in motion_stage.BUILDERS for b in plan.beats)
    if plan.beats and (settings.maps or settings.cards or settings.newspapers
                       or settings.case_files or designed or has_motion):
        report_progress(0.60, "Drawing maps, charts, newspapers and case files")
        drawn = await stages.run(
            "maps_charts",
            asyncio.to_thread(_draw_local_assets, plan.beats, PROJECTS_DIR / project_id,
                              timeline.width, timeline.height, genre, settings, generated),
            default=([], []))
        local_assets, local_failures = drawn if drawn else ([], [])
        report.assets_failed = list(report.assets_failed) + list(local_failures)
    # A `stat_callout` a motion render already turned into its own animated
    # cutaway (MOTION_GRAPHICS_PLAN.md M2) must not also get cards.py's plain
    # text card below -- that would draw the same number twice.
    motion_stat_ids = {
        a.beat_id for a in local_assets
        if getattr(a, "source", "") == "motion"
        and next((b.kind for b in plan.beats if b.id == a.beat_id), None) == "stat_callout"
    }
    if local_assets:
        # A drawn card replaces anything generated for the same beat (a
        # source quote's background picture is inside the card already).
        drawn_ids = {a.beat_id for a in local_assets}
        generated = [a for a in generated if a.beat_id not in drawn_ids] + local_assets

    # --- C3. 2.5D parallax on a share of the stills (MOTION_GRAPHICS_PLAN.md
    # M6) -- after every still (generated + drawn) exists, before placement,
    # so a chosen still is placed exactly like any other motion render
    # (placement.py: source=="motion" -> no 5s floor, no extra Ken Burns) and
    # one that is skipped or fails keeps the ordinary Ken Burns still it
    # already was.
    if plan.beats and generated and settings.parallax_share > 0:
        from . import motion_parallax as motion_parallax_stage
        generated, parallax_stats = await stages.run(
            "parallax",
            asyncio.to_thread(motion_parallax_stage.apply_parallax, generated, plan.beats,
                              PROJECTS_DIR / project_id, timeline.width, timeline.height,
                              genre, settings, seed),
            default=(generated, {}))
        if parallax_stats:
            report.generation["parallax"] = parallax_stats

    # --- D. placement ------------------------------------------------------
    report_progress(0.62, "Cutting in the B-roll")
    live_now = live_mod.read(PROJECTS_DIR / project_id)
    if live_now:
        before = settings
        settings = live_mod.apply(settings, live_now)
        made_clips = {a.beat_id for a in generated if getattr(a, "kind", "") == "video"}
        trimmed, live_dropped = live_mod.trim_to_coverage(plan.beats, settings, program.duration_s)
        trimmed = live_mod.rebalance_clips(trimmed, settings, made_clips)
        kept_ids = {b.id for b in trimmed}
        kinds = {b.id: b.kind for b in trimmed}
        # Assets follow the beats: a dropped beat loses its picture, a beat
        # demoted to a still keeps only its still (every clip began as one).
        kept_assets = []
        for a in generated:
            if a.beat_id not in kept_ids:
                continue
            if getattr(a, "kind", "") == "video" and kinds.get(a.beat_id) != "broll_video":
                continue
            kept_assets.append(a)
        demoted = [a.beat_id for a in generated if getattr(a, "kind", "") == "video"
                   and a.beat_id in kept_ids and kinds.get(a.beat_id) != "broll_video"]
        for beat_id in demoted:
            still = next((a for a in generated if a.beat_id == beat_id and getattr(a, "kind", "") != "video"), None)
            if still is None:
                # The clip replaced its still in `generated`: keep the clip
                # rather than show nothing for an important beat.
                kept_assets.extend(a for a in generated if a.beat_id == beat_id and a not in kept_assets)
        generated = kept_assets
        plan.beats = trimmed
        report.live_overrides = {
            "settings": dict(live_now),
            "coverage": {"from": before.target_coverage, "to": settings.target_coverage},
            "video_share": {"from": before.video_broll_share, "to": settings.video_broll_share},
            "beats_dropped": len(live_dropped),
            "clips_kept": sum(1 for b in trimmed if b.kind == "broll_video"),
        }
        report.beats_dropped = list(report.beats_dropped) + live_dropped
        logger.info("Presentation: live settings applied before placement (%s): %d beat(s) dropped, %d clip(s)",
                    live_mod.describe(live_now), len(live_dropped), report.live_overrides["clips_kept"])
    # A paper-collage sub-beat's picture was generated to be cut out inside
    # the collage (MOTION_GRAPHICS_PLAN.md M5), never as its own cutaway.
    generated = motion_collage_stage.drop_collage_subs(generated, plan.beats)
    # A stick_scene background sub-beat's picture is drawn *under* the
    # figures inside the StickScene composition (STICK_WORLD_PLAN.md SW2),
    # never its own cutaway.
    generated = motion_stick_stage.drop_stick_background_subs(generated, plan.beats)
    # A callout sub-beat's picture is drawn onto inside the Callout
    # composition (MOTION_GRAPHICS_PLAN.md M10), never its own cutaway.
    generated = motion_ui_stage.drop_callout_subs(generated, plan.beats)
    rejected: List[Dict[str, str]] = []
    if (settings.broll or local_assets) and generated:
        report.broll_placed = await stages.run(
            "place_broll",
            _sync(placement.place_broll, timeline, plan.beats, generated,
                  program, settings, seed, rejected),
            default=0) or 0
    report.beats_placement_rejected = rejected
    def _placed_from(folder: str) -> int:
        return sum(
            1 for i in timeline.items if i.origin == placement.BROLL_ORIGIN and i.source_id
            and folder in Path(timeline.sources[i.source_id].path).parts)
    report.maps_placed = _placed_from("maps")
    report.newspapers_placed = _placed_from("newspaper")
    report.case_files_placed = _placed_from("dossier")

    busy = placement.broll_windows(timeline)
    report.coverage_target = settings.target_coverage
    if program.duration_s > 0:
        report.coverage_achieved = round(
            sum(end - start for start, end in busy) / program.duration_s, 3)
    cut_cover = placement.jump_cut_coverage(timeline)
    report.jump_cuts_total = cut_cover["total"]
    report.jump_cuts_covered = cut_cover["covered"]

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

    # --- face detection (early: cards' face-avoidance below and the text-fx
    #     pass later both need it; "E. the zooms" further down reuses this
    #     same dict instead of detecting a second time) ----------------------
    face_anchors: Dict[str, Tuple[float, float]] = {}
    if (settings.face_zoom or getattr(settings, "text_fx", True)) and source_video:
        face_anchors = dict(await stages.run(
            "face_detect", asyncio.to_thread(facezoom.detect_faces, source_video, program),
            default={}) or {})
        followed = shorts_stage.follow_faces(timeline, face_anchors)
        if followed:
            logger.info("Presentation: %d clip(s) of the 9:16 crop now follow the face", followed)

    # --- designed graphics: the live speaker in each layout's slot, the
    #     overlays on the speaker, and the AI disclosure tag -----------------
    if plan.beats:
        designed_counts: Dict[str, int] = {}
        speakers = await stages.run(
            "layout_speakers",
            _sync(graphics_stage.place_layout_speakers, timeline, plan.beats, face_anchors),
            default=0) or 0
        if speakers:
            designed_counts["speaker_slots"] = speakers
        overlay_counts = await stages.run(
            "overlays",
            _sync(graphics_stage.place_overlays, timeline, plan.beats, program,
                  PROJECTS_DIR / project_id, placement.broll_windows(timeline), face_anchors),
            default={}) or {}
        designed_counts.update(overlay_counts)
        for beat_kind in graphics_stage.DRAWN_CUTAWAY_KINDS + ("canvas_card",):
            placed_n = sum(1 for i in timeline.items
                           if i.origin == placement.BROLL_ORIGIN and i.source_id
                           and any(i.source_id.startswith(f"src_broll_{b.id}_")
                                   for b in plan.beats if b.kind == beat_kind))
            if placed_n:
                designed_counts[beat_kind] = placed_n
        report.designed_graphics = designed_counts
    if getattr(settings, "ai_label", True) and generated:
        ai_ids = [a.beat_id for a in generated
                  if getattr(a, "source", "generated") == "generated" and a.workflow_file]
        report.ai_labels = await stages.run(
            "ai_labels", _sync(graphics_stage.place_ai_labels, timeline, ai_ids), default=0) or 0

    # --- text cards ----------------------------------------------------------
    if settings.cards and plan.beats:
        report_progress(0.66, "Placing the cards")
        popup_windows = placement._windows_of(timeline, placement.POPUP_ORIGIN)
        placed_cards = await stages.run(
            "place_cards",
            _sync(cards_stage.place_cards, timeline, plan.beats, program, settings,
                  genre, genre_secondary, popup_windows, face_anchors, motion_stat_ids),
            default={})
        report.cards_placed = placed_cards or {}

    # --- composites: the speaker in a corner, freeze-frames ----------------------
    report.splits_placed = sum(1 for i in timeline.items
                               if i.origin == placement.BROLL_ORIGIN
                               and (i.label or "").startswith("split left"))
    if settings.pip and settings.pip.lower() not in ("off", "none", ""):
        report.pip_placed = await stages.run(
            "pip", _sync(composite_stage.place_pip, timeline, program, settings, genre),
            default=0) or 0
    if settings.freeze_on_stats and settings.cards and plan.beats:
        report.freezes_placed = await stages.run(
            "freezes",
            asyncio.to_thread(composite_stage.place_freezes, timeline, program, plan.beats,
                              settings, PROJECTS_DIR / project_id),
            default=0) or 0

    # --- E. the zooms ------------------------------------------------------
    # The resting frame first: every move below composes on top of it.
    report.base_zoom = await stages.run(
        "base_zoom", _sync(facezoom.apply_base_zoom, timeline, settings), default=1.0) or 1.0
    window_zoom_times: List[float] = []
    if settings.face_zoom:
        report_progress(0.70, "Finding the speaker")
        # Detected earlier, before the cards section, so it is not redone here.
        anchors = face_anchors
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
        window_zoom_times = [z.start_s for z in window_zooms]
        applied = await stages.run(
            "apply_zooms",
            _sync(facezoom.apply_zooms, timeline, segment_zooms, window_zooms, anchors),
            default=(0, 0))
        report.zooms_segment, report.zooms_windowed = applied if applied else (0, 0)
        report.zooms_suppressed_by_broll = int(zoom_stats.get("suppressed") or 0)
        report.zooms_punch_in = sum(1 for z in window_zooms if z.push_in)
        report.zooms_punch_out = sum(1 for z in window_zooms if not z.push_in)

    # --- captions ----------------------------------------------------------
    if settings.captions:
        report_progress(0.80, "Writing the captions")
        items = await stages.run(
            "captions",
            _sync(_regenerate_captions, timeline, settings, data),
            default=[])
        report.captions = len(items or [])
        from timeline.authoring import caption_script_for
        report.caption_script = caption_script_for(data.get("settings") or {})
        if settings.caption_emphasis:
            report.caption_emphasis = await stages.run(
                "caption_emphasis", _sync(captions_stage.mark_emphasis, timeline, program),
                default=0) or 0
        if translations:
            report.caption_translations = await stages.run(
                "caption_translations",
                _sync(captions_stage.attach_second_lines, timeline, translations),
                default=0) or 0

    # --- the opening title -------------------------------------------------
    #
    # The thumbnail title doubles as the video title: it was written while the
    # language model was up, so this stage only draws it. The default preset is
    # an overlay (pop-in over the opening footage); a card preset pushes the
    # whole programme back behind a title card, which apply_intro handles by
    # shifting every clip.
    if settings.title and thumb_title:
        report_progress(0.83, "Placing the title")
        applied = await stages.run(
            "title",
            _sync(_apply_title, timeline, settings, thumb_title),
            default=None)
        if applied:
            report.title_text = thumb_title

    # --- atmosphere --------------------------------------------------------
    #
    # One subtle full-length layer — grain, sunlight, fog — mapped from the
    # genre, so the video has a look rather than raw camera footage. Replaced
    # on every run (only the generated one is touched; effects the user added
    # stay).
    if settings.atmosphere and settings.atmosphere.lower() not in ("off", "none"):
        applied_effect = await stages.run(
            "atmosphere",
            _sync(_apply_atmosphere, timeline, settings, genre),
            default="")
        report.atmosphere_applied = applied_effect or ""

    # --- script-directed effects ------------------------------------------
    #
    # Effect windows the writer placed in the script ([fx: flicker], [atmos:
    # fog], [grade: saturation=0.7]) become adjustment layers over the moment
    # they were spoken; the five transition stingers ([fx: shape_wipe dur=0.8],
    # MOTION_GRAPHICS_PLAN.md M3) become a Remotion overlay instead -- hence
    # the extra project/size/genre/settings args _apply_script_fx now takes.
    # Replaced on every run; effects the user added stay.
    if directives.get("fx"):
        await stages.run(
            "script_fx",
            _sync(_apply_script_fx, timeline, directives.get("fx") or [], program,
                  PROJECTS_DIR / project_id, timeline.width, timeline.height, genre, settings),
            default=0)

    # --- the cold open ------------------------------------------------------
    #
    # Last of the layers: it moves the whole programme back to make room, so
    # everything placed above already sits where it will end up. Only the sound
    # lanes, placed below, are laid over the shifted timeline.
    if settings.structure and settings.cold_open and hook_box["hook"] is not None:
        report_progress(0.835, "Cutting the cold open")
        report.cold_open = await stages.run(
            "cold_open",
            _sync(structure_stage.apply_cold_open, timeline, program, hook_box["hook"], settings),
            default=None)
        if report.cold_open and settings.metadata:
            # Chapter stamps moved with the programme.
            await stages.run(
                "metadata_reshift",
                structure_stage.write_metadata(
                    PROJECTS_DIR / project_id, program, plan.topics, None, genre, thumb_title,
                    offset_s=timeline.cold_open_frames / (timeline.fps_num / max(1, timeline.fps_den))),
                default=None)
    else:
        structure_stage.remove_cold_open(timeline)

    # --- the progress bar ----------------------------------------------------
    #
    # After the cold open (its chapter times use the same offset as the
    # metadata) and the captions (it is placed clear of them). Replaced on
    # every run; removed when switched off.
    report.progress_bar = await stages.run(
        "progress_bar",
        _sync(_apply_progress_bar, timeline, settings, genre, plan.topics),
        default=None)

    # --- the mood recipes ----------------------------------------------------
    #
    # Grade shifts, pushes and atmosphere windows per topic, and the hits at
    # each key moment. Their sound (stingers, thunder, risers, heartbeat) and
    # their transition choices are handed to the stages that own those.
    mood_plan = mood_stage.MoodPlan()
    if settings.moods and plan.topics:
        report_progress(0.84, "Applying the moods")
        mood_plan = await stages.run(
            "apply_moods",
            _sync(mood_stage.apply_moods, timeline, plan.topics, program, settings,
                  genre, seed),
            default=mood_stage.MoodPlan()) or mood_stage.MoodPlan()
        report.mood_layers = mood_plan.layers
        report.mood_hits = mood_plan.hits

    # --- the look: grade and chapter transitions --------------------------
    if settings.grade and settings.grade.lower() not in ("off", "none"):
        applied_grade = await stages.run(
            "grade", _sync(look.apply_genre_grade, timeline, settings, genre),
            default=None)
        report.grade_applied = applied_grade or ""
    if settings.recreations:
        report.recreations_placed = await stages.run(
            "recreations", _sync(_apply_recreations, timeline, settings, genre),
            default=0) or 0
    if settings.topic_transitions and plan.topics:
        report.topic_transitions = await stages.run(
            "topic_transitions",
            _sync(look.apply_topic_transitions, timeline, plan.topics, genre, settings,
                  mood_plan.transitions),
            default=0) or 0
    # "Edit like this channel": a measured reference profile wins over the
    # genre look. Its motion is skipped when the face zooms already moved the
    # segments; its caption geometry never overrides the chosen preset.
    if settings.style_profile:
        applied_profile = await stages.run(
            "style_profile",
            _sync(_apply_style_profile, timeline, settings, source_video),
            default="")
        report.style_profile_applied = applied_profile or ""

    # --- sound -------------------------------------------------------------
    #
    # Music bed, effects on the cutaways and pop-ups, an ambience loop, and the
    # voice master. Placed last among the layers so every whoosh lands on the
    # final position of the cutaway it marks (the title stage above may have
    # shifted the programme behind a card).
    if settings.music or settings.sfx or settings.ambience or settings.sfx_diegetic:
        # A best-effort ComfyUI pre-pass for "auto"/"generated" sourcing: it
        # only fills gaps the library doesn't cover, and needs to run (and be
        # awaited) before the synchronous planning/placement below, which
        # only ever READS the cache it fills. Skipped whenever ComfyUI's
        # status was never checked (broll/thumbnail/graphics all off) — no
        # extra connectivity probe just for this.
        if settings.sfx_source in ("auto", "generated") or settings.music_source in ("auto", "generated"):
            await stages.run(
                "sound_generation",
                sound_stage.warm_generated_cache(
                    settings, genre, seed, comfyui_online=bool(report.comfyui_online),
                    acts=[t.act or "" for t in plan.topics],
                    music_cues=list(directives.get("music_cues") or []),
                    duration=program.duration_s,
                    progress=lambda msg: report_progress(0.845, msg)),
                default=None)
        report_progress(0.85, "Sound design")
        # Scene sounds: the script's [sfx: <effect>] cues first, then the sound
        # designer's (a model pass, or keywords from the beats), with the
        # effects the library has nothing for generated before placement.
        diegetic: List[Any] = []
        if settings.sfx_diegetic:
            from . import sfx_library, sound_design
            planned = await stages.run(
                "sound_design", sound_design.plan_diegetic(program, plan.beats, ask, genre),
                default=[])
            diegetic = sound_design.merge_cues(directives.get("diegetic") or [], planned or [])
            if diegetic and settings.sfx_source in ("auto", "generated"):
                await stages.run(
                    "sound_effects_generation",
                    sfx_library.warm_effects(diegetic, seed,
                                             comfyui_online=bool(report.comfyui_online),
                                             library_dirs=settings.sfx_library_dirs),
                    default=None)
        sound_result = await stages.run(
            "sound",
            _sync(_apply_sound, timeline, program, settings, genre, seed,
                  window_zoom_times, list(directives["sfx"]) + list(mood_plan.sfx),
                  mood_plan.loops,
                  [(t.start_s, t.end_s, t.act or "") for t in plan.topics],
                  list(directives.get("music_cues") or []), diegetic=diegetic),
            default=None)
        if sound_result:
            report.music_used = sound_result.get("music", "")
            report.sfx_placed = int(sound_result.get("sfx", 0))
            report.ambience_used = sound_result.get("ambience", "")
            report.sfx_source_used = sound_result.get("sfx_sources", {})
            report.music_source_used = sound_result.get("music_source", "")
            report.diegetic_placed = int(sound_result.get("diegetic", 0))
            report.diegetic_sources = sound_result.get("diegetic_sources", {})
            report.diegetic_unresolved = sound_result.get("diegetic_unresolved", [])
            for note in sound_result.get("notes", []):
                report.degraded.append(note)
    if settings.voice_preset and settings.voice_preset.lower() not in ("off", "none"):
        applied_voice = await stages.run(
            "voice_master", _sync(sound_stage.apply_voice_master, timeline, settings),
            default=None)
        report.voice_preset = applied_voice or ""
        if applied_voice and timeline.audio_master is not None:
            from render.audio import resolve_voice_enhance
            report.voice_enhance_used = resolve_voice_enhance(timeline.audio_master.voice_enhance)
            logger.info("Voice master: preset=%s denoise_engine=%s",
                       applied_voice, report.voice_enhance_used)

    # --- text effects --------------------------------------------------------
    #
    # Text behind the speaker's head, knockout, focus, newspaper highlighter
    # sweeps, kinetic keyword pop-ins, hand-drawn annotations, typewriter and
    # glitch reveals (presentation.text_fx). Runs after every other overlay
    # (cards, pip, zooms, captions) has settled onto the timeline, so its own
    # spacing has an accurate picture of what is already busy, and before the
    # timeline below is saved/rendered.
    if getattr(settings, "text_fx", True):
        report_progress(0.865, "Adding text effects")
        card_windows = cards_stage.card_windows(timeline)
        broll_busy = placement.broll_windows(timeline)
        text_fx_result = await stages.run(
            "text_fx",
            # Off the event loop thread, not `_sync`: the torchvision matte
            # backend leases the GPU broker (runtime.gpu_broker) around its
            # inference, which needs a thread with no event loop of its own
            # already running — see presentation.matte._run_with_gpu_lease.
            asyncio.to_thread(_apply_text_fx, timeline, program, plan.beats, generated, settings,
                              genre, face_anchors, broll_busy, card_windows,
                              PROJECTS_DIR / project_id, source_video, seed),
            default=({}, "off", []))
        placed, matte_used, fx_skipped = text_fx_result if text_fx_result else ({}, "off", [])
        report.text_fx_placed = placed or {}
        report.person_matte_used = matte_used or "off"
        report.text_fx_skipped = fx_skipped or []

    # --- emoji reactions (presentation.emoji_fx) ----------------------------
    #
    # After text_fx so the busy windows it reads off the timeline include the
    # text effects. Always runs: it clears the last run's items (origin
    # "emoji_fx") even when the setting or the genre now says off.
    from . import emoji_fx as emoji_stage
    report.emoji_reactions = await stages.run(
        "emoji_reactions",
        _sync(emoji_stage.plan_and_apply, timeline, program, settings, genre, face_anchors,
              PROJECTS_DIR / project_id, seed),
        default=0) or 0

    # --- save before rendering --------------------------------------------
    #
    # The render is a file; the timeline is the edit. Saving first means that
    # even if the render dies at 4am, the morning still has something to open.
    report_progress(0.86, "Saving the timeline")
    data["timeline"] = timeline.model_dump()
    data["status"] = "presented"
    # When this timeline was made, so a retry can tell it came from the job
    # whose render failed and re-render it (rerender_presented) instead of
    # running the whole pass again.
    data["presented_at"] = datetime.now().isoformat(timespec="seconds")
    store.save_project(project_id, data)

    # --- thumbnail ---------------------------------------------------------
    if settings.thumbnail:
        report_progress(0.88, "Making a thumbnail")
        # Title was written up front (before the LLM was ejected); this stage is
        # only the ComfyUI image now.
        report.thumbnail = await stages.run(
            "thumbnail", _make_thumbnail(project_id, data, thumb_title, genre, thumb_scene,
                                         thumb_words, thumb_label, art_style=settings.art_style),
            default=None)

    # --- render ------------------------------------------------------------
    if settings.render:
        # Written now as well as at the end: a render that dies leaves the
        # report of everything before it for a render-only retry to finish.
        _write_report(project_id, report)
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

    # --- Shorts ------------------------------------------------------------------
    #
    # After the main render, so a failure here costs only the extras. Each
    # clip is its own render of a slice of the dressed timeline.
    if settings.render and (settings.shorts_clips > 0 or settings.shorts_full) and plan.topics:
        report_progress(0.985, "Cutting Shorts")
        made = await stages.run(
            "shorts",
            shorts_stage.render_shorts(timeline, program, plan.topics, settings,
                                       OUTPUT_DIR / project_id, face_anchors),
            default=[])
        report.shorts = list(made or [])

    # --- verification ----------------------------------------------------------
    if settings.verify:
        report_progress(0.99, "Checking the result")
        checks = await stages.run(
            "verify_timeline",
            _sync(verify_stage.verify_timeline, timeline, program, settings, face_anchors),
            default=[]) or []
        if settings.render and report.output_path:
            checks += await stages.run(
                "verify_render",
                asyncio.to_thread(verify_stage.verify_render, report.output_path, settings,
                                  timeline),
                default=[]) or []
        report.verification = [c.model_dump() for c in checks]
        for check in checks:
            if not check.ok:
                report.degraded.append(f"check:{check.name}")
        verify_stage.write_verification(PROJECTS_DIR / project_id, checks)

    # --- deliver -----------------------------------------------------------
    if output_dir:
        _collect(Path(output_dir), report.output_path, report.thumbnail)

    report.finished_at = datetime.now().isoformat()
    _write_report(project_id, report)
    report_progress(1.0, "Done")
    logger.info("Presentation pass complete for %s: %s", project_id,
                report.model_dump(exclude={"timings", "beats_dropped"}))
    return report


async def rerender_presented(
    project_id: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    output_dir: Optional[Path] = None,
) -> PresentationReport:
    """Render the timeline a presentation pass already saved, and nothing else.

    For a pass whose render failed (ffmpeg killed, the machine slept): every
    stage before the render -- planning, pictures, music, placement -- is
    already in the saved timeline, so re-running them would only cost time.
    Raises when there is no presented timeline to render.
    """
    report_progress = progress_cb or (lambda fraction, message="": None)
    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    data = store.get_project(project_id)
    if not data:
        raise FileNotFoundError(f"Project {project_id} not found")
    if data.get("status") != "presented" or not data.get("timeline"):
        raise RuntimeError(f"Project {project_id} has no presented timeline to re-render "
                           f"(status {data.get('status')!r})")
    timeline = Timeline.model_validate(data["timeline"])

    path = PROJECTS_DIR / project_id / "presentation_report.json"
    try:
        report = PresentationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        report = PresentationReport(project_id=project_id, started_at=datetime.now().isoformat())
    stages = _Stages(report)

    report_progress(0.90, "Rendering (re-using the saved edit)")
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

    if output_dir:
        _collect(Path(output_dir), report.output_path, report.thumbnail)
    report.finished_at = datetime.now().isoformat()
    _write_report(project_id, report)
    report_progress(1.0, "Done")
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
    from asr.auto_edit import extract_project_audio, pacing_kwargs, plan_auto_edit, record_cut_coverage
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
    precut = bool(settings.get("precut"))
    if precut:
        # Already cut by the user: keep every frame (Timeline.keep_full_source).
        plan = None
        plan_words, plan_report = words, {"precut": True}
    else:
        plan = await plan_auto_edit(
            words, audio_path,
            aggressiveness=float(settings.get("fumble_aggressiveness", 0.5)),
            settings=settings)
        plan_words, plan_report = plan.words, plan.report
    timeline = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=info.get("duration", 0.0),
        transcript_words=plan_words,
        fps_num=info.get("fps_num", 30),
        fps_den=info.get("fps_den", 1),
        width=info.get("width", 1920),
        height=info.get("height", 1080),
        has_audio=info.get("audio_codec") not in (None, "none"),
        speech_regions=plan_report.get("speech"),
        energy_envelope=plan_report.get("energy"),
        keep_full_source=precut,
        **pacing_kwargs(settings),
    )
    if plan is not None:
        record_cut_coverage(timeline, plan_report)
    data["timeline"] = timeline.model_dump()
    store.save_project(project_id, data)
    return timeline


def _apply_title(timeline: Timeline, settings: PresentationSettings,
                 title: str) -> bool:
    from timeline.authoring import apply_intro
    apply_intro(timeline, settings.title_preset, title=title)
    return True


ATMOSPHERE_ORIGIN = "presentation"


def _apply_atmosphere(timeline: Timeline, settings: PresentationSettings,
                      genre: str) -> str:
    """Replace the pass's own atmosphere layer; effects the user added stay."""
    from timeline.presets import effect_preset
    from timeline.schema import AtmosphereEffect
    from . import genre as genre_mod

    timeline.effects = [e for e in timeline.effects
                        if getattr(e, "origin", None) != ATMOSPHERE_ORIGIN]

    choice = (settings.atmosphere or "").lower()
    if choice == "auto":
        mapped = genre_mod.atmosphere_for(genre)
        if not mapped:
            return ""
        effect_type, intensity = mapped
        values: Dict[str, Any] = {"type": effect_type, "intensity": intensity}
    else:
        values = effect_preset(choice)
        if not values:
            logger.warning("Unknown atmosphere preset %r; skipping", choice)
            return ""

    timeline.effects.append(AtmosphereEffect(
        type=str(values["type"]),
        intensity=float(values.get("intensity", 0.5)),
        speed=float(values.get("speed", 1.0)),
        color=values.get("color"),
        origin=ATMOSPHERE_ORIGIN,
    ))
    timeline.revision += 1
    return str(values["type"])


PROGRESS_ORIGIN = "presentation:progress"
# Genres whose viewers navigate by chapter (explainers, infotainment). A story
# (horror, true crime, vlog, ...) keeps its frame clean, and so does "general",
# which is also the fallback when genre detection fails.
PROGRESS_GENRES = ("tech", "science_education", "finance", "news", "documentary",
                   "geopolitics", "health_fitness")
_PROGRESS_BAR_PX = 6     # at 1080p, like everything below
_PROGRESS_GAP_PX = 8


def _progress_bar_wanted(settings: PresentationSettings, genre: str) -> bool:
    if not settings.progress_bar:
        return False
    style = (settings.progress_bar_style or "auto").lower()
    if style in ("off", "none"):
        return False
    if style == "auto":
        from . import genre as genre_mod
        return genre_mod.normalise(genre) in PROGRESS_GENRES
    return True


def _caption_band(timeline: Timeline) -> Optional[Tuple[float, float]]:
    """(top, bottom) of the caption block in 1080p pixels, or None without captions.

    Both caption engines centre the block on its pos_y anchor (drawtext
    `_position_expressions`, ASS `_position`). The block is estimated as the
    wrapped main lines (about 0.55 em per character over 90% of the width) plus
    the smaller translation line (0.55x font size), the outline and any box.
    """
    band: Optional[Tuple[float, float]] = None
    for item in timeline.items:
        if item.origin != CAPTION_ORIGIN or item.text is None or not item.enabled:
            continue
        style = item.text.style
        size = float(style.font_size)
        per_line = max(1, int(1920 * 0.9 / max(1.0, size * 0.55)))
        lines = max(1, -(-len(item.text.content or "") // per_line))
        block = lines * size * 1.25 + (lines - 1) * style.line_spacing
        if item.text.second_line:
            block += size * 0.55 * 1.25
        block += 2 * style.stroke_width + (2 * style.box_padding if style.box else 0)
        centre = (max(-1.0, min(1.0, style.pos_y)) + 1.0) / 2.0 * 1080
        top, bottom = centre - block / 2.0, centre + block / 2.0
        band = (top, bottom) if band is None else (min(band[0], top), max(band[1], bottom))
    return band


def _progress_placement(timeline: Timeline) -> Dict[str, Any]:
    """Where the bar goes so it never overlaps the captions (1080p pixels).

    The usable bottom edge is the bottom of the picture, above the bottom
    letterbox bar when `aspect_bars` letterboxes, because the bars are matted
    over the bar (compiler 7c). Captions are burnt in after the bar, so the
    rule only has to keep them apart:
      * no captions, or captions in the top half: the bar sits at the bottom
        edge with its chapter label;
      * captions in the bottom half with room under them: the bar is 6 px at
        the very bottom edge, under them, and the label is hidden;
      * captions reaching the bottom edge: the bar sits above the caption band
        (the margin is measured from the band's top) with its label above it.
    """
    edge = 0.0
    ratio = float(timeline.aspect_bars or 0)
    if ratio > 1920 / 1080 + 0.01:
        edge = (1080 - 1920 / ratio) / 2.0
    band = _caption_band(timeline)
    if band is None or band[0] + band[1] < 1080:   # centred in the top half
        return {"position": "bottom", "margin_px": int(round(edge)), "label": "name"}
    usable_bottom = 1080 - edge
    if band[1] + _PROGRESS_GAP_PX <= usable_bottom - _PROGRESS_BAR_PX:
        return {"position": "bottom", "margin_px": int(round(edge)), "label": "none"}
    return {"position": "bottom",
            "margin_px": int(round(1080 - band[0] + _PROGRESS_GAP_PX)), "label": "name"}


def _progress_accent(genre: str) -> str:
    """The genre's accent: the thumbnail title look's second colour (yellow by
    default). text_fx.palette_for lists text-effect styles, not colours."""
    from .thumbnail_compose import style_for
    rgb = style_for(genre).get("label_second") or (255, 212, 0)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _apply_progress_bar(timeline: Timeline, settings: PresentationSettings,
                        genre: str, topics) -> Optional[Dict[str, Any]]:
    """Replace the pass's own progress bar; one the user added stays.

    Returns {"chapters": n} for the report, None when the bar is off.
    """
    from render.atmosphere import PROGRESS_BAR_DEFAULTS
    from timeline.schema import AtmosphereEffect

    before = len(timeline.effects)
    timeline.effects = [e for e in timeline.effects
                        if getattr(e, "origin", None) != PROGRESS_ORIGIN]
    if not _progress_bar_wanted(settings, genre):
        if len(timeline.effects) != before:
            timeline.revision += 1
        return None

    fps = timeline.fps_num / max(1, timeline.fps_den)
    offset_s = timeline.cold_open_frames / fps   # the metadata's offset
    chapters = [{"at_s": round(at, 3), "name": name}
                for at, name in structure_stage.chapter_seconds(topics or [], offset_s)]
    extra: Dict[str, Any] = {**PROGRESS_BAR_DEFAULTS, **_progress_placement(timeline),
                             "chapters": chapters, "height_px": _PROGRESS_BAR_PX,
                             "fill_color": _progress_accent(genre)}
    timeline.effects.append(AtmosphereEffect(
        type="progress_bar", intensity=1.0, origin=PROGRESS_ORIGIN, extra=extra))
    timeline.revision += 1
    return {"chapters": len(chapters)}


RECREATION_ORIGIN = "recreation"
RECREATION_GENRES = ("true_crime", "horror")


def _apply_recreations(timeline: Timeline, settings: PresentationSettings,
                       genre: str) -> int:
    """Give reenactment B-roll a 'recreation' look.

    In the investigative genres the generated cutaways ARE reconstructions of a
    past event, so they get a desaturated, grainy grade. The broadcast-style
    'DRAMATIZATION' super that used to accompany it is gone (2026-10-21, the
    creator: "no need for this"; a story channel carries no text over its
    pictures). Only the pass's own layer is replaced on a re-run; anything the
    user added stays."""
    from timeline.authoring import clear_generated
    from timeline.schema import AtmosphereEffect

    clear_generated(timeline, RECREATION_ORIGIN)
    from . import art_style as art_mod
    if genre not in RECREATION_GENRES or not art_mod.is_photoreal_base(settings.art_style):
        # A drawing is not a reconstruction: no grain on a stick figure.
        return 0

    broll = sorted(
        (i for i in timeline.items
         if i.origin == placement.BROLL_ORIGIN and i.source_id and i.kind == "media"
         and "generated" in Path(timeline.sources[i.source_id].path).parts),
        key=lambda i: i.timeline_start_frame)
    if not broll:
        return 0

    count = 0
    for item in broll:
        start_frame = item.timeline_start_frame
        frames = max(1, item.timeline_end_frame - item.timeline_start_frame)
        layer = clip_ops.add_adjustment_item(timeline, start_frame, frames, track="V7")
        layer.origin = RECREATION_ORIGIN
        layer.label = "recreation look"
        clip_ops.set_color(timeline, layer.id,
                           {"saturation": 0.72, "contrast": 1.06, "vignette": 0.22})
        layer.atmosphere.append(AtmosphereEffect(
            type="grain", intensity=0.25, origin=RECREATION_ORIGIN))
        count += 1

    timeline.recalculate_duration()
    timeline.revision += 1
    logger.info("Recreations: %d clips graded", count)
    return count


SCRIPT_FX_ORIGIN = "script_fx"


def _apply_script_fx(timeline: Timeline, fx_list: List[Dict[str, Any]], program,
                     project_dir: Optional[Path] = None, width: int = 0, height: int = 0,
                     genre: str = "general",
                     settings: Optional[PresentationSettings] = None) -> int:
    """Place the effect windows the script asked for.

    Each `[fx: ...]` / `[atmos: ...]` / `[grade: ...]` directive from the script
    becomes an adjustment item at the moment it was spoken, carrying an
    AtmosphereEffect (or a colour grade) over its window -- except the five
    Remotion transition stingers (MOTION_GRAPHICS_PLAN.md M3: shape_wipe, iris,
    ink, bars, paper_tear), which are not a filter over the footage but a
    transparent overlay of their own, rendered and placed by
    `motion_transitions.place_stinger` centred on `at` instead. `project_dir`/
    `width`/`height`/`genre`/`settings` are only needed for those; omitted (as
    every existing caller still may), a stinger directive simply falls back to
    whiteflash like a render failure would. Only this pass's own layers are
    replaced on a re-run; effects the user added by hand stay."""
    from timeline.authoring import clear_generated
    from timeline.schema import AtmosphereEffect, time_to_frame

    from . import motion_transitions as transitions_stage

    clear_generated(timeline, SCRIPT_FX_ORIGIN)
    if not fx_list:
        return 0

    count = 0
    for fx in fx_list:
        at = max(0.0, float(fx.get("at") or 0.0))
        dur = max(0.2, float(fx.get("dur") or 1.0))
        effect_name = fx.get("effect")
        if fx.get("kind") == "fx" and effect_name in transitions_stage.STINGER_NAMES:
            if project_dir is not None and settings is not None and transitions_stage.place_stinger(
                    timeline, effect_name, at, dur, program, project_dir, width, height, genre,
                    settings, origin=SCRIPT_FX_ORIGIN):
                count += 1
                continue
            logger.warning("Transition stinger %r unavailable; falling back to whiteflash",
                           effect_name)
            fx = {**fx, "effect": "whiteflash"}
        start_frame = time_to_frame(at, timeline.fps_num, timeline.fps_den)
        frames = max(1, time_to_frame(dur, timeline.fps_num, timeline.fps_den))
        try:
            layer = clip_ops.add_adjustment_item(timeline, start_frame, frames, track="V7")
        except Exception as exc:
            logger.warning("script fx %r could not be placed: %s", fx.get("effect"), exc)
            continue
        layer.origin = SCRIPT_FX_ORIGIN
        layer.label = f"fx: {fx.get('effect') or fx.get('kind')}"
        if fx.get("kind") == "grade":
            grade = fx.get("grade") or {}
            if grade:
                try:
                    clip_ops.set_color(timeline, layer.id, grade)
                except Exception as exc:
                    logger.warning("script grade could not be applied: %s", exc)
        else:
            layer.atmosphere.append(AtmosphereEffect(
                type=fx.get("effect") or "grain",
                intensity=float(fx.get("intensity", 0.5)),
                speed=float(fx.get("speed", 1.0)),
                color=fx.get("color"),
                origin=SCRIPT_FX_ORIGIN))
        count += 1

    timeline.recalculate_duration()
    timeline.revision += 1
    logger.info("Script FX: %d effect window(s) placed", count)
    return count


def _apply_text_fx(timeline: Timeline, program, beats, generated_assets,
                   settings: PresentationSettings, genre: str,
                   face_anchors: Dict[str, Tuple[float, float]],
                   busy: List[Tuple[float, float]], card_windows: List[Tuple[float, float]],
                   project_dir: Path, source_video: Optional[str], seed: int):
    """Plan and place behind_head/knockout/focus/annotation/newspaper-sweep/
    kinetic-words/typewriter/glitch. Returns (counts_by_style, matte_backend,
    skipped) — see presentation.text_fx.apply_moments."""
    from . import text_fx as text_fx_stage
    canvas_w = timeline.width or 1920
    canvas_h = timeline.height or 1080
    moments = text_fx_stage.plan_moments(program, beats, timeline, generated_assets, settings,
                                         genre, face_anchors, busy, card_windows, seed=seed)
    return text_fx_stage.apply_moments(timeline, moments, settings, project_dir, source_video,
                                       canvas_w, canvas_h)


def _apply_sound(timeline: Timeline, program, settings: PresentationSettings,
                 genre: str, seed: int,
                 punch_in_times: Optional[List[float]] = None,
                 extra_sfx: Optional[List[Tuple[float, str]]] = None,
                 loops: Optional[List[Tuple[float, float, str, float]]] = None,
                 sections: Optional[List[Tuple[float, float, str]]] = None,
                 music_cues: Optional[List[Tuple[float, str]]] = None,
                 diegetic: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Plan and place the music, effects and ambience lanes."""
    from timeline.schema import frame_to_time
    broll = placement.broll_windows(timeline)
    popups = [frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den)
              for i in timeline.items if i.origin == placement.POPUP_ORIGIN]
    extra = [(t, "whoosh_soft") for t in (punch_in_times or [])]
    extra.extend(extra_sfx or [])
    plan = sound_stage.plan_sound(timeline, program, settings, genre, seed,
                                  broll_windows=broll, popup_times=popups,
                                  extra_sfx=extra, loops=loops, sections=sections,
                                  music_cues=music_cues, diegetic=diegetic)
    sfx_sources: Dict[str, int] = {}
    counts = sound_stage.apply_sound(timeline, plan, settings, seed, source_counts=sfx_sources)
    music = ""
    if plan.music:
        names = []
        for cue in plan.music:
            name = f"synth:{Path(cue.path).stem}" if cue.synthesised else Path(cue.path).name
            names.append(f"{name} [{cue.act}]" if cue.act else name)
        music = " → ".join(dict.fromkeys(names))
    ambience = ""
    if plan.ambience:
        cue = plan.ambience[0]
        ambience = f"synth:{cue.kind}" if cue.synthesised else cue.kind
    music_source = plan.music[0].source if plan.music else ""
    return {"music": music, "sfx": counts.get("sfx", 0), "ambience": ambience,
            "notes": plan.notes, "sfx_sources": sfx_sources, "music_source": music_source,
            "diegetic": counts.get("diegetic", 0),
            "diegetic_sources": counts.get("diegetic_sources", {}),
            "diegetic_unresolved": counts.get("diegetic_unresolved", [])}


def _apply_style_profile(timeline: Timeline, settings: PresentationSettings,
                         source_video: Optional[str]) -> str:
    from style import ApplyOptions, apply_profile
    from style.profile import load
    profile = load(str(settings.style_profile))
    if profile is None:
        logger.warning("Style profile %r not found", settings.style_profile)
        return ""
    apply_profile(timeline, profile, source_video,
                  ApplyOptions(look=True, motion=not settings.face_zoom, captions=False,
                               transitions=True))
    return profile.name


def _draw_local_assets(beats, project_dir: Path, width: int, height: int, genre: str,
                       settings: PresentationSettings, generated=()):
    """Maps, charts, newspapers and case files for the beats that want them
    (synchronous; run in a thread). None of these touch ComfyUI, so they run
    whether or not it is online and take the same placement path as B-roll."""
    from . import charts as charts_stage
    from . import dossier as dossier_stage
    from . import maps as maps_stage
    from . import motion as motion_stage
    from . import newspaper as newspaper_stage
    assets: List = []
    failures: List[Dict[str, str]] = []

    # Motion-template graphics (presentation/motion.py; MOTION_GRAPHICS_PLAN.md)
    # run first: a beat they handle is skipped by every PIL renderer below, and
    # one they could not (engine down, a bad prop) gets its static fallback --
    # a motion failure must never fail the render.
    motion_handled: set = set()
    if settings.motion_graphics:
        m_assets, m_failures, motion_handled = motion_stage.render_motion_assets(
            beats, project_dir, width, height, genre, settings)
        assets += m_assets
        by_id = {b.id: b for b in beats}
        for failure in m_failures:
            beat = by_id.get(failure.get("beat_id"))
            fallback = motion_stage.render_fallback(beat, project_dir, width, height)
            if fallback is not None:
                assets.append(fallback)
                motion_handled.add(beat.id)
                failure = dict(failure, reason=f"{failure.get('reason')}; drew a static card instead")
            failures.append(failure)
    beats = [b for b in beats if b.id not in motion_handled]

    if settings.maps:
        drawn, failed = maps_stage.render_map_assets(beats, project_dir, width, height,
                                                     genre, settings.map_style)
        assets += drawn
        failures += failed
    drawn, failed = charts_stage.render_chart_assets(beats, project_dir, width, height, genre)
    assets += drawn
    failures += failed
    if settings.newspapers:
        drawn, failed = newspaper_stage.render_newspaper_assets(beats, project_dir, width,
                                                               height, genre)
        assets += drawn
        failures += failed
    if settings.case_files:
        drawn, failed = dossier_stage.render_case_file_assets(beats, project_dir, width,
                                                             height, genre)
        assets += drawn
        failures += failed
    drawn, failed = graphics_stage.render_graphic_assets(beats, project_dir, width, height,
                                                         generated)
    assets += drawn
    failures += failed
    return assets, failures


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
    from timeline.authoring import caption_script_for
    return generate_captions(
        timeline,
        preset=settings.caption_preset,
        style_overrides=overrides,
        script=caption_script_for(project_settings),
    )


async def _make_thumbnail(project_id: str, data: Dict[str, Any],
                          title: Optional[str],
                          genre: str = "general",
                          scene: Optional[str] = None,
                          words: Optional[str] = None,
                          label: Optional[str] = None,
                          art_style: str = "real") -> Optional[str]:
    """The thumbnail image, from a title already written up front (no LLM here)."""
    from agents.thumbnail_agent import ThumbnailAgent
    from runtime import gpu_handover
    # Its own model: hand the card over from whatever phase ran before.
    await gpu_handover.prepare_for_phase("thumbnail", need_vram_mb=6000, need_ram_mb=4000)
    from models import Project

    project = Project.model_validate(data)
    agent = ThumbnailAgent()
    return await agent.generate_thumbnail(project, PROJECTS_DIR / project_id,
                                          title=title, genre=genre, scene=scene,
                                          words=words, label=label, art_style=art_style)


def _comfyui_online() -> bool:
    """One probe, module-level so tests can stand it in."""
    from comfyui_bridge import queue_manager
    return queue_manager.client.is_connected()


async def _eject_llm() -> None:
    """Unload every LM Studio model so ComfyUI gets the whole GPU.

    Through the `lms` CLI unconditionally: the HTTP check behind
    `unload_all_models` asks whichever server the LLM client points at, which
    is not LM Studio when the passes run on a remote model -- and then a model
    loaded by anything else on the machine was never ejected."""
    from runtime.gpu_handover import eject_lm_studio
    await eject_lm_studio()


def _write_shot_plan(project_id: str, plan, thumb_title: Optional[str],
                     thumb_scene: Optional[str] = None) -> None:
    """Save the one-shot LLM output — beat prompts and the thumbnail title — so
    generation reads it from disk, and so a run can be inspected afterwards."""
    try:
        path = PROJECTS_DIR / project_id / "shot_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": plan.source,
            "genre": plan.genre,
            "thumbnail_title": thumb_title,
            "thumbnail_scene": thumb_scene,
            "beats": [b.model_dump() for b in plan.beats],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write the shot plan for %s: %s", project_id, e)


def _output_resolution(timeline: Timeline, data: Dict[str, Any]) -> Optional[str]:
    """The project's resolution setting, unless it would letterbox the picture.

    The project default is 1920x1080; a phone recording is 1080x1920, and
    forcing it into the default pads a vertical video into a landscape frame
    with black bars either side. When the aspects disagree the timeline's own
    size wins.
    """
    wanted = (data.get("settings") or {}).get("resolution")
    if not wanted:
        return None
    try:
        w, h = (int(v) for v in str(wanted).lower().split("x"))
    except ValueError:
        return None
    if timeline.width and timeline.height:
        if (w > h) != (timeline.width > timeline.height):
            return f"{timeline.width}x{timeline.height}"
    return wanted


async def _render(timeline: Timeline, output_path: str, data: Dict[str, Any],
                  progress) -> str:
    from render.runner import render_timeline_async
    return await render_timeline_async(
        timeline, output_path,
        progress_callback=progress,
        output_resolution=_output_resolution(timeline, data),
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
