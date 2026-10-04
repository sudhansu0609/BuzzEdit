"""The effects sample: ~20 s of the creator's own cut, dressed on purpose.

The full presentation pass takes hours, so its look was only ever judged the
next morning. This builds a short, deliberately dense stretch instead -- the
captions where the full render will put them, a full-frame knockout word,
kinetic keywords, a word behind the speaker's head, one eased punch-in, a
couple of sound effects and the music bed -- and renders it small. No LLM and
no ComfyUI are involved, so it takes a minute or two and can be re-run after
every settings change.

It works on a copy: the project's own timeline is never modified or saved.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import OUTPUT_DIR, PROJECTS_DIR
from store.project_store import ProjectStore
from timeline.schema import Timeline, frame_to_time

from . import facezoom, graphics, placement
from . import sound as sound_stage
from . import text_fx
from .graphics import plain
from .models import Beat, PresentationSettings, WindowZoom
from .program import build_program

logger = logging.getLogger("presentation.effects_sample")

SAMPLE_SECONDS = 30.0
SAMPLE_HEIGHT = 540
# The cut itself: the edited talking head and its own sound. Everything else
# on a timeline is dressing from an earlier pass and is dropped from the copy.
_CUT_TRACKS = {"V1", "A1"}
# Every effect gets one slot across the window (fraction of the window, effect),
# so a single sample shows each of them once.
SAMPLE_SLOTS: Tuple[Tuple[float, str], ...] = (
    (0.03, "knockout"), (0.14, "kinetic_words"), (0.25, "behind_head"),
    (0.37, "statement_card"), (0.53, "pill_labels"), (0.67, "numbered_point"),
    (0.80, "punch"), (0.90, "knockout"),
)
TEXT_FX_SLOTS = ("knockout", "kinetic_words", "behind_head")
MIN_SPACING_S = 3.5


def sample_path(project_id: str) -> Path:
    return Path(OUTPUT_DIR) / f"{project_id}_effects_sample.mp4"


def _speech_start(program, after_s: float, span_s: float = 10.0, min_words: int = 12) -> float:
    """Where the talking actually starts: the first word with a steady run of
    speech after it. A Life3Baje edit opens on ~14 s of silent B-roll, and a
    sample window that started there showed no words to put effects on."""
    words = [w for w in program.words if w.tl_start_s >= after_s]
    for index, word in enumerate(words):
        run = sum(1 for w in words[index:] if w.tl_start_s < word.tl_start_s + span_s)
        if run >= min_words:
            return max(after_s, word.tl_start_s - 0.3)
    return max(after_s, words[0].tl_start_s - 0.3) if words else after_s


def _pick_moments(program, start_s: float, end_s: float, count: int) -> List[Any]:
    """The loudest distinct words inside the window, spaced out, in time order."""
    candidates = [w for w in program.words
                  if start_s + 0.8 <= w.tl_start_s <= end_s - 2.4
                  and len(w.text.strip(".,!?;:\"'")) >= 3]
    candidates.sort(key=lambda w: w.emphasis_z, reverse=True)
    chosen: List[Any] = []
    for word in candidates:
        if all(abs(word.tl_start_s - c.tl_start_s) >= MIN_SPACING_S for c in chosen):
            chosen.append(word)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda w: w.tl_start_s)


def _word_near(program, at_s: float, start_s: float, end_s: float):
    inside = [w for w in program.words if start_s <= w.tl_start_s <= end_s - 1.0]
    return min(inside, key=lambda w: abs(w.tl_start_s - at_s)) if inside else None


def _phrase_at(program, word, n: int) -> str:
    """`n` words from `word` on, the loudest one marked `*like this*`."""
    index = program.words.index(word)
    run = program.words[index:index + n]
    loudest = max(run, key=lambda w: w.emphasis_z) if run else None
    return " ".join(f"*{w.text}*" if w is loudest else w.text for w in run)


def _keywords_after(program, word, n: int) -> List[str]:
    """The `n` longest distinct words of the next few seconds, title-cased."""
    index = program.words.index(word)
    seen, out = set(), []
    for w in sorted(program.words[index:index + 18], key=lambda w: len(w.text), reverse=True):
        key = w.text.strip(".,!?").lower()
        if len(key) >= 4 and key not in seen:
            seen.add(key)
            out.append(w.text.strip(".,!?").title())
        if len(out) >= n:
            break
    return out


def _following_words(program, word, n: int = 3) -> List[str]:
    index = program.words.index(word)
    return [w.text for w in program.words[index:index + n]]


async def render_effects_sample(project_id: str, settings: PresentationSettings,
                                seconds: float = SAMPLE_SECONDS,
                                progress_cb: Optional[Callable[[float, str], None]] = None
                                ) -> Dict[str, Any]:
    """Render the sample to `sample_path(project_id)`; returns what went in."""
    report = progress_cb or (lambda fraction, message="": None)
    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    data = store.get_project(project_id)
    if not data or not data.get("timeline"):
        raise ValueError("This project has no edit yet -- transcribe (and cut) it first.")

    timeline = Timeline.model_validate(data["timeline"]).model_copy(deep=True)
    timeline.items = [i for i in timeline.items if (i.track or "").upper() in _CUT_TRACKS]
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    program = build_program(timeline)
    if not program.words:
        raise ValueError("The edit has no words to build a sample from.")

    start_s = _speech_start(program, frame_to_time(timeline.program_offset_frames or 0, fps_num, fps_den))
    end_s = min(program.duration_s, start_s + max(8.0, float(seconds)))
    source_video = data.get("source_video")
    project_dir = PROJECTS_DIR / project_id
    canvas_w, canvas_h = timeline.width or 1920, timeline.height or 1080

    report(0.1, "Finding the speaker")
    face_anchors: Dict[str, Tuple[float, float]] = {}
    if source_video:
        try:
            face_anchors = dict(await asyncio.to_thread(facezoom.detect_faces, source_video, program) or {})
        except Exception as e:
            logger.info("Effects sample: face detection skipped (%s)", e)

    moments: List[text_fx.Moment] = []
    zooms: List[WindowZoom] = []
    sfx: List[Tuple[float, str]] = []
    beats: List[Beat] = []
    allowed = set(settings.text_fx_styles) if settings.text_fx_styles else None
    span = end_s - start_s
    for fraction, style in SAMPLE_SLOTS:
        word = _word_near(program, start_s + fraction * span, start_s, end_s)
        if word is None:
            continue
        at = max(start_s, word.tl_start_s - 0.15)
        if style == "punch":
            if settings.face_zoom:
                zooms.append(WindowZoom(start_s=at, end_s=at + 2.0,
                                        depth=min(max(0.12, settings.zoom_depth or 0.2),
                                                  facezoom.PUNCH_DEPTH_MAX)))
            continue
        if style in TEXT_FX_SLOTS:
            if not settings.text_fx or (allowed and style not in allowed):
                continue
            face_x, face_y = text_fx._face_anchor(face_anchors, program, word.tl_start_s)
            pos_x, pos_y = text_fx._dodge_face(face_x, face_y)
            moments.append(text_fx.Moment(at, min(end_s, at + text_fx._duration_for(style)), style,
                                          word.text, words=_following_words(program, word),
                                          pos_x=pos_x, pos_y=pos_y, face_x=face_x, face_y=face_y))
            if style == "knockout":
                sfx.append((at, "whoosh"))
            continue
        # The designed graphics, fed with the speaker's own words.
        phrase = _phrase_at(program, word, 9)
        if style == "statement_card":
            beats.append(Beat(id="sample_statement", kind="statement_card", start_s=at,
                              end_s=at + 3.5, planned_duration_s=3.5, topic="effects sample",
                              priority=1.0, text=phrase, data={"with_speaker": True}))
        elif style == "pill_labels":
            items = _keywords_after(program, word, 3)
            if items:
                beats.append(Beat(id="sample_pills", kind="pill_labels", start_s=at, end_s=at + 3.0,
                                  topic="effects sample", priority=1.0, text=items[0],
                                  data={"items": items, "accent_first": True}))
        elif style == "numbered_point":
            beats.append(Beat(id="sample_point", kind="numbered_point", start_s=at, end_s=at + 3.0,
                              topic="effects sample", priority=1.0,
                              text=" ".join(plain(phrase).split()[:5]), data={"number": "1"}))

    report(0.3, "Placing the effects")
    if zooms:
        facezoom.apply_zooms(timeline, [], zooms, face_anchors)
    designed: Dict[str, int] = {}
    if beats:
        drawn, _failed = graphics.render_graphic_assets(beats, project_dir, canvas_w, canvas_h)
        if drawn:
            placement.place_broll(timeline, beats, drawn, program, settings, 0)
            designed["statement_card"] = graphics.place_layout_speakers(timeline, beats, face_anchors)
        designed.update(graphics.place_overlays(timeline, beats, program, project_dir,
                                                placement.broll_windows(timeline), face_anchors))

    from .director import _apply_sound, _regenerate_captions
    if settings.captions:
        _regenerate_captions(timeline, settings, data)

    placed, matte_used, skipped = {}, "off", []
    if moments and settings.text_fx:
        placed, matte_used, skipped = await asyncio.to_thread(
            text_fx.apply_moments, timeline, moments, settings, project_dir, source_video,
            canvas_w, canvas_h)

    report(0.5, "Adding the sound")
    genre = settings.genre or "general"
    if settings.sfx or settings.music:
        # Fills (and re-makes, if they were static) the generated one-shots
        # when ComfyUI is up; a no-op otherwise.
        try:
            from comfyui_bridge import queue_manager
            online = await asyncio.to_thread(queue_manager.client.is_connected, 5.0)
            await sound_stage.warm_generated_cache(settings, genre, 0, comfyui_online=online)
        except Exception as e:
            logger.info("Effects sample: sound generation skipped (%s)", e)
    sound = {}
    if settings.music or settings.sfx:
        sound = _apply_sound(timeline, program, settings, genre, 0,
                             punch_in_times=[z.start_s for z in zooms], extra_sfx=sfx)
    if settings.voice_preset and settings.voice_preset.lower() not in ("off", "none"):
        sound_stage.apply_voice_master(timeline, settings)

    report(0.6, "Rendering the sample")
    from render.runner import render_timeline_async
    out = sample_path(project_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    height = SAMPLE_HEIGHT
    width = int(round(canvas_w * height / canvas_h / 2.0)) * 2
    await render_timeline_async(
        timeline, str(out), prefer_nvenc=True, output_resolution=f"{width}x{height}",
        window=(start_s, end_s),
        progress_callback=lambda f: report(0.6 + 0.4 * max(0.0, min(1.0, f)), "Rendering the sample"))

    return {
        "path": str(out),
        "window": [round(start_s, 2), round(end_s, 2)],
        "text_fx": placed or {},
        "person_matte": matte_used,
        "skipped": skipped or [],
        "zooms": len(zooms),
        "designed": designed,
        "sfx": int((sound or {}).get("sfx", 0)),
        "music": (sound or {}).get("music", ""),
    }
