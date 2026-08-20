"""Stage B — deciding what appears on screen, and writing the prompts for it.

Two model calls, not one. Asking a local model to segment a transcript *and*
write generation prompts in a single answer produces mush: it either segments
well and writes generic prompts, or writes good prompts for three topics and
forgets the rest. Splitting the job means each call has one thing to do and a
worked example of doing it.

**The model plans; this module decides what to believe.** Nothing it returns
reaches the timeline unchecked — times are clamped to the programme, beats
without speech under them are dropped, the density budget caps how much of the
video can be covered, and near-duplicate topics collapse. A model that returns
nonsense yields an empty plan, never a bad edit. That asymmetry is the same one
the fluency pass is built on: a missed opportunity is a blemish, a wrong cutaway
over the speaker's face is a defect.
"""

import hashlib
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .models import Beat, Program, PresentationSettings, ShotPlan, Topic
from .program import transcript_lines

logger = logging.getLogger("presentation.shotplan")

# --- limits, each protecting against something specific --------------------

# Shorter than this and a cutaway registers as a flash rather than a shot.
MIN_BEAT_SECONDS = 2.0
# Pop-ups are punctuation. Any denser and they are decoration.
MIN_POPUP_GAP_SECONDS = 12.0
# Longer than this cannot be read before it leaves the screen.
MAX_POPUP_WORDS = 6
# Two beats about the same thing waste a generation and read as a stutter.
TOPIC_DEDUP_JACCARD = 0.6
# Diffusion models ignore the tail of a very long prompt; keep them tight.
MAX_PROMPT_CHARS = 400
# Ceiling on how many shots one topic is split into. Generous so a long topic
# fills its budget with many images; the coverage budget is the real limiter, so
# this only bounds a single very long topic. Beyond the six named shot angles the
# fallback reuses them with different generation seeds, so images stay distinct.
MAX_SHOT_SLICES = 30
# The cutaway gap and screen-time budget are DERIVED from the settings
# (`PresentationSettings.cutaway_gap_s` / `.broll_budget_seconds`), and
# placement imports them from there too. They were duplicated as constants
# here and in placement.py once — placement silently re-rejected beats this
# module had already budgeted.

# Windowing for the transcript, mirroring the fluency pass: enough context to see
# a whole topic, short enough that a local model does not lose its place.
WINDOW_LINES = 60
OVERLAP_LINES = 8
TOPIC_BATCH = 8


TOPIC_SYSTEM = (
    "You are the researcher for a YouTube video. You are given the transcript of a "
    "finished edit, with a timestamp in seconds at the start of each line.\n\n"
    "The speaker talks in Hindi written in Latin letters (Hinglish), mixing in English "
    "words as Hindi speakers do. A machine wrote this transcript down and romanises "
    "Hindi badly — 'jaz' means judge, 'sabsakraaib' means subscribe, 'phinamaanaa' "
    "means phenomenon, 'ooniversitee' means university. Read past the spelling; the "
    "speaker said these words perfectly.\n\n"
    "Split the transcript into the TOPICS it actually covers. A topic is a thing the "
    "speaker is talking about for a stretch of time — a story, a study, an example, a "
    "piece of advice. Not every sentence starts a new topic; a topic usually runs for "
    "15 to 90 seconds.\n\n"
    "For each topic give:\n"
    "  start_s, end_s  - when it runs, in seconds, taken from the timestamps given\n"
    "  topic           - three to six words naming it, IN ENGLISH\n"
    "  summary         - one sentence in ENGLISH saying what the speaker says about it\n"
    "  visual          - one sentence in ENGLISH describing a single image that would "
    "illustrate it for a viewer. Describe a SCENE, not a concept: 'a student sitting "
    "alone in a lecture hall wearing a bright yellow t-shirt', not 'embarrassment'.\n"
    "  priority        - 0.0 to 1.0, how much this topic would benefit from a picture. "
    "A concrete story or example is high. A general statement or an aside is low.\n\n"
    "RULES:\n"
    "1. Times must come from the timestamps in the transcript and must not overlap.\n"
    "2. Everything you write in topic, summary and visual must be in ENGLISH, however "
    "the transcript is written.\n"
    "3. Never invent a topic the speaker does not discuss.\n"
    "4. Answer with JSON only, no commentary:\n"
    '   {"topics": [{"start_s": 0.0, "end_s": 0.0, "topic": "", "summary": "", '
    '"visual": "", "priority": 0.0}]}'
)


BEAT_SYSTEM = (
    "You are the art director for a YouTube video. You are given a list of topics from "
    "the video, each with a summary and a suggested visual. For each topic you decide "
    "what appears on screen and write the prompt that will generate it.\n\n"
    "You are writing prompts for a text-to-image model. It does not know what the video "
    "is about. It only sees your prompt.\n\n"
    "For each topic return one beat:\n"
    "  kind         - 'broll_image' for a still, 'broll_video' for a moving shot, or "
    "'popup' when the point is better made by a few words on screen than by a picture.\n"
    "  image_prompt - for broll_image. Describe the SHOT: subject, setting, lighting, "
    "lens or medium. 20 to 40 words. English only. No text, no words, no logos in the "
    "image — the model renders them as gibberish.\n"
    "  video_prompt - for broll_video. As above, plus ONE simple camera or subject "
    "motion ('slow push in', 'leaves drifting past'). Nothing complex; a short clip "
    "cannot carry a complicated action.\n"
    "  popup_text   - for popup. AT MOST SIX WORDS naming the idea. English, or the "
    "speaker's own English term if they used one. No full stop at the end.\n"
    "  negative_prompt - what must not appear. Always include 'text, watermark, logo'.\n"
    "  style_hint   - 'photoreal', 'illustration', 'diagram' or 'abstract'.\n\n"
    "RULES:\n"
    "1. Prefer 'broll_image'. Use 'broll_video' only when motion is the point of the "
    "shot. Use 'popup' for abstract ideas, numbers, names and definitions, which "
    "generated pictures render badly.\n"
    "2. Never put readable text in an image prompt.\n"
    "3. Never describe a real identifiable person.\n"
    "4. Keep every prompt under 40 words.\n"
    "5. Answer with JSON only, no commentary:\n"
    '   {"beats": [{"topic": "", "kind": "", "image_prompt": null, '
    '"video_prompt": null, "popup_text": null, "negative_prompt": "", '
    '"style_hint": ""}]}\n\n'
    "WORKED EXAMPLE:\n"
    "Topic: Cornwall University experiment | A psychology study where students had to "
    "wear an embarrassing t-shirt and walk into a full room.\n"
    'Beat: {"topic": "Cornwall University experiment", "kind": "broll_image", '
    '"image_prompt": "A young student pausing in the doorway of a crowded university '
    'lecture hall, wearing a bright yellow t-shirt, other students seated and looking '
    'down at notes, warm afternoon light through tall windows, 35mm documentary '
    'photograph, shallow depth of field", "video_prompt": null, "popup_text": null, '
    '"negative_prompt": "text, watermark, logo, deformed hands, extra limbs", '
    '"style_hint": "photoreal"}\n\n'
    "Note what happened: the abstract idea (embarrassment) became a concrete scene a "
    "camera could photograph, and nothing in the image needs to be read."
)


THUMBNAIL_TITLE_SYSTEM = (
    "Write a YouTube thumbnail title for this video: at most five words, in the "
    "language the speaker is using, in capitals, no punctuation. It must promise what "
    "the video actually delivers — never invent a claim the transcript does not make. "
    "Answer with the title only."
)


# The answer shapes, sent to the server so it constrains generation rather than
# relying on the prompt alone. Each call must send its OWN schema: LM Studio
# honours whatever it is given, so a caller that inherits somebody else's schema
# gets a perfectly valid 200 containing entirely the wrong object.
TOPIC_SCHEMA = {
    "name": "topics",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_s": {"type": "number"},
                        "end_s": {"type": "number"},
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "visual": {"type": "string"},
                        "priority": {"type": "number"},
                    },
                    "required": ["start_s", "end_s", "topic", "summary", "visual",
                                 "priority"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topics"],
        "additionalProperties": False,
    },
}

BEAT_SCHEMA = {
    "name": "beats",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "kind": {"type": "string",
                                 "enum": ["broll_image", "broll_video", "popup"]},
                        "image_prompt": {"type": ["string", "null"]},
                        "video_prompt": {"type": ["string", "null"]},
                        "popup_text": {"type": ["string", "null"]},
                        "negative_prompt": {"type": "string"},
                        "style_hint": {"type": "string"},
                    },
                    "required": ["topic", "kind", "image_prompt", "video_prompt",
                                 "popup_text", "negative_prompt", "style_hint"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["beats"],
        "additionalProperties": False,
    },
}

# (system_prompt, user_prompt, schema) -> the model's text
AskJson = Callable[..., Awaitable[Optional[str]]]


# --- parsing ---------------------------------------------------------------

def first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in an answer, tolerating prose and code fences.

    Small models wrap their answers in explanation however firmly they are told
    not to. Refusing those answers costs real content for no benefit.
    """
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    depth = 0
    start = -1
    for index, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


async def _ask(ask: AskJson, system: str, user: str,
               schema: Optional[Dict[str, Any]]) -> Optional[str]:
    """Call the model, passing a schema only if the caller can accept one.

    Tests hand in a two-argument stand-in; the real client takes a schema. Trying
    with the schema and retrying without keeps both working without every test
    having to know about response formats.
    """
    try:
        return await ask(system, user, schema)
    except TypeError:
        return await ask(system, user)


def _clean_prompt(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    flat = " ".join(str(text).split())
    if len(flat) <= MAX_PROMPT_CHARS:
        return flat or None
    trimmed = flat[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]
    return trimmed or None


def _clean_popup(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    words = str(text).split()
    if not words:
        return None
    return " ".join(words[:MAX_POPUP_WORDS]).rstrip(".,;:")


def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^\w]+", (text or "").lower()) if len(t) > 2}


def _similar(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- validation ------------------------------------------------------------

def validate_plan(beats: List[Beat], program: Program,
                  settings: PresentationSettings) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Keep only the beats that are safe to act on.

    Returns (kept, dropped) where each dropped entry carries the reason, so the
    report can say what the model wanted and why it did not happen.
    """
    candidates, dropped = sanitize_beats(beats, program, settings)
    unique, dup_dropped = dedup_beats(candidates)
    kept, budget_dropped = budget_beats(unique, program, settings)
    return kept, dropped + dup_dropped + budget_dropped


def sanitize_beats(beats: List[Beat], program: Program,
                   settings: PresentationSettings) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Clamp, type-check and field-check every beat before anything trusts it."""
    dropped: List[Dict[str, str]] = []
    candidates: List[Beat] = []

    for beat in beats:
        start = max(0.0, min(float(beat.start_s), program.duration_s))
        end = max(0.0, min(float(beat.end_s), program.duration_s))
        if end <= start:
            dropped.append({"topic": beat.topic, "reason": "times are empty or reversed"})
            continue
        if end - start < MIN_BEAT_SECONDS:
            dropped.append({"topic": beat.topic, "reason": "shorter than the minimum beat"})
            continue
        if not program.has_speech_between(start, end):
            dropped.append({"topic": beat.topic, "reason": "no speech in that span"})
            continue

        beat = beat.model_copy(update={"start_s": start, "end_s": end})

        if beat.kind not in ("broll_image", "broll_video", "popup", "graphic"):
            dropped.append({"topic": beat.topic, "reason": f"unknown kind {beat.kind!r}"})
            continue

        # A video beat with no video workflow configured is better served as a
        # still than dropped — the topic still deserves a picture.
        if beat.kind == "broll_video" and not settings.broll_video:
            beat = beat.model_copy(update={"kind": "broll_image"})

        beat = beat.model_copy(update={
            "image_prompt": _clean_prompt(beat.image_prompt),
            "video_prompt": _clean_prompt(beat.video_prompt),
            "popup_text": _clean_popup(beat.popup_text),
            "priority": max(0.0, min(1.0, float(beat.priority or 0.5))),
        })

        if beat.kind == "broll_image" and not beat.image_prompt:
            dropped.append({"topic": beat.topic, "reason": "no image prompt"})
            continue
        if beat.kind == "broll_video" and not (beat.video_prompt or beat.image_prompt):
            dropped.append({"topic": beat.topic, "reason": "no video prompt"})
            continue
        if beat.kind == "popup" and not beat.popup_text:
            dropped.append({"topic": beat.topic, "reason": "no pop-up text"})
            continue
        if beat.kind == "graphic" and not settings.graphics:
            dropped.append({"topic": beat.topic, "reason": "graphics are switched off"})
            continue

        candidates.append(beat)

    return candidates, dropped


def dedup_beats(candidates: List[Beat]) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Collapse near-duplicate topics, keeping the higher priority one.

    Runs BEFORE beat multiplication — sub-beats of one long topic deliberately
    share a topic name, and deduping after would collapse them straight back
    into one shot.
    """
    dropped: List[Dict[str, str]] = []
    unique: List[Beat] = []
    for beat in sorted(candidates, key=lambda b: -b.priority):
        clash = next((k for k in unique
                      if _similar(k.topic, beat.topic) >= TOPIC_DEDUP_JACCARD), None)
        if clash is not None:
            dropped.append({"topic": beat.topic,
                            "reason": f"duplicate of {clash.topic!r}"})
            continue
        unique.append(beat)
    return unique, dropped


def budget_beats(unique: List[Beat], program: Program,
                 settings: PresentationSettings) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Accept beats until the coverage budget is spent.

    The budget is SECONDS of screen time (`target_coverage` × programme), not a
    per-minute count — a count budget with one beat per topic capped coverage
    near 20% whatever the settings said. Highest priority first, so when the
    video is busier than the budget allows, what survives is what the model
    cared about most. Each accepted cutaway carries its `planned_duration_s`;
    placement must honour it rather than re-deciding.
    """
    dropped: List[Dict[str, str]] = []
    gap = settings.cutaway_gap_s
    budget = settings.broll_budget_seconds(program.duration_s)
    max_cutaways = None
    if settings.max_broll_per_minute:
        minutes = max(1.0, program.duration_s / 60.0)
        max_cutaways = max(1, int(round(minutes * settings.max_broll_per_minute)))

    # Two separate passes: pop-ups have their own gap rule and must not starve
    # because priority-sorted iteration interleaved them with cutaways.
    kept_cutaways: List[Beat] = []
    spent = 0.0
    for beat in sorted([b for b in unique if b.is_cutaway or b.kind == "graphic"],
                       key=lambda b: -b.priority):
        # The shot must leave the on-camera gap INSIDE its own span, or two
        # adjacent shots (multiplication slices are contiguous by construction)
        # would always crowd each other out of the spacing check.
        planned = max(settings.broll_seconds_min,
                      min(settings.broll_seconds_max, beat.duration_s - gap))
        planned = min(planned, max(MIN_BEAT_SECONDS, beat.duration_s))
        if spent + planned > budget + 0.001:
            dropped.append({"topic": beat.topic, "reason": "over the coverage budget"})
            continue
        if max_cutaways is not None and len(kept_cutaways) >= max_cutaways:
            dropped.append({"topic": beat.topic, "reason": "over the cutaway budget"})
            continue
        if _too_close(beat, planned, kept_cutaways, gap):
            dropped.append({"topic": beat.topic, "reason": "too close to another cutaway"})
            continue
        kept_cutaways.append(beat.model_copy(update={"planned_duration_s": round(planned, 3)}))
        spent += planned

    kept_popups: List[Beat] = []
    for beat in sorted([b for b in unique if not (b.is_cutaway or b.kind == "graphic")],
                       key=lambda b: -b.priority):
        if _too_close(beat, beat.duration_s, kept_popups, MIN_POPUP_GAP_SECONDS):
            dropped.append({"topic": beat.topic, "reason": "too close to another pop-up"})
            continue
        kept_popups.append(beat)

    kept = sorted(kept_cutaways + kept_popups, key=lambda b: b.start_s)
    for index, beat in enumerate(kept):
        beat.id = f"b{index:02d}"
    return kept, dropped


def _planned_length(beat: Beat) -> float:
    # Spacing is measured on what the beat will occupy ON SCREEN, never on the
    # topic span — consecutive topics are adjacent by definition, and judging by
    # span rejected two of three beats on a three-topic video.
    return beat.planned_duration_s or beat.duration_s


def _too_close(beat: Beat, planned: float, placed: List[Beat], gap: float) -> bool:
    # The 0.01s epsilon keeps back-to-back multiplication slices, whose spacing
    # equals the gap exactly by construction, from failing on float rounding.
    start, end = beat.start_s, beat.start_s + planned
    for other in placed:
        other_start = other.start_s
        other_end = other_start + _planned_length(other)
        if start < other_end + gap - 0.01 and other_start < end + gap - 0.01:
            return True
    return False


# --- the model calls -------------------------------------------------------

async def find_topics(program: Program, ask: AskJson) -> List[Topic]:
    """Ask the model what the video is about, one window at a time."""
    lines = transcript_lines(program).splitlines()
    if not lines:
        return []

    topics: List[Topic] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + WINDOW_LINES)
        window = "\n".join(lines[start:end])
        answer = await _ask(ask, TOPIC_SYSTEM, f"Transcript:\n{window}\n\nTopics:",
                            TOPIC_SCHEMA)
        parsed = first_json_object(answer or "")
        if not parsed:
            logger.warning("Topics: no usable answer for lines %d-%d", start, end)
        else:
            for raw in parsed.get("topics", []) or []:
                try:
                    topics.append(Topic(
                        start_s=float(raw.get("start_s", 0.0)),
                        end_s=float(raw.get("end_s", 0.0)),
                        topic=str(raw.get("topic", "")).strip(),
                        summary=str(raw.get("summary", "")).strip(),
                        visual=str(raw.get("visual", "")).strip(),
                        priority=float(raw.get("priority", 0.5)),
                    ))
                except (TypeError, ValueError):
                    continue
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)

    topics = [t for t in topics if t.topic and t.end_s > t.start_s]
    topics.sort(key=lambda t: t.start_s)
    logger.info("Topics: %d found", len(topics))
    return topics


async def write_beats(topics: List[Topic], ask: AskJson) -> List[Beat]:
    """Turn topics into on-screen beats with generation prompts."""
    beats: List[Beat] = []
    by_topic = {t.topic.lower(): t for t in topics}

    for start in range(0, len(topics), TOPIC_BATCH):
        batch = topics[start:start + TOPIC_BATCH]
        listing = "\n".join(
            f"{i + 1}. {t.topic} | {t.summary} | suggested visual: {t.visual}"
            for i, t in enumerate(batch))
        answer = await _ask(ask, BEAT_SYSTEM, f"Topics:\n{listing}\n\nBeats:",
                            BEAT_SCHEMA)
        parsed = first_json_object(answer or "")
        if not parsed:
            logger.warning("Beats: no usable answer for topics %d-%d",
                           start, start + len(batch))
            continue

        for index, raw in enumerate(parsed.get("beats", []) or []):
            # Match the beat back to its topic by name, falling back to position:
            # models reorder and rename, and a beat with no time is useless.
            name = str(raw.get("topic", "")).strip()
            topic = by_topic.get(name.lower())
            if topic is None:
                best = max(batch, key=lambda t: _similar(t.topic, name), default=None)
                topic = best if best and _similar(best.topic, name) >= 0.34 else None
            if topic is None and index < len(batch):
                topic = batch[index]
            if topic is None:
                continue

            kind = str(raw.get("kind", "broll_image")).strip().lower()
            beats.append(Beat(
                start_s=topic.start_s,
                end_s=topic.end_s,
                topic=topic.topic,
                summary=topic.summary,
                kind=kind if kind in ("broll_image", "broll_video", "popup", "graphic")
                     else "broll_image",
                priority=topic.priority,
                image_prompt=raw.get("image_prompt") or topic.visual or None,
                video_prompt=raw.get("video_prompt"),
                popup_text=raw.get("popup_text"),
                negative_prompt=str(raw.get("negative_prompt")
                                    or "text, watermark, logo, deformed hands, blurry"),
                style_hint=(str(raw.get("style_hint", "photoreal")).strip().lower()
                            if str(raw.get("style_hint", "")).strip().lower()
                            in ("photoreal", "illustration", "diagram", "abstract")
                            else "photoreal"),
            ))
    return beats


# --- beat multiplication ----------------------------------------------------

# Deterministic shot-type variations, used when the model cannot be asked for
# distinct shots. Appended to the base prompt so each sub-beat generates a
# different image of the same subject.
SHOT_VARIATIONS = [
    "wide establishing shot",
    "medium shot, subject in motion",
    "close-up on a telling detail",
    "over-the-shoulder perspective",
    "low angle, dramatic composition",
    "high angle overview of the scene",
]

VARIATION_SYSTEM = (
    "You are the art director for a YouTube video. You are given ONE scene "
    "description and a number N. Write N distinct text-to-image prompts showing the "
    "same subject as N different shots — for example a wide establishing shot, a "
    "medium shot of the action, a close-up on a telling detail. Keep the same world, "
    "style and lighting across all of them so they cut together as one sequence.\n"
    "Each prompt: 20 to 40 words, English only, no readable text, no logos, no real "
    "identifiable people.\n"
    'Answer with JSON only: {"prompts": ["...", "..."]}'
)

VARIATION_SCHEMA = {
    "name": "variations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
        "required": ["prompts"],
        "additionalProperties": False,
    },
}


async def multiply_beats(beats: List[Beat], settings: PresentationSettings,
                         ask: Optional[AskJson]) -> List[Beat]:
    """Split each long cutaway topic into several distinct shots.

    One beat per topic is a hard ceiling near two cutaways a minute — a 60s
    story illustrated by one 7s still leaves the speaker bare for 53s, and no
    budget can fix what the planner never proposed. Runs after dedup (siblings
    share a topic name and would collapse) and before budgeting (the budget
    must see everything it might spend on).
    """
    # Tile each long topic into cells of one image plus its on-camera gap, so the
    # slices fall exactly where placement's spacing wants them and every one
    # survives. Sizing by a bare factor of the shot length instead made slices
    # placement then dropped as "too close", shipping ~half the images at ~40%
    # coverage. `floor` keeps every slice at least a full cell — a shorter slice
    # gets its length clamped back up past the gap and collides with its neighbour.
    cell = settings.avg_broll_seconds + settings.cutaway_gap_s
    out: List[Beat] = []
    for beat in beats:
        span = beat.duration_s
        count = int(span // cell) if cell > 0 else 1
        if not beat.is_cutaway or count < 2 or not beat.image_prompt:
            out.append(beat)
            continue
        count = min(MAX_SHOT_SLICES, count)
        prompts: List[str] = []
        if ask is not None:
            try:
                answer = await _ask(
                    ask, VARIATION_SYSTEM,
                    f"Scene: {beat.image_prompt}\nN: {count}\n\nPrompts:",
                    VARIATION_SCHEMA)
                parsed = first_json_object(answer or "") or {}
                prompts = [_clean_prompt(p) for p in parsed.get("prompts", []) or []]
                prompts = [p for p in prompts if p][:count]
            except Exception as e:
                logger.warning("Shot variation call failed (%s); using suffixes", e)
        if len(prompts) < count:
            # Deterministic fallback: the base scene from different shot types.
            prompts = [f"{beat.image_prompt}, {SHOT_VARIATIONS[i % len(SHOT_VARIATIONS)]}"
                       for i in range(count)]

        slice_s = span / count
        for index in range(count):
            sub = beat.model_copy(update={
                "start_s": round(beat.start_s + index * slice_s, 3),
                "end_s": round(beat.start_s + (index + 1) * slice_s, 3),
                "image_prompt": _clean_prompt(prompts[index]) or beat.image_prompt,
                # Only the first slice may stay a video shot; generating N
                # videos of one topic is minutes of GPU for seconds of screen.
                "kind": beat.kind if index == 0 else "broll_image",
                "video_prompt": beat.video_prompt if index == 0 else None,
            })
            out.append(sub)
        logger.info("Topic %r (%.0fs) split into %d shots", beat.topic, span, count)

    return out


# --- the deterministic fallback --------------------------------------------

STOP_WORDS = {
    "the", "and", "that", "this", "with", "have", "from", "they", "will", "what",
    "when", "your", "which", "there", "their", "would", "about", "could", "been",
    "hai", "hain", "aur", "kya", "yah", "vah", "mein", "men", "kar", "rahe", "ko",
    "ki", "ka", "ke", "se", "par", "bhi", "hee", "toh", "dosto", "aap", "aapako",
}


def fallback_plan(program: Program, settings: PresentationSettings) -> List[Beat]:
    """Beats without a language model: the loudest phrase in each window.

    Deliberately worse than the planned version and deliberately still
    shippable — a night where LM Studio failed to load should not produce a bare
    talking head.
    """
    if not program.words:
        return []
    # Sized from the coverage target like the planned path — the old clamp of
    # two beats a minute left a fallback night at ~16% coverage.
    wanted = max(1, int(round(settings.broll_budget_seconds(program.duration_s)
                              / settings.avg_broll_seconds)))
    window = program.duration_s / wanted

    beats: List[Beat] = []
    for index in range(wanted):
        lo = index * window
        hi = lo + window
        candidates = program.words_between(lo, hi)
        if len(candidates) < 5:
            continue
        # The most emphatic run of five words in this stretch.
        best_at = max(range(len(candidates) - 4),
                      key=lambda i: sum(w.emphasis_z for w in candidates[i:i + 5]))
        run = candidates[best_at:best_at + 5]
        keywords = [w.text.strip(".,!?").lower() for w in run
                    if len(w.text.strip(".,!?")) > 3
                    and w.text.strip(".,!?").lower() not in STOP_WORDS]
        if not keywords:
            continue
        subject = " ".join(keywords[:3])
        beats.append(Beat(
            start_s=run[0].tl_start_s,
            end_s=min(program.duration_s,
                      run[0].tl_start_s + settings.broll_seconds_max),
            topic=subject,
            summary="",
            kind="broll_image",
            priority=0.5,
            image_prompt=(f"Cinematic documentary photograph representing {subject}, "
                          f"natural lighting, 35mm film still, shallow depth of field"),
        ))
    return beats


async def plan_shots(program: Program, settings: PresentationSettings,
                     ask: Optional[AskJson]) -> ShotPlan:
    """The whole of stage B: topics, beats, validation — or the fallback."""
    beats: List[Beat] = []
    source = "llm"

    if ask is not None:
        try:
            topics = await find_topics(program, ask)
            if topics:
                beats = await write_beats(topics, ask)
        except Exception as e:
            logger.warning("Shot planning failed (%s); falling back to keywords", e)
            beats = []

    if not beats:
        source = "fallback"
        beats = fallback_plan(program, settings)

    candidates, dropped = sanitize_beats(beats, program, settings)
    unique, dup_dropped = dedup_beats(candidates)
    if source == "llm":
        # The fallback already proposes one beat per shot slot; only topic-sized
        # LLM beats need splitting into a sequence of shots.
        unique = await multiply_beats(unique, settings, ask)
    kept, budget_dropped = budget_beats(unique, program, settings)
    dropped = dropped + dup_dropped + budget_dropped

    if not settings.popups:
        kept = [b for b in kept if b.kind != "popup"]
    if not settings.broll:
        kept = [b for b in kept if not b.is_cutaway]

    logger.info("Shot plan (%s): %d beats kept, %d dropped", source, len(kept), len(dropped))
    return ShotPlan(beats=kept, dropped=dropped, source=source)


def seed_for(project_id: str, extra: str = "") -> int:
    """A stable seed, so two runs of one project make the same decisions."""
    digest = hashlib.sha1(f"{project_id}:{extra}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
