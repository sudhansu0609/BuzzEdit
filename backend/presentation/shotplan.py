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

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from . import character as character_mod
from . import genre as genre_mod
from .models import BEAT_KINDS, Beat, Character, Program, PresentationSettings, ShotPlan, Topic
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


THUMBNAIL_CONCEPT_SYSTEM = (
    "You design the YouTube thumbnail for this video. Read the transcript excerpts and "
    "the topic list, work out the ONE core idea or payoff the whole video is about, and "
    "answer with exactly two lines:\n"
    "TITLE: at most five words, in the language the speaker is using, in capitals, no "
    "punctuation. It names that core idea and must promise only what the video actually "
    "delivers -- never invent a claim the transcript does not make, never a generic "
    "phrase that would fit any video.\n"
    "SCENE: one sentence for an image generator: a single clear subject doing something "
    "that shows that core idea literally (the specific objects, place and emotion the "
    "speaker talks about), close framing, strong contrast. No text, no letters, no logos "
    "in the scene -- the title is added on top later.\n"
    "Answer with the two lines only."
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

# BEAT_SCHEMA plus the per-beat main-character flag, sent only when the story
# has a lead -- a video without one keeps the plain schema.
BEAT_SCHEMA_WITH_CHARACTER = json.loads(json.dumps(BEAT_SCHEMA))
_beat_item = BEAT_SCHEMA_WITH_CHARACTER["schema"]["properties"]["beats"]["items"]
_beat_item["properties"]["shows_main_character"] = {"type": "boolean"}
_beat_item["required"].append("shows_main_character")

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


# How many planning calls run at once. The calls within one pass are
# independent (each topic batch, each scene to vary); run one after another
# they were ~12 round trips and 4+ minutes of the pass. Four matches LM
# Studio's default parallel slots and is well inside a remote API's limits.
PLAN_CALL_CONCURRENCY = 4


async def _ask_all(ask: AskJson, calls: List[Tuple[str, str, Any]],
                   limit: int = PLAN_CALL_CONCURRENCY) -> List[Optional[str]]:
    """Every (system, user, schema) call at most `limit` at a time; answers
    in call order, None where a call raised."""
    gate = asyncio.Semaphore(max(1, limit))

    async def one(system: str, user: str, schema: Any) -> Optional[str]:
        async with gate:
            try:
                return await _ask(ask, system, user, schema)
            except Exception as e:
                logger.warning("Planning call failed (%s)", e)
                return None

    return list(await asyncio.gather(*(one(*call) for call in calls)))


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

        if beat.kind not in BEAT_KINDS:
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
        if beat.kind == "map" and not beat.place:
            dropped.append({"topic": beat.topic, "reason": "map with no place"})
            continue
        if beat.kind == "split" and not (beat.data.get("left") and beat.data.get("right")):
            dropped.append({"topic": beat.topic, "reason": "split with one side"})
            continue
        if beat.kind == "chart" and not (beat.data.get("values") and beat.data.get("labels")):
            dropped.append({"topic": beat.topic, "reason": "chart with no data"})
            continue
        if beat.kind == "newspaper" and not settings.newspapers:
            dropped.append({"topic": beat.topic, "reason": "newspapers are switched off"})
            continue
        if beat.kind == "newspaper" and not (beat.data.get("headline") or (beat.text or "").strip()):
            dropped.append({"topic": beat.topic, "reason": "newspaper with no headline"})
            continue
        if beat.kind == "case_file" and not settings.case_files:
            dropped.append({"topic": beat.topic, "reason": "case files are switched off"})
            continue
        if beat.kind == "case_file" and not (
                beat.data.get("fields") or beat.data.get("title") or (beat.text or "").strip()):
            dropped.append({"topic": beat.topic, "reason": "case file with no content"})
            continue
        if beat.is_text and not (beat.text or "").strip():
            dropped.append({"topic": beat.topic, "reason": f"{beat.kind} with no text"})
            continue

        candidates.append(beat)

    return candidates, dropped


# --- claim beats that read as a newspaper clipping ---------------------------
#
# Newspapers used to come only from entities.py's dated-event beats. Genres
# whose text-fx palette reaches for a highlighter sweep (text_fx.palette_for)
# read a claim, a stat or a quote as a clipping too, so some of those cards
# become full-frame newspaper cutaways instead — the highlighter sweep then
# has something to land on beyond the handful of dated events.
_CLAIM_KINDS: Tuple[str, ...] = ("stat_callout", "quote_card", "definition_card")
_CLAIM_CUE_RE = re.compile(
    r"\b(research|study|studies|studied|report|reports|reported|according to|"
    r"survey|surveys|scientists?|researchers?)\b", re.IGNORECASE)
NEWSPAPER_CONVERT_SECONDS = 4.5


def _claim_worthy(beat: Beat) -> bool:
    """A stat/quote/definition beat that reads as an assertion worth clipping:
    a quote always does, a stat always does (it is a number by definition),
    otherwise it needs a number or a research/report cue of its own — or an
    explicit `data["claim"]` flag from whatever wrote the beat."""
    if beat.kind not in _CLAIM_KINDS:
        return False
    if beat.kind in ("quote_card", "stat_callout"):
        return True
    text = f"{beat.text or ''} {beat.subtext or ''}"
    if re.search(r"\d", text) or _CLAIM_CUE_RE.search(text):
        return True
    return bool(isinstance(beat.data, dict) and beat.data.get("claim"))


def _newspaper_fields(beat: Beat) -> Dict[str, Optional[str]]:
    """Headline + a short highlight phrase, both drawn from the beat's own
    spoken text. The highlight must land inside a single wrapped headline
    line (presentation.newspaper.render_newspaper) to get a highlight box at
    all, so it stays to the phrase's first few words rather than the whole
    thing."""
    phrase = (beat.text or beat.topic or "").strip().strip("“”\"'").strip()
    words = phrase.split()
    headline = " ".join(words[:9]).upper() if words else "BREAKING NEWS"
    highlight = " ".join(words[:4]) if words else None
    return {"headline": headline, "highlight": highlight}


def newspaper_conversion_gap_s(settings: PresentationSettings) -> float:
    """The minimum spacing this pass enforces between newspaper cutaways it
    creates: one every 90s at calm/balanced density, one every 45s at
    busy/max — mirroring the busy/max split PresentationSettings.crowd_gap_s
    already makes, just at a newspaper's much coarser scale."""
    if (settings.density or "").strip().lower() in ("busy", "max"):
        return 45.0
    return 90.0


def convert_claims_to_newspapers(beats: List[Beat], program: Program,
                                 settings: PresentationSettings, genre: str,
                                 genre_secondary: Optional[str] = None) -> List[Beat]:
    """Some claim/stat/quote beats become newspaper cutaways instead of text
    cards, in the genres whose blended text-fx palette includes
    "newspaper_highlight" (presentation.text_fx.palette_for — primary genre at
    weight 3, `genre_secondary` at weight 1). A no-op everywhere else, and
    when `settings.newspapers` is off.

    Density-capped (`newspaper_conversion_gap_s`) and additionally spaced at
    least `settings.crowd_gap_s` from any newspaper beat already in the plan —
    a run of claims must not turn into a flip-book of clippings.
    """
    if not settings.newspapers or not beats:
        return beats
    from . import text_fx as text_fx_mod
    palette = text_fx_mod.palette_for(genre, genre_secondary)
    if not any(style == "newspaper_highlight" for style, _weight in palette):
        return beats

    min_gap = max(newspaper_conversion_gap_s(settings), settings.crowd_gap_s)
    placed_times = sorted(b.start_s for b in beats if b.kind == "newspaper")

    out: List[Beat] = []
    for beat in sorted(beats, key=lambda b: b.start_s):
        if beat.kind == "newspaper" or not _claim_worthy(beat) or not (beat.text or "").strip():
            out.append(beat)
            continue
        if any(abs(beat.start_s - t) < min_gap for t in placed_times):
            out.append(beat)
            continue
        fields = _newspaper_fields(beat)
        if not fields["highlight"]:
            out.append(beat)
            continue
        end_s = min(program.duration_s, beat.start_s + NEWSPAPER_CONVERT_SECONDS)
        if end_s <= beat.start_s:
            out.append(beat)
            continue
        data = dict(beat.data or {})
        data.update(headline=fields["headline"], highlight=fields["highlight"])
        out.append(beat.model_copy(update={
            "kind": "newspaper", "end_s": end_s, "text": fields["headline"],
            "subtext": None, "data": data, "priority": max(beat.priority, 0.72),
        }))
        placed_times.append(beat.start_s)

    return out


def dedup_beats(candidates: List[Beat]) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Collapse near-duplicate topics, keeping the higher priority one.

    Runs BEFORE beat multiplication — sub-beats of one long topic deliberately
    share a topic name, and deduping after would collapse them straight back
    into one shot.
    """
    dropped: List[Dict[str, str]] = []
    unique: List[Beat] = []
    for beat in sorted(candidates, key=lambda b: -b.priority):
        if beat.origin != "llm" or beat.is_text:
            unique.append(beat)
            continue
        clash = next((k for k in unique
                      if k.origin == "llm" and not k.is_text
                      and _similar(k.topic, beat.topic) >= TOPIC_DEDUP_JACCARD), None)
        if clash is not None:
            dropped.append({"topic": beat.topic,
                            "reason": f"duplicate of {clash.topic!r}"})
            continue
        unique.append(beat)
    return unique, dropped


def _ensure_coverage_candidates(unique: List[Beat], program: Program,
                                settings: PresentationSettings, genre: str,
                                genre_secondary: Optional[str] = None) -> List[Beat]:
    """Top up cutaway candidates so there is enough screen time on offer to
    spend the whole coverage budget: beats x avg seconds >= coverage x
    duration. However correctly the budget itself is sized, it can only ever
    keep what it is offered — an LLM plan that proposes too few beats still
    ships under target no matter how generous the budget math is.
    """
    budget = settings.broll_budget_seconds(program.duration_s)
    have = sum((b.planned_duration_s or min(settings.broll_seconds_max, b.duration_s))
              for b in unique if b.is_cutaway)
    if have >= budget:
        return unique
    filler = fallback_plan(program, settings, genre, genre_secondary)
    if not filler:
        return unique
    covered = [(b.start_s, b.end_s) for b in unique if b.is_cutaway]
    extra: List[Beat] = []
    for beat in filler:
        if any(beat.start_s < e and s < beat.end_s for s, e in covered):
            continue
        # Never outrank what the model itself proposed — this is a top-up for
        # unclaimed seconds, not a competitor for the budget's best slots.
        beat = beat.model_copy(update={"priority": min(beat.priority, 0.35),
                                       "origin": "fallback"})
        extra.append(beat)
        covered.append((beat.start_s, beat.end_s))
    if extra:
        logger.info("Shot plan: %d filler beats added to reach the coverage budget",
                    len(extra))
    return unique + extra


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
        # A stage direction is the user asking for this picture here, by name.
        # It is exempt from the coverage budget and keeps only the minimum
        # on-camera gap; it still cannot sit on top of another cutaway.
        requested = beat.origin == "script"
        # A direction that timed itself (`seconds=`) runs exactly that long and
        # may sit back to back with the next one: the caller already spaced them.
        held = requested and bool(beat.data.get("hold")) and bool(beat.planned_duration_s)
        if held:
            planned = beat.planned_duration_s
        if not requested and spent + planned > budget + 0.001:
            dropped.append({"topic": beat.topic, "reason": "over the coverage budget"})
            continue
        if not requested and max_cutaways is not None and len(kept_cutaways) >= max_cutaways:
            dropped.append({"topic": beat.topic, "reason": "over the cutaway budget"})
            continue
        if _too_close(beat, planned, kept_cutaways,
                      0.0 if held else settings.min_oncamera_gap_s if requested else settings.crowd_gap_s):
            dropped.append({"topic": beat.topic, "reason": "too close to another cutaway"})
            continue
        kept_cutaways.append(beat.model_copy(update={"planned_duration_s": round(planned, 3)}))
        spent += planned

    kept_popups: List[Beat] = []
    for beat in sorted([b for b in unique if b.kind == "popup"],
                       key=lambda b: -b.priority):
        if _too_close(beat, beat.duration_s, kept_popups, MIN_POPUP_GAP_SECONDS):
            dropped.append({"topic": beat.topic, "reason": "too close to another pop-up"})
            continue
        kept_popups.append(beat)

    # Text cards (chapter titles, stat call-outs, …) cost no coverage budget and
    # have their own spacing rules in placement.
    cards = [b for b in unique if b.is_text]

    kept = sorted(kept_cutaways + kept_popups + cards, key=lambda b: b.start_s)
    for index, beat in enumerate(kept):
        beat.id = f"b{index:02d}"
    return kept, dropped


def balance_video_share(beats: List[Beat], settings: PresentationSettings) -> List[Beat]:
    """Give the most important cutaways motion, at the configured share.

    A generated video clip costs minutes where a still costs seconds, so which
    beats get one cannot be left to the model's whim: the HIGHEST-priority beats
    — the parts of the video the planner cared about most — are promoted to
    `broll_video` until video makes up ~`video_broll_share` of B-roll screen
    time, and any excess the model proposed is demoted lowest-priority-first.
    Share is measured in seconds, not beat count; with ~2.5s beats the landing
    is within one beat of the target, which is as exact as it can be.
    """
    share = settings.video_broll_share
    if not settings.broll_video or share <= 0:
        return beats

    out = list(beats)
    cutaway_idx = [i for i, b in enumerate(out) if b.is_cutaway]
    total = sum(_planned_length(out[i]) for i in cutaway_idx)
    if total <= 0:
        return beats
    video_secs = sum(_planned_length(out[i]) for i in cutaway_idx
                     if out[i].kind == "broll_video")

    # Demote the model's excess, least-important first. The tolerance keeps a
    # plan that is already roughly right from churning.
    ceiling = min(1.0, share + 0.05)
    # Stage directions name their kind ([broll:] is a still, [video:] a clip)
    # and are neither promoted nor demoted; maps and charts are drawn locally
    # and have no video form.
    movable = [i for i in cutaway_idx if out[i].origin == "llm" and out[i].is_generated]
    for i in sorted((i for i in movable if out[i].kind == "broll_video"),
                    key=lambda i: out[i].priority):
        if video_secs / total <= ceiling:
            break
        out[i] = out[i].model_copy(update={"kind": "broll_image"})
        video_secs -= _planned_length(out[i])

    # Promote the most important stills up to the share.
    for i in sorted((i for i in movable if out[i].kind == "broll_image"),
                    key=lambda i: -out[i].priority):
        if video_secs / total >= share:
            break
        beat = out[i]
        if not beat.image_prompt:
            continue
        out[i] = beat.model_copy(update={
            "kind": "broll_video",
            # The scene prompt doubles as the motion prompt: the video stage
            # only needs to know what world it is moving through.
            "video_prompt": beat.video_prompt or _clean_prompt(
                f"{beat.image_prompt}, gentle cinematic camera movement, "
                f"natural motion in the scene"),
        })
        video_secs += _planned_length(out[i])
    return out


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

async def find_topics(program: Program, ask: AskJson,
                      genre: str = "general") -> List[Topic]:
    """Ask the model what the video is about, one window at a time."""
    lines = transcript_lines(program).splitlines()
    if not lines:
        return []

    system = TOPIC_SYSTEM + genre_mod.genre_block(genre)
    topics: List[Topic] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + WINDOW_LINES)
        window = "\n".join(lines[start:end])
        answer = await _ask(ask, system, f"Transcript:\n{window}\n\nTopics:",
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


PARAGRAPH_SYSTEM = (
    "You are the researcher for a YouTube video. You are given the paragraphs of "
    "the speaker's own SCRIPT, numbered. The paragraphs are the topics; do not "
    "merge or split them. The script may be in Hindi (Devanagari or Latin letters) "
    "mixed with English.\n\n"
    "For each paragraph give:\n"
    "  index    - its number, unchanged\n"
    "  topic    - three to six words naming it, IN ENGLISH\n"
    "  summary  - one sentence in ENGLISH saying what the speaker says\n"
    "  visual   - one sentence in ENGLISH describing a single SCENE a camera could "
    "photograph that illustrates it\n"
    "  priority - 0.0 to 1.0, how much it would benefit from a picture\n\n"
    "Answer with JSON only:\n"
    '{"topics": [{"index": 0, "topic": "", "summary": "", "visual": "", "priority": 0.0}]}'
)

PARAGRAPH_SCHEMA = {
    "name": "paragraph_topics",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "visual": {"type": "string"},
                        "priority": {"type": "number"},
                    },
                    "required": ["index", "topic", "summary", "visual", "priority"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topics"],
        "additionalProperties": False,
    },
}


async def describe_paragraphs(topics: List[Topic], ask: AskJson,
                              genre: str = "general") -> List[Topic]:
    """Name and illustrate script-derived topics; the boundaries are the
    script's and are never moved. A topic the model skips keeps its
    provisional name."""
    if not topics:
        return topics
    system = PARAGRAPH_SYSTEM + genre_mod.genre_block(genre)
    out = [t.model_copy() for t in topics]
    starts = list(range(0, len(out), TOPIC_BATCH))
    calls = []
    for start in starts:
        listing = "\n".join(
            f"{start + i}. {t.summary[:500]}" for i, t in enumerate(out[start:start + TOPIC_BATCH]))
        calls.append((system, f"Paragraphs:\n{listing}\n\nTopics:", PARAGRAPH_SCHEMA))
    answers = await _ask_all(ask, calls)
    for start, answer in zip(starts, answers):
        parsed = first_json_object(answer or "")
        if not parsed:
            continue
        for raw in parsed.get("topics", []) or []:
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(out)):
                continue
            topic = out[index]
            name = str(raw.get("topic", "")).strip()
            if name and not topic.heading:
                topic.topic = name[:80]
            topic.summary = str(raw.get("summary") or topic.summary).strip()[:600]
            topic.visual = str(raw.get("visual") or "").strip()
            try:
                topic.priority = max(0.0, min(1.0, float(raw.get("priority", topic.priority))))
            except (TypeError, ValueError):
                pass
    return out


async def write_beats(topics: List[Topic], ask: AskJson,
                      genre: str = "general",
                      genre_secondary: Optional[str] = None,
                      character: Optional[Character] = None) -> List[Beat]:
    """Turn topics into on-screen beats with generation prompts. With a main
    character, each beat also says whether it shows them."""
    beats: List[Beat] = []
    by_topic = {t.topic.lower(): t for t in topics}

    system = BEAT_SYSTEM + genre_mod.genre_block(genre)
    schema = BEAT_SCHEMA
    if character is not None:
        system += character_mod.beat_block(character)
        schema = BEAT_SCHEMA_WITH_CHARACTER
    starts = list(range(0, len(topics), TOPIC_BATCH))
    calls = []
    for start in starts:
        listing = "\n".join(
            f"{i + 1}. {t.topic} | {t.summary} | suggested visual: {t.visual}"
            for i, t in enumerate(topics[start:start + TOPIC_BATCH]))
        calls.append((system, f"Topics:\n{listing}\n\nBeats:", schema))
    answers = await _ask_all(ask, calls)
    for start, answer in zip(starts, answers):
        batch = topics[start:start + TOPIC_BATCH]
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
                # Stamp the genre's look on deterministically: the system prompt
                # asked for it, but a local model forgets, and one off-mood
                # picture in a horror edit reads as a mistake.
                image_prompt=genre_mod.apply_look(
                    raw.get("image_prompt") or topic.visual or None, genre, genre_secondary),
                video_prompt=genre_mod.apply_look(raw.get("video_prompt"), genre, genre_secondary),
                popup_text=raw.get("popup_text"),
                negative_prompt=genre_mod.apply_negative(
                    str(raw.get("negative_prompt")
                        or "text, watermark, logo, deformed hands, blurry"),
                    genre, genre_secondary),
                style_hint=(str(raw.get("style_hint", "photoreal")).strip().lower()
                            if str(raw.get("style_hint", "")).strip().lower()
                            in ("photoreal", "illustration", "diagram", "abstract")
                            else "photoreal"),
                shows_character=bool(character is not None and raw.get("shows_main_character") is True),
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
                         ask: Optional[AskJson],
                         genre: str = "general",
                         genre_secondary: Optional[str] = None) -> List[Beat]:
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

    def slices_for(beat: Beat) -> int:
        count = int(beat.duration_s // cell) if cell > 0 else 1
        if not beat.is_cutaway or count < 2 or not beat.image_prompt:
            return 0
        return min(MAX_SHOT_SLICES, count)

    to_vary = [b for b in beats if slices_for(b)]
    answers: Dict[int, Optional[str]] = {}
    if ask is not None and to_vary:
        system = VARIATION_SYSTEM + genre_mod.genre_block(genre)
        replies = await _ask_all(ask, [
            (system, f"Scene: {b.image_prompt}\nN: {slices_for(b)}\n\nPrompts:", VARIATION_SCHEMA)
            for b in to_vary])
        answers = {id(b): reply for b, reply in zip(to_vary, replies)}

    out: List[Beat] = []
    for beat in beats:
        span = beat.duration_s
        count = slices_for(beat)
        if not count:
            out.append(beat)
            continue
        prompts: List[str] = []
        if ask is not None:
            try:
                answer = answers.get(id(beat))
                parsed = first_json_object(answer or "") or {}
                # Re-stamp the look: the model rewrites the scene and routinely
                # drops the mood suffix the base prompt carried.
                prompts = [_clean_prompt(genre_mod.apply_look(p, genre, genre_secondary))
                           for p in parsed.get("prompts", []) or []]
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


# --- guaranteed pop-in labels ------------------------------------------------

# The opening title owns the first seconds of the video; a pop-up under it is
# clutter.
TITLE_ZONE_SECONDS = 3.0


def topic_popups(topics: List[Topic], settings: PresentationSettings) -> List[Beat]:
    """A pop-in label per topic, for when the model planned no pop-ups at all.

    The model is asked for pop-ups but routinely plans every beat as a cutaway
    (a real run planned 18 beats, zero pop-ups), and then the video has no text
    accents whatever the user asked for. The topics themselves are the reliable
    fallback: each one's name pops in as the speaker reaches it, which is the
    standard section-label treatment. Placement still owns spacing and keeps
    them off the B-roll."""
    popups: List[Beat] = []
    for topic in topics:
        text = _clean_popup(topic.topic)
        if not text or topic.start_s < TITLE_ZONE_SECONDS:
            continue
        popups.append(Beat(
            start_s=topic.start_s,
            end_s=max(topic.start_s + 4.0, min(topic.end_s, topic.start_s + 12.0)),
            topic=topic.topic,
            summary=topic.summary,
            kind="popup",
            priority=topic.priority,
            popup_text=text,
            negative_prompt="",
        ))
    return popups


# --- the deterministic fallback --------------------------------------------

STOP_WORDS = {
    "the", "and", "that", "this", "with", "have", "from", "they", "will", "what",
    "when", "your", "which", "there", "their", "would", "about", "could", "been",
    "hai", "hain", "aur", "kya", "yah", "vah", "mein", "men", "kar", "rahe", "ko",
    "ki", "ka", "ke", "se", "par", "bhi", "hee", "toh", "dosto", "aap", "aapako",
}


# Without a model, topics are cut at the longest pauses so each runs roughly
# this long — the length a spoken subject tends to take.
FALLBACK_TOPIC_TARGET_S = 40.0
FALLBACK_TOPIC_MIN_S = 15.0


def fallback_topics(program: Program) -> List[Topic]:
    """Topics from the speech rhythm alone: windows of ~40s split at the
    longest pauses, named by their most emphatic words."""
    if not program.words or program.duration_s < FALLBACK_TOPIC_MIN_S:
        return []
    wanted = max(1, int(round(program.duration_s / FALLBACK_TOPIC_TARGET_S)))
    if wanted == 1:
        return [Topic(start_s=0.0, end_s=program.duration_s,
                      topic=_keywords(program.words) or "the video", origin="fallback")]
    # Candidate boundaries: the gaps between words, biggest first.
    gaps = []
    for previous, word in zip(program.words, program.words[1:]):
        gaps.append((word.tl_start_s - previous.tl_end_s, word.tl_start_s))
    gaps.sort(reverse=True)
    boundaries: List[float] = []
    for _, at in gaps:
        if len(boundaries) >= wanted - 1:
            break
        if at < FALLBACK_TOPIC_MIN_S or at > program.duration_s - FALLBACK_TOPIC_MIN_S:
            continue
        if all(abs(at - b) >= FALLBACK_TOPIC_MIN_S for b in boundaries):
            boundaries.append(at)
    edges = [0.0] + sorted(boundaries) + [program.duration_s]
    topics: List[Topic] = []
    for start, end in zip(edges, edges[1:]):
        words = program.words_between(start, end)
        name = _keywords(words) or f"part {len(topics) + 1}"
        topics.append(Topic(start_s=start, end_s=end, topic=name,
                            summary=" ".join(w.text for w in words[:40]),
                            priority=0.5, origin="fallback"))
    return topics


def _keywords(words) -> str:
    """Two or three of the most emphatic content words, as a topic name."""
    ranked = sorted(
        (w for w in words if len(w.text.strip(".,!?")) > 3
         and w.text.strip(".,!?").lower() not in STOP_WORDS),
        key=lambda w: -w.emphasis_z)
    picked: List[str] = []
    for word in ranked:
        token = word.text.strip(".,!?").lower()
        if token not in picked:
            picked.append(token)
        if len(picked) == 3:
            break
    return " ".join(picked)


def fallback_plan(program: Program, settings: PresentationSettings,
                  genre: str = "general",
                  genre_secondary: Optional[str] = None) -> List[Beat]:
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
            image_prompt=genre_mod.apply_look(
                f"Cinematic documentary photograph representing {subject}, "
                f"natural lighting, 35mm film still, shallow depth of field",
                genre, genre_secondary),
            negative_prompt=genre_mod.apply_negative(
                "text, watermark, logo, deformed hands, blurry", genre, genre_secondary),
        ))
    return beats


async def plan_shots(program: Program, settings: PresentationSettings,
                     ask: Optional[AskJson],
                     genre: Optional[str] = None,
                     script_topics: Optional[List[Topic]] = None,
                     extra_beats: Optional[List[Beat]] = None,
                     enrich: Optional[Callable[[List[Topic]], Awaitable[None]]] = None) -> ShotPlan:
    """The whole of stage B: topics, beats, validation — or the fallback.

    `genre` styles every prompt to the kind of video this is (a horror video
    gets horror imagery). None means detect it from the transcript here.
    `script_topics` are the user's script paragraphs: when given, they ARE the
    topics (the model only names and illustrates them). `extra_beats` are
    beats other planners insist on (stage directions, extracted entities);
    they are validated like everything else but never deduplicated away.
    """
    if genre is None:
        genre = await genre_mod.detect_genre(program, ask)
    genre = genre_mod.normalise(genre)
    # The second genre, when the caller named one distinct from the primary —
    # blended into the FX palette and B-roll style tags only; see genre.py.
    genre_secondary = (genre_mod.normalise(settings.genre_secondary)
                       if settings.genre_secondary else None)
    if genre_secondary in (None, "general", genre):
        genre_secondary = None

    beats: List[Beat] = []
    topics: List[Topic] = list(script_topics or [])
    source = "llm"
    # A lead the caller named (or ruled out) wins over asking the model.
    named = settings.main_character is not None
    character: Optional[Character] = (
        character_mod.from_settings(settings.main_character)
        if named and settings.character_consistency else None)

    if ask is not None:
        try:
            if topics:
                topics = await describe_paragraphs(topics, ask, genre)
            else:
                topics = await find_topics(program, ask, genre)
            if topics and enrich is not None:
                # Story structure tags the topics (acts) before the beats are
                # weighed: an act's density multiplier scales its beats'
                # priority, so the budget favours the parts that want pictures.
                try:
                    await enrich(topics)
                except Exception as e:
                    logger.warning("Topic enrichment failed (%s); acts unweighted", e)
            if topics and not named and settings.character_consistency and settings.broll:
                character = await character_mod.find_main_character(program, ask, topics)
            if topics:
                beats = await write_beats(topics, ask, genre, genre_secondary, character)
                beats = _weigh_by_act(beats, topics)
        except Exception as e:
            logger.warning("Shot planning failed (%s); falling back to keywords", e)
            beats = []

    if not beats:
        source = "fallback"
        if not named:
            character = None    # nothing was tagged by the model
        beats = fallback_plan(program, settings, genre, genre_secondary)
        if not topics:
            # Everything topic-based — chapters, moods, cards, transitions —
            # needs topics, and a night without the model used to get none.
            topics = fallback_topics(program)
            if topics and enrich is not None:
                try:
                    await enrich(topics)
                except Exception as e:
                    logger.warning("Topic enrichment failed (%s)", e)

    beats = list(beats) + list(extra_beats or [])
    candidates, dropped = sanitize_beats(beats, program, settings)
    unique, dup_dropped = dedup_beats(candidates)
    if source == "llm":
        # The fallback already proposes one beat per shot slot; only topic-sized
        # LLM beats need splitting into a sequence of shots.
        planned = [b for b in unique if b.origin == "llm"]
        others = [b for b in unique if b.origin != "llm"]
        unique = await multiply_beats(planned, settings, ask, genre, genre_secondary) + others
        # The model's own beats sometimes fall short of the seconds the
        # coverage target asks for however well they are split — top up with
        # keyword-derived filler in whatever time they left uncovered, so the
        # budget below has enough candidates to actually spend on.
        unique = _ensure_coverage_candidates(unique, program, settings, genre, genre_secondary)
    kept, budget_dropped = budget_beats(unique, program, settings)
    dropped = dropped + dup_dropped + budget_dropped
    kept = balance_video_share(kept, settings)
    kept = character_mod.apply_to_beats(kept, character)
    if character is not None:
        logger.info("Shot plan: %d of %d beats show the main character (%s)",
                    sum(b.shows_character for b in kept), len(kept), character.name)

    # The model plans pop-ups so rarely that a run with none is the normal case;
    # the topics themselves then supply one label each. Added after budgeting —
    # pop-ups cost no screen-time budget — and validated the same way.
    # When chapter titles are on they carry each topic's name; a pop-up saying
    # the same thing a second later would be a stutter.
    if (settings.popups and topics and not any(b.kind == "popup" for b in kept)
            and not (settings.cards and settings.chapter_titles)):
        labels, _ = sanitize_beats(topic_popups(topics, settings), program, settings)
        for index, beat in enumerate(labels):
            beat.id = f"p{index:02d}"
        kept = sorted(kept + labels, key=lambda b: b.start_s)
        if labels:
            logger.info("Shot plan: %d topic labels added as pop-ups", len(labels))

    if not settings.popups:
        kept = [b for b in kept if b.kind != "popup"]
    if not settings.broll:
        kept = [b for b in kept if not b.is_cutaway]

    logger.info("Shot plan (%s, genre=%s): %d beats kept, %d dropped",
                source, genre, len(kept), len(dropped))
    return ShotPlan(beats=kept, dropped=dropped, source=source, genre=genre,
                    genre_secondary=genre_secondary or "", topics=topics,
                    character=character)


def _weigh_by_act(beats: List[Beat], topics: List[Topic]) -> List[Beat]:
    """Scale each cutaway's priority by its topic's act density."""
    from .structure import act_weight
    weights = {t.topic.lower(): act_weight(t.act) for t in topics}
    out: List[Beat] = []
    for beat in beats:
        weight = weights.get(beat.topic.lower(), 1.0)
        if beat.is_cutaway and weight != 1.0:
            beat = beat.model_copy(update={"priority": max(0.0, min(1.0, beat.priority * weight))})
        out.append(beat)
    return out


def seed_for(project_id: str, extra: str = "") -> int:
    """A stable seed, so two runs of one project make the same decisions."""
    digest = hashlib.sha1(f"{project_id}:{extra}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
