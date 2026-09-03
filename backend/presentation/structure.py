"""The shape of the whole video: its acts, its cold open, and its metadata.

Three things fall out of reading the topics as a story rather than a list:

* **Acts.** A horror narration runs hook → setup → build → reveal → climax →
  aftermath → call to action; an explainer runs hook → context → points →
  takeaway → call to action. Each act has a picture density — sparse in the
  setup, dense in the build, the speaker's face at the reveal — and the
  planner scales its beats' priority by it, so the coverage budget goes where
  the story wants pictures.
* **The cold open.** The single most gripping sentence, played before the
  title, then a beat of black, then the video from its start. It is a copy of
  the V1/A1 span placed in the programme offset (the same mechanism an intro
  card uses), so the transcript rebuild never touches it and the original
  sentence still plays in place later.
* **Metadata.** Title options, a description with chapter timestamps, tags —
  written next to the report as `metadata.json` and `chapters.txt`.

One schema-constrained model call does the acts and picks the hook; the
metadata is a second, small one. Without a model the acts are positional and
the hook is scored from the transcript (a question, a turn word, loudness).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from timeline import clip_ops
from timeline.authoring import CAPTION_TRACK, _shift_all_items, clear_generated
from timeline.presets import caption_preset
from timeline.schema import Timeline, TimelineItem, time_to_frame

from . import genre as genre_mod
from .models import PresentationSettings, Program, Topic
from .program import transcript_lines

logger = logging.getLogger("presentation.structure")

AskJson = Callable[..., Awaitable[Optional[str]]]

STORY_ACTS = ("hook", "setup", "build", "reveal", "climax", "aftermath", "cta")
EXPLAINER_ACTS = ("hook", "context", "point", "takeaway", "cta")
STORY_GENRES = {"horror", "true_crime"}

# Picture density per act: a multiplier on the topic's beats' priority.
DENSITY: Dict[str, float] = {
    "hook": 0.5, "setup": 0.7, "build": 1.0, "reveal": 0.5, "climax": 0.75,
    "aftermath": 0.85, "cta": 0.3, "context": 0.85, "point": 1.1, "takeaway": 0.7,
}

COLD_OPEN_ORIGIN = "coldopen"
COLD_OPEN_VIDEO_TRACK = "V2"
COLD_OPEN_AUDIO_TRACK = "A5"
MAX_HOOK_S = 8.0
MIN_HOOK_S = 2.0
# A hook this close to the start would play twice in a row.
MIN_HOOK_START_S = 12.0
COLD_OPEN_GAP_S = 0.6

_TURN_WORDS = {"achanak", "suddenly", "lekin", "but", "kyunki", "because", "kabhi", "never",
               "sach", "truth", "secret", "raaz", "pata", "actually", "asli", "sabse", "most"}
_CTA_WORDS = {"subscribe", "sabsakraaib", "like", "comment", "share", "channel", "bell",
              "video", "dekhna", "watch"}


def acts_for(genre: str) -> Tuple[str, ...]:
    return STORY_ACTS if genre in STORY_GENRES else EXPLAINER_ACTS


def act_weight(act: Optional[str]) -> float:
    return DENSITY.get((act or "").lower(), 1.0)


class Hook(BaseModel):
    start_s: float
    end_s: float
    text: str
    source: str = "fallback"


# --- the model call -----------------------------------------------------------------------

def _system(genre: str) -> str:
    acts = acts_for(genre)
    if genre in STORY_GENRES:
        guide = ("hook (the line that makes the viewer stay), setup (where and who), build "
                 "(unease rising), reveal (the turn), climax (the peak), aftermath (what it "
                 "cost), cta (asking for a like, subscribe, the next video)")
    else:
        guide = ("hook (the question or promise), context (the background), point (one "
                 "of the main things explained), takeaway (what it means for the viewer), "
                 "cta (asking for a like, subscribe, the next video)")
    return (
        "You are the story editor for a YouTube video. You are given its TOPICS in order, "
        "with a summary each, and the transcript as timestamped lines. The speaker may talk "
        "in Hindi written in Latin letters; read past the spelling.\n\n"
        f"1. Give each topic its ACT, one of: {', '.join(acts)}. Meaning: {guide}. Acts "
        "run in story order; several topics in a row may share one.\n"
        "2. Pick the HOOK: the ONE sentence from the transcript that would most make a "
        "viewer keep watching if it were the first thing they heard — a question, a "
        "promise, a shock, a number. Quote it EXACTLY as written in the transcript line "
        "and give that line's timestamp as start_s. Never the very first line.\n\n"
        "Answer with JSON only:\n"
        '{"acts": [{"index": 0, "act": ""}], "hook": {"start_s": 0.0, "text": ""}}'
    )


def _schema(genre: str) -> Dict[str, Any]:
    return {
        "name": "structure",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "acts": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"},
                                   "act": {"type": "string", "enum": list(acts_for(genre))}},
                    "required": ["index", "act"], "additionalProperties": False}},
                "hook": {"type": "object",
                         "properties": {"start_s": {"type": "number"}, "text": {"type": "string"}},
                         "required": ["start_s", "text"], "additionalProperties": False},
            },
            "required": ["acts", "hook"],
            "additionalProperties": False,
        },
    }


async def tag_structure(topics: Sequence[Topic], program: Program, ask: Optional[AskJson],
                        genre: str = "general") -> Optional[Hook]:
    """Fill each topic's `act`; return the hook. Never raises."""
    ordered = sorted(topics, key=lambda t: t.start_s)
    for index, topic in enumerate(ordered):
        topic.act = _fallback_act(index, len(ordered), genre, program, topic)
    hook: Optional[Hook] = None
    if ask is not None and ordered:
        try:
            hook = await _ask_structure(ordered, program, ask, genre)
        except Exception as e:
            logger.warning("Structure call failed (%s); positional acts", e)
    if hook is None:
        hook = fallback_hook(program)
    logger.info("Acts: %s; hook: %r at %.1fs", ", ".join(t.act or "-" for t in ordered),
                hook.text[:50] if hook else None, hook.start_s if hook else -1)
    return hook


def _fallback_act(index: int, count: int, genre: str, program: Program, topic: Topic) -> str:
    text = program.text_between(topic.start_s, topic.end_s).lower()
    words = set(re.findall(r"[a-z]+", text))
    last = index == count - 1
    if last and count > 1 and len(words & _CTA_WORDS) >= 2:
        return "cta"
    position = index / max(1, count - 1) if count > 1 else 0.0
    if genre in STORY_GENRES:
        if index == 0:
            return "hook" if count >= 5 else "setup"
        arc = ["setup", "build", "reveal", "climax", "aftermath"]
        return arc[min(len(arc) - 1, int(position * (len(arc) - 0.001)))]
    if index == 0:
        return "hook" if count >= 4 else "context"
    if last and count > 2:
        return "takeaway"
    return "context" if position < 0.25 and count > 4 else "point"


async def _ask_structure(topics: List[Topic], program: Program, ask: AskJson,
                         genre: str) -> Optional[Hook]:
    from .shotplan import _ask, first_json_object
    listing = "\n".join(f"{i}. [{t.start_s:.1f}s] {t.topic}: {t.summary}"
                        for i, t in enumerate(topics))
    lines = transcript_lines(program)
    answer = await _ask(ask, _system(genre) + genre_mod.genre_block(genre),
                        f"Topics:\n{listing}\n\nTranscript:\n{lines[:6000]}\n\nAnswer:",
                        _schema(genre))
    parsed = first_json_object(answer or "")
    if not parsed:
        return None
    valid = set(acts_for(genre))
    for raw in parsed.get("acts", []) or []:
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        act = str(raw.get("act", "")).strip().lower()
        if 0 <= index < len(topics) and act in valid:
            topics[index].act = act
    raw_hook = parsed.get("hook") or {}
    try:
        start = float(raw_hook.get("start_s", -1))
    except (TypeError, ValueError):
        start = -1.0
    return _validate_hook(program, start, str(raw_hook.get("text", "")))


# --- the hook --------------------------------------------------------------------------

def _sentences(program: Program) -> List[Tuple[float, float, str]]:
    """(start, end, text) per transcript line, from the programme's words."""
    out: List[Tuple[float, float, str]] = []
    for line in transcript_lines(program).splitlines():
        match = re.match(r"\[(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if not match:
            continue
        start = float(match.group(1))
        text = match.group(2).strip()
        words = [w for w in program.words if w.tl_start_s >= start - 0.01]
        count = len(text.split())
        if not words or count == 0:
            continue
        end = words[min(len(words), count) - 1].tl_end_s
        out.append((start, end, text))
    return out


def _validate_hook(program: Program, start: float, text: str) -> Optional[Hook]:
    """The model's hook only counts when it names a real transcript line."""
    if start < 0 or not text.strip():
        return None
    sentences = _sentences(program)
    if not sentences:
        return None
    nearest = min(sentences, key=lambda s: abs(s[0] - start))
    if abs(nearest[0] - start) > 2.0:
        # Fall back on the text: does it appear as a line?
        key = re.sub(r"\W+", " ", text.lower()).strip()
        for sentence in sentences:
            if key and key in re.sub(r"\W+", " ", sentence[2].lower()):
                nearest = sentence
                break
        else:
            return None
    s_start, s_end, s_text = nearest
    if s_end - s_start > MAX_HOOK_S:
        # Take the sentence's opening words up to the limit, on a word edge.
        words = [w for w in program.words if s_start - 0.01 <= w.tl_start_s < s_start + MAX_HOOK_S]
        if len(words) < 4:
            return None
        s_end = words[-1].tl_end_s
        s_text = " ".join(w.text for w in words)
    if s_end - s_start < MIN_HOOK_S:
        return None
    return Hook(start_s=s_start, end_s=s_end, text=s_text, source="llm")


def fallback_hook(program: Program) -> Optional[Hook]:
    """The best-scoring sentence between 5% and 70% of the programme."""
    sentences = _sentences(program)
    if len(sentences) < 3:
        return None
    lo, hi = program.duration_s * 0.05, program.duration_s * 0.7
    best: Optional[Tuple[float, Tuple[float, float, str]]] = None
    for index, (start, end, text) in enumerate(sentences):
        if index == 0 or start < lo or start > hi:
            continue
        count = len(text.split())
        if count < 4 or count > 22 or end - start < MIN_HOOK_S:
            continue
        score = 0.0
        if "?" in text:
            score += 2.0
        words = set(re.findall(r"[\w]+", text.lower()))
        score += 1.2 * len(words & _TURN_WORDS)
        if re.search(r"\d", text):
            score += 0.8
        span_words = program.words_between(start, end)
        if span_words and program.has_energy:
            score += max(0.0, sum(w.emphasis_z for w in span_words) / len(span_words))
        if best is None or score > best[0]:
            best = (score, (start, end, text))
    if best is None or best[0] <= 0.0:
        return None
    start, end, text = best[1]
    end = min(end, start + MAX_HOOK_S)
    return Hook(start_s=start, end_s=end, text=text, source="fallback")


# --- applying the cold open ------------------------------------------------------------------

def remove_cold_open(timeline: Timeline) -> int:
    """Undo a previous cold open: drop its items and pull the programme back."""
    removed = clear_generated(timeline, COLD_OPEN_ORIGIN)
    frames = max(0, timeline.cold_open_frames)
    if frames:
        _shift_all_items(timeline, -frames)
        timeline.program_offset_frames = max(0, timeline.program_offset_frames - frames)
        timeline.cold_open_frames = 0
        timeline.recalculate_duration()
    return removed


def apply_cold_open(timeline: Timeline, program: Program, hook: Optional[Hook],
                    settings: PresentationSettings) -> Optional[Dict[str, Any]]:
    """Play the hook before everything else. Returns what was done, or None.

    The hook's frames are copied from the primary source onto V2/A5 at the
    head of the timeline; everything else moves back by the hook plus a beat
    of black. The programme offset carries the shift so a transcript rebuild
    keeps it, and `cold_open_frames` lets the intro logic leave it alone.
    """
    remove_cold_open(timeline)
    if hook is None or not settings.cold_open:
        return None
    if hook.start_s < MIN_HOOK_START_S:
        logger.info("Cold open skipped: the hook is already the opening")
        return None
    duration = hook.end_s - hook.start_s
    if not (MIN_HOOK_S <= duration <= MAX_HOOK_S + 0.5):
        return None

    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    fps = fps_num / max(1, fps_den)
    # The hook's span in the cut maps onto one or more V1 segments.
    pieces: List[Tuple[str, int, int]] = []
    for segment in program.segments:
        lo = max(hook.start_s, segment.tl_start_s)
        hi = min(hook.end_s, segment.tl_end_s)
        if hi - lo <= 0.05:
            continue
        item = next((i for i in timeline.items if i.id == segment.item_id), None)
        if item is None or not item.source_id:
            continue
        src_start = segment.source_start_frame + int(round((lo - segment.tl_start_s) * fps))
        src_end = src_start + int(round((hi - lo) * fps))
        pieces.append((item.source_id, src_start, src_end))
    if not pieces:
        return None

    hook_frames = sum(end - start for _, start, end in pieces)
    gap_frames = time_to_frame(COLD_OPEN_GAP_S, fps_num, fps_den)
    offset = hook_frames + gap_frames

    # Move the whole programme back, then lay the hook in the space.
    _shift_all_items(timeline, offset)
    timeline.program_offset_frames += offset
    timeline.cold_open_frames = offset
    timeline.program_offset_color = timeline.program_offset_color or "black"

    cursor = 0
    for source_id, src_start, src_end in pieces:
        for track in (COLD_OPEN_VIDEO_TRACK, COLD_OPEN_AUDIO_TRACK):
            source = timeline.sources.get(source_id)
            if track.startswith("A") and (source is None or not source.has_audio):
                continue
            item = clip_ops.add_media_item(timeline, source_id, track, cursor, src_start,
                                           src_end, origin=COLD_OPEN_ORIGIN)
            item.label = "cold open"
            if track.startswith("A"):
                item.audio_fade_in = 0.05
                item.audio_fade_out = 0.25
        cursor += src_end - src_start

    # The bed and the ambience should start under the hook, not after it.
    for item in timeline.items:
        if item.origin in ("music", "ambience") and item.timeline_start_frame == offset:
            item.timeline_start_frame = 0
            item.source_end_frame += offset

    # A karaoke caption for the hook, in the programme's caption style.
    _caption_for_hook(timeline, program, hook, settings, hook_frames)

    timeline.recalculate_duration()
    timeline.revision += 1
    logger.info("Cold open: %.1fs hook from %.1fs (%s)", duration, hook.start_s, hook.source)
    return {"text": hook.text, "seconds": round(hook_frames / fps, 2),
            "from_s": round(hook.start_s, 2), "source": hook.source}


def _caption_for_hook(timeline: Timeline, program: Program, hook: Hook,
                      settings: PresentationSettings, hook_frames: int) -> None:
    if not settings.captions:
        return
    words = [w for w in program.words if w.tl_start_s >= hook.start_s - 0.01
             and w.tl_end_s <= hook.end_s + 0.01]
    if not words:
        return
    config = caption_preset(settings.caption_preset)
    uppercase = bool(config.get("uppercase", False))
    per_card = int(config.get("words_per_caption", 4))
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    for start in range(0, len(words), per_card):
        card = words[start:start + per_card]
        card_start = time_to_frame(card[0].tl_start_s - hook.start_s, fps_num, fps_den)
        card_end = time_to_frame(card[-1].tl_end_s - hook.start_s, fps_num, fps_den)
        if start + per_card < len(words):
            card_end = min(card_end, time_to_frame(words[start + per_card].tl_start_s - hook.start_s,
                                                   fps_num, fps_den))
        card_end = min(max(card_end, card_start + 6), hook_frames)
        if card_end <= card_start:
            continue
        content = " ".join(w.text for w in card)
        if uppercase:
            content = content.upper()
        item = clip_ops.add_text_item(timeline, content, card_start, card_end - card_start,
                                      track=CAPTION_TRACK, style=dict(config.get("style", {})))
        item.origin = COLD_OPEN_ORIGIN
        item.text.preset = settings.caption_preset
        item.text.words = [{"text": (w.text.upper() if uppercase else w.text),
                            "start_s": round(w.tl_start_s - card[0].tl_start_s, 3),
                            "end_s": round(w.tl_end_s - card[0].tl_start_s, 3)} for w in card]


# --- metadata ----------------------------------------------------------------------------------

METADATA_SYSTEM = (
    "You write the YouTube listing for a video from its topics. The speaker may talk in "
    "Hindi written in Latin letters; write in the same language mix the speaker uses.\n"
    "Give: three title options (each under 70 characters, no clickbait the video does not "
    "deliver), a description of two or three sentences, and 10 to 15 tags (single words "
    "or short phrases).\n"
    'Answer with JSON only: {"titles": ["", "", ""], "description": "", "tags": [""]}'
)

METADATA_SCHEMA = {
    "name": "metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "titles": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["titles", "description", "tags"],
        "additionalProperties": False,
    },
}


def _stamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def chapters_for(topics: Sequence[Topic], offset_s: float = 0.0,
                 min_gap_s: float = 10.0) -> List[Tuple[str, str]]:
    """(timestamp, name) per chapter; YouTube wants the first at 0:00 and at
    least ten seconds between chapters."""
    out: List[Tuple[str, str]] = []
    last = -min_gap_s
    for index, topic in enumerate(sorted(topics, key=lambda t: t.start_s)):
        at = 0.0 if index == 0 else topic.start_s + offset_s
        if index and at - last < min_gap_s:
            continue
        name = (topic.heading or topic.topic or f"Part {index + 1}").strip()
        out.append((_stamp(at), name[:60]))
        last = at
    if out and out[0][0] != "0:00":
        out[0] = ("0:00", out[0][1])
    return out


async def write_metadata(project_dir: Path, program: Program, topics: Sequence[Topic],
                         ask: Optional[AskJson], genre: str, title: Optional[str],
                         offset_s: float = 0.0) -> Dict[str, Any]:
    """metadata.json + chapters.txt next to the report. Never raises."""
    listing = "\n".join(f"- {t.topic}: {t.summary}" for t in topics) or program.text_between(
        0.0, min(90.0, program.duration_s))
    titles: List[str] = [title] if title else []
    description = ""
    tags: List[str] = []
    if ask is not None:
        try:
            from .shotplan import _ask, first_json_object
            answer = await _ask(ask, METADATA_SYSTEM + genre_mod.genre_block(genre),
                                f"Topics:\n{listing}\n\nListing:", METADATA_SCHEMA)
            parsed = first_json_object(answer or "") or {}
            titles += [str(t).strip()[:70] for t in parsed.get("titles", []) if str(t).strip()]
            description = str(parsed.get("description") or "").strip()[:1200]
            tags = [str(t).strip()[:30] for t in parsed.get("tags", []) if str(t).strip()][:15]
        except Exception as e:
            logger.warning("Metadata call failed (%s)", e)
    if not description:
        description = " ".join(program.text_between(0.0, min(40.0, program.duration_s)).split()[:45])
    if not tags:
        style = genre_mod.style_for(genre)
        text = program.text_between(0.0, program.duration_s).lower()
        tags = [k for k in style.keywords if k in text][:12]
    chapters = chapters_for(topics, offset_s)
    chapter_lines = [f"{stamp} {name}" for stamp, name in chapters]
    payload = {
        "titles": list(dict.fromkeys(t for t in titles if t)),
        "description": description + ("\n\n" + "\n".join(chapter_lines) if chapter_lines else ""),
        "tags": tags,
        "chapters": [{"at": stamp, "name": name} for stamp, name in chapters],
        "genre": genre,
    }
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (project_dir / "chapters.txt").write_text("\n".join(chapter_lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write metadata: %s", e)
    return payload
