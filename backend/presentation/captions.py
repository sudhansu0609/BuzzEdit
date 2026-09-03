"""Caption extras: emphasised words and the bilingual line.

**Emphasis.** In a karaoke caption every word lights up as it is spoken; an
emphasised word is drawn larger and in the accent colour the whole time it is
on screen. Which words: numbers and percentages always (they are why the
viewer is reading), and words the speaker actually stressed — the programme
carries a loudness z-score per word from the energy envelope — capped at two
per card so emphasis stays emphasis.

**The bilingual line.** For an Indic-language project, a smaller English line
under each caption. The translation is asked of the model sentence by sentence
while it is still resident (captions are generated after the model is
ejected), then attached to the cards by time. Without a model, a transcript
that carries per-segment English (`text_english`) is used; otherwise nothing.
"""

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from timeline.authoring import CAPTION_ORIGIN
from timeline.schema import Timeline, frame_to_time

from .models import Program
from .program import transcript_lines

logger = logging.getLogger("presentation.captions")

AskJson = Callable[..., Awaitable[Optional[str]]]

# A word louder than this many standard deviations is stressed.
EMPHASIS_Z = 1.1
MAX_EMPHASIS_PER_CARD = 2
_NUMBER_RE = re.compile(r"\d")

TRANSLATE_SYSTEM = (
    "Translate each numbered line of a spoken transcript into short, natural "
    "English for a subtitle. The lines are Hindi (Devanagari or Latin letters) "
    "mixed with English; keep names and numbers as they are. One translation per "
    "line, same numbering, at most twelve words each.\n"
    'Answer with JSON only: {"lines": [{"index": 0, "english": ""}]}'
)

TRANSLATE_SCHEMA = {
    "name": "translations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "lines": {"type": "array", "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "english": {"type": "string"}},
                "required": ["index", "english"], "additionalProperties": False}},
        },
        "required": ["lines"],
        "additionalProperties": False,
    },
}
TRANSLATE_BATCH = 25


# --- emphasis ------------------------------------------------------------------------

def mark_emphasis(timeline: Timeline, program: Program) -> int:
    """Flag the emphasised words on every karaoke caption card. Returns how many."""
    by_start: Dict[int, float] = {}
    for word in program.words:
        by_start[int(round(word.tl_start_s * 1000))] = word.emphasis_z
    count = 0
    for item in timeline.items:
        if item.origin != CAPTION_ORIGIN or item.text is None or not item.text.words:
            continue
        card_start = frame_to_time(item.timeline_start_frame, timeline.fps_num, timeline.fps_den)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for word in item.text.words:
            word.pop("emphasis", None)
            text = str(word.get("text", ""))
            at = int(round((card_start + float(word.get("start_s", 0.0))) * 1000))
            loud = _nearest(by_start, at)
            if _NUMBER_RE.search(text):
                scored.append((10.0, word))
            elif loud is not None and loud >= EMPHASIS_Z and len(text.strip(".,!?")) >= 3:
                scored.append((loud, word))
        for _, word in sorted(scored, key=lambda s: -s[0])[:MAX_EMPHASIS_PER_CARD]:
            word["emphasis"] = True
            count += 1
    return count


def _nearest(by_start: Dict[int, float], at_ms: int, tolerance_ms: int = 60) -> Optional[float]:
    if at_ms in by_start:
        return by_start[at_ms]
    for delta in range(1, tolerance_ms + 1):
        for candidate in (at_ms - delta, at_ms + delta):
            if candidate in by_start:
                return by_start[candidate]
    return None


# --- translation ---------------------------------------------------------------------

async def translate_lines(program: Program, ask: Optional[AskJson]) -> List[Tuple[float, float, str]]:
    """(start, end, english) per transcript line, from the model. Empty without one."""
    if ask is None:
        return []
    lines: List[Tuple[float, str]] = []
    for raw in transcript_lines(program).splitlines():
        match = re.match(r"\[(\d+(?:\.\d+)?)\]\s*(.*)", raw)
        if match and match.group(2).strip():
            lines.append((float(match.group(1)), match.group(2).strip()))
    if not lines:
        return []
    out: List[Tuple[float, float, str]] = []
    from .shotplan import _ask, first_json_object
    for start in range(0, len(lines), TRANSLATE_BATCH):
        batch = lines[start:start + TRANSLATE_BATCH]
        listing = "\n".join(f"{start + i}. {text}" for i, (_, text) in enumerate(batch))
        try:
            answer = await _ask(ask, TRANSLATE_SYSTEM, f"Lines:\n{listing}\n\nTranslations:",
                                TRANSLATE_SCHEMA)
        except Exception as e:
            logger.warning("Translation call failed (%s)", e)
            continue
        parsed = first_json_object(answer or "") or {}
        for raw in parsed.get("lines", []) or []:
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            english = " ".join(str(raw.get("english") or "").split())
            if not (0 <= index < len(lines)) or not english:
                continue
            line_start = lines[index][0]
            line_end = lines[index + 1][0] if index + 1 < len(lines) else program.duration_s
            out.append((line_start, line_end, english[:120]))
    out.sort(key=lambda t: t[0])
    logger.info("Translations: %d of %d lines", len(out), len(lines))
    return out


def translations_from_transcript(data: Dict[str, Any], program: Program,
                                 timeline: Timeline) -> List[Tuple[float, float, str]]:
    """Per-segment English the transcription stored, projected onto the cut."""
    from timeline.authoring import _program_segments, source_to_timeline_frame
    transcript = (data.get("transcript") or {})
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not segments:
        return []
    fps = timeline.fps_num / max(1, timeline.fps_den)
    cuts = _program_segments(timeline)
    out: List[Tuple[float, float, str]] = []
    for segment in segments:
        english = " ".join(str(segment.get("text_english") or "").split())
        if not english:
            continue
        start_frame = source_to_timeline_frame(cuts, int(float(segment.get("start", 0.0)) * fps))
        end_frame = source_to_timeline_frame(cuts, int(float(segment.get("end", 0.0)) * fps) - 1)
        if start_frame is None and end_frame is None:
            continue
        start = (start_frame if start_frame is not None else end_frame) / fps
        end = (end_frame if end_frame is not None else start_frame) / fps
        out.append((start, max(start + 0.1, end), english[:120]))
    out.sort(key=lambda t: t[0])
    return out


def attach_second_lines(timeline: Timeline,
                        translations: Sequence[Tuple[float, float, str]]) -> int:
    """Give each caption card the English of the sentence it belongs to.

    Several cards share one sentence and every one of them carries the whole
    translation: a subtitle line that stays put while the karaoke words change
    above it reads far better than fragments handed out card by card.
    """
    if not translations:
        return 0
    cards = sorted(
        [i for i in timeline.items if i.origin == CAPTION_ORIGIN and i.text is not None],
        key=lambda i: i.timeline_start_frame)
    fps = timeline.fps_num / max(1, timeline.fps_den)
    grouped: Dict[int, List] = {}
    for card in cards:
        card.text.second_line = None
        mid = (card.timeline_start_frame + card.timeline_end_frame) / 2 / fps
        for index, (start, end, _) in enumerate(translations):
            if start - 0.05 <= mid < end + 0.05:
                grouped.setdefault(index, []).append(card)
                break
    count = 0
    for index, members in grouped.items():
        english = translations[index][2]
        for card in members:
            card.text.second_line = english
            count += 1
    return count
