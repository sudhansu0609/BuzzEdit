"""The user's script, as the presentation pass sees it.

`asr.script_align` does the aligning; this module is the bridge to the pass:
it applies a stored script to a project's timeline words (fixing caption
spelling), projects the script's paragraphs and stage directions from source
time into the CUT programme's time, and turns the directions into beats.

A script is optional. Everything here returns empty structures when there is
none, and the planner behaves exactly as before.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from asr import script_align
from timeline.authoring import _program_segments, source_to_timeline_frame
from timeline.schema import Timeline, frame_to_time, time_to_frame

from .models import Beat, PresentationSettings, Program, Topic

logger = logging.getLogger("presentation.script")

# Stage directions that ask for a full-frame picture run this long on screen.
DIRECTIVE_CUTAWAY_SECONDS = 4.0
DIRECTIVE_CARD_SECONDS = 4.0
# Paragraphs shorter than this are joined onto the previous one as a topic: a
# one-line paragraph is a beat, not a subject.
MIN_PARAGRAPH_TOPIC_S = 6.0


class ScriptParagraph(BaseModel):
    index: int
    tl_start_s: float
    tl_end_s: float
    text: str = ""
    heading: Optional[str] = None
    estimated: bool = False


class ScriptDirective(BaseModel):
    kind: str
    arg: str
    tl_at_s: float
    paragraph: int = 0


class ScriptContext(BaseModel):
    paragraphs: List[ScriptParagraph] = Field(default_factory=list)
    directives: List[ScriptDirective] = Field(default_factory=list)
    aligned_words: int = 0
    token_count: int = 0

    @property
    def present(self) -> bool:
        return bool(self.paragraphs)


# --- applying a script to a project ---------------------------------------------

def apply_project_script(data: Dict[str, Any], timeline: Timeline,
                         text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Align `text` (or the project's stored script) to the timeline's words.

    Rewrites the words' spelling in place and stores the record (times in
    source seconds) at `data["script"]`. Returns the record, or None when
    there is no script.
    """
    script_text = text if text is not None else ((data.get("script") or {}).get("text"))
    if not script_text or not script_text.strip() or not timeline.words:
        return None

    fps = timeline.fps_num / max(1, timeline.fps_den)
    words = [w.model_dump() for w in timeline.words]
    record = script_align.ingest(script_text, words, start_key="start_frame",
                                 end_key="end_frame")
    for word, updated in zip(timeline.words, words):
        word.text = updated.get("text", word.text)
        word.word_native = updated.get("word_native", word.word_native)

    # Store times in source SECONDS: the timeline's frame rate may change on a
    # re-transcription, and seconds are what the rest of the project speaks.
    for paragraph in record["paragraphs"]:
        paragraph["start"] = round(paragraph["start"] / fps, 3)
        paragraph["end"] = round(paragraph["end"] / fps, 3)
    for directive in record["directives"]:
        directive["at"] = round(directive["at"] / fps, 3)
    data["script"] = record
    timeline.revision += 1
    return record


def context_from(data: Dict[str, Any], timeline: Timeline) -> ScriptContext:
    """The stored script projected onto the cut programme's clock."""
    record = data.get("script") or {}
    if not record.get("paragraphs"):
        return ScriptContext()
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    segments = _program_segments(timeline)
    duration_s = frame_to_time(timeline.duration_frames, fps_num, fps_den)

    def to_timeline(source_s: float) -> Optional[float]:
        frame = _nearest_surviving(segments, time_to_frame(source_s, fps_num, fps_den))
        return None if frame is None else frame_to_time(frame, fps_num, fps_den)

    paragraphs: List[ScriptParagraph] = []
    for raw in record["paragraphs"]:
        start = to_timeline(float(raw["start"]))
        end = to_timeline(float(raw["end"]))
        if start is None or end is None:
            continue
        end = min(duration_s, max(end, start + 0.5))
        paragraphs.append(ScriptParagraph(
            index=int(raw.get("index", len(paragraphs))), tl_start_s=start, tl_end_s=end,
            text=str(raw.get("text") or ""), heading=raw.get("heading"),
            estimated=bool(raw.get("estimated"))))
    # Paragraph ends: each runs to the start of the next so the topics tile
    # the programme without gaps a beat could fall into.
    paragraphs.sort(key=lambda p: p.tl_start_s)
    for current, following in zip(paragraphs, paragraphs[1:]):
        current.tl_end_s = max(current.tl_start_s + 0.5, following.tl_start_s)

    directives: List[ScriptDirective] = []
    for raw in record.get("directives") or []:
        at = to_timeline(float(raw["at"]))
        if at is None:
            continue
        directives.append(ScriptDirective(
            kind=str(raw["kind"]).lower(), arg=str(raw["arg"]), tl_at_s=at,
            paragraph=int(raw.get("paragraph", 0))))

    return ScriptContext(paragraphs=paragraphs, directives=directives,
                         aligned_words=int(record.get("aligned_words", 0)),
                         token_count=int(record.get("token_count", 0)))


def _nearest_surviving(segments: List[Tuple[int, int, int]], source_frame: int) -> Optional[int]:
    """The timeline frame for a source frame, or of the next surviving frame
    when that one was cut (a directive on a cut word still happens)."""
    frame = source_to_timeline_frame(segments, source_frame)
    if frame is not None:
        return frame
    following = [(src_start, tl_start) for src_start, _, tl_start in segments
                 if src_start > source_frame]
    if following:
        return min(following)[1]
    preceding = [(src_end, tl_start + (src_end - src_start))
                 for src_start, src_end, tl_start in segments if src_end <= source_frame]
    if preceding:
        return max(preceding)[1] - 1
    return None


# --- topics from paragraphs ----------------------------------------------------------

def paragraph_topics(ctx: ScriptContext, program: Program) -> List[Topic]:
    """The script's paragraphs as topics, short ones merged into the previous.

    Names and visuals are provisional (the heading, or the first words); the
    planner asks the model to describe them properly when it has one.
    """
    topics: List[Topic] = []
    for paragraph in ctx.paragraphs:
        if paragraph.tl_end_s <= paragraph.tl_start_s:
            continue
        span = paragraph.tl_end_s - paragraph.tl_start_s
        if topics and span < MIN_PARAGRAPH_TOPIC_S and not paragraph.heading:
            previous = topics[-1]
            previous.end_s = paragraph.tl_end_s
            previous.summary = (previous.summary + " " + paragraph.text).strip()[:600]
            continue
        words = paragraph.text.split()
        name = paragraph.heading or " ".join(words[:5])
        topics.append(Topic(
            start_s=paragraph.tl_start_s, end_s=min(program.duration_s, paragraph.tl_end_s),
            topic=name[:80], summary=paragraph.text[:600], visual="",
            priority=0.6, heading=paragraph.heading, origin="script"))
    return [t for t in topics if t.end_s > t.start_s]


# --- beats from stage directions --------------------------------------------------------

def _end(at: float, seconds: float, program: Program) -> float:
    return min(program.duration_s, at + seconds)


def directive_beats(ctx: ScriptContext, program: Program,
                    settings: PresentationSettings) -> Dict[str, Any]:
    """Turn the script's stage directions into beats and side requests.

    Returns {"beats": [...], "sfx": [(at, tag)], "title": str|None,
             "moods": {paragraph_index: mood}, "music_mood": str|None}.
    Every beat carries priority 1.0 and origin "script": the user asked for
    it by name, so it wins any budget contest with the model's own ideas.
    """
    beats: List[Beat] = []
    sfx: List[Tuple[float, str]] = []
    moods: Dict[int, str] = {}
    title: Optional[str] = None
    music_mood: Optional[str] = None
    cut = DIRECTIVE_CUTAWAY_SECONDS
    card = DIRECTIVE_CARD_SECONDS

    for d in ctx.directives:
        at = max(0.0, d.tl_at_s)
        arg = d.arg.strip()
        if not arg:
            continue
        common = dict(start_s=at, topic=arg[:60], priority=1.0, origin="script",
                      negative_prompt="text, watermark, logo, deformed hands, blurry")
        if d.kind == "broll":
            beats.append(Beat(kind="broll_image", end_s=_end(at, cut, program),
                              image_prompt=arg, **common))
        elif d.kind == "video":
            beats.append(Beat(kind="broll_video", end_s=_end(at, cut + 1.0, program),
                              video_prompt=arg, image_prompt=arg, **common))
        elif d.kind == "map":
            beats.append(Beat(kind="map", end_s=_end(at, cut, program), place=arg,
                              text=arg, **common))
        elif d.kind == "text":
            beats.append(Beat(kind="popup", end_s=_end(at, 3.5, program),
                              popup_text=arg, **common))
        elif d.kind in ("card", "chapter"):
            beats.append(Beat(kind="chapter_title", end_s=_end(at, 3.0, program),
                              text=arg, **common))
        elif d.kind == "stat":
            beats.append(Beat(kind="stat_callout", end_s=_end(at, card, program),
                              text=arg, data=_stat_data(arg), **common))
        elif d.kind == "quote":
            text, sub = _split_attribution(arg)
            beats.append(Beat(kind="quote_card", end_s=_end(at, card + 1.0, program),
                              text=text, subtext=sub, **common))
        elif d.kind == "source":
            beats.append(Beat(kind="source_card", end_s=_end(at, card, program),
                              text=arg, **common))
        elif d.kind == "character":
            text, sub = _split_attribution(arg)
            beats.append(Beat(kind="character_card", end_s=_end(at, card, program),
                              text=text, subtext=sub, **common))
        elif d.kind == "location":
            text, sub = _split_attribution(arg)
            beats.append(Beat(kind="location_card", end_s=_end(at, card, program),
                              text=text, subtext=sub, **common))
        elif d.kind == "definition":
            text, sub = _split_attribution(arg, separators=("=", ":", " - ", " – "))
            beats.append(Beat(kind="definition_card", end_s=_end(at, card + 1.0, program),
                              text=text, subtext=sub, **common))
        elif d.kind == "chart":
            data = _chart_data(arg)
            if data:
                beats.append(Beat(kind="chart", end_s=_end(at, cut + 1.0, program),
                                  data=data, text=data.get("title") or arg[:40], **common))
        elif d.kind == "split":
            left, right = _split_attribution(arg, separators=("|", " vs ", " versus "))
            if right:
                beats.append(Beat(kind="split", end_s=_end(at, cut + 1.0, program),
                                  image_prompt=left,
                                  data={"left": left, "right": right,
                                        "labels": [left[:24], right[:24]]}, **common))
        elif d.kind == "sfx":
            sfx.append((at, arg.lower().replace(" ", "_")))
        elif d.kind == "title":
            title = arg
        elif d.kind == "mood":
            moods[d.paragraph] = arg.lower()
        elif d.kind == "music":
            music_mood = arg.lower()

    beats.sort(key=lambda b: b.start_s)
    return {"beats": beats, "sfx": sfx, "title": title, "moods": moods,
            "music_mood": music_mood}


def _split_attribution(arg: str, separators=("|", " — ", " - ", ",")) -> Tuple[str, Optional[str]]:
    for sep in separators:
        if sep in arg:
            head, tail = arg.split(sep, 1)
            if head.strip() and tail.strip():
                return head.strip(), tail.strip()
    return arg, None


def _stat_data(arg: str) -> Dict[str, Any]:
    """'25%' → value 25, suffix '%'; '3.5 crore' → 3.5, ' crore'; '$120' → prefix."""
    import re
    match = re.search(r"([^\d\-]*?)\s*(-?\d[\d,]*(?:\.\d+)?)\s*(.*)$", arg)
    if not match:
        return {}
    prefix, number, suffix = match.groups()
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return {}
    return {"value": value, "prefix": prefix.strip(), "suffix": suffix.strip(),
            "decimals": 1 if "." in number else 0}


def _chart_data(arg: str) -> Dict[str, Any]:
    """'Title: A=40, B=12' or 'A=40, B=12' → labels/values (+ optional title)."""
    import re
    title = None
    body = arg
    if ":" in arg and "=" not in arg.split(":", 1)[0]:
        title, body = arg.split(":", 1)
    labels: List[str] = []
    values: List[float] = []
    unit = ""
    for part in re.split(r"[,;]", body):
        if "=" not in part:
            continue
        label, value = part.split("=", 1)
        match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*(\S*)", value)
        if not match:
            continue
        try:
            values.append(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
        labels.append(label.strip())
        unit = unit or match.group(2)
    if len(values) < 2:
        return {}
    return {"labels": labels, "values": values, "unit": unit,
            "title": title.strip() if title else None}
