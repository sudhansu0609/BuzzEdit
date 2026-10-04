"""The user's script, as the presentation pass sees it.

`asr.script_align` does the aligning; this module is the bridge to the pass:
it applies a stored script to a project's timeline words (fixing caption
spelling), projects the script's paragraphs and stage directions from source
time into the CUT programme's time, and turns the directions into beats.

A script is optional. Everything here returns empty structures when there is
none, and the planner behaves exactly as before.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from asr import script_align
from timeline.authoring import _program_segments, source_to_timeline_frame
from timeline.schema import Timeline, frame_to_time, time_to_frame

from .models import BROLL_SECONDS_CAP, BROLL_SECONDS_FLOOR, Beat, PresentationSettings, Program, Topic

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


# Effect windows the script places. Hit effects are short punctuation; the rest
# are ambient textures held longer. `atmos` defaults longer still than `fx`.
_FX_DEFAULT_DUR = {"flash": 0.4, "whiteflash": 0.5, "shutter": 0.6, "shake": 0.6, "glitch": 0.6,
                   "strobe": 0.8, "redaction": 0.8, "flicker": 1.2}


def _parse_fx(kind: str, arg: str) -> Dict[str, Any]:
    """Parse `[fx: name k=v ...]`, `[atmos: name ...]` or `[grade: k=v ...]`.

    The first bare token is the effect name; `key=value` tokens are options.
    For `grade` the options are colour adjustments (saturation, contrast, …).
    """
    name = ""
    params: Dict[str, str] = {}
    for tok in arg.split():
        if "=" in tok:
            key, value = tok.split("=", 1)
            params[key.strip().lower()] = value.strip()
        elif not name:
            name = tok.strip().lower()

    def _num(key: str, default: float) -> float:
        try:
            return float(params[key])
        except (KeyError, ValueError):
            return default

    record: Dict[str, Any] = {
        "kind": kind,
        "effect": name,
        "intensity": max(0.0, min(1.0, _num("intensity", 0.5))),
        "speed": max(0.1, min(4.0, _num("speed", 1.0))),
        "color": params.get("color") or None,
    }
    if kind == "grade":
        grade: Dict[str, float] = {}
        for key in ("saturation", "contrast", "brightness", "vignette",
                    "temperature", "gamma", "exposure", "sharpen"):
            if key in params:
                try:
                    grade[key] = float(params[key])
                except ValueError:
                    pass
        record["grade"] = grade
    record["dur"] = max(0.2, _num("dur", _FX_DEFAULT_DUR.get(name, 6.0 if kind == "atmos" else 3.0)))
    return record


_OPTION_RE = re.compile(r"^\s*([a-z_]+)\s*=\s*(\S+)\s*$", re.IGNORECASE)


def _parts(arg: str) -> Tuple[List[str], Dict[str, str]]:
    """`a | b | key=value` -> (["a", "b"], {"key": "value"}): the pipe-separated
    fields of a designed-graphic directive, with `key=value` fields pulled out
    as options wherever they sit."""
    fields: List[str] = []
    options: Dict[str, str] = {}
    for piece in arg.split("|"):
        match = _OPTION_RE.match(piece)
        if match:
            options[match.group(1).lower()] = match.group(2).strip().lower()
        elif piece.strip():
            fields.append(piece.strip())
    return fields, options


def _hold_seconds(options: Dict[str, str]) -> Optional[float]:
    """`seconds=6.5` on a `[broll:]`/`[video:]` direction: the caller timed the
    shot itself (BuzzcafStudio does, so clips fill an exact share of the
    video) and it runs exactly that long, back to back with the next shot if
    that is where the next one starts. None leaves the fixed default."""
    raw = options.get("seconds") or options.get("hold")
    if not raw:
        return None
    try:
        value = float(raw.rstrip("s"))
    except ValueError:
        return None
    return round(max(BROLL_SECONDS_FLOOR, min(BROLL_SECONDS_CAP, value)), 3)


def _face_main(options: Dict[str, str]) -> bool:
    """`face=main` on a `[broll:]`/`[video:]` direction: the shot shows the
    story's main character, so it gets their face (presentation/character.py).
    A direction without it is never face-swapped."""
    return options.get("face") == "main"


def _speaker_on(options: Dict[str, str]) -> bool:
    return options.get("speaker", "on") not in ("off", "no", "false", "0")


def _timeline_data(field: str) -> Dict[str, Any]:
    """"1975, 1980, *1985*, 1990" -> ticks, with the starred one active (the
    last one when none is starred)."""
    ticks, active = [], None
    for raw in [t.strip() for t in field.replace(";", ",").split(",") if t.strip()]:
        if raw.startswith("*") and raw.endswith("*") and len(raw) > 2:
            active = len(ticks)
            raw = raw.strip("*").strip()
        ticks.append(raw)
    return {"ticks": ticks[:9], "active": active if active is not None else max(0, len(ticks) - 1)}


def directive_beats(ctx: ScriptContext, program: Program,
                    settings: PresentationSettings) -> Dict[str, Any]:
    """Turn the script's stage directions into beats and side requests.

    Returns {"beats": [...], "sfx": [(at, tag)], "title": str|None,
             "moods": {paragraph_index: mood}, "music_mood": str|None,
             "music_cues": [(at, cue text)] -- each `[music: ...]` in order}.
    Every beat carries priority 1.0 and origin "script": the user asked for
    it by name, so it wins any budget contest with the model's own ideas.
    """
    beats: List[Beat] = []
    sfx: List[Tuple[float, str]] = []
    moods: Dict[int, str] = {}
    title: Optional[str] = None
    music_mood: Optional[str] = None
    music_cues: List[Tuple[float, str]] = []
    fx: List[Dict[str, Any]] = []
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
            fields, options = _parts(arg)
            data = {"look": options["look"]} if options.get("look") else {}
            hold = _hold_seconds(options)
            beats.append(Beat(kind="broll_image", end_s=_end(at, hold or cut, program),
                              planned_duration_s=hold, image_prompt=" | ".join(fields) or arg,
                              data=dict(data, hold=True) if hold else data,
                              shows_character=_face_main(options), **common))
        elif d.kind == "statement":
            fields, options = _parts(arg)
            beats.append(Beat(kind="statement_card", end_s=_end(at, card + 1.0, program),
                              text=" ".join(fields), data={"with_speaker": _speaker_on(options)},
                              **common))
        elif d.kind == "canvas":
            fields, options = _parts(arg)
            data = {"with_speaker": _speaker_on(options)}
            if len(fields) > 1:
                data["caption"] = fields[1]
            if options.get("look"):
                data["look"] = options["look"]
            beats.append(Beat(kind="canvas_card", end_s=_end(at, cut, program),
                              image_prompt=fields[0] if fields else arg, data=data, **common))
        elif d.kind == "pills":
            items = [p.strip() for p in arg.replace("|", ";").split(";") if p.strip()]
            if items:
                beats.append(Beat(kind="pill_labels", end_s=_end(at, 1.2 + 0.6 * len(items), program),
                                  text=items[0], data={"items": items[:4]}, **common))
        elif d.kind == "point":
            fields, _ = _parts(arg)
            number, text = (fields[0], fields[1]) if len(fields) > 1 else ("1", fields[0] if fields else arg)
            beats.append(Beat(kind="numbered_point", end_s=_end(at, card, program),
                              text=text, data={"number": number[:3]}, **common))
        elif d.kind == "name":
            fields, _ = _parts(arg)
            beats.append(Beat(kind="name_title", end_s=_end(at, card, program),
                              text=fields[0] if fields else arg,
                              subtext=fields[1] if len(fields) > 1 else None, **common))
        elif d.kind == "source_quote":
            fields, _ = _parts(arg)
            source, quote = (fields[0], fields[1]) if len(fields) > 1 else ("", fields[0] if fields else arg)
            beats.append(Beat(kind="source_quote", end_s=_end(at, card + 1.5, program),
                              text=quote, data={"source": source},
                              image_prompt=fields[2] if len(fields) > 2 else None, **common))
        elif d.kind == "document":
            fields, _ = _parts(arg)
            fmt = fields[0].lower() if fields and fields[0].lower() in ("web", "book") else "web"
            rest = fields[1:] if fields and fields[0].lower() in ("web", "book") else fields
            if len(rest) >= 3:
                site, title, body = rest[0], rest[1], " ".join(rest[2:])
            else:
                site, title, body = "", (rest[0] if rest else arg), (rest[1] if len(rest) > 1 else "")
            beats.append(Beat(kind="document", end_s=_end(at, cut + 1.0, program),
                              text=title, subtext=body or None,
                              data={"format": fmt, "site": site}, **common))
        elif d.kind == "timeline":
            fields, _ = _parts(arg)
            data = _timeline_data(fields[0]) if fields else {"ticks": [], "active": 0}
            beats.append(Beat(kind="timeline", end_s=_end(at, card + 1.0, program),
                              text=fields[1] if len(fields) > 1 else "",
                              subtext=fields[2] if len(fields) > 2 else None, data=data, **common))
        elif d.kind == "video":
            fields, options = _parts(arg)
            hold = _hold_seconds(options)
            # A timed or face=main shot drops its options from the prompt; any
            # other direction keeps its text whole, as it always has.
            prompt = (" | ".join(fields) or arg) if (hold or _face_main(options)) else arg
            beats.append(Beat(kind="broll_video", end_s=_end(at, hold or cut + 1.0, program),
                              planned_duration_s=hold, video_prompt=prompt, image_prompt=prompt,
                              data={"hold": True} if hold else {},
                              shows_character=_face_main(options), **common))
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
        elif d.kind == "newspaper":
            data = _newspaper_data(arg)
            beats.append(Beat(kind="newspaper", end_s=_end(at, cut + 1.0, program),
                              data=data, text=data.get("headline") or arg[:60], **common))
        elif d.kind == "case_file":
            data = _case_file_data(arg)
            beats.append(Beat(kind="case_file", end_s=_end(at, cut + 1.0, program),
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
            music_cues.append((at, arg))
        elif d.kind in ("fx", "atmos", "grade"):
            record = _parse_fx(d.kind, arg)
            record["at"] = at
            record["dur"] = min(record["dur"], max(0.2, program.duration_s - at))
            fx.append(record)

    beats.sort(key=lambda b: b.start_s)
    return {"beats": beats, "sfx": sfx, "title": title, "moods": moods,
            "music_mood": music_mood, "music_cues": music_cues, "fx": fx}


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
    """'Title: A=40, B=12' or 'A=40, B=12' → labels/values (+ optional title).

    A leading type word picks the shape: '[chart: pie: A=40, B=12]',
    '[chart: line: 2019=10, 2020=22]', '[chart: bar: ...]' ('graph' == line).
    Without one it stays a bar chart, exactly as before.
    """
    import re
    chart_type = ""
    body = arg.strip()
    type_match = re.match(r"(?i)^(pie|line|graph|bar)\b[\s:]*", body)
    if type_match:
        picked = type_match.group(1).lower()
        chart_type = "line" if picked == "graph" else picked
        body = body[type_match.end():]
    title = None
    if ":" in body and "=" not in body.split(":", 1)[0]:
        title, body = body.split(":", 1)
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
    data: Dict[str, Any] = {"labels": labels, "values": values, "unit": unit,
                            "title": title.strip() if title else None}
    if chart_type:
        data["chart_type"] = chart_type
    return data


def _newspaper_data(arg: str) -> Dict[str, Any]:
    """'MASTHEAD :: Headline :: Dateline' or 'Headline | Dateline' → newspaper.

    The headline is the only required part. '::' names a masthead first; a
    plain '|' splits headline from dateline.
    """
    text = arg.strip()
    if "::" in text:
        parts = [p.strip() for p in text.split("::")]
        masthead = parts[0] or None
        headline = parts[1] if len(parts) > 1 else parts[0]
        dateline = parts[2] if len(parts) > 2 else None
        return {"masthead": masthead, "headline": headline, "dateline": dateline,
                "highlight": headline}
    headline, dateline = _split_attribution(text, separators=("|", " — ", " - "))
    return {"headline": headline, "dateline": dateline, "highlight": headline}


def _case_file_data(arg: str) -> Dict[str, Any]:
    """'TITLE | label=value; label=value | STAMP' → a dossier.

    Only the title (or a stamp) is required; fields and stamp are optional.
    """
    import re
    chunks = [c.strip() for c in arg.split("|")]
    title = chunks[0] if chunks and chunks[0] else "CASE FILE"
    fields: List[Dict[str, str]] = []
    stamp = None
    for chunk in chunks[1:]:
        if "=" in chunk:
            for part in re.split(r"[;,]", chunk):
                if "=" in part:
                    label, value = part.split("=", 1)
                    if label.strip():
                        fields.append({"label": label.strip(), "value": value.strip()})
        elif chunk:
            stamp = chunk
    data: Dict[str, Any] = {"title": title, "fields": fields}
    if stamp:
        data["stamp"] = stamp
    return data
