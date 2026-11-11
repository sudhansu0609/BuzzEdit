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
# A kinetic-text beat without a `| seconds=N` hold: long enough for the words
# to spring in and settle. presentation/motion.py clamps the beat's actual
# render length to 2.5-12s on top of this.
DIRECTIVE_KINETIC_SECONDS = 5.0
_KINETIC_STYLES = ("slam", "stack", "reveal", "typewriter")
# A stick-figure beat without a `| seconds=N` hold: long enough for a pose to
# land (~40% in) and keep moving a while. presentation/motion.py clamps the
# beat's actual render length to 2.5-12s on top of this.
DIRECTIVE_STICK_SECONDS = 6.0
# A paper-collage beat without a `| seconds=N` hold: long enough for 2-4
# cutouts to step in (landing by ~40%, Architecture) and the doodled arrows to
# finish drawing. presentation/motion.py clamps the beat's actual render
# length to 2.5-12s on top of this.
DIRECTIVE_COLLAGE_SECONDS = 7.0
# A flow-diagram beat without a `| seconds=N` hold: long enough for 2-6 boxes
# and their arrows to draw in order (landing by ~40%, Architecture).
# presentation/motion.py clamps the beat's actual render length to 2.5-12s on
# top of this.
DIRECTIVE_FLOW_SECONDS = 7.0
# An icon-list beat without a `| seconds=N` hold: long enough for 2-6
# rows/cells to stagger-pop in. presentation/motion.py clamps the beat's
# actual render length to 2.5-12s on top of this.
DIRECTIVE_ICONS_SECONDS = 6.0
# A ui-mockup beat without a `| seconds=N` hold: long enough for the card to
# float in and its content to land (~40%, Architecture). presentation/
# motion.py clamps the beat's actual render length to 2.5-12s on top of this.
DIRECTIVE_UI_SECONDS = 6.0
_UI_VARIANTS = ("post", "forum", "chat", "search", "headline")
# A callout beat without a `| seconds=N` hold: long enough for the push-in
# and the hand-drawn mark/magnifier to land (~40%, Architecture).
# presentation/motion.py clamps the beat's actual render length to 2.5-12s on
# top of this.
DIRECTIVE_CALLOUT_SECONDS = 6.0
_CALLOUT_MARKS = ("circle", "arrow", "magnify")
_CALLOUT_REGIONS = ("top_left", "top", "top_right", "left", "center", "right",
                    "bottom_left", "bottom", "bottom_right")
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
# shape_wipe/iris/ink/bars/paper_tear (MOTION_GRAPHICS_PLAN.md M3) are the
# Remotion-rendered transition stingers -- same `[fx: ...]` grammar, routed to
# presentation.motion_transitions instead of an AtmosphereEffect adjustment
# (presentation/director.py::_apply_script_fx).
_FX_DEFAULT_DUR = {"flash": 0.4, "whiteflash": 0.5, "shutter": 0.6, "shake": 0.6, "glitch": 0.6,
                   "strobe": 0.8, "redaction": 0.8, "flicker": 1.2,
                   "shape_wipe": 0.8, "iris": 0.8, "ink": 0.8, "bars": 0.8, "paper_tear": 0.8}


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


# `map`/`chart`/`timeline` are timed like a still picture, never a clip and
# never split into continuation parts (BuzzcafStudio times them this way so
# they fill an exact share of the video): a trailing `| seconds=N` on the arg.
# Pulled off with a regex, before each kind's own parsing, rather than through
# `_parts` -- a chart body already contains '=' and ',' and would misparse.
_TRAILING_HOLD_RE = re.compile(r"\s*\|\s*seconds\s*=\s*(\S+)\s*$", re.IGNORECASE)


def _strip_trailing_hold(arg: str) -> Tuple[str, Optional[float]]:
    """Returns (arg with a trailing `| seconds=N` removed, hold seconds or
    None). Without the option, `arg` comes back unchanged."""
    match = _TRAILING_HOLD_RE.search(arg)
    if not match:
        return arg, None
    return arg[:match.start()].rstrip(), _hold_seconds({"seconds": match.group(1)})


def _stick_options(arg: str) -> Dict[str, str]:
    """`[stick: action=swim | scene=escaping a flooded house | prop=ship |
    era=ancient_greece | outfit=chiton+laurel+scroll+beard ; toga+staff |
    figures=2 | bg=ancient Greek library courtyard, marble columns |
    caption=Help!]` (STICK_WORLD_PLAN.md SW2 adds `era`/`outfit`/`bg`) --
    every field is `key=value`, but unlike `_parts` (whose option values are
    a single token) `scene`/`outfit`/`bg`/`caption` are short phrases, so
    each `|`-separated piece is split on only its first `=` and the rest
    kept whole, case preserved."""
    options: Dict[str, str] = {}
    for piece in arg.split("|"):
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return options


def _collage_options(arg: str) -> Tuple[List[str], Dict[str, str]]:
    """`[collage: prompt A ; prompt B ; prompt C | caption=Three places he
    never returned | seconds=8]` (MOTION_GRAPHICS_PLAN.md M5) -- the first
    `|`-separated piece is always the semicolon-separated list of 2-4 picture
    prompts; every later piece is `key=value` with the value kept whole (a
    caption may have spaces), the same convention `_stick_options` uses."""
    pieces = arg.split("|")
    prompts = [p.strip() for p in pieces[0].split(";") if p.strip()] if pieces else []
    options: Dict[str, str] = {}
    for piece in pieces[1:]:
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return prompts, options


def _list_and_options(arg: str) -> Tuple[List[str], Dict[str, str]]:
    """`item ; item ; ... | key=value | key=value` (MOTION_GRAPHICS_PLAN.md
    M9's `[flow:]`/`[icons:]`) -- the same shape `_collage_options` uses for
    `[collage:]`: the first `|`-piece is the semicolon-separated item list,
    every later piece is `key=value` with the value kept whole."""
    pieces = arg.split("|")
    items = [p.strip() for p in pieces[0].split(";") if p.strip()] if pieces else []
    options: Dict[str, str] = {}
    for piece in pieces[1:]:
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return items, options


def _ui_options(arg: str) -> Tuple[str, Dict[str, str]]:
    """`[ui: post | text=... | name=... | handle=... | meta=...]`
    (MOTION_GRAPHICS_PLAN.md M10) -- the first `|`-separated piece is the
    bare variant name, every later piece is `key=value` with the value kept
    whole (text/name/meta may have spaces), the same convention
    `_stick_options` uses."""
    pieces = arg.split("|")
    variant = pieces[0].strip().lower() if pieces else ""
    options: Dict[str, str] = {}
    for piece in pieces[1:]:
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return variant, options


def _callout_options(arg: str) -> Tuple[str, Dict[str, str]]:
    """`[callout: <image prompt> | mark=... | at=... | label=...]`
    (MOTION_GRAPHICS_PLAN.md M10) -- the first `|`-separated piece is the
    image prompt, every later piece is `key=value` with the value kept
    whole, the same shape `_collage_options` uses for `[collage:]`."""
    pieces = arg.split("|")
    prompt = pieces[0].strip() if pieces else ""
    options: Dict[str, str] = {}
    for piece in pieces[1:]:
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return prompt, options


def _face_main(options: Dict[str, str]) -> bool:
    """`face=main` on a `[broll:]`/`[video:]` direction: the shot shows the
    story's main character, so it gets their face (presentation/character.py).
    A direction without it is never face-swapped."""
    return options.get("face") == "main"


# `own=C:\My Clips\a b.mp4` on a `[video:]` direction (OWN_FOOTAGE_PLAN.md): the
# creator's own clip. The path is cut out with a regex of its own, before
# `_parts`, since it holds spaces and a drive colon and `_parts` lower-cases
# values and takes one word. It runs to the next `| key=` or the end.
_OWN_RE = re.compile(r"(?:^|\|)\s*own\s*=\s*(.+?)\s*(?=\|\s*[a-z_]+\s*=|$)",
                     re.IGNORECASE | re.DOTALL)
NATSOUND_DEFAULT = 0.15


def _own_clip(arg: str) -> Tuple[str, Optional[str]]:
    """Returns (arg without its `own=` option, the path or None)."""
    match = _OWN_RE.search(arg)
    if not match or not match.group(1).strip():
        return arg, None
    rest = (arg[:match.start()] + arg[match.end():]).strip(" |")
    return rest, match.group(1).strip().strip('"')


def _own_numbers(options: Dict[str, str]) -> Tuple[float, float]:
    """(`in=` seconds into the clip, `natsound=` gain 0..1) of an `own=` shot."""
    def _num(key: str, default: float) -> float:
        try:
            return float((options.get(key) or "").rstrip("s"))
        except ValueError:
            return default
    return max(0.0, _num("in", 0.0)), max(0.0, min(1.0, _num("natsound", NATSOUND_DEFAULT)))


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


def _diegetic_cue(at: float, effect: str, options: Dict[str, str]) -> Optional[Any]:
    """`[sfx: footsteps on gravel | kind=accent | level=subtle | seconds=3]`: a
    scene sound in words (at most 8), not a tagged one-shot."""
    from .sound_design import DiegeticCue
    words = effect.split()[:8]
    if not words:
        return None
    kind = options.get("kind", "accent")
    level = options.get("level", "normal")
    seconds: Optional[float] = None
    if options.get("seconds"):
        try:
            seconds = float(options["seconds"].rstrip("s")) or None
        except ValueError:
            seconds = None
    return DiegeticCue(at_s=at, effect=" ".join(words),
                       kind=kind if kind in ("accent", "bed") else "accent",
                       level=level if level in ("subtle", "normal") else "normal",
                       seconds=seconds if seconds and seconds > 0 else None, source="script")


def directive_beats(ctx: ScriptContext, program: Program,
                    settings: PresentationSettings) -> Dict[str, Any]:
    """Turn the script's stage directions into beats and side requests.

    Returns {"beats": [...], "sfx": [(at, tag)], "title": str|None,
             "moods": {paragraph_index: mood}, "music_mood": str|None,
             "music_cues": [(at, cue text)] -- each `[music: ...]` in order,
             "diegetic": [DiegeticCue] -- each `[sfx: ...]` naming a scene sound
             rather than a known tag (presentation/sound_design.py)}.
    Every beat carries priority 1.0 and origin "script": the user asked for
    it by name, so it wins any budget contest with the model's own ideas.
    """
    beats: List[Beat] = []
    sfx: List[Tuple[float, str]] = []
    diegetic: List[Any] = []
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
            body, hold = _strip_trailing_hold(arg)
            fields, _ = _parts(body)
            data = _timeline_data(fields[0]) if fields else {"ticks": [], "active": 0}
            if hold:
                data["hold"] = True
            beats.append(Beat(kind="timeline", end_s=_end(at, hold or card + 1.0, program),
                              planned_duration_s=hold,
                              text=fields[1] if len(fields) > 1 else "",
                              subtext=fields[2] if len(fields) > 2 else None, data=data, **common))
        elif d.kind == "video":
            arg, own_path = _own_clip(arg)
            fields, options = _parts(arg)
            hold = _hold_seconds(options)
            # A timed or face=main shot drops its options from the prompt; any
            # other direction keeps its text whole, as it always has.
            prompt = (" | ".join(fields) or arg) if (hold or own_path or _face_main(options)) else arg
            own = {}
            if own_path:
                own_in, natsound = _own_numbers(options)
                own = dict(own_path=own_path, own_in_s=own_in, natsound=natsound)
            beats.append(Beat(kind="broll_video", end_s=_end(at, hold or cut + 1.0, program),
                              planned_duration_s=hold, video_prompt=prompt, image_prompt=prompt,
                              data={"hold": True} if hold else {},
                              shows_character=_face_main(options), **own, **common))
        elif d.kind == "kinetic":
            body, hold = _strip_trailing_hold(arg)
            fields, options = _parts(body)
            text = " | ".join(fields) if fields else body
            kinetic_style = options.get("style", "slam").lower()
            if kinetic_style not in _KINETIC_STYLES:
                kinetic_style = "slam"
            data = {"style": kinetic_style}
            if hold:
                data["hold"] = True
            beats.append(Beat(kind="kinetic_text", end_s=_end(at, hold or DIRECTIVE_KINETIC_SECONDS, program),
                              planned_duration_s=hold, text=text, data=data, **common))
        elif d.kind == "stick":
            body, hold = _strip_trailing_hold(arg)
            options = _stick_options(body)
            action = options.get("action", "idle").lower()
            prop = options.get("prop", "none").lower()
            scene = options.get("scene", "")
            caption = (options.get("caption") or "")[:80] or None
            # era/outfit/bg (STICK_WORLD_PLAN.md SW2): `outfit` is one spec
            # per figure, `;` between figures; `figures` defaults to the
            # number of outfits named, else 1 (unchanged for an old
            # directive naming neither).
            era = (options.get("era") or "").strip().lower() or None
            outfits = [o.strip().lower() for o in (options.get("outfit") or "").split(";")
                      if o.strip()][:3]
            default_figures = len(outfits) if outfits else 1
            try:
                figures = max(1, min(3, int(options.get("figures", str(default_figures)))))
            except ValueError:
                figures = default_figures
            bg = (options.get("bg") or "").strip() or None
            data = {"action": action, "scene": scene, "prop": prop, "figures": figures}
            if era:
                data["era"] = era
            if outfits:
                data["outfits"] = outfits
            if bg:
                data["bg"] = bg
            if caption:
                data["caption"] = caption
            if hold:
                data["hold"] = True
            beats.append(Beat(kind="stick_scene", end_s=_end(at, hold or DIRECTIVE_STICK_SECONDS, program),
                              planned_duration_s=hold, text=caption or "", data=data, **common))
        elif d.kind == "collage":
            body, hold = _strip_trailing_hold(arg)
            prompts, options = _collage_options(body)
            prompts = prompts[:4]
            if len(prompts) >= 2:
                caption = (options.get("caption") or "")[:80] or None
                data = {"prompts": prompts}
                if caption:
                    data["caption"] = caption
                if hold:
                    data["hold"] = True
                beats.append(Beat(kind="paper_collage",
                                  end_s=_end(at, hold or DIRECTIVE_COLLAGE_SECONDS, program),
                                  planned_duration_s=hold, text=caption or "", data=data, **common))
        elif d.kind == "flow":
            body, hold = _strip_trailing_hold(arg)
            steps, options = _list_and_options(body)
            steps = steps[:6]
            if len(steps) >= 2:
                layout = options.get("layout", "chain").lower()
                if layout not in ("chain", "cycle", "branch"):
                    layout = "chain"
                data = {"steps": steps, "layout": layout}
                if hold:
                    data["hold"] = True
                beats.append(Beat(kind="flow_diagram", end_s=_end(at, hold or DIRECTIVE_FLOW_SECONDS, program),
                                  planned_duration_s=hold, text=" -> ".join(steps)[:80], data=data, **common))
        elif d.kind == "icons":
            body, hold = _strip_trailing_hold(arg)
            items, options = _list_and_options(body)
            items = items[:6]
            if len(items) >= 2:
                mode = options.get("mode", "list").lower()
                if mode not in ("list", "pros_cons"):
                    mode = "list"
                data = {"items": items, "mode": mode}
                if hold:
                    data["hold"] = True
                beats.append(Beat(kind="icon_list", end_s=_end(at, hold or DIRECTIVE_ICONS_SECONDS, program),
                                  planned_duration_s=hold, text=", ".join(items)[:80], data=data, **common))
        elif d.kind == "ui":
            body, hold = _strip_trailing_hold(arg)
            variant, options = _ui_options(body)
            if variant not in _UI_VARIANTS:
                variant = "post"
            ui_text = options.get("text", "")
            data = {
                "variant": variant, "text": ui_text,
                "name": options.get("name", ""), "handle": options.get("handle", ""),
                "meta": options.get("meta", ""),
            }
            if hold:
                data["hold"] = True
            beats.append(Beat(kind="ui_mockup", end_s=_end(at, hold or DIRECTIVE_UI_SECONDS, program),
                              planned_duration_s=hold, text=ui_text[:80] or variant, data=data, **common))
        elif d.kind == "callout":
            body, hold = _strip_trailing_hold(arg)
            prompt, options = _callout_options(body)
            if prompt:
                mark = options.get("mark", "circle").lower()
                if mark not in _CALLOUT_MARKS:
                    mark = "circle"
                region = options.get("at", "center").lower()
                if region not in _CALLOUT_REGIONS:
                    region = "center"
                label = (options.get("label") or "")[:40] or None
                data = {"mark": mark, "at": region}
                if label:
                    data["label"] = label
                if hold:
                    data["hold"] = True
                beats.append(Beat(kind="callout", end_s=_end(at, hold or DIRECTIVE_CALLOUT_SECONDS, program),
                                  planned_duration_s=hold, image_prompt=prompt, text=label or "",
                                  data=data, **common))
        elif d.kind == "map":
            place, hold = _strip_trailing_hold(arg)
            beats.append(Beat(kind="map", end_s=_end(at, hold or cut, program),
                              planned_duration_s=hold, place=place, text=place,
                              data={"hold": True} if hold else {}, **common))
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
            body, hold = _strip_trailing_hold(arg)
            data = _chart_data(body)
            if data:
                if hold:
                    data["hold"] = True
                beats.append(Beat(kind="chart", end_s=_end(at, hold or cut + 1.0, program),
                                  planned_duration_s=hold, data=data,
                                  text=data.get("title") or body[:40], **common))
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
            # A known tag (whoosh, stinger...) is a one-shot as before; any
            # other effect is a scene sound.
            from .sound import SFX_TAGS
            fields, options = _parts(arg)
            tag = (fields[0] if fields else "").lower().replace(" ", "_")
            if tag in SFX_TAGS:
                sfx.append((at, tag))
            elif fields:
                cue = _diegetic_cue(at, fields[0], options)
                if cue is not None:
                    diegetic.append(cue)
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
            "music_mood": music_mood, "music_cues": music_cues, "fx": fx,
            "diegetic": diegetic}


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
