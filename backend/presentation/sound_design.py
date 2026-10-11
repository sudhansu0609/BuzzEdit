"""Scene sounds: what the story would really sound like, and where.

The music bed and the tagged one-shots (whoosh, pop, stinger) are decoration;
a diegetic sound is part of the scene -- footsteps on the trail, rain starting,
a jeep engine, a door, a dog. This pass reads the programme a minute at a time
(the words with their times, and what is on screen) and asks the model, as a
sound designer, for those sounds: `{"cues": [{"anchor", "effect", "kind",
"level", "seconds"}]}`. An anchor is resolved to a time the way a script
directive is -- the first time its exact words are said inside the window --
and a cue whose anchor is not there is dropped. With no model, the beats'
own prompts give a few (walking -> footsteps, rain -> a rain bed...).

The script's `[sfx: <effect> | kind= | level= | seconds=]` cues (script.py)
arrive as the same `DiegeticCue`s and win over the planner's within 2 s.
`sound.plan_sound` turns them into levelled, ducked items; `sound.apply_sound`
finds each one's file through `sfx_library.resolve_effect`.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("presentation.sound_design")

SOUND_SYSTEM = (
    "You are the sound designer for a Hindi/English narrated video. You read what is said "
    "(with times) and what is on screen at each moment, and list the diegetic sounds a careful "
    "designer would add — what the scene would really sound like: footsteps, running, a door, "
    "rain starting, thunder, a jeep engine, a phone ringing, a shout, a dog, night insects. "
    "Never music, never whooshes or stingers. At most one sound per ~20 s; a bed (rain, wind, "
    "insects, engine idle) covers the stretch it belongs to; an accent lands on the word where "
    "the thing happens. The voice always stays on top; you only say subtle or normal. Answer "
    "with one JSON object and nothing else: {\"cues\": [{\"anchor\": \"exact words from the text "
    "where it happens\", \"effect\": \"footsteps running on wet leaves\", \"kind\": "
    "\"accent|bed\", \"level\": \"subtle|normal\", \"seconds\": 3}]}"
)

SOUND_SCHEMA = {
    "name": "sound_cues",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "anchor": {"type": "string"},
                        "effect": {"type": "string"},
                        "kind": {"type": "string", "enum": ["accent", "bed"]},
                        "level": {"type": "string", "enum": ["subtle", "normal"]},
                        "seconds": {"type": "number"},
                    },
                    "required": ["anchor", "effect", "kind", "level", "seconds"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["cues"],
        "additionalProperties": False,
    },
}

WINDOW_S = 60.0
# A time mark in the words every this many seconds.
MARK_EVERY_S = 5.0
MAX_EFFECT_WORDS = 8
MIN_CUE_S, ACCENT_MAX_S, BED_MAX_S = 0.5, 20.0, 60.0
CALL_CONCURRENCY = 3
# A planner cue this close to a script cue gives way to it.
SCRIPT_WINS_S = 2.0
# Not scene sounds, whatever the model says.
_NOT_DIEGETIC = {"music", "musical", "song", "melody", "soundtrack", "whoosh", "swoosh",
                 "stinger", "riser"}
_TOKEN_RE = re.compile(r"[\W_]+")


@dataclass
class DiegeticCue:
    at_s: float
    effect: str
    kind: str = "accent"          # "accent" | "bed"
    level: str = "normal"         # "subtle" | "normal"
    seconds: Optional[float] = None
    source: str = "planner"       # "script" | "planner" | "keyword"


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _token(text: str) -> str:
    return _TOKEN_RE.sub("", (text or "").casefold())


def _windows(words: Sequence[Any], window_s: float) -> List[List[Any]]:
    """The programme's words in consecutive stretches of about `window_s`."""
    out: List[List[Any]] = []
    current: List[Any] = []
    start = 0.0
    for word in words:
        if current and word.tl_start_s - start >= window_s:
            out.append(current)
            current = []
        if not current:
            start = word.tl_start_s
        current.append(word)
    if current:
        out.append(current)
    return out


def _window_text(window: Sequence[Any], beats: Sequence[Any], genre: str) -> str:
    lines: List[str] = []
    line: List[str] = []
    mark: Optional[float] = None
    for word in window:
        if mark is None or word.tl_start_s - mark >= MARK_EVERY_S:
            if line:
                lines.append(" ".join(line))
            line = [f"[{_clock(word.tl_start_s)}]"]
            mark = word.tl_start_s
        line.append(word.text)
    if line:
        lines.append(" ".join(line))
    start, end = window[0].tl_start_s, window[-1].tl_end_s
    screen: List[str] = []
    for beat in beats or []:
        if beat.start_s < end and beat.end_s > start:
            what = (beat.video_prompt or beat.image_prompt or beat.text or beat.popup_text or "")
            label = " -- ".join(x for x in ((beat.topic or "").strip(), " ".join(what.split())[:200]) if x)
            if label:
                screen.append(f"[{_clock(beat.start_s)}] {label}")
    return (f"Genre: {genre or 'general'}\n\nWords (with times):\n" + "\n".join(lines)
            + "\n\nOn screen:\n" + ("\n".join(screen) if screen else "(the speaker)")
            + "\n\nCues:")


def _anchor_time(anchor: str, window: Sequence[Any]) -> Optional[float]:
    """When the anchor's exact words are first said inside the window."""
    target = [t for t in (_token(w) for w in (anchor or "").split()) if t]
    if not target:
        return None
    said = [(t, w) for t, w in ((_token(w.text), w) for w in window) if t]
    for i in range(len(said) - len(target) + 1):
        if all(said[i + j][0] == target[j] for j in range(len(target))):
            return said[i][1].tl_start_s
    return None


def _seconds(raw: Any, kind: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return round(max(MIN_CUE_S, min(BED_MAX_S if kind == "bed" else ACCENT_MAX_S, value)), 2)


def _cue_from(raw: Any, window: Sequence[Any]) -> Optional[DiegeticCue]:
    if not isinstance(raw, dict):
        return None
    effect_words = str(raw.get("effect") or "").split()[:MAX_EFFECT_WORDS]
    if not effect_words or any(_token(w) in _NOT_DIEGETIC for w in effect_words):
        return None
    at = _anchor_time(str(raw.get("anchor") or ""), window)
    if at is None:
        return None
    kind = str(raw.get("kind") or "accent").strip().lower()
    kind = kind if kind in ("accent", "bed") else "accent"
    level = str(raw.get("level") or "normal").strip().lower()
    level = level if level in ("subtle", "normal") else "normal"
    return DiegeticCue(at_s=round(at, 3), effect=" ".join(effect_words), kind=kind,
                       level=level, seconds=_seconds(raw.get("seconds"), kind), source="planner")


# Keyword fallback: (pattern in a beat's prompt, effect, kind, seconds).
_KEYWORD_ACCENTS = (
    (re.compile(r"\brunning\b"), "running footsteps"),
    (re.compile(r"\b(walking|footsteps)\b"), "footsteps"),
    (re.compile(r"\bdoors?\b"), "door"),
    (re.compile(r"\bthunder"), "thunder"),
)
_KEYWORD_BEDS = (
    (re.compile(r"\brain\b"), "rain", None),
    (re.compile(r"\b(car|jeep)\b"), "jeep engine", 8.0),
)
KEYWORD_MIN_GAP_S = 20.0


def keyword_cues(beats: Sequence[Any]) -> List[DiegeticCue]:
    """With no model: a few subtle sounds from the beats' own prompts, at the
    beat's start -- the first accent and the first bed a prompt names, the
    same effect at most once per KEYWORD_MIN_GAP_S."""
    cues: List[DiegeticCue] = []
    last: Dict[str, float] = {}
    for beat in sorted(beats or [], key=lambda b: b.start_s):
        prompt = " ".join(x for x in (beat.video_prompt, beat.image_prompt) if x).lower()
        if not prompt:
            continue
        found = []
        accent = next((effect for pattern, effect in _KEYWORD_ACCENTS if pattern.search(prompt)), None)
        if accent:
            found.append((accent, "accent", None))
        bed = next(((effect, seconds) for pattern, effect, seconds in _KEYWORD_BEDS
                    if pattern.search(prompt)), None)
        if bed:
            found.append((bed[0], "bed", bed[1]))
        for effect, kind, seconds in found:
            at = round(max(0.0, beat.start_s), 3)
            if effect in last and at - last[effect] < KEYWORD_MIN_GAP_S:
                continue
            last[effect] = at
            cues.append(DiegeticCue(at_s=at, effect=effect, kind=kind, level="subtle",
                                    seconds=seconds, source="keyword"))
    return cues


async def plan_diegetic(program: Any, beats: Sequence[Any], ask: Any,
                        genre: str = "general", window_s: float = WINDOW_S) -> List[DiegeticCue]:
    """The scene sounds for the programme, one model call per ~`window_s` of
    words (at most CALL_CONCURRENCY at once). No model, or no usable answer
    at all, falls back to `keyword_cues`."""
    words = list(getattr(program, "words", None) or [])
    if ask is None or not words:
        return keyword_cues(beats)
    from .shotplan import _ask_all, first_json_object
    windows = _windows(words, window_s)
    calls = [(SOUND_SYSTEM, _window_text(window, beats, genre), SOUND_SCHEMA) for window in windows]
    answers = await _ask_all(ask, calls, limit=CALL_CONCURRENCY)
    cues: List[DiegeticCue] = []
    usable = 0
    for window, answer in zip(windows, answers):
        parsed = first_json_object(answer or "")
        if not isinstance(parsed, dict):
            logger.warning("Sound design: no usable answer for %s-%s", _clock(window[0].tl_start_s),
                           _clock(window[-1].tl_end_s))
            continue
        usable += 1
        for raw in parsed.get("cues") or []:
            cue = _cue_from(raw, window)
            if cue is not None:
                cues.append(cue)
    if not usable:
        return keyword_cues(beats)
    cues.sort(key=lambda c: c.at_s)
    logger.info("Sound design: %d scene sounds from %d windows", len(cues), len(windows))
    return cues


def merge_cues(script_cues: Sequence[DiegeticCue],
               planned: Sequence[DiegeticCue]) -> List[DiegeticCue]:
    """The script's cues first; a planner cue within SCRIPT_WINS_S of one is dropped."""
    merged = list(script_cues or [])
    for cue in planned or []:
        if any(abs(cue.at_s - s.at_s) < SCRIPT_WINS_S for s in script_cues or []):
            continue
        merged.append(cue)
    merged.sort(key=lambda c: c.at_s)
    return merged
