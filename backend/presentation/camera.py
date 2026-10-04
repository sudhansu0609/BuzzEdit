"""Gradual camera moves for genres that live on slowness (horror).

A B-roll clip is 2-4 s on screen, and a video model asked for "a zoom in" or "a push toward the
door" fits the whole move into those few seconds: on a horror story the result reads as a snap,
not a creep. The model's speed cannot be set; the timeline's can. So for those genres a clip whose
prompt asks for a zoom, push or pull is generated with the camera locked off, and the move is put
back on the timeline as a slow zoom in the asked direction across the clip's whole time on screen
(placement.place_broll).
"""

import re
from typing import Iterable, Optional

#: Genres whose zooms must creep rather than snap.
GRADUAL_GENRES = {"horror"}

# Zoom depth per second on screen, and its bounds: about 2% a second, 4-10% per clip. The
# stills' Ken Burns moves 1-2% a second; a requested move should read as deliberate, still slow.
GRADUAL_RATE = 0.02
GRADUAL_MIN = 0.04
GRADUAL_MAX = 0.10

STEADY = "static locked-off shot, the frame does not move"
NO_MOVE_NEGATIVE = "fast zoom, sudden zoom, rapid camera movement, whip pan, shaky camera"

_SPEED = (r"(?:(?:very|extremely|super|really)\s+)?(?:slow(?:ly)?|gentle|gently|gradual(?:ly)?|steady|"
          r"steadily|smooth(?:ly)?|quick(?:ly)?|fast|sudden(?:ly)?|rapid(?:ly)?|creeping|subtle|subtly|"
          r"dramatic(?:ally)?)")
_LEAD = r"(?:\b(?:an?|the)\s+)?(?:\b" + _SPEED + r"[\s,]+)*"
_TAIL = r"(?:\s+" + _SPEED + r")?(?:\s+(?:in\s+on|on(?:to)?|toward|towards|into|to|at|from|across|over)\b)?"
# Only the move, its speed and its preposition go; what it moves toward ("her face") is the
# subject of the shot and stays.
_OUT = re.compile(
    _LEAD + r"\b(?:zoom(?:s|ing|ed)?[\s-]+out|pull(?:s|ing|ed)?[\s-]+(?:back|out|away)|"
    r"dolly(?:ing)?[\s-]+out|widen(?:s|ing)?(?:\s+out)?)\b" + _TAIL,
    re.IGNORECASE)
_IN = re.compile(
    _LEAD + r"\b(?:zoom(?:s|ing|ed)?(?:[\s-]+in)?|push(?:es|ing|ed)?(?:[\s-]+in)?|"
    r"dolly(?:ing)?(?:[\s-]+in)?|creep(?:s|ing)?\s+(?:closer|in)|mov(?:e|es|ing)\s+(?:closer|in))\b" + _TAIL,
    re.IGNORECASE)


def camera_move(prompt: Optional[str]) -> Optional[str]:
    """"in", "out" or None: the move a prompt asks the camera for."""
    if not prompt:
        return None
    if _OUT.search(prompt):
        return "out"
    if _IN.search(prompt):
        return "in"
    return None


def steady_prompt(prompt: str) -> str:
    """`prompt` with its zoom/push/pull phrases taken out and a locked-off camera asked for."""
    text = _IN.sub(" ", _OUT.sub(" ", prompt))
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"([,;])\s*(?:[,;]\s*)+", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.")
    text = text[:1].upper() + text[1:] if text and prompt[:1].isupper() else text
    return f"{text}, {STEADY}" if text else STEADY


def gradual_depth(duration_s: float) -> float:
    return max(GRADUAL_MIN, min(GRADUAL_MAX, GRADUAL_RATE * max(0.0, duration_s)))


def steady_for_genre(beats: Iterable, genre: Optional[str]) -> int:
    """For a gradual genre, lock the camera of every video beat that asks for a zoom and
    remember the direction on the beat (`camera_move`). Returns how many beats changed."""
    if (genre or "").lower() not in GRADUAL_GENRES:
        return 0
    changed = 0
    for beat in beats:
        if getattr(beat, "kind", None) != "broll_video" or getattr(beat, "camera_move", None):
            continue
        move = camera_move(beat.video_prompt)
        if not move:
            continue
        beat.camera_move = move
        beat.video_prompt = steady_prompt(beat.video_prompt)
        if NO_MOVE_NEGATIVE not in (beat.negative_prompt or ""):
            beat.negative_prompt = (f"{beat.negative_prompt}, {NO_MOVE_NEGATIVE}"
                                    if beat.negative_prompt else NO_MOVE_NEGATIVE)
        changed += 1
    return changed
