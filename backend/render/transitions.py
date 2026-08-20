"""Transitions between programme clips.

The programme is built by concatenating cut segments, which gives hard cuts. A
real transition has to *overlap* two streams instead, which is what `xfade` does —
so the V1 chain is assembled as a mix of the two: runs of hard-cut clips are
concatenated into groups, and groups are joined with `xfade`.

The timing rule is the whole trick. `xfade` places the overlap at `offset` seconds
into the first input and the result is `a + b - duration` long — so a plain xfade
shortens the programme by its own length, and since audio concatenates at full
length, every dissolve walked picture ahead of sound. The compiler therefore
extends both sides of each junction with cloned edge frames (`tpad`) and runs the
xfade over the cloned material: the programme keeps its full length, audio needs
no crossfade at all, and A/V sync holds by construction.
"""

from typing import Dict, List, Tuple

# Every transition this FFmpeg build supports, grouped so the UI can present them
# as something other than a wall of 57 names.
CATALOGUE: Dict[str, List[str]] = {
    "Dissolve": ["fade", "dissolve", "fadeblack", "fadewhite", "fadegrays",
                 "fadefast", "fadeslow", "distance", "hblur", "pixelize"],
    "Wipe": ["wipeleft", "wiperight", "wipeup", "wipedown",
             "wipetl", "wipetr", "wipebl", "wipebr"],
    "Slide": ["slideleft", "slideright", "slideup", "slidedown",
              "coverleft", "coverright", "coverup", "coverdown",
              "revealleft", "revealright", "revealup", "revealdown"],
    "Smooth": ["smoothleft", "smoothright", "smoothup", "smoothdown",
               "circleopen", "circleclose", "vertopen", "vertclose",
               "horzopen", "horzclose", "radial"],
    "Shape": ["circlecrop", "rectcrop", "diagtl", "diagtr", "diagbl", "diagbr",
              "hlslice", "hrslice", "vuslice", "vdslice"],
    "Motion": ["zoomin", "squeezeh", "squeezev",
               "hlwind", "hrwind", "vuwind", "vdwind"],
}

ALL = sorted({name for group in CATALOGUE.values() for name in group})

# Kept short: a transition longer than this on a talking-head cut reads as a fault.
MAX_DURATION = 3.0
MIN_DURATION = 0.08


def is_valid(name: str) -> bool:
    return name in ALL


def clamp_duration(duration: float, first: float, second: float) -> float:
    """A transition cannot be longer than the clips it joins.

    `xfade` needs both inputs to cover the overlap; asking for a one-second
    dissolve between two half-second clips produces a broken graph rather than a
    shorter transition, so it is clamped here instead.
    """
    limit = max(0.0, min(first, second) - 0.04)
    return max(0.0, min(duration, MAX_DURATION, limit))


def build_video(first_label: str, second_label: str, out_label: str,
                name: str, duration: float, first_duration: float) -> str:
    offset = max(0.0, first_duration - duration)
    return (f"{first_label}{second_label}xfade=transition={name}:"
            f"duration={duration:.3f}:offset={offset:.3f}{out_label}")


def build_audio(first_label: str, second_label: str, out_label: str,
                duration: float) -> str:
    """Audio crossfade. Unused on the programme path — video transitions no
    longer shorten the stream, so A1 concatenates plainly (its segments carry
    their own 8ms declick fades). Kept for callers that join free audio."""
    return (f"{first_label}{second_label}acrossfade=d={duration:.3f}"
            f":c1=tri:c2=tri{out_label}")


def catalogue() -> List[Dict[str, object]]:
    """Grouped listing for the UI."""
    return [{"group": group, "transitions": names} for group, names in CATALOGUE.items()]
