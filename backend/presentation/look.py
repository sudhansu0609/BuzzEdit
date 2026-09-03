"""The programme's look, chosen from what the video is: the master grade and
the transitions at its chapter boundaries.

Both were available to the user by hand — a colour preset on the master, a
transition on any clip — and nothing chose them automatically. They are the
two things a viewer registers before a single picture: whether the footage
looks like *a horror video* or like a phone recording, and whether the video
moves between its subjects or merely runs on.

Both are tagged with the presentation origin so a re-run replaces only its own
choices. A grade or a transition the user set stays.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from timeline.presets import color_preset_values
from timeline.schema import ColorGrade, Timeline, TimelineItem, Transition, frame_to_time

from . import genre as genre_mod
from .models import PresentationSettings, Topic

logger = logging.getLogger("presentation.look")

GRADE_ORIGIN_PREFIX = "presentation:"
TRANSITION_ORIGIN = "presentation"

# A topic boundary this far from the nearest V1 join still counts as landing
# on it. Topics come from the model reading timestamps, and a chapter change
# a few hundred milliseconds off its cut is the same chapter change.
JOIN_TOLERANCE_S = 0.75
# The first topic starts the video; a transition INTO it has nothing before it.
# Anything closer than this to the start is that same opening, not a chapter.
MIN_BOUNDARY_S = 3.0


# --- grade -------------------------------------------------------------------

def apply_genre_grade(timeline: Timeline, settings: PresentationSettings,
                      genre: str) -> Optional[str]:
    """Set the master grade for the genre. Returns the preset used, or None.

    `settings.grade` is "auto" (from the genre), a COLOR_PRESETS name, or
    "off". A master grade the user set by hand is never touched.
    """
    current = timeline.master_color
    ours = current is not None and (current.preset or "").startswith(GRADE_ORIGIN_PREFIX)
    if current is not None and not ours and not current.is_identity():
        return None

    choice = (settings.grade or "auto").lower()
    if choice in ("off", "none", ""):
        if ours:
            timeline.master_color = None
            timeline.revision += 1
        return None

    if choice == "auto":
        preset, overrides = genre_mod.grade_for(genre)
    else:
        preset, overrides = choice, {}
    values = dict(color_preset_values(preset))
    if not values and choice != "auto":
        logger.warning("Unknown colour preset %r; no grade applied", choice)
        return None
    values.update(overrides)
    grade = ColorGrade(**values)
    grade.preset = f"{GRADE_ORIGIN_PREFIX}{preset}"
    timeline.master_color = grade
    timeline.revision += 1
    return preset


# --- transitions ---------------------------------------------------------------

def _v1_items(timeline: Timeline) -> List[TimelineItem]:
    return sorted(
        [i for i in timeline.items
         if i.track == "V1" and i.enabled and i.kind == "media"],
        key=lambda i: i.timeline_start_frame)


def clear_topic_transitions(timeline: Timeline) -> int:
    count = 0
    for item in timeline.items:
        if item.transition is not None and item.transition.origin == TRANSITION_ORIGIN:
            item.transition = None
            count += 1
    return count


def apply_topic_transitions(timeline: Timeline, topics: Sequence[Topic], genre: str,
                            settings: PresentationSettings,
                            overrides: Optional[Dict[int, Tuple[str, float]]] = None) -> int:
    """A transition into the first clip of each topic after the first.

    A topic boundary can only take a transition where a V1 join already exists
    (a transition is a property of a junction); a boundary that falls inside a
    continuous take is left as it is. `overrides` maps a topic index to an
    (xfade type, seconds) pair — the mood recipes use it for flashbacks and
    reveals.
    """
    clear_topic_transitions(timeline)
    if not settings.topic_transitions or len(topics) < 2:
        return 0

    items = _v1_items(timeline)
    if len(items) < 2:
        return 0
    joins = [(frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den), i)
             for i in items[1:]]
    kind, seconds = genre_mod.topic_transition_for(genre)
    if settings.topic_transition not in ("", "auto"):
        kind = settings.topic_transition
    overrides = overrides or {}

    count = 0
    used = set()
    for index, topic in enumerate(sorted(topics, key=lambda t: t.start_s)):
        if index == 0 or topic.start_s < MIN_BOUNDARY_S:
            continue
        nearest = min(joins, key=lambda j: abs(j[0] - topic.start_s), default=None)
        if nearest is None or abs(nearest[0] - topic.start_s) > JOIN_TOLERANCE_S:
            continue
        _, item = nearest
        if item.id in used:
            continue
        if item.transition is not None:      # the user's own choice
            continue
        chosen_kind, chosen_seconds = overrides.get(index, (kind, seconds))
        item.transition = Transition(type=chosen_kind, duration=chosen_seconds,
                                     origin=TRANSITION_ORIGIN)
        used.add(item.id)
        count += 1
    if count:
        timeline.revision += 1
    logger.info("Topic transitions: %d of %d boundaries landed on a join (%s)",
                count, max(0, len(topics) - 1), kind)
    return count
