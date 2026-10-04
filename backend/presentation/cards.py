"""Text cards on the picture: chapter titles, stat call-outs, location and
character cards, sources, definitions, quotes, the end screen.

Each kind has a **zone** — centre, top-left, lower-left, lower-right — and
cards only collide within a zone: a location card in the corner and a stat in
the middle can share a moment, two stats cannot. A colliding card slides
forward a little inside its topic; past that it is dropped rather than drawn
on top of another, because two cards at once is noise.

All cards live on one named text track (`TX`) with one origin so a re-run
replaces exactly the previous batch and hand-placed text stays. Presets come
from `timeline.presets.TEXT_PRESETS`; the genre can restyle a kind (a horror
chapter title is a slow serif, not a slam).
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from timeline import clip_ops
from timeline.presets import TEXT_ZONE_MAX_POS_Y
from timeline.authoring import clear_generated
from timeline.schema import Timeline, time_to_frame

from . import genre as genre_mod
from .models import Beat, PresentationSettings, Program

logger = logging.getLogger("presentation.cards")

CARD_TRACK = "TX"
CARD_ORIGIN = "card"

# Kinds placed dead centre — exactly where a talking head's face usually sits
# — so they are the ones the "text_clear_of_face" render check catches. Below
# this, `_dodge_face` nudges them into the upper or lower band instead,
# whichever the speaker's face is farther from at that moment.
FACE_AVOID_KINDS = {"chapter_title", "stat_callout", "definition_card", "quote_card", "end_screen"}
# Half the face box verify.py checks against (its own FACE_HALF_H), doubled
# for room to spare rather than a graze.
_FACE_BAND = 0.32
_STAT_LABEL_OFFSET = 0.27

ZONES: Dict[str, str] = {
    "chapter_title": "centre",
    "stat_callout": "centre",
    "definition_card": "centre",
    "quote_card": "centre",
    "end_screen": "centre",
    "location_card": "top_left",
    "character_card": "lower_left",
    "source_card": "lower_right",
}
# Cards in one zone stay this far apart; a centre card also keeps clear of the
# opening title so the video's first seconds carry one message, not two.
ZONE_GAP_S = 1.0
TITLE_ZONE_S = 3.0
SLIDE_STEP_S = 0.25
SLIDE_MAX_S = 4.0
MIN_CARD_S = 1.2

# Per-genre restyling of a kind's preset. Sparse on purpose.
# Which card kinds a genre reaches for on its own: a motivational video wants
# its quote or its stat to win a shared centre slot over a plain chapter
# title, true crime wants its case file and its sources to. Blended with
# `genre_secondary` the same ~65/35 way as genre.py's blended_fx_palette (see
# `_affinity_boost`), so a vlog+motivational video leans toward motivational's
# favourites without vlog (which favours nothing in particular) losing out.
_KIND_AFFINITY: Dict[str, Tuple[str, ...]] = {
    "motivational": ("quote_card", "stat_callout"),
    "finance": ("stat_callout", "source_card"),
    "true_crime": ("case_file", "source_card", "quote_card"),
    "horror": ("quote_card",),
    "mystery": ("quote_card", "source_card"),
    "news": ("source_card", "stat_callout"),
    "science_education": ("definition_card", "stat_callout", "source_card"),
    "geopolitics": ("source_card", "stat_callout"),
    "documentary": ("quote_card", "source_card"),
    "devotional": ("quote_card",),
    "health_fitness": ("stat_callout",),
    "tech": ("stat_callout",),
}
# A primary-genre favourite outranks an equal-priority rival outright; a
# secondary-genre favourite gets the same ~35/65 fraction of that boost that
# genre.py's own blend gives a secondary genre's names.
_AFFINITY_PRIMARY_BOOST = 0.20
_AFFINITY_SECONDARY_BOOST = _AFFINITY_PRIMARY_BOOST * (
    genre_mod._BLEND_SECONDARY_WEIGHT / genre_mod._BLEND_PRIMARY_WEIGHT)

GENRE_STYLES: Dict[str, Dict[str, Dict[str, object]]] = {
    "horror": {
        "chapter_title": {"font_family": "Georgia", "bold": False, "color": "#D8D0C4",
                          "stroke_width": 0, "shadow_x": 0, "shadow_y": 0,
                          "animation": "blur_in", "animation_duration": 0.8},
        "stat_callout": {"color": "#C8102E", "stroke_color": "#1A0000"},
        "end_screen": {"font_family": "Georgia", "color": "#D8D0C4", "stroke_width": 0},
    },
    "true_crime": {
        "chapter_title": {"font_family": "Georgia", "bold": True, "color": "#E8E2D6",
                          "animation": "fade", "animation_duration": 0.5},
    },
    "devotional": {
        "chapter_title": {"color": "#FFE9B0", "stroke_color": "#4A2A00"},
        "stat_callout": {"color": "#FFE9B0"},
    },
    "finance": {
        "stat_callout": {"color": "#8CFF9E", "stroke_color": "#003311"},
    },
    "comedy": {
        "chapter_title": {"font_family": "Impact", "color": "#FFE23A"},
    },
}


def _content(beat: Beat) -> str:
    kind = beat.kind
    text = (beat.text or "").strip()
    sub = (beat.subtext or "").strip()
    if kind == "quote_card":
        return f"{text}\n— {sub}" if sub else text
    if kind in ("location_card", "character_card", "definition_card"):
        return f"{text}\n{sub}" if sub else text
    return text


def _style_for(kind: str, genre: str) -> Dict[str, object]:
    return dict(GENRE_STYLES.get(genre, {}).get(kind, {}))


def _affinity_boost(kind: str, genre: str, genre_secondary: Optional[str]) -> float:
    """How much `kind` outranks an equal-priority rival for a shared zone
    slot, given the genre blend — 0.0 when neither genre reaches for it."""
    if kind in _KIND_AFFINITY.get(genre, ()):
        return _AFFINITY_PRIMARY_BOOST
    normalised_secondary = genre_mod.normalise(genre_secondary) if genre_secondary else None
    if normalised_secondary and normalised_secondary not in ("general", genre_mod.normalise(genre)):
        if kind in _KIND_AFFINITY.get(normalised_secondary, ()):
            return _AFFINITY_SECONDARY_BOOST
    return 0.0


def _dodge_face(face_anchors: Optional[Dict[str, Tuple[float, float]]], program: Program,
                at_s: float) -> Optional[float]:
    """`pos_y` for a centre card at this moment, or None to leave the preset
    alone — no anchor for the segment on screen there means nothing to dodge."""
    if not face_anchors:
        return None
    segment = program.segment_at(at_s)
    anchor = face_anchors.get(segment.item_id) if segment else None
    if anchor is None:
        return None
    # Dodging down stops short of the caption band, with room for a stat's
    # label under its value (`_STAT_LABEL_OFFSET`).
    return -_FACE_BAND if anchor[1] >= 0.0 else min(_FACE_BAND, TEXT_ZONE_MAX_POS_Y - _STAT_LABEL_OFFSET + 0.1)


def place_cards(timeline: Timeline, beats: Sequence[Beat], program: Program,
                settings: PresentationSettings, genre: str = "general",
                genre_secondary: Optional[str] = None,
                popup_windows: Optional[Sequence[Tuple[float, float]]] = None,
                face_anchors: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, int]:
    """Put every text beat on the timeline that can be placed without a clash.

    Returns counts per kind. Beats of other kinds are ignored.

    `face_anchors` (segment item id -> (x, y), from facezoom.detect_faces) is
    optional: with it, a centre-zone card (chapter title, stat, definition,
    quote, end screen) sharing a moment with a detected face nudges up or down
    out of its way; without it every card keeps its preset's position exactly
    as before.

    `genre_secondary`, when given, biases which kind wins a shared zone slot
    at a tie — see `_KIND_AFFINITY` / `_affinity_boost` — without changing
    anything else about placement.
    """
    clear_generated(timeline, CARD_ORIGIN)
    cards = [b for b in beats if b.is_text and (b.text or "").strip()]
    if not cards:
        return {}

    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    placed: Dict[str, List[Tuple[float, float]]] = {z: [] for z in set(ZONES.values())}
    # Pop-ups live in the upper third and never touch a centre card; they are
    # accepted here only so a caller can pass them without a special case.
    del popup_windows
    if settings.title:
        placed["centre"].append((0.0, TITLE_ZONE_S))
    counts: Dict[str, int] = {}

    def _sort_key(b: Beat) -> Tuple[float, float]:
        return b.start_s, -(b.priority + _affinity_boost(b.kind, genre, genre_secondary))

    for beat in sorted(cards, key=_sort_key):
        zone = ZONES.get(beat.kind, "centre")
        duration = max(MIN_CARD_S, beat.duration_s)
        limit = program.duration_s - duration
        if limit <= 0:
            continue
        start = _slot(beat.start_s, duration, placed[zone], min(limit, beat.end_s + SLIDE_MAX_S - duration))
        if start is None:
            logger.info("Card %s %r skipped: zone %s busy", beat.kind, beat.text, zone)
            continue
        end = start + duration
        pos_y = (_dodge_face(face_anchors, program, start) if beat.kind in FACE_AVOID_KINDS else None)
        _create(timeline, beat, start, duration, genre, fps_num, fps_den, pos_y_override=pos_y)
        placed[zone].append((start, end))
        counts[beat.kind] = counts.get(beat.kind, 0) + 1
        logger.info("Card %s %r at %.1fs", beat.kind, (beat.text or "")[:30], start)

    timeline.recalculate_duration()
    timeline.revision += 1
    return counts


def _slot(wanted: float, duration: float, busy: List[Tuple[float, float]],
          latest: float) -> Optional[float]:
    candidate = max(0.0, wanted)
    while candidate <= latest + 1e-6:
        end = candidate + duration
        if not any(candidate < b_end + ZONE_GAP_S and b_start < end + ZONE_GAP_S
                   for b_start, b_end in busy):
            return candidate
        candidate += SLIDE_STEP_S
    return None


def _create(timeline: Timeline, beat: Beat, start_s: float, duration_s: float,
            genre: str, fps_num: int, fps_den: int,
            pos_y_override: Optional[float] = None) -> None:
    start_frame = time_to_frame(start_s, fps_num, fps_den)
    frames = max(1, time_to_frame(duration_s, fps_num, fps_den))
    style = _style_for(beat.kind, genre)
    if pos_y_override is not None:
        style["pos_y"] = pos_y_override

    if beat.kind == "stat_callout" and beat.data.get("value") is not None:
        item = clip_ops.add_text_item(timeline, beat.text or "", start_frame, frames,
                                      track=CARD_TRACK, preset="stat_callout", style=style)
        item.origin = CARD_ORIGIN
        item.label = f"stat: {beat.text}"
        item.text.counter = {
            "from": 0, "to": float(beat.data["value"]),
            "prefix": beat.data.get("prefix") or "", "suffix": beat.data.get("suffix") or "",
            "decimals": int(beat.data.get("decimals") or 0),
            "seconds": min(1.4, max(0.6, duration_s * 0.4)),
        }
        if beat.subtext:
            label_style = _style_for("stat_label", genre)
            if pos_y_override is not None:
                # Keep the label's usual offset below the value rather than
                # collapsing both onto the same line (presets:
                # stat_callout pos_y=-0.05, stat_label pos_y=0.22).
                label_style["pos_y"] = min(pos_y_override + _STAT_LABEL_OFFSET, TEXT_ZONE_MAX_POS_Y + 0.1)
            label = clip_ops.add_text_item(timeline, beat.subtext, start_frame, frames,
                                           track=CARD_TRACK, preset="stat_label",
                                           style=label_style)
            label.origin = CARD_ORIGIN
            label.label = f"stat label: {beat.subtext[:30]}"
        return

    preset = beat.kind if beat.kind != "quote_card" else "quote_card"
    item = clip_ops.add_text_item(timeline, _content(beat), start_frame, frames,
                                  track=CARD_TRACK, preset=preset, style=style)
    item.origin = CARD_ORIGIN
    item.label = f"{beat.kind}: {(beat.text or '')[:30]}"

    if beat.kind == "end_screen":
        # Pull the picture back so the platform's end-screen elements read.
        shade = clip_ops.add_adjustment_item(timeline, start_frame, frames, track="V9")
        shade.origin = CARD_ORIGIN
        shade.label = "end screen shade"
        clip_ops.set_color(timeline, shade.id, {"brightness": -0.22, "saturation": 0.7,
                                                "vignette": 0.5, "fade_in": 0.6})


def card_windows(timeline: Timeline) -> List[Tuple[float, float]]:
    from timeline.schema import frame_to_time
    return [(frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den),
             frame_to_time(i.timeline_end_frame, timeline.fps_num, timeline.fps_den))
            for i in timeline.items if i.origin == CARD_ORIGIN and i.kind == "text"]
