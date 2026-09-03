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
from timeline.authoring import clear_generated
from timeline.schema import Timeline, time_to_frame

from .models import Beat, PresentationSettings, Program

logger = logging.getLogger("presentation.cards")

CARD_TRACK = "TX"
CARD_ORIGIN = "card"

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


def place_cards(timeline: Timeline, beats: Sequence[Beat], program: Program,
                settings: PresentationSettings, genre: str = "general",
                popup_windows: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, int]:
    """Put every text beat on the timeline that can be placed without a clash.

    Returns counts per kind. Beats of other kinds are ignored.
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

    for beat in sorted(cards, key=lambda b: (b.start_s, -b.priority)):
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
        _create(timeline, beat, start, duration, genre, fps_num, fps_den)
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
            genre: str, fps_num: int, fps_den: int) -> None:
    start_frame = time_to_frame(start_s, fps_num, fps_den)
    frames = max(1, time_to_frame(duration_s, fps_num, fps_den))
    style = _style_for(beat.kind, genre)

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
            label = clip_ops.add_text_item(timeline, beat.subtext, start_frame, frames,
                                           track=CARD_TRACK, preset="stat_label",
                                           style=_style_for("stat_label", genre))
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
