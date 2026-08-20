"""Stage D — putting the generated material onto the timeline.

Everything here goes into the EDL rather than being rendered into an
intermediate file. That is the difference between a video the user can adjust in
the morning and one they can only accept or throw away: a cutaway that lands two
seconds late is a drag of the mouse, not another twenty-minute run.

It is also faster and sharper. The compiler already animates a still natively —
`_is_still` promotes it to a full-canvas `zoompan` and `_retime_still` stretches
its single decoded frame across the clip — so a Ken Burns move costs one filter
in the existing single-pass render, against a whole extra encode for the
pre-baked approach it replaces.
"""

import logging
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import SourceFile, Timeline, time_to_frame

from .models import Asset, Beat, PresentationSettings, Program

logger = logging.getLogger("presentation.placement")

BROLL_TRACK = "V3"
BROLL_ORIGIN = "broll"
POPUP_TRACK = "TP"
POPUP_ORIGIN = "popup"

# Enough movement to feel alive, not enough to notice as an effect.
BROLL_ZOOM_MIN = 0.06
BROLL_ZOOM_MAX = 0.14
BROLL_PAN = 0.04
# Cutaway durations and the on-camera gap between cutaways come from the
# settings (`planned_duration_s` per beat, `PresentationSettings.cutaway_gap_s`)
# — the planner budgeted with them, and a second set of constants here silently
# re-rejected beats the plan had already paid for.

POPUP_SECONDS = 3.5
# Upper third. Captions own the lower third (`lower_third` sits at pos_y 0.62),
# and a pop-up over the caption band is unreadable.
POPUP_POS_Y = -0.55
MIN_POPUP_GAP_SECONDS = 8.0


def place_broll(timeline: Timeline, beats: List[Beat], assets: List[Asset],
                program: Program, settings: PresentationSettings,
                seed: int = 0,
                rejected: Optional[List[Dict[str, str]]] = None) -> int:
    """Register generated assets as sources and cut them in over the speaker.

    `rejected`, when given, collects `{beat, topic, reason}` for every beat that
    had an asset but could not be placed — a silent drop here is exactly how a
    70% coverage plan quietly ships at 20%.
    """
    clear_generated(timeline, BROLL_ORIGIN)
    _drop_orphan_sources(timeline)
    if not assets:
        return 0

    by_id = {beat.id: beat for beat in beats}
    fps = timeline.fps_num / max(1, timeline.fps_den)
    rng = random.Random(seed)
    placed: List[Tuple[float, float]] = []
    count = 0

    for index, asset in enumerate(sorted(assets, key=lambda a: by_id.get(a.beat_id).start_s
                                         if a.beat_id in by_id else 0.0)):
        beat = by_id.get(asset.beat_id)
        if beat is None or not Path(asset.path).exists():
            continue

        start_s, duration_s, reason = _window_for(beat, asset, program, placed, settings)
        if duration_s <= 0:
            logger.info("Beat %s: not placed (%s)", beat.id, reason)
            if rejected is not None:
                rejected.append({"beat": beat.id, "topic": beat.topic,
                                 "reason": reason or "no room"})
            continue

        source_id = f"src_broll_{beat.id}_{uuid.uuid4().hex[:4]}"
        timeline.sources[source_id] = SourceFile(
            id=source_id,
            path=asset.path,
            duration_seconds=asset.duration_s or duration_s,
            width=asset.width or timeline.width,
            height=asset.height or timeline.height,
            fps_num=timeline.fps_num,
            fps_den=timeline.fps_den,
            has_audio=False,
            kind="video" if asset.kind == "video" else "image",
        )

        duration_frames = max(1, time_to_frame(duration_s, timeline.fps_num, timeline.fps_den))
        item = clip_ops.add_media_item(
            timeline, source_id, BROLL_TRACK,
            time_to_frame(start_s, timeline.fps_num, timeline.fps_den),
            0, duration_frames,
            origin=BROLL_ORIGIN,
        )

        # Stills get the move. A generated clip already moves, and adding a zoom
        # on top of its own motion looks like a mistake.
        if asset.kind == "image":
            _apply_ken_burns(timeline, item.id, index, rng, duration_s)

        placed.append((start_s, start_s + duration_s))
        count += 1
        logger.info("Beat %s: %s cutaway at %.1fs for %.1fs (%s)",
                    beat.id, asset.kind, start_s, duration_s, beat.topic)

    return count


def _window_for(beat: Beat, asset: Asset, program: Program,
                placed: List[Tuple[float, float]],
                settings: PresentationSettings) -> Tuple[float, float, str]:
    """(start, duration, reason) — duration 0 with the reason when it cannot run."""
    start_s = max(0.0, beat.start_s)
    # Land on a word boundary. Cutting away mid-syllable draws attention to the
    # cut itself, the same instinct behind snapping audio cuts to a quiet moment.
    following = [w for w in program.words if w.tl_start_s >= start_s - 0.15]
    if following:
        start_s = following[0].tl_start_s

    # The planner already decided how long this shot runs; honour it. Falling
    # back to the settings clamp covers plans stored before the field existed.
    planned = beat.planned_duration_s or max(
        settings.broll_seconds_min, min(settings.broll_seconds_max, beat.duration_s))
    available = min(beat.end_s, program.duration_s) - start_s
    duration_s = min(planned, max(0.0, available))
    if asset.kind == "video" and asset.duration_s > 0:
        duration_s = min(duration_s, asset.duration_s)
    floor = min(settings.broll_seconds_min, 3.0)
    if duration_s < floor:
        # Allow a short one only if that is genuinely all the beat has.
        if available < floor * 0.8:
            return start_s, 0.0, "no room inside the beat's span"
        duration_s = min(floor, available)

    # The planner already spaced the beats; this is a safety net, so it runs
    # 0.2s looser than the planning gap — the word-boundary snap above can move
    # a start slightly, and a hair's drift must not throw a budgeted shot away.
    gap = max(0.0, settings.cutaway_gap_s - 0.2)
    for other_start, other_end in placed:
        if start_s < other_end + gap and other_start < start_s + duration_s:
            return start_s, 0.0, "would overlap or crowd the previous cutaway"
    return start_s, duration_s, ""


def _apply_ken_burns(timeline: Timeline, item_id: str, index: int,
                     rng: random.Random, duration_s: float = 6.0) -> None:
    """A slow push-in or pull-back, alternating so consecutive shots differ.

    Depth scales mildly with duration — the 6-14% calibrated for 6s shots is
    imperceptibly slow spread over 10s, and a still that long reads frozen.
    """
    depth = BROLL_ZOOM_MIN + rng.random() * (BROLL_ZOOM_MAX - BROLL_ZOOM_MIN)
    depth *= max(1.0, min(1.5, duration_s / 6.0))
    pan = BROLL_PAN * (1.0 if index % 4 < 2 else -1.0)
    if index % 2 == 0:
        scale, scale_end = 1.0, 1.0 + depth
    else:
        scale, scale_end = 1.0 + depth, 1.0
    clip_ops.set_transform(timeline, item_id, {
        "scale": scale,
        "scale_end": scale_end,
        "pos_x": -pan,
        "pos_x_end": pan,
    })


def _drop_orphan_sources(timeline: Timeline) -> None:
    """Forget sources no item references any more.

    `clear_generated` removes the clips but not the SourceFile entries behind
    them, and every re-run would otherwise leave another set of dead sources in
    the project file.
    """
    used = {item.source_id for item in timeline.items if item.source_id}
    for source_id in [s for s in timeline.sources
                      if s.startswith("src_broll_") and s not in used]:
        timeline.sources.pop(source_id, None)


def place_popups(timeline: Timeline, beats: List[Beat], program: Program,
                 settings: PresentationSettings,
                 broll_windows: Optional[List[Tuple[float, float]]] = None) -> int:
    """Topic text on its own track, out of the caption band."""
    clear_generated(timeline, POPUP_ORIGIN)
    popups = [b for b in beats if b.kind == "popup" and b.popup_text]
    if not popups:
        return 0

    busy = list(broll_windows or _windows_of(timeline, BROLL_ORIGIN))
    placed: List[Tuple[float, float]] = []
    count = 0

    for beat in sorted(popups, key=lambda b: b.start_s):
        start_s = max(0.0, beat.start_s)
        duration_s = min(POPUP_SECONDS, max(1.2, beat.duration_s),
                         max(0.0, program.duration_s - start_s))
        if duration_s < 1.2:
            continue
        end_s = start_s + duration_s

        # A full-frame cutaway plus floating text is clutter, and the pop-up is
        # the one of the two that can simply wait.
        if any(start_s < b_end and b_start < end_s for b_start, b_end in busy):
            logger.info("Pop-up %r skipped: a cutaway is on screen", beat.popup_text)
            continue
        if any(start_s < p_end + MIN_POPUP_GAP_SECONDS and p_start < end_s
               for p_start, p_end in placed):
            continue

        style = {"pos_y": POPUP_POS_Y, "animation": "pop", "animation_duration": 0.25}
        item = clip_ops.add_text_item(
            timeline, beat.popup_text,
            time_to_frame(start_s, timeline.fps_num, timeline.fps_den),
            max(1, time_to_frame(duration_s, timeline.fps_num, timeline.fps_den)),
            track=POPUP_TRACK,
            preset=settings.popup_preset,
            style=style,
        )
        item.origin = POPUP_ORIGIN
        placed.append((start_s, end_s))
        count += 1
        logger.info("Pop-up %r at %.1fs", beat.popup_text, start_s)

    return count


def _windows_of(timeline: Timeline, origin: str) -> List[Tuple[float, float]]:
    from timeline.schema import frame_to_time
    return [(frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den),
             frame_to_time(i.timeline_end_frame, timeline.fps_num, timeline.fps_den))
            for i in timeline.items if i.origin == origin]


def broll_windows(timeline: Timeline) -> List[Tuple[float, float]]:
    """When a cutaway is on screen — the face zoom needs this too."""
    return _windows_of(timeline, BROLL_ORIGIN)
