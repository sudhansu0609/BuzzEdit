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

from .models import BROLL_SECONDS_FLOOR, Asset, Beat, PresentationSettings, Program

logger = logging.getLogger("presentation.placement")

BROLL_TRACK = "V3"
BROLL_ORIGIN = "broll"
# Graphics (icons, diagrams with alpha) ride above the B-roll, never full-frame.
GRAPHIC_TRACK = "V4"
# The right half of a split screen sits one track above the left.
SPLIT_RIGHT_TRACK = "V4"
GRAPHIC_ORIGIN = "graphic"
GRAPHIC_SCALE = 0.42
POPUP_TRACK = "TP"
POPUP_ORIGIN = "popup"

# Enough movement to feel alive, not enough to notice as an effect.
# A shot timed by the script may start this far inside, or after, the one
# before it (rounding) and is moved to start exactly where that one ends.
HELD_OVERLAP_SLACK_S = 0.25
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

# A jump cut is invisible while a full-frame cutaway is over it — the switch to
# B-roll and back then happens inside one continuous take, and the join
# underneath is never seen. Each window therefore shifts (within its beat) to
# straddle the nearest V1 join, with this much cover either side so neither the
# cutaway's own edge nor the join lands on the same frame.
CUT_COVER_MARGIN_S = 0.35
# How far outside its topic span a window may drift to reach a join. The span is
# the model's estimate of when a topic runs, so half a second of spill keeps the
# picture on-topic while letting it catch a join sitting right at the boundary.
CUT_COVER_SLACK_S = 0.5


def jump_cut_times(timeline: Timeline) -> List[float]:
    """Timeline seconds where one V1 clip ends and the next begins.

    Every one of these joins is a visible jump — the auto-edit removed speech
    between them, so the speaker's head position snaps. They are what the
    cut-covering shift below hides.
    """
    from timeline.schema import frame_to_time
    v1 = sorted((i for i in timeline.items
                 if i.track == "V1" and i.enabled and i.kind == "media"),
                key=lambda i: i.timeline_start_frame)
    return [frame_to_time(item.timeline_end_frame, timeline.fps_num, timeline.fps_den)
            for item in v1[:-1]]


def _cuts_covered(start_s: float, end_s: float, cuts: List[float],
                  margin: float = CUT_COVER_MARGIN_S) -> int:
    # The candidate start is rounded to milliseconds, so a join covered exactly
    # at the margin can land a fraction of a millisecond outside it; the epsilon
    # keeps that from reading as uncovered.
    return sum(1 for cut in cuts
               if start_s + margin - 0.005 <= cut <= end_s - margin + 0.005)


def jump_cut_coverage(timeline: Timeline) -> Dict[str, int]:
    """How many of the edit's jump cuts a cutaway hides — for the report."""
    cuts = jump_cut_times(timeline)
    windows = broll_windows(timeline)
    covered = sum(1 for cut in cuts
                  if any(start + 0.1 <= cut <= end - 0.1 for start, end in windows))
    return {"total": len(cuts), "covered": covered}


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
    clear_generated(timeline, GRAPHIC_ORIGIN)
    clear_generated(timeline, "split")
    _drop_orphan_sources(timeline)
    if not assets:
        return 0

    by_id = {beat.id: beat for beat in beats}
    fps = timeline.fps_num / max(1, timeline.fps_den)
    rng = random.Random(seed)
    placed: List[Tuple[float, float]] = []
    cuts = jump_cut_times(timeline)
    count = 0

    assets_by_beat = {a.beat_id: a for a in assets}
    split_pairs: List[Tuple[float, float, Optional[str], Optional[str]]] = []
    for index, asset in enumerate(sorted(assets, key=lambda a: by_id.get(a.beat_id).start_s
                                         if a.beat_id in by_id else 0.0)):
        beat = by_id.get(asset.beat_id)
        if beat is None or not Path(asset.path).exists():
            continue

        # Split screens: the left half places the pair; the right half rides
        # along. A half whose partner never generated goes up full-frame.
        side = beat.data.get("split") if beat.data else None
        partner = None
        if side == "right" and by_id.get(f"{beat.data.get('pair')}_l") is not None \
                and f"{beat.data.get('pair')}_l" in assets_by_beat:
            continue
        if side == "left":
            partner_id = f"{beat.data.get('pair')}_r"
            partner = assets_by_beat.get(partner_id)
            if partner is not None and not Path(partner.path).exists():
                partner = None

        start_s, duration_s, reason = _window_for(beat, asset, program, placed, settings,
                                                  cuts)
        if partner is not None and duration_s > 0:
            from .composite import split_transform
            frames = max(1, time_to_frame(duration_s, timeline.fps_num, timeline.fps_den))
            start_frame = time_to_frame(start_s, timeline.fps_num, timeline.fps_den)
            for half, half_asset, track in (("left", asset, BROLL_TRACK),
                                            ("right", partner, SPLIT_RIGHT_TRACK)):
                source_id = f"src_broll_{half_asset.beat_id}_{uuid.uuid4().hex[:4]}"
                timeline.sources[source_id] = SourceFile(
                    id=source_id, path=half_asset.path,
                    duration_seconds=half_asset.duration_s or duration_s,
                    width=half_asset.width or timeline.width,
                    height=half_asset.height or timeline.height,
                    fps_num=timeline.fps_num, fps_den=timeline.fps_den,
                    has_audio=False, kind="video" if half_asset.kind == "video" else "image")
                item = clip_ops.add_media_item(timeline, source_id, track, start_frame, 0,
                                               frames, origin=BROLL_ORIGIN)
                clip_ops.set_transform(timeline, item.id, split_transform(half))
                item.label = f"split {half}: {beat.topic[:24]}"
            labels = beat.data.get("label"), (by_id[partner.beat_id].data.get("label")
                                              if partner.beat_id in by_id else None)
            split_pairs.append((start_s, start_s + duration_s, labels[0], labels[1]))
            placed.append((start_s, start_s + duration_s))
            count += 1
            logger.info("Beat %s: split screen at %.1fs for %.1fs", beat.id, start_s, duration_s)
            continue
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
        if beat.kind == "graphic":
            # A graphic is an element ON the picture, not a picture: it sits in
            # a corner at under half size and slides in, keeping its alpha. It
            # is never zoomed — an animated zoom goes through zoompan, which
            # pads a transparent PNG with black.
            item = clip_ops.add_media_item(
                timeline, source_id, GRAPHIC_TRACK,
                time_to_frame(start_s, timeline.fps_num, timeline.fps_den),
                0, duration_frames, origin=GRAPHIC_ORIGIN)
            side = 1.0 if index % 2 == 0 else -1.0
            clip_ops.set_transform(timeline, item.id, {
                "scale": GRAPHIC_SCALE, "pos_x": side * 0.62, "pos_x_end": side * 0.52,
                "pos_y": -0.42, "pos_y_end": -0.42})
            count += 1
            logger.info("Beat %s: graphic at %.1fs for %.1fs", beat.id, start_s, duration_s)
            continue

        item = clip_ops.add_media_item(
            timeline, source_id, BROLL_TRACK,
            time_to_frame(start_s, timeline.fps_num, timeline.fps_den),
            0, duration_frames,
            origin=BROLL_ORIGIN,
        )

        # Stills get the move. A generated clip already moves, and adding a zoom
        # on top of its own motion looks like a mistake. A designed layout
        # stays still: its speaker slot is drawn at a fixed place.
        if beat.kind in ("statement_card", "canvas_card"):
            item.label = f"layout: {beat.kind}"
        elif asset.kind == "image":
            _apply_ken_burns(timeline, item.id, index, rng, duration_s)

        placed.append((start_s, start_s + duration_s))
        count += 1
        logger.info("Beat %s: %s cutaway at %.1fs for %.1fs (%s)",
                    beat.id, asset.kind, start_s, duration_s, beat.topic)

    if split_pairs:
        from .composite import place_split_labels
        place_split_labels(timeline, split_pairs, "")
    return count


def _window_for(beat: Beat, asset: Asset, program: Program,
                placed: List[Tuple[float, float]],
                settings: PresentationSettings,
                cuts: Optional[List[float]] = None) -> Tuple[float, float, str]:
    """(start, duration, reason) — duration 0 with the reason when it cannot run."""
    start_s = max(0.0, beat.start_s)
    if beat.data.get("hold") and beat.planned_duration_s:
        return _held_window(beat, asset, program, placed, start_s)
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

    # Slide the window onto the nearest jump cut. The word-boundary snap above
    # is a preference; hiding a join is the point of the shot being here at all,
    # so where a join is reachable the window moves to straddle it — the picture
    # then switches to B-roll inside one continuous take and back inside the
    # next, and the join underneath is never seen. (The audio never cuts, so a
    # start mid-word costs nothing.)
    start_s = _shift_to_cover_cuts(start_s, duration_s, beat, cuts or [],
                                   placed, program, settings)

    # The planner already spaced the beats; this is a safety net, so it runs
    # 0.2s looser than the planning gap — the word-boundary snap above can move
    # a start slightly, and a hair's drift must not throw a budgeted shot away.
    # Density-scaled (see PresentationSettings.crowd_gap_s), not the larger
    # coverage-derived cutaway_gap_s — busy/max allows back-to-back cutaways,
    # and this safety net must not re-reject what budgeting already allowed.
    gap = max(0.0, settings.crowd_gap_s - 0.2)
    for other_start, other_end in placed:
        if start_s < other_end + gap and other_start < start_s + duration_s:
            return start_s, 0.0, "would overlap or crowd the previous cutaway"
    return start_s, duration_s, ""


def _held_window(beat: Beat, asset: Asset, program: Program,
                 placed: List[Tuple[float, float]], start_s: float) -> Tuple[float, float, str]:
    """A shot the script timed itself (`seconds=`): it starts on its word and
    runs exactly its length. The caller spaced these shots, often back to
    back, so no snapping or sliding -- either could open a one-frame gap of
    the speaker between two shots or push one onto the next. Only a sliver of
    overlap (float rounding) is trimmed rather than the shot thrown away."""
    duration_s = min(beat.planned_duration_s, max(0.0, program.duration_s - start_s))
    if asset.kind == "video" and asset.duration_s > 0:
        duration_s = min(duration_s, asset.duration_s)
    for other_start, other_end in placed:
        if other_start <= start_s < other_end:
            if other_end - start_s > HELD_OVERLAP_SLACK_S:
                return start_s, 0.0, "would overlap the previous cutaway"
            duration_s -= other_end - start_s
            start_s = other_end
        elif 0.0 < start_s - other_end <= HELD_OVERLAP_SLACK_S:
            # Meant to follow straight on: close the sliver, or the speaker
            # flashes up for a frame between two shots.
            duration_s += start_s - other_end
            start_s = other_end
        elif start_s < other_start < start_s + duration_s:
            duration_s = other_start - start_s
    if duration_s < BROLL_SECONDS_FLOOR:
        return start_s, 0.0, "no room inside the beat's span"
    return start_s, duration_s, ""


def _shift_to_cover_cuts(start_s: float, duration_s: float, beat: Beat,
                         cuts: List[float], placed: List[Tuple[float, float]],
                         program: Program,
                         settings: PresentationSettings) -> float:
    """The window start that hides the most V1 joins, nearest the planned start.

    Candidates are the planned start plus, for every reachable join, the start
    nearest the planned one that still covers that join with the margin. Every
    candidate must stay inside the beat's span (± a little slack), clear of the
    cutaways already placed, and the planned start always remains on the table —
    a shot that can reach no join simply runs where the planner put it.
    """
    if not cuts or duration_s <= 2 * CUT_COVER_MARGIN_S:
        return start_s
    lo = max(0.0, beat.start_s - CUT_COVER_SLACK_S)
    hi = min(program.duration_s, beat.end_s + CUT_COVER_SLACK_S) - duration_s
    if hi <= lo:
        return start_s

    gap = max(0.0, settings.crowd_gap_s - 0.2)

    def clear(candidate: float) -> bool:
        end = candidate + duration_s
        return all(not (candidate < other_end + gap and other_start < end)
                   for other_start, other_end in placed)

    candidates = {round(start_s, 3)}
    for cut in cuts:
        if cut < lo or cut > hi + duration_s:
            continue
        # The start nearest the planned one that covers this join with margin.
        candidate = min(max(start_s, cut + CUT_COVER_MARGIN_S - duration_s),
                        cut - CUT_COVER_MARGIN_S)
        candidates.add(round(min(max(candidate, lo), hi), 3))

    def score(candidate: float) -> Tuple[int, float]:
        return (_cuts_covered(candidate, candidate + duration_s, cuts),
                -abs(candidate - start_s))

    viable = [c for c in candidates if clear(c)] or [round(start_s, 3)]
    best = max(viable, key=score)
    return best if score(best) > score(round(start_s, 3)) else start_s


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

        # The pop-up prefers a moment with the speaker on screen, so it slides
        # forward within its topic's span to the first stretch clear of
        # cutaways. But at 60-75% coverage the on-camera gaps are shorter than
        # a pop-up, and "skip on collision" shipped videos with no pop-ups at
        # all — so when no clear moment exists it draws over the B-roll
        # instead, which is where YouTube text accents live anyway; only a
        # collision with another pop-up still skips it.
        slid = _first_clear_moment(start_s, duration_s, beat, busy, placed,
                                   program)
        if slid is not None:
            start_s = slid
        else:
            if any(start_s < p_end + MIN_POPUP_GAP_SECONDS
                   and p_start < start_s + duration_s
                   for p_start, p_end in placed):
                logger.info("Pop-up %r skipped: too close to another pop-up",
                            beat.popup_text)
                continue
            logger.info("Pop-up %r drawn over the B-roll: no clear moment in "
                        "its span", beat.popup_text)
        end_s = start_s + duration_s

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


def _first_clear_moment(start_s: float, duration_s: float, beat: Beat,
                        busy: List[Tuple[float, float]],
                        placed: List[Tuple[float, float]],
                        program: Program,
                        step: float = 0.5) -> Optional[float]:
    """The earliest start in the beat's span clear of cutaways and other pop-ups.

    None when the whole span is occupied — a pop-up drifting outside its topic
    would label the wrong content, so past the span it is dropped, not moved.
    """
    limit = min(beat.end_s, program.duration_s) - duration_s
    candidate = start_s
    while candidate <= limit + 1e-6:
        end = candidate + duration_s
        clear_of_broll = not any(candidate < b_end and b_start < end
                                 for b_start, b_end in busy)
        clear_of_popups = not any(candidate < p_end + MIN_POPUP_GAP_SECONDS
                                  and p_start < end
                                  for p_start, p_end in placed)
        if clear_of_broll and clear_of_popups:
            return candidate
        candidate += step
    return None


def _windows_of(timeline: Timeline, origin: str) -> List[Tuple[float, float]]:
    from timeline.schema import frame_to_time
    return [(frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den),
             frame_to_time(i.timeline_end_frame, timeline.fps_num, timeline.fps_den))
            for i in timeline.items if i.origin == origin]


def broll_windows(timeline: Timeline) -> List[Tuple[float, float]]:
    """When a cutaway is on screen — the face zoom needs this too."""
    return _windows_of(timeline, BROLL_ORIGIN)
