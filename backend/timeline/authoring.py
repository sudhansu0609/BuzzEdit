"""Generated text: burned-in captions from the transcript, and intro sequences.

Both features produce ordinary text items on the timeline rather than a separate
render pass, so they compose with everything else (grades, overlays, manual text)
in the same single FFmpeg run and stay editable clip-by-clip afterwards.

Generated items are tagged via `origin` ("caption" / "intro") so regenerating
replaces exactly the previous batch and leaves hand-made text alone.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import clip_ops
from .ops import rebuild_primary_tracks
from .presets import caption_preset, intro_preset, text_preset_style
from .schema import Timeline, TimelineItem, time_to_frame

# Named (rather than numbered) tracks for generated content: `next_track` only
# counts digits, so these never collide with the T1/T2 lanes a user creates.
CAPTION_TRACK = "TC"
INTRO_TRACK = "TI"

CAPTION_ORIGIN = "caption"
INTRO_ORIGIN = "intro"


def _fps(timeline: Timeline) -> float:
    return timeline.fps_num / max(1, timeline.fps_den)


def _program_segments(timeline: Timeline) -> List[Tuple[int, int, int]]:
    """(source_start, source_end, timeline_start) for each V1 cut, in order."""
    return sorted(
        [(i.source_start_frame, i.source_end_frame, i.timeline_start_frame)
         for i in timeline.items if i.track == "V1" and i.enabled and i.kind == "media"],
        key=lambda s: s[2],
    )


def source_to_timeline_frame(segments: List[Tuple[int, int, int]], source_frame: int) -> Optional[int]:
    """Where a source frame ends up on the timeline after cutting, or None if cut.

    Words carry source timestamps, but every removed filler ripples the program
    earlier — so a caption placed at the word's raw timestamp would drift further
    out of sync with every cut. Mapping through the surviving V1 cuts is what
    keeps captions locked to speech.
    """
    for src_start, src_end, tl_start in segments:
        if src_start <= source_frame < src_end:
            return tl_start + (source_frame - src_start)
    return None


def clear_generated(timeline: Timeline, origin: str) -> int:
    before = len(timeline.items)
    timeline.items = [i for i in timeline.items if i.origin != origin]
    return before - len(timeline.items)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def generate_captions(
    timeline: Timeline,
    preset: str = "classic",
    track: str = CAPTION_TRACK,
    style_overrides: Optional[Dict[str, Any]] = None,
    min_duration_sec: float = 0.4,
) -> List[TimelineItem]:
    """Turn the enabled transcript words into styled caption clips.

    Replaces any previous generated captions. Cards break on the preset's word
    count, on a pause longer than `max_gap_seconds`, or wherever a cut makes the
    words non-contiguous on the timeline.
    """
    config = caption_preset(preset)
    words_per_card = int(config.get("words_per_caption", 8))
    max_gap_frames = int(float(config.get("max_gap_seconds", 0.8)) * _fps(timeline))
    uppercase = bool(config.get("uppercase", False))

    clear_generated(timeline, CAPTION_ORIGIN)

    segments = _program_segments(timeline)
    if not segments:
        timeline.recalculate_duration()
        return []

    # Project each enabled word onto the cut program, dropping anything the edit
    # removed.
    placed: List[Tuple[str, int, int]] = []
    for word in timeline.words:
        if not word.enabled or not word.text.strip():
            continue
        start = source_to_timeline_frame(segments, word.start_frame)
        if start is None:
            continue
        end = source_to_timeline_frame(segments, max(word.start_frame, word.end_frame - 1))
        end = (end + 1) if end is not None else start + 1
        if end <= start:
            end = start + 1
        placed.append((word.text.strip(), start, end))

    if not placed:
        timeline.recalculate_duration()
        return []

    cards: List[List[Tuple[str, int, int]]] = []
    current: List[Tuple[str, int, int]] = [placed[0]]
    for entry in placed[1:]:
        previous = current[-1]
        gap = entry[1] - previous[2]
        if len(current) >= words_per_card or gap > max_gap_frames or gap < 0:
            cards.append(current)
            current = [entry]
        else:
            current.append(entry)
    cards.append(current)

    min_frames = max(1, int(min_duration_sec * _fps(timeline)))
    created: List[TimelineItem] = []
    for index, card in enumerate(cards):
        content = " ".join(w[0] for w in card)
        if uppercase:
            content = content.upper()
        start = card[0][1]
        end = max(card[-1][2], start + min_frames)
        # Never let a card outlive the next one; overlapping drawtext reads as a
        # double-exposure on screen.
        if index + 1 < len(cards):
            end = min(end, cards[index + 1][0][1])
        if end <= start:
            continue

        item = TimelineItem(
            id=f"cap_{uuid.uuid4().hex[:8]}",
            track=track,
            source_id=None,
            kind="text",
            timeline_start_frame=start,
            timeline_end_frame=end,
            origin=CAPTION_ORIGIN,
            text=clip_ops.build_text_clip(content, preset=None,
                                          style={**config.get("style", {}),
                                                 **(style_overrides or {})}),
            label=content[:40],
        )
        item.text.preset = preset
        timeline.items.append(item)
        created.append(item)

    timeline.recalculate_duration()
    timeline.revision += 1
    return created


def remove_captions(timeline: Timeline) -> int:
    removed = clear_generated(timeline, CAPTION_ORIGIN)
    timeline.recalculate_duration()
    timeline.revision += 1
    return removed


# ---------------------------------------------------------------------------
# Intros
# ---------------------------------------------------------------------------

def _shift_all_items(timeline: Timeline, delta: int) -> None:
    """Slide every clip along the timeline so an intro can be inserted in front."""
    if delta == 0:
        return
    for item in timeline.items:
        item.timeline_start_frame = max(0, item.timeline_start_frame + delta)
        item.timeline_end_frame = max(1, item.timeline_end_frame + delta)


def _primary_source_id(timeline: Timeline) -> Optional[str]:
    for item in timeline.items:
        if item.track == "V1" and item.source_id:
            return item.source_id
    return next(iter(timeline.sources), None)


def apply_intro(
    timeline: Timeline,
    preset: str,
    title: str = "",
    subtitle: str = "",
    track: str = INTRO_TRACK,
) -> Dict[str, Any]:
    """Put a preset intro in front of the program.

    Presets with a `background` push the whole program back by the intro's length
    and fill the gap with a generated colour card (see Timeline.program_offset_frames).
    A preset with `background: None` is a hook overlay — it draws over the opening
    footage and shifts nothing.
    """
    config = intro_preset(preset)
    if config is None:
        raise clip_ops.ClipOpError(f"Unknown intro preset: {preset}")

    fps = _fps(timeline)
    duration_frames = max(1, time_to_frame(float(config.get("duration", 3.0)),
                                           timeline.fps_num, timeline.fps_den))
    background = config.get("background")

    clear_generated(timeline, INTRO_ORIGIN)

    if background:
        previous_offset = timeline.program_offset_frames
        delta = duration_frames - previous_offset
        _shift_all_items(timeline, delta)
        timeline.program_offset_frames = duration_frames
        timeline.program_offset_color = background
        # Re-cut V1/A1 so the program starts cleanly after the card. Without a
        # transcript there is nothing to rebuild from and the shift above already
        # did the job.
        source_id = _primary_source_id(timeline)
        if timeline.words and source_id:
            rebuild_primary_tracks(timeline, source_id)
    elif timeline.program_offset_frames:
        # Switching to an overlay-style intro: drop any card left by the last one.
        remove_intro(timeline, keep_text=True)

    values = {"title": title or "", "subtitle": subtitle or ""}
    created: List[TimelineItem] = []
    for beat in config.get("beats", []):
        content = str(beat.get("text", "")).format(**values).strip()
        if not content:
            continue
        start = time_to_frame(float(beat.get("start", 0.0)), timeline.fps_num, timeline.fps_den)
        end = time_to_frame(float(beat.get("end", config.get("duration", 3.0))),
                            timeline.fps_num, timeline.fps_den)
        if end <= start:
            end = start + max(1, int(fps))

        style = {**text_preset_style(beat.get("preset")), **(beat.get("style") or {})}
        item = TimelineItem(
            id=f"intro_{uuid.uuid4().hex[:8]}",
            track=track,
            source_id=None,
            kind="text",
            timeline_start_frame=start,
            timeline_end_frame=end,
            origin=INTRO_ORIGIN,
            text=clip_ops.build_text_clip(content, preset=None, style=style),
            label=content[:40],
        )
        item.text.preset = beat.get("preset")
        timeline.items.append(item)
        created.append(item)

    timeline.recalculate_duration()
    timeline.revision += 1
    return {
        "preset": preset,
        "duration_frames": duration_frames if background else 0,
        "items": created,
    }


def remove_intro(timeline: Timeline, keep_text: bool = False) -> None:
    """Drop the intro card and pull the program back to the head of the timeline."""
    if not keep_text:
        clear_generated(timeline, INTRO_ORIGIN)

    offset = timeline.program_offset_frames
    if offset:
        _shift_all_items(timeline, -offset)
        timeline.program_offset_frames = 0
        source_id = _primary_source_id(timeline)
        if timeline.words and source_id:
            rebuild_primary_tracks(timeline, source_id)

    timeline.recalculate_duration()
    timeline.revision += 1
