"""Vertical variants for distribution: a 9:16 reframe and the best topics as Shorts.

The dressed timeline is the source of truth; nothing here re-plans. A **vertical
variant** is the same timeline on a 1080×1920 canvas: a landscape source is
scaled to cover the height and cropped around the speaker's face (the anchors
the zoom stage found, one per segment), the full-frame cutaways cover the
canvas the same way, and the text keeps its relative positions. A portrait
source needs no reframing and is used as it is.

**Shorts** are slices of that vertical timeline: the top topics by priority,
each clipped to its span (or to the Shorts limit), with every layer that
overlaps the slice trimmed to it. They render as separate files next to the
main output.
"""

import asyncio
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from timeline.schema import Timeline, TimelineItem, Transform, frame_to_time, time_to_frame

from . import facezoom
from .models import PresentationSettings, Program, Topic

logger = logging.getLogger("presentation.shorts")

SHORT_W, SHORT_H = 1080, 1920
# YouTube Shorts run up to 60 s; a topic longer than this is cut to its opening.
SHORT_MAX_S = 58.0
SHORT_MIN_S = 12.0


def is_portrait(timeline: Timeline) -> bool:
    return bool(timeline.width and timeline.height and timeline.height > timeline.width)


def cover_scale(source_w: int, source_h: int, canvas_w: int, canvas_h: int) -> float:
    """The Transform.scale that makes a source fill the canvas (crop, not fit).

    The canvas transform scales the source to fit inside (canvas_w·s, canvas_h·s)
    keeping its aspect, so to cover the canvas the limiting side has to reach it.
    """
    if not source_w or not source_h:
        return 1.0
    fit_w = canvas_w
    fit_h = fit_w * source_h / source_w
    if fit_h >= canvas_h:
        return 1.0
    return canvas_h / fit_h


def face_pos_x(anchor: Optional[Tuple[float, float]]) -> float:
    """The crop's horizontal position (-1..1) that centres the face.

    The zoom stage pulls its anchors toward the centre (so a punch-in never
    aims at the frame's edge); a reframe wants the face itself, so the pull is
    undone. Positive pos_x moves the crop window right.
    """
    if anchor is None:
        return 0.0
    raw = anchor[0] / max(0.05, facezoom.FACE_ANCHOR_PULL)
    return max(-1.0, min(1.0, raw))


def vertical_variant(timeline: Timeline, program: Program,
                     anchors: Optional[Dict[str, Tuple[float, float]]] = None) -> Timeline:
    """The timeline on a 9:16 canvas, reframed around the speaker."""
    vertical = copy.deepcopy(timeline)
    if is_portrait(timeline):
        return vertical
    src_w, src_h = timeline.width or 1920, timeline.height or 1080
    vertical.width, vertical.height = SHORT_W, SHORT_H
    scale = cover_scale(src_w, src_h, SHORT_W, SHORT_H)
    anchors = anchors or {}

    for item in vertical.items:
        if item.kind != "media" or not item.source_id:
            continue
        source = vertical.sources.get(item.source_id)
        if source is None or source.kind == "audio":
            continue
        if item.track == "V1":
            # Follow the face; keep any push-in the zoom stage put on the clip.
            base = item.transform or Transform()
            pos = face_pos_x(anchors.get(item.id))
            item.transform = Transform(
                scale=base.scale * scale,
                scale_end=(base.scale_end * scale) if base.scale_end is not None else None,
                pos_x=pos, pos_x_end=pos if base.is_animated() else None,
                pos_y=base.pos_y, pos_y_end=base.pos_y_end)
        elif item.track.upper().startswith("V"):
            # Cutaways generated at the landscape size: cover the tall canvas.
            item_scale = cover_scale(source.width or src_w, source.height or src_h,
                                     SHORT_W, SHORT_H)
            base = item.transform or Transform()
            item.transform = base.model_copy(update={
                "scale": base.scale * item_scale,
                "scale_end": (base.scale_end * item_scale) if base.scale_end is not None else None,
            })
    # The primary source's own size drives the compiler's canvas; the copy
    # declares the canvas explicitly by re-sizing the primary source entry.
    for source in vertical.sources.values():
        if source.kind == "video" and any(i.source_id == source.id and i.track == "V1"
                                          for i in vertical.items):
            source.width, source.height = SHORT_W, SHORT_H
    return vertical


# --- slicing ----------------------------------------------------------------------------------

def slice_timeline(timeline: Timeline, start_s: float, end_s: float) -> Timeline:
    """Every layer of the timeline between two moments, re-based to zero."""
    out = copy.deepcopy(timeline)
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    start_f = time_to_frame(start_s, fps_num, fps_den)
    end_f = time_to_frame(end_s, fps_num, fps_den)
    kept: List[TimelineItem] = []
    for item in out.items:
        lo = max(item.timeline_start_frame, start_f)
        hi = min(item.timeline_end_frame, end_f)
        if hi - lo <= 0:
            continue
        head = lo - item.timeline_start_frame
        tail = item.timeline_end_frame - hi
        if item.kind == "media" and item.source_id:
            source = out.sources.get(item.source_id)
            still = source is not None and source.kind == "image"
            if not still and not item.loop:
                item.source_start_frame += head
                item.source_end_frame -= tail
            elif item.loop:
                item.source_end_frame = item.source_start_frame + (hi - lo)
        if item.text is not None and item.text.words:
            shift = frame_to_time(head, fps_num, fps_den)
            item.text.words = [
                {**w, "start_s": round(max(0.0, float(w["start_s"]) - shift), 3),
                 "end_s": round(max(0.0, float(w["end_s"]) - shift), 3)}
                for w in item.text.words if float(w["end_s"]) > shift]
        item.timeline_start_frame = lo - start_f
        item.timeline_end_frame = hi - start_f
        item.transition = None if item.timeline_start_frame == 0 else item.transition
        kept.append(item)
    out.items = kept
    # The pad (title card, cold open) belongs to the full programme only.
    pad = max(0, out.program_offset_frames - start_f)
    out.program_offset_frames = pad if start_f < out.program_offset_frames else 0
    out.cold_open_frames = 0
    out.recalculate_duration()
    out.revision += 1
    return out


# A word that closes a sentence: trailing full stop / question / exclamation /
# danda, or a real breath (a pause this long) before the next word.
_SENTENCE_END_CHARS = (".", "?", "!", "।", "…")
_SENTENCE_PAUSE_S = 0.7
# How far past a topic's end a short may run to finish the sentence it cuts into.
_FINISH_SENTENCE_S = 4.0


def _sentence_bounds(words) -> Tuple[List[float], List[float]]:
    """(sentence start times, sentence end times) over the programme words."""
    starts: List[float] = []
    ends: List[float] = []
    at_sentence_start = True
    for i, w in enumerate(words):
        if at_sentence_start:
            starts.append(w.tl_start_s)
        nxt = words[i + 1] if i + 1 < len(words) else None
        at_sentence_start = (w.text.strip().rstrip("\"'”’").endswith(_SENTENCE_END_CHARS)
                             or nxt is None or nxt.tl_start_s - w.tl_end_s >= _SENTENCE_PAUSE_S)
        if at_sentence_start:
            ends.append(w.tl_end_s)
    return starts, ends


def pick_shorts(topics: Sequence[Topic], program: Program, count: int = 3) -> List[Tuple[float, float, Topic]]:
    """The best `count` topics as (start, end, topic).

    A short has to land its point, so every clip ends on a sentence boundary
    (running up to `_FINISH_SENTENCE_S` past the topic to finish the sentence
    the topic's edge cuts into). A topic longer than `SHORT_MAX_S` is trimmed
    from the FRONT, to a sentence start, so the clip keeps the topic's
    conclusion rather than stopping mid-argument at the 58 s mark."""
    words = sorted(program.words, key=lambda w: w.tl_start_s)
    sentence_starts, sentence_ends = _sentence_bounds(words)
    candidates = []
    for topic in topics:
        if (topic.act or "") == "cta":
            continue
        start = max(0.0, topic.start_s)
        end = min(program.duration_s, topic.end_s)
        # Begin on the sentence the topic opens in, end on the one it closes in.
        before = [t for t in sentence_starts if start - 2.0 <= t <= start + 0.2]
        if before:
            start = max(0.0, before[-1] - 0.1)
        after = [t for t in sentence_ends if end - 0.3 <= t <= end + _FINISH_SENTENCE_S]
        if after:
            end = min(program.duration_s, after[0] + 0.3)
        if end - start > SHORT_MAX_S:
            # Keep the payoff: the last sentence-aligned SHORT_MAX_S of the topic.
            tail = [t for t in sentence_starts if end - SHORT_MAX_S <= t < end - SHORT_MIN_S]
            start = (tail[0] - 0.1) if tail else end - SHORT_MAX_S
        if end - start < SHORT_MIN_S:
            continue
        candidates.append((topic.priority, start, end, topic))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    picked = [(s, e, t) for _, s, e, t in candidates[:max(0, count)]]
    picked.sort(key=lambda c: c[0])
    return picked


async def render_shorts(timeline: Timeline, program: Program, topics: Sequence[Topic],
                        settings: PresentationSettings, output_stem: Path,
                        anchors: Optional[Dict[str, Tuple[float, float]]] = None,
                        progress=None) -> List[Dict[str, object]]:
    """Render the vertical variant (when asked) and the Shorts clips."""
    from render.runner import render_timeline_async
    results: List[Dict[str, object]] = []
    vertical = vertical_variant(timeline, program, anchors)
    offset_s = frame_to_time(timeline.cold_open_frames, timeline.fps_num, timeline.fps_den)

    if settings.shorts_full:
        path = str(output_stem) + "_vertical.mp4"
        await render_timeline_async(vertical, path, progress_callback=progress,
                                    output_resolution=f"{SHORT_W}x{SHORT_H}")
        results.append({"kind": "vertical", "path": path,
                        "seconds": round(frame_to_time(vertical.duration_frames,
                                                       vertical.fps_num, vertical.fps_den), 2)})

    # The Shorts are independent renders: run them together rather than one
    # after another (~4.6 min back to back on a real run). Each ffmpeg keeps
    # to its own few threads and the GPU encoder takes several sessions.
    pad_s = frame_to_time(timeline.program_offset_frames, timeline.fps_num, timeline.fps_den)

    async def one(index: int, start: float, end: float, topic) -> Optional[Dict[str, object]]:
        # Topic times are on the programme's clock; the timeline may carry a
        # cold open / title pad in front of it.
        clip = slice_timeline(vertical, start + pad_s, end + pad_s)
        if clip.duration_frames <= 0:
            return None
        path = f"{output_stem}_short{index + 1}.mp4"
        try:
            await render_timeline_async(clip, path, progress_callback=progress,
                                        output_resolution=f"{SHORT_W}x{SHORT_H}")
        except Exception as e:
            logger.warning("Short %d (%s) failed: %s", index + 1, topic.topic, e)
            return {"kind": "short", "topic": topic.topic, "error": str(e)[:200]}
        logger.info("Short %d: %r %.1fs → %s", index + 1, topic.topic, end - start, path)
        return {"kind": "short", "path": path, "topic": topic.topic,
                "from_s": round(start + offset_s, 2), "seconds": round(end - start, 2)}

    picked = pick_shorts(topics, program, settings.shorts_clips)
    done = await asyncio.gather(*(one(i, s, e, t) for i, (s, e, t) in enumerate(picked)))
    results.extend(r for r in done if r is not None)
    return results
