"""Composites: picture-in-picture, split screen and the freeze-frame.

Three things an explainer does with the picture that a full-frame cutaway
cannot:

* **Picture-in-picture.** During a long cutaway the speaker stays on screen
  in a corner, so a point they make with their face and hands is not lost
  behind a still. The corner clip is the same source over the same frames
  as the V1 programme beneath it, scaled down and parked top-right.
* **Split screen.** "A versus B": two generated stills side by side, each
  cropped to its half, with a label under each. Planned as a `split` beat
  (from a `[split: A | B]` stage direction or a comparison the entity pass
  found) and expanded into a pair of ordinary B-roll beats before
  generation, so ComfyUI makes two pictures and placement puts them up
  together.
* **Freeze-frame.** At a stat call-out the speaker freezes for a moment
  while the number counts up over them and the voice carries on. The frame
  is pulled from the source once and placed as a still on the composite
  track.

All three live on their own track and origin, above the B-roll and below the
cards, and a re-run replaces exactly them.
"""

import hashlib
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import FFMPEG_BIN
from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import SourceFile, Timeline, frame_to_time, time_to_frame

from .models import Beat, PresentationSettings, Program

logger = logging.getLogger("presentation.composite")

COMPOSITE_TRACK = "V5"
COMPOSITE_ORIGIN = "composite"
SPLIT_LEFT_TRACK = "V3"
SPLIT_RIGHT_TRACK = "V4"
SPLIT_LABEL_TRACK = "TX"

PIP_SCALE = 0.3
PIP_POS = (0.66, -0.58)         # top-right, clear of pop-ups (top centre) and cards (top-left)
PIP_MIN_CUTAWAY_S = 4.0
# Genres where a corner speaker suits the form. Horror wants the picture alone.
PIP_GENRES = {"science_education", "tech", "finance", "news", "gaming", "health_fitness",
              "cooking", "general", "vlog", "motivational", "travel"}

FREEZE_SECONDS = 1.4
FREEZE_ZOOM = 0.05


# --- picture-in-picture -----------------------------------------------------------------

def place_pip(timeline: Timeline, program: Program, settings: PresentationSettings,
              genre: str) -> int:
    """The speaker in a corner over every long cutaway. Returns how many."""
    clear_generated(timeline, COMPOSITE_ORIGIN)
    mode = (settings.pip or "auto").lower()
    if mode in ("off", "none", "") or (mode == "auto" and genre not in PIP_GENRES):
        return 0
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    cutaways = sorted(
        (i for i in timeline.items if i.origin == "broll" and i.kind == "media"),
        key=lambda i: i.timeline_start_frame)
    v1 = sorted((i for i in timeline.items if i.track == "V1" and i.kind == "media" and i.enabled),
                key=lambda i: i.timeline_start_frame)
    count = 0
    for cut in cutaways:
        seconds = frame_to_time(cut.duration_frames, fps_num, fps_den)
        if seconds < settings.pip_min_seconds:
            continue
        # The same frames as the programme beneath, segment by segment.
        for segment in v1:
            lo = max(cut.timeline_start_frame, segment.timeline_start_frame)
            hi = min(cut.timeline_end_frame, segment.timeline_end_frame)
            if hi - lo <= 2 or not segment.source_id:
                continue
            offset = lo - segment.timeline_start_frame
            item = clip_ops.add_media_item(
                timeline, segment.source_id, COMPOSITE_TRACK, lo,
                segment.source_start_frame + offset,
                segment.source_start_frame + offset + (hi - lo),
                origin=COMPOSITE_ORIGIN)
            item.mute = True
            item.label = "pip: speaker"
            clip_ops.set_transform(timeline, item.id, {
                "scale": PIP_SCALE, "pos_x": PIP_POS[0], "pos_y": PIP_POS[1]})
        count += 1
    if count:
        timeline.recalculate_duration()
        timeline.revision += 1
    logger.info("Picture-in-picture over %d cutaways", count)
    return count


# --- split screen -----------------------------------------------------------------------

def expand_splits(beats: Sequence[Beat]) -> List[Beat]:
    """A `split` beat becomes two B-roll beats sharing a window, marked as a pair."""
    out: List[Beat] = []
    for beat in beats:
        if beat.kind != "split":
            out.append(beat)
            continue
        left = beat.data.get("left") or beat.image_prompt
        right = beat.data.get("right")
        if not left or not right:
            continue
        pair = beat.id or f"split_{uuid.uuid4().hex[:6]}"
        labels = beat.data.get("labels") or [None, None]
        for side, prompt, label in (("left", left, labels[0]), ("right", right, labels[1])):
            out.append(beat.model_copy(update={
                "id": f"{pair}_{side[0]}",
                "kind": "broll_image",
                "image_prompt": str(prompt),
                "video_prompt": None,
                "data": {"split": side, "pair": pair, "label": label},
                "planned_duration_s": beat.planned_duration_s,
            }))
    return out


def split_transform(side: str) -> Dict[str, float]:
    """Each half shows the centre of its picture on its side of the canvas."""
    if side == "left":
        return {"crop_left": 0.25, "crop_right": 0.25, "scale": 1.0, "pos_x": -1.0}
    return {"crop_left": 0.25, "crop_right": 0.25, "scale": 1.0, "pos_x": 1.0}


def place_split_labels(timeline: Timeline, pairs: Sequence[Tuple[float, float, Optional[str], Optional[str]]],
                       genre: str) -> int:
    """'A' and 'B' under their halves, for each placed split."""
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    count = 0
    for start_s, end_s, left, right in pairs:
        frames = max(1, time_to_frame(end_s - start_s, fps_num, fps_den))
        for label, pos_x in ((left, -0.5), (right, 0.5)):
            if not label:
                continue
            item = clip_ops.add_text_item(
                timeline, str(label), time_to_frame(start_s, fps_num, fps_den), frames,
                track=SPLIT_LABEL_TRACK, preset="character_card",
                style={"align": "center", "pos_x": pos_x, "pos_y": 0.72,
                       "animation": "slide-up"})
            item.origin = "split"
            item.label = f"split label: {label}"
            count += 1
    return count


# --- freeze frame -----------------------------------------------------------------------

def grab_frame(source_path: str, source_seconds: float, destination: Path) -> Optional[Path]:
    """One frame of the source as a PNG, or None when it cannot be read."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{max(0.0, source_seconds):.3f}", "-i", source_path,
             "-frames:v", "1", str(destination)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except Exception as e:
        logger.warning("Could not grab a frame at %.2fs: %s", source_seconds, e)
        return None
    return destination if destination.exists() else None


def place_freezes(timeline: Timeline, program: Program, beats: Sequence[Beat],
                  settings: PresentationSettings, project_dir: Path,
                  grab=grab_frame) -> int:
    """Freeze the speaker under every stat call-out that lands on the speaker."""
    if not settings.freeze_on_stats:
        return 0
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    fps = fps_num / max(1, fps_den)
    busy = [(frame_to_time(i.timeline_start_frame, fps_num, fps_den),
             frame_to_time(i.timeline_end_frame, fps_num, fps_den))
            for i in timeline.items if i.origin == "broll"]
    count = 0
    for beat in beats:
        if beat.kind != "stat_callout":
            continue
        at = beat.start_s
        if any(s <= at < e for s, e in busy):
            continue                       # a cutaway is on screen: nothing to freeze
        segment = program.segment_at(at)
        if segment is None:
            continue
        item = next((i for i in timeline.items if i.id == segment.item_id), None)
        if item is None or not item.source_id:
            continue
        source = timeline.sources.get(item.source_id)
        if source is None:
            continue
        source_frame = segment.source_start_frame + int(round((at - segment.tl_start_s) * fps))
        key = hashlib.sha1(f"{source.path}:{source_frame}".encode()).hexdigest()[:10]
        path = grab(source.path, source_frame / fps, Path(project_dir) / "assets" / "freezes" / f"freeze_{key}.png")
        if path is None:
            continue
        source_id = f"src_freeze_{key}"
        if source_id not in timeline.sources:
            timeline.sources[source_id] = SourceFile(
                id=source_id, path=str(path), duration_seconds=FREEZE_SECONDS,
                width=source.width, height=source.height, has_audio=False, kind="image")
        frames = max(1, time_to_frame(min(FREEZE_SECONDS, max(0.5, beat.duration_s)), fps_num, fps_den))
        placed = clip_ops.add_media_item(timeline, source_id, COMPOSITE_TRACK,
                                         time_to_frame(at, fps_num, fps_den), 0, frames,
                                         origin=COMPOSITE_ORIGIN)
        placed.label = f"freeze: {beat.text}"
        clip_ops.set_transform(timeline, placed.id, {"scale": 1.0, "scale_end": 1.0 + FREEZE_ZOOM})
        clip_ops.set_color(timeline, placed.id, {"saturation": 0.75, "contrast": 1.08})
        count += 1
    if count:
        timeline.recalculate_duration()
        timeline.revision += 1
    logger.info("Freeze-frames under %d stat call-outs", count)
    return count
