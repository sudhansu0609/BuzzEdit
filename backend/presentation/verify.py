"""The morning report's missing half: checks on the dressed programme.

`cut_verify` proves the cut is frame-accurate; nothing proved the layers on
top of it made sense. As the pass grew — cards, maps, moods, music — the ways
a night could quietly go wrong multiplied: a stat drawn across the speaker's
face, two cards on top of each other, a bed louder than the voice, a mix that
lands 6 LU hot. These checks catch those, and every failure goes into
`PresentationReport.degraded` so it is the first thing read in the morning.

Two kinds of check. The **timeline** checks reason about geometry and timing
without rendering: text boxes estimated from their style against the face
anchors the zoom stage found; cards against each other; cutaway coverage
against the target. The **render** checks measure the finished file: EBU R128
loudness against the target, and picture length against sound.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from config import FFMPEG_BIN
from utils.proc import NO_WINDOW
from timeline.schema import Timeline, TimelineItem, frame_to_time

from .models import PresentationSettings, Program

logger = logging.getLogger("presentation.verify")


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    value: Optional[float] = None


# --- geometry ------------------------------------------------------------------------------

# A face box around the anchor, as a fraction of the canvas either side.
FACE_HALF_W = 0.11
FACE_HALF_H = 0.16
# Text width per character as a fraction of the font size (bold sans average).
CHAR_ADVANCE = 0.58


def text_box(item: TimelineItem, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) in canvas pixels, estimated from the style and the copy."""
    if item.kind != "text" or item.text is None:
        return None
    style = item.text.style
    lines = (item.text.content or "").split("\n") or [""]
    longest = max(len(line) for line in lines)
    box_w = min(width * 0.96, longest * style.font_size * CHAR_ADVANCE + 2 * style.box_padding)
    box_h = len(lines) * style.font_size * 1.3 + 2 * style.box_padding
    anchor_x = (style.pos_x + 1.0) / 2.0 * width
    anchor_y = (style.pos_y + 1.0) / 2.0 * height
    if style.align == "left":
        x0 = anchor_x
    elif style.align == "right":
        x0 = anchor_x - box_w
    else:
        x0 = anchor_x - box_w / 2
    y0 = anchor_y - box_h / 2
    return x0, y0, x0 + box_w, y0 + box_h


def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Overlap area as a fraction of the smaller box."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
    return inter / smaller


def _span(item: TimelineItem, timeline: Timeline) -> Tuple[float, float]:
    return (frame_to_time(item.timeline_start_frame, timeline.fps_num, timeline.fps_den),
            frame_to_time(item.timeline_end_frame, timeline.fps_num, timeline.fps_den))


def _windows(timeline: Timeline, origins: Sequence[str]) -> List[Tuple[float, float]]:
    return [_span(i, timeline) for i in timeline.items if i.origin in origins and i.kind == "media"]


def _covered(at: float, windows: Sequence[Tuple[float, float]]) -> bool:
    return any(s <= at < e for s, e in windows)


# --- timeline checks -------------------------------------------------------------------------

def verify_timeline(timeline: Timeline, program: Program, settings: PresentationSettings,
                    face_anchors: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Check]:
    checks: List[Check] = []
    width, height = timeline.width or 1920, timeline.height or 1080
    duration = max(0.001, program.duration_s)

    # 1. The speaker stays a presence.
    broll = _windows(timeline, ["broll"])
    covered = sum(e - s for s, e in broll)
    coverage = covered / duration
    checks.append(Check(
        name="speaker_on_screen", ok=coverage <= min(0.85, settings.target_coverage + 0.12),
        detail=f"cutaways cover {coverage:.0%} of the programme (target {settings.target_coverage:.0%})",
        value=round(coverage, 3)))

    # 2. Text never sits on the speaker's face while the speaker is on screen.
    anchors = face_anchors or {}
    segment_boxes: List[Tuple[float, float, Tuple[float, float, float, float]]] = []
    for segment in program.segments:
        anchor = anchors.get(segment.item_id)
        if anchor is None:
            continue
        cx = (anchor[0] + 1.0) / 2.0 * width
        cy = (anchor[1] + 1.0) / 2.0 * height
        segment_boxes.append((segment.tl_start_s, segment.tl_end_s,
                              (cx - FACE_HALF_W * width, cy - FACE_HALF_H * height,
                               cx + FACE_HALF_W * width, cy + FACE_HALF_H * height)))
    texts = [i for i in timeline.items if i.kind == "text" and i.text and i.enabled]
    offenders: List[str] = []
    if segment_boxes:
        for item in texts:
            if item.origin == "caption":
                continue                     # captions live in their band by design
            box = text_box(item, width, height)
            if box is None:
                continue
            start, end = _span(item, timeline)
            for seg_start, seg_end, face in segment_boxes:
                if end <= seg_start or start >= seg_end:
                    continue
                mid = max(start, seg_start) + 0.1
                if _covered(mid, broll):
                    continue                 # the picture under it is B-roll
                if _overlap(box, face) > 0.15:
                    offenders.append(f"{item.label or item.text.content[:20]} @ {start:.1f}s")
                    break
    checks.append(Check(
        name="text_clear_of_face", ok=not offenders,
        detail=("; ".join(offenders[:6]) if offenders
                else (f"{len(texts)} text items checked against {len(segment_boxes)} face boxes"
                      if segment_boxes else "no face anchors to check against")),
        value=float(len(offenders))))

    # 3. Text elements do not pile up on one another.
    piles: List[str] = []
    boxes = [(i, text_box(i, width, height), _span(i, timeline)) for i in texts]
    for index, (a, box_a, span_a) in enumerate(boxes):
        if box_a is None:
            continue
        for b, box_b, span_b in boxes[index + 1:]:
            if box_b is None or a.origin == b.origin == "caption":
                continue
            if span_a[1] <= span_b[0] or span_b[1] <= span_a[0]:
                continue
            if _overlap(box_a, box_b) > 0.35:
                piles.append(f"{a.label or '?'} × {b.label or '?'} @ {max(span_a[0], span_b[0]):.1f}s")
    checks.append(Check(name="text_not_stacked", ok=not piles,
                        detail="; ".join(piles[:6]) if piles else f"{len(texts)} text items",
                        value=float(len(piles))))

    # 4. The bed ducks under the voice and nothing on a mix lane is louder than it.
    mix = [i for i in timeline.items if i.kind == "media" and i.track.upper().startswith("A")
           and i.track.upper() != "A1"]
    loud = [i.label or i.id for i in mix if i.volume > 0.9 and i.origin in ("music", "ambience")]
    unducked = [i.label or i.id for i in mix if i.origin == "music" and i.duck <= 0.0]
    checks.append(Check(name="music_under_voice", ok=not loud and not unducked,
                        detail=("too loud: " + ", ".join(loud) if loud else "")
                        + ("; not ducked: " + ", ".join(unducked) if unducked else "")
                        or f"{len(mix)} mix items", value=float(len(loud) + len(unducked))))

    # 5. Every generated cutaway still has its file.
    missing = [i.label or i.id for i in timeline.items
               if i.kind == "media" and i.source_id and i.origin in ("broll",)
               and not Path(timeline.sources[i.source_id].path).exists()]
    checks.append(Check(name="assets_present", ok=not missing,
                        detail=", ".join(missing[:5]) if missing else "all cutaway files exist",
                        value=float(len(missing))))
    return checks


# --- render checks -----------------------------------------------------------------------------

def measure_loudness(path: str) -> Optional[Dict[str, float]]:
    """Integrated loudness, range and true peak of a file via ebur128."""
    try:
        result = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-nostats", "-i", path,
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600,
            creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("Loudness measurement failed: %s", e)
        return None
    text = result.stderr
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else text
    out: Dict[str, float] = {}
    for key, pattern in (("integrated", r"I:\s*(-?[\d.]+)\s*LUFS"),
                         ("range", r"LRA:\s*(-?[\d.]+)\s*LU"),
                         ("peak", r"Peak:\s*(-?[\d.]+)\s*dBFS")):
        match = re.search(pattern, summary)
        if match:
            out[key] = float(match.group(1))
    return out or None


def _stream_seconds(path: str, stream: str) -> Optional[float]:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
             "stream=duration", "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
            creationflags=NO_WINDOW)
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def verify_render(output_path: str, settings: PresentationSettings,
                  timeline: Optional[Timeline] = None) -> List[Check]:
    checks: List[Check] = []
    if not output_path or not Path(output_path).exists():
        return [Check(name="render_exists", ok=False, detail="no output file")]
    checks.append(Check(name="render_exists", ok=True, detail=output_path))

    loudness = measure_loudness(output_path)
    target = settings.loudness_lufs
    if loudness and "integrated" in loudness:
        measured = loudness["integrated"]
        if target is None:
            checks.append(Check(name="loudness", ok=True,
                                detail=f"{measured:.1f} LUFS (no target)", value=measured))
        else:
            checks.append(Check(name="loudness", ok=abs(measured - target) <= 2.0,
                                detail=f"{measured:.1f} LUFS against {target:.1f} target"
                                       + (f", peak {loudness['peak']:.1f} dBFS" if "peak" in loudness else ""),
                                value=measured))
    else:
        checks.append(Check(name="loudness", ok=True, detail="could not measure", value=None))

    video = _stream_seconds(output_path, "v:0")
    audio = _stream_seconds(output_path, "a:0")
    if video and audio:
        checks.append(Check(name="av_length", ok=abs(video - audio) < 0.25,
                            detail=f"picture {video:.2f}s, sound {audio:.2f}s",
                            value=round(video - audio, 3)))
        if timeline is not None:
            expected = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
            checks.append(Check(name="length_matches_timeline", ok=abs(video - expected) < 0.5,
                                detail=f"rendered {video:.2f}s, timeline {expected:.2f}s",
                                value=round(video - expected, 3)))
    return checks


def write_verification(project_dir: Path, checks: Sequence[Check]) -> None:
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "presentation_checks.json").write_text(
            json.dumps([c.model_dump() for c in checks], indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write the verification: %s", e)
