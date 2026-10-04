"""Stage E — punch-ins on the speaker, timed to how they are talking.

Two things have to be true for an automatic zoom to look deliberate rather than
random: it has to be **aimed at the face**, and it has to happen **when the
speaker leans into something**. The pieces for both are already in the project
and were simply never connected — the auto-edit measures loudness at 50Hz and
stores it on the timeline, and OpenCV ships a face detector in the version this
app already depends on.

Two kinds of move come out of it:

*Segment zooms* run the whole length of one programme clip. They are the
baseline rhythm, they survive a transcript edit (the rebuild carries a clip's
transform across by anchor word), and they are what `style.apply_motion` already
does for a reference-matched edit.

*Windowed punch-ins* start and end inside a clip, on a moment of emphasis. These
cannot be done with a clip transform — a transform animates across the clip's
whole length, and V1 clips cannot be split because the rebuild owns them. They
go on an **adjustment layer** instead, which the compiler windows with
`enable=between(t,a,b)` over whatever is below it. That mechanism already exists
for hand-made adjustment clips; this is the first thing to generate them.
"""

import logging
import math
import urllib.request
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR
from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import Timeline, time_to_frame

from .models import Program, ProgramSegment, PresentationSettings, SegmentZoom, WindowZoom

logger = logging.getLogger("presentation.facezoom")

ZOOM_ORIGIN = "autozoom"
ZOOM_TRACK = "V2"

# --- the rhythm ------------------------------------------------------------

# Roughly one move across cut segments. Enough that the frame is dynamic
# throughout the video, covering ~90% of segments with alternating in/out zooms.
ZOOMS_PER_MINUTE = 60.0
# Minimum segment length to receive a zoom move.
MIN_ZOOM_SEGMENT_SECONDS = 0.5
# ...and the most. A segment move exists to disguise a jump cut: a clip of a
# few seconds drifts tighter so the next one can step back wide. On one long
# take (a recording that arrives already cut) it became a single 1.00 -> 1.31
# drift across all 473 s of a real render -- invisible as motion (0.07%/s),
# 31% softer by the end, and a 3x-supersampled zoompan on every one of ~11,800
# frames, the costliest chain in the whole render. Longer shots get their
# movement from the punch-ins instead.
MAX_ZOOM_SEGMENT_SECONDS = 20.0

# --- the punches -----------------------------------------------------------

# How loud, relative to the speaker's own average, counts as emphasis.
PUNCH_Z_THRESHOLD = 1.2
# Long enough for an ease in, a hold and an ease out that all read as motion
# rather than a twitch (see `_punch_envelope`).
PUNCH_MIN_SECONDS = 1.2
PUNCH_MAX_SECONDS = 2.5
# Past this a 1080p source visibly softens when blown up.
PUNCH_DEPTH_MAX = 0.25
MIN_PUNCH_GAP_SECONDS = 6.0
# Historical fallback when settings carry neither an explicit rate nor a
# density (see PresentationSettings.effective_punch_rate_per_minute).
MAX_PUNCHES_PER_MINUTE = 1.5
# A punch that begins on a cut reads as a mistake rather than a choice.
PUNCH_EDGE_MARGIN = 0.5
# When there are too few peaks at PUNCH_Z_THRESHOLD to fill the allowance, the
# threshold eases down rather than leaving punches unfilled — but never below
# this, or "emphasis" stops meaning anything.
PUNCH_Z_FLOOR = 0.6
PUNCH_Z_STEP = 0.1
# A quick, shallow-duration "whip" push on top of an already-chosen punch —
# busy/max density only, every 4th push-in, alternating with pull-backs.
WHIP_MAX_SECONDS = 0.5
WHIP_DEPTH_BONUS = 0.05
WHIP_EVERY = 4

# --- the face --------------------------------------------------------------

# The zoo stores its weights in Git LFS: the raw.githubusercontent URL returns
# a 131-byte pointer file, which is what sat on disk for weeks and failed to
# parse on every run. The media endpoint serves the actual bytes.
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_PATH = DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
# The real model is ~230 KB; anything under this is a pointer or an error page.
MODEL_MIN_BYTES = 100_000
FACE_SAMPLE_FPS = 2.0
FACE_SAMPLE_WIDTH = 640
FACE_SAMPLE_HEIGHT = 360
# Below this many hits in a segment there is no reliable face to aim at.
MIN_FACE_SAMPLES = 3
# Zooming hard toward a face at the edge of frame crops the composition badly,
# so the anchor is pulled back toward centre rather than used raw.
FACE_ANCHOR_PULL = 0.6
MAX_ANCHOR_OFFSET = 0.35


def ensure_model() -> Optional[Path]:
    """The YuNet weights, downloading them once. None if unavailable offline."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MODEL_MIN_BYTES:
        return MODEL_PATH
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading the face detection model (once)")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp:
            payload = resp.read()
        if len(payload) < MODEL_MIN_BYTES or payload.startswith(b"version https://git-lfs"):
            logger.warning("Face model download is not a model (%d bytes); zooms will be centred",
                           len(payload))
            return None
        MODEL_PATH.write_bytes(payload)
        return MODEL_PATH
    except Exception as e:
        logger.warning("Face model unavailable (%s); zooms will be centred", e)
        return None


def detect_faces(source_path: str, program: Program) -> Dict[str, Tuple[float, float]]:
    """Where the speaker's face sits in each programme clip, in canvas coords.

    Frames are decoded once through `style.frames.sample` rather than seeking per
    frame — the codebase settled on that after measuring OpenCV's per-frame seek
    as punishingly slow on long files.
    """
    model = ensure_model()
    if model is None or not program.segments:
        return {}

    try:
        import cv2
        import numpy as np
        from style.frames import sample
    except Exception as e:
        logger.warning("Face detection unavailable (%s); zooms will be centred", e)
        return {}

    try:
        frames, fps = sample(source_path, FACE_SAMPLE_WIDTH, FACE_SAMPLE_HEIGHT,
                             FACE_SAMPLE_FPS, gray=False)
    except Exception as e:
        logger.warning("Could not sample frames for face detection (%s)", e)
        return {}
    if frames is None or len(frames) == 0:
        return {}

    height, width = frames.shape[1], frames.shape[2]
    try:
        detector = cv2.FaceDetectorYN.create(str(model), "", (width, height),
                                             score_threshold=0.7, nms_threshold=0.3,
                                             top_k=5)
    except Exception as e:
        logger.warning("Could not create the face detector (%s)", e)
        return {}

    # Detections keyed by source time, so they can be bucketed per segment below.
    hits: List[Tuple[float, float, float]] = []
    for index in range(len(frames)):
        frame = frames[index]
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
        else:
            bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_GRAY2BGR)
        try:
            _, faces = detector.detect(bgr)
        except Exception:
            continue
        if faces is None or len(faces) == 0:
            continue
        # The largest face is the speaker; anything smaller is a bystander, a
        # poster on the wall, or a false positive.
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        cx = (float(best[0]) + float(best[2]) / 2.0) / width
        cy = (float(best[1]) + float(best[3]) / 2.0) / height
        hits.append((index / max(1e-6, fps), cx, cy))

    if not hits:
        logger.info("Face detection found nothing; zooms will be centred")
        return {}

    anchors: Dict[str, Tuple[float, float]] = {}
    source_fps = program.fps or 30.0
    for segment in program.segments:
        lo = segment.source_start_frame / source_fps
        hi = segment.source_end_frame / source_fps
        inside = [(cx, cy) for t, cx, cy in hits if lo <= t < hi]
        if len(inside) < MIN_FACE_SAMPLES:
            continue
        # Median, not mean: one false positive on a background object would drag
        # a mean off the speaker and aim every zoom at the wall behind them.
        cx = median(v[0] for v in inside)
        cy = median(v[1] for v in inside)
        anchors[segment.item_id] = _to_canvas(cx, cy)

    logger.info("Face anchors for %d of %d segments", len(anchors), len(program.segments))
    return anchors


def _to_canvas(cx: float, cy: float) -> Tuple[float, float]:
    """Frame fractions to the -1..1 canvas coordinates Transform uses."""
    x = (cx * 2.0 - 1.0) * FACE_ANCHOR_PULL
    y = (cy * 2.0 - 1.0) * FACE_ANCHOR_PULL
    clamp = lambda v: max(-MAX_ANCHOR_OFFSET, min(MAX_ANCHOR_OFFSET, v))
    return clamp(x), clamp(y)


def plan_zooms(program: Program, settings: PresentationSettings,
               busy_windows: Optional[List[Tuple[float, float]]] = None,
               seed: int = 0,
               stats: Optional[Dict[str, int]] = None) -> Tuple[List[SegmentZoom], List[WindowZoom]]:
    """Which clips get a move, and where the punches land.

    `stats`, when given, receives `{"suppressed": n}` — punches that a B-roll
    cutaway squeezed out. At 75% coverage that number decides whether any
    punch-ins survive at all, so the report has to be able to show it.
    """
    busy = list(busy_windows or [])
    depth = max(0.04, min(0.6, settings.zoom_depth))

    # A segment only needs enough screen time for the move to be seen.
    # At high B-roll coverage, even partially visible clips receive transforms.
    visible_floor = max(MIN_ZOOM_SEGMENT_SECONDS, 0.4)
    eligible = [s for s in program.segments
                if MIN_ZOOM_SEGMENT_SECONDS <= s.duration_s <= MAX_ZOOM_SEGMENT_SECONDS
                and _uncovered_seconds(s.tl_start_s, s.tl_end_s, busy) >= visible_floor]
    if not eligible:
        eligible = [s for s in program.segments
                    if MIN_ZOOM_SEGMENT_SECONDS <= s.duration_s <= MAX_ZOOM_SEGMENT_SECONDS]

    segment_zooms: List[SegmentZoom] = []
    if eligible:
        # Cover ~90% of eligible segments. Every move pushes IN, deliberately:
        # each clip then starts wide and drifts tighter, so at every join the
        # frame steps back to wide by the whole depth of the move — a punch-out,
        # which is the standard disguise for a talking-head jump cut. The old
        # alternation (push, pull, push …) made the scale CONTINUOUS across
        # every join — a push-in ends at 1+d exactly where the following
        # pull-back starts — so the head-position jump played completely bare.
        # Verified on real footage: adjacent takes differ by a lean or a hand
        # move, and an 8-14% scale step across the join reads as an edit where
        # equal scale reads as a glitch. Depth is jittered per segment (seeded,
        # so re-runs are stable) to keep the rhythm from feeling mechanical.
        import random as _random
        rng = _random.Random(seed)
        wanted = max(1, int(round(len(eligible) * 0.90))) if len(eligible) > 1 else len(eligible)
        if wanted > 0:
            step = len(eligible) / wanted
            for index in range(wanted):
                segment = eligible[int(index * step)]
                jitter = 0.7 + 0.6 * rng.random()
                segment_zooms.append(SegmentZoom(
                    item_id=segment.item_id,
                    push_in=True,
                    depth=round(min(0.6, max(0.06, depth * jitter)), 3),
                ))

    window_zooms = _plan_punches(program, busy, settings, seed, stats)
    logger.info("Zooms planned: %d segment moves, %d punch-ins",
                len(segment_zooms), len(window_zooms))
    return segment_zooms, window_zooms


def _uncovered_seconds(start: float, end: float,
                       busy: List[Tuple[float, float]]) -> float:
    """Screen time in [start, end) not hidden behind a busy (B-roll) window."""
    total = max(0.0, end - start)
    for b_start, b_end in busy:
        total -= max(0.0, min(end, b_end) - max(start, b_start))
    return total


def _trim_outside(start: float, end: float,
                  busy: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """The largest stretch of [start, end] no busy window touches, or None."""
    best: Optional[Tuple[float, float]] = None

    def offer(lo: float, hi: float) -> None:
        nonlocal best
        if hi > lo and (best is None or hi - lo > best[1] - best[0]):
            best = (lo, hi)

    cursor = start
    for b_start, b_end in sorted(busy):
        if b_end <= start or b_start >= end:
            continue
        offer(cursor, min(b_start, end))
        cursor = max(cursor, b_end)
    offer(cursor, end)
    return best


def _plan_punches(program: Program, busy: List[Tuple[float, float]],
                  settings: Optional[PresentationSettings],
                  seed: int, stats: Optional[Dict[str, int]] = None) -> List[WindowZoom]:
    """Emphasis peaks that deserve a punch-in, after every safety filter."""
    if not program.has_energy or not program.words:
        return []

    settings = settings or PresentationSettings()
    counters = stats if stats is not None else {}
    counters.setdefault("suppressed", 0)

    minutes = max(1.0, program.duration_s / 60.0)
    allowance = max(1, int(round(minutes * settings.effective_punch_rate_per_minute)))

    # Too few peaks at the strict threshold to fill the allowance: ease the
    # threshold down rather than leaving a busy/max programme under-punched.
    # Strength (below) still measures against PUNCH_Z_THRESHOLD, so an eased-in
    # peak still gets the short, shallow end of the scale it earned.
    threshold = PUNCH_Z_THRESHOLD
    peaks = [w for w in program.words if w.emphasis_z >= threshold]
    while len(peaks) < allowance and threshold > PUNCH_Z_FLOOR:
        threshold = max(PUNCH_Z_FLOOR, round(threshold - PUNCH_Z_STEP, 2))
        peaks = [w for w in program.words if w.emphasis_z >= threshold]
    if not peaks:
        return []
    peaks.sort(key=lambda w: -w.emphasis_z)

    chosen: List[WindowZoom] = []

    for word in peaks:
        if len(chosen) >= allowance:
            break
        segment = program.segment_at(word.tl_start_s)
        if segment is None:
            continue

        # Scale the length and the depth with how emphatic the moment is.
        strength = max(0.0, min(1.0, (word.emphasis_z - PUNCH_Z_THRESHOLD) / 1.5))
        length = PUNCH_MIN_SECONDS + strength * (PUNCH_MAX_SECONDS - PUNCH_MIN_SECONDS)
        start = word.tl_start_s - 0.3
        end = start + length

        # Never cross a cut: the adjustment layer would carry the zoom over the
        # join and the two shots would appear to be one moving frame.
        start = max(start, segment.tl_start_s + PUNCH_EDGE_MARGIN)
        end = min(end, segment.tl_end_s - PUNCH_EDGE_MARGIN)
        if end - start < PUNCH_MIN_SECONDS:
            continue
        # A punch fully hidden behind a cutaway is wasted work and a pop on the
        # return; one that merely brushes a cutaway is trimmed to its visible
        # part instead of thrown away — at high coverage, dropping on any
        # overlap eliminated punch-ins entirely.
        if _overlaps(start, end, busy):
            visible = _trim_outside(start, end, busy)
            if visible is None or visible[1] - visible[0] < PUNCH_MIN_SECONDS:
                counters["suppressed"] += 1
                continue
            start, end = visible
        if any(start < c.end_s + MIN_PUNCH_GAP_SECONDS and c.start_s < end + MIN_PUNCH_GAP_SECONDS
               for c in chosen):
            continue

        chosen.append(WindowZoom(
            start_s=start, end_s=end,
            depth=min(PUNCH_DEPTH_MAX, 0.10 + 0.15 * strength),
            reason=f"emphasis z={word.emphasis_z:.2f}",
        ))

    chosen.sort(key=lambda w: w.start_s)
    return _shape_punches(chosen, settings)


def _shape_punches(chosen: List[WindowZoom],
                   settings: PresentationSettings) -> List[WindowZoom]:
    """Direction and the occasional whip, applied in chronological order.

    Every other punch (by time, not by how it was chosen) pulls back instead
    of pushing in, so consecutive punches read as two different moves rather
    than one repeated one. At busy/max density, every WHIP_EVERY-th push-in
    additionally shortens and deepens into a quick whip on the same moment.
    """
    density = (settings.density or "").strip().lower()
    whip_eligible = density in ("busy", "max")
    shaped: List[WindowZoom] = []
    for index, zoom in enumerate(chosen):
        push_in = index % 2 == 0
        updates: Dict[str, object] = {"push_in": push_in}
        if whip_eligible and push_in and index > 0 and (index // 2) % WHIP_EVERY == 0:
            whip_len = min(zoom.end_s - zoom.start_s, WHIP_MAX_SECONDS)
            updates["end_s"] = zoom.start_s + whip_len
            updates["depth"] = min(PUNCH_DEPTH_MAX, zoom.depth + WHIP_DEPTH_BONUS)
            updates["reason"] = f"whip {zoom.reason}"
        shaped.append(zoom.model_copy(update=updates))
    return shaped


def _overlaps(start: float, end: float, windows: List[Tuple[float, float]]) -> bool:
    return any(start < w_end and w_start < end for w_start, w_end in windows)


def apply_zooms(timeline: Timeline, segment_zooms: List[SegmentZoom],
                window_zooms: List[WindowZoom],
                anchors: Dict[str, Tuple[float, float]]) -> Tuple[int, int]:
    """Write the moves onto the timeline. Returns (segment count, punch count)."""
    clear_generated(timeline, ZOOM_ORIGIN)

    applied_segments = 0
    for zoom in segment_zooms:
        anchor_x, anchor_y = anchors.get(zoom.item_id, (0.0, 0.0))
        # Drift *toward* the face rather than starting on it: beginning a shot
        # already off-centre looks like a framing error, not a move.
        if zoom.push_in:
            values = {"scale": 1.0, "scale_end": 1.0 + zoom.depth,
                      "pos_x": anchor_x * 0.4, "pos_x_end": anchor_x,
                      "pos_y": anchor_y * 0.4, "pos_y_end": anchor_y}
        else:
            values = {"scale": 1.0 + zoom.depth, "scale_end": 1.0,
                      "pos_x": anchor_x, "pos_x_end": anchor_x * 0.4,
                      "pos_y": anchor_y, "pos_y_end": anchor_y * 0.4}
        try:
            clip_ops.set_transform(timeline, zoom.item_id, values)
            applied_segments += 1
        except Exception as e:
            logger.debug("Could not zoom segment %s: %s", zoom.item_id, e)

    applied_windows = 0
    for zoom in window_zooms:
        start_frame = time_to_frame(zoom.start_s, timeline.fps_num, timeline.fps_den)
        duration = max(1, time_to_frame(zoom.end_s - zoom.start_s,
                                        timeline.fps_num, timeline.fps_den))
        segment = _segment_item_at(timeline, zoom.start_s)
        anchor_x, anchor_y = anchors.get(segment, (0.0, 0.0)) if segment else (0.0, 0.0)
        try:
            for offset, frames, values in _punch_envelope(duration, zoom.depth,
                                                          anchor_x, anchor_y):
                item = clip_ops.add_adjustment_item(timeline, start_frame + offset, frames,
                                                    track=ZOOM_TRACK)
                item.origin = ZOOM_ORIGIN
                clip_ops.set_transform(timeline, item.id, values)
            applied_windows += 1
        except Exception as e:
            logger.debug("Could not add a punch-in at %.1fs: %s", zoom.start_s, e)

    logger.info("Zooms applied: %d segment moves, %d punch-ins",
                applied_segments, applied_windows)
    return applied_segments, applied_windows


# Share of a punch spent easing in, and again easing out; the rest holds.
PUNCH_EASE_SHARE = 0.3
# Frames below which a punch is one straight ease in (no room for a hold).
_PUNCH_MIN_ENVELOPE_FRAMES = 9


def _punch_envelope(duration: int, depth: float, anchor_x: float, anchor_y: float
                    ) -> List[Tuple[int, int, Dict[str, float]]]:
    """(frame offset, frames, transform) pieces for one punch: ease in to the
    face, hold, ease back out to where the shot was.

    A punch used to be a single push-in whose window then simply ended, so the
    frame snapped from 1.25x straight back to 1.0x on the next frame -- and a
    pull-back did the same at its start. Each piece's own move is smoothstep-
    eased by the renderer, so joining them end-to-start at the same scale gives
    one continuous in-hold-out move with no jump at either edge.
    """
    tight = {"scale": 1.0 + depth, "pos_x": anchor_x, "pos_y": anchor_y}
    wide = {"scale": 1.0, "pos_x": 0.0, "pos_y": 0.0}

    def move(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        return {"scale": a["scale"], "scale_end": b["scale"],
                "pos_x": a["pos_x"], "pos_x_end": b["pos_x"],
                "pos_y": a["pos_y"], "pos_y_end": b["pos_y"]}

    if duration < _PUNCH_MIN_ENVELOPE_FRAMES:
        return [(0, duration, move(wide, tight))]
    ease = max(3, int(round(duration * PUNCH_EASE_SHARE)))
    hold = duration - 2 * ease
    pieces = [(0, ease, move(wide, tight))]
    if hold > 0:
        pieces.append((ease, hold, dict(tight)))
    pieces.append((ease + max(0, hold), ease, move(tight, wide)))
    return pieces


def _segment_item_at(timeline: Timeline, time_s: float) -> Optional[str]:
    from timeline.schema import frame_to_time
    for item in timeline.items:
        if item.track != "V1" or item.kind != "media":
            continue
        start = frame_to_time(item.timeline_start_frame, timeline.fps_num, timeline.fps_den)
        end = frame_to_time(item.timeline_end_frame, timeline.fps_num, timeline.fps_den)
        if start <= time_s < end:
            return item.id
    return None
