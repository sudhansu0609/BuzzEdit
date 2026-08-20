"""Zoom and pan estimation.

Each shot is sampled at roughly one-second steps and a similarity transform
(scale + rotation + translation) is fitted between consecutive samples with
RANSAC, then composed across the shot. That gives the total move directly.

The first version of this integrated a per-frame optical-flow divergence rate
over the shot, which does not work: the per-frame signal is dominated by noise,
and multiplying a noisy mean by the frame count amplifies it instead of cancelling
it. A completely static clip measured 0.156 of false zoom that way. Fitting a
transform per step and composing keeps every measurement a direct one, and RANSAC
throws out the moving subjects that a whole-frame average would fold in.

An honest limit, stated once: this measures motion *in the reference picture*. It
cannot separate a physical camera push-in from a post-production punch-in, because
on screen they are the same thing. For copying editing style that rarely matters —
what carries over is "shots here drift in about 12% over a couple of seconds".
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Below these a "move" is tracking jitter or codec noise, not an intentional one.
MIN_ZOOM_RATIO = 0.04
MIN_PAN_FRACTION = 0.05

_MAX_FEATURES = 300
_MIN_INLIERS = 12
_STEP_SECONDS = 1.0


@dataclass
class MotionAnalysis:
    zoom_share: float = 0.0        # fraction of shots with a deliberate zoom
    pan_share: float = 0.0
    mean_zoom_ratio: float = 0.0   # e.g. 0.15 -> pushes in to 1.15x
    mean_pan_fraction: float = 0.0 # as a fraction of frame width
    mean_move_seconds: float = 0.0
    samples: int = 0


def estimate_step(previous: np.ndarray, current: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """(scale, dx, dy) between two frames, or None when there is nothing to track.

    Returning None rather than an identity guess matters: a flat, featureless
    shot should report "unknown", which the caller treats as no move, instead of
    contributing a fabricated zero to the average.
    """
    import cv2

    corners = cv2.goodFeaturesToTrack(
        previous, maxCorners=_MAX_FEATURES, qualityLevel=0.01,
        minDistance=8, blockSize=7)
    if corners is None or len(corners) < _MIN_INLIERS:
        return None

    tracked, status, _err = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    if tracked is None or status is None:
        return None

    keep = status.reshape(-1).astype(bool)
    source, destination = corners[keep], tracked[keep]
    if len(source) < _MIN_INLIERS:
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
        return None

    # A similarity matrix stores scale folded into the rotation block.
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not np.isfinite(scale) or scale <= 0.2 or scale >= 5.0:
        return None
    return scale, float(matrix[0, 2]), float(matrix[1, 2])


def measure_shot(frames: np.ndarray, start: int, end: int, step: int) -> Optional[Tuple[float, float]]:
    """(total_scale, total_translation_px) across one shot, or None if untrackable."""
    scale = 1.0
    dx = dy = 0.0
    measured = 0

    # Whole steps, plus a final partial one so the tail of the shot is included.
    # Leaving it out silently under-reports every move by up to a full step —
    # a 0.30 push measured as 0.255.
    positions = list(range(start, end - 1, step))
    pairs = [(a, min(a + step, end - 1)) for a in positions]
    for first, second in pairs:
        if second <= first:
            continue
        result = estimate_step(frames[first], frames[second])
        if result is None:
            continue
        step_scale, step_dx, step_dy = result
        scale *= step_scale
        dx += step_dx
        dy += step_dy
        measured += 1

    if measured == 0:
        return None
    return scale, float(np.hypot(dx, dy))


def analyse_motion(
    frames: np.ndarray,
    fps: float,
    shot_bounds: Optional[List[Tuple[float, float]]] = None,
) -> MotionAnalysis:
    """Measure how often, and how far, shots zoom and pan.

    `shot_bounds` keeps each measurement inside one shot — tracking across a cut
    would read the cut itself as an enormous camera move.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        logger.warning("OpenCV unavailable; skipping motion analysis")
        return MotionAnalysis()

    if frames.shape[0] < 3:
        return MotionAnalysis()

    total = frames.shape[0]
    width = frames.shape[2]
    duration = total / fps
    if not shot_bounds:
        shot_bounds = [(0.0, duration)]

    step = max(1, int(round(_STEP_SECONDS * fps)))
    zoom_ratios: List[float] = []
    pan_fractions: List[float] = []
    move_durations: List[float] = []
    shots_measured = 0

    for start_sec, end_sec in shot_bounds:
        # Drop a frame either side so a transition's own blended frames stay out.
        start = int(round(start_sec * fps)) + 1
        end = min(total, int(round(end_sec * fps))) - 1
        if end - start < step + 1:
            continue

        result = measure_shot(frames, start, end, step)
        if result is None:
            continue
        shots_measured += 1

        scale, translation = result
        shot_seconds = (end - start) / fps
        zoom_ratio = abs(scale - 1.0)
        pan_fraction = translation / max(1.0, width)

        if zoom_ratio >= MIN_ZOOM_RATIO:
            zoom_ratios.append(zoom_ratio)
            move_durations.append(shot_seconds)
        # A zoom leaves a little residual translation in the fit, so only count a
        # pan that is bigger than the zoom could account for. Otherwise every
        # centred push-in would also be reported as a pan.
        if pan_fraction >= MIN_PAN_FRACTION and pan_fraction > zoom_ratio * 0.8:
            pan_fractions.append(pan_fraction)

    if shots_measured == 0:
        return MotionAnalysis()

    return MotionAnalysis(
        zoom_share=round(len(zoom_ratios) / shots_measured, 3),
        pan_share=round(len(pan_fractions) / shots_measured, 3),
        mean_zoom_ratio=round(float(np.mean(zoom_ratios)), 3) if zoom_ratios else 0.0,
        mean_pan_fraction=round(float(np.mean(pan_fractions)), 3) if pan_fractions else 0.0,
        mean_move_seconds=round(float(np.mean(move_durations)), 2) if move_durations else 0.0,
        samples=shots_measured,
    )
