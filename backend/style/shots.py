"""Shot boundary detection and transition classification.

Works off a single luma difference curve. The three things we care about have
distinct signatures in that curve:

* a **hard cut** is one isolated spike — the picture changes completely between
  two adjacent sampled frames and is stable either side;
* a **dissolve** is a broad plateau — the frames keep changing moderately for as
  long as the mix lasts, so the difference stays elevated over many frames;
* a **fade** is a dissolve whose luma also collapses towards black (or blows out
  to white) at its midpoint.

The threshold adapts to the footage rather than being a fixed constant: a locked-off
talking head and a handheld action reel have wildly different baseline motion, and
a constant threshold would find hundreds of phantom cuts in one and none in the other.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Boundary:
    time: float
    kind: str          # "cut" | "dissolve" | "fade"
    duration: float    # 0 for a hard cut
    strength: float


@dataclass
class ShotAnalysis:
    boundaries: List[Boundary] = field(default_factory=list)
    shot_lengths: List[float] = field(default_factory=list)
    duration: float = 0.0

    @property
    def cuts_per_minute(self) -> float:
        if self.duration <= 0:
            return 0.0
        return len(self.boundaries) / (self.duration / 60.0)


def difference_curve(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive sampled frames.

    Colour matters here. Cutting between two shots of similar brightness — a
    common, unremarkable edit — leaves a luma-only curve almost flat, so a
    greyscale detector walks straight past it. Differencing all three channels
    catches those.
    """
    if frames.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    return np.abs(np.diff(flat, axis=0)).mean(axis=1)


# Below this, a "peak" is compression noise on an essentially static picture.
_NOISE_FLOOR = 3.0
# How far above its neighbourhood a frame must jump to count as an edit.
_LOCAL_RATIO = 3.2


def local_thresholds(diff: np.ndarray, fps: float, window_seconds: float = 4.0) -> np.ndarray:
    """A per-frame threshold that tracks the footage's own motion.

    One global threshold cannot serve a video that is calm for a minute and then
    handheld for the next: set it for the calm half and the busy half turns into
    phantom cuts; set it for the busy half and the calm half detects nothing.
    Comparing each frame against the median of the seconds around it sidesteps
    that, and the median ignores the cut spikes themselves.
    """
    if diff.size == 0:
        return diff

    block = max(4, int(round(window_seconds * fps)))
    block_count = max(1, int(np.ceil(diff.size / block)))

    # Block medians, each widened by its neighbours so a cut sitting at a block
    # edge is still judged against calm material.
    medians = np.array([
        float(np.median(diff[i * block:(i + 1) * block])) for i in range(block_count)
    ], dtype=np.float32)
    widened = medians.copy()
    for i in range(block_count):
        low, high = max(0, i - 1), min(block_count, i + 2)
        widened[i] = float(np.median(medians[low:high]))

    per_frame = np.repeat(widened, block)[: diff.size]
    return np.maximum(_NOISE_FLOOR, per_frame * _LOCAL_RATIO)


def detect_boundaries(
    frames: np.ndarray,
    fps: float,
    duration: float,
    max_transition_sec: float = 1.6,
) -> ShotAnalysis:
    """Find shot boundaries and label each as a cut, dissolve or fade."""
    diff = difference_curve(frames)
    analysis = ShotAnalysis(duration=duration)
    if diff.size == 0:
        return analysis

    threshold = local_thresholds(diff, fps)
    luma = frames.reshape(frames.shape[0], -1).astype(np.float32).mean(axis=1)
    max_span = max(1, int(round(max_transition_sec * fps)))

    above = diff > threshold
    index = 0
    while index < above.size:
        if not above[index]:
            index += 1
            continue

        run_start = index
        while index < above.size and above[index]:
            index += 1
        run_end = index                      # exclusive
        run = diff[run_start:run_end]
        span = run_end - run_start

        # A run longer than a realistic transition is just sustained motion
        # (a whip pan, a busy crowd) — not an edit point.
        if span > max_span:
            continue

        centre = run_start + int(np.argmax(run))
        time = centre / fps
        strength = float(run.max())

        if span <= 1:
            kind, transition_duration = "cut", 0.0
        else:
            transition_duration = span / fps
            window = luma[run_start:run_end + 1]
            # Through-black or through-white shows up as an extreme in the middle
            # of the mix that neither end reaches.
            edge = max(window[0], window[-1])
            kind = "fade" if (window.min() < 18.0 and window.min() < edge * 0.35) else "dissolve"
        analysis.boundaries.append(Boundary(time, kind, transition_duration, strength))

    previous = 0.0
    for boundary in analysis.boundaries:
        analysis.shot_lengths.append(max(0.0, boundary.time - previous))
        previous = boundary.time
    analysis.shot_lengths.append(max(0.0, duration - previous))
    analysis.shot_lengths = [s for s in analysis.shot_lengths if s > 0.05]
    return analysis


def transition_mix(analysis: ShotAnalysis) -> dict:
    """Counts and mean durations per transition kind."""
    mix = {}
    for kind in ("cut", "dissolve", "fade"):
        matching = [b for b in analysis.boundaries if b.kind == kind]
        mix[kind] = {
            "count": len(matching),
            "mean_duration": round(float(np.mean([b.duration for b in matching])), 3) if matching else 0.0,
        }
    total = len(analysis.boundaries) or 1
    for kind in mix:
        mix[kind]["share"] = round(mix[kind]["count"] / total, 3)
    return mix
