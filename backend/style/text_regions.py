"""Where the reference puts its captions, and how big they are.

No OCR: reading the words back is not the goal and would add a heavy dependency
for no benefit — the copy comes from the user's own transcript. What transfers is
*geometry and rhythm*: which band of the frame the text sits in, how large it is
relative to frame height, whether it has a box behind it, and how much of the
runtime carries text at all.

Detection leans on the one thing burned-in captions reliably are: rows with far
more vertical edges than the picture around them, appearing in the same place for
seconds at a time. Both halves matter — busy footage has edge-rich rows too, but
they wander, whereas a caption band stays put.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CaptionAnalysis:
    present: bool = False
    pos_y: float = 0.0          # -1..1 canvas coordinate, matching TextStyle
    size_fraction: float = 0.0  # band height / frame height
    coverage: float = 0.0       # share of sampled frames carrying text
    boxed: bool = False
    confidence: float = 0.0


_EDGE_LEVEL = 24     # luma step that counts as a stroke edge


def _edge_rows(frames: np.ndarray) -> np.ndarray:
    """Per-row count of strong horizontal transitions, normalised, shape (N, H).

    Counting edges rather than averaging gradient magnitude is what separates
    text from graphics. A line of letters crosses dozens of light/dark boundaries
    across its width; a solid rectangle — however bright its border — crosses two.
    An energy average cannot tell those apart and reports any hard-edged shape as
    a caption.
    """
    gradient = np.abs(np.diff(frames.astype(np.int16), axis=2))
    crossings = (gradient > _EDGE_LEVEL).sum(axis=2).astype(np.float32)
    return crossings / max(1, frames.shape[2])


def analyse_captions(
    frames: np.ndarray,
    min_coverage: float = 0.15,
    min_band_fraction: float = 0.035,
    max_band_fraction: float = 0.18,
) -> CaptionAnalysis:
    """Find a persistent text band in a stack of greyscale frames (N, H, W)."""
    if frames.ndim != 3 or frames.shape[0] < 4:
        return CaptionAnalysis()

    count, height, _width = frames.shape
    rows = _edge_rows(frames)

    # A row is "texty" when it carries much more edge energy than that frame's
    # own average, which keeps the test independent of overall busyness.
    per_frame_mean = rows.mean(axis=1, keepdims=True)
    per_frame_std = rows.std(axis=1, keepdims=True) + 1e-6
    texty = (rows - per_frame_mean) / per_frame_std > 1.6

    # How often each row is texty across the whole video. A caption band shows up
    # as a contiguous run of rows with a high score; moving subjects do not.
    row_score = texty.mean(axis=0)
    if row_score.max() < min_coverage:
        return CaptionAnalysis()

    threshold = max(min_coverage, row_score.max() * 0.5)
    active = row_score >= threshold

    best_start = best_end = None
    best_weight = 0.0
    index = 0
    while index < height:
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < height and active[index]:
            index += 1
        weight = float(row_score[run_start:index].sum())
        if weight > best_weight:
            best_weight, best_start, best_end = weight, run_start, index

    # A caption band has to be thick enough to read but is never a large slab of
    # the frame. Anything outside that range is picture texture, not type: one or
    # two edgy rows below, a quarter of the screen above.
    if best_start is None:
        return CaptionAnalysis()
    thickness = best_end - best_start
    if not (max(2, int(height * min_band_fraction)) <= thickness <= height * max_band_fraction):
        return CaptionAnalysis()

    # Text occupies part of the width, not all of it. Dense picture texture — noise,
    # foliage, a checkerboard — produces edges right across every row, so a band
    # whose edges span the full frame is scenery, not a caption.
    band = frames[:, best_start:best_end, :]
    band_gradient = np.abs(np.diff(band.astype(np.int16), axis=2)) > _EDGE_LEVEL
    column_hits = band_gradient.any(axis=1).mean(axis=0)      # per column, over time
    extent = float((column_hits > 0.25).mean())
    if not 0.10 <= extent <= 0.88:
        return CaptionAnalysis()

    band_centre = (best_start + best_end) / 2.0
    band_height = best_end - best_start
    coverage = float(row_score[best_start:best_end].max())

    # A box reads as unusually low pixel variance in the rows just outside the
    # glyphs — a flat plate rather than picture.
    margin = max(1, band_height // 2)
    above = slice(max(0, best_start - margin), best_start)
    below = slice(best_end, min(height, best_end + margin))
    band_variance = float(frames[:, best_start:best_end, :].std())
    surround = np.concatenate([
        frames[:, above, :].reshape(count, -1),
        frames[:, below, :].reshape(count, -1),
    ], axis=1) if (best_start > 0 or best_end < height) else frames[:, best_start:best_end, :].reshape(count, -1)
    surround_variance = float(surround.std()) if surround.size else band_variance
    boxed = surround_variance < band_variance * 0.55

    return CaptionAnalysis(
        present=True,
        pos_y=round((band_centre / height) * 2.0 - 1.0, 3),
        size_fraction=round(band_height / height, 3),
        coverage=round(coverage, 3),
        boxed=boxed,
        confidence=round(min(1.0, coverage * 1.5), 3),
    )
