"""Colour look extraction and matching.

What this does: measure a reference's exposure, contrast, saturation and colour
cast, measure the same on the project's own footage, and solve for the ColorGrade
that moves one to the other.

What this does *not* do, and cannot: recover the original grade. A finished video
only shows the result, and the result depends as much on what was filmed as on how
it was graded. Matching statistics gets the mood right — brightness, punch,
warmth, how saturated it feels — but it will not reproduce split toning or a
bespoke curve. Fitting the existing `eq`-based ColorGrade rather than baking a LUT
is a deliberate trade: every value stays visible and editable in the Inspector
afterwards instead of being frozen into an opaque file.
"""

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from timeline.schema import ColorGrade


@dataclass
class ColorStats:
    mean: float                 # overall luma, 0..1
    std: float                  # luma spread -> contrast
    saturation: float           # mean chroma distance from grey, 0..1
    channel_mean: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def measure(frames: np.ndarray) -> ColorStats:
    """Colour statistics for a stack of RGB frames, shape (N, H, W, 3)."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("measure() expects RGB frames of shape (N, H, W, 3)")

    pixels = frames.reshape(-1, 3).astype(np.float32) / 255.0
    # Luma weights so a change in green counts more than the same change in blue,
    # matching how the eye reads brightness.
    luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    channel_mean = pixels.mean(axis=0)
    saturation = float(np.abs(pixels - luma[:, None]).mean())

    return ColorStats(
        mean=float(luma.mean()),
        std=float(luma.std()),
        saturation=saturation,
        channel_mean={"r": float(channel_mean[0]),
                      "g": float(channel_mean[1]),
                      "b": float(channel_mean[2])},
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fit_grade(source: ColorStats, target: ColorStats) -> ColorGrade:
    """The ColorGrade that moves `source` footage towards the `target` look."""
    # Contrast is the ratio of spreads: a reference with twice the luma spread
    # wants roughly twice the contrast.
    contrast = _clamp(target.std / max(source.std, 1e-3), 0.4, 2.2)

    # eq applies contrast about mid-grey, so predict where the mean lands after
    # that stretch and let brightness make up the remaining difference.
    predicted_mean = 0.5 + (source.mean - 0.5) * contrast
    brightness = _clamp(target.mean - predicted_mean, -0.4, 0.4)

    saturation = _clamp(target.saturation / max(source.saturation, 1e-3), 0.0, 2.5)

    # Colour cast: compare how far each of red and blue sits from the frame's own
    # average, so a simply brighter reference does not read as "warm".
    def cast(stats: ColorStats) -> float:
        mean = (stats.channel_mean["r"] + stats.channel_mean["g"] + stats.channel_mean["b"]) / 3.0 or 1e-3
        return (stats.channel_mean["r"] - stats.channel_mean["b"]) / mean

    temperature = _clamp((cast(target) - cast(source)) * 1.6, -1.0, 1.0)

    return ColorGrade(
        preset=None,
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        saturation=round(saturation, 3),
        temperature=round(temperature, 3),
    )


def describe(stats: ColorStats) -> str:
    """A short human sentence for the UI, so the numbers mean something."""
    exposure = "dark" if stats.mean < 0.35 else "bright" if stats.mean > 0.6 else "mid-exposed"
    punch = "flat" if stats.std < 0.15 else "punchy" if stats.std > 0.26 else "moderate contrast"
    cast = (stats.channel_mean["r"] - stats.channel_mean["b"])
    tone = "warm" if cast > 0.02 else "cool" if cast < -0.02 else "neutral"
    vividness = "muted" if stats.saturation < 0.05 else "saturated" if stats.saturation > 0.12 else "natural"
    return f"{exposure}, {punch}, {tone}, {vividness}"
