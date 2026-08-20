"""Musical rhythm and how tightly the edit is cut to it.

Tempo comes from the autocorrelation of a spectral-flux onset envelope: flux rises
wherever new energy appears (a drum hit, a downbeat), and a track with a steady
pulse makes that envelope repeat at the beat period. The lag with the strongest
correlation inside a plausible tempo range is the beat.

The number that actually matters for style is `alignment`: what fraction of the
edit's cuts land on a beat. An editor cutting to music sits near 1.0; one cutting
to speech sits near chance. That single figure separates two very different
editing grammars, and it is cheap to measure.

Implemented on numpy alone rather than pulling in librosa — this is a couple of
FFTs and an autocorrelation, and the dependency would be far heavier than the code.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

MIN_BPM = 60.0
MAX_BPM = 190.0
_HOP = 512
_WINDOW = 1024


@dataclass
class RhythmAnalysis:
    bpm: float = 0.0
    alignment: float = 0.0     # 0..1 share of cuts landing on a beat
    confidence: float = 0.0    # how periodic the onset envelope is
    has_audio: bool = False


def onset_envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral flux: summed positive change in magnitude spectrum per hop."""
    if samples.size < _WINDOW * 4:
        return np.zeros(0, dtype=np.float32)

    frame_count = 1 + (samples.size - _WINDOW) // _HOP
    if frame_count < 8:
        return np.zeros(0, dtype=np.float32)

    # Strided view avoids materialising a copy of every overlapping window.
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    strides = (samples.strides[0] * _HOP, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(frame_count, _WINDOW), strides=strides)
    windowed = frames * np.hanning(_WINDOW).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))

    flux = np.diff(spectrum, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux.astype(np.float32)


def estimate_tempo(envelope: np.ndarray, sample_rate: int) -> tuple:
    """(bpm, confidence) from the autocorrelation peak of the onset envelope."""
    if envelope.size < 16:
        return 0.0, 0.0

    centred = envelope - envelope.mean()
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:]
    if correlation[0] <= 0:
        return 0.0, 0.0
    correlation = correlation / correlation[0]

    hop_rate = sample_rate / _HOP                      # envelope frames per second
    min_lag = max(1, int(round(hop_rate * 60.0 / MAX_BPM)))
    max_lag = min(correlation.size - 1, int(round(hop_rate * 60.0 / MIN_BPM)))
    if max_lag <= min_lag:
        return 0.0, 0.0

    window = correlation[min_lag:max_lag + 1]
    best = int(np.argmax(window)) + min_lag

    # Autocorrelation peaks just as hard at two beats as at one, so the raw
    # maximum lands on half the real tempo about as often as not — a 120 BPM
    # track reads as 60. Score the octave candidates against a preference for
    # ordinary dance/vlog tempi and take the best of them.
    def prior(bpm: float) -> float:
        return float(np.exp(-0.5 * (np.log(bpm / 120.0) / 0.45) ** 2))

    best_lag, best_score = best, 0.0
    for candidate in {best, best // 2, best * 2, best // 3, best * 3}:
        if not (min_lag <= candidate <= max_lag):
            continue
        bpm = 60.0 * hop_rate / candidate
        score = float(correlation[candidate]) * prior(bpm)
        if score > best_score:
            best_lag, best_score = candidate, score

    # Sub-lag precision by fitting a parabola through the peak and its neighbours.
    # At this hop rate one lag step is several BPM, and that error compounds into
    # the alignment measure, so the refinement is worth the three lines.
    refined = float(best_lag)
    if 0 < best_lag < correlation.size - 1:
        left, centre, right = (correlation[best_lag - 1],
                               correlation[best_lag],
                               correlation[best_lag + 1])
        denominator = left - 2.0 * centre + right
        if abs(denominator) > 1e-9:
            refined = best_lag + 0.5 * (left - right) / denominator

    confidence = float(np.clip(correlation[best_lag], 0.0, 1.0))
    return round(60.0 * hop_rate / max(1e-6, refined), 1), round(confidence, 3)


def beat_alignment(cut_times: List[float], bpm: float, tolerance: float = 0.16) -> float:
    """Share of consecutive cut *intervals* that span a whole number of beats.

    Measured on intervals rather than against an absolute beat grid on purpose.
    A grid needs both the tempo and its phase to be right, and a 2% tempo error —
    well within what any estimator gives you — walks the grid a full beat out of
    step over half a minute, so a perfectly beat-cut montage scores near zero.
    Intervals are local: each one is judged on its own, and small tempo error
    stays small. `tolerance` is a fraction of one beat.
    """
    if bpm <= 0 or len(cut_times) < 2:
        return 0.0
    period = 60.0 / bpm
    intervals = np.diff(np.asarray(sorted(cut_times), dtype=np.float64))
    intervals = intervals[intervals > period * 0.4]      # ignore sub-beat flurries
    if intervals.size == 0:
        return 0.0

    beats = intervals / period
    distance = np.abs(beats - np.round(beats))
    return round(float((distance <= tolerance).mean()), 3)


def analyse_rhythm(samples: np.ndarray, sample_rate: int, cut_times: List[float]) -> RhythmAnalysis:
    if samples.size == 0:
        return RhythmAnalysis()
    envelope = onset_envelope(samples, sample_rate)
    bpm, confidence = estimate_tempo(envelope, sample_rate)
    return RhythmAnalysis(
        bpm=bpm,
        alignment=beat_alignment(cut_times, bpm),
        confidence=confidence,
        has_audio=True,
    )
