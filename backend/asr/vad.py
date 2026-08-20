"""Voice activity detection over an audio file.

Speech/silence is the one signal in this pipeline that is genuinely reliable: it
does not depend on the language being recognised, on the transcript being right,
or on word timestamps being sane. Everything the auto-edit does is anchored to it.

The threshold adapts to the recording. The previous version compared against a
fixed -35 dB, which is meaningless across different microphones and rooms — a
quiet room-tone recording sits entirely below it (everything reads as silence)
and a noisy one entirely above (nothing does). Estimating this file's own noise
floor and speech peak, then thresholding between them, works on both.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("vad")

_FRAME_SECONDS = 0.02          # 20 ms analysis frames


@dataclass
class SpeechMap:
    """Where speech is in a file, and the levels it was decided from."""
    speech: List[Tuple[float, float]]      # (start, end) seconds
    duration: float
    noise_floor_db: float
    speech_peak_db: float
    threshold_db: float
    # 20ms loudness envelope the decisions came from. Kept so downstream cut
    # placement can find the quietest moment near a boundary.
    frame_db: Optional[List[int]] = None
    frame_rate: float = 1.0 / _FRAME_SECONDS

    def envelope(self) -> Optional[dict]:
        if not self.frame_db:
            return None
        return {"rate": self.frame_rate, "db": self.frame_db}

    @property
    def speech_seconds(self) -> float:
        return sum(end - start for start, end in self.speech)

    def silences(self, min_duration: float = 0.0) -> List[Tuple[float, float]]:
        """Gaps between speech regions, including any lead-in and tail."""
        gaps: List[Tuple[float, float]] = []
        previous = 0.0
        for start, end in self.speech:
            if start - previous >= min_duration:
                gaps.append((previous, start))
            previous = end
        if self.duration - previous >= min_duration:
            gaps.append((previous, self.duration))
        return gaps

    def speech_fraction(self, start: float, end: float) -> float:
        """How much of a span is speech, 0..1. Used to judge a word's timing."""
        if end <= start:
            return 0.0
        covered = 0.0
        for s_start, s_end in self.speech:
            overlap = min(end, s_end) - max(start, s_start)
            if overlap > 0:
                covered += overlap
        return min(1.0, covered / (end - start))

    def clamp_to_speech(self, start: float, end: float) -> Optional[Tuple[float, float]]:
        """Tighten a span to the speech inside it, or None if it holds none."""
        overlapping = [(max(start, s), min(end, e)) for s, e in self.speech
                       if min(end, e) > max(start, s)]
        if not overlapping:
            return None
        return overlapping[0][0], overlapping[-1][1]


def _load_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def _frame_db(data: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = max(1, int(sample_rate * _FRAME_SECONDS))
    count = len(data) // hop
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = data[: count * hop].reshape(count, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def _smooth(mask: np.ndarray, frames: int) -> np.ndarray:
    """Close gaps shorter than `frames` in a boolean mask."""
    if frames <= 0 or mask.size == 0:
        return mask
    out = mask.copy()
    index = 0
    while index < out.size:
        if out[index]:
            index += 1
            continue
        run_start = index
        while index < out.size and not out[index]:
            index += 1
        # Only close an interior gap; leading and trailing silence stay silence.
        if (index - run_start) < frames and run_start > 0 and index < out.size:
            out[run_start:index] = True
    return out


def analyse_speech(
    audio_path: str,
    min_speech_duration: float = 0.12,
    merge_gap: float = 0.12,
    sensitivity: float = 0.22,
) -> Optional[SpeechMap]:
    """Map the speech in an audio file.

    `sensitivity` positions the threshold between this file's noise floor and its
    speech peak: lower keeps more marginal audio as speech (safer, cuts less),
    higher is more aggressive.
    """
    try:
        data, sample_rate = _load_mono(audio_path)
    except Exception as exc:
        logger.warning("VAD could not read %s: %s", audio_path, exc)
        return None

    db = _frame_db(data, sample_rate)
    if db.size == 0:
        return None

    duration = len(data) / sample_rate
    # Percentiles rather than min/max: a single click or dropout should not
    # define the range the threshold is placed in.
    noise_floor = float(np.percentile(db, 10))
    speech_peak = float(np.percentile(db, 95))
    span = speech_peak - noise_floor

    if span < 6.0:
        # Almost no dynamic range — either pure silence or a constant tone.
        # Treat the whole file as speech rather than cutting it to pieces.
        logger.info("VAD: %s has no usable dynamic range (%.1f dB); treating as all speech",
                    audio_path, span)
        return SpeechMap([(0.0, duration)], duration, noise_floor, speech_peak,
                         noise_floor, [int(round(v)) for v in db])

    threshold = noise_floor + span * sensitivity
    mask = db > threshold
    mask = _smooth(mask, int(round(merge_gap / _FRAME_SECONDS)))

    regions: List[Tuple[float, float]] = []
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < mask.size and mask[index]:
            index += 1
        start_sec = start * _FRAME_SECONDS
        end_sec = min(duration, index * _FRAME_SECONDS)
        if end_sec - start_sec >= min_speech_duration:
            regions.append((round(start_sec, 3), round(end_sec, 3)))

    logger.info("VAD %s: %d speech regions, %.1f%% speech (floor %.1f dB, peak %.1f dB)",
                audio_path, len(regions), 100.0 * sum(e - s for s, e in regions) / max(duration, 1e-6),
                noise_floor, speech_peak)
    return SpeechMap(regions, duration, noise_floor, speech_peak, threshold,
                     [int(round(v)) for v in db])


def detect_speech_silence_intervals(
    audio_path: str,
    min_silence_duration: float = 0.8,
    silence_threshold_db: float = -35.0,
) -> List[Tuple[float, float]]:
    """Silent intervals in seconds. Kept for the existing analysis endpoints.

    `silence_threshold_db` is ignored: the threshold is now derived from the
    file's own noise floor, which is what makes this work across recordings.
    """
    speech_map = analyse_speech(audio_path)
    if speech_map is None:
        return []
    return [(round(s, 3), round(e, 3))
            for s, e in speech_map.silences(min_duration=min_silence_duration)]
