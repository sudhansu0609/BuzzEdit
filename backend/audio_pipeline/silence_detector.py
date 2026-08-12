import numpy as np
import logging
from typing import List, Dict, Any, Optional
from models import TranscriptSegment

logger = logging.getLogger(__name__)


def detect_silence_segments(
    video_path: str,
    transcript_segments: List[TranscriptSegment],
    min_duration: float = 1.5,
    gap_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    silence_segments = []

    if not transcript_segments:
        return silence_segments

    sorted_segments = sorted(transcript_segments, key=lambda s: s.start)

    for i in range(len(sorted_segments) - 1):
        current_end = sorted_segments[i].end
        next_start = sorted_segments[i + 1].start
        gap = next_start - current_end

        if gap >= min_duration:
            silence_start = current_end + gap_threshold
            silence_end = next_start - gap_threshold
            if silence_end > silence_start:
                silence_segments.append({
                    "start": round(silence_start, 2),
                    "end": round(silence_end, 2),
                    "duration": round(silence_end - silence_start, 2),
                    "confidence": min(1.0, gap / (min_duration * 2)),
                })

    if len(sorted_segments) > 0:
        first_start = sorted_segments[0].start
        if first_start >= min_duration:
            silence_segments.append({
                "start": 0.0,
                "end": round(first_start - gap_threshold, 2),
                "duration": round(first_start - gap_threshold, 2),
                "confidence": 0.9,
            })

    logger.info(f"Detected {len(silence_segments)} silence segments")
    return silence_segments


def detect_audio_silence(
    audio_path: str,
    threshold_db: float = -40.0,
    min_duration: float = 1.5,
) -> List[Dict[str, Any]]:
    import soundfile as sf

    data, samplerate = sf.read(audio_path, dtype="float32")

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    window_size = int(samplerate * 0.1)
    hop_size = window_size // 2

    silence_segments = []
    in_silence = False
    silence_start = 0.0

    for i in range(0, len(data) - window_size, hop_size):
        window = data[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))

        db = 20 * np.log10(rms + 1e-10)
        timestamp = i / samplerate

        if db < threshold_db:
            if not in_silence:
                silence_start = timestamp
                in_silence = True
        else:
            if in_silence:
                duration = timestamp - silence_start
                if duration >= min_duration:
                    silence_segments.append({
                        "start": round(silence_start, 2),
                        "end": round(timestamp, 2),
                        "duration": round(duration, 2),
                        "confidence": min(1.0, (-db) / 60),
                    })
                in_silence = False

    if in_silence:
        total_duration = len(data) / samplerate
        duration = total_duration - silence_start
        if duration >= min_duration:
            silence_segments.append({
                "start": round(silence_start, 2),
                "end": round(total_duration, 2),
                "duration": round(duration, 2),
                "confidence": 0.8,
            })

    logger.info(f"Audio analysis: {len(silence_segments)} silence gaps found")
    return silence_segments
