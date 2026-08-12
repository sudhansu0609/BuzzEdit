import soundfile as sf
import numpy as np
from typing import List, Tuple

def detect_speech_silence_intervals(
    audio_path: str,
    min_silence_duration: float = 0.8,
    silence_threshold_db: float = -35.0
) -> List[Tuple[float, float]]:
    """
    Detect silent intervals (in seconds) from an audio file.
    Returns list of (start_sec, end_sec) tuples representing pauses/silences.
    """
    try:
        data, sample_rate = sf.read(audio_path)
    except Exception:
        return []

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    chunk_size = int(sample_rate * 0.05)  # 50ms chunks
    if chunk_size == 0:
        return []

    num_chunks = len(data) // chunk_size
    silence_intervals: List[Tuple[float, float]] = []
    
    in_silence = False
    silence_start = 0.0

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)

        time_sec = (i * chunk_size) / sample_rate

        if db < silence_threshold_db:
            if not in_silence:
                in_silence = True
                silence_start = time_sec
        else:
            if in_silence:
                in_silence = False
                duration = time_sec - silence_start
                if duration >= min_silence_duration:
                    silence_intervals.append((round(silence_start, 3), round(time_sec, 3)))

    if in_silence:
        end_sec = len(data) / sample_rate
        if end_sec - silence_start >= min_silence_duration:
            silence_intervals.append((round(silence_start, 3), round(end_sec, 3)))

    return silence_intervals
