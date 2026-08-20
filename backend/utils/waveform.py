"""Audio waveform peak extraction for the timeline UI.

Decodes a media file's audio to low-rate mono PCM via ffmpeg and reduces it to a
compact array of normalized peaks (one value per time bucket). Results are cached
to a sidecar JSON so we only compute once per source.
"""

import json
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from config import FFMPEG_BIN, TEMP_DIR

logger = logging.getLogger("waveform")

_SAMPLE_RATE = 8000          # plenty for a visual waveform
_POINTS_PER_SECOND = 60      # peak buckets per second of audio


def _cache_path(media_path: str) -> Path:
    key = hashlib.md5(media_path.encode("utf-8")).hexdigest()[:16]
    return TEMP_DIR / f"wave_{key}.json"


def compute_waveform(media_path: str, points_per_second: int = _POINTS_PER_SECOND) -> Optional[dict]:
    """Return {peaks: [0..1], points_per_second, duration} for a media file, or
    None if it has no decodable audio. Cached to disk."""
    path = Path(media_path)
    if not path.exists():
        return None

    cache = _cache_path(media_path)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("points_per_second") == points_per_second:
                return data
        except Exception:
            pass

    cmd = [
        FFMPEG_BIN, "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(_SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except Exception as e:
        logger.warning(f"waveform ffmpeg failed for {media_path}: {e}")
        return None

    if proc.returncode != 0 or not proc.stdout:
        return None  # no audio stream, or decode error

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        return None

    bucket = max(1, _SAMPLE_RATE // points_per_second)
    n_buckets = int(np.ceil(samples.size / bucket))
    # Pad to a whole number of buckets, then peak (max abs) per bucket.
    padded = np.zeros(n_buckets * bucket, dtype=np.float32)
    padded[: samples.size] = np.abs(samples)
    peaks = padded.reshape(n_buckets, bucket).max(axis=1)

    peak_max = float(peaks.max()) or 1.0
    peaks = (peaks / peak_max).clip(0.0, 1.0)

    data = {
        "peaks": [round(float(p), 3) for p in peaks],
        "points_per_second": points_per_second,
        "duration": samples.size / _SAMPLE_RATE,
    }
    try:
        cache.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data
