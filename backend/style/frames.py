"""Fast frame sampling for style analysis.

Every analyser needs the same thing: a few thousand small frames spread across a
video. Seeking with OpenCV frame-by-frame is punishingly slow on long files, so
this decodes once through ffmpeg into a raw pipe at a reduced size and frame rate
and hands back a single numpy array.

Sizes are deliberately tiny. Cut detection works fine at 64x36, and optical flow
at 160x90 — analysing a 20-minute reference should cost seconds, not minutes.
"""

import logging
import subprocess
from typing import Optional, Tuple

import numpy as np

from config import FFMPEG_BIN
from utils.ffmpeg_utils import get_video_info

logger = logging.getLogger(__name__)


class FrameSampleError(RuntimeError):
    """Raised when a video cannot be decoded for analysis."""


def probe_duration(path: str) -> float:
    try:
        return float(get_video_info(path).get("duration") or 0.0)
    except Exception as exc:
        raise FrameSampleError(f"Could not read {path}: {exc}") from exc


def sample(
    path: str,
    width: int,
    height: int,
    fps: float,
    gray: bool = True,
    max_frames: int = 6000,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Decode a video into a small array of frames.

    Returns (frames, effective_fps) where frames is (N, H, W) for gray or
    (N, H, W, 3) for colour. When the requested rate would exceed `max_frames`
    the rate is lowered rather than truncating the video, so the sample still
    covers the whole edit — a profile built from only the first two minutes
    would describe the intro, not the style.
    """
    if duration is None:
        duration = probe_duration(path)
    if duration <= 0:
        raise FrameSampleError(f"{path} reports no duration")

    effective_fps = fps
    if duration * fps > max_frames:
        effective_fps = max(0.5, max_frames / duration)

    pix_fmt = "gray" if gray else "rgb24"
    channels = 1 if gray else 3
    frame_bytes = width * height * channels

    cmd = [
        FFMPEG_BIN, "-v", "error", "-i", path,
        "-vf", f"fps={effective_fps:.4f},scale={width}:{height}",
        "-pix_fmt", pix_fmt, "-f", "rawvideo", "-",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=600)
    except Exception as exc:
        raise FrameSampleError(f"ffmpeg failed sampling {path}: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="ignore")[-300:]
        raise FrameSampleError(f"ffmpeg could not decode {path}: {detail}")

    count = len(proc.stdout) // frame_bytes
    if count < 2:
        raise FrameSampleError(f"{path} yielded too few frames to analyse")

    buffer = np.frombuffer(proc.stdout[: count * frame_bytes], dtype=np.uint8)
    shape = (count, height, width) if gray else (count, height, width, 3)
    return buffer.reshape(shape), effective_fps


def sample_audio(path: str, sample_rate: int = 22050, max_seconds: float = 1800.0) -> np.ndarray:
    """Mono float32 PCM for rhythm analysis, or an empty array if there is no audio."""
    cmd = [
        FFMPEG_BIN, "-v", "error", "-i", path, "-vn",
        "-ac", "1", "-ar", str(sample_rate), "-t", f"{max_seconds:.0f}",
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    except Exception as exc:
        logger.warning("Audio sampling failed for %s: %s", path, exc)
        return np.zeros(0, dtype=np.float32)

    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32)
