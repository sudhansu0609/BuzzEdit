import subprocess
from typing import List

def get_nvenc_available() -> bool:
    """Check if h264_nvenc encoder is available in FFmpeg."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

def get_encoder_flags(prefer_nvenc: bool = True) -> List[str]:
    """
    Return hardware-accelerated NVENC flags if available, otherwise CPU x264 fallback flags.
    NVENC quality flags use `-cq:v 23 -preset p4` (NOT the buggy 51 - CRF calculation).
    """
    if prefer_nvenc and get_nvenc_available():
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq:v", "23",
            "-b:v", "0",
            "-c:a", "aac",
            "-b:a", "192k"
        ]
    else:
        return [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k"
        ]
