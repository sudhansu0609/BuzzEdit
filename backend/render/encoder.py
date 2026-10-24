import subprocess
from typing import List
from utils.proc import NO_WINDOW

def get_nvenc_available() -> bool:
    """Check if h264_nvenc encoder is available in FFmpeg."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            creationflags=NO_WINDOW,
        )
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

def get_encoder_flags(prefer_nvenc: bool = True, profile: str = "final") -> List[str]:
    """
    Return hardware-accelerated NVENC flags if available, otherwise CPU x264 fallback flags.
    NVENC quality flags use `-cq:v 23 -preset p4` (NOT the buggy 51 - CRF calculation).

    `profile="draft"` is the FX preview's: the fastest NVENC preset at a low
    quality target and a 15-frame GOP so scrubbing it stays responsive. It is
    watched once and thrown away. "final" (the default) is the delivery render.
    """
    if profile == "draft":
        if prefer_nvenc and get_nvenc_available():
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-rc", "vbr",
                "-cq:v", "32",
                "-b:v", "0",
                "-g", "15",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "96k",
                "-ar", "48000",
                "-movflags", "+faststart",
            ]
        return [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-g", "15",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-ar", "48000",
            "-movflags", "+faststart",
        ]
    # yuv420p + high profile keeps the file playable everywhere (QuickTime,
    # browsers, mobile); faststart moves the moov atom to the front so the
    # in-app preview and web players can start before the full download; aac at
    # 48 kHz matches the audio graph and avoids A/V drift.
    if prefer_nvenc and get_nvenc_available():
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq:v", "21",
            "-b:v", "0",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
        ]
    else:
        return [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
        ]
