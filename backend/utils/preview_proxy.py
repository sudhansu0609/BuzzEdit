"""Seek-friendly preview proxies.

The live "Cut" preview reconstructs the edit by seeking the source `<video>`
element to each kept range and skipping the gaps between them. That only works
if the browser can seek quickly and accurately — which Chromium/Electron cannot
do on 10-bit HEVC (`hvc1`) iPhone footage. A large forward seek across a removed
region (some gaps here are 40s+) stalls or never lands, so the removed fumble
just keeps playing and the edit looks like it did nothing. The transcript, the
EDL and the final render are all correct; only the *preview* is affected.

The fix is what every NLE does: preview a lightweight H.264 8-bit proxy instead
of the raw camera file. H.264 8-bit with a dense keyframe interval seeks
instantly and frame-accurately, so the existing gap-skipper works as designed.

Proxies build once per source in a background thread, cached beside the source
as `<name>_proxy.mp4`. Sources in this app get unique per-project names and
never change in place, so an existing proxy file is trusted as-is.
"""

import logging
import threading
from pathlib import Path
from typing import Dict

from utils.ffmpeg_utils import run_ffmpeg, get_video_info

logger = logging.getLogger("preview_proxy")

# Codecs the browser already seeks reliably; these need no proxy.
_SEEKABLE_CODECS = {"h264", "avc1"}

# Longest side of the proxy. Small enough to encode fast and stream light, large
# enough that the preview still reads sharply on a vertical phone clip.
_MAX_DIMENSION = 1280

# out-path -> {"status": "building"|"ready"|"error", "detail"?: str}
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def proxy_path(source: str) -> Path:
    src = Path(source)
    return src.with_name(f"{src.stem}_proxy.mp4")


def needs_proxy(source: str) -> bool:
    """True when the browser cannot seek this source reliably (so the live cut
    preview would play through removed regions). False for plain H.264, which
    seeks fine as-is."""
    try:
        codec = str(get_video_info(source).get("video_codec", "")).lower()
    except Exception:
        return False
    return codec not in _SEEKABLE_CODECS


def ensure(source: str) -> dict:
    """Return the proxy state for `source`, kicking off a build if needed.

    Never blocks on the transcode: the encode runs in a daemon thread and the
    caller polls this until it reports `ready`. Statuses: `ready` (use `path`),
    `building`, `not_needed` (source seeks fine — preview the original),
    `none` (no such file), `error`.
    """
    src = Path(source)
    if not src.exists():
        return {"status": "none"}
    if not needs_proxy(source):
        return {"status": "not_needed"}

    out = proxy_path(source)
    key = str(out)
    with _lock:
        job = _jobs.get(key)
        if out.exists() and (job is None or job.get("status") == "ready"):
            _jobs[key] = {"status": "ready"}
            return {"status": "ready", "path": key}
        if job and job.get("status") == "building":
            return {"status": "building"}
        if job and job.get("status") == "error":
            # Let a caller retry a previously failed build.
            pass
        _jobs[key] = {"status": "building"}

    threading.Thread(target=_build, args=(str(src), key), daemon=True).start()
    return {"status": "building"}


def _build(source: str, out: str) -> None:
    partial = out + ".partial.mp4"
    try:
        run_ffmpeg([
            "-i", source,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",                       # 8-bit: the browser's happy path
            "-vf", (f"scale=w={_MAX_DIMENSION}:h={_MAX_DIMENSION}"
                    ":force_original_aspect_ratio=decrease:force_divisible_by=2"),
            # A keyframe every second so even a backward-then-forward seek lands
            # within ~1s of the target — the whole reason the raw HEVC failed.
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-y", partial,
        ], timeout=3600)
        Path(partial).replace(out)
        with _lock:
            _jobs[out] = {"status": "ready"}
        logger.info("Preview proxy ready: %s", out)
    except Exception as e:
        logger.warning("Preview proxy build failed for %s: %s", source, e)
        try:
            Path(partial).unlink(missing_ok=True)
        except Exception:
            pass
        with _lock:
            _jobs[out] = {"status": "error", "detail": str(e)[:200]}
