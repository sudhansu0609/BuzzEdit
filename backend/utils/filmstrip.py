"""Filmstrip strips for the timeline UI.

A video clip on a timeline lane used to be a flat coloured rectangle with a text
label, which says nothing about what is actually in the shot. This renders the
source once into a single horizontal strip of evenly spaced frames; the UI then
shows the part of that strip covering each clip's trimmed range, so a lane reads
as pictures rather than as a row of identical blocks.

One strip per source, not per clip: a source cut into thirty pieces would
otherwise mean thirty ffmpeg runs for the same footage. Results are cached to
TEMP_DIR alongside the waveform sidecars.
"""

import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Optional

from config import FFMPEG_BIN, TEMP_DIR
from utils.ffmpeg_utils import get_video_info
from utils.proc import NO_WINDOW

logger = logging.getLogger("filmstrip")

TILE_HEIGHT = 48             # tall enough to stay sharp on a ~38px lane clip
_SECONDS_PER_TILE = 2.0      # target sampling density
_MAX_TILES = 300
# Chromium will not upload a texture wider than 16384px, and a background image
# that exceeds it is silently dropped -- so the whole strip has to stay well
# under that. This budget also keeps the JPEG around 100-200KB.
_MAX_STRIP_WIDTH = 8192


def _cache_key(media_path: str, height: int) -> str:
    digest = hashlib.md5(f"{media_path}|{height}".encode("utf-8")).hexdigest()[:16]
    return f"strip_{digest}"


def _meta_path(key: str) -> Path:
    return TEMP_DIR / f"{key}.json"


def _image_path(key: str) -> Path:
    return TEMP_DIR / f"{key}.jpg"


def compute_filmstrip(media_path: str, kind: str = "video",
                      height: int = TILE_HEIGHT) -> Optional[dict]:
    """Build (or reuse) a strip for `media_path`.

    Returns {image_path, columns, tile_width, tile_height, duration}, or None
    when the file has no picture to sample. `columns` tiles are spaced evenly
    across the whole source, so tile *i* is the frame at
    ``(i + 0.5) * duration / columns`` seconds.
    """
    path = Path(media_path)
    if not path.exists():
        return None
    if kind == "audio":
        return None

    key = _cache_key(str(path), height)
    meta_file, image_file = _meta_path(key), _image_path(key)
    if meta_file.exists() and image_file.exists() and image_file.stat().st_size > 0:
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            data["image_path"] = str(image_file)
            return data
        except Exception:
            pass

    try:
        info = get_video_info(str(path))
    except Exception as exc:
        logger.warning("Could not probe %s for a filmstrip: %s", path, exc)
        return None

    src_w = int(info.get("width") or 0)
    src_h = int(info.get("height") or 0)
    if src_w <= 0 or src_h <= 0:
        return None

    # Even width: JPEG's 4:2:0 chroma subsampling rejects odd dimensions, and an
    # odd tile would also put every later tile on a half-pixel boundary.
    tile_w = max(2, int(round(height * src_w / src_h)) & ~1)
    duration = float(info.get("duration") or 0.0)

    if kind == "image" or duration <= 0:
        # A still has nothing to sample over time: one tile, which the UI repeats
        # across however long the clip is.
        columns = 1
        vfilter = f"scale={tile_w}:{height}"
        duration = duration if duration > 0 else 0.0
    else:
        columns = int(min(_MAX_TILES,
                          _MAX_STRIP_WIDTH // tile_w,
                          max(1, math.ceil(duration / _SECONDS_PER_TILE))))
        # fps is what spaces the samples evenly; tile packs them into one image.
        # A rounding shortfall of one frame leaves the final cell blank, which is
        # invisible at this size, so an exact frame count is not worth chasing.
        rate = columns / duration
        vfilter = f"fps={rate:.6f},scale={tile_w}:{height},tile={columns}x1"

    cmd = [FFMPEG_BIN, "-y", "-v", "error", "-i", str(path),
           "-vf", vfilter, "-frames:v", "1", "-q:v", "6", str(image_file)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=180, creationflags=NO_WINDOW)
    except Exception as exc:
        logger.warning("Filmstrip ffmpeg failed for %s: %s", path, exc)
        return None

    if proc.returncode != 0 or not image_file.exists() or image_file.stat().st_size == 0:
        logger.warning("Filmstrip produced nothing for %s: %s", path,
                       proc.stderr.decode("utf-8", "ignore")[-200:])
        return None

    data = {
        "columns": columns,
        "tile_width": tile_w,
        "tile_height": height,
        "duration": duration,
    }
    try:
        meta_file.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    data["image_path"] = str(image_file)
    return data
