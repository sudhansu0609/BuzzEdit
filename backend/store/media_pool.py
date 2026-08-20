"""The project media library — the bin you import into and drag onto the timeline.

Two decisions shape this module:

**Imports link by default rather than copy.** Copying a 300 MB phone clip into the
project directory takes long enough that the import looks broken, and it doubles the
disk cost of every file. Entries therefore point at the original path, exactly like
Filmora's project media. `copy=True` is still available for material that needs to
travel with the project, and because an entry can then point at a file that later
moves, every listing re-checks existence and flags `missing`.

**Thumbnails are generated lazily and addressed by convention.** The poster path is
derived from the project and media ids, so a pool entry that never recorded a thumb
(or whose thumb was deleted) heals itself the next time the tile is requested instead
of showing a permanent hole.
"""

import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import FFMPEG_BIN, PROJECTS_DIR
from utils.ffmpeg_utils import get_video_duration, get_video_info

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".mts", ".m2ts"}
MEDIA_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

# Folder imports walk into subdirectories, but only so far — dropping a drive root
# should not spend minutes probing thousands of files.
MAX_FOLDER_DEPTH = 4
MAX_FOLDER_FILES = 500


def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return "video"


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTS


def expand_paths(paths: Iterable[str]) -> List[Path]:
    """Turn a mixed list of files and folders into a flat list of media files."""
    found: List[Path] = []
    seen: set = set()

    def add(path: Path) -> None:
        key = str(path).lower()
        if key not in seen and is_media_file(path) and path.is_file():
            seen.add(key)
            found.append(path)

    for raw in paths:
        candidate = Path(raw)
        if candidate.is_file():
            add(candidate)
        elif candidate.is_dir():
            base_depth = len(candidate.parts)
            for child in sorted(candidate.rglob("*")):
                if len(found) >= MAX_FOLDER_FILES:
                    logger.warning("Folder import truncated at %d files", MAX_FOLDER_FILES)
                    break
                if len(child.parts) - base_depth > MAX_FOLDER_DEPTH:
                    continue
                add(child)
    return found


def probe(path: Path, kind: str) -> Dict[str, Any]:
    """Duration/geometry/audio for one file, degrading to defaults on unreadable media."""
    info: Dict[str, Any] = {"duration": 0.0, "width": 0, "height": 0,
                            "has_audio": kind != "image", "fps": 0.0}
    try:
        if kind == "audio":
            info["duration"] = float(get_video_duration(str(path)) or 0.0)
            info["has_audio"] = True
            return info

        probed = get_video_info(str(path))
        info["width"] = int(probed.get("width") or 0)
        info["height"] = int(probed.get("height") or 0)
        if kind == "image":
            info["duration"] = 5.0          # stills get a default on-timeline length
            info["has_audio"] = False
        else:
            info["duration"] = float(probed.get("duration") or 0.0)
            info["has_audio"] = probed.get("audio_codec") not in (None, "none")
            fps_num = probed.get("fps_num")
            fps_den = probed.get("fps_den") or 1
            if fps_num:
                info["fps"] = round(float(fps_num) / float(fps_den), 3)
    except Exception as exc:
        logger.warning("Could not probe %s: %s", path, exc)
    return info


def thumb_path(project_id: str, media_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}_{media_id}_thumb.jpg"


def ensure_thumbnail(project_id: str, entry: Dict[str, Any]) -> Optional[Path]:
    """Return the entry's poster image, generating it on first request.

    Deriving the path from the ids (instead of trusting a stored `thumb` field)
    means an entry saved before thumbnails existed, or one whose file was cleaned
    up, still gets a picture rather than a broken tile.
    """
    media_id = entry.get("id")
    if not media_id:
        return None
    target = thumb_path(project_id, media_id)
    if target.exists() and target.stat().st_size > 0:
        return target

    source = Path(entry.get("path", ""))
    kind = entry.get("kind") or media_kind(source)
    if kind == "audio" or not source.exists():
        return None

    if kind == "image":
        cmd = [FFMPEG_BIN, "-y", "-v", "error", "-i", str(source),
               "-vf", "scale=320:-2", "-frames:v", "1", "-update", "1", str(target)]
    else:
        # Seek before -i so the grab is instant even on a multi-gigabyte source.
        cmd = [FFMPEG_BIN, "-y", "-v", "error", "-ss", "1", "-i", str(source),
               "-vf", "scale=320:-2", "-frames:v", "1", "-update", "1", str(target)]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except Exception as exc:
        logger.warning("Thumbnail failed for %s: %s", source, exc)
        return None

    if target.exists() and target.stat().st_size > 0:
        return target
    # A clip shorter than the 1s seek point yields nothing — retry from frame one.
    if kind == "video":
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-v", "error", "-i", str(source),
                 "-vf", "scale=320:-2", "-frames:v", "1", "-update", "1", str(target)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except Exception:
            return None
    return target if target.exists() and target.stat().st_size > 0 else None


def _existing_sources(pool: List[Dict[str, Any]]) -> set:
    """Lower-cased original paths already in the pool, for duplicate rejection."""
    keys = set()
    for entry in pool:
        for field in ("source_path", "path"):
            value = entry.get(field)
            if value:
                keys.add(str(value).lower())
    return keys


def import_paths(
    project_id: str,
    pool: List[Dict[str, Any]],
    paths: Iterable[str],
    copy: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Add files (and the contents of folders) to a pool.

    Returns (added_entries, skipped_names). Duplicates and unreadable files are
    skipped rather than failing the whole import — dropping a folder of 40 clips
    should not be undone by one corrupt file.
    """
    known = _existing_sources(pool)
    added: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for source in expand_paths(paths):
        if str(source).lower() in known:
            skipped.append(f"{source.name} (already in library)")
            continue

        kind = media_kind(source)
        media_id = f"media_{uuid.uuid4().hex[:8]}"
        stored = source

        if copy:
            stored = PROJECTS_DIR / f"{project_id}_{media_id}{source.suffix}"
            try:
                shutil.copy(str(source), str(stored))
            except Exception as exc:
                logger.warning("Could not copy %s: %s", source, exc)
                skipped.append(f"{source.name} (copy failed)")
                continue

        info = probe(stored, kind)
        if kind != "image" and info["duration"] <= 0:
            # Zero duration means ffprobe could not read it as media at all.
            if copy:
                Path(stored).unlink(missing_ok=True)
            skipped.append(f"{source.name} (unreadable)")
            continue

        try:
            size = source.stat().st_size
        except Exception:
            size = 0

        entry = {
            "id": media_id,
            "name": source.name,
            "path": str(stored),
            "source_path": str(source),
            "kind": kind,
            "duration": info["duration"],
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "has_audio": info["has_audio"],
            "size_bytes": size,
            "linked": not copy,
        }
        pool.append(entry)
        added.append(entry)
        known.add(str(source).lower())

    return added, skipped


def decorate(project_id: str, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add view-only fields (`missing`, `has_thumb`) without touching stored state."""
    decorated = []
    for entry in pool:
        view = dict(entry)
        path = Path(entry.get("path", ""))
        view["missing"] = not path.exists()
        view["has_thumb"] = (entry.get("kind") != "audio") and not view["missing"]
        decorated.append(view)
    return decorated


def relink(pool: List[Dict[str, Any]], media_id: str, new_path: str) -> Optional[Dict[str, Any]]:
    """Point an entry at a new file after the original moved."""
    candidate = Path(new_path)
    if not candidate.is_file():
        return None
    for entry in pool:
        if entry.get("id") == media_id:
            entry["path"] = str(candidate)
            entry["source_path"] = str(candidate)
            entry["name"] = candidate.name
            info = probe(candidate, entry.get("kind") or media_kind(candidate))
            entry.update({"duration": info["duration"], "width": info["width"],
                          "height": info["height"], "has_audio": info["has_audio"]})
            return entry
    return None
