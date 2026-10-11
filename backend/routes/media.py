import asyncio
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import anyio
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from config import DATA_DIR, FFMPEG_BIN, PROJECTS_DIR
from utils.proc import NO_WINDOW

router = APIRouter(prefix="/api/media", tags=["Media"])

VOICES_DIR = DATA_DIR / "voices"
# A download must come from a project's own media or the voices library --
# never an arbitrary path on disk.
_ALLOWED_DOWNLOAD_ROOTS = (PROJECTS_DIR, VOICES_DIR)

# Streams read on a limiter of their own. A sync generator behind
# StreamingResponse takes a default-pool thread per chunk, and when a project
# opens that pool is full of thumbnail, filmstrip and waveform work for every
# library entry: the preview's first 2 MB took 3.5 s to arrive and the player
# showed "Loading video stream" for ~20 s (2026-10-08). 1 MB chunks, so a
# 25-minute proxy is a few hundred reads rather than thousands.
STREAM_CHUNK = 1 << 20
_stream_limiter = anyio.CapacityLimiter(8)


async def _read_range(file_path: Path, start: int, length: int):
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            data = await anyio.to_thread.run_sync(f.read, min(STREAM_CHUNK, remaining),
                                                  limiter=_stream_limiter)
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/proxy")
async def preview_proxy(path: str):
    """Ensure a seek-friendly H.264 preview proxy for `path` and report its state.

    The live "Cut" preview seeks the source `<video>` to skip removed regions;
    on 10-bit HEVC phone footage those seeks stall and the cuts appear to do
    nothing. This hands the player a proxy it can actually seek. Non-blocking:
    the transcode runs in the background and the client polls until `ready`.
    """
    from utils.preview_proxy import ensure

    result = await asyncio.to_thread(ensure, path)
    if result.get("status") == "ready" and result.get("path"):
        result["url"] = f"/api/media/stream?path={quote(result['path'])}"
    return result

@router.get("/stream")
async def stream_media(path: str, request: Request, range: str = Header(None)):
    """
    Stream media file with HTTP Range support for HTML5 video/audio playback and seeking.
    """
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    file_size = file_path.stat().st_size
    content_type = "video/mp4"
    if file_path.suffix.lower() in (".mp3", ".wav", ".aac", ".flac"):
        content_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "audio/wav"

    if not range:
        return StreamingResponse(
            _read_range(file_path, 0, file_size),
            status_code=200,
            headers={
                "Content-Length": str(file_size),
                "Content-Type": content_type,
                "Accept-Ranges": "bytes",
            }
        )

    # Range header present, parse bytes range e.g. "bytes=0-1024"
    try:
        units, range_str = range.split("=")
        if units.strip() != "bytes":
            raise ValueError()
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1
    except Exception:
        raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": content_type,
    }
    return StreamingResponse(_read_range(file_path, start, length), status_code=206, headers=headers)


def _within_allowed_roots(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in _ALLOWED_DOWNLOAD_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _safe_download_name(name: str) -> str:
    """Strip characters that would break a `Content-Disposition` header or a
    filesystem path -- the value can come straight from the Voice panel."""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]', "_", name).strip()
    return cleaned or "download"


@router.get("/download")
async def download_media(path: str, format: str = "wav", name: Optional[str] = None):
    """Download an audio file (a Voice panel recording, or anything in the
    media library) as WAV or MP3. Only files inside BuzzEdit's own project and
    voice data are servable -- an arbitrary path on disk is refused.
    """
    file_path = Path(path)
    if not _within_allowed_roots(file_path):
        raise HTTPException(status_code=403, detail="Path is outside BuzzEdit's project and voice data")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    fmt = (format or "wav").strip().lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be wav or mp3")

    download_name = _safe_download_name(name or file_path.stem)

    if fmt == "wav" and file_path.suffix.lower() == ".wav":
        return FileResponse(str(file_path), media_type="audio/wav", filename=f"{download_name}.wav")

    media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"
    out_path = Path(tempfile.gettempdir()) / f"buzzedit_dl_{uuid.uuid4().hex}.{fmt}"
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(file_path), "-vn", "-ar", "48000"]
    if fmt == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    else:
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    cmd.append(str(out_path))

    result = await asyncio.to_thread(
        subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, creationflags=NO_WINDOW)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        err = (result.stderr or b"").decode("utf-8", "replace")[-500:]
        raise HTTPException(status_code=500, detail=f"Could not transcode to {fmt}: {err}")

    return FileResponse(str(out_path), media_type=media_type, filename=f"{download_name}.{fmt}",
                        background=BackgroundTask(out_path.unlink, missing_ok=True))
