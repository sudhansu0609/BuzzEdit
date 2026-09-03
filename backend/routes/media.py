import asyncio
import os
from pathlib import Path
from urllib.parse import quote
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/media", tags=["Media"])


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
        def iter_full():
            with open(file_path, "rb") as f:
                yield from f
        return StreamingResponse(
            iter_full(),
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

    def iter_range(chunk_size=65536):
        with open(file_path, "rb") as f:
            f.seek(start)
            bytes_remaining = length
            while bytes_remaining > 0:
                read_size = min(chunk_size, bytes_remaining)
                data = f.read(read_size)
                if not data:
                    break
                bytes_remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": content_type,
    }
    return StreamingResponse(iter_range(), status_code=206, headers=headers)
