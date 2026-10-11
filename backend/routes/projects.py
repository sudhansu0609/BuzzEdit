import asyncio
import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from models import Project
from config import FFMPEG_BIN, PROJECTS_DIR
from utils.ffmpeg_utils import FFmpegError, get_video_info
from utils.proc import NO_WINDOW
from store import media_pool
from store.project_store import ProjectStore
from store.app_settings import app_settings

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

class PathImportRequest(BaseModel):
    path: str

class MediaImportRequest(BaseModel):
    # Files *or* folders; folders are walked for media. `copy` stores a private
    # duplicate in the project instead of linking the original in place. The field
    # is aliased because a plain `copy` attribute shadows BaseModel.copy().
    paths: List[str]
    copy_files: bool = Field(False, alias="copy")


class MediaRenameRequest(BaseModel):
    name: str


class MediaRelinkRequest(BaseModel):
    path: str


def _load(project_id: str) -> Dict[str, Any]:
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    return p_data


async def _load_readonly(project_id: str) -> Dict[str, Any]:
    """`_load` for routes that only read: parsed once per saved version, and
    off the event loop, so a burst of library requests cannot stall the
    responses everything else is waiting on (the preview's video stream)."""
    p_data = await asyncio.to_thread(project_store.get_project_readonly, project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    return p_data


def _pool_entry(p_data: Dict[str, Any], media_id: str) -> Dict[str, Any]:
    entry = next((m for m in p_data.get("media_pool", []) if m.get("id") == media_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Media not found in library")
    return entry


def _seed_media_pool(project: Project, video_path: str, original_path: str, display_name: str) -> None:
    """Put the project's own source video in the library.

    A new project should not open onto an empty bin — the footage you just imported
    is the first thing you want to drag onto a track.

    The entry points at the project's copy (which is what transcription and the
    timeline use) but records the file the user actually chose as `source_path`,
    so re-importing that same file later is recognised as a duplicate instead of
    landing in the library twice.
    """
    try:
        added, _skipped = media_pool.import_paths(project.id, project.media_pool, [video_path])
        if added:
            added[0]["name"] = display_name
            added[0]["source_path"] = original_path
            added[0]["linked"] = False   # the project owns this copy
    except Exception:
        pass  # an empty library is a cosmetic loss, never a failed import

def _disk_full(e: OSError) -> bool:
    # ENOSPC, or Windows' ERROR_DISK_FULL (112) / ERROR_HANDLE_DISK_FULL (39).
    return e.errno == 28 or getattr(e, "winerror", None) in (39, 112) or "not enough space" in str(e).lower()


def place_source(src: Path, dest: Path) -> str:
    """Put the recording at `dest`: a hard link when it is on the same drive
    (instant, and no second copy of a multi-GB take -- re-imports used to fill
    the disk), else a copy. Nothing in BuzzEdit writes into the source file,
    and removing the project removes only its own name for it. A copy checks
    the free space first and never leaves a partial file behind."""
    try:
        dest.hardlink_to(src)
        return "linked"
    except OSError:
        pass   # another drive, a filesystem without hard links, or not allowed
    size = src.stat().st_size
    free = shutil.disk_usage(str(dest.parent)).free
    if free < size + 512 * 1024 * 1024:
        raise OSError(28, f"Not enough space on {dest.anchor or dest.parent} to copy the recording: "
                          f"it needs {size / 1e9:.1f} GB, {free / 1e9:.1f} GB is free. "
                          "Free some space (or keep recordings on the same drive as BuzzEdit) and try again.")
    try:
        shutil.copy(str(src), str(dest))
    except OSError as e:
        dest.unlink(missing_ok=True)
        raise OSError(e.errno, f"Failed to copy the recording: {e}") from e
    return "copied"


@router.post("/import_path", response_model=Project)
async def import_video_path(body: PathImportRequest):
    src_path = Path(body.path)
    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"File not found at path: {body.path}")

    file_ext = src_path.suffix
    project_id = str(uuid.uuid4())[:8]
    dest_video_path = PROJECTS_DIR / f"{project_id}_source{file_ext}"

    try:
        place_source(src_path, dest_video_path)
    except OSError as e:
        raise HTTPException(status_code=507 if _disk_full(e) else 500, detail=str(e))

    try:
        video_info = get_video_info(str(dest_video_path))
    except Exception as e:
        dest_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")

    project = Project(
        id=project_id,
        name=src_path.name,
        source_video=str(dest_video_path),
        status="draft",
    )

    project.settings["source_width"] = video_info["width"]
    project.settings["source_height"] = video_info["height"]
    project.settings["source_duration"] = video_info["duration"]
    _seed_media_pool(project, str(dest_video_path), str(src_path), src_path.name)

    project_store.save_project(project_id, project.model_dump())
    app_settings.set_last_project(project_id)
    return project

@router.get("/{project_id}/media")
async def list_media(
    project_id: str,
    kind: Optional[str] = Query(None, description="Filter to video | audio | image"),
    q: Optional[str] = Query(None, description="Substring match on the file name"),
):
    """The project's media library — everything available to drag onto the timeline."""
    p_data = _load(project_id)
    pool = p_data.setdefault("media_pool", [])
    # Backfill anything the timeline uses that never went through an explicit
    # import (a clip dragged onto a track, or media a generation pass wrote
    # straight in) so the library shows everything the project actually plays.
    if media_pool.sync_from_timeline(pool, p_data.get("timeline")):
        project_store.save_project(project_id, p_data)
    media = media_pool.decorate(project_id, pool)
    if kind:
        media = [m for m in media if m.get("kind") == kind]
    if q:
        needle = q.lower()
        media = [m for m in media if needle in str(m.get("name", "")).lower()]
    counts: Dict[str, int] = {"video": 0, "audio": 0, "image": 0}
    for entry in pool:
        counts[entry.get("kind", "video")] = counts.get(entry.get("kind", "video"), 0) + 1
    return {"media": media, "counts": counts, "total": len(pool)}


@router.post("/{project_id}/media/import")
async def import_media(project_id: str, body: MediaImportRequest):
    """Add files or whole folders to the library.

    Linking is the default, so importing a multi-gigabyte clip is instant; pass
    `copy: true` to store a private duplicate inside the project instead.
    """
    p_data = _load(project_id)
    pool: List[dict] = p_data.get("media_pool", [])

    # Probing runs ffprobe per file, so keep it off the event loop — a folder
    # import would otherwise stall every other request.
    added, skipped = await asyncio.to_thread(
        media_pool.import_paths, project_id, pool, body.paths, body.copy_files
    )

    if not added:
        detail = "; ".join(skipped) if skipped else "No supported media files found"
        raise HTTPException(status_code=400, detail=detail)

    p_data["media_pool"] = pool
    project_store.save_project(project_id, p_data)
    return {
        "status": "success",
        "added": added,
        "skipped": skipped,
        "media": media_pool.decorate(project_id, pool),
    }


MAX_UPLOAD_BYTES = 2 * 1024 ** 3  # 2 GiB

_UPLOAD_CONTENT_TYPE_EXT = {
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/ogg": ".ogg", "video/ogg": ".ogv",
    "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "video/mp4": ".mp4",
}


def _upload_suffix(filename: Optional[str], content_type: Optional[str]) -> str:
    """A MediaRecorder blob names itself (`recording.webm`); fall back to the
    multipart content type only when the browser sent no filename at all."""
    suffix = Path(filename or "").suffix.lower()
    if suffix:
        return suffix
    return _UPLOAD_CONTENT_TYPE_EXT.get((content_type or "").split(";")[0].strip().lower(), ".bin")


@router.post("/{project_id}/media/upload")
async def upload_media(project_id: str, file: UploadFile = File(...),
                       name: Optional[str] = Form(None)):
    """Add a browser microphone recording (or any audio/video file) to the library.

    A MediaRecorder blob arrives as `audio/webm` or `audio/ogg`. Anything audio
    that is not already WAV is transcoded to 48 kHz 16-bit WAV so it mixes
    cleanly with the rest of the project; video is stored untouched.
    """
    p_data = _load(project_id)
    pool: List[dict] = p_data.get("media_pool", [])

    display_name = (name or file.filename or "recording").strip() or "recording"
    media_id = f"media_{uuid.uuid4().hex[:8]}"
    staged = PROJECTS_DIR / f"{project_id}_{media_id}{_upload_suffix(file.filename, file.content_type)}"

    size = 0
    try:
        with open(staged, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds the 2 GB upload limit")
                out.write(chunk)
    except HTTPException:
        staged.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    try:
        info = await asyncio.to_thread(get_video_info, str(staged))
    except FFmpegError as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unreadable media file: {exc}")

    has_video = info.get("video_codec", "none") != "none"
    if not has_video and info.get("audio_codec", "none") == "none":
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File has no audio or video stream ffmpeg can read")

    final_path = staged
    if not has_video and staged.suffix.lower() != ".wav":
        wav_path = PROJECTS_DIR / f"{project_id}_{media_id}.wav"
        cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(staged),
               "-vn", "-ar", "48000", "-c:a", "pcm_s16le", str(wav_path)]
        result = await asyncio.to_thread(
            subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120, creationflags=NO_WINDOW)
        if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            staged.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)
            err = (result.stderr or b"").decode("utf-8", "replace")[-500:]
            raise HTTPException(status_code=400, detail=f"Could not transcode audio: {err}")
        staged.unlink(missing_ok=True)
        final_path = wav_path

    added, skipped = await asyncio.to_thread(media_pool.import_paths, project_id, pool, [str(final_path)])
    if not added:
        final_path.unlink(missing_ok=True)
        detail = "; ".join(skipped) if skipped else "Could not add the uploaded file to the library"
        raise HTTPException(status_code=400, detail=detail)

    entry = added[0]
    entry["name"] = display_name
    entry["linked"] = False  # the project owns this copy

    p_data["media_pool"] = pool
    project_store.save_project(project_id, p_data)

    return {
        "media_id": entry["id"],
        "path": entry["path"],
        "name": entry["name"],
        "kind": entry["kind"],
        "seconds": entry["duration"],
    }


@router.get("/{project_id}/media/{media_id}/thumb")
async def media_thumbnail(project_id: str, media_id: str):
    """Poster image for a library tile, generated on first request."""
    p_data = await _load_readonly(project_id)
    entry = _pool_entry(p_data, media_id)
    thumb = await asyncio.to_thread(media_pool.ensure_thumbnail, project_id, entry)
    if thumb:
        return FileResponse(str(thumb), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No thumbnail available")


@router.get("/{project_id}/media/{media_id}/waveform")
async def media_waveform(project_id: str, media_id: str):
    """Audio peaks so music and voiceover tiles show a waveform instead of an icon."""
    from utils.waveform import compute_waveform
    p_data = await _load_readonly(project_id)
    entry = _pool_entry(p_data, media_id)
    data = await asyncio.to_thread(compute_waveform, entry.get("path", ""))
    if not data:
        return {"peaks": [], "points_per_second": 60, "duration": 0.0}
    return data


@router.patch("/{project_id}/media/{media_id}")
async def rename_media(project_id: str, media_id: str, body: MediaRenameRequest):
    """Rename a library entry. The file on disk is left alone."""
    p_data = _load(project_id)
    entry = _pool_entry(p_data, media_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    entry["name"] = name
    project_store.save_project(project_id, p_data)
    return {"status": "success", "media": media_pool.decorate(project_id, p_data["media_pool"])}


@router.post("/{project_id}/media/{media_id}/relink")
async def relink_media(project_id: str, media_id: str, body: MediaRelinkRequest):
    """Point an entry at a new file after the original was moved or renamed."""
    p_data = _load(project_id)
    _pool_entry(p_data, media_id)
    updated = await asyncio.to_thread(media_pool.relink, p_data["media_pool"], media_id, body.path)
    if not updated:
        raise HTTPException(status_code=400, detail=f"File not found: {body.path}")
    project_store.save_project(project_id, p_data)
    return {"status": "success", "media": media_pool.decorate(project_id, p_data["media_pool"])}


@router.delete("/{project_id}/media/{media_id}")
async def remove_media(project_id: str, media_id: str,
                       delete_file: bool = Query(False, description="Also delete the file on disk")):
    """Remove a library entry. Clips already on the timeline are unaffected.

    Only files the project owns (imported with `copy`) can be deleted from disk —
    a linked entry points at the user's own file and must never be removed.
    """
    p_data = _load(project_id)
    entry = _pool_entry(p_data, media_id)
    p_data["media_pool"] = [m for m in p_data.get("media_pool", []) if m.get("id") != media_id]
    project_store.save_project(project_id, p_data)

    if delete_file and not entry.get("linked", False):
        try:
            Path(entry["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    media_pool.thumb_path(project_id, media_id).unlink(missing_ok=True)
    return {"status": "success", "media": media_pool.decorate(project_id, p_data["media_pool"])}

@router.post("/upload", response_model=Project)
async def upload_video(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix
    project_id = str(uuid.uuid4())[:8]
    video_path = PROJECTS_DIR / f"{project_id}_source{file_ext}"

    content = await file.read()
    video_path.write_bytes(content)

    try:
        video_info = get_video_info(str(video_path))
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")

    project = Project(
        id=project_id,
        name=file.filename,
        source_video=str(video_path),
        status="draft",
    )

    project.settings["source_width"] = video_info["width"]
    project.settings["source_height"] = video_info["height"]
    project.settings["source_duration"] = video_info["duration"]
    # An upload has no original path on disk to dedupe against, so the copy is both.
    _seed_media_pool(project, str(video_path), str(video_path), file.filename)

    project_store.save_project(project_id, project.model_dump())
    app_settings.set_last_project(project_id)
    return project

@router.get("/list")
async def list_projects():
    projs = project_store.list_projects()
    result = []
    for p in projs:
        result.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "status": p.get("status"),
            "source_video": p.get("source_video"),
            "sourceVideo": p.get("source_video"),
        })
    return {"projects": result}

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    data = project_store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    app_settings.set_last_project(project_id)
    project = Project.model_validate(data)
    # Stamp the rendered file's mtime so the UI can cache-bust the Rendered
    # preview — the render always writes the same filename, so the stream URL
    # would otherwise be identical after a re-render and the browser would keep
    # showing the stale (e.g. pre-zoom) video.
    if project.output_path:
        try:
            project.output_version = Path(project.output_path).stat().st_mtime
        except OSError:
            project.output_version = None
    return project

@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, updates: Dict[str, Any]):
    data = project_store.get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.model_validate(data)
    updated = project.model_copy(update=updates)
    project_store.save_project(project_id, updated.model_dump())
    return updated

@router.delete("/{project_id}")
async def delete_project(project_id: str):
    success = project_store.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "id": project_id}
