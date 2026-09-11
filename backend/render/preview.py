"""Previews of the *dressed* programme — what the render will actually look like.

The editor's live preview plays the source file and skips the struck words, which
is honest about the cut and silent about everything else: a grade, rain, a light
leak, captions, B-roll and the aspect bars all exist only in the timeline until a
render happens. That is what made effects feel like they had not been applied —
you could set one, and there was nowhere to see it.

Two answers, because they trade off differently and a cutting room wants both:

* **`compose_frame`** — one exact frame at the playhead, in about a third of a
  second. It is built by handing the *real* `FilterGraphCompiler` a timeline
  trimmed to a single frame, so nothing here re-implements a look: whatever the
  render would draw at that moment is what comes back. Always current, cheap
  enough to fire as the playhead settles, and cached so scrubbing back over
  visited ground is instant. It cannot show motion.

* **`build_proxy`** — a low-resolution encode of the whole dressed programme,
  which the preview then plays like any other video: rain falls, dissolves
  dissolve, captions animate. It costs roughly 2.5x realtime to build and goes
  stale the moment the edit changes, so it is something you ask for rather than
  something that happens to you.

The one-frame trick has three sharp edges, all handled below:

1. **Stills, not seeks.** Feeding the compiler the original sources means ffmpeg
   decodes from the top of the file to reach the frame — eight seconds at the end
   of a one-minute programme. Instead each live clip's frame is pulled separately
   with a keyframe seek and the timeline is repointed at those stills, so the
   compose step reads a handful of tiny images and every position costs the same.

2. **The audio branch must terminate.** With no A-track items the compiler falls
   back to an *infinite* `anullsrc`. Sink that and ffmpeg never returns. It is
   bounded here before it is discarded.

3. **Generated layers start at t=0.** Rain, flicker and lightning are lavfi
   sources whose phase comes from the frame's timestamp, and a one-frame timeline
   always starts at zero — so a preview shows the effect at its opening phase
   rather than its phase at that moment. Intensity, colour and coverage are
   exact; only *where in its cycle* the animation is differs, which is what the
   proxy is for.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import FFMPEG_BIN, OUTPUT_DIR, TEMP_DIR
from render.compiler import FilterGraphCompiler, flatten_items
from timeline.schema import Timeline
from utils.proc import NO_WINDOW

logger = logging.getLogger("preview")

# Enough frames to scrub a minute of programme without re-running ffmpeg; a
# 640-wide JPEG is ~60KB, so this is tens of megabytes at worst.
_CACHE_LIMIT = 240
_cache: "OrderedDict[Tuple[Any, ...], bytes]" = OrderedDict()
_cache_lock = threading.Lock()

PREVIEW_DIR = OUTPUT_DIR / "preview"


def _cache_get(key) -> Optional[bytes]:
    with _cache_lock:
        data = _cache.get(key)
        if data is not None:
            _cache.move_to_end(key)
        return data


def _cache_put(key, data: bytes) -> None:
    with _cache_lock:
        _cache[key] = data
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)


def invalidate(project_id: str) -> None:
    """Drop every cached frame for a project. Called when its timeline changes."""
    with _cache_lock:
        for key in [k for k in _cache if k[0] == project_id]:
            del _cache[key]


# --- one exact frame ------------------------------------------------------

def _pull_still(job: Tuple[str, float, str]) -> bool:
    """One frame of one source, as a JPEG on disk.

    `-noaccurate_seek` lands on the preceding keyframe rather than decoding
    forward to the exact frame. On a preview that is at most a few frames early
    and it is the difference between 30ms and several seconds.
    """
    src_path, seconds, dest = job

    def attempt(at: float) -> bool:
        proc = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-noaccurate_seek",
             "-ss", f"{max(0.0, at):.3f}", "-i", src_path,
             "-frames:v", "1", "-q:v", "2", "-y", dest],
            capture_output=True, creationflags=NO_WINDOW,
        )
        if proc.returncode != 0:
            logger.debug("still pull failed for %s @%.2fs: %s", src_path, at,
                         proc.stderr.decode(errors="replace")[:200])
        return proc.returncode == 0 and Path(dest).exists()

    if attempt(seconds):
        return True
    # A seek that lands past the last keyframe — rounding at the very end of a
    # clip, or a source shorter than its timeline entry claims — yields no frame
    # at all. Backing off a beat is better than a blank preview.
    return seconds > 0.5 and attempt(seconds - 0.5)


def _one_frame_timeline(timeline: Timeline, frame: int, tmp: Path
                        ) -> Tuple[Optional[Timeline], List[Tuple[str, float, str]]]:
    """A timeline holding only what is on screen at `frame`, one frame long.

    Returns (timeline, still-pull jobs) or (None, []) when nothing is there —
    a gap between clips, or past the end.
    """
    fps = timeline.fps_num / timeline.fps_den
    sub = timeline.model_copy(deep=True)
    sub.sources = {}
    items: List[Any] = []
    jobs: List[Tuple[str, float, str]] = []

    for index, item in enumerate(flatten_items(timeline.items)):
        if not (item.timeline_start_frame <= frame < item.timeline_end_frame):
            continue
        # Audio contributes nothing to a still and its chain is the slowest part
        # of the graph; dropping it also removes the concat/mix work entirely.
        if item.track.upper().startswith("A"):
            continue

        clip = item.model_copy(deep=True)
        offset = frame - item.timeline_start_frame
        clip.timeline_start_frame = 0
        clip.timeline_end_frame = 1
        clip.children = []
        clip.mute = True

        if clip.source_id:
            source = timeline.sources.get(clip.source_id)
            if source is None:
                continue
            still_id = f"still_{index}"
            replacement = source.model_copy(deep=True)
            replacement.id = still_id
            replacement.has_audio = False
            if (source.kind or "video") == "image":
                # A still has one frame and no timeline to seek; asking for
                # `-ss 4.0` on a B-roll JPEG yields nothing at all, which is what
                # made every frame over generated B-roll come back empty.
                replacement.path = source.path
            else:
                still = str(tmp / f"still_{index}.jpg")
                jobs.append((source.path, (item.source_start_frame + offset) / fps, still))
                # Each clip gets its *own* still, so two items reading the same
                # file at different times do not have to share one seek.
                replacement.path = still
            replacement.kind = "image"
            sub.sources[still_id] = replacement
            clip.source_id = still_id
            clip.source_start_frame = 0
            clip.source_end_frame = 1

        items.append(clip)

    if not any(i.track.upper() == "V1" for i in items):
        return None, []

    sub.items = items
    sub.duration_frames = 1
    sub.program_offset_frames = 0
    # A single frame is not a junction, and a transition here would only ask the
    # graph for a second clip that does not exist.
    sub.default_transition = None
    for clip in sub.items:
        clip.transition = None
    return sub, jobs


def compose_frame(project_id: str, timeline: Timeline, seconds: float,
                  width: int = 640) -> Optional[bytes]:
    """The dressed frame at `seconds` of programme time, as JPEG bytes.

    None means there is nothing on V1 there — a gap, or past the end — and the
    caller should show black rather than a stale frame.
    """
    fps = timeline.fps_num / timeline.fps_den
    frame = max(0, int(round(seconds * fps)))
    key = (project_id, timeline.revision, frame, width)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fxframe_", dir=str(TEMP_DIR)) as tmpdir:
        tmp = Path(tmpdir)
        sub, jobs = _one_frame_timeline(timeline, frame, tmp)
        if sub is None:
            return None

        if jobs:
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                if not all(pool.map(_pull_still, jobs)):
                    logger.info("preview frame %d: a source still could not be pulled", frame)
                    return None

        try:
            inputs, graph, v_label, a_label = FilterGraphCompiler(sub).compile()
        except Exception as exc:                       # a half-built timeline
            logger.info("preview frame %d could not be compiled: %s", frame, exc)
            return None

        # The audio branch is discarded, but an unconnected output aborts the
        # graph and an *unbounded* one hangs it, so it is trimmed then sunk.
        graph = (f"{graph};{v_label}scale={int(width)}:-2:flags=bilinear[fx_out]"
                 f";{a_label}atrim=duration=0.05,anullsink")
        script = tmp / "graph.txt"
        script.write_text(graph, encoding="utf-8")

        proc = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex_script", str(script), "-map", "[fx_out]",
             "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1"],
            capture_output=True, creationflags=NO_WINDOW,
        )

    if proc.returncode != 0 or not proc.stdout:
        logger.info("preview frame %d failed: %s", frame,
                    proc.stderr.decode(errors="replace")[:300])
        return None

    logger.debug("preview frame %d in %.0fms", frame, 1000 * (time.perf_counter() - started))
    _cache_put(key, proc.stdout)
    return proc.stdout


# --- the whole dressed programme, small ------------------------------------

class ProxyJob:
    """One in-flight proxy build. Cheap enough to keep in memory per project."""

    def __init__(self, project_id: str, revision: int):
        self.project_id = project_id
        self.revision = revision
        self.state = "building"           # building | ready | error
        self.detail: str = ""
        self.path: Optional[str] = None
        self.started = time.time()
        self.finished: Optional[float] = None


_jobs: Dict[str, ProxyJob] = {}
_jobs_lock = threading.Lock()


def proxy_path(project_id: str) -> Path:
    return PREVIEW_DIR / f"{project_id}_fx.mp4"


def proxy_status(project_id: str, revision: int) -> Dict[str, Any]:
    """What the UI needs to decide between the proxy and a still."""
    with _jobs_lock:
        job = _jobs.get(project_id)
    path = proxy_path(project_id)
    if job and job.state == "building":
        return {"state": "building", "elapsed": round(time.time() - job.started, 1),
                "revision": job.revision, "stale": job.revision != revision}
    if job and job.state == "error":
        return {"state": "error", "detail": job.detail, "revision": job.revision}
    if path.exists():
        built_for = job.revision if job else None
        return {
            "state": "ready",
            "path": str(path),
            "revision": built_for,
            # A proxy built before the last edit still plays; it just is not the
            # current cut, and saying so is better than quietly showing old work.
            "stale": built_for is None or built_for != revision,
            "size_bytes": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }
    return {"state": "none", "stale": True}


def _run_proxy(project_id: str, timeline: Timeline, height: int) -> None:
    job = _jobs[project_id]
    dest = proxy_path(project_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    script_path: Optional[Path] = None
    try:
        inputs, graph, v_label, a_label = FilterGraphCompiler(timeline).compile()
        # Scaled to an even width inside the graph: the output of a complex
        # filtergraph cannot also be fed through a simple -vf.
        graph = f"{graph};{v_label}scale=-2:{int(height)}:flags=bilinear[fx_v]"
        script_path = Path(tempfile.mkstemp(suffix=".txt", prefix="fxproxy_",
                                            dir=str(TEMP_DIR))[1])
        script_path.write_text(graph, encoding="utf-8")

        partial = dest.with_suffix(".partial.mp4")
        proc = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *inputs,
             "-filter_complex_script", str(script_path),
             "-map", "[fx_v]", "-map", a_label,
             # Speed over size: this is watched once and thrown away. The short
             # GOP is what keeps scrubbing it responsive.
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-g", "15",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "96k", str(partial)],
            capture_output=True, creationflags=NO_WINDOW,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-600:])
        # Swapped in only once complete, so the player never picks up a
        # half-written file mid-build.
        partial.replace(dest)
        job.path = str(dest)
        job.state = "ready"
    except Exception as exc:
        job.state = "error"
        job.detail = str(exc)[:600]
        logger.warning("FX proxy for %s failed: %s", project_id, exc)
    finally:
        job.finished = time.time()
        if script_path:
            try:
                script_path.unlink()
            except OSError:
                pass


def start_proxy(project_id: str, timeline: Timeline, height: int = 540) -> Dict[str, Any]:
    """Kick off a dressed proxy build, unless one is already running."""
    with _jobs_lock:
        existing = _jobs.get(project_id)
        if existing and existing.state == "building":
            return {"state": "building", "already_running": True,
                    "elapsed": round(time.time() - existing.started, 1)}
        job = ProxyJob(project_id, timeline.revision)
        _jobs[project_id] = job

    thread = threading.Thread(
        target=_run_proxy, args=(project_id, timeline, height),
        name=f"fxproxy-{project_id}", daemon=True)
    thread.start()
    return {"state": "building", "already_running": False, "revision": timeline.revision}
