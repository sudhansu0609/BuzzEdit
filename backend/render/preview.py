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

* **`start_proxy`** — a 540p *draft* of the whole dressed programme (or of a
  window of it), which the preview then plays like any other video: dissolves
  dissolve, captions animate, zooms move. It is composed small, shows stills
  for generated video B-roll and skips the noise layers (see `draft_timeline`
  below), and renders in parallel chunks on NVENC. It goes stale the moment
  the edit changes, so it is something you ask for rather than something that
  happens to you.

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

        from render.cmdline import fit_command
        with fit_command(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex_script", str(script), "-map", "[fx_out]",
             "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1"]
        ) as (cmd, cwd):
            proc = subprocess.run(cmd, capture_output=True, cwd=cwd, creationflags=NO_WINDOW)

    if proc.returncode != 0 or not proc.stdout:
        logger.info("preview frame %d failed: %s", frame,
                    proc.stderr.decode(errors="replace")[:300])
        return None

    logger.debug("preview frame %d in %.0fms", frame, 1000 * (time.perf_counter() - started))
    _cache_put(key, proc.stdout)
    return proc.stdout


# --- the whole dressed programme, small: the draft ----------------------------
#
# The proxy is a *draft* of the dressed programme, not the render scaled down.
# It is composed at 540p from the start (text scaled with it, by the same
# helper the real render uses), generated video B-roll is shown as the still
# each clip was made from, the noise atmosphere layers are dropped, Ken Burns
# zooms run without supersampling, and it renders in parallel chunks on NVENC
# with a draft encoder profile. What is left is everything that decides an
# edit -- the cut, captions, cards, B-roll placement, grade, hit effects.

# Generated noise layers: full-frame lavfi sources (or per-pixel noise) that
# cost a pass over every frame and decide nothing about the edit.
NOISE_EFFECTS = frozenset({"grain", "vhs", "wind", "rain", "snow", "fog", "lightning",
                           "light_leak", "sunlight"})
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DRAFT_WORKERS = 2

_sidecar_index_cache: Dict[str, Tuple[int, Dict[str, Any]]] = {}
_sidecar_index_lock = threading.Lock()


def _generation_sidecars(directory: Path) -> Dict[str, Any]:
    """{stem: sidecar dict} for a generated-assets folder, re-read when it changes."""
    import json
    try:
        stamp = directory.stat().st_mtime_ns
    except OSError:
        return {}
    key = str(directory)
    with _sidecar_index_lock:
        cached = _sidecar_index_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
    index: Dict[str, Any] = {}
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            index[path.stem] = data
    with _sidecar_index_lock:
        _sidecar_index_cache[key] = (stamp, index)
    return index


def still_for_video(source_path: str) -> Optional[str]:
    """The still a generated B-roll clip was made from, or None.

    A clip's still (Asset.still_path) lives only in memory during generation;
    on disk the link is the generation sidecar `<key>.json` beside each file in
    `assets/generated`: a `broll_video` clip and the `broll_image` it started
    from share a `beat_id`. Exactly one such image must exist -- anything else
    is ambiguous and the clip is kept.
    """
    clip = Path(source_path)
    index = _generation_sidecars(clip.parent)
    side = index.get(clip.stem)
    if not side or side.get("kind") != "broll_video" or not side.get("beat_id"):
        return None
    matches: List[Path] = []
    for stem, data in index.items():
        if data.get("kind") != "broll_image" or data.get("beat_id") != side["beat_id"]:
            continue
        for suffix in _IMAGE_SUFFIXES:
            candidate = clip.parent / f"{stem}{suffix}"
            if candidate.exists():
                matches.append(candidate)
                break
    return str(matches[0]) if len(matches) == 1 else None


def _image_size(path: str) -> Tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 1920, 1080


def _walk(items: List[Any]):
    for item in items:
        yield item
        yield from _walk(item.children)


def draft_timeline(timeline: Timeline, height: int = 540, stills_for_video: bool = True,
                   drop_noise: bool = True) -> Timeline:
    """A copy of `timeline` made cheap to render, for the draft proxy.

    (a) composed at `height` (the programme's aspect, text scaled to match --
        runner.compose_canvas, the real render's own helper); width/height are
        set to the canvas it composes at;
    (b) generated video B-roll on V2+ shows the still it was made from (same
        span, transform and colour) when that still is found;
    (c) the noise atmosphere layers (NOISE_EFFECTS) are removed from the
        programme, clips and adjustment items; the cheap ones stay.
    """
    from render.runner import compose_canvas

    draft = timeline.model_copy(deep=True)

    if stills_for_video:
        for item in _walk(draft.items):
            track = item.track.upper()
            if item.kind != "media" or not track.startswith("V") or track == "V1":
                continue
            source = draft.sources.get(item.source_id or "")
            if source is None or (source.kind or "video") != "video":
                continue
            still = still_for_video(source.path)
            if still is None:
                continue
            still_id = f"{source.id}__still"
            if still_id not in draft.sources:
                width, height_px = _image_size(still)
                draft.sources[still_id] = source.model_copy(update={
                    "id": still_id, "path": still, "kind": "image", "has_audio": False,
                    "width": width, "height": height_px, "audio_override": None,
                    "audio_history": [], "duration_seconds": 0.0})
            item.source_id = still_id
            item.source_start_frame = 0
            item.source_end_frame = max(1, item.duration_frames)
        used = {i.source_id for i in _walk(draft.items) if i.source_id}
        draft.sources = {k: v for k, v in draft.sources.items()
                         if k in used or not k.endswith("__still")}

    if drop_noise:
        def keep(effects):
            return [e for e in effects if (e.type or "").lower() not in NOISE_EFFECTS]
        draft.effects = keep(draft.effects)
        for item in _walk(draft.items):
            if item.atmosphere:
                item.atmosphere = keep(item.atmosphere)

    program_w, program_h = FilterGraphCompiler(draft)._canvas_size()
    want_w = max(2, int(round(program_w * height / max(1, program_h) / 2.0)) * 2)
    draft, canvas = compose_canvas(draft, (want_w, int(height)))
    canvas = canvas or (program_w, program_h)
    return draft.model_copy(update={"width": canvas[0], "height": canvas[1]})


def _safe_cut(timeline: Timeline, frame: int, direction: int) -> int:
    """The nearest frame at or beyond `frame` (towards `direction`, -1 or +1)
    that no unsplittable picture item crosses -- where chunked.sub_timeline
    can cut without restarting a caption, a card or an animated overlay."""
    from render.chunked import _splittable, _visual
    total = int(timeline.duration_frames)
    spans = [(i.timeline_start_frame, i.timeline_end_frame) for i in timeline.items
             if i.enabled and _visual(i) and not _splittable(i)]
    while 0 < frame < total:
        crossing = [(s, e) for s, e in spans if s < frame < e]
        if not crossing:
            break
        frame = min(s for s, _ in crossing) if direction < 0 else max(e for _, e in crossing)
    return max(0, min(total, frame))


def window_timeline(timeline: Timeline, start_s: float, end_s: float
                    ) -> Tuple[Timeline, int, int]:
    """Programme time [start_s, end_s) as a timeline of its own, starting at 0,
    plus the (start, end) frames it really covers.

    The edges move outward to frames nothing animated crosses, so the picture
    is exactly the programme's (render/chunked.py:sub_timeline does the
    picture); audio items are trimmed to the same frames and the master kept.
    A late window then costs only its own length.
    """
    from render.chunked import _visual, sub_timeline
    fps = timeline.fps_num / max(1, timeline.fps_den)
    total = int(timeline.duration_frames)
    a = max(0, min(total - 1, int(round(float(start_s) * fps))))
    b = max(a + 1, min(total, int(round(float(end_s) * fps))))
    a, b = _safe_cut(timeline, a, -1), _safe_cut(timeline, b, +1)

    sub = sub_timeline(timeline, a, b)
    for item in timeline.items:
        if not item.enabled or _visual(item):
            continue
        s, e = item.timeline_start_frame, item.timeline_end_frame
        if e <= a or s >= b:
            continue
        clip = item.model_copy(deep=True)
        cut_s, cut_e = max(s, a), min(e, b)
        source = timeline.sources.get(clip.source_id or "")
        if cut_s > s:
            shift = cut_s - s
            if clip.loop and source is not None and source.duration_seconds > 0:
                period = max(1, int(round(source.duration_seconds * fps)))
                clip.source_start_frame = (clip.source_start_frame + shift) % period
            else:
                clip.source_start_frame += shift
            clip.audio_fade_in = 0.0
        if cut_e < e:
            clip.audio_fade_out = 0.0
        clip.source_end_frame = clip.source_start_frame + (cut_e - cut_s)
        clip.timeline_start_frame, clip.timeline_end_frame = cut_s - a, cut_e - a
        sub.items.append(clip)
        if source is not None:
            sub.sources[source.id] = source.model_copy(deep=True)
    sub.audio_master = (timeline.audio_master.model_copy(deep=True)
                        if timeline.audio_master is not None else None)
    return sub, a, b


class ProxyJob:
    """One in-flight proxy build. Cheap enough to keep in memory per project."""

    def __init__(self, project_id: str, revision: int,
                 window: Optional[Tuple[float, float]] = None):
        self.project_id = project_id
        self.revision = revision
        self.window = window
        self.state = "building"           # building | ready | error
        self.detail: str = ""
        self.path: Optional[str] = None
        self.started = time.time()
        self.finished: Optional[float] = None
        self.progress: float = 0.0
        self.offset_s: float = float(window[0]) if window else 0.0
        self.seconds: Optional[float] = None
        self.realtime_x: Optional[float] = None


_jobs: Dict[str, ProxyJob] = {}
_jobs_lock = threading.Lock()


def _window_key(window: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
    if not window:
        return None
    return int(round(float(window[0]) * 1000)), int(round(float(window[1]) * 1000))


def _job_key(project_id: str, window: Optional[Tuple[float, float]]) -> str:
    key = _window_key(window)
    return project_id if key is None else f"{project_id}@{key[0]}-{key[1]}"


def proxy_path(project_id: str, window: Optional[Tuple[float, float]] = None) -> Path:
    """The full draft, or the draft of one window (named by its requested
    start/end in ms) so the two coexist."""
    key = _window_key(window)
    if key is None:
        return PREVIEW_DIR / f"{project_id}_fx.mp4"
    return PREVIEW_DIR / f"{project_id}_fx_w{key[0]}-{key[1]}.mp4"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _read_meta(path: Path) -> Dict[str, Any]:
    import json
    try:
        return json.loads(_meta_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _existing_windows(project_id: str) -> List[Dict[str, Any]]:
    import re
    found = []
    for path in sorted(PREVIEW_DIR.glob(f"{project_id}_fx_w*.mp4")):
        match = re.fullmatch(rf"{re.escape(project_id)}_fx_w(\d+)-(\d+)\.mp4", path.name)
        if not match:
            continue
        start_s, end_s = int(match.group(1)) / 1000.0, int(match.group(2)) / 1000.0
        meta = _read_meta(path)
        found.append({"start_s": start_s, "end_s": end_s, "path": str(path),
                      "offset_s": meta.get("offset_s", start_s),
                      "revision": meta.get("revision")})
    return found


def proxy_status(project_id: str, revision: int,
                 window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """What the UI needs to decide between the proxy and a still.

    The shape is the old one ({state, stale, path, mtime, revision, ...}) plus
    the draft's fields: `draft`, `window`, `offset_s` (programme time of the
    proxy's first frame), `seconds` / `realtime_x` (build time), and which
    drafts exist (`full_exists`, `windows`)."""
    with _jobs_lock:
        job = _jobs.get(_job_key(project_id, window))
    path = proxy_path(project_id, window)
    extra = {"draft": True, "window": list(window) if window else None,
             "full_exists": proxy_path(project_id).exists(),
             "windows": _existing_windows(project_id) if PREVIEW_DIR.exists() else []}
    if job and job.state == "building":
        return {"state": "building", "elapsed": round(time.time() - job.started, 1),
                "revision": job.revision, "stale": job.revision != revision,
                "progress": round(job.progress, 1), **extra}
    if job and job.state == "error":
        return {"state": "error", "detail": job.detail, "revision": job.revision, **extra}
    if path.exists():
        meta = _read_meta(path)
        built_for = job.revision if job else meta.get("revision")
        return {
            "state": "ready",
            "path": str(path),
            "revision": built_for,
            # A proxy built before the last edit still plays; it just is not the
            # current cut, and saying so is better than quietly showing old work.
            "stale": built_for is None or built_for != revision,
            "size_bytes": path.stat().st_size,
            "mtime": path.stat().st_mtime,
            "offset_s": job.offset_s if job else meta.get("offset_s", 0.0),
            "seconds": job.seconds if job else meta.get("seconds"),
            "realtime_x": job.realtime_x if job else meta.get("realtime_x"),
            **extra,
        }
    return {"state": "none", "stale": True, **extra}


def _run_proxy(project_id: str, timeline: Timeline, height: int,
               window: Optional[Tuple[float, float]] = None) -> None:
    """Build the draft: (window ->) draft_timeline -> chunked NVENC render."""
    import asyncio
    import json
    from render.effects import ZOOM_SS_OVERRIDE
    from render.runner import render_timeline_async

    job = _jobs[_job_key(project_id, window)]
    dest = proxy_path(project_id, window)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial.mp4")
    started = time.perf_counter()
    token = ZOOM_SS_OVERRIDE.set(1)   # Ken Burns still moves, at 1x
    try:
        fps = timeline.fps_num / max(1, timeline.fps_den)
        # Drafted first, windowed second: a mood layer is unsplittable only
        # while it carries a noise layer, and with that gone the window's
        # edges no longer have to grow out to the mood's own span.
        draft = draft_timeline(timeline, height=height)
        if window is not None:
            draft, a, _b = window_timeline(draft, window[0], window[1])
            job.offset_s = a / fps
        canvas = (int(draft.width), int(draft.height))

        def on_progress(pct: float) -> None:
            job.progress = float(pct)

        # The loop's tasks and worker threads copy this context, so the zoom
        # override reaches every chunk's compile.
        asyncio.run(render_timeline_async(
            draft, str(partial), on_progress, prefer_nvenc=True,
            output_resolution=f"{canvas[0]}x{canvas[1]}", window=None,
            prerender=False, chunked=True, workers=DRAFT_WORKERS,
            encoder_profile="draft", composed_canvas=canvas))
        # Swapped in only once complete, so the player never picks up a
        # half-written file mid-build.
        partial.replace(dest)
        seconds = time.perf_counter() - started
        programme_s = draft.duration_frames / fps
        job.seconds = round(seconds, 1)
        job.realtime_x = round(programme_s / max(1e-6, seconds), 2)
        try:
            _meta_path(dest).write_text(json.dumps({
                "revision": job.revision, "window": list(window) if window else None,
                "offset_s": job.offset_s, "seconds": job.seconds,
                "realtime_x": job.realtime_x, "programme_s": round(programme_s, 2)}),
                encoding="utf-8")
        except OSError:
            pass
        job.path = str(dest)
        job.state = "ready"
        logger.info("FX draft for %s%s: %.1fs of programme in %.1fs (%.2fx realtime)",
                    project_id, f" window {window[0]:.1f}-{window[1]:.1f}s" if window else "",
                    programme_s, seconds, job.realtime_x)
    except Exception as exc:
        job.state = "error"
        job.detail = str(exc)[:600]
        logger.warning("FX draft for %s failed: %s", project_id, exc)
    finally:
        ZOOM_SS_OVERRIDE.reset(token)
        job.finished = time.time()


def _run_proxy_legacy(project_id: str, timeline: Timeline, height: int) -> None:
    """The pre-draft proxy: full-size single graph, scaled at the end, libx264.
    Kept only for before/after timing (start_proxy(legacy=True))."""
    job = _jobs[project_id]
    dest = proxy_path(project_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    script_path: Optional[Path] = None
    started = time.perf_counter()
    try:
        inputs, graph, v_label, a_label = FilterGraphCompiler(timeline).compile()
        # Scaled to an even width inside the graph: the output of a complex
        # filtergraph cannot also be fed through a simple -vf.
        graph = f"{graph};{v_label}scale=-2:{int(height)}:flags=bilinear[fx_v]"
        script_path = Path(tempfile.mkstemp(suffix=".txt", prefix="fxproxy_",
                                            dir=str(TEMP_DIR))[1])
        script_path.write_text(graph, encoding="utf-8")

        partial = dest.with_suffix(".partial.mp4")
        from render.cmdline import fit_command
        with fit_command(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *inputs,
             "-filter_complex_script", str(script_path),
             "-map", "[fx_v]", "-map", a_label,
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-g", "15",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "96k", str(partial)]
        ) as (cmd, cwd):
            proc = subprocess.run(cmd, capture_output=True, cwd=cwd, creationflags=NO_WINDOW)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-600:])
        partial.replace(dest)
        job.path = str(dest)
        job.state = "ready"
        job.seconds = round(time.perf_counter() - started, 1)
        fps = timeline.fps_num / max(1, timeline.fps_den)
        job.realtime_x = round(timeline.duration_frames / fps / max(1e-6, job.seconds), 2)
    except Exception as exc:
        job.state = "error"
        job.detail = str(exc)[:600]
        logger.warning("FX proxy (legacy) for %s failed: %s", project_id, exc)
    finally:
        job.finished = time.time()
        if script_path:
            try:
                script_path.unlink()
            except OSError:
                pass


def start_proxy(project_id: str, timeline: Timeline, height: int = 540,
                window: Optional[Tuple[float, float]] = None,
                legacy: bool = False) -> Dict[str, Any]:
    """Kick off a draft build (of the whole programme, or of `window` =
    (start_s, end_s)), unless that one is already running."""
    if window is not None:
        start_s, end_s = max(0.0, float(window[0])), float(window[1])
        window = (start_s, end_s) if end_s > start_s else None
    key = _job_key(project_id, window)
    with _jobs_lock:
        existing = _jobs.get(key)
        if existing and existing.state == "building":
            return {"state": "building", "already_running": True,
                    "elapsed": round(time.time() - existing.started, 1),
                    "window": list(window) if window else None}
        job = ProxyJob(project_id, timeline.revision, window)
        _jobs[key] = job

    if legacy:
        target, args = _run_proxy_legacy, (project_id, timeline, height)
    else:
        target, args = _run_proxy, (project_id, timeline, height, window)
    thread = threading.Thread(target=target, args=args,
                              name=f"fxproxy-{key}", daemon=True)
    thread.start()
    return {"state": "building", "already_running": False, "revision": timeline.revision,
            "window": list(window) if window else None, "draft": not legacy}
