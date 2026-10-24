"""Chunked, parallel rendering of a long timeline.

One filter graph for a whole presented video holds every overlay, caption and
effect of the programme (~1,000 items, ~240 overlay branches on a 14-minute
Raat3Baje episode). Every frame passes through all of them whether they are on
screen or not, so the cost grows with items x frames: a 14-minute render ran at
0.06x realtime, single graph, ~3 cores busy out of 28, 49 GB of memory.

Here the picture is cut into chunks of ~CHUNK_TARGET_S, each compiled from only
the items it shows, rendered in parallel (each on its own NVENC session), and
joined without re-encoding. The audio is rendered once for the whole programme
(cheap, and the two-pass loudness stays one measurement), then muxed in.

A chunk boundary is only placed where nothing animated crosses it: the one
thing allowed to be split is a plain V1 clip (no transform), away from its own
edges so no transition is cut. Everything that runs across a boundary would
otherwise restart its animation there. Programme-wide atmosphere effects are
the exception; they are noise and simply continue with a new seed.
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import TEMP_DIR
from render.audio import audio_subgraph
from render.cmdline import fit_command, prune_inputs
from render.compiler import FilterGraphCompiler
from render.encoder import get_encoder_flags
from runtime import sentinel_priority
from timeline.schema import Timeline, TimelineItem
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)

CHUNK_TARGET_S = 45.0
# How far either side of the target a boundary may move to find a safe frame.
SEARCH_S = 20.0
# A remainder shorter than this joins the chunk before it.
MIN_CHUNK_S = 12.0
# Below this length the single graph is cheap enough.
MIN_PROGRAMME_S = 90.0
# Parallel chunk renders: NVENC on consumer cards allows 8 sessions; each ffmpeg
# also runs its own filter threads, so a few at once fills a 28-thread CPU.
MAX_WORKERS = 4
# A chunk this many times over the target is worth a log line naming what
# blocked it -- there is no settings system in this backend for render
# internals (every other constant here is a plain module constant too), so
# this is the override point for the next person who needs one.
_LONG_CHUNK_FACTOR = 3.0

# How long an ffmpeg chunk may go with neither playback progress (`out_time_ms`)
# nor its own output file growing before `_follow` kills it. Overridable like
# any other BUZZEDIT_* knob (see config.py); no settings store exists for the
# render internals to read instead.
STALL_SECONDS = float(os.environ.get("BUZZEDIT_CHUNK_STALL_S", "600"))
# How often the stall watchdog samples progress/output size.
_STALL_POLL_S = 5.0
# A chunk is retried this many times (alone, after everything else running
# finishes) before render_chunked gives up and raises.
MAX_CHUNK_RETRIES = 2
# chunks_* work directories older than this are swept at the start of a run.
_WORK_DIR_MAX_AGE_DAYS = 3.0


class ChunkStalledError(RuntimeError):
    """A chunk's ffmpeg made no progress for STALL_SECONDS and was killed."""


# Peak memory of one chunk's ffmpeg at 1080p, measured on a 14-minute episode
# (3.2 GB typical, 6.3 GB for the busiest chunk). It does not grow with the
# pixel count: a 4K chunk of the same episode peaked at ~6 GB, so the scale is
# a gentle power of the pixel ratio.
_CHUNK_GB_1080P = 3.5
_CHUNK_GB_EXPONENT = 0.4
# Left free for everything else on the machine (the Studio, a browser, ComfyUI idle).
_RAM_RESERVE_GB = 6.0


def _free_ram_gb() -> float:
    """Available physical memory in GB (Windows), or a cautious 8 when unknown."""
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 1024 ** 3
    except Exception:
        pass
    return 8.0


def parallel_chunks(canvas: Tuple[int, int], free_gb: Optional[float] = None) -> int:
    """How many chunks to render at once: as many as fit in free RAM at this
    canvas size (a 4K chunk needs ~1.7x a 1080p one), between 1 and MAX_WORKERS.
    Paging is what made the old renders crawl, so this errs low."""
    ratio = max(0.25, (canvas[0] * canvas[1]) / (1920 * 1080))
    per_chunk = _CHUNK_GB_1080P * ratio ** _CHUNK_GB_EXPONENT
    free = _free_ram_gb() if free_gb is None else free_gb
    return max(1, min(MAX_WORKERS, int((free - _RAM_RESERVE_GB) // max(0.5, per_chunk))))


def _visual(item: TimelineItem) -> bool:
    return not item.track.upper().startswith("A")


def _splittable(item: TimelineItem) -> bool:
    """Whether a chunk boundary may cut through this item's own stream.

    True for:
      * a `media` item on any visual track with no children, since its only
        time-dependent behaviour is its own transform animation (Ken Burns)
        and/or colour fades -- `sub_timeline` re-anchors both across the cut
        (`Transform.anim_total_frames`/`anim_offset_frames`, and the fade
        split rule: the left piece keeps fade_in, the right keeps fade_out).
      * an `adjustment` item (kind == "adjustment") with no children, for the
        same reason -- `_apply_adjustment` windows its transform/colour the
        same way a media item's own chain does.

    False for anything else with time-dependent behaviour this module cannot
    re-anchor: a `transition` (only ever meaningful on a V1 junction, and the
    existing ±1s edge guard below is what protects it), `children` (a
    compound moves and trims as one block), or an `effects`/animated
    `atmosphere` list (named effects and generated overlays, neither
    describable as a simple before/after split).
    """
    if item.children or item.transition is not None or item.effects:
        return False
    if any(e.enabled for e in item.atmosphere):
        return False
    return item.kind == "adjustment" or (item.kind == "media" and _visual(item))


def _fade_frames(seconds: float, fps: float) -> int:
    return max(0, int(round(seconds * fps)))


def plan_chunks(timeline: Timeline, target_s: float = CHUNK_TARGET_S
                ) -> List[Tuple[int, int]]:
    """[(start_frame, end_frame)] covering the programme, cut only at safe frames."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    total = int(timeline.duration_frames)
    if total <= 0:
        return []
    blocked = bytearray(total + 1)
    # (lo, hi, label) for every unsplittable item's interior -- the usual
    # cause of a chunk that cannot shrink to the target; kept so an oversize
    # chunk can be explained in the log rather than just measured.
    blockers: List[Tuple[int, int, str]] = []

    def block(lo: int, hi: int) -> None:
        for f in range(max(0, lo), min(total, hi) + 1):
            blocked[f] = 1

    guard = int(round(fps))  # a second either side of a V1 edge (transitions)
    for item in timeline.items:
        if not item.enabled or not _visual(item):
            continue
        s, e = item.timeline_start_frame, item.timeline_end_frame
        if _splittable(item):
            block(s - guard, s + guard)
            block(e - guard, e + guard)
            # A cut inside a fade region cannot be split by the plain
            # keep-this-side/drop-that-side rule sub_timeline uses, so the
            # fade's own span -- on whichever side(s) it is set -- stays whole.
            fi = _fade_frames(item.color.fade_in, fps) if item.color else 0
            fo = _fade_frames(item.color.fade_out, fps) if item.color else 0
            if fi > 0:
                block(s, s + fi - 1)
            if fo > 0:
                block(e - fo + 1, e)
            afi = _fade_frames(item.audio_fade_in, fps)
            afo = _fade_frames(item.audio_fade_out, fps)
            if afi > 0:
                block(s, s + afi - 1)
            if afo > 0:
                block(e - afo + 1, e)
        else:
            block(s + 1, e - 1)
            blockers.append((s, e, f"{item.track}:{item.id} ({item.kind})"))
    # The intro pad and cold open belong to the first chunk.
    block(0, max(timeline.program_offset_frames, timeline.cold_open_frames) + guard)

    step, search, min_len = int(target_s * fps), int(SEARCH_S * fps), int(MIN_CHUNK_S * fps)
    chunks: List[Tuple[int, int]] = []
    a = 0
    while total - a > step + min_len:
        want = a + step
        cut = None
        for d in range(0, search + 1):
            for f in (want - d, want + d):
                if a + min_len <= f < total - min_len and not blocked[f]:
                    cut = f
                    break
            if cut is not None:
                break
        if cut is None:  # a long animated stretch: the next safe frame after it
            cut = next((f for f in range(want + search, total - min_len) if not blocked[f]), None)
        if cut is None:
            break
        chunks.append((a, cut))
        a = cut
    chunks.append((a, total))

    limit = target_s * _LONG_CHUNK_FACTOR
    for a, b in chunks:
        length_s = (b - a) / fps
        if length_s <= limit:
            continue
        names = sorted({name for lo, hi, name in blockers if lo < b and hi > a})
        logger.warning(
            "Chunked render: a %.0fs chunk is %.1fx the %.0fs target -- blocked by: %s",
            length_s, length_s / target_s, target_s, ", ".join(names) or "(nothing recorded)")
    return chunks


def sub_timeline(timeline: Timeline, a: int, b: int) -> Timeline:
    """The picture of frames [a, b) as a timeline of its own, starting at 0."""
    copy = timeline.model_copy(deep=True)
    items = []
    for item in copy.items:
        if not item.enabled or not _visual(item):
            continue
        s, e = item.timeline_start_frame, item.timeline_end_frame
        if e <= a or s >= b:
            continue
        if s < a or e > b:
            if not _splittable(item):
                raise ValueError(f"item {item.id} on {item.track} crosses a chunk boundary")
            cut_s, cut_e = max(s, a), min(e, b)
            if item.kind != "adjustment":
                item.source_start_frame += cut_s - s
                item.source_end_frame -= e - cut_e
            left_trimmed, right_trimmed = cut_s > s, cut_e < e
            if item.transform is not None and item.transform.is_animated():
                t = item.transform
                if t.anim_total_frames is None:
                    t.anim_total_frames = e - s
                t.anim_offset_frames = (t.anim_offset_frames or 0) + (cut_s - s)
            if item.color is not None:
                # The left piece (keeps the original start) keeps fade_in and
                # drops fade_out; the right piece (keeps the original end)
                # keeps fade_out and drops fade_in. A piece that is neither
                # (both trimmed) drops both -- plan_chunks never cuts inside
                # either fade's own span, so this never clips one short.
                if left_trimmed:
                    item.color.fade_in = 0.0
                if right_trimmed:
                    item.color.fade_out = 0.0
            if left_trimmed:
                item.audio_fade_in = 0.0
            if right_trimmed:
                item.audio_fade_out = 0.0
            s, e = cut_s, cut_e
        item.timeline_start_frame, item.timeline_end_frame = s - a, e - a
        items.append(item)
    copy.items = items
    # A progress bar is drawn on programme time, not chunk time: tell it where
    # this chunk starts and how long the whole programme is (render/atmosphere
    # `_progress_bar`). Per-clip copies (adjustment layers) too.
    # The text-fx effects (knockout, behind_head, focus, annotation,
    # newspaper_sweep, ...) carry their window as `extra` start_s/end_s on
    # programme time, read as between(t, start_s, end_s) (and as the matte's
    # setpts anchor). The chunk restarts t at 0, so the window moves with it.
    # An effect whose window misses this chunk is disabled. `matte_offset` is
    # a seek into the matte file, not programme time, so it stays.
    fps = timeline.fps_num / max(1, timeline.fps_den)
    off_s, span_s = a / fps, (b - a) / fps
    for effect in list(copy.effects) + [e for i in items for e in i.atmosphere]:
        if (effect.type or "").lower() == "progress_bar":
            effect.extra["_offset_s"] = float(effect.extra.get("_offset_s") or 0.0) + off_s
            effect.extra["_programme_s"] = float(effect.extra.get("_programme_s")
                                                 or timeline.duration_frames / fps)
        elif "start_s" in effect.extra or "end_s" in effect.extra:
            start = float(effect.extra.get("start_s") or 0.0)
            end = float(effect.extra.get("end_s", start) or 0.0)
            if end <= off_s or start >= off_s + span_s:
                effect.enabled = False
            effect.extra["start_s"] = start - off_s
            if "end_s" in effect.extra:
                effect.extra["end_s"] = end - off_s
    copy.duration_frames = b - a
    if a > 0:
        copy.program_offset_frames = 0
        copy.cold_open_frames = 0
    used = set()
    for item in items:
        used.add(item.source_id)
        used.update(c.source_id for c in item.children)
    copy.sources = {k: v for k, v in copy.sources.items() if k in used}
    copy.audio_master = None
    return copy


async def _run(cmd: List[str], on_progress: Optional[Callable[[float], None]] = None,
               output_path: Optional[Path] = None) -> None:
    with fit_command(cmd) as (cmd, cwd):
        await _run_fitted(cmd, cwd, on_progress, output_path)


async def _run_fitted(cmd: List[str], cwd: Optional[str],
                      on_progress: Optional[Callable[[float], None]],
                      output_path: Optional[Path] = None) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=cwd, creationflags=NO_WINDOW)
    # Not the first thing Sentinel kills under memory pressure.
    with sentinel_priority.ffmpeg(proc.pid):
        await _follow(proc, on_progress, output_path)


async def _kill_tree(proc: "asyncio.subprocess.Process") -> None:
    """Kill `proc` and anything it spawned. ffmpeg is a leaf process on every
    platform this ships on, but `/T` still catches a wrapper in front of it."""
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/T", "/F", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                creationflags=NO_WINDOW)
            await killer.wait()
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.debug("Killing stalled ffmpeg (pid %s) failed (%s); trying proc.kill()", proc.pid, e)
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _follow(proc: "asyncio.subprocess.Process",
                  on_progress: Optional[Callable[[float], None]],
                  output_path: Optional[Path] = None) -> None:
    tail: List[bytes] = []
    loop = asyncio.get_event_loop()
    state = {"sec": None, "changed": loop.time(), "stalled": False}

    def progress(sec: float) -> None:
        if state["sec"] is None or sec > state["sec"]:
            state["changed"] = loop.time()
        state["sec"] = sec
        if on_progress:
            on_progress(sec)

    async def drain() -> None:
        while True:
            chunk = await proc.stderr.read(65536)
            if not chunk:
                break
            tail.append(chunk)
            del tail[:-8]

    async def watchdog() -> None:
        """Kill ffmpeg if neither its reported progress nor its output file's
        size has moved for STALL_SECONDS -- the 578s chunk of the 25eff765
        incident ran at 0.05x realtime on one thread with _follow having no
        timeout at all, so nothing noticed it had effectively hung."""
        last_size = -1
        while True:
            await asyncio.sleep(min(_STALL_POLL_S, max(0.1, STALL_SECONDS)))
            size = -1
            if output_path is not None:
                try:
                    size = os.path.getsize(output_path)
                except OSError:
                    size = -1
            if size != last_size:
                last_size = size
                state["changed"] = loop.time()
            if loop.time() - state["changed"] >= STALL_SECONDS:
                state["stalled"] = True
                logger.warning("ffmpeg stalled (no progress for %.0fs): killing pid %s (%s)",
                               STALL_SECONDS, proc.pid, output_path)
                await _kill_tree(proc)
                return

    drainer = asyncio.create_task(drain())
    watcher = asyncio.create_task(watchdog())
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "ignore").strip()
            if text.startswith("out_time_ms="):
                try:
                    progress(int(text.split("=", 1)[1]) / 1_000_000.0)
                except ValueError:
                    pass
    finally:
        # Cancel before awaiting the drainer: a killed process closes its
        # pipes and lets drain() finish on its own, but the watchdog's own
        # sleep would otherwise outlive this chunk and keep the task alive.
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
    try:
        await asyncio.wait_for(drainer, timeout=30)
    except asyncio.TimeoutError:
        logger.warning("ffmpeg stderr drain did not finish within 30s (pid %s)", proc.pid)
        drainer.cancel()
    await proc.wait()
    if state["stalled"]:
        raise ChunkStalledError(f"ffmpeg made no progress for {STALL_SECONDS:.0f}s and was killed")
    if proc.returncode != 0:
        from render.runner import summarize_ffmpeg_error
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: "
                           f"{summarize_ffmpeg_error(b''.join(tail).decode('utf-8', 'ignore'))}")


def _work_dir_hash(timeline: Timeline, canvas: Optional[Tuple[int, int]],
                    target: Optional[Tuple[int, int]], video_flags: List[str]) -> str:
    """A stable name for this render's work directory, so a restarted job finds
    the same chunks it already rendered rather than starting over in a fresh
    tempdir -- `tempfile.mkdtemp`'s random suffix used to make resume impossible."""
    payload = {
        "timeline": timeline.model_dump(mode="json"),
        "canvas": list(canvas) if canvas else None,
        "target": list(target) if target else None,
        "video_flags": video_flags,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _cleanup_old_work_dirs(max_age_days: float = _WORK_DIR_MAX_AGE_DAYS) -> None:
    """Sweep chunks_* work directories left behind by a job that never
    finished (or whose output was never re-requested) and is unlikely to be
    resumed any more."""
    cutoff = time.time() - max_age_days * 86400
    try:
        for entry in Path(TEMP_DIR).glob("chunks_*"):
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


async def render_chunked(
    timeline: Timeline,
    output_path: str,
    canvas: Optional[Tuple[int, int]],
    target: Optional[Tuple[int, int]],
    hw_decode: bool,
    prefer_nvenc: bool,
    progress_callback: Optional[Callable[[float], None]] = None,
    workers: Optional[int] = None,
    encoder_profile: str = "final",
) -> Optional[str]:
    """Render `timeline` in parallel chunks. Returns the output path, or None
    when the programme is too short or offers no safe cut (the caller then
    renders it in one graph). Raises if a chunk or the join fails."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    total_s = timeline.duration_frames / fps
    if total_s < MIN_PROGRAMME_S:
        return None
    chunks = plan_chunks(timeline)
    if len(chunks) < 2:
        return None
    logger.info("Chunked render: %d chunks (%s s)", len(chunks),
                ", ".join(f"{(b - a) / fps:.0f}" for a, b in chunks))

    _cleanup_old_work_dirs()

    encoder = [f for f in get_encoder_flags(prefer_nvenc=prefer_nvenc, profile=encoder_profile)]
    # Video only per chunk; the audio flags and faststart belong to the final mux.
    video_flags: List[str] = []
    skip = 0
    for i, flag in enumerate(encoder):
        if skip:
            skip -= 1
            continue
        if flag in ("-c:a", "-b:a", "-ar", "-movflags"):
            skip = 1
            continue
        video_flags.append(flag)

    # A deterministic directory name (rather than tempfile.mkdtemp's random
    # suffix) so a job restarted after a crash or a Sentinel kill finds the
    # chunks it already finished instead of re-rendering the whole programme.
    work = Path(TEMP_DIR) / f"chunks_{_work_dir_hash(timeline, canvas, target, video_flags)}"
    work.mkdir(parents=True, exist_ok=True)

    done_s: Dict[int, float] = {}

    def report() -> None:
        if progress_callback:
            progress_callback(min(97.0, sum(done_s.values()) / total_s * 97.0))

    async def render_one(index: int, a: int, b: int) -> Path:
        part = sub_timeline(timeline, a, b)
        compiler = FilterGraphCompiler(part, assets_dir=work / f"text_{index}",
                                       canvas=canvas, hw_decode=hw_decode)
        inputs, graph, final_v, _final_a = compiler.compile()
        graph = audio_subgraph(graph, final_v)  # the picture's statements only
        if target and canvas != target:
            w, h = target
            graph += (f";{final_v}scale={w}:{h}:force_original_aspect_ratio=decrease,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[chunk_out]")
            final_v = "[chunk_out]"
        inputs, graph, (final_v,) = prune_inputs(inputs, graph, final_v)
        out = work / f"chunk_{index:03d}.mp4"
        frames = b - a

        graph_path = work / f"graph_{index:03d}.txt"
        done_marker = work / f"chunk_{index:03d}.done"
        previous_graph = graph_path.read_text(encoding="utf-8") if graph_path.exists() else None
        if done_marker.exists() and out.exists() and previous_graph == graph:
            logger.info("Chunked render: chunk %d already rendered, resuming", index)
            done_s[index] = frames / fps
            report()
            return out

        graph_path.write_text(graph, encoding="utf-8")
        part_out = work / f"chunk_{index:03d}.part.mp4"
        cmd = ["ffmpeg", "-y", "-hide_banner", *inputs,
               "-filter_complex_script", str(graph_path),
               "-map", final_v, *video_flags, "-an",
               "-r", f"{fps:.5f}", "-fps_mode", "cfr", "-frames:v", str(frames),
               "-progress", "pipe:1", str(part_out)]

        def on_progress(sec: float) -> None:
            done_s[index] = min(sec, frames / fps)
            report()

        await _run(cmd, on_progress, output_path=part_out)
        os.replace(part_out, out)
        done_marker.write_text("", encoding="utf-8")
        done_s[index] = frames / fps
        report()
        return out

    if workers is None:
        workers = parallel_chunks(canvas or FilterGraphCompiler(timeline)._canvas_size())
    workers = max(1, workers)
    logger.info("Chunked render: %d at a time", workers)
    gate = asyncio.Semaphore(workers)

    async def bounded(index: int, a: int, b: int) -> Path:
        async with gate:
            try:
                return await render_one(index, a, b)
            except (RuntimeError, ChunkStalledError) as e:
                last_exc: Exception = e
        # The first attempt ran alongside whatever else was in flight and
        # failed (a bad exit or the stall watchdog). Retry alone: claim every
        # permit so nothing else is competing for the same CPU/NVENC session
        # while this one gets another try -- an 8.2 GB chunk thrashing memory
        # next to three others is exactly how the 25eff765 incident's single
        # overgrown chunk took 40 minutes in the first place.
        for attempt in range(1, MAX_CHUNK_RETRIES + 1):
            logger.warning("Chunked render: chunk %d failed (attempt %d/%d): %s; retrying alone",
                           index, attempt, MAX_CHUNK_RETRIES + 1, last_exc)
            held = 0
            try:
                for _ in range(workers):
                    await gate.acquire()
                    held += 1
                return await render_one(index, a, b)
            except (RuntimeError, ChunkStalledError) as e:
                last_exc = e
            finally:
                for _ in range(held):
                    gate.release()
        raise last_exc

    audio_task = asyncio.create_task(_render_audio(timeline, work))
    try:
        parts = await asyncio.gather(*(bounded(i, a, b) for i, (a, b) in enumerate(chunks)))
        audio = await audio_task
    except Exception:
        audio_task.cancel()
        logger.warning("Chunked render failed; keeping %s for a resume", work)
        raise

    listing = work / "chunks.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c", "copy"]
    else:
        cmd += ["-map", "0:v", "-c", "copy"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", output_path]
    try:
        await _run(cmd, output_path=Path(output_path))
    except Exception:
        logger.warning("Chunked render's final concat failed; keeping %s for a resume", work)
        raise
    # Only the join of already-finished chunks happened here, so there is
    # nothing left worth resuming -- the work dir can go.
    shutil.rmtree(work, ignore_errors=True)
    if progress_callback:
        progress_callback(100.0)
    logger.info("Chunked render complete: %s", output_path)
    return output_path


async def _render_audio(timeline: Timeline, work: Path) -> Optional[Path]:
    """The whole programme's audio, alone: the same statements the single graph
    would run for it, without any of the picture.

    The mix is rendered once, before loudness, to a WAV; the two loudness
    passes then read that file and take seconds. Measuring by re-running the
    mix (noise reduction and all) took over 300 s on a 14-minute episode,
    timed out, and fell back to single-pass loudnorm."""
    master = timeline.audio_master
    wants_loudness = master is not None and master.loudness_lufs is not None
    mix_timeline = timeline
    if wants_loudness:
        mix_timeline = timeline.model_copy(deep=True)
        mix_timeline.audio_master = master.model_copy(update={"loudness_lufs": None})
    inputs, graph, _final_v, final_a = FilterGraphCompiler(
        mix_timeline, assets_dir=work / "text_audio").compile()
    if not graph:
        return None
    sub = audio_subgraph(graph, final_a)
    if not sub:
        return None
    # The whole programme's inputs include every still and B-roll clip; the
    # audio graph reads a handful of them. Passing all ~370 overflowed the
    # Windows command line (WinError 206) on a 14-minute episode.
    inputs, sub, (final_a,) = prune_inputs(inputs, sub, final_a)

    out = work / "audio.m4a"
    done = work / "audio.done"
    graph_path = work / "graph_audio.txt"
    previous_graph = graph_path.read_text(encoding="utf-8") if graph_path.exists() else None
    if done.exists() and out.exists() and previous_graph == sub:
        logger.info("Chunked render: audio mix already rendered, resuming")
        return out

    graph_path.write_text(sub, encoding="utf-8")
    mix = work / "mix.wav"
    mix_part = work / "mix.part.wav"
    await _run(["ffmpeg", "-y", "-hide_banner", *inputs,
                "-filter_complex_script", str(graph_path),
                "-map", final_a, "-vn", "-c:a", "pcm_f32le", "-ar", "48000",
                "-progress", "pipe:1", str(mix_part)], output_path=mix_part)
    os.replace(mix_part, mix)

    chain: List[str] = []
    if wants_loudness:
        from render.audio import build_loudness_chain
        measured = await asyncio.to_thread(_measure_loudness, mix, master)
        chain = build_loudness_chain(master, measured)
    out_part = work / "audio.part.m4a"
    await _run(["ffmpeg", "-y", "-hide_banner", "-i", str(mix),
                *(["-af", ",".join(chain)] if chain else []),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-progress", "pipe:1", str(out_part)],
               output_path=out_part)
    os.replace(out_part, out)
    done.write_text("", encoding="utf-8")
    return out


def _measure_loudness(mix: Path, master) -> Optional[Dict[str, float]]:
    """loudnorm's first-pass stats for a rendered mix, or None."""
    import json
    import re
    import subprocess
    target = max(-40.0, min(-5.0, float(master.loudness_lufs)))
    peak = max(-9.0, min(0.0, float(master.true_peak_db)))
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(mix), "-af",
             f"loudnorm=I={target:.1f}:TP={peak:.1f}:LRA=11:print_format=json", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, creationflags=NO_WINDOW)
        found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", (result.stderr or b"").decode("utf-8", "replace"))
        if not found:
            return None
        stats = json.loads(found.group(0))
        return {k: float(stats[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh")} | {
            "target_offset": float(stats.get("target_offset", 0.0))}
    except Exception as e:
        logger.warning("Loudness measurement of the mix failed (%s); using single-pass", e)
        return None
