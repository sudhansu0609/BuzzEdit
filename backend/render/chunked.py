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
import logging
import os
import shutil
import tempfile
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
    return (item.kind == "media" and item.track == "V1" and not item.children
            and (item.transform is None or item.transform.is_identity()))


def plan_chunks(timeline: Timeline, target_s: float = CHUNK_TARGET_S
                ) -> List[Tuple[int, int]]:
    """[(start_frame, end_frame)] covering the programme, cut only at safe frames."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    total = int(timeline.duration_frames)
    if total <= 0:
        return []
    blocked = bytearray(total + 1)

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
        else:
            block(s + 1, e - 1)
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
            item.source_start_frame += cut_s - s
            item.source_end_frame -= e - cut_e
            s, e = cut_s, cut_e
        item.timeline_start_frame, item.timeline_end_frame = s - a, e - a
        items.append(item)
    copy.items = items
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


async def _run(cmd: List[str], on_progress: Optional[Callable[[float], None]] = None) -> None:
    with fit_command(cmd) as (cmd, cwd):
        await _run_fitted(cmd, cwd, on_progress)


async def _run_fitted(cmd: List[str], cwd: Optional[str],
                      on_progress: Optional[Callable[[float], None]]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=cwd, creationflags=NO_WINDOW)
    # Not the first thing Sentinel kills under memory pressure.
    with sentinel_priority.ffmpeg(proc.pid):
        await _follow(proc, on_progress)


async def _follow(proc: "asyncio.subprocess.Process",
                  on_progress: Optional[Callable[[float], None]]) -> None:
    tail: List[bytes] = []

    async def drain() -> None:
        while True:
            chunk = await proc.stderr.read(65536)
            if not chunk:
                break
            tail.append(chunk)
            del tail[:-8]

    drainer = asyncio.create_task(drain())
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", "ignore").strip()
        if on_progress and text.startswith("out_time_ms="):
            try:
                on_progress(int(text.split("=", 1)[1]) / 1_000_000.0)
            except ValueError:
                pass
    await drainer
    await proc.wait()
    if proc.returncode != 0:
        from render.runner import summarize_ffmpeg_error
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: "
                           f"{summarize_ffmpeg_error(b''.join(tail).decode('utf-8', 'ignore'))}")


def _script(graph: str, directory: Path, name: str) -> str:
    path = directory / name
    path.write_text(graph, encoding="utf-8")
    return str(path)


async def render_chunked(
    timeline: Timeline,
    output_path: str,
    canvas: Optional[Tuple[int, int]],
    target: Optional[Tuple[int, int]],
    hw_decode: bool,
    prefer_nvenc: bool,
    progress_callback: Optional[Callable[[float], None]] = None,
    workers: Optional[int] = None,
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

    work = Path(tempfile.mkdtemp(prefix="chunks_", dir=str(TEMP_DIR)))
    encoder = [f for f in get_encoder_flags(prefer_nvenc=prefer_nvenc)]
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
        cmd = ["ffmpeg", "-y", "-hide_banner", *inputs,
               "-filter_complex_script", _script(graph, work, f"graph_{index:03d}.txt"),
               "-map", final_v, *video_flags, "-an",
               "-r", f"{fps:.5f}", "-fps_mode", "cfr", "-frames:v", str(frames),
               "-progress", "pipe:1", str(out)]

        def on_progress(sec: float) -> None:
            done_s[index] = min(sec, frames / fps)
            report()

        await _run(cmd, on_progress)
        done_s[index] = frames / fps
        report()
        return out

    if workers is None:
        workers = parallel_chunks(canvas or FilterGraphCompiler(timeline)._canvas_size())
    logger.info("Chunked render: %d at a time", workers)
    gate = asyncio.Semaphore(max(1, workers))

    async def bounded(index: int, a: int, b: int) -> Path:
        async with gate:
            return await render_one(index, a, b)

    try:
        audio_task = asyncio.create_task(_render_audio(timeline, work))
        parts = await asyncio.gather(*(bounded(i, a, b) for i, (a, b) in enumerate(chunks)))
        audio = await audio_task

        listing = work / "chunks.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(listing)]
        if audio:
            cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c", "copy"]
        else:
            cmd += ["-map", "0:v", "-c", "copy"]
        cmd += ["-movflags", "+faststart", "-progress", "pipe:1", output_path]
        await _run(cmd)
    finally:
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
    mix = work / "mix.wav"
    await _run(["ffmpeg", "-y", "-hide_banner", *inputs,
                "-filter_complex_script", _script(sub, work, "graph_audio.txt"),
                "-map", final_a, "-vn", "-c:a", "pcm_f32le", "-ar", "48000",
                "-progress", "pipe:1", str(mix)])

    chain: List[str] = []
    if wants_loudness:
        from render.audio import build_loudness_chain
        measured = await asyncio.to_thread(_measure_loudness, mix, master)
        chain = build_loudness_chain(master, measured)
    out = work / "audio.m4a"
    await _run(["ffmpeg", "-y", "-hide_banner", "-i", str(mix),
                *(["-af", ",".join(chain)] if chain else []),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-progress", "pipe:1", str(out)])
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
