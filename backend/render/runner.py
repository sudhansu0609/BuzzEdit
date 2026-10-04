import asyncio
import os
import re
import logging
import tempfile
from typing import List, Callable, Optional, Tuple
from timeline.schema import Timeline, frame_to_time
from render.compiler import FilterGraphCompiler
from render.encoder import get_encoder_flags, get_nvenc_available
from config import TEMP_DIR
from utils.proc import NO_WINDOW

logger = logging.getLogger("render_runner")

_ERROR_LINE = re.compile(r"error|invalid|not found|no such|unable|failed|cannot", re.IGNORECASE)


def summarize_ffmpeg_error(stderr: str, limit: int = 900) -> str:
    """The lines of FFmpeg's stderr that say what went wrong.

    The raw tail is useless for a graph failure: FFmpeg echoes the offending
    filter statement (often hundreds of characters) right before the one line
    that names the problem, so a 500-char tail showed the drawtext around it
    and "Invalid argument" but never "Option not found" or which filter said
    so. Keep the diagnostic lines (each capped), falling back to the tail.
    """
    hits = [ln.strip()[:300] for ln in stderr.splitlines() if _ERROR_LINE.search(ln)]
    if not hits:
        return stderr[-limit:]
    text = " | ".join(dict.fromkeys(hits))  # dedupe, keep order
    return text[:limit]

def _parse_resolution(resolution: Optional[str]) -> Optional[tuple]:
    """Parse a "WIDTHxHEIGHT" string into (w, h) ints, or None if invalid."""
    if not resolution:
        return None
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX]\s*(\d{2,5})\s*", str(resolution))
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    return w, h


def _render_canvas(program: Tuple[int, int], target: Optional[Tuple[int, int]]
                   ) -> Optional[Tuple[int, int]]:
    """The size to compose at when it should be smaller than the programme's
    own: the programme scaled to fit inside `target` (even dimensions; the
    final pad still letterboxes a different aspect). None = keep the programme
    size (no target, or the target is not smaller)."""
    if not target:
        return None
    pw, ph = program
    tw, th = target
    if pw <= 0 or ph <= 0:
        return None
    f = min(tw / pw, th / ph)
    if f >= 1.0:
        return None
    w, h = max(2, int(pw * f) // 2 * 2), max(2, int(ph * f) // 2 * 2)
    return w, h


# Text styles (presets, captions, cards) are authored in pixels for a 1080p
# frame. On any other canvas they are scaled with it, or a 4K render shows them
# at half their intended size.
TEXT_DESIGN_HEIGHT = 1080
_TEXT_PIXEL_FIELDS = ("font_size", "line_spacing", "stroke_width", "shadow_x", "shadow_y",
                      "box_padding")


def _scale_text(timeline: Timeline, canvas: Tuple[int, int]) -> Timeline:
    """A copy of `timeline` whose text pixel sizes suit `canvas` (short edge
    against TEXT_DESIGN_HEIGHT). The same timeline when no scaling is needed."""
    k = min(canvas) / TEXT_DESIGN_HEIGHT
    if abs(k - 1.0) < 0.05:
        return timeline
    copy = timeline.model_copy(deep=True)

    def scale(item) -> None:
        if item.text is not None:
            style = item.text.style
            for field in _TEXT_PIXEL_FIELDS:
                value = getattr(style, field, None)
                if isinstance(value, (int, float)) and value:
                    setattr(style, field, int(round(value * k)))
        for child in item.children:
            scale(child)

    for item in copy.items:
        scale(item)
    return copy


async def render_timeline_async(
    timeline: Timeline,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    prefer_nvenc: bool = True,
    output_resolution: Optional[str] = None,
    window: Optional[Tuple[float, float]] = None,
    prerender: bool = True,
    chunked: bool = True,
) -> str:
    """
    Render a Timeline EDL into a single video file using single-pass FFmpeg compilation.
    Parses -progress pipe:1 to report percentage completion.

    `output_resolution` ("WxH") scales the final video to that size, letterboxing
    to preserve aspect ratio. When omitted, the source resolution is kept.

    `window` ((start_s, end_s) of programme time) writes only that stretch --
    the effects sample (presentation/effects_sample.py). It skips the two-pass
    loudness measurement, which would read the whole programme for a clip.
    """
    # Compose at the delivery size. A 4K recording delivered at 1080p used to
    # be composed at 4K and scaled down at the very end: every overlay, zoom
    # and grade ran on four times the pixels, each of ~230 input branches held
    # 4K frames, and a 14-minute render reached 49 GB on a 32 GB machine and
    # paged at 0.06x. Text sizes are authored for 1080p, so this also stops
    # captions and cards coming out at half size on 4K footage.
    target = _parse_resolution(output_resolution)
    canvas = _render_canvas(FilterGraphCompiler(timeline)._canvas_size(), target)
    if canvas is not None:
        logger.info("Composing at %dx%d (the delivery size) instead of the source's size", *canvas)
        timeline = timeline.model_copy(update={"width": canvas[0], "height": canvas[1]})
    timeline = _scale_text(timeline, canvas or FilterGraphCompiler(timeline)._canvas_size())

    # Ken Burns moves on stills are rendered first, in parallel on the GPU
    # encoder (render/prerender.py) -- half of a real render's time was
    # those chains running one after another inside the big graph.
    if prerender:
        from render.prerender import prerender_animated_stills
        timeline = await asyncio.to_thread(prerender_animated_stills, timeline)
    hw_decode = prefer_nvenc and get_nvenc_available()
    compiler = FilterGraphCompiler(timeline, canvas=canvas, hw_decode=hw_decode)

    # A long programme renders in parallel chunks (render/chunked.py): one
    # graph pushes every frame through every overlay of the whole video. Any
    # failure there falls back to the single graph below, as before. It does
    # its own two-pass loudness on the rendered mix, so the measuring pass
    # below only runs for the single graph.
    if window is None and chunked:
        from render.chunked import render_chunked
        try:
            done = await render_chunked(
                timeline, output_path, canvas, target, hw_decode, prefer_nvenc,
                progress_callback)
            if done:
                return done
        except Exception as e:
            logger.warning("Chunked render failed (%s); rendering in one graph", e)

    # Two-pass loudnorm: measure the pre-loudness mix first (a throwaway null-
    # muxer pass on the SAME inputs/graph, minus the final loudness stage), and
    # feed those stats into the real compile so loudnorm runs in linear mode
    # against real numbers instead of guessing frame-by-frame — that guessing
    # is what makes single-pass loudnorm audibly pump on a talky programme.
    # Only attempted when a loudness target is actually set, and any failure
    # (ffmpeg missing, a timeout, a malformed graph) falls back to the single-
    # pass chain exactly as if this block were never here.
    loudness_measured: Optional[dict] = None
    master = timeline.audio_master
    if master is not None and master.loudness_lufs is not None and window is None:
        try:
            from render.audio import measure_loudness_stats
            measure_timeline = timeline.model_copy(deep=True)
            measure_timeline.audio_master = master.model_copy(update={"loudness_lufs": None})
            m_inputs, m_graph, _, m_final_a = FilterGraphCompiler(
                measure_timeline, canvas=canvas, hw_decode=hw_decode).compile()
            if m_final_a != "[anull]":
                target_lufs = max(-40.0, min(-5.0, float(master.loudness_lufs)))
                peak = max(-9.0, min(0.0, float(master.true_peak_db)))
                loudness_measured = await asyncio.to_thread(
                    measure_loudness_stats, "ffmpeg", m_inputs, m_graph, m_final_a,
                    target_lufs, peak)
        except Exception as e:
            logger.warning("Two-pass loudness measurement failed (%s); using single-pass", e)
            loudness_measured = None

    inputs, filter_complex, final_v, final_a = compiler.compile(
        loudness_measured=loudness_measured)
    from render.cmdline import prune_inputs
    inputs, filter_complex, (final_v, final_a) = prune_inputs(
        inputs, filter_complex, final_v, final_a)

    # Optionally scale the final video stream to a target resolution, preserving
    # aspect ratio and padding to fill (letterbox/pillarbox). Nothing to do when
    # the programme was composed at exactly that size.
    if target and canvas != target:
        w, h = target
        scale_filter = (
            f"{final_v}scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[vout]"
        )
        filter_complex = f"{filter_complex};{scale_filter}" if filter_complex else scale_filter
        final_v = "[vout]"

    total_duration_sec = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
    window_args: list = []
    if window is not None:
        start_s, end_s = max(0.0, float(window[0])), float(window[1])
        window_args = ["-ss", f"{start_s:.3f}", "-t", f"{max(0.1, end_s - start_s):.3f}"]
        total_duration_sec = max(0.1, end_s - start_s)
    if total_duration_sec <= 0:
        total_duration_sec = 1.0

    encoder_flags = get_encoder_flags(prefer_nvenc=prefer_nvenc)

    # Force a constant output frame rate matching the timeline so concatenated /
    # overlaid clips from mixed-fps sources don't drift or produce VFR output.
    fps = timeline.fps_num / max(1, timeline.fps_den)

    # A long edit (dozens of jump-cuts, B-roll overlays, zooms, fades) produces a
    # filtergraph that can exceed the Windows ~32KB command-line limit, which
    # surfaces as "[WinError 206] The filename or extension is too long" the moment
    # the process is spawned. Pass the graph through a file with
    # -filter_complex_script so its size stops mattering.
    script_path: Optional[str] = None
    if filter_complex:
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            fd, script_path = tempfile.mkstemp(suffix=".txt", prefix="filtergraph_", dir=str(TEMP_DIR))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(filter_complex)
            filter_args = ["-filter_complex_script", script_path]
        except Exception as e:
            logger.warning("Could not write filter_complex script (%s); using inline arg", e)
            script_path = None
            filter_args = ["-filter_complex", filter_complex]
    else:
        filter_args = []

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        *filter_args,
        "-map", final_v,
        "-map", final_a,
        *encoder_flags,
        "-r", f"{fps:.5f}",
        "-fps_mode", "cfr",
        *window_args,
        "-progress", "pipe:1",
        output_path
    ]

    logger.info(f"Executing render: {' '.join(cmd[:10])}... -> {output_path} "
                f"(filtergraph {len(filter_complex)} chars via "
                f"{'script file' if script_path else 'inline'})")

    from render.cmdline import fit_command
    from runtime import sentinel_priority
    fitted = fit_command(cmd)
    sentinel_client = None
    try:
        cmd, cwd = fitted.__enter__()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            creationflags=NO_WINDOW
        )
        # Not the first thing Sentinel kills under memory pressure (see
        # runtime/sentinel_priority.py).
        sentinel_client = f"buzzedit-ffmpeg-{proc.pid}"
        sentinel_priority.hold(sentinel_client, proc.pid)

        pattern_time = re.compile(r"out_time_ms=(\d+)")

        # stderr has to be drained WHILE stdout is read. A chatty filter (libass
        # warning about every font it scans, a decoder complaining per frame)
        # fills the 64KB pipe long before the render ends, and FFmpeg then
        # blocks on its next write — at frame 0, for ever, with no error.
        stderr_chunks: list = []

        async def drain_stderr() -> None:
            while True:
                chunk = await proc.stderr.read(65536)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
                # Keep the tail only; a failure message is what matters.
                if len(stderr_chunks) > 64:
                    del stderr_chunks[:32]

        drain = asyncio.create_task(drain_stderr())

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="ignore").strip()
            match = pattern_time.search(decoded)
            if match and progress_callback:
                out_time_us = int(match.group(1))
                current_sec = out_time_us / 1_000_000.0
                progress_pct = min(99.0, max(0.0, (current_sec / total_duration_sec) * 100.0))
                try:
                    progress_callback(progress_pct)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        await drain
        await proc.wait()
        stderr = b"".join(stderr_chunks)
    finally:
        if sentinel_client:
            sentinel_priority.drop(sentinel_client)
        fitted.__exit__(None, None, None)
        if script_path:
            try:
                os.remove(script_path)
            except Exception:
                pass

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore")
        logger.error(f"FFmpeg render failed (code {proc.returncode}): {err_msg}")
        raise RuntimeError(f"FFmpeg render failed with exit code {proc.returncode}: {summarize_ffmpeg_error(err_msg)}")

    if progress_callback:
        progress_callback(100.0)

    logger.info(f"Render completed successfully: {output_path}")
    return output_path
