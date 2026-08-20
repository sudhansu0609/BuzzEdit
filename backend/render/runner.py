import asyncio
import os
import re
import logging
import tempfile
from typing import List, Callable, Optional
from timeline.schema import Timeline, frame_to_time
from render.compiler import FilterGraphCompiler
from render.encoder import get_encoder_flags
from config import TEMP_DIR

logger = logging.getLogger("render_runner")

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


async def render_timeline_async(
    timeline: Timeline,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    prefer_nvenc: bool = True,
    output_resolution: Optional[str] = None
) -> str:
    """
    Render a Timeline EDL into a single video file using single-pass FFmpeg compilation.
    Parses -progress pipe:1 to report percentage completion.

    `output_resolution` ("WxH") scales the final video to that size, letterboxing
    to preserve aspect ratio. When omitted, the source resolution is kept.
    """
    compiler = FilterGraphCompiler(timeline)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    # Optionally scale the final video stream to a target resolution, preserving
    # aspect ratio and padding to fill (letterbox/pillarbox).
    target = _parse_resolution(output_resolution)
    if target:
        w, h = target
        scale_filter = (
            f"{final_v}scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[vout]"
        )
        filter_complex = f"{filter_complex};{scale_filter}" if filter_complex else scale_filter
        final_v = "[vout]"

    total_duration_sec = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
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
        "-progress", "pipe:1",
        output_path
    ]

    logger.info(f"Executing render: {' '.join(cmd[:10])}... -> {output_path} "
                f"(filtergraph {len(filter_complex)} chars via "
                f"{'script file' if script_path else 'inline'})")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        pattern_time = re.compile(r"out_time_ms=(\d+)")

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

        stdout, stderr = await proc.communicate()
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except Exception:
                pass

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore")
        logger.error(f"FFmpeg render failed (code {proc.returncode}): {err_msg}")
        raise RuntimeError(f"FFmpeg render failed with exit code {proc.returncode}: {err_msg[-500:]}")

    if progress_callback:
        progress_callback(100.0)

    logger.info(f"Render completed successfully: {output_path}")
    return output_path
