import asyncio
import re
import logging
from typing import List, Callable, Optional
from timeline.schema import Timeline, frame_to_time
from render.compiler import FilterGraphCompiler
from render.encoder import get_encoder_flags

logger = logging.getLogger("render_runner")

async def render_timeline_async(
    timeline: Timeline,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    prefer_nvenc: bool = True
) -> str:
    """
    Render a Timeline EDL into a single video file using single-pass FFmpeg compilation.
    Parses -progress pipe:1 to report percentage completion.
    """
    compiler = FilterGraphCompiler(timeline)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    total_duration_sec = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
    if total_duration_sec <= 0:
        total_duration_sec = 1.0

    encoder_flags = get_encoder_flags(prefer_nvenc=prefer_nvenc)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", final_v,
        "-map", final_a,
        *encoder_flags,
        "-progress", "pipe:1",
        output_path
    ]

    logger.info(f"Executing render: {' '.join(cmd[:10])}... -> {output_path}")

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

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore")
        logger.error(f"FFmpeg render failed (code {proc.returncode}): {err_msg}")
        raise RuntimeError(f"FFmpeg render failed with exit code {proc.returncode}: {err_msg[-500:]}")

    if progress_callback:
        progress_callback(100.0)

    logger.info(f"Render completed successfully: {output_path}")
    return output_path
