"""Pre-rendering animated stills (Ken Burns moves) in parallel, on the GPU encoder.

A still with an animated zoom is the costliest chain in a presentation render:
zoompan runs on a 3x-supersampled frame (5760x3240 for 1080p) so the move
glides instead of stepping. Inside the one big render filtergraph those chains
run one after another -- measured on a real 8-minute edit, the 86 B-roll moves
were half the render time.

Here each move is rendered on its own, with exactly the chain the compiler
would have built (`render.effects.build_overlay_transform` plus the compiler's
own still retiming), many at a time, encoded with NVENC at a near-lossless
constant quality. The timeline copy that gets rendered then points those
items at the clips with no transform, so the main render only overlays them.
Colour grade and chroma key stay on the item and are applied in the main
render in the same order as before (geometry first).

Clips are cached by everything that shapes them, so a re-render after a
caption or sound change reuses them all.
"""

import concurrent.futures
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from config import FFMPEG_BIN, TEMP_DIR
from timeline.schema import SourceFile, Timeline, TimelineItem
from utils.proc import NO_WINDOW

from .effects import build_overlay_transform

logger = logging.getLogger("render.prerender")

CACHE_DIR = Path(TEMP_DIR) / "kenburns"
# Encoder quality for the intermediate clips: NVENC constant-quality 16 is
# visually lossless for a generated still moving on screen for 2-5 s (the
# final render then encodes at 21, as it always did).
NVENC_CQ = 16
X264_CRF = 14
# Parallel jobs. Consumer NVIDIA cards run up to 8 NVENC sessions; each job's
# zoompan is single-threaded, so 6 at once uses the CPU the one big graph
# left idle without starving the final render of encoder sessions.
MAX_WORKERS = 6


def _is_animated_still(item: TimelineItem, timeline: Timeline) -> bool:
    if item.kind != "media" or not item.enabled or item.track == "V1" or not item.source_id:
        return False
    if not item.track.upper().startswith("V"):
        return False
    source = timeline.sources.get(item.source_id)
    t = item.transform
    is_image = source is not None and ((source.kind or "") == "image" or
                                       Path(source.path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    return is_image and t is not None and t.scale_end is not None and t.scale_end != t.scale


def _job(item: TimelineItem, source_path: str, canvas: Tuple[int, int], fps: float
         ) -> Tuple[List[str], Path, int]:
    frames = max(1, item.duration_frames)
    chain, _x, _y = build_overlay_transform(item.transform, canvas[0], canvas[1], frames, fps,
                                            0.0, frames / fps)
    # The compiler's still retiming: zoompan expands the one input frame.
    chain = [f.replace(":d=1:", f":d={frames}:") if f.startswith("zoompan=") else f for f in chain]
    key = hashlib.sha1(json.dumps({
        "src": source_path, "mtime": os.path.getmtime(source_path) if os.path.exists(source_path) else 0,
        "t": item.transform.model_dump(), "frames": frames, "canvas": canvas, "fps": fps,
        "chain": chain}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return chain, CACHE_DIR / f"kb_{key}.mp4", frames


def _encode(chain: List[str], source_path: str, out: Path, frames: int, fps: float,
            nvenc: bool) -> bool:
    if out.exists() and out.stat().st_size > 1000:
        return True
    tmp = out.with_suffix(".part.mp4")
    codec = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(NVENC_CQ), "-b:v", "0"]
             if nvenc else ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(X264_CRF)])
    cmd = [FFMPEG_BIN, "-y", "-v", "error", "-i", source_path,
           "-vf", ",".join(chain + ["format=yuv420p"]),
           "-frames:v", str(frames), "-r", f"{fps:.5f}", *codec, "-pix_fmt", "yuv420p", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600, creationflags=NO_WINDOW)
        os.replace(tmp, out)
        return True
    except Exception as e:
        detail = getattr(e, "stderr", b"") or b""
        logger.debug("Ken Burns pre-render (%s) failed: %s %s", "nvenc" if nvenc else "x264", e,
                     detail[-300:] if isinstance(detail, bytes) else detail)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def prerender_animated_stills(timeline: Timeline, workers: Optional[int] = None) -> Timeline:
    """A copy of `timeline` whose animated stills point at pre-rendered clips.
    Anything that fails to pre-render is left exactly as it was."""
    items = [i for i in timeline.items if _is_animated_still(i, timeline)]
    if not items:
        return timeline
    from render.encoder import get_nvenc_available
    nvenc = get_nvenc_available()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fps = timeline.fps_num / max(1, timeline.fps_den)
    canvas = (timeline.width or 1920, timeline.height or 1080)
    copy = timeline.model_copy(deep=True)
    by_id = {i.id: i for i in copy.items}
    jobs = []
    for item in items:
        source = timeline.sources[item.source_id]
        chain, out, frames = _job(item, source.path, canvas, fps)
        jobs.append((item.id, chain, source.path, out, frames))

    def run(job) -> Tuple[str, Optional[Path], int]:
        item_id, chain, path, out, frames = job
        ok = _encode(chain, path, out, frames, fps, nvenc) or (nvenc and _encode(chain, path, out, frames, fps, False))
        return item_id, (out if ok else None), frames

    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers or MAX_WORKERS) as pool:
        for item_id, out, frames in pool.map(run, jobs):
            if out is None:
                continue
            item = by_id[item_id]
            source_id = f"src_kb_{out.stem}"
            copy.sources[source_id] = SourceFile(
                id=source_id, path=str(out), duration_seconds=frames / fps,
                width=canvas[0], height=canvas[1], fps_num=copy.fps_num, fps_den=copy.fps_den,
                has_audio=False, kind="video")
            item.source_id = source_id
            item.source_start_frame = 0
            item.source_end_frame = frames
            item.transform = None
            count += 1
    logger.info("Pre-rendered %d of %d Ken Burns moves (%s)", count, len(items),
                "NVENC" if nvenc else "x264")
    return copy
