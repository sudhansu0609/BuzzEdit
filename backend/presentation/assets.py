"""Stage C2 — turning prompts into files, through the user's ComfyUI.

Two rules govern everything here.

**A failure is one missing picture, never a failed night.** Every beat is
generated inside its own try/except; a workflow that errors, a model that will
not load, a timeout — all of them cost that beat and nothing else. The pass
carries on and the report says which beats went missing and why.

**Never generate the same thing twice.** Assets are keyed by the hash of what
produced them, so re-running a project after tweaking the zoom settings reuses
every image instead of spending twenty minutes on the GPU again.
"""

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from runtime import gpu_handover

from .models import BROLL_SECONDS_CAP, Asset, Beat, Character, PresentationSettings
from . import character as character_mod
from . import workflows

logger = logging.getLogger("presentation.assets")

# Generate at least the canvas size: a 1280x720 image pushed into a 1080p frame
# and then zoomed is visibly soft, and Ken Burns needs headroom besides.
DEFAULT_IMAGE_SIZE = (1920, 1080)
DEFAULT_VIDEO_SIZE = (1280, 720)
DEFAULT_VIDEO_LENGTH = 81
# Wan's native rate: what clip length is counted in. A workflow that
# interpolates (RIFE x2) saves at a multiple of it -- see ResolvedWorkflow.build.
DEFAULT_VIDEO_FPS = 16
IMAGE_TIMEOUT = 300
# A clip that has not finished in 15 minutes on this card is stuck (a healthy
# 3s Wan clip takes ~2); the old 30-minute limit let one bad clip eat half an
# hour before the still fallback kicked in.
VIDEO_TIMEOUT = 900

# Measured on the RTX 5060 Ti 16GB (ComfyUI log, 2026-09-26): Wan 2.2 14B Q4
# samples at ~37 s/step for 81 frames at 832x480, so frames are the cost. Clips
# are cut to the beat (2-4s on screen), so generating a fixed 5s (81 frames)
# threw ~40% of every clip's sampling away. Wan wants 4n+1 frames.
# The ceiling follows BROLL_SECONDS_CAP (8 s -> 133 frames at 16 fps); a
# clip that long costs ~1.6x an 81-frame one, so only a long max asks for it.
MIN_VIDEO_FRAMES = 33
MAX_VIDEO_FRAMES = ((int(round((BROLL_SECONDS_CAP + 0.3) * DEFAULT_VIDEO_FPS)) - 1) // 4) * 4 + 1

# Z-Image Turbo cost scales with pixels: 1920x1080 (2.1MP) spent ~25s of each
# ~35s image sampling. ~1.4MP samples in about half that, and the 1.2x Lanczos
# step up to 1080p at composite time is not visible on a 2-4s cutaway.
MAX_IMAGE_GEN_PIXELS = 1600 * 896


def _video_length_for(seconds: float, fps: int) -> int:
    """Frames for a clip that covers `seconds` on screen, plus a little slack
    for the cut, snapped to Wan's 4n+1 and clamped."""
    frames = int(round((max(0.0, seconds) + 0.3) * max(1, fps)))
    frames = ((max(frames, MIN_VIDEO_FRAMES) - 1 + 3) // 4) * 4 + 1
    cap = ((int(round((BROLL_SECONDS_CAP + 0.3) * max(1, fps))) - 1) // 4) * 4 + 1
    return min(cap, frames)


def _image_gen_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """`size`'s aspect at no more than MAX_IMAGE_GEN_PIXELS, snapped."""
    width, height = size
    if width * height <= MAX_IMAGE_GEN_PIXELS:
        return size
    scale = (MAX_IMAGE_GEN_PIXELS / (width * height)) ** 0.5
    return (_snap(int(width * scale), 16), _snap(int(height * scale), 16))

# Three beats in a row producing nothing is when the pass stops to ask whether
# ComfyUI is still there. Offline ends it; online (each failure has already been
# cleared off ComfyUI's queue and VRAM) gets up to twice this before the
# workflow itself is blamed. Aborting on three outright cost a real run 52 of
# 67 pictures to what was only an LLM squatting on the GPU. For video beats it
# is also how many failed clips switch the rest of the pass to stills.
MAX_CONSECUTIVE_FAILURES = 3

# What each phase needs free before it starts (see runtime/gpu_handover.py).
# Wan 2.2 14B Q4 loads ~8.5 GB per expert plus the ~6.4 GB UMT5 encoder; on
# less than ~11 GB free it spills into shared memory and a 2-minute clip takes
# over 15. The GGUF experts are staged through system RAM, hence the RAM bar.
VIDEO_PHASE_MIN_VRAM_MB = 11000.0
VIDEO_PHASE_MIN_RAM_MB = 10000.0
# The image phase is advisory: stills still run on a tight card, only slower,
# so its hand-over is for the eviction, and its verdict is only reported.
IMAGE_PHASE_MIN_VRAM_MB = 9000.0
IMAGE_PHASE_MIN_RAM_MB = 4000.0
# Wall-clock ceiling for the whole video phase; clips past it keep their stills.
# A healthy clip is ~2.3-2.6 min on the reference 16 GB card (measured), so
# this is ~18-20 clips -- the most important first (see the phase loop).
# `gen_video_budget_s` raises it for a video that wants more motion.
VIDEO_PHASE_BUDGET_S = 45 * 60
# The ceiling grows with the clips the plan asks for, so a video that wants
# 70% motion gets the time its clips need instead of losing most of them to
# stills (Buzzcaf Studio holds the runtime share it was asked for). Generous
# against the measured ~2.3-3.5 min per clip; a starved card is still caught
# by the per-clip timeout and the consecutive-failure guard.
VIDEO_PHASE_SECONDS_PER_CLIP = 6 * 60

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".gif", ".mov", ".mkv"}


def _setting(key: str, default):
    try:
        from store.app_settings import AppSettings
        value = AppSettings().get(key)
        return value if value not in (None, "") else default
    except Exception:
        return default


def cache_key(workflow_file: str, positive: str, negative: str,
              seed: int, width: int, height: int, extra: str = "") -> str:
    raw = f"{workflow_file}|{positive}|{negative}|{seed}|{width}x{height}|{extra}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _snap(value: int, multiple: int = 8) -> int:
    """Round a dimension down to a multiple ComfyUI's latent nodes accept.

    A source cropped to an odd size (1918x1080, say) would otherwise hand
    EmptyLatentImage a width it rejects. Never go below one tile.
    """
    return max(multiple, (int(value) // multiple) * multiple)


def _canvas_default(canvas_size: Optional[Tuple[int, int]],
                    fallback: Tuple[int, int]) -> Tuple[int, int]:
    """The media's own dimensions, snapped, so generated pictures match the
    frame they are cut into rather than being letterboxed or cropped. Falls back
    to the fixed default when the media size is unknown."""
    if canvas_size and all(canvas_size):
        return (_snap(canvas_size[0]), _snap(canvas_size[1]))
    return fallback


# Video diffusion cost scales with pixels × frames. ~0.4MP (832x480-class) is
# what the 14B Q4 model renders in ~2.5 min on a 16GB card; generating at a full
# 1080p canvas instead takes ~5x longer and blows the per-clip timeout. The HQ
# workflow (zimage_wan22_i2v_hq.json, not active) was built for 1024x576 and
# measured ~7 min a clip here -- docs/GENERATION_UPGRADE_PLAN.md.
MAX_VIDEO_GEN_PIXELS = 832 * 480


# Every delivered clip is full HD (the creator's rule): generated at the size
# above, then upscaled -- RealESRGAN in the HQ workflow, an ffmpeg Lanczos fit
# (_fit_clip) for any graph that delivers another size. A larger canvas (4K)
# stays at 1080p-class: 2.1MP x 161 float frames is already ~4 GB of RAM.
MAX_VIDEO_OUTPUT_PIXELS = 1920 * 1080
# A cut-in clip is re-encoded once more by the final render, so encode the
# fitted file near-lossless (same as render/prerender.py).
CLIP_FIT_NVENC_CQ = 16
CLIP_FIT_X264_CRF = 14


def _video_output_size(canvas_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """The delivered clip size: the media's aspect at 1920x1080-class pixels
    (1080x1920 for a vertical short). An explicit gen_video_output_size
    setting wins."""
    explicit = _setting("gen_video_output_size", None)
    if explicit:
        return tuple(int(v) for v in explicit)
    width, height = canvas_size if canvas_size and all(canvas_size) else (1920, 1080)
    scale = (MAX_VIDEO_OUTPUT_PIXELS / (width * height)) ** 0.5
    return (_snap(int(round(width * scale)), 2), _snap(int(round(height * scale)), 2))


def _fit_clip(path: Path, width: int, height: int) -> bool:
    """Re-encode a clip to exactly width x height (fill, centre crop) when the
    workflow delivered another size. GPU encode when NVENC is there."""
    from config import FFMPEG_BIN
    from render.encoder import get_nvenc_available
    from utils.proc import NO_WINDOW
    tmp = path.with_suffix(".fit.mp4")
    codec = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(CLIP_FIT_NVENC_CQ), "-b:v", "0"]
             if get_nvenc_available() else ["-c:v", "libx264", "-preset", "medium", "-crf", str(CLIP_FIT_X264_CRF)])
    chain = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
             f"crop={width}:{height},setsar=1,format=yuv420p")
    cmd = [FFMPEG_BIN, "-y", "-v", "error", "-i", str(path), "-vf", chain, *codec,
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)]
    try:
        import subprocess
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=300, creationflags=NO_WINDOW)
        tmp.replace(path)
        return True
    except Exception as e:
        logger.warning("Could not fit %s to %dx%d: %s", path.name, width, height, e)
        tmp.unlink(missing_ok=True)
        return False


def _video_canvas_default(canvas_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """The media's aspect at a generation-sized resolution, snapped for video
    latents. An explicit gen_video_size setting bypasses this entirely."""
    width, height = _canvas_default(canvas_size, DEFAULT_VIDEO_SIZE)
    pixels = width * height
    if pixels > MAX_VIDEO_GEN_PIXELS:
        scale = (MAX_VIDEO_GEN_PIXELS / pixels) ** 0.5
        width, height = width * scale, height * scale
    return (_snap(int(width), 16), _snap(int(height), 16))


def _upscale_image(path: Path, factor: float) -> None:
    """Enlarge a generated still by `factor` with a high-quality Lanczos resample.

    A resize, not an AI super-resolution pass — for that, add an upscale-model
    node to the ComfyUI workflow itself. This just gives Ken Burns more pixels to
    push into so a 1024px generation doesn't go soft on a 1080p/4K canvas.
    """
    if factor <= 1.0:
        return
    try:
        from PIL import Image
        with Image.open(path) as image:
            w, h = int(image.width * factor), int(image.height * factor)
            image.resize((w, h), Image.LANCZOS).save(path)
    except Exception as e:
        logger.warning("Upscale of %s failed (%s); keeping the original size", path.name, e)


def _cache_dir(project_dir: Path) -> Path:
    path = Path(project_dir) / "assets" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached(directory: Path, key: str) -> Optional[Path]:
    for candidate in directory.glob(f"{key}.*"):
        if candidate.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES:
            return candidate
    return None


def _probe(path: Path) -> Tuple[int, int, float]:
    """(width, height, duration) for a generated file; zeros if unreadable."""
    try:
        from utils.ffmpeg_utils import get_video_info
        info = get_video_info(str(path))
        return (int(info.get("width") or 0), int(info.get("height") or 0),
                float(info.get("duration") or 0.0))
    except Exception:
        try:
            from PIL import Image
            with Image.open(path) as image:
                return image.width, image.height, 0.0
        except Exception:
            return 0, 0, 0.0


async def generate_assets(beats: List[Beat], project_dir: Path,
                          settings: PresentationSettings,
                          progress_cb: Optional[Callable[[float, str], None]] = None,
                          seed_base: int = 0,
                          canvas_size: Optional[Tuple[int, int]] = None,
                          stats: Optional[Dict[str, Any]] = None,
                          character: Optional[Character] = None) -> Tuple[List[Asset], List[Dict[str, str]]]:
    """Generate what every cutaway beat needs. Returns (assets, failures).

    `character`, the story's lead (ShotPlan.character): every beat tagged
    `shows_character` gets that one face swapped into its still, so it and the
    clip made from it show the same person. Other beats are never swapped.

    `stats`, when given, is filled with how each phase went (the memory
    hand-over verdicts, clips made / kept as stills and why) for the report.

    `canvas_size` is the media's (width, height); when given, pictures generate
    at that size so they fill the frame instead of being letterboxed or cropped.
    An explicit gen_image_size / gen_video_size setting still overrides it.
    """
    report = progress_cb or (lambda fraction, message="": None)
    wanted = [b for b in beats if b.kind in ("broll_image", "broll_video", "graphic", "canvas_card")
              or (b.kind == "source_quote" and b.image_prompt)]
    if not wanted:
        return [], []

    resolved: Dict[str, Optional[workflows.ResolvedWorkflow]] = {}
    for role in ("broll_image", "broll_video", "graphic"):
        resolved[role] = workflows.resolve(role)

    if resolved["broll_image"] is None:
        logger.warning("No image workflow is configured; skipping asset generation")
        failures = [{"beat_id": b.id, "reason": "no image workflow configured"}
                   for b in wanted]
        return await _fill_stock_fallback(wanted, project_dir, settings, failures, canvas_size)

    from comfyui_bridge import queue_manager

    face: Optional[_Face] = None
    if (character is not None and settings.character_consistency
            and any(b.shows_character for b in wanted)):
        face = _Face(key=_face_key(character, seed_base))

    if not await asyncio.to_thread(queue_manager.client.is_connected):
        # Pictures an earlier run already made are on disk: a retry after a
        # crash must still use them rather than render with none at all.
        directory = _cache_dir(project_dir)
        cached: List[Asset] = []
        failures = []
        for b in wanted:
            found = None
            for candidate in ([b, b.model_copy(update={"kind": "broll_image"})]
                              if b.kind == "broll_video" else [b]):
                try:
                    found = await _generate_one(candidate, resolved, directory, queue_manager,
                                                seed_base, canvas_size, cache_only=True,
                                                face=face if b.shows_character else None)
                except Exception as e:
                    logger.debug("Cache lookup for beat %s failed: %s", b.id, e)
                if found:
                    found.beat_id = b.id
                    break
            if found:
                cached.append(found)
            else:
                failures.append({"beat_id": b.id, "reason": "ComfyUI offline"})
        logger.warning("ComfyUI is offline; using %d cached picture(s), %d beat(s) without one",
                       len(cached), len(failures))
        missing = [b for b in wanted if any(f["beat_id"] == b.id for f in failures)]
        more, failures = await _fill_stock_fallback(missing, project_dir, settings, failures, canvas_size)
        return cached + list(more), failures

    directory = _cache_dir(project_dir)
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []

    # Phase 1 -- stills for EVERY beat, video beats included: they take
    # seconds on a clear card, and every beat then has a picture whatever the
    # video phase manages. Phase 2 (below) upgrades video beats to clips.
    video_beats = [b for b in wanted if b.kind == "broll_video"]
    stills_by_beat: Dict[str, Asset] = {}

    async def attempt(b: Beat, start_image: Optional[str] = None) -> Tuple[Optional[Asset], str]:
        try:
            made = await _generate_one(b, resolved, directory, queue_manager,
                                       seed_base, canvas_size, start_image=start_image,
                                       face=face if b.shows_character else None)
            return made, "" if made else "no output produced"
        except Exception as e:
            logger.warning("Beat %s (%s) failed to generate: %s", b.id, b.topic, e)
            return None, str(e)[:200]

    stats = stats if stats is not None else {}
    image_ready = await gpu_handover.prepare_for_phase(
        "images", need_vram_mb=IMAGE_PHASE_MIN_VRAM_MB, need_ram_mb=IMAGE_PHASE_MIN_RAM_MB)
    stats["image_phase"] = image_ready

    if face is not None:
        # The lead's reference face, made once (cached per project) before any
        # still that shows them.
        character_stats = {"name": character.name, "description": character.description,
                           "beats": sum(1 for b in wanted if b.shows_character)}
        stats["character"] = character_stats
        face = await _prepare_face(character, face, resolved["broll_image"], directory,
                                   queue_manager, seed_base, character_stats)

    consecutive_failures = 0
    total = len(wanted) + len(video_beats)
    for index, beat in enumerate(wanted):
        report(index / max(1, total), f"Generating picture {index + 1}/{len(wanted)}")
        as_still = beat.model_copy(update={"kind": "broll_image"}) if beat.kind == "broll_video" else beat
        asset, reason = await attempt(as_still)
        if asset:
            asset.beat_id = beat.id
            assets.append(asset)
            stills_by_beat[beat.id] = asset
            consecutive_failures = 0
            continue
        failures.append({"beat_id": beat.id, "reason": reason})
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and index + 1 < len(wanted):
            # Only a ComfyUI that is actually gone ends the pass early. A live
            # one that failed a few beats has been cleared and gets another go,
            # up to twice the limit before the workflow itself is suspect.
            online = await asyncio.to_thread(queue_manager.client.is_connected)
            if online and consecutive_failures < 2 * MAX_CONSECUTIVE_FAILURES:
                continue
            remaining = wanted[index + 1:]
            why = "ComfyUI offline" if not online else f"{consecutive_failures} consecutive failures"
            logger.warning("Aborting asset generation (%s); %d beats not attempted", why, len(remaining))
            failures.extend(
                {"beat_id": b.id, "reason": f"not attempted: aborted ({why})"}
                for b in remaining)
            video_beats = []
            break

    # Phase 2 -- the video clips, on a card handed over for them alone.
    video_workflow = resolved.get("broll_video")
    video_stats: Dict[str, Any] = {"planned": len(video_beats), "made": 0, "kept_still": 0,
                                   "skipped_reason": ""}
    stats["video_phase"] = video_stats
    if video_beats and (video_workflow is None or video_workflow.output_kind != "video"):
        video_stats["skipped_reason"] = "no video workflow configured"
        video_beats = []
    if video_beats:
        ready = await gpu_handover.prepare_for_phase(
            "video", need_vram_mb=float(_setting("gen_video_min_vram_mb", VIDEO_PHASE_MIN_VRAM_MB)),
            need_ram_mb=float(_setting("gen_video_min_ram_mb", VIDEO_PHASE_MIN_RAM_MB)))
        video_stats["handover"] = ready
        if not ready["ready"]:
            video_stats["skipped_reason"] = f"not enough memory for video: {ready['reason']}"
            logger.warning("Video phase skipped (%s); %d video beats keep their stills",
                           ready["reason"], len(video_beats))
            video_beats = []

    budget_s = max(float(_setting("gen_video_budget_s", VIDEO_PHASE_BUDGET_S)),
                   len(video_beats) * float(_setting("gen_video_seconds_per_clip", VIDEO_PHASE_SECONDS_PER_CLIP)))
    video_stats["budget_s"] = round(budget_s)
    # Most important shots first, so a budget that runs out leaves the
    # least important ones as stills rather than whatever came last.
    video_beats = sorted(video_beats, key=lambda b: (-float(b.priority or 0.0), b.start_s))
    phase_started = asyncio.get_running_loop().time()
    video_failures = 0
    for number, beat in enumerate(video_beats):
        report((len(wanted) + number) / max(1, total), f"Generating video clip {number + 1}/{len(video_beats)}")
        if asyncio.get_running_loop().time() - phase_started > budget_s:
            video_stats["skipped_reason"] = (f"video budget of {budget_s / 60:.0f} min used; "
                                             f"{len(video_beats) - number} clip(s) kept as stills")
            logger.warning("Video phase budget spent; the remaining %d video beats keep their stills",
                           len(video_beats) - number)
            break
        # Something may have loaded a language model onto the card since the
        # hand-over (a Studio agent on a local LLM, say): eject it before the
        # clip. Only LM Studio -- ComfyUI's own resident Wan experts are what
        # make the next clip fast, and freeing them would reload ~17 GB of
        # weights per clip.
        if await asyncio.to_thread(gpu_handover.lm_studio_has_models):
            logger.info("LM Studio loaded a model mid-phase; ejecting it before the next clip")
            await gpu_handover.eject_lm_studio()
        still = stills_by_beat.get(beat.id)
        clip, reason = await attempt(beat, start_image=still.path if still else None)
        fallback = video_workflow.fallback if video_workflow is not None else None
        if (clip is None or clip.kind != "video") and "timed out" in reason and not video_stats.get("timeout_retried"):
            # One timeout can be a model swapped in behind our back: free the
            # card again and give this clip one more go before giving up.
            video_stats["timeout_retried"] = True
            logger.warning("Beat %s: clip timed out; freeing the GPU and retrying once", beat.id)
            await gpu_handover.prepare_for_phase(
                "video", need_vram_mb=float(_setting("gen_video_min_vram_mb", VIDEO_PHASE_MIN_VRAM_MB)),
                need_ram_mb=float(_setting("gen_video_min_ram_mb", VIDEO_PHASE_MIN_RAM_MB)))
            clip, reason = await attempt(beat, start_image=still.path if still else None)
        elif (clip is None or clip.kind != "video") and "timed out" not in reason and fallback is None:
            # A one-off failure (a bad sample, a transient ComfyUI error) gets
            # a second try before the shot is downgraded to its still.
            video_stats["retried"] = video_stats.get("retried", 0) + 1
            clip, reason = await attempt(beat, start_image=still.path if still else None)
        if (clip is None or clip.kind != "video") and fallback is not None and "timed out" not in reason:
            # The upgraded graph failed on something the older one does not
            # need (a node or model not installed yet): finish the phase on
            # the older workflow rather than losing every clip to stills.
            logger.warning("Video workflow %s failed (%s); switching to %s for this pass",
                           video_workflow.file, reason[:160], fallback.file)
            video_stats["fallback"] = {"from": video_workflow.file, "to": fallback.file,
                                       "reason": reason[:200]}
            video_workflow = resolved["broll_video"] = fallback
            clip, reason = await attempt(beat, start_image=still.path if still else None)
        if clip is not None and clip.kind == "video":
            if still is not None:
                assets.remove(still)
            assets.append(clip)
            video_stats["made"] += 1
            video_failures = 0
            continue
        video_failures += 1
        logger.info("Beat %s: video failed (%s); keeping its still", beat.id, reason)
        if "timed out" in reason:
            # A clip that ran out the clock means the card is starved, not that
            # this prompt was unlucky -- every later clip would do the same.
            video_stats["skipped_reason"] = f"a clip timed out ({reason[:120]}); remaining clips kept as stills"
            logger.warning("Video clip timed out; the remaining video beats keep their stills")
            break
        if video_failures >= MAX_CONSECUTIVE_FAILURES:
            video_stats["skipped_reason"] = f"{video_failures} clips failed in a row ({reason[:120]})"
            logger.warning("Video generation failed %d times; remaining video beats keep their stills",
                           video_failures)
            break
    video_stats["kept_still"] = video_stats["planned"] - video_stats["made"]

    # Colour looks on stills, and canvas cards framed on their canvas
    # (presentation/graphics.py) -- both before anything is placed.
    from .graphics import post_process_generated
    frame_w, frame_h = canvas_size if canvas_size and all(canvas_size) else (1920, 1080)
    assets = post_process_generated(assets, wanted, frame_w, frame_h)

    logger.info("Assets: %d generated (%d cached), %d failed",
                len(assets), sum(1 for a in assets if a.cache_hit), len(failures))
    stock_assets, failures = await _fill_stock_fallback(
        wanted, project_dir, settings, failures, canvas_size)
    return assets + stock_assets, failures


async def _fill_stock_fallback(wanted: List[Beat], project_dir: Path,
                               settings: PresentationSettings,
                               failures: List[Dict[str, str]],
                               canvas_size: Optional[Tuple[int, int]]
                               ) -> Tuple[List[Asset], List[Dict[str, str]]]:
    """Free stock (Pexels/Pixabay) for whichever beats above still have no
    asset — only when the project opted in and a key is configured. A no-op,
    without so much as an import, otherwise. See presentation.stock."""
    if not failures:
        return [], failures
    from . import stock as stock_fallback
    if not stock_fallback.stock_available(settings):
        return [], failures
    new_assets, still_failed, stock_used = await stock_fallback.fill_missing(
        wanted, project_dir, settings, failures, canvas_size)
    if stock_used:
        logger.info("Stock fallback filled %d/%d missing beat(s) from free stock",
                   len(stock_used), len(failures))
    return new_assets, still_failed


async def _generate_one(beat: Beat, resolved: Dict[str, Optional[workflows.ResolvedWorkflow]],
                        directory: Path, queue_manager, seed_base: int,
                        canvas_size: Optional[Tuple[int, int]] = None,
                        start_image: Optional[str] = None,
                        cache_only: bool = False,
                        face: Optional["_Face"] = None) -> Optional[Asset]:
    """One beat's asset: from the cache when this exact prompt was made
    before, otherwise generated. `cache_only` never generates (ComfyUI down).
    `face`, for a beat that shows the main character: a still gets that face
    swapped in (a clip only keys its cache on it -- it starts from the
    swapped still)."""
    role = "graphic" if beat.kind == "graphic" else (
        "broll_video" if beat.kind == "broll_video" else "broll_image")
    workflow = resolved.get(role)
    # A video beat with no video workflow becomes a still rather than nothing:
    # the topic still deserves a picture.
    if workflow is None or (role == "broll_video" and workflow.output_kind != "video"):
        workflow = resolved["broll_image"]
        role = "broll_image"
    if workflow is None:
        return None

    is_video = workflow.output_kind == "video"
    positive = beat.prompt() or beat.image_prompt or beat.topic
    if not positive:
        return None
    negative = beat.negative_prompt or ""

    # Optional generation overrides from app settings; None/blank keeps whatever
    # the workflow itself specifies.
    steps_s = _setting("gen_steps", None)
    cfg_s = _setting("gen_cfg", None)
    steps = int(steps_s) if steps_s not in (None, "") else None
    cfg = float(cfg_s) if cfg_s not in (None, "") else None
    model_override = (_setting("gen_broll_model", None)
                      if role in ("broll_image", "broll_video") else None) or None
    try:
        upscale = float(_setting("gen_upscale", 1.0) or 1.0)
    except (TypeError, ValueError):
        upscale = 1.0

    if is_video:
        # Match the media's aspect by default, capped to a generation-sized
        # resolution so a clip takes minutes, not the whole night.
        width, height = tuple(_setting(
            "gen_video_size", _video_canvas_default(canvas_size)))
        fps = int(_setting("gen_video_fps", DEFAULT_VIDEO_FPS))
        explicit_length = _setting("gen_video_length", None)
        length = (int(explicit_length) if explicit_length
                  else _video_length_for(beat.planned_duration_s or (beat.end_s - beat.start_s), fps))
        timeout = int(_setting("gen_video_timeout_s", VIDEO_TIMEOUT))
        out_width, out_height = _video_output_size(canvas_size)
    else:
        width, height = tuple(_setting(
            "gen_image_size", _image_gen_size(_canvas_default(canvas_size, DEFAULT_IMAGE_SIZE))))
        length = fps = out_width = out_height = None
        timeout = int(_setting("gen_image_timeout_s", IMAGE_TIMEOUT))

    # Seeded from the project and the beat, so a re-run reproduces the same
    # picture rather than quietly generating a different one.
    seed = (seed_base + int(hashlib.sha1(beat.id.encode()).hexdigest()[:6], 16)) % 2_000_000_000
    key = cache_key(workflow.file, positive, negative, seed, width, height,
                    extra=f"{steps}|{cfg}|{model_override}|{upscale}"
                          + (f"|out{out_width}x{out_height}" if is_video else "")
                          + (f"|face:{face.key}" if face is not None else ""))

    existing = _cached(directory, key)
    if existing:
        w, h, duration = _probe(existing)
        logger.info("Beat %s: reusing cached asset %s", beat.id, existing.name)
        return Asset(beat_id=beat.id,
                     kind="video" if existing.suffix.lower() in _VIDEO_SUFFIXES else "image",
                     path=str(existing), width=w, height=h, duration_s=duration,
                     cache_hit=True, workflow_file=workflow.file, seed=seed, prompt_sha=key)
    if cache_only:
        return None

    graph = workflow.build(positive=positive, negative=negative, width=width,
                           height=height, seed=seed, length=length, fps=fps,
                           prefix=f"buzzedit_{beat.id}_{key}",
                           steps=steps, cfg=cfg, model=model_override,
                           out_width=out_width, out_height=out_height)
    if is_video and start_image:
        graph = await _start_from_still(graph, start_image, queue_manager)
    swapped = face is not None and face.image is not None and not is_video
    if swapped:
        graph = character_mod.with_face_swap(graph, face.image, face.gender)

    logger.info("Beat %s: generating via %s (%s%s)", beat.id, workflow.file, role,
                ", main character's face" if swapped else "")
    try:
        outputs = await queue_manager.submit_and_wait(graph, timeout=timeout)
    except Exception as e:
        if not swapped or "timed out" in str(e):
            raise
        outputs = []
        logger.warning("Beat %s: the face swap failed (%s); making the still without it", beat.id, e)
    if not outputs and swapped:
        # A picture without the face beats no picture; cached under its own key.
        return await _generate_one(beat, resolved, directory, queue_manager, seed_base,
                                   canvas_size, start_image=start_image)
    if not outputs:
        return None

    # Prefer a file of the kind we asked for: a video workflow often saves a
    # preview PNG alongside the clip, and picking that would silently turn a
    # moving shot into a still.
    wanted_suffixes = _VIDEO_SUFFIXES if is_video else _IMAGE_SUFFIXES
    chosen = next((o for o in outputs if Path(o).suffix.lower() in wanted_suffixes),
                  outputs[0])

    destination = directory / f"{key}{Path(chosen).suffix.lower()}"
    try:
        shutil.copy2(chosen, destination)
    except Exception as e:
        logger.warning("Could not copy %s into the project: %s", chosen, e)
        destination = Path(chosen)

    if not is_video and upscale > 1.0:
        _upscale_image(destination, upscale)

    w, h, duration = _probe(destination)
    if is_video and destination.suffix.lower() in _VIDEO_SUFFIXES and w and (w, h) != (out_width, out_height):
        # A graph without its own upscale (the fallback, say) still delivers
        # a full-size clip.
        logger.info("Beat %s: clip came back %dx%d; fitting to %dx%d", beat.id, w, h, out_width, out_height)
        if await asyncio.to_thread(_fit_clip, destination, out_width, out_height):
            w, h, duration = _probe(destination)
    _write_sidecar(directory, key, beat, workflow.file, positive, negative, seed,
                   width, height)
    return Asset(beat_id=beat.id,
                 kind="video" if destination.suffix.lower() in _VIDEO_SUFFIXES else "image",
                 path=str(destination), width=w, height=h, duration_s=duration,
                 cache_hit=False, workflow_file=workflow.file, seed=seed, prompt_sha=key)


class _Face:
    """The main character's face for this pass: `key` names it in cache keys
    (known without ComfyUI), `image` is the reference's file name in ComfyUI's
    input folder once uploaded (None = do not swap)."""

    def __init__(self, key: str, image: Optional[str] = None, path: Optional[str] = None,
                 gender: str = ""):
        self.key = key
        self.image = image
        self.path = path
        self.gender = gender


CHARACTER_PORTRAIT_SIZE = (1024, 1024)


def _face_key(character: Character, seed_base: int) -> str:
    return hashlib.sha1(f"{character.description}|{character.gender}|{seed_base}"
                        .encode("utf-8")).hexdigest()[:12]


async def _prepare_face(character: Character, face: "_Face",
                        workflow: Optional[workflows.ResolvedWorkflow], directory: Path,
                        queue_manager, seed_base: int, stats: Dict[str, Any]) -> Optional["_Face"]:
    """Make (or reuse) the reference portrait and hand it to ComfyUI. None, with
    the reason in `stats`, when the swap cannot run -- the beats then get plain
    stills, as before."""
    client = queue_manager.client
    if not await asyncio.to_thread(client.has_node, character_mod.FACE_SWAP_CLASS):
        stats["skipped_reason"] = f"{character_mod.FACE_SWAP_CLASS} is not installed in ComfyUI"
        logger.warning("Main character: %s; stills keep their own faces", stats["skipped_reason"])
        return None
    if workflow is None:
        stats["skipped_reason"] = "no image workflow"
        return None
    width, height = CHARACTER_PORTRAIT_SIZE
    seed = int(face.key[:8], 16) % 2_000_000_000
    prompt = character.portrait_prompt()
    key = cache_key(workflow.file, prompt, "", seed, width, height, extra="character_reference")
    path = _cached(directory, key)
    if path is None:
        try:
            outputs = await queue_manager.submit_and_wait(
                workflow.build(positive=prompt, negative="", width=width, height=height, seed=seed,
                               prefix=f"buzzedit_character_{key}"),
                timeout=int(_setting("gen_image_timeout_s", IMAGE_TIMEOUT)))
            chosen = next((o for o in outputs or [] if Path(o).suffix.lower() in _IMAGE_SUFFIXES), None)
            if chosen is None:
                raise RuntimeError("no image produced")
            path = directory / f"{key}{Path(chosen).suffix.lower()}"
            shutil.copy2(chosen, path)
        except Exception as e:
            stats["skipped_reason"] = f"reference portrait failed: {str(e)[:160]}"
            logger.warning("Main character: %s; stills keep their own faces", stats["skipped_reason"])
            return None
    try:
        image = await asyncio.to_thread(client.upload_image, str(path))
    except Exception as e:
        stats["skipped_reason"] = f"could not upload the reference: {str(e)[:160]}"
        logger.warning("Main character: %s", stats["skipped_reason"])
        return None
    stats["reference"] = str(path)
    logger.info("Main character %s: reference face %s", character.name, path.name)
    return _Face(key=face.key, image=image, path=str(path), gender=character.gender)


_I2V_CLASSES = ("WanImageToVideo", "WanFirstLastFrameToVideo", "WanVaceToVideo")


async def _start_from_still(graph: Dict[str, Any], still_path: str, queue_manager) -> Dict[str, Any]:
    """Point the i2v node's `start_image` at the phase-1 still (uploaded to
    ComfyUI) instead of the graph's own first-frame generator.

    ComfyUI only runs nodes an output depends on, so once nothing reads the
    in-graph image generator it never loads -- the video phase then holds Wan
    alone rather than Wan plus the image model, and the clip opens on exactly
    the picture that stood in for it. Any failure leaves the graph unchanged.
    """
    target = next((node_id for node_id, node in graph.items()
                   if isinstance(node, dict) and node.get("class_type") in _I2V_CLASSES
                   and isinstance((node.get("inputs") or {}).get("start_image"), list)), None)
    if target is None:
        return graph
    try:
        name = await asyncio.to_thread(queue_manager.client.upload_image, still_path)
    except Exception as e:
        logger.info("Could not upload %s as a start frame (%s); the graph makes its own", still_path, e)
        return graph
    rewired = json.loads(json.dumps(graph))
    loader = "bz_start_image"
    rewired[loader] = {"class_type": "LoadImage", "inputs": {"image": name}}
    rewired[target]["inputs"]["start_image"] = [loader, 0]
    return rewired


def _write_sidecar(directory: Path, key: str, beat: Beat, workflow_file: str,
                   positive: str, negative: str, seed: int,
                   width: int, height: int) -> None:
    """Record what produced this asset, so the cache can be audited."""
    try:
        (directory / f"{key}.json").write_text(json.dumps({
            "beat_id": beat.id,
            "topic": beat.topic,
            "kind": beat.kind,
            "workflow": workflow_file,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "width": width,
            "height": height,
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


def _thumbnail_excerpt(program, topics=()) -> str:
    """The opening, the middle and the ending of the transcript plus the topic
    list -- the first two minutes alone describe the hook, not what the video
    is about or where it lands."""
    total = max(0.0, program.duration_s)
    parts = [("Opening", 0.0, min(75.0, total))]
    if total > 150.0:
        mid = total / 2.0
        parts.append(("Middle", mid - 30.0, mid + 30.0))
    if total > 90.0:
        parts.append(("Ending", max(0.0, total - 60.0), total))
    lines = []
    for label, a, b in parts:
        text = program.text_between(a, b)
        if text:
            lines.append(f"{label}: {text}")
    names = [t.heading or t.topic for t in topics or () if (t.heading or t.topic)]
    if names:
        lines.append("Topics: " + "; ".join(names[:12]))
    return "\n\n".join(lines)


async def generate_thumbnail_concept(program, ask, topics=()) -> Optional[Dict[str, str]]:
    """{"title", "scene"} for the thumbnail, read from the whole video, or None."""
    if ask is None:
        return None
    from .shotplan import THUMBNAIL_CONCEPT_SYSTEM
    excerpt = _thumbnail_excerpt(program, topics)
    if not excerpt:
        return None
    try:
        answer = await ask(THUMBNAIL_CONCEPT_SYSTEM, f"{excerpt}\n\nAnswer:")
    except Exception:
        return None
    if not answer:
        return None
    concept: Dict[str, str] = {}
    for line in str(answer).splitlines():
        key, _, value = line.partition(":")
        key = key.strip().strip("*# ").lower()
        if key in ("title", "scene") and value.strip():
            concept[key] = " ".join(value.split()).strip(' "\'*')
    if "title" in concept:
        concept["title"] = concept["title"][:60].strip()
    elif str(answer).strip() and "scene" not in concept:
        # A model that ignored the format and just wrote a title.
        concept["title"] = " ".join(str(answer).split())[:60].strip(' "\'')
    return concept or None


async def generate_thumbnail_title(program, ask) -> Optional[str]:
    """A short thumbnail title from the transcript, or None."""
    concept = await generate_thumbnail_concept(program, ask)
    return (concept or {}).get("title") or None
