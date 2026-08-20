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
from typing import Callable, Dict, List, Optional, Tuple

from .models import Asset, Beat, PresentationSettings
from . import workflows

logger = logging.getLogger("presentation.assets")

# Generate at least the canvas size: a 1280x720 image pushed into a 1080p frame
# and then zoomed is visibly soft, and Ken Burns needs headroom besides.
DEFAULT_IMAGE_SIZE = (1920, 1080)
DEFAULT_VIDEO_SIZE = (1280, 720)
DEFAULT_VIDEO_LENGTH = 81
DEFAULT_VIDEO_FPS = 16
IMAGE_TIMEOUT = 300
VIDEO_TIMEOUT = 1800

# Three beats in a row producing nothing means the *night* is broken — ComfyUI
# crashed mid-run, the workflow is wrong, a model refuses to load — and every
# further attempt costs a full timeout. Stop and say so, rather than "completing"
# hours later with zero images. One bad beat between good ones is still just one
# missing picture; the counter resets on every success.
MAX_CONSECUTIVE_FAILURES = 3

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
                          canvas_size: Optional[Tuple[int, int]] = None) -> Tuple[List[Asset], List[Dict[str, str]]]:
    """Generate what every cutaway beat needs. Returns (assets, failures).

    `canvas_size` is the media's (width, height); when given, pictures generate
    at that size so they fill the frame instead of being letterboxed or cropped.
    An explicit gen_image_size / gen_video_size setting still overrides it.
    """
    report = progress_cb or (lambda fraction, message="": None)
    wanted = [b for b in beats if b.kind in ("broll_image", "broll_video", "graphic")]
    if not wanted:
        return [], []

    resolved: Dict[str, Optional[workflows.ResolvedWorkflow]] = {}
    for role in ("broll_image", "broll_video", "graphic"):
        resolved[role] = workflows.resolve(role)

    if resolved["broll_image"] is None:
        logger.warning("No image workflow is configured; skipping asset generation")
        return [], [{"beat_id": b.id, "reason": "no image workflow configured"}
                    for b in wanted]

    from comfyui_bridge import queue_manager

    if not await asyncio.to_thread(queue_manager.client.is_connected):
        logger.warning("ComfyUI is offline; the pass will run without generated assets")
        return [], [{"beat_id": b.id, "reason": "ComfyUI offline"} for b in wanted]

    directory = _cache_dir(project_dir)
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []

    consecutive_failures = 0
    for index, beat in enumerate(wanted):
        report(index / max(1, len(wanted)), f"Generating {index + 1}/{len(wanted)}")
        try:
            asset = await _generate_one(beat, resolved, directory, queue_manager,
                                        seed_base, canvas_size)
            if asset:
                assets.append(asset)
                consecutive_failures = 0
            else:
                failures.append({"beat_id": beat.id, "reason": "no output produced"})
                consecutive_failures += 1
        except Exception as e:
            logger.warning("Beat %s (%s) failed to generate: %s", beat.id, beat.topic, e)
            failures.append({"beat_id": beat.id, "reason": str(e)[:200]})
            consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and index + 1 < len(wanted):
            remaining = wanted[index + 1:]
            logger.warning(
                "Aborting asset generation after %d consecutive failures; "
                "%d beats not attempted", consecutive_failures, len(remaining))
            failures.extend(
                {"beat_id": b.id,
                 "reason": f"not attempted: aborted after {consecutive_failures} "
                           "consecutive generation failures"}
                for b in remaining)
            break

    logger.info("Assets: %d generated (%d cached), %d failed",
                len(assets), sum(1 for a in assets if a.cache_hit), len(failures))
    return assets, failures


async def _generate_one(beat: Beat, resolved: Dict[str, Optional[workflows.ResolvedWorkflow]],
                        directory: Path, queue_manager, seed_base: int,
                        canvas_size: Optional[Tuple[int, int]] = None) -> Optional[Asset]:
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
        # Match the media's aspect/size by default so the clip fills the frame.
        width, height = tuple(_setting(
            "gen_video_size", _canvas_default(canvas_size, DEFAULT_VIDEO_SIZE)))
        length = int(_setting("gen_video_length", DEFAULT_VIDEO_LENGTH))
        fps = int(_setting("gen_video_fps", DEFAULT_VIDEO_FPS))
        timeout = int(_setting("gen_video_timeout_s", VIDEO_TIMEOUT))
    else:
        width, height = tuple(_setting(
            "gen_image_size", _canvas_default(canvas_size, DEFAULT_IMAGE_SIZE)))
        length = fps = None
        timeout = int(_setting("gen_image_timeout_s", IMAGE_TIMEOUT))

    # Seeded from the project and the beat, so a re-run reproduces the same
    # picture rather than quietly generating a different one.
    seed = (seed_base + int(hashlib.sha1(beat.id.encode()).hexdigest()[:6], 16)) % 2_000_000_000
    key = cache_key(workflow.file, positive, negative, seed, width, height,
                    extra=f"{steps}|{cfg}|{model_override}|{upscale}")

    existing = _cached(directory, key)
    if existing:
        w, h, duration = _probe(existing)
        logger.info("Beat %s: reusing cached asset %s", beat.id, existing.name)
        return Asset(beat_id=beat.id,
                     kind="video" if existing.suffix.lower() in _VIDEO_SUFFIXES else "image",
                     path=str(existing), width=w, height=h, duration_s=duration,
                     cache_hit=True, workflow_file=workflow.file, seed=seed, prompt_sha=key)

    graph = workflow.build(positive=positive, negative=negative, width=width,
                           height=height, seed=seed, length=length, fps=fps,
                           prefix=f"buzzedit_{beat.id}_{key}",
                           steps=steps, cfg=cfg, model=model_override)

    logger.info("Beat %s: generating via %s (%s)", beat.id, workflow.file, role)
    outputs = await queue_manager.submit_and_wait(graph, timeout=timeout)
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
    _write_sidecar(directory, key, beat, workflow.file, positive, negative, seed,
                   width, height)
    return Asset(beat_id=beat.id,
                 kind="video" if destination.suffix.lower() in _VIDEO_SUFFIXES else "image",
                 path=str(destination), width=w, height=h, duration_s=duration,
                 cache_hit=False, workflow_file=workflow.file, seed=seed, prompt_sha=key)


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


async def generate_thumbnail_title(program, ask) -> Optional[str]:
    """A short thumbnail title from the transcript, or None."""
    if ask is None:
        return None
    from .shotplan import THUMBNAIL_TITLE_SYSTEM
    excerpt = program.text_between(0.0, min(120.0, program.duration_s))
    if not excerpt:
        return None
    try:
        answer = await ask(THUMBNAIL_TITLE_SYSTEM, f"Transcript:\n{excerpt}\n\nTitle:")
    except Exception:
        return None
    if not answer:
        return None
    title = " ".join(str(answer).split())[:60].strip(' "\'')
    return title or None
