"""M6 2.5D parallax (MOTION_GRAPHICS_PLAN.md) -- depth + the ``Parallax``
builder.

Unlike every other motion kind this is not a ``BUILDERS`` entry: there is no
``[parallax: ...]`` directive, no beat kind of its own. ``apply_parallax`` runs
once, after every still already exists (generated B-roll, drawn maps/cards),
right before ``placement.place_broll`` (see ``director.py``'s hook) and turns a
*share* of the already-placed stills into a small-camera-dolly render over a
depth-displaced plane, in place -- the Asset for that beat is swapped for the
render; every other beat's Asset is untouched.

Depth comes from Depth Anything V2 Small (Apache-2.0), run as ONNX
(onnxruntime, CPU -- never a GPU slot that ComfyUI might want) and downloaded
once into BuzzEdit's own ``data/models/`` (see ``eyecontact/landmarks.py`` for
the same lazy-download-and-verify shape this mirrors). A still whose raw depth
has too little contrast (a close, flat framing -- see ``depth_contrast_ok``) is
left alone, and so is any still a render fails for -- the existing Ken Burns
still is always still sitting there in ``assets``, since this only ever swaps
an entry, never removes one.
"""

import base64
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from config import DATA_DIR

from .models import Asset, Beat, PresentationSettings
from .motion import MotionContext, _beat_duration

logger = logging.getLogger("presentation.motion_parallax")

# --- the depth model ---------------------------------------------------------

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "depth_anything_v2_small.onnx"
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
# Apache-2.0 (depth-anything/Depth-Anything-V2-Small, ONNX export by
# onnx-community -- checked 2026-10-28). Pinned by sha256 so a corrupt or
# swapped download is never silently used.
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
MODEL_MIN_BYTES = 90_000_000  # the real file is ~94.5 MB; well short of that is a bad download

# DPTImageProcessor's own preprocessing (preprocessor_config.json on the model
# repo): resize keeping aspect ratio to ~518 on the axis closer to it already,
# both dimensions rounded to a multiple of 14, then ImageNet normalise.
_DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DEPTH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEPTH_TARGET = 518
DEPTH_MULTIPLE = 14

# Raw (pre-normalisation) depth std below this reads as "no real depth contrast"
# -- a close, flat framing rather than a scene with real foreground/background
# separation. Measured on 4 real Raat3Baje/Life3Baje stills 2026-10-28 (a
# corridor, a held phone against a blurred corridor, a mountain-top silhouette,
# a noir diary/building composite): raw std 1.19-3.00. Set comfortably below
# that range; see tests/test_motion_parallax.py::test_depth_contrast_ok.
DEPTH_RAW_STD_MIN = 1.0

# The displacement grid handed to the Parallax composition as a prop (see
# Parallax.tsx) -- (PARALLAX_GRID+1)^2 vertices.
PARALLAX_GRID = 48
# Mirrors Parallax.tsx's own defaults; kept here too so a cache key changes if
# either side's constant ever does (PARALLAX_VERSION below is the real guard).
PARALLAX_DOLLY_PERCENT = 4
PARALLAX_DRIFT_PERCENT = 2
# Bumped whenever this module's job shape changes in a way that should
# invalidate every cached parallax render.
PARALLAX_VERSION = 1

_session = None


def _verify_model(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MODEL_MIN_BYTES:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256


def ensure_model() -> Path:
    """Downloads the depth model once, verifies it by size and sha256, and
    never touches the network again once it is on disk. Never called at
    import time -- only the first still a render actually wants to parallax
    pays for this."""
    if _verify_model(MODEL_PATH):
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the Depth Anything V2 Small model (once, ~95 MB)")
    import urllib.request
    tmp = MODEL_PATH.with_suffix(".tmp")
    with urllib.request.urlopen(MODEL_URL, timeout=180) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    if not _verify_model(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Depth model download did not verify ({len(data)} bytes)")
    tmp.replace(MODEL_PATH)
    logger.info("Depth model ready at %s", MODEL_PATH)
    return MODEL_PATH


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        ensure_model()
        # CPU only -- ComfyUI may be holding the GPU (GPU_MODEL_POLICY); ~1s
        # per still at 518px is fine for the handful this ever runs on.
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def reset_session_cache() -> None:
    """Tests call this between cases; _get_session otherwise memoises per process."""
    global _session
    _session = None


# --- depth inference ----------------------------------------------------------

def _constrain_to_multiple(value: float, multiple: int, min_value: int = 0) -> int:
    x = round(value / multiple) * multiple
    if x < min_value:
        x += multiple
    return int(x)


def _resize_dims(height: int, width: int, target: int = DEPTH_TARGET,
                 multiple: int = DEPTH_MULTIPLE) -> Tuple[int, int]:
    """DPTImageProcessor.resize's own rule: scale whichever axis is already
    closer to `target` by that axis's own ratio, apply the same scale to both
    (keeps aspect), round each to a multiple of 14."""
    scale_h = target / height
    scale_w = target / width
    if abs(scale_w - 1) < abs(scale_h - 1):
        scale_h = scale_w
    else:
        scale_w = scale_h
    new_h = _constrain_to_multiple(scale_h * height, multiple, multiple)
    new_w = _constrain_to_multiple(scale_w * width, multiple, multiple)
    return new_h, new_w


def infer_depth(image: Image.Image) -> np.ndarray:
    """The model's raw per-pixel depth prediction, resized back to `image`'s
    own size (bilinear). Higher value = nearer the camera -- the model's own
    convention (checked against a real corridor still: the near doorway comes
    out bright, the dark far end comes out dark); Parallax.tsx relies on this
    directly."""
    rgb = image.convert("RGB")
    w0, h0 = rgb.size
    new_h, new_w = _resize_dims(h0, w0)
    resized = rgb.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - _DEPTH_MEAN) / _DEPTH_STD
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
    session = _get_session()
    out = session.run(None, {"pixel_values": arr})[0]
    depth = out[0].astype(np.float32)
    # Resize the (small, resized-input-sized) prediction back up to the
    # still's own resolution for the cached PNG and the displacement grid.
    depth_img = Image.fromarray(depth, mode="F").resize((w0, h0), Image.BILINEAR)
    return np.asarray(depth_img, dtype=np.float32)


def depth_contrast_ok(depth: np.ndarray) -> bool:
    """True when `depth` (the raw, pre-normalisation prediction) has real
    foreground/background separation rather than reading as a single close,
    flat plane."""
    return float(depth.std()) >= DEPTH_RAW_STD_MIN


def _normalized_u8(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(depth.min()), float(depth.max())
    norm = (depth - dmin) / max(1e-6, dmax - dmin)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _still_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def depth_map_for(still_path: Path) -> Optional[Path]:
    """The cached depth-map PNG for `still_path` -- grayscale, normalised
    0-255, saved next to the still and named by the still's own content hash
    (MOTION_GRAPHICS_PLAN.md M6). Only ever written for a still that passes
    ``depth_contrast_ok``; a flat/low-contrast still, or any inference
    failure, returns None (the still's cheap enough to re-check next time
    rather than caching a negative result). Never raises."""
    try:
        key = _still_hash(still_path)
        cache_path = still_path.parent / f"{key}.depth.png"
        if cache_path.exists():
            return cache_path
        with Image.open(still_path) as img:
            depth = infer_depth(img)
        if not depth_contrast_ok(depth):
            logger.info("Parallax: %s has too little depth contrast (std=%.2f); kept as a still",
                       still_path.name, float(depth.std()))
            return None
        gray = _normalized_u8(depth)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.png")
        Image.fromarray(gray, mode="L").save(tmp)
        tmp.replace(cache_path)
        return cache_path
    except Exception as e:  # noqa: BLE001 - a depth failure must never fail the still
        logger.warning("Depth map for %s failed: %s", still_path.name, e)
        return None


def _depth_grid(depth_png: Path, grid: int = PARALLAX_GRID) -> List[float]:
    """A flat (grid+1)^2 list, 0..1, for Parallax.tsx's `depthGrid` prop.
    Rows are flipped (PNG row 0 = top of the image -> list row 0 = bottom of
    the image) to match three.js's PlaneGeometry vertex order -- see the
    matching comment in Parallax.tsx."""
    with Image.open(depth_png) as img:
        small = img.convert("L").resize((grid + 1, grid + 1), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = arr[::-1, :]
    return [round(float(v), 4) for v in arr.flatten().tolist()]


def _data_uri(still_path: Path) -> Optional[str]:
    """The still as a `data:` URI -- it lives outside motion/public/, so it
    travels in the job's props rather than through `staticFile` (the way
    MotionMap's world outline does)."""
    try:
        with Image.open(still_path) as img:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001
        logger.warning("Parallax image data for %s failed: %s", still_path.name, e)
        return None


# --- selection ----------------------------------------------------------------

def _eligible(asset: Asset, beat: Optional[Beat]) -> bool:
    """Only a freshly generated still (never a drawn map/card/chart, a stock
    photo, or an already-animated clip), whose beat does not show the main
    character (a displaced face reads as a mistake, not an effect)."""
    if asset.kind != "image":
        return False
    if getattr(asset, "source", "generated") != "generated":
        return False
    if beat is None or beat.shows_character:
        return False
    if not Path(asset.path).exists():
        return False
    return True


def _select(candidates: Sequence[Asset], share: float, seed: int) -> List[Asset]:
    """A deterministic, seeded subset of `candidates`, capped at `share` of
    them -- same seed, same stills in, same selection out (so a re-render
    reuses the same parallax cache instead of picking a new share each time).
    """
    if not candidates or share <= 0:
        return []
    count = round(len(candidates) * min(1.0, share))
    if count <= 0:
        return []

    def _rank(asset: Asset) -> str:
        return hashlib.sha1(f"{seed}:parallax:{asset.beat_id}".encode()).hexdigest()

    ranked = sorted(candidates, key=_rank)
    return ranked[:count]


# --- job building ---------------------------------------------------------

def _build_spec(asset: Asset, beat: Beat, ctx: MotionContext, depth_png: Path) -> Optional[Dict[str, Any]]:
    image_uri = _data_uri(Path(asset.path))
    if image_uri is None:
        return None
    depth_grid = _depth_grid(depth_png)
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    seed_val = int(hashlib.sha1(beat.id.encode()).hexdigest()[:8], 16)
    return {
        "composition": "Parallax",
        "props": {
            "imageSrc": image_uri,
            "depthGrid": depth_grid,
            "gridSize": PARALLAX_GRID,
            "seed": seed_val,
            "dollyPercent": PARALLAX_DOLLY_PERCENT,
            "driftPercent": PARALLAX_DRIFT_PERCENT,
            "theme": ctx.theme,
        },
        "width": ctx.width,
        "height": ctx.height,
        "fps": ctx.fps,
        "frames": frames,
        "transparent": False,
    }


def _asset_for(old: Asset, duration_s: float, out: Path, ctx: MotionContext, cache_hit: bool) -> Asset:
    return Asset(beat_id=old.beat_id, kind="video", path=str(out), width=int(ctx.width),
                height=int(ctx.height), duration_s=duration_s, cache_hit=cache_hit, source="motion")


# --- entry point ---------------------------------------------------------

def apply_parallax(assets: List[Asset], beats: Sequence[Beat], project_dir: Path,
                   width: int, height: int, genre: str, settings: PresentationSettings,
                   seed: int = 0) -> Tuple[List[Asset], Dict[str, Any]]:
    """Swaps a share of the already-generated stills in `assets` for a
    Parallax render, in place -- called right before `placement.place_broll`
    (director.py), after every still (generated + drawn maps/cards) already
    exists. Every still this does not touch -- not selected, too little depth
    contrast, a failed render, the engine not installed -- is left exactly as
    it was, so it still gets the ordinary Ken Burns move placement already
    gives a still (MOTION_GRAPHICS_PLAN.md M6). Never raises.
    """
    stats: Dict[str, Any] = {"candidates": 0, "selected": 0, "rendered": 0,
                             "skipped_contrast": 0, "failed": 0}
    share = float(getattr(settings, "parallax_share", 0.0) or 0.0)
    if not settings.motion_graphics or share <= 0:
        return list(assets), stats
    from . import motion as motion_stage
    if not motion_stage.available():
        return list(assets), stats

    try:
        by_beat = {b.id: b for b in beats}
        candidates = [a for a in assets if _eligible(a, by_beat.get(a.beat_id))]
        stats["candidates"] = len(candidates)
        chosen = _select(candidates, share, seed)
        stats["selected"] = len(chosen)
        if not chosen:
            return list(assets), stats

        from runtime.master_control import wait_if_paused

        theme = motion_stage.theme_for(genre, settings.art_style)
        ctx = MotionContext(width=width, height=height, fps=motion_stage.DEFAULT_FPS, genre=genre,
                            theme=theme, settings=settings)
        directory = Path(project_dir) / "assets" / "motion"
        directory.mkdir(parents=True, exist_ok=True)

        out_assets = list(assets)
        pending: List[Dict[str, Any]] = []
        by_job_id: Dict[str, Tuple[Asset, float, Path]] = {}
        for asset in chosen:
            beat = by_beat.get(asset.beat_id)
            depth_png = depth_map_for(Path(asset.path))
            if depth_png is None:
                stats["skipped_contrast"] += 1
                continue
            spec = _build_spec(asset, beat, ctx, depth_png)
            if spec is None:
                stats["failed"] += 1
                continue
            key = motion_stage._cache_key(spec)
            out = directory / f"parallax_{key}.mp4"
            duration_s = spec["frames"] / max(1.0, float(spec["fps"]))
            if out.exists():
                out_assets = [_asset_for(asset, duration_s, out, ctx, cache_hit=True)
                             if a is asset else a for a in out_assets]
                stats["rendered"] += 1
                continue
            job_id = f"parallax_{asset.beat_id}"
            job = {"id": job_id, "composition": spec["composition"], "props": spec["props"],
                  "width": spec["width"], "height": spec["height"], "fps": spec["fps"],
                  "frames": spec["frames"], "out": str(out), "transparent": False,
                  "concurrency": 1, "gl": "angle"}
            pending.append(job)
            by_job_id[job_id] = (asset, duration_s, out)

        if not pending:
            return out_assets, stats

        concurrency = max(1, int(getattr(settings, "motion_concurrency", 4) or 4))
        for start in range(0, len(pending), concurrency):
            batch = pending[start:start + concurrency]
            wait_if_paused("motion_render")
            results = motion_stage._run_batch(batch, project_dir)
            for job in batch:
                asset, duration_s, out = by_job_id[job["id"]]
                result = results.get(job["id"])
                if result and result.get("ok") and out.exists():
                    out_assets = [_asset_for(asset, duration_s, out, ctx, cache_hit=False)
                                 if a is asset else a for a in out_assets]
                    stats["rendered"] += 1
                else:
                    reason = (result or {}).get("error") or "render failed"
                    logger.warning("Parallax render failed for beat %s: %s", asset.beat_id, reason)
                    stats["failed"] += 1
        return out_assets, stats
    except Exception as e:  # noqa: BLE001 - a parallax failure must never fail the pass
        logger.warning("apply_parallax failed; every still kept as-is: %s", e)
        return list(assets), stats
