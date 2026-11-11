"""M5 paper-cutout collage (MOTION_GRAPHICS_PLAN.md) -- the cutout model, the
`expand_collages`/`annotate_collage_assets`/`drop_collage_subs` beat plumbing,
and the ``PaperCollage`` builder.

A `[collage: prompt A ; prompt B ; ... | caption=... | seconds=N]` directive
(presentation/script.py::directive_beats) becomes one `paper_collage` beat
carrying 2-4 prompts. Unlike every other motion kind, that beat needs
pictures of its own before the Remotion render can run at all:

1. ``expand_collages`` (called right after `composite.expand_splits`,
   director.py, before anything counts assets) keeps the `paper_collage` beat
   on the timeline and appends one zero-length `broll_image` sub-beat per
   prompt, each tagged `data["collage_sub"]`. `assets_stage.generate_assets`
   makes a real ComfyUI picture for each exactly like any other B-roll beat
   (mirrors `canvas_card`: generate first, frame/compose after -- see
   graphics.py's own docstring).
2. ``annotate_collage_assets`` (called right after that generation pass)
   stashes each sub-beat's resulting file path on the parent beat's own
   `data["sub_asset_paths"]`, so `build_paper_collage` below and
   `motion.render_fallback`'s `paper_collage` branch can both find them
   without threading the `generated` list through `MotionContext`.
3. ``drop_collage_subs`` (called right before `placement.place_broll`) drops
   the sub-beats' own assets -- their picture is cut out *inside* the
   collage, never its own full-frame cutaway.
4. ``build_paper_collage`` (a normal `BUILDERS` entry) turns those files into
   the `PaperCollage` composition's props: each cut out of its background
   with a small, permissively-licensed ONNX matting model (IS-Net
   general-use, Apache-2.0 training code / MIT ONNX export via rembg --
   checked 2026-10-28) run on CPU, or left as a plain rectangular photo
   (`cutout: False`) when the mask is missing or too small/large to read as
   a clean subject -- PaperCollage.tsx draws that as a polaroid instead of a
   die-cut sticker.

Every function here is defensive the way motion_parallax.py's depth model is:
a download failure, a bad mask, an unreadable file -- none of it may ever
fail the render. The worst case is `build_paper_collage` returning None (the
beat falls back to `motion.render_fallback`'s plain still, or is dropped).
"""

import base64
import hashlib
import io
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from config import DATA_DIR

from .models import Asset, Beat, PresentationSettings
from .motion import MotionContext, _beat_duration

logger = logging.getLogger("presentation.motion_collage")

# --- the cutout model ---------------------------------------------------------

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "isnet-general-use.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
# IS-Net (xuebinqin/DIS, "Highly Accurate Dichotomous Image Segmentation") is
# Apache-2.0; this ONNX export is redistributed by rembg (MIT) -- checked
# 2026-10-28. Pinned by sha256 (computed from the real download, not quoted
# from memory) so a corrupt or swapped file is never silently used.
MODEL_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
MODEL_MIN_BYTES = 170_000_000  # the real file is ~170.4 MiB; well short of that is a bad download

# A still whose subject covers less than this share of the frame (a stray
# object in an otherwise empty scene) or more than this share (the mask
# missed the background entirely -- usually a close, flat framing ISNet
# cannot separate) reads as a bad cutout; PaperCollage.tsx draws it as a
# plain polaroid instead (MOTION_GRAPHICS_PLAN.md M5).
MASK_MIN_RATIO = 0.08
MASK_MAX_RATIO = 0.85
# How far past the mask's own bounding box the cutout is cropped, as a
# fraction of the longer side -- enough that a soft edge is not clipped.
CUTOUT_CROP_PAD = 0.04
# A cutout/polaroid is shown a few hundred px wide on screen (PaperCollage.tsx
# fits it inside its own slot box); a raw ~1500-1900px B-roll still is far
# more pixel data than that needs, and a real render timed out at 1920x1080
# shipping it uncapped (several MB of base64 image props, each frame's
# drop-shadow filter working that large) -- measured on real stills
# 2026-10-28. Downscaled once here, before it is ever cached or filtered.
MAX_SIDE = 820
# ~8px of white outline at the typical on-screen display size, baked once
# into the cached PNG on this (downscaled) resolution rather than redrawn by
# Chromium's drop-shadow filter every frame -- the fix for that timeout.
OUTLINE_PX = 12

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
    """Downloads the IS-Net matting model once, verifies it by size and
    sha256, and never touches the network again once it is on disk. Never
    called at import time -- only the first still a collage actually wants
    cut out pays for this."""
    if _verify_model(MODEL_PATH):
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the IS-Net general-use cutout model (once, ~170 MB)")
    import urllib.request
    tmp = MODEL_PATH.with_suffix(".tmp")
    with urllib.request.urlopen(MODEL_URL, timeout=180) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    if not _verify_model(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cutout model download did not verify ({len(data)} bytes)")
    tmp.replace(MODEL_PATH)
    logger.info("Cutout model ready at %s", MODEL_PATH)
    return MODEL_PATH


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        ensure_model()
        # CPU only -- ComfyUI may be holding the GPU (GPU_MODEL_POLICY); a
        # 1024x1024 IS-Net pass is fine for the 2-4 stills one collage uses.
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def reset_session_cache() -> None:
    """Tests call this between cases; _get_session otherwise memoises per process."""
    global _session
    _session = None


# --- matting --------------------------------------------------------------------

def _isnet_input(img: Image.Image) -> np.ndarray:
    """rembg's own `DisSession.predict` preprocessing: resize to 1024x1024,
    scale by the resized image's own max (not a fixed 255), then (x-0.5)/1.0
    per channel, CHW, batched."""
    im = img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = (arr - 0.5) / 1.0
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return arr


def infer_mask(img: Image.Image) -> Optional[Image.Image]:
    """The soft matte for `img` (0-255 "L" image, same size as `img`), or
    None on any failure -- a missing/unverified model, a bad onnxruntime
    session, a malformed image. Never raises."""
    try:
        session = _get_session()
        tensor = _isnet_input(img)
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: tensor})[0]
        pred = np.squeeze(out[:, 0, :, :])
        mn, mx = float(pred.min()), float(pred.max())
        norm = (pred - mn) / max(1e-6, mx - mn)
        mask = Image.fromarray((norm * 255.0).astype("uint8"), mode="L")
        return mask.resize(img.size, Image.Resampling.LANCZOS)
    except Exception as e:  # noqa: BLE001 - a matting failure must never fail the render
        logger.warning("Collage mask inference failed: %s", e)
        return None


def _sharpen_mask(mask: Image.Image, low: int = 60, high: int = 180) -> Image.Image:
    """A levels stretch on the raw matte: below `low` -> 0, above `high` ->
    255, linear between. ISNet's raw mask is mostly a clean 0/255 split with
    a real edge, but also a wide, noisy mid-grey band (confirmed by a real
    histogram 2026-10-28 -- roughly a third of a busy still's non-zero
    pixels) that reads as a faint, haloed "ghost" of nearby clutter once
    composited -- a real render found this on a studio mic shot (a blurred
    camera corner ghosted in at ~30% alpha). This pushes that band decisively
    toward transparent or opaque while keeping the true edge's own
    anti-aliasing (the narrow ramp right around it)."""
    arr = np.asarray(mask, dtype=np.float32)
    arr = np.clip((arr - low) / max(1, (high - low)), 0.0, 1.0) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _mask_coverage(mask: Image.Image) -> float:
    arr = np.asarray(mask, dtype=np.uint8)
    return float(np.count_nonzero(arr >= 127)) / float(arr.size)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """`binary` (0/1 uint8) with every interior hole filled -- a border flood
    fill marks every background pixel actually reachable from the image edge;
    whatever is left unreached is enclosed by foreground and filled back in."""
    import cv2
    h, w = binary.shape
    flood = (binary * 255).astype(np.uint8)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or((binary * 255).astype(np.uint8), holes)
    return (filled > 0).astype(np.uint8)


def _clean_mask(mask: Image.Image) -> Image.Image:
    """Drops every mask fragment but the subject itself. A lone blob of
    background clutter ISNet mis-reads as foreground -- far from the real
    subject -- inflates the crop bbox below to cover both, so the subject
    renders tiny and the stray blob renders as what looks like its own
    separate item once pasted onto the paper (found by a real render + frame
    inspection, MOTION_GRAPHICS_PLAN.md M5 review). Keeps the largest
    connected component plus any other whose area is at least 15% of the
    mask's own total foreground area (so a real second subject -- two
    people, say, each a large minority of the whole -- survives; measured
    against the *largest* component alone, a real ghost this review found
    -- a blurred camera corner, 15.7% of its own still's main subject but
    only 13.3% of that still's total foreground -- would have survived the
    same check it was meant to catch). Fills interior holes in what is
    kept, and leaves every kept pixel's own soft (anti-aliased) matte value
    untouched. Returns `mask` unchanged on any failure -- never raises."""
    try:
        import cv2
        binary = (np.asarray(mask, dtype=np.uint8) >= 127).astype(np.uint8)
        if not binary.any():
            return mask
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        if num_labels > 2:  # more than one real component (label 0 = background)
            areas = {lbl: int(np.count_nonzero(labels == lbl)) for lbl in range(1, num_labels)}
            largest = max(areas.values())
            total = sum(areas.values())
            keep_labels = [lbl for lbl, a in areas.items() if a == largest or a >= total * 0.15]
            binary = np.isin(labels, keep_labels).astype(np.uint8)
        binary = _fill_holes(binary)
        arr = np.asarray(mask, dtype=np.uint8).copy()
        arr[binary == 0] = 0
        # A filled hole had no real matte value of its own (it was read as
        # background, near 0) -- give it a solid opaque value so it reads as
        # part of the sticker rather than a hole in it.
        hole = (binary == 1) & (arr < 16)
        arr[hole] = 255
        return Image.fromarray(arr, mode="L")
    except Exception as e:  # noqa: BLE001 - mask cleanup must never fail the render
        logger.warning("Collage mask cleanup failed: %s", e)
        return mask


def _mask_bbox(mask: Image.Image, pad_frac: float = CUTOUT_CROP_PAD) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.where(arr >= 16)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = arr.shape
    pad = int(round(max(h, w) * pad_frac))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return (x0, y0, x1, y1)


def _downscale(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Shrinks `img` so its longer side is at most `max_side`; a still
    already at or under that size is returned unchanged."""
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale >= 0.999:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def _add_outline(rgba: Image.Image, outline_px: int = OUTLINE_PX,
                 color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    """Bakes a solid-colour ring `outline_px` wide around `rgba`'s own alpha
    silhouette -- the "~8 px white outline" (Architecture) -- directly into
    the PNG, so Chromium never has to recompute it with a filter every frame.
    The canvas is padded first so the ring is never clipped at the edge."""
    padded = ImageOps.expand(rgba, border=outline_px, fill=(0, 0, 0, 0))
    alpha = padded.split()[-1]
    k = outline_px * 2 + 1
    dilated = alpha.filter(ImageFilter.MaxFilter(k))
    outline_layer = Image.new("RGBA", padded.size, color)
    outline_layer.putalpha(dilated)
    return Image.alpha_composite(outline_layer, padded)


def cutout_rgba(still_path: Path) -> Optional[Image.Image]:
    """`still_path` cut out of its background -- an RGBA image cropped tight
    to the subject's own bounding box and downscaled to `MAX_SIDE` -- or None
    when the model is unavailable or the mask reads as too small/large a
    share of the frame (``MASK_MIN_RATIO``/``MASK_MAX_RATIO``): the caller
    then draws it as a plain polaroid instead. Never raises."""
    try:
        with Image.open(still_path) as src:
            img = src.convert("RGB")
        mask = infer_mask(img)
        if mask is None:
            return None
        mask = _sharpen_mask(mask)
        mask = _clean_mask(mask)
        coverage = _mask_coverage(mask)
        if coverage < MASK_MIN_RATIO or coverage > MASK_MAX_RATIO:
            logger.info("Collage cutout for %s skipped (coverage=%.2f); polaroid fallback",
                       still_path.name, coverage)
            return None
        rgba = img.convert("RGBA")
        rgba.putalpha(mask)
        bbox = _mask_bbox(mask)
        if bbox:
            rgba = rgba.crop(bbox)
        return _downscale(rgba)
    except Exception as e:  # noqa: BLE001
        logger.warning("Cutout for %s failed: %s", still_path.name, e)
        return None


def _still_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cutout_cached(still_path: Path) -> Optional[Path]:
    """The cached, outline-baked cutout PNG for `still_path`, named by the
    still's own content hash (mirrors motion_parallax.py's depth cache) --
    computed once, never recomputed for identical bytes. None (not cached)
    when the cutout itself returned None; cheap enough to re-check next time
    rather than caching a negative result."""
    try:
        key = _still_hash(still_path)
        cache_path = still_path.parent / f"{key}.cutout.png"
        if cache_path.exists():
            return cache_path
        rgba = cutout_rgba(still_path)
        if rgba is None:
            return None
        outlined = _add_outline(rgba)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.png")
        outlined.save(tmp)
        tmp.replace(cache_path)
        return cache_path
    except Exception as e:  # noqa: BLE001
        logger.warning("Cutout cache for %s failed: %s", still_path.name, e)
        return None


def _data_uri_rgba(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _data_uri_rgb(path: Path) -> Optional[str]:
    """The still, downscaled to `MAX_SIDE`, as a plain `data:` JPEG URI
    (mirrors motion_parallax.py's own `_data_uri`) -- PaperCollage.tsx draws
    this as a polaroid card."""
    try:
        with Image.open(path) as src:
            rgb = _downscale(src.convert("RGB"))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001
        logger.warning("Collage image data for %s failed: %s", path.name, e)
        return None


# --- beat plumbing: expand, annotate, drop ------------------------------------

def expand_collages(beats: Sequence[Beat], genre: str, genre_secondary: Optional[str],
                    art_style: str) -> List[Beat]:
    """A `paper_collage` beat keeps its own place on the timeline (the
    Remotion render happens for it, like any other motion beat) and also
    gets one zero-length `broll_image` sub-beat per prompt appended -- zero
    length so it costs nothing in any coverage/budget math downstream
    (`presentation.live`'s own trim). `assets_stage.generate_assets` makes a
    real picture for each exactly like any other B-roll beat; `drop_collage_
    subs` keeps them out of `placement.place_broll`. Call the same way
    `composite.expand_splits` already is, right after it, before anything
    counts assets (director.py)."""
    from . import genre as genre_mod
    out: List[Beat] = []
    for beat in beats:
        if beat.kind != "paper_collage":
            out.append(beat)
            continue
        prompts = [p for p in ((beat.data or {}).get("prompts") or []) if p][:4]
        if len(prompts) < 2:
            out.append(beat)
            continue
        beat_id = beat.id or f"collage_{uuid.uuid4().hex[:6]}"
        parent = beat if beat.id else beat.model_copy(update={"id": beat_id})
        out.append(parent)
        for i, raw_prompt in enumerate(prompts):
            prompt = genre_mod.apply_look(raw_prompt, genre, genre_secondary, art_style) or raw_prompt
            out.append(parent.model_copy(update={
                "id": f"{beat_id}_c{i}",
                "kind": "broll_image",
                "end_s": parent.start_s,
                "image_prompt": prompt,
                "video_prompt": None,
                "text": None,
                "subtext": None,
                "place": None,
                "data": {"collage_sub": True, "parent_beat": beat_id, "index": i},
                "planned_duration_s": None,
                "shows_character": False,
                "sfx": None,
            }))
    return out


def annotate_collage_assets(beats: Sequence[Beat], generated: Sequence[Asset]) -> None:
    """After every still already exists (assets_stage.generate_assets), stash
    each `paper_collage` beat's own sub-beat picture paths on its `data` --
    `build_paper_collage` and `motion.render_fallback`'s `paper_collage`
    branch both read them from there, so neither needs the `generated` list
    threaded through `MotionContext`. Mutates `beats` in place; never raises."""
    try:
        by_id = {a.beat_id: a for a in generated}
        for beat in beats:
            if beat.kind != "paper_collage":
                continue
            prompts = (beat.data or {}).get("prompts") or []
            paths: List[Optional[str]] = []
            for i in range(len(prompts)):
                asset = by_id.get(f"{beat.id}_c{i}")
                paths.append(asset.path if asset is not None and Path(asset.path).exists() else None)
            beat.data["sub_asset_paths"] = paths
            first_real = next((p for p in paths if p), None)
            if first_real:
                beat.data["fallback_image"] = first_real
    except Exception as e:  # noqa: BLE001
        logger.warning("annotate_collage_assets failed: %s", e)


def drop_collage_subs(assets: Sequence[Asset], beats: Sequence[Beat]) -> List[Asset]:
    """A `paper_collage` sub-beat's picture is cut out *inside* the collage
    (MOTION_GRAPHICS_PLAN.md M5) -- generated like any other `broll_image`
    beat, but never its own cutaway. Drops its asset right before
    `placement.place_broll`; every other asset is returned untouched."""
    sub_ids = {b.id for b in beats if (b.data or {}).get("collage_sub")}
    if not sub_ids:
        return list(assets)
    return [a for a in assets if a.beat_id not in sub_ids]


# --- the builder ----------------------------------------------------------

def build_paper_collage(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """`beat.data["sub_asset_paths"]` (set by `annotate_collage_assets`) -> a
    PaperCollage job spec: each path becomes a cut-out sticker (``cutout:
    True``) or, when the mask is missing or unusable, a plain photo
    (``cutout: False``) that PaperCollage.tsx draws as a polaroid instead.
    None when fewer than 2 of the 2-4 prompts actually generated a picture --
    too thin a collage to render; the beat falls back to a plain still
    (`motion.render_fallback`) or is dropped."""
    sub_paths = (beat.data or {}).get("sub_asset_paths") or []
    images: List[Dict[str, Any]] = []
    for raw_path in sub_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        cutout_path = cutout_cached(path)
        added = False
        if cutout_path is not None:
            try:
                with Image.open(cutout_path) as img:
                    w, h = img.size
                    src = _data_uri_rgba(img)
                images.append({"src": src, "cutout": True, "aspect": w / max(1, h)})
                added = True
            except Exception as e:  # noqa: BLE001
                logger.warning("Reading cached cutout %s failed: %s", cutout_path, e)
        if not added:
            src = _data_uri_rgb(path)
            if src is None:
                continue
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:  # noqa: BLE001
                w, h = 4, 3
            images.append({"src": src, "cutout": False, "aspect": w / max(1, h)})
    if len(images) < 2:
        return None
    images = images[:4]
    caption = (beat.data or {}).get("caption") or (beat.text or None)
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    seed_val = int(hashlib.sha1(beat.id.encode()).hexdigest()[:8], 16)
    grid = str(getattr(ctx.settings, "art_style", "real") or "real") == "explainer"
    return {
        "composition": "PaperCollage",
        "props": {"images": images, "caption": caption, "grid": grid, "seed": seed_val, "theme": ctx.theme},
        "width": ctx.width,
        "height": ctx.height,
        "fps": ctx.fps,
        "frames": frames,
        "transparent": False,
    }
