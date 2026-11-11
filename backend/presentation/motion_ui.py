"""Builders for ``UiMockup`` and ``Callout`` (MOTION_GRAPHICS_PLAN.md M10).

``[ui: <variant> | text=<body> | name=<display name> | handle=<@handle> |
meta=<e.g. 2h . 14.2K likes> | seconds=N]`` -> a ``ui_mockup`` beat ->
``build_ui_mockup`` -> a UiMockup job: a generic social/UI card (post, forum,
chat, search or headline -- no real platform logos, colours or names) that
floats in on the theme background. A render the engine cannot make either
drops (no text to fall back on) or redraws the body text as the closest
existing PIL card (a statement card, ``render_ui_mockup_fallback``), used by
``motion.render_fallback``.

``[callout: <image prompt> | mark=circle|arrow|magnify | at=<3x3 region> |
label=<short text> | seconds=N]`` -> a ``callout`` beat. Unlike every other
motion kind but ``paper_collage``, that beat needs a picture of its own
before the Remotion render can run at all -- ``expand_callouts`` (called
right after ``motion_collage.expand_collages``, director.py, before anything
counts assets) keeps the ``callout`` beat on the timeline and appends one
zero-length ``broll_image`` sub-beat carrying the image prompt, mirroring
``motion_collage.expand_collages``'s own pattern for ``paper_collage`` (that
module is edited in parallel elsewhere, so this is its own small copy of the
idea rather than a shared import). ``assets_stage.generate_assets`` makes a
real ComfyUI picture for it exactly like any other B-roll beat;
``annotate_callout_assets`` (called right after that generation pass) stashes
the resulting file path on the parent beat's own ``data``, so
``build_callout`` and ``motion.render_fallback``'s ``callout`` branch can
both find it without threading ``generated`` through ``MotionContext``;
``drop_callout_subs`` (called right before ``placement.place_broll``) keeps
the sub-beat's own picture out of placement -- the mark/magnifier is drawn
*onto* it inside the Callout composition, never as its own cutaway. A render
the engine cannot make falls back to the plain generated still, placed as a
normal ``broll_image`` (``motion.render_fallback``'s ``callout`` branch).

A planner only ever names a 3x3 ``at`` region, which lands the mark on
whatever happens to sit in that cell -- often not the thing the label is
actually about (MOTION_GRAPHICS_PLAN.md M10 review: a circle around "The oil
lamp" on the empty floor beside it, a magnifier on "Her expression" finding
the dark wall instead of her face). ``aim_callout`` runs before
``build_callout`` returns its props: a label about a face/expression/eyes/
smile/look/he/she is aimed at the nearest face (facezoom.py's own YuNet
detector, built for video punch-ins, run here on this one still); any other
label is aimed at the centroid of the nearest IS-Net salient component
(motion_collage.py's own cutout model, imported read-only) inside the cell
expanded by 60%; neither found anything -> no target prop, and Callout.tsx
falls back to the plain cell centre exactly as before.
"""

import base64
import hashlib
import io
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Asset, Beat
from .motion import MotionContext, _beat_duration

logger = logging.getLogger("presentation.motion_ui")

_UI_VARIANTS = ("post", "forum", "chat", "search", "headline")
_CALLOUT_MARKS = ("circle", "arrow", "magnify")
_CALLOUT_REGIONS = ("top_left", "top", "top_right", "left", "center", "right",
                    "bottom_left", "bottom", "bottom_right")


# --- UiMockup ------------------------------------------------------------

def build_ui_mockup(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``ui_mockup`` beat -> a UiMockup job. None when there is no body text
    to show -- every variant needs at least that."""
    data = beat.data or {}
    variant = data.get("variant", "post")
    if variant not in _UI_VARIANTS:
        variant = "post"
    text = (data.get("text") or beat.text or "").strip()
    if not text:
        return None
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "UiMockup",
        "props": {
            "theme": ctx.theme,
            "variant": variant,
            "text": text,
            "name": data.get("name") or "",
            "handle": data.get("handle") or "",
            "meta": data.get("meta") or "",
        },
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


def render_ui_mockup_fallback(beat: Beat, width: int, height: int):
    """A static stand-in for a failed UiMockup render: the body text as a
    plain statement card (graphics.render_statement_card) -- the closest
    existing PIL card to a floating text card. Never raises -- motion.
    render_fallback treats any exception as 'no fallback for this beat'."""
    from . import graphics as graphics_stage

    data = beat.data or {}
    text = (data.get("text") or beat.text or "").strip()
    if not text:
        raise ValueError("no text for a ui_mockup fallback")
    return graphics_stage.render_statement_card(text, width, height, with_speaker=False)


# --- Callout: expand, annotate, drop -------------------------------------

def expand_callouts(beats: Sequence[Beat], genre: str, genre_secondary: Optional[str],
                    art_style: str) -> List[Beat]:
    """A ``callout`` beat keeps its own place on the timeline and also gets
    one zero-length ``broll_image`` sub-beat appended for its own image
    prompt -- zero length so it costs nothing in any coverage/budget math
    downstream (``presentation.live``'s own trim), mirroring
    ``motion_collage.expand_collages``'s pattern for ``paper_collage``.
    ``assets_stage.generate_assets`` then makes a real picture for it exactly
    like any other B-roll beat; ``annotate_callout_assets`` stashes the file
    path back on the parent; ``drop_callout_subs`` keeps the sub-beat's own
    asset out of placement."""
    from . import genre as genre_mod
    out: List[Beat] = []
    for beat in beats:
        if beat.kind != "callout":
            out.append(beat)
            continue
        prompt = (beat.image_prompt or beat.topic or "").strip()
        if not prompt:
            out.append(beat)
            continue
        beat_id = beat.id or f"callout_{uuid.uuid4().hex[:6]}"
        parent = beat if beat.id else beat.model_copy(update={"id": beat_id})
        out.append(parent)
        looked = genre_mod.apply_look(prompt, genre, genre_secondary, art_style) or prompt
        out.append(parent.model_copy(update={
            "id": f"{beat_id}_still",
            "kind": "broll_image",
            "end_s": parent.start_s,
            "image_prompt": looked,
            "video_prompt": None,
            "text": None,
            "subtext": None,
            "place": None,
            "data": {"callout_sub": True, "parent_beat": beat_id},
            "planned_duration_s": None,
            "shows_character": False,
            "sfx": None,
        }))
    return out


def annotate_callout_assets(beats: Sequence[Beat], generated: Sequence[Asset]) -> None:
    """After the still already exists (assets_stage.generate_assets), stash
    a ``callout`` beat's own picture path on its ``data`` -- ``build_callout``
    and ``motion.render_fallback``'s ``callout`` branch both read it from
    there, so neither needs ``generated`` threaded through ``MotionContext``.
    Mutates ``beats`` in place; never raises."""
    try:
        by_id = {a.beat_id: a for a in generated}
        for beat in beats:
            if beat.kind != "callout":
                continue
            asset = by_id.get(f"{beat.id}_still")
            if asset is not None and Path(asset.path).exists():
                beat.data["still_path"] = asset.path
                beat.data["fallback_image"] = asset.path
    except Exception as e:  # noqa: BLE001 - annotation must never fail the render
        logger.warning("annotate_callout_assets failed: %s", e)


def drop_callout_subs(assets: Sequence[Asset], beats: Sequence[Beat]) -> List[Asset]:
    """A ``callout`` sub-beat's picture is the still the mark/magnifier is
    drawn onto *inside* the Callout composition (MOTION_GRAPHICS_PLAN.md
    M10), never its own cutaway. Drops its asset right before
    ``placement.place_broll``; every other asset is returned untouched."""
    sub_ids = {b.id for b in beats if (b.data or {}).get("callout_sub")}
    if not sub_ids:
        return list(assets)
    return [a for a in assets if a.beat_id not in sub_ids]


def _data_uri(path: Path) -> Optional[str]:
    """The still as a plain ``data:`` JPEG URI -- a small local copy of
    motion_collage.py's own ``_data_uri_rgb`` idea (that module is edited in
    parallel elsewhere, so copied rather than imported)."""
    try:
        from PIL import Image
        with Image.open(path) as src:
            rgb = src.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001 - must never fail the render
        logger.warning("Callout image data for %s failed: %s", path.name, e)
        return None


# --- aiming the mark at the actual subject ---------------------------------

_FACE_WORDS = re.compile(r"\b(face|expression|eyes?|smile|looks?|looking|he|she|his|her|him)\b",
                         re.IGNORECASE)
_REGION_CELL = {
    "top_left": (0, 0), "top": (1, 0), "top_right": (2, 0),
    "left": (0, 1), "center": (1, 1), "right": (2, 1),
    "bottom_left": (0, 2), "bottom": (1, 2), "bottom_right": (2, 2),
}
# A salient component smaller than this share of the still's own area is
# noise (a stray highlight, a compression artefact), not a real subject.
_SALIENCY_MIN_AREA_RATIO = 0.002
# How much bigger than the chosen cell the saliency search still looks --
# the planner's 3x3 region rarely lines up with the subject's own edges.
_SALIENCY_CELL_EXPAND = 1.6


def _cover_scale(img_w: float, img_h: float, canvas_w: float, canvas_h: float) -> float:
    return max(canvas_w / img_w, canvas_h / img_h)


def _canvas_to_image(px: float, py: float, img_w: float, img_h: float,
                     canvas_w: float, canvas_h: float) -> Tuple[float, float]:
    """A canvas-space point to the still's own pixel coordinates, inverting
    the CSS ``background-size: cover`` fit Callout.tsx paints the still
    with (the still's aspect ratio is not generally the canvas's own)."""
    scale = _cover_scale(img_w, img_h, canvas_w, canvas_h)
    off_x = (canvas_w - img_w * scale) / 2.0
    off_y = (canvas_h - img_h * scale) / 2.0
    return (px - off_x) / scale, (py - off_y) / scale


def _image_to_canvas(px: float, py: float, img_w: float, img_h: float,
                     canvas_w: float, canvas_h: float) -> Tuple[float, float]:
    scale = _cover_scale(img_w, img_h, canvas_w, canvas_h)
    off_x = (canvas_w - img_w * scale) / 2.0
    off_y = (canvas_h - img_h * scale) / 2.0
    return off_x + px * scale, off_y + py * scale


def _detect_faces_in_image(image_path: Path) -> List[Tuple[float, float, float, float]]:
    """Face boxes (x, y, w, h), in the still's own pixel coordinates --
    facezoom.py's own YuNet detector (it otherwise only ever runs on video),
    run here on one frame. Empty on any failure -- model unavailable, a
    corrupt still, cv2 missing. Never raises."""
    try:
        from . import facezoom
        model = facezoom.ensure_model()
        if model is None:
            return []
        import cv2
        import numpy as np
        from PIL import Image as PILImage
        with PILImage.open(image_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            arr = np.asarray(rgb)
        bgr = np.ascontiguousarray(arr[:, :, ::-1])
        detector = cv2.FaceDetectorYN.create(str(model), "", (w, h),
                                             score_threshold=0.6, nms_threshold=0.3, top_k=10)
        _, faces = detector.detect(bgr)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces]
    except Exception as e:  # noqa: BLE001 - aiming must never fail the render
        logger.warning("Callout face detection failed: %s", e)
        return []


def _salient_component_near(image_path: Path, cx: float, cy: float,
                            search_w: float, search_h: float,
                            img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """The IS-Net salient component (motion_collage.py's own cutout model,
    imported read-only -- that module is edited in parallel elsewhere, so
    only its loader/runner are reused here, never its own beat plumbing)
    whose centroid is nearest ``(cx, cy)`` among those inside the search
    window, as ``(centroid_x, centroid_y, half_w, half_h)`` in the still's
    own pixel coordinates. None on any failure or when nothing salient
    falls inside the window. Never raises."""
    try:
        from . import motion_collage
        from PIL import Image as PILImage
        import numpy as np
        import cv2
        with PILImage.open(image_path) as im:
            rgb = im.convert("RGB")
            mask = motion_collage.infer_mask(rgb)
        if mask is None:
            return None
        mask = motion_collage._sharpen_mask(mask)
        binary = (np.asarray(mask, dtype=np.uint8) >= 127).astype(np.uint8)
        if not binary.any():
            return None
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        if num_labels <= 1:
            return None
        x0, x1 = max(0.0, cx - search_w / 2.0), min(float(img_w), cx + search_w / 2.0)
        y0, y1 = max(0.0, cy - search_h / 2.0), min(float(img_h), cy + search_h / 2.0)
        min_area = img_w * img_h * _SALIENCY_MIN_AREA_RATIO
        best = None
        best_d = None
        for lbl in range(1, num_labels):
            ys, xs = np.where(labels == lbl)
            if ys.size < min_area:
                continue
            ccx, ccy = float(xs.mean()), float(ys.mean())
            if not (x0 <= ccx <= x1 and y0 <= ccy <= y1):
                continue
            d = (ccx - cx) ** 2 + (ccy - cy) ** 2
            if best_d is None or d < best_d:
                hx = float(xs.max() - xs.min()) / 2.0
                hy = float(ys.max() - ys.min()) / 2.0
                best_d, best = d, (ccx, ccy, hx, hy)
        return best
    except Exception as e:  # noqa: BLE001 - aiming must never fail the render
        logger.warning("Callout saliency aim failed: %s", e)
        return None


def _nearest_box(boxes: Sequence[Tuple[float, float, float, float]],
                 cx: float, cy: float) -> Optional[Tuple[float, float, float, float]]:
    """The box ``(centre_x, centre_y, half_w, half_h)`` whose centre is
    nearest ``(cx, cy)``, or None."""
    best = None
    best_d = None
    for b in boxes:
        d = (b[0] - cx) ** 2 + (b[1] - cy) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, b
    return best


def aim_callout(label: str, image_path: Path, region: str,
                canvas_w: int, canvas_h: int) -> Optional[Dict[str, float]]:
    """Where a callout mark should actually land, in canvas-fraction
    coordinates (MOTION_GRAPHICS_PLAN.md M10 review -- see the module
    docstring). Returns ``{"x", "y", "rx", "ry"}``, all 0..1 fractions of
    the canvas (``rx``/``ry`` the half-extent, of width/height respectively)
    -- the still is painted onto a canvas of a different aspect ratio via
    CSS ``cover``, so the mapping from the still's own pixels is not 1:1.
    None when nothing was found (no face, nothing salient nearby) --
    callers keep the plain cell centre they already draw. Never raises."""
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as im:
            img_w, img_h = im.size
    except Exception as e:  # noqa: BLE001
        logger.warning("Callout aim: could not read %s (%s)", image_path, e)
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    col, row = _REGION_CELL.get(region, (1, 1))
    cell_w, cell_h = canvas_w / 3.0, canvas_h / 3.0
    cell_cx_c = col * cell_w + cell_w / 2.0
    cell_cy_c = row * cell_h + cell_h / 2.0
    cell_cx_i, cell_cy_i = _canvas_to_image(cell_cx_c, cell_cy_c, img_w, img_h, canvas_w, canvas_h)

    box = None
    if _FACE_WORDS.search(label or ""):
        faces = _detect_faces_in_image(image_path)
        centered = [(x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0) for (x, y, w, h) in faces]
        box = _nearest_box(centered, cell_cx_i, cell_cy_i)
    if box is None:
        scale = _cover_scale(img_w, img_h, canvas_w, canvas_h)
        search_w = (cell_w / scale) * _SALIENCY_CELL_EXPAND
        search_h = (cell_h / scale) * _SALIENCY_CELL_EXPAND
        box = _salient_component_near(image_path, cell_cx_i, cell_cy_i, search_w, search_h, img_w, img_h)
    if box is None:
        return None

    cx_i, cy_i, hx_i, hy_i = box
    cx_c, cy_c = _image_to_canvas(cx_i, cy_i, img_w, img_h, canvas_w, canvas_h)
    scale = _cover_scale(img_w, img_h, canvas_w, canvas_h)
    return {
        "x": max(0.0, min(1.0, cx_c / canvas_w)),
        "y": max(0.0, min(1.0, cy_c / canvas_h)),
        "rx": max(0.025, min(0.45, (hx_i * scale) / canvas_w)),
        "ry": max(0.025, min(0.45, (hy_i * scale) / canvas_h)),
    }


# --- the builder ----------------------------------------------------------

def build_callout(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``beat.data["still_path"]`` (set by ``annotate_callout_assets``) -> a
    Callout job spec. None when the still never generated -- the beat falls
    back to the plain still (``motion.render_fallback``) or is dropped."""
    data = beat.data or {}
    still_path = data.get("still_path")
    if not still_path or not Path(still_path).exists():
        return None
    src = _data_uri(Path(still_path))
    if src is None:
        return None
    mark = data.get("mark", "circle")
    if mark not in _CALLOUT_MARKS:
        mark = "circle"
    region = data.get("at", "center")
    if region not in _CALLOUT_REGIONS:
        region = "center"
    label = data.get("label") or None
    target = aim_callout(label or "", Path(still_path), region, ctx.width, ctx.height)
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    props: Dict[str, Any] = {
        "theme": ctx.theme,
        "image": src,
        "mark": mark,
        "at": region,
        "label": label,
    }
    if target:
        props["target"] = target
    return {
        "composition": "Callout",
        "props": props,
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }
