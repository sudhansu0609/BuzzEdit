"""Compose a thumbnail the way the creator's references do (2026-10-21).

The layout, from the two Hindi horror-channel thumbnails he gave as the
model:

    +---------------------------------------------+----------------+
    | HORROR                                      \\  the creator   |
    | STORY     the story's scene (AI picture)     \\ (his own      |
    |                                               \\ picture,     |
    |             जंगल का                            \\ expressive, |
    |             रेंजर      <- big, red, dripping    \\ behind a    |
    |                                                 \\ torn white  |
    |                                                  \\ edge)      |
    +---------------------------------------------------+-----------+

- the scene fills the frame; the creator's picture takes the right third
  behind a jagged, torn-paper edge with a light border and a shadow;
- the title is two to four Devanagari words, set in a heavy face, blood red
  with a dark 3D extrusion, cracks, a black outline, a faint glow and blood
  drips running off the letters;
- a small channel label ("HORROR STORY") sits top-left in the same manner;
- a vignette and a little grain tie it together.

Everything here is Pillow + numpy; the scene itself is generated elsewhere
(`agents/thumbnail_agent.py`) and handed in as a file. Fonts come from
Windows: Nirmala UI Bold for Devanagari (clean and heavy), Impact for the
Latin label. `compose()` never raises for a missing part: no creator picture
means a full-width scene, no scene means the creator's own frame darkened.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CANVAS = (1920, 1080)
FONT_DEVANAGARI = ("C:/Windows/Fonts/Nirmala.ttc", 1)      # Nirmala UI Bold
FONT_DEVANAGARI_ALT = ("C:/Windows/Fonts/mangalb.ttf", 0)
FONT_LATIN = ("C:/Windows/Fonts/impact.ttf", 0)
FONT_LATIN_ALT = ("C:/Windows/Fonts/arialbd.ttf", 0)

#: Looks per genre: the title colours, whether blood drips run off the letters.
STYLES: Dict[str, Dict[str, object]] = {
    "horror": {"top": (255, 70, 55), "bottom": (196, 8, 8), "depth": (70, 0, 0),
               "glow": (255, 0, 0), "drips": True, "label_second": (230, 20, 20)},
    "true_crime": {"top": (255, 60, 40), "bottom": (140, 0, 0), "depth": (80, 0, 0),
                   "glow": (255, 40, 0), "drips": True, "label_second": (230, 20, 20)},
    "default": {"top": (255, 230, 60), "bottom": (255, 150, 0), "depth": (110, 50, 0),
                "glow": (255, 200, 0), "drips": False, "label_second": (255, 210, 0)},
}

BASE_PICTURE_NAMES = ("thumbnail_base.png", "thumbnail_base.jpg", "thumbnail_base.jpeg", "thumbnail_base.webp")


def style_for(genre: Optional[str]) -> Dict[str, object]:
    return STYLES.get((genre or "").lower(), STYLES["default"])


def find_base_picture(source_video: Optional[str], settings: Optional[Dict] = None) -> Optional[Path]:
    """The creator's own picture for the right panel: the `thumbnail_base_image`
    setting, else `thumbnail_base.*` beside the recording, else one in the
    recording's parent (the channel folder, one picture for every episode)."""
    explicit = str((settings or {}).get("thumbnail_base_image") or "").strip()
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    if not source_video:
        return None
    here = Path(source_video).resolve().parent
    for folder in (here, here.parent):
        for name in BASE_PICTURE_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


# --- pictures -----------------------------------------------------------------------

def _load_font(spec: Tuple[str, int], size: int, fallback: Tuple[str, int]):
    from PIL import ImageFont
    for path, index in (spec, fallback):
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _cover(image, size: Tuple[int, int], focus: Tuple[float, float] = (0.5, 0.5)):
    """Scale to cover `size` and crop around `focus` (fractions of the source)."""
    from PIL import Image
    w, h = image.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
    scaled = image.resize((nw, nh), Image.LANCZOS)
    cx, cy = focus[0] * nw, focus[1] * nh
    left = int(min(max(0, cx - tw / 2), nw - tw))
    top = int(min(max(0, cy - th / 2), nh - th))
    return scaled.crop((left, top, left + tw, top + th))


def face_box(image) -> Optional[Tuple[int, int, int, int]]:
    """The largest frontal face (x, y, w, h), or None."""
    try:
        import cv2
        import numpy as np
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY))
        for name, neighbours in (("haarcascade_frontalface_default.xml", 5), ("haarcascade_frontalface_alt2.xml", 3),
                                 ("haarcascade_frontalface_default.xml", 2)):
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=neighbours,
                                             minSize=(max(24, image.width // 24),) * 2)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return int(x), int(y), int(w), int(h)
        return None
    except Exception as e:  # noqa: BLE001 - no cv2, no face: the centre is used
        logger.debug("Face detection skipped: %s", e)
        return None


def creator_panel(image, size: Tuple[int, int]):
    """The creator's picture cropped to the panel: the face about a third of the
    panel's height, placed in its upper half, like the references."""
    from PIL import ImageEnhance
    tw, th = size
    box = face_box(image)
    if box:
        x, y, w, h = box
        # Scale so the face is ~30% of the panel height, then crop around it.
        scale = (0.36 * th) / max(1, h)
        scale = min(max(scale, max(tw / image.width, th / image.height)), 2.5)
        nw, nh = int(image.width * scale), int(image.height * scale)
        big = image.resize((max(tw, nw), max(th, nh)))
        cx, cy = (x + w / 2) * big.width / image.width, (y + h / 2) * big.height / image.height
        left = int(min(max(0, cx - tw / 2), big.width - tw))
        top = int(min(max(0, cy - th * 0.34), big.height - th))
        panel = big.crop((left, top, left + tw, top + th))
    else:
        # No face found (a dark frame): the creator sits centre-frame in his
        # recordings, so zoom on the upper centre instead of showing the room.
        from PIL import Image
        zoom = 2.0
        region_h = image.height / zoom
        region_w = region_h * tw / th
        if region_w > image.width:
            region_w, region_h = image.width, image.width * th / tw
        cx, cy = image.width * 0.5, image.height * 0.55     # his eyes sit near 0.47H; face in the panel's upper half
        left = int(min(max(0, cx - region_w / 2), image.width - region_w))
        top = int(min(max(0, cy - region_h / 2), image.height - region_h))
        panel = image.crop((left, top, int(left + region_w), int(top + region_h))).resize((tw, th), Image.LANCZOS)
    panel = ImageEnhance.Brightness(panel).enhance(1.18)
    panel = ImageEnhance.Contrast(panel).enhance(1.08)
    panel = ImageEnhance.Color(panel).enhance(1.08)
    return panel


def torn_edge(width: int, height: int, x_top: float, x_bottom: float, rng: random.Random,
              step: int = 26, amp: int = 16) -> List[Tuple[int, int]]:
    """Points of a jagged line from (x_top, 0) to (x_bottom, height)."""
    points = []
    y = 0
    while y <= height:
        t = y / height
        x = x_top + (x_bottom - x_top) * t + rng.uniform(-amp, amp)
        points.append((int(x), y))
        y += step
    points[-1] = (points[-1][0], height)
    return points


def _edge_polygon(points: Sequence[Tuple[int, int]], width: int, height: int, shift: int = 0):
    poly = [(x + shift, y) for x, y in points]
    return poly + [(width + 50, height), (width + 50, 0)]


# --- text ---------------------------------------------------------------------------

def _wrap_title(title: str) -> List[str]:
    words = [w for w in title.replace("\n", " ").split() if w]
    if len(words) <= 2:
        return [" ".join(words)] if words else []
    if len(words) == 3:
        return [" ".join(words[:2]), words[2]] if len(words[0]) + len(words[1]) <= len(words[2]) + 4 \
            else [words[0], " ".join(words[1:])]
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def _gradient(size: Tuple[int, int], top: Tuple[int, int, int], bottom: Tuple[int, int, int]):
    import numpy as np
    from PIL import Image
    w, h = size
    t = np.linspace(0.0, 1.0, max(1, h), dtype="float32")[:, None]
    rows = (np.asarray(top, dtype="float32")[None, :] * (1 - t) + np.asarray(bottom, dtype="float32")[None, :] * t)
    return Image.fromarray(np.repeat(rows[:, None, :], max(1, w), axis=1).astype("uint8"), "RGB")


def _cracks(size: Tuple[int, int], rng: random.Random, strength: float = 0.22):
    """A texture of dark veins to multiply into the letters."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    w, h = size
    layer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(max(8, (w * h) // 40000)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        for _seg in range(rng.randint(4, 10)):
            nx, ny = x + rng.uniform(-40, 40), y + rng.uniform(-30, 30)
            draw.line((x, y, nx, ny), fill=255, width=rng.randint(1, 3))
            x, y = nx, ny
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.asarray(layer, dtype="float32") / 255.0
    return (1.0 - strength * arr)


def _drips(mask, rng: random.Random, scale: float, count: int = 7):
    """A mask of drips hanging off the ink's bottom edge."""
    import numpy as np
    from PIL import Image, ImageDraw
    arr = np.asarray(mask) > 96
    h, w = arr.shape
    drips = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(drips)
    bottoms = []
    for x in range(0, w):
        rows = np.flatnonzero(arr[:, x])
        if rows.size:
            bottoms.append((x, int(rows.max())))
    if not bottoms:
        return drips
    deepest = max(y for _x, y in bottoms)
    candidates = [x for x, y in bottoms if y >= deepest - int(0.18 * deepest)]
    rng.shuffle(candidates)
    chosen: List[int] = []
    for x in candidates:
        if all(abs(x - c) > 70 * scale for c in chosen):
            chosen.append(x)
        if len(chosen) >= count:
            break
    by_x = dict(bottoms)
    for x in chosen:
        y0 = by_x[x]
        length = rng.uniform(40, 170) * scale
        width = rng.uniform(14, 26) * scale
        draw.rounded_rectangle((x - width / 2, y0 - 6, x + width / 2, y0 + length), radius=width / 2, fill=255)
        draw.ellipse((x - width * 0.7, y0 + length - width * 1.2, x + width * 0.7, y0 + length + width * 0.3), fill=255)
    return drips


def render_text_block(lines: Sequence[str], font, style: Dict[str, object], rng: random.Random,
                      scale: float = 1.0, depth: int = 14, stroke: int = 5, line_gap: float = 0.08,
                      fill_top=None, fill_bottom=None, drips: Optional[bool] = None):
    """An RGBA layer with the styled text: glow, outline, 3D depth, gradient
    fill with cracks, and drips. Returns (layer, ink mask)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    top = fill_top or style["top"]
    bottom = fill_bottom or style["bottom"]
    want_drips = style["drips"] if drips is None else drips

    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max(b[3] - b[1] for b in boxes) if boxes else 0
    gap = int(line_h * line_gap)
    text_w = max(b[2] - b[0] for b in boxes) if boxes else 0
    text_h = sum(b[3] - b[1] for b in boxes) + gap * max(0, len(lines) - 1)
    pad = int(60 * scale) + depth + stroke
    extra_drip = int(220 * scale) if want_drips else 0
    size = (text_w + pad * 2, text_h + pad * 2 + extra_drip)

    ink = Image.new("L", size, 0)
    draw = ImageDraw.Draw(ink)
    y = pad
    for line, box in zip(lines, boxes):
        x = pad + (text_w - (box[2] - box[0])) // 2 - box[0]
        draw.text((x, y - box[1]), line, font=font, fill=255)
        y += (box[3] - box[1]) + gap
    if want_drips:
        drip_mask = _drips(ink, rng, scale)
        ink = Image.fromarray(np.maximum(np.asarray(ink), np.asarray(drip_mask)))

    outline = ink.filter(ImageFilter.MaxFilter(stroke * 2 + 1))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    # Glow behind everything.
    glow = outline.filter(ImageFilter.GaussianBlur(18 * scale))
    glow_rgba = Image.new("RGBA", size, tuple(style["glow"]) + (0,))
    glow_rgba.putalpha(glow.point(lambda v: int(v * 0.55)))
    layer.alpha_composite(glow_rgba)
    # 3D depth: the outline stepped down and to the right in the dark colour.
    depth_rgba = Image.new("RGBA", size, tuple(style["depth"]) + (255,))
    for d in range(depth, 0, -1):
        shifted = Image.new("L", size, 0)
        shifted.paste(outline, (d, d))
        layer.paste(depth_rgba, (0, 0), shifted)
    # Black outline, then the gradient fill textured with cracks.
    layer.paste(Image.new("RGBA", size, (0, 0, 0, 255)), (0, 0), outline)
    fill = _gradient(size, top, bottom)
    cracks = _cracks(size, rng)
    fill = Image.fromarray((np.asarray(fill, dtype="float32") * cracks[:, :, None]).clip(0, 255).astype("uint8"), "RGB")
    layer.paste(fill, (0, 0), ink)
    return layer, ink


def _fit_title(lines: List[str], max_w: int, max_h: int, font_spec, fallback_spec, start: int = 300):
    from PIL import Image, ImageDraw
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    size = start
    while size > 40:
        font = _load_font(font_spec, size, fallback_spec)
        boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
        w = max(b[2] - b[0] for b in boxes)
        h = sum(b[3] - b[1] for b in boxes) * 1.08
        if w <= max_w and h <= max_h:
            return font
        size = int(size * 0.92)
    return _load_font(font_spec, 40, fallback_spec)


def _vignette(image, strength: float = 0.42):
    import numpy as np
    from PIL import Image
    w, h = image.size
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    dx, dy = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    r = np.sqrt(dx * dx + dy * dy)
    shade = (1.0 - strength * np.clip((r - 0.55) / 0.7, 0.0, 1.0) ** 1.5)
    arr = np.asarray(image.convert("RGB"), dtype="float32") * shade[:, :, None]
    return Image.fromarray(arr.clip(0, 255).astype("uint8"), "RGB")


def _grain(image, amount: float = 7.0, seed: int = 0):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype="float32")
    arr += rng.normal(0.0, amount, size=arr.shape[:2])[:, :, None]
    return Image.fromarray(arr.clip(0, 255).astype("uint8"), "RGB")


# --- the composition -----------------------------------------------------------------

def compose(scene_path: Optional[Path], creator_path: Optional[Path], title: str,
            out_path: Path, label: str = "", genre: Optional[str] = None,
            size: Tuple[int, int] = CANVAS, seed: int = 7) -> Path:
    """Write the thumbnail to `out_path` (JPEG) and return it."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    rng = random.Random(seed)
    W, H = size
    style = style_for(genre)

    scene = None
    if scene_path and Path(scene_path).is_file():
        try:
            scene = _cover(Image.open(scene_path).convert("RGB"), size)
        except Exception as e:  # noqa: BLE001
            logger.warning("Thumbnail: could not read the scene %s (%s)", scene_path, e)
    creator = None
    if creator_path and Path(creator_path).is_file():
        try:
            creator = Image.open(creator_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            logger.warning("Thumbnail: could not read the creator picture %s (%s)", creator_path, e)
    if scene is None and creator is None:
        scene = Image.new("RGB", size, (12, 8, 10))
    if scene is None:
        scene = ImageEnhance.Brightness(_cover(creator, size).filter(ImageFilter.GaussianBlur(6))).enhance(0.45)

    canvas = scene.copy()
    text_right = W  # where the title may extend to
    if creator is not None:
        x_top, x_bottom = int(W * 0.615), int(W * 0.705)
        points = torn_edge(W, H, x_top, x_bottom, rng)
        panel_left = min(x for x, _ in points) - 40
        panel = creator_panel(creator, (W - panel_left, H))
        # Shadow, light border, then the picture through the torn mask.
        shadow = Image.new("L", size, 0)
        ImageDraw.Draw(shadow).polygon(_edge_polygon(points, W, H, shift=-34), fill=255)
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        canvas.paste(Image.new("RGB", size, (0, 0, 0)), (0, 0), shadow.point(lambda v: int(v * 0.75)))
        border = Image.new("L", size, 0)
        ImageDraw.Draw(border).polygon(_edge_polygon(points, W, H, shift=-14), fill=255)
        canvas.paste(Image.new("RGB", size, (246, 241, 232)), (0, 0), border)
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon(_edge_polygon(points, W, H), fill=255)
        full = Image.new("RGB", size, (0, 0, 0))
        full.paste(panel, (panel_left, 0))
        canvas.paste(full, (0, 0), mask)
        text_right = x_top - int(W * 0.02)

    canvas = _vignette(canvas)

    # The title: left of the creator, vertically a little below centre.
    lines = _wrap_title(title)
    if lines:
        max_w = int(text_right * 0.86)
        max_h = int(H * 0.46)
        font = _fit_title(lines, max_w, max_h, FONT_DEVANAGARI, FONT_DEVANAGARI_ALT)
        layer, ink = render_text_block(lines, font, style, rng, scale=W / 1920.0)
        # A dark pool under the words so they read on a bright scene.
        pool = Image.new("L", size, 0)
        bw, bh = layer.size
        x = max(10, (text_right - bw) // 2)
        y = max(int(H * 0.24), int(H * 0.54 - bh * 0.5))
        ImageDraw.Draw(pool).rounded_rectangle((x + 20, y + 30, x + bw - 20, y + bh - 60), radius=120, fill=255)
        pool = pool.filter(ImageFilter.GaussianBlur(60))
        canvas.paste(Image.new("RGB", size, (0, 0, 0)), (0, 0), pool.point(lambda v: int(v * 0.5)))
        canvas.paste(layer, (x, y), layer)

    if label:
        words = label.split()
        first = words[0] if words else label
        rest = " ".join(words[1:])
        font = _load_font(FONT_LATIN, int(H * 0.068), FONT_LATIN_ALT)
        lx, ly = int(W * 0.02), int(H * 0.025)
        first_layer, _ = render_text_block([first], font, style, rng, scale=W / 1920.0 * 0.5, depth=6, stroke=4,
                                           fill_top=(255, 255, 255), fill_bottom=(225, 225, 225))
        canvas.paste(first_layer, (lx, ly), first_layer)
        if rest:
            second, _ = render_text_block([rest], font, style, rng, scale=W / 1920.0 * 0.5, depth=6, stroke=4,
                                          fill_top=style["label_second"], fill_bottom=style["bottom"])
            canvas.paste(second, (lx, ly + int(first_layer.height * 0.62)), second)

    canvas = _grain(canvas, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=93, subsampling=0)
    try:
        small = canvas.resize((1280, 720), Image.LANCZOS)
        small.save(out_path.with_name(out_path.stem + "_1280" + out_path.suffix), quality=92)
    except Exception:  # noqa: BLE001
        pass
    return out_path
