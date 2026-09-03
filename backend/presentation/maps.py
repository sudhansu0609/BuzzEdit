"""Map cutaways, drawn offline from Natural Earth data.

When the speaker names a place, the video shows where it is: a map centred
on it with land, borders, neighbouring cities and a marker, pushed in slowly
by the same Ken Burns move the B-roll gets. Nothing here touches the network.
The data is three GeoJSON files in `data/geo/` (countries, states, populated
places — Natural Earth, public domain) and the drawing is PIL.

Geocoding is a lookup, not a service: city names (with their ASCII forms and a
small alias table for the names Hindi speakers use), then countries, then
states. A name that matches nothing draws nothing — a wrong map is worse than
no map, so nothing is guessed from partial matches.

Two looks. `clean` is the explainer map: pale land, blue water, a red marker.
`noir` is the horror map: near-black land, faint borders, a blood-red marker
and a vignette, so the map belongs in the same dark edit as the footage.
"""

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import DATA_DIR

from .models import Asset, Beat

logger = logging.getLogger("presentation.maps")

GEO_DIR = Path(DATA_DIR) / "geo"
COUNTRIES_FILE = GEO_DIR / "ne_50m_admin_0_countries.geojson"
STATES_FILE = GEO_DIR / "ne_10m_admin_1_states_provinces.geojson"
PLACES_FILE = GEO_DIR / "ne_10m_populated_places_simple.geojson"

# Names the speaker uses that the data spells differently.
ALIASES: Dict[str, str] = {
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "bangalore": "bengaluru", "banaras": "varanasi", "benares": "varanasi",
    "kashi": "varanasi", "allahabad": "prayagraj", "gurgaon": "gurugram",
    "poona": "pune", "trivandrum": "thiruvananthapuram", "cochin": "kochi",
    "baroda": "vadodara", "mysore": "mysuru", "simla": "shimla", "dilli": "delhi",
    "new delhi": "delhi", "burma": "myanmar", "usa": "united states of america",
    "america": "united states of america", "us": "united states of america",
    "uk": "united kingdom", "britain": "united kingdom", "england": "united kingdom",
    "bharat": "india", "hindustan": "india", "uae": "united arab emirates",
    "dubai": "dubai", "russia": "russia", "korea": "south korea",
}

# How wide the view is, in degrees of longitude, per kind of place.
SPAN_CITY_DEG = 9.0
SPAN_STATE_DEG = 14.0
SPAN_COUNTRY_MIN_DEG = 12.0

STYLES: Dict[str, Dict[str, Any]] = {
    "clean": {
        "water": (191, 216, 232), "land": (237, 231, 218), "border": (168, 159, 140),
        "state": (196, 189, 172), "marker": (224, 57, 62), "ring": (224, 57, 62),
        "label": (28, 28, 30), "label_halo": (255, 255, 255), "city": (96, 92, 84),
        "city_label": (70, 66, 60), "vignette": 0.0, "grain": 0,
    },
    "noir": {
        "water": (9, 11, 14), "land": (26, 29, 34), "border": (62, 68, 76),
        "state": (46, 51, 58), "marker": (200, 16, 46), "ring": (200, 16, 46),
        "label": (216, 208, 196), "label_halo": (0, 0, 0), "city": (110, 116, 124),
        "city_label": (150, 150, 150), "vignette": 0.55, "grain": 14,
    },
}


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    kind: str                 # "city" | "country" | "state"
    country: str = ""
    admin1: str = ""
    population: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None   # lon0, lat0, lon1, lat1


# --- data ---------------------------------------------------------------------------

_cache: Dict[str, Any] = {}


def data_available() -> bool:
    return COUNTRIES_FILE.exists() and PLACES_FILE.exists()


def _load(path: Path) -> Optional[Dict[str, Any]]:
    key = str(path)
    if key in _cache:
        return _cache[key]
    if not path.exists():
        _cache[key] = None
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        data = None
    _cache[key] = data
    return data


def _norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def _geometry_bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    lons: List[float] = []
    lats: List[float] = []
    for ring in _rings(geometry):
        for lon, lat in ring:
            lons.append(lon)
            lats.append(lat)
    if not lons:
        return 0.0, 0.0, 0.0, 0.0
    return min(lons), min(lats), max(lons), max(lats)


def _rings(geometry: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    rings: List[List[Tuple[float, float]]] = []
    if kind == "Polygon":
        polygons = [coords]
    elif kind == "MultiPolygon":
        polygons = coords
    else:
        return rings
    for polygon in polygons:
        for ring in polygon:
            rings.append([(float(p[0]), float(p[1])) for p in ring])
    return rings


def _country_index() -> Dict[str, Dict[str, Any]]:
    if "countries" in _cache:
        return _cache["countries"]
    index: Dict[str, Dict[str, Any]] = {}
    data = _load(COUNTRIES_FILE)
    for feature in (data or {}).get("features", []):
        props = feature.get("properties") or {}
        names = {props.get(k) for k in ("NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT",
                                        "NAME_EN", "FORMAL_EN", "BRK_NAME")}
        entry = {"props": props, "geometry": feature.get("geometry") or {},
                 "name": props.get("NAME") or props.get("ADMIN") or ""}
        for name in names:
            if name:
                index.setdefault(_norm(str(name)), entry)
    _cache["countries"] = index
    return index


def _state_index() -> Dict[str, Dict[str, Any]]:
    if "states" in _cache:
        return _cache["states"]
    index: Dict[str, Dict[str, Any]] = {}
    data = _load(STATES_FILE)
    for feature in (data or {}).get("features", []):
        props = feature.get("properties") or {}
        entry = {"props": props, "geometry": feature.get("geometry") or {},
                 "name": props.get("name") or "", "country": props.get("admin") or ""}
        for key in ("name", "name_en", "gn_name", "name_alt"):
            value = props.get(key)
            if value:
                for part in str(value).split("|"):
                    index.setdefault(_norm(part), entry)
    _cache["states"] = index
    return index


def _place_index() -> Dict[str, List[Dict[str, Any]]]:
    if "places" in _cache:
        return _cache["places"]
    index: Dict[str, List[Dict[str, Any]]] = {}
    data = _load(PLACES_FILE)
    for feature in (data or {}).get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        entry = {"name": props.get("name") or props.get("nameascii") or "",
                 "lon": float(coords[0]), "lat": float(coords[1]),
                 "country": props.get("adm0name") or props.get("sov0name") or "",
                 "admin1": props.get("adm1name") or "",
                 "pop": float(props.get("pop_max") or props.get("pop_min") or 0.0)}
        for key in ("name", "nameascii", "namealt", "ls_name"):
            value = props.get(key)
            if value:
                index.setdefault(_norm(str(value)), []).append(entry)
    _cache["places"] = index
    return index


# --- geocoding ------------------------------------------------------------------------

def geocode(name: str) -> Optional[Place]:
    """The place a name refers to, or None. Cities first (the most common
    thing a story names), then countries, then states."""
    if not data_available():
        return None
    key = _norm(name)
    if not key:
        return None
    # "Jaipur, Rajasthan" / "Jaipur city": try the whole, then the first part.
    candidates = [key]
    if "," in name:
        candidates.append(_norm(name.split(",")[0]))
    stripped = re.sub(r"\b(city|town|village|district|state|province|ka|ke|ki)\b", "", key).strip()
    if stripped and stripped != key:
        candidates.append(stripped)

    places = _place_index()
    countries = _country_index()
    states = _state_index()
    for candidate in candidates:
        if candidate in countries:
            entry = countries[candidate]
            bbox = _geometry_bbox(entry["geometry"])
            return Place(name=entry["name"], lat=(bbox[1] + bbox[3]) / 2,
                         lon=(bbox[0] + bbox[2]) / 2, kind="country",
                         country=entry["name"], bbox=bbox)
        if candidate in places:
            best = max(places[candidate], key=lambda e: e["pop"])
            return Place(name=best["name"], lat=best["lat"], lon=best["lon"], kind="city",
                         country=best["country"], admin1=best["admin1"], population=best["pop"])
        if candidate in states:
            entry = states[candidate]
            bbox = _geometry_bbox(entry["geometry"])
            return Place(name=entry["name"], lat=(bbox[1] + bbox[3]) / 2,
                         lon=(bbox[0] + bbox[2]) / 2, kind="state",
                         country=entry["country"], bbox=bbox)
    return None


# --- drawing ------------------------------------------------------------------------------

class _Projection:
    """A flat local projection: degrees to pixels around a centre, with the
    longitude scale corrected for latitude so a city block is not stretched."""

    def __init__(self, lon0: float, lat0: float, span_deg: float, width: int, height: int):
        self.lon0, self.lat0 = lon0, lat0
        self.cos = max(0.2, math.cos(math.radians(lat0)))
        self.px_per_deg = width / max(1e-6, span_deg)
        self.width, self.height = width, height

    def to_px(self, lon: float, lat: float) -> Tuple[float, float]:
        x = self.width / 2 + (lon - self.lon0) * self.cos * self.px_per_deg
        y = self.height / 2 - (lat - self.lat0) * self.px_per_deg
        return x, y

    def visible(self, bbox: Tuple[float, float, float, float]) -> bool:
        x0, y1 = self.to_px(bbox[0], bbox[1])
        x1, y0 = self.to_px(bbox[2], bbox[3])
        return x1 >= -50 and x0 <= self.width + 50 and y1 >= -50 and y0 <= self.height + 50


def _font(size: int, bold: bool = True):
    from PIL import ImageFont
    try:
        from utils.fonts import resolve_font_file
        path = resolve_font_file("Montserrat", bold=bold) or resolve_font_file("Arial", bold=bold)
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _span_for(places: Sequence[Place], aspect: float = 16 / 9) -> Tuple[float, float, float]:
    """(lon0, lat0, span_deg) framing every place with room around it.

    `aspect` is width/height: the span is in longitude, so a portrait canvas
    needs the latitude extent scaled up to fit.
    """
    if len(places) == 1:
        place = places[0]
        if place.kind == "city":
            return place.lon, place.lat, SPAN_CITY_DEG * (aspect / (16 / 9)) ** 0.5
        if place.bbox:
            lon0, lat0, lon1, lat1 = place.bbox
            cos = max(0.2, math.cos(math.radians((lat0 + lat1) / 2)))
            span = max(SPAN_COUNTRY_MIN_DEG if place.kind == "country" else SPAN_STATE_DEG,
                       (lon1 - lon0) * cos * 1.35, (lat1 - lat0) * 1.35 * aspect)
            return (lon0 + lon1) / 2, (lat0 + lat1) / 2, min(140.0, span)
        return place.lon, place.lat, SPAN_STATE_DEG
    lons = [p.lon for p in places]
    lats = [p.lat for p in places]
    cos = max(0.2, math.cos(math.radians(sum(lats) / len(lats))))
    span = max(SPAN_CITY_DEG, (max(lons) - min(lons)) * cos * 1.8,
               (max(lats) - min(lats)) * 1.8 * aspect)
    return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2, min(140.0, span)


def render_map(places: Sequence[Place], width: int, height: int, style: str = "clean",
               label: Optional[str] = None) -> "Image.Image":
    """A map image framing `places`. The first place gets the marker and the
    label; the rest get smaller markers joined by a dashed route."""
    from PIL import Image, ImageDraw, ImageFilter

    look = STYLES.get(style, STYLES["clean"])
    lon0, lat0, span = _span_for(places, width / max(1, height))
    # Supersample for clean edges on the polygons, then scale down.
    ss = 2
    W, H = width * ss, height * ss
    proj = _Projection(lon0, lat0, span, W, H)
    image = Image.new("RGB", (W, H), look["water"])
    draw = ImageDraw.Draw(image)

    countries = _load(COUNTRIES_FILE) or {}
    for feature in countries.get("features", []):
        geometry = feature.get("geometry") or {}
        bbox = _geometry_bbox(geometry)
        if not proj.visible(bbox):
            continue
        for ring in _rings(geometry):
            points = [proj.to_px(lon, lat) for lon, lat in ring]
            if len(points) >= 3:
                draw.polygon(points, fill=look["land"], outline=look["border"])

    # A country or state is a shape, not a point: tint its polygon.
    highlight = places[0] if places and places[0].kind in ("country", "state") else None
    if highlight is not None:
        source = _country_index() if highlight.kind == "country" else _state_index()
        entry = source.get(_norm(highlight.name))
        if entry:
            tint = tuple(int(l * 0.55 + m * 0.45) for l, m in zip(look["land"], look["marker"]))
            for ring in _rings(entry["geometry"]):
                points = [proj.to_px(lon, lat) for lon, lat in ring]
                if len(points) >= 3:
                    draw.polygon(points, fill=tint, outline=look["marker"])

    if span <= 40.0:
        states = _load(STATES_FILE) or {}
        for feature in states.get("features", []):
            geometry = feature.get("geometry") or {}
            bbox = _geometry_bbox(geometry)
            if not proj.visible(bbox):
                continue
            for ring in _rings(geometry):
                points = [proj.to_px(lon, lat) for lon, lat in ring]
                if len(points) >= 2:
                    draw.line(points, fill=look["state"], width=max(1, ss))

        # Neighbouring cities give the place a context.
        placed_labels: List[Tuple[float, float]] = [proj.to_px(p.lon, p.lat) for p in places]
        own_names = {_norm(p.name) for p in places}
        city_font = _font(int(15 * ss * min(1.0, width / 1280)), bold=False)
        for entries in _place_index().values():
            for entry in entries:
                if entry["pop"] < 300000 or _norm(entry["name"]) in own_names:
                    continue
                x, y = proj.to_px(entry["lon"], entry["lat"])
                if not (0 < x < W and 0 < y < H):
                    continue
                if any(math.hypot(x - px, y - py) < 110 * ss for px, py in placed_labels[:len(places)]):
                    continue
                if any(abs(x - px) < 60 * ss and abs(y - py) < 24 * ss for px, py in placed_labels):
                    continue
                placed_labels.append((x, y))
                r = 3 * ss
                draw.ellipse([x - r, y - r, x + r, y + r], fill=look["city"])
                draw.text((x + 6 * ss, y - 8 * ss), entry["name"], fill=look["city_label"],
                          font=city_font)

    # The route and the markers.
    pixels = [proj.to_px(p.lon, p.lat) for p in places]
    if len(pixels) > 1:
        for (x0, y0), (x1, y1) in zip(pixels, pixels[1:]):
            _dashed(draw, (x0, y0), (x1, y1), look["marker"], width=3 * ss, dash=14 * ss)
    small_font = _font(int(24 * ss * max(0.6, min(1.2, width / 1280))), bold=True)
    for index, (x, y) in enumerate(pixels):
        if places[index].kind != "city" and index == 0:
            continue                      # the shape is the marker
        if index > 0:
            for dx in (-1, 1):
                draw.text((x + 14 * ss + dx * ss, y + 6 * ss), places[index].name,
                          font=small_font, fill=look["label_halo"])
            draw.text((x + 14 * ss, y + 6 * ss), places[index].name, font=small_font,
                      fill=look["label"])
        radius = (16 if index == 0 else 9) * ss * min(1.0, max(0.6, width / 1280))
        for ring_scale, alpha in ((3.2, 60), (2.2, 110)):
            rr = radius * ring_scale
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).ellipse([x - rr, y - rr, x + rr, y + rr],
                                            outline=look["ring"] + (alpha,), width=max(2, 2 * ss))
            image.paste(overlay, (0, 0), overlay)
            draw = ImageDraw.Draw(image)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=look["marker"],
                     outline=look["label_halo"], width=max(2, ss))

    # The label, haloed so it reads on either land or water.
    text = label or places[0].name
    if text:
        font = _font(int(44 * ss * max(0.6, min(1.2, width / 1280))))
        x, y = pixels[0]
        if places[0].kind == "city":
            tx, ty = x + 26 * ss, y - 30 * ss
        else:
            tw = draw.textlength(text, font=font)
            tx, ty = x - tw / 2, y - 26 * ss
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                draw.text((tx + dx * ss, ty + dy * ss), text, font=font, fill=look["label_halo"])
        draw.text((tx, ty), text, font=font, fill=look["label"])

    image = image.resize((width, height), Image.LANCZOS)
    if look["vignette"] > 0:
        image = _vignette(image, look["vignette"])
    if look["grain"] > 0:
        image = _grain(image, look["grain"])
    return image


def _dashed(draw, start, end, colour, width: int, dash: int) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    steps = int(length // dash)
    for i in range(0, steps, 2):
        t0, t1 = i / max(1, steps), min(1.0, (i + 1) / max(1, steps))
        draw.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                   (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=colour, width=width)


def _vignette(image, strength: float):
    from PIL import Image, ImageDraw, ImageFilter
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([-w * 0.15, -h * 0.25, w * 1.15, h * 1.25], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.18))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    faded = Image.composite(image, dark, mask)
    return Image.blend(image, faded, strength)


def _grain(image, amount: int):
    import random
    from PIL import Image
    w, h = image.size
    rng = random.Random(7)
    noise = Image.effect_noise((w, h), amount).convert("RGB")
    return Image.blend(image, Image.blend(image, noise, 0.5), 0.12)


# --- assets ------------------------------------------------------------------------------------

def style_for(genre: str, requested: str = "auto") -> str:
    if requested in STYLES:
        return requested
    return "noir" if genre in ("horror", "true_crime") else "clean"


def render_map_assets(beats: Sequence[Beat], project_dir: Path, width: int, height: int,
                      genre: str = "general", requested_style: str = "auto") -> Tuple[List[Asset], List[Dict[str, str]]]:
    """A PNG per map beat, cached by place + style + size. Returns (assets, failures)."""
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []
    wanted = [b for b in beats if b.kind == "map" and b.place]
    if not wanted:
        return assets, failures
    if not data_available():
        logger.warning("Map data missing in %s; no maps drawn", GEO_DIR)
        return assets, [{"beat_id": b.id, "reason": "map data missing"} for b in wanted]
    style = style_for(genre, requested_style)
    directory = Path(project_dir) / "assets" / "maps"
    directory.mkdir(parents=True, exist_ok=True)

    for beat in wanted:
        names = [n.strip() for n in re.split(r"\s*(?:→|->|to)\s*", beat.place) if n.strip()] \
            if beat.data.get("route") else [beat.place]
        places = [p for p in (geocode(n) for n in names) if p is not None]
        if not places:
            failures.append({"beat_id": beat.id, "reason": f"unknown place {beat.place!r}"})
            continue
        key = hashlib.sha1(f"{[p.name for p in places]}:{style}:{width}x{height}".encode()).hexdigest()[:12]
        path = directory / f"map_{key}.png"
        if not path.exists():
            try:
                render_map(places, width, height, style, label=beat.text or None).save(path)
            except Exception as e:
                logger.warning("Map for %r failed: %s", beat.place, e)
                failures.append({"beat_id": beat.id, "reason": f"render failed: {e}"})
                continue
        assets.append(Asset(beat_id=beat.id, kind="image", path=str(path), width=width,
                            height=height, cache_hit=True))
        logger.info("Map for %r → %s (%s)", beat.place, places[0].name, style)
    return assets, failures
