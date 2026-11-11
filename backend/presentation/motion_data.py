"""Builders for ``MotionMap``, ``MotionChart``, ``MotionStat`` and
``MotionTimeline`` (MOTION_GRAPHICS_PLAN.md M2) -- the Remotion-rendered
replacements for the ``map``, ``chart``, ``stat_callout`` and ``timeline``
PIL stills. Each mirrors the exact data a PIL renderer would have used
(``maps.py``, ``charts.py``, ``graphics.py``) so a motion render and its PIL
fallback always carry the same information -- one of them animated, the
other a still. Registered one line each in ``presentation/motion.py::BUILDERS``.

A builder that finds nothing worth animating returns ``None`` -- the beat is
then reported as a motion failure and left for whichever PIL stage already
draws that kind, exactly as before this module existed.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Beat
from .motion import MotionContext, _beat_duration

logger = logging.getLogger("presentation.motion_data")

# charts.py::render_chart_assets treats these chart_type values as a line chart.
_CHART_LINE_TYPES = ("line", "graph")

# M2b (review 2026-10-28): a clean MotionMap read as a blank beige field --
# no country/state borders, no neighbour cities -- because only the world
# outline and the one highlighted place ever travelled as props. These caps
# keep the extra context small enough to still count as "a few seconds of
# motion graphic" (maps.motion_feature's own words) rather than shipping a
# whole region's Natural Earth data over the wire.
_MOTION_MAX_BORDER_FEATURES = 36
_MOTION_MAX_NEIGHBOURS = 8
_MOTION_NEIGHBOUR_MIN_POP = 300_000.0


def _motion_view_window(lon0: float, lat0: float, span: float, aspect: float
                        ) -> Tuple[float, float, float, float]:
    """A generous degree-space (lon0, lat0, lon1, lat1) box around the
    zoomed-in view -- only used to pick which borders/neighbours are worth
    sending; the browser's own ``geoOrthographic().clipAngle(90)`` still
    clips the exact screen edge at render time, so over-including here only
    costs a little payload, never correctness."""
    cos = max(0.2, math.cos(math.radians(lat0)))
    half_lon = (span / cos) * 0.75
    half_lat = (span / max(0.3, aspect)) * 0.75
    return lon0 - half_lon, lat0 - half_lat, lon0 + half_lon, lat0 + half_lat


def _motion_borders(maps_stage, lon0: float, lat0: float, span: float, aspect: float
                    ) -> List[Dict[str, Any]]:
    """Country borders (always) plus state/province lines (only once zoomed
    in, mirroring ``maps.render_map``'s own ``span <= 40.0`` gate) for
    whatever is inside the view window -- simplified and rounded exactly the
    way ``maps.motion_feature`` already does for the highlight, so the payload
    stays small."""
    window = _motion_view_window(lon0, lat0, span, aspect)

    def _overlaps(bbox: Tuple[float, float, float, float]) -> bool:
        lo0, la0, lo1, la1 = bbox
        # A country/state whose own rings cross the antimeridian (Alaska's
        # Aleutians, Russia's Far East, Fiji) makes maps._geometry_bbox's
        # plain min/max come out almost global -- that bbox would then
        # "overlap" every window on Earth. None of those belong in a small
        # per-beat props payload, so they are skipped rather than trusted.
        if lo1 - lo0 > 180:
            return False
        return lo1 >= window[0] and lo0 <= window[2] and la1 >= window[1] and la0 <= window[3]

    def _feature(geometry: Dict[str, Any], is_state: bool) -> Optional[Dict[str, Any]]:
        rings = [r for r in maps_stage._rings(geometry) if len(r) >= 2]
        if not rings:
            return None
        return {
            "rings": [[[round(lon, 2), round(lat, 2)] for lon, lat in maps_stage._simplify_ring(r)]
                     for r in rings],
            "state": is_state,
        }

    out: List[Dict[str, Any]] = []
    countries = maps_stage._load(maps_stage.COUNTRIES_FILE) or {}
    for feature in countries.get("features", []):
        if len(out) >= _MOTION_MAX_BORDER_FEATURES:
            return out
        geometry = feature.get("geometry") or {}
        if not _overlaps(maps_stage._geometry_bbox(geometry)):
            continue
        shape = _feature(geometry, False)
        if shape:
            out.append(shape)
    if span <= 40.0:
        states = maps_stage._load(maps_stage.STATES_FILE) or {}
        for feature in states.get("features", []):
            if len(out) >= _MOTION_MAX_BORDER_FEATURES:
                break
            geometry = feature.get("geometry") or {}
            if not _overlaps(maps_stage._geometry_bbox(geometry)):
                continue
            shape = _feature(geometry, True)
            if shape:
                out.append(shape)
    return out


def _motion_neighbours(maps_stage, places: Sequence[Any], lon0: float, lat0: float,
                       span: float, aspect: float) -> List[Dict[str, Any]]:
    """4-8 nearby cities (population >= 300k, Natural Earth's own populated
    places -- the same source ``maps.render_map``'s neighbour dots use) to
    give the zoomed-in map context, picked biggest-first and spaced out in
    degrees so they don't crowd the places already on the map. Only once
    zoomed in enough to be legible (``span <= 40.0``, same gate as the state
    lines)."""
    if span > 40.0:
        return []
    window = _motion_view_window(lon0, lat0, span, aspect)
    own_names = {maps_stage._norm(p.name) for p in places}
    picked_points = [(p.lon, p.lat) for p in places]
    min_separation = max(0.25, span * 0.1)

    candidates = []
    for entries in maps_stage._place_index().values():
        for entry in entries:
            if entry["pop"] < _MOTION_NEIGHBOUR_MIN_POP:
                continue
            if maps_stage._norm(entry["name"]) in own_names:
                continue
            lon, lat = entry["lon"], entry["lat"]
            if window[0] <= lon <= window[2] and window[1] <= lat <= window[3]:
                candidates.append(entry)
    candidates.sort(key=lambda e: -e["pop"])

    out: List[Dict[str, Any]] = []
    seen_names = set()
    for entry in candidates:
        name = maps_stage._norm(entry["name"])
        if name in seen_names:
            continue
        lon, lat = entry["lon"], entry["lat"]
        if any(math.hypot(lon - px, lat - py) < min_separation for px, py in picked_points):
            continue
        picked_points.append((lon, lat))
        seen_names.add(name)
        out.append({"name": entry["name"], "lat": round(lat, 2), "lon": round(lon, 2)})
        if len(out) >= _MOTION_MAX_NEIGHBOURS:
            break
    return out


# --- map ----------------------------------------------------------------

def build_map(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``map`` beat -> a MotionMap job: an orthographic globe that turns to
    the place and zooms in, mirroring ``maps.render_map_assets``'s place
    lookup (``maps.places_for_beat``) and country/state highlight
    (``maps.motion_feature``, already clipped to that one place and rounded
    to 2 decimals). Also sends country/state borders and a handful of
    neighbour cities in view (M2b review: the clean map read as a blank
    beige field with no context) -- picked the same way ``maps.render_map``
    picks them for the PIL fallback, simplified and capped so the extra
    props stay small."""
    from . import maps as maps_stage
    if not beat.place or not maps_stage.data_available():
        return None
    places = maps_stage.places_for_beat(beat)
    if not places:
        return None
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    is_route = bool(beat.data.get("route")) and len(places) > 1
    aspect = ctx.width / max(1, ctx.height)
    lon0, lat0, span = maps_stage._span_for(places, aspect)
    return {
        "composition": "MotionMap",
        "props": {
            "theme": ctx.theme,
            "places": [
                {"name": p.name, "lat": round(p.lat, 2), "lon": round(p.lon, 2), "kind": p.kind}
                for p in places
            ],
            "highlight": maps_stage.motion_feature(places[0]),
            "borders": _motion_borders(maps_stage, lon0, lat0, span, aspect),
            "neighbours": _motion_neighbours(maps_stage, places, lon0, lat0, span, aspect),
            "label": beat.text or places[0].name,
            "route": is_route,
        },
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


# --- chart ----------------------------------------------------------------

def build_chart(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``chart`` beat -> a MotionChart job. Mirrors
    ``charts.render_chart_assets``'s own dispatch exactly: a beat carrying
    ``years`` becomes the years/ruler strip, otherwise ``chart_type`` picks
    bars growing, a line drawing, or pie/donut slices sweeping."""
    data = beat.data or {}
    if not data:
        return None
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    title = data.get("title") or beat.text or None

    if data.get("years"):
        years = [str(y) for y in data["years"]]
        active = data.get("active")
        props: Dict[str, Any] = {
            "theme": ctx.theme, "mode": "years", "years": years,
            "active": str(active) if active is not None else (years[-1] if years else ""),
            "title": title,
        }
    else:
        labels, values = data.get("labels"), data.get("values")
        if not labels or not values:
            return None
        chart_type = str(data.get("chart_type", "")).lower()
        mode = "pie" if chart_type == "pie" else "line" if chart_type in _CHART_LINE_TYPES else "bar"
        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError):
            return None
        props = {
            "theme": ctx.theme, "mode": mode, "labels": [str(l) for l in labels],
            "values": values, "unit": data.get("unit") or "", "title": title,
        }

    return {
        "composition": "MotionChart", "props": props,
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


# --- stat callout -----------------------------------------------------------

def build_stat(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``stat_callout`` beat -> a MotionStat job: a big number counting up to
    ``data["value"]`` plus its label. Mirrors ``cards.py``'s own gate
    (``beat.data.get("value") is not None``) -- a stat beat with no number to
    count has nothing for a counter to animate, so it is left for cards.py's
    plain text-card path exactly as today."""
    data = beat.data or {}
    if data.get("value") is None:
        return None
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "MotionStat",
        "props": {
            "theme": ctx.theme,
            "value": float(data["value"]),
            "prefix": data.get("prefix") or "",
            "suffix": data.get("suffix") or "",
            "decimals": int(data.get("decimals") or 0),
            "label": beat.subtext or "",
        },
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


# --- timeline --------------------------------------------------------------

def build_timeline(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``timeline`` beat -> a MotionTimeline job: the ruler draws, the marker
    slides to the active tick, the headline (the "big figure") counts or
    pops in, with the caption below. Mirrors ``graphics.render_timeline``'s
    own defaults (``DRAWN_CUTAWAY_KINDS`` draws every ``timeline`` beat
    unconditionally, so this never reports "no content")."""
    data = beat.data or {}
    ticks = [str(t) for t in (data.get("ticks") or [])] or ["", ""]
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "MotionTimeline",
        "props": {
            "theme": ctx.theme,
            "ticks": ticks,
            "active": int(data.get("active", len(ticks) - 1)),
            "headline": (beat.text or "").strip(),
            "caption": (beat.subtext or "").strip(),
        },
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }
