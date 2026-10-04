"""Stage C fallback — free stock photos/video for a beat ComfyUI could not
make a picture for.

Studio's own asset sourcing is licensed/generated only (ComfyUI, running on
the user's own machine). This module is the one opt-in exception:
`PresentationSettings.allow_free_stock`, plus a configured key, lets a beat
that would otherwise stay blank pull from Pexels or Pixabay instead — both
free, both requiring the user's own free API key. Never a paid source, never
scraped.

Same rule as the rest of Stage C: one missing picture, never a failed night.
Every search/download here is wrapped so a network hiccup costs one beat and
nothing else.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .assets import _cache_dir, _cached, _probe, _IMAGE_SUFFIXES, _VIDEO_SUFFIXES
from .models import Asset, Beat, PresentationSettings

logger = logging.getLogger("presentation.stock")

SEARCH_TIMEOUT_S = 6.0
DOWNLOAD_TIMEOUT_S = 20.0

PEXELS_PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PIXABAY_IMAGE_URL = "https://pixabay.com/api/"
PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"

# "Prefer clips 3-10s" from the brief.
VIDEO_DURATION_MIN_S = 3.0
VIDEO_DURATION_MAX_S = 10.0
# "<=1080p" is a hard cap on the rendition we actually download.
MAX_VIDEO_HEIGHT = 1080

PEXELS_LICENCE = ("Pexels License — free for commercial & personal use, "
                  "no attribution required (https://www.pexels.com/license/)")
PIXABAY_LICENCE = ("Pixabay License — free for commercial use, no "
                   "attribution required (https://pixabay.com/service/license/)")


def _api_key(name: str) -> Optional[str]:
    try:
        from store.app_settings import get_stock_api_key
        return get_stock_api_key(name)
    except Exception:
        return None


def stock_available(settings: PresentationSettings) -> bool:
    """Whether the stock fallback can even be attempted: the project opted
    in AND at least one of the two keys is configured (stored or via its
    environment variable)."""
    if not getattr(settings, "allow_free_stock", False):
        return False
    return bool(_api_key("pexels_api_key") or _api_key("pixabay_api_key"))


def _orientation_for(canvas_size: Optional[Tuple[int, int]]) -> str:
    if canvas_size and all(canvas_size) and canvas_size[1] > canvas_size[0]:
        return "portrait"
    return "landscape"


def _canvas_aspect(canvas_size: Optional[Tuple[int, int]]) -> float:
    if canvas_size and all(canvas_size):
        return canvas_size[0] / canvas_size[1]
    return 16.0 / 9.0


def _query_for(beat: Beat) -> str:
    text = beat.topic or beat.prompt() or beat.image_prompt or ""
    words = str(text).split()
    return " ".join(words[:8]) or "video background"


# --- low-level HTTP (kept as thin, mockable seams) --------------------------

def _http_get_json(url: str, params: Dict[str, Any],
                   headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import requests
    resp = requests.get(url, params=params, headers=headers or {}, timeout=SEARCH_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _http_download(url: str) -> bytes:
    import requests
    resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


# --- search + normalise ------------------------------------------------------
#
# Every source's hits get flattened into the same shape so picking and
# downloading do not need to know which API they came from:
# {source, id, page_url, download_url, author, width, height, duration,
#  licence, ext}

def search_pexels_photos(query: str, orientation: str, api_key: str) -> List[Dict[str, Any]]:
    data = _http_get_json(PEXELS_PHOTO_SEARCH_URL,
                          {"query": query, "orientation": orientation, "per_page": 5},
                          headers={"Authorization": api_key})
    hits = []
    for photo in data.get("photos", []) or []:
        src = photo.get("src") or {}
        download_url = src.get("large2x") or src.get("large") or src.get("original")
        if not download_url:
            continue
        hits.append({
            "source": "pexels", "id": photo.get("id"), "page_url": photo.get("url", ""),
            "download_url": download_url, "author": photo.get("photographer", "") or "",
            "width": int(photo.get("width") or 0), "height": int(photo.get("height") or 0),
            "duration": 0.0, "licence": PEXELS_LICENCE, "ext": ".jpg",
        })
    return hits


def search_pexels_videos(query: str, orientation: str, api_key: str) -> List[Dict[str, Any]]:
    data = _http_get_json(PEXELS_VIDEO_SEARCH_URL,
                          {"query": query, "orientation": orientation, "per_page": 5},
                          headers={"Authorization": api_key})
    hits = []
    for video in data.get("videos", []) or []:
        files = [f for f in (video.get("video_files") or [])
                if (f.get("file_type") or "").startswith("video")]
        capped = [f for f in files if int(f.get("height") or 0) <= MAX_VIDEO_HEIGHT]
        pool = capped or files
        if not pool:
            continue
        best = max(pool, key=lambda f: int(f.get("height") or 0))
        hits.append({
            "source": "pexels", "id": video.get("id"), "page_url": video.get("url", ""),
            "download_url": best.get("link"),
            "author": (video.get("user") or {}).get("name", "") or "",
            "width": int(best.get("width") or 0), "height": int(best.get("height") or 0),
            "duration": float(video.get("duration") or 0.0), "licence": PEXELS_LICENCE,
            "ext": ".mp4",
        })
    return hits


def search_pixabay_images(query: str, orientation: str, api_key: str) -> List[Dict[str, Any]]:
    orientation_param = "horizontal" if orientation == "landscape" else "vertical"
    data = _http_get_json(PIXABAY_IMAGE_URL,
                          {"key": api_key, "q": query, "image_type": "photo",
                           "orientation": orientation_param, "per_page": 5, "safesearch": "true"})
    hits = []
    for hit in data.get("hits", []) or []:
        download_url = hit.get("largeImageURL") or hit.get("webformatURL")
        if not download_url:
            continue
        hits.append({
            "source": "pixabay", "id": hit.get("id"), "page_url": hit.get("pageURL", ""),
            "download_url": download_url, "author": hit.get("user", "") or "",
            "width": int(hit.get("imageWidth") or 0), "height": int(hit.get("imageHeight") or 0),
            "duration": 0.0, "licence": PIXABAY_LICENCE, "ext": ".jpg",
        })
    return hits


def search_pixabay_videos(query: str, orientation: str, api_key: str) -> List[Dict[str, Any]]:
    data = _http_get_json(PIXABAY_VIDEO_URL,
                          {"key": api_key, "q": query, "per_page": 5, "safesearch": "true"})
    hits = []
    for hit in data.get("hits", []) or []:
        renditions = hit.get("videos") or {}
        capped = [r for r in renditions.values()
                 if r.get("url") and int(r.get("height") or 0) <= MAX_VIDEO_HEIGHT]
        pool = capped or [r for r in renditions.values() if r.get("url")]
        if not pool:
            continue
        best = max(pool, key=lambda r: int(r.get("height") or 0))
        hits.append({
            "source": "pixabay", "id": hit.get("id"), "page_url": hit.get("pageURL", ""),
            "download_url": best.get("url"), "author": hit.get("user", "") or "",
            "width": int(best.get("width") or 0), "height": int(best.get("height") or 0),
            "duration": float(hit.get("duration") or 0.0), "licence": PIXABAY_LICENCE,
            "ext": ".mp4",
        })
    return hits


# --- picking -----------------------------------------------------------------

def _score(hit: Dict[str, Any], canvas_aspect: float, is_video: bool) -> float:
    score = 0.0
    width, height = hit.get("width") or 0, hit.get("height") or 0
    if width and height:
        score -= abs((width / height) - canvas_aspect) * 5.0
    if is_video:
        duration = hit.get("duration") or 0.0
        if VIDEO_DURATION_MIN_S <= duration <= VIDEO_DURATION_MAX_S:
            score += 10.0
        elif duration > 0:
            mid = (VIDEO_DURATION_MIN_S + VIDEO_DURATION_MAX_S) / 2
            score -= min(abs(duration - mid), 20.0)
    return score


def pick_best(hits: List[Dict[str, Any]], canvas_aspect: float,
             is_video: bool) -> Optional[Dict[str, Any]]:
    """The highest-scoring candidate across every source's hits, or None."""
    if not hits:
        return None
    return max(hits, key=lambda h: _score(h, canvas_aspect, is_video))


# --- orchestration -------------------------------------------------------

def _stock_cache_key(beat_id: str, source: str, hit_id: Any) -> str:
    raw = f"stock|{beat_id}|{source}|{hit_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _write_stock_sidecar(directory: Path, key: str, beat: Beat, hit: Dict[str, Any],
                         query: str) -> None:
    try:
        (directory / f"{key}.json").write_text(json.dumps({
            "beat_id": beat.id, "topic": beat.topic, "kind": beat.kind, "query": query,
            "source": hit["source"], "page_url": hit.get("page_url", ""),
            "author": hit.get("author", ""), "licence": hit.get("licence", ""),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


def _search_all_sources(query: str, orientation: str, is_video: bool) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    pexels_key = _api_key("pexels_api_key")
    if pexels_key:
        try:
            fn = search_pexels_videos if is_video else search_pexels_photos
            hits.extend(fn(query, orientation, pexels_key))
        except Exception as e:
            logger.info("Pexels %s search failed for %r: %s",
                       "video" if is_video else "photo", query, e)
    pixabay_key = _api_key("pixabay_api_key")
    if pixabay_key:
        try:
            fn = search_pixabay_videos if is_video else search_pixabay_images
            hits.extend(fn(query, orientation, pixabay_key))
        except Exception as e:
            logger.info("Pixabay %s search failed for %r: %s",
                       "video" if is_video else "photo", query, e)
    return hits


def _fetch_one(beat: Beat, directory: Path,
               canvas_size: Optional[Tuple[int, int]]) -> Optional[Tuple[Asset, Dict[str, str]]]:
    """One beat, one attempt: video sources first when the beat wants video,
    falling back to a still (the same "a picture beats nothing" rule the
    ComfyUI path already follows), else straight to a still."""
    orientation = _orientation_for(canvas_size)
    canvas_aspect = _canvas_aspect(canvas_size)
    query = _query_for(beat)
    wants_video = beat.kind == "broll_video"

    hit = None
    is_video = False
    if wants_video:
        hits = _search_all_sources(query, orientation, is_video=True)
        hit = pick_best(hits, canvas_aspect, is_video=True)
        is_video = hit is not None
    if hit is None:
        hits = _search_all_sources(query, orientation, is_video=False)
        hit = pick_best(hits, canvas_aspect, is_video=False)
        is_video = False
    if hit is None or not hit.get("download_url"):
        return None

    key = _stock_cache_key(beat.id, hit["source"], hit["id"])
    existing = _cached(directory, key)
    if existing is None:
        try:
            content = _http_download(hit["download_url"])
        except Exception as e:
            logger.info("Stock download failed for beat %s (%s): %s", beat.id, hit["source"], e)
            return None
        destination = directory / f"{key}{hit['ext']}"
        try:
            destination.write_bytes(content)
        except Exception as e:
            logger.warning("Could not write stock asset for beat %s: %s", beat.id, e)
            return None
        _write_stock_sidecar(directory, key, beat, hit, query)
        existing = destination

    w, h, duration = _probe(existing)
    asset = Asset(
        beat_id=beat.id,
        kind="video" if existing.suffix.lower() in _VIDEO_SUFFIXES else "image",
        path=str(existing), width=w, height=h, duration_s=duration, cache_hit=False,
        seed=0, prompt_sha=key,
        source=hit["source"], stock_url=hit.get("page_url", ""),
        stock_author=hit.get("author", ""), stock_licence=hit.get("licence", ""),
    )
    record = {"beat_id": beat.id, "source": hit["source"],
             "url": hit.get("page_url", ""), "author": hit.get("author", "")}
    return asset, record


async def fill_missing(beats: List[Beat], project_dir: Path, settings: PresentationSettings,
                       failures: List[Dict[str, str]],
                       canvas_size: Optional[Tuple[int, int]] = None
                       ) -> Tuple[List[Asset], List[Dict[str, str]], List[Dict[str, str]]]:
    """For every beat named in `failures`, try the free stock fallback.

    Returns (new_assets, still_failed, stock_used) — `still_failed` keeps the
    original reason for beats stock could not fill either, so nothing is ever
    silently dropped from the report.
    """
    if not stock_available(settings) or not failures:
        return [], list(failures), []

    by_id = {b.id: b for b in beats}
    new_assets: List[Asset] = []
    still_failed: List[Dict[str, str]] = []
    stock_used: List[Dict[str, str]] = []
    directory = _cache_dir(project_dir)

    for failure in failures:
        beat = by_id.get(failure.get("beat_id"))
        if beat is None:
            still_failed.append(failure)
            continue
        try:
            import asyncio
            result = await asyncio.to_thread(_fetch_one, beat, directory, canvas_size)
        except Exception as e:
            logger.info("Stock fallback errored for beat %s: %s", beat.id, e)
            result = None
        if result is None:
            still_failed.append(failure)
            continue
        asset, record = result
        new_assets.append(asset)
        stock_used.append(record)
        logger.info("Beat %s: filled from free stock (%s)", beat.id, record["source"])

    return new_assets, still_failed, stock_used
