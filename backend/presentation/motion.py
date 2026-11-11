"""The Python side of the Remotion motion-template engine
(MOTION_GRAPHICS_PLAN.md M1 Architecture).

Builds a Node render job for every beat a registered builder knows how to
make, runs ``BuzzEdit/motion/render.mjs`` on them in small batches (one Node
subprocess per batch, BELOW_NORMAL priority, no console window -- the PC must
not lag while a video is being cut), and turns a finished render into an
``Asset`` the rest of the presentation pass places exactly like a generated
clip (``source="motion"``; see placement.py).

A beat this module cannot render -- the engine is not installed, a job times
out, Node exits non-zero -- is simply left out of ``render_motion_assets``'s
return. ``director._draw_local_assets`` then tries ``render_fallback`` for it
(a static PIL card, today only for ``kinetic_text``) and otherwise drops it
with a reason. No motion failure may ever fail a render.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import BASE_DIR
from utils.proc import NO_WINDOW

from .models import Asset, Beat, PresentationSettings

logger = logging.getLogger("presentation.motion")

MOTION_DIR = BASE_DIR / "motion"
RENDER_SCRIPT = MOTION_DIR / "render.mjs"

# Bumped whenever a builder's prop shape changes in a way that should
# invalidate every cached render; folded into every cache key.
ENGINE_VERSION = 1

# Below-normal priority, no console window -- a Chromium render must never be
# the thing that makes the PC lag mid-shoot (GPU_MODEL_POLICY / creator rule).
_BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if sys.platform == "win32" else 0
MOTION_PRIORITY_FLAGS = NO_WINDOW | _BELOW_NORMAL

# Timeout for one Node batch, scaled to how many frames it has to render --
# generous enough for a cold Chromium start under a busy GPU.
SECONDS_PER_FRAME = 0.6
MIN_TIMEOUT_S = 60.0

# A motion asset is rendered at the beat's planned length, clamped (Architecture).
MOTION_BEAT_SECONDS_MIN = 2.5
MOTION_BEAT_SECONDS_MAX = 12.0
DEFAULT_FPS = 30.0

_KINETIC_STYLES = ("slam", "stack", "reveal", "typewriter")

# Folded into every render cache key (FX1) so a composition/dependency edit
# under motion/src or a package-lock bump invalidates every stale cached
# render instead of silently reusing it. mtime+size per file, not file
# contents -- cheap, and editing a .tsx/.ts always changes its mtime.
# Computed once per process and memoized.
_engine_fingerprint_cache: Optional[str] = None


def _engine_fingerprint() -> str:
    global _engine_fingerprint_cache
    if _engine_fingerprint_cache is not None:
        return _engine_fingerprint_cache
    paths: List[Path] = []
    src_dir = MOTION_DIR / "src"
    if src_dir.is_dir():
        paths.extend(sorted(p for p in src_dir.rglob("*") if p.is_file()))
    lock = MOTION_DIR / "package-lock.json"
    if lock.exists():
        paths.append(lock)
    digest = hashlib.sha1()
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        digest.update(str(p.relative_to(MOTION_DIR)).encode("utf-8"))
        digest.update(str(st.st_mtime_ns).encode("utf-8"))
        digest.update(str(st.st_size).encode("utf-8"))
    _engine_fingerprint_cache = digest.hexdigest()[:12]
    return _engine_fingerprint_cache


@dataclass
class MotionContext:
    """What every builder needs besides the beat itself."""
    width: int
    height: int
    fps: float
    genre: str
    theme: Dict[str, Any]
    settings: PresentationSettings


# --- availability ------------------------------------------------------------

_availability_cache: Optional[bool] = None


def _node_path() -> Optional[str]:
    import shutil
    return shutil.which("node")


def available() -> bool:
    """node on PATH + motion/node_modules + render.mjs present.

    A fast, cheap precondition check -- not a guarantee the engine actually
    renders (the browser may still need to download, a job may still fail).
    The real proof is installer/selftest.py's ``--selftest`` pass at install
    time; every render attempt here still falls back per-beat on its own
    failure regardless of what this returns.
    """
    global _availability_cache
    if _availability_cache is not None:
        return _availability_cache
    ok = bool(_node_path()) and RENDER_SCRIPT.exists() and (MOTION_DIR / "node_modules").is_dir()
    _availability_cache = ok
    return ok


def reset_availability_cache() -> None:
    """Tests call this between cases; available() otherwise memoises per process."""
    global _availability_cache
    _availability_cache = None


# --- theme --------------------------------------------------------------------

def _hex(rgb: Sequence[int]) -> str:
    r, g, b = (int(c) for c in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def theme_for(genre: str, art_style: str = "real") -> Dict[str, Any]:
    """The Theme every composition takes as ``props.theme`` -- built from the
    same tables the PIL renderers already use (maps.STYLES, charts.STYLES,
    art_style.card_style), so a motion composition and its PIL fallback always
    read as the same video."""
    from . import art_style as art_style_stage
    from . import charts as charts_stage
    from . import maps as maps_stage

    style_name = charts_stage.style_for(genre)
    chart_look = charts_stage.STYLES.get(style_name, charts_stage.STYLES["clean"])
    map_look = maps_stage.STYLES.get(style_name, maps_stage.STYLES["clean"])
    card = art_style_stage.card_style(art_style)
    accent = card.get("color")
    font = card.get("font_family") or "Poppins"
    return {
        "bg": _hex(chart_look["bg"]),
        "panel": _hex(chart_look["panel"]),
        "fg": _hex(chart_look["text"]),
        "muted": _hex(chart_look["muted"]),
        "accent": accent if isinstance(accent, str) and accent.startswith("#") else _hex(chart_look["bar"]),
        "accent2": _hex(chart_look["bar_alt"]),
        "water": _hex(map_look["water"]),
        "land": _hex(map_look["land"]),
        "border": _hex(map_look["border"]),
        "fontHead": font,
        "fontBody": "Poppins",
        # Hand-drawn art styles get the paper-blotch overlay instead of a flat fill.
        "paper": art_style in ("stick_figure", "cartoon", "painterly", "illustrated"),
        "grain": bool(map_look.get("grain")) or style_name == "noir",
        "noir": style_name == "noir",
    }


# --- builders -------------------------------------------------------------

def _parse_marks(text: str) -> List[Dict[str, Any]]:
    """"This is *huge*" -> [{"text":"This"},{"text":"is"},{"text":"huge","mark":True}].
    Mirrors motion/src/compositions/KineticText.tsx::parseMarks."""
    words: List[Dict[str, Any]] = []
    for raw in text.strip().split():
        mark = len(raw) > 2 and raw.startswith("*") and raw.endswith("*")
        words.append({"text": raw[1:-1] if mark else raw, "mark": mark})
    return words


def _beat_duration(beat: Beat) -> float:
    duration = beat.planned_duration_s or (beat.end_s - beat.start_s) or 4.0
    return max(MOTION_BEAT_SECONDS_MIN, min(MOTION_BEAT_SECONDS_MAX, duration))


def build_kinetic_text(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """`[kinetic: text with *marks* | style=...]` -> a KineticText job spec."""
    text = (beat.text or beat.topic or "").strip()
    if not text:
        return None
    words = _parse_marks(text)
    if not words:
        return None
    style = (beat.data or {}).get("style", "slam")
    if style not in _KINETIC_STYLES:
        style = "slam"
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "KineticText",
        "props": {"words": words, "style": style, "theme": ctx.theme},
        "width": ctx.width,
        "height": ctx.height,
        "fps": ctx.fps,
        "frames": frames,
        "transparent": False,
    }


# One entry per motion-handled beat kind. M2/M3 add one line each here; each
# composition's own builder lives in its own module (motion_data.py,
# motion_transitions.py, ...) the way build_kinetic_text lives in this one.
from .motion_data import build_chart, build_map, build_stat, build_timeline  # noqa: E402
from .motion_transitions import build_transition_stinger  # noqa: E402
from .motion_stick import build_stick_scene  # noqa: E402
from .motion_collage import build_paper_collage  # noqa: E402
from .motion_diagrams import build_flow_diagram, build_icon_list  # noqa: E402
from .motion_ui import build_callout, build_ui_mockup  # noqa: E402

BUILDERS: Dict[str, Callable[[Beat, MotionContext], Optional[Dict[str, Any]]]] = {
    "kinetic_text": build_kinetic_text,
    "map": build_map,
    "chart": build_chart,
    "stat_callout": build_stat,
    "timeline": build_timeline,
    "transition_stinger": build_transition_stinger,
    "stick_scene": build_stick_scene,
    "paper_collage": build_paper_collage,
    "flow_diagram": build_flow_diagram,
    "icon_list": build_icon_list,
    "ui_mockup": build_ui_mockup,
    "callout": build_callout,
}

# A motion kind that has a PIL equivalent gated by its own setting (maps,
# cards) must honour that same setting -- a user who turned maps off must not
# still get a MotionMap. A kind absent here (kinetic_text, chart, timeline,
# transition_stinger) has no such independent toggle and is only gated by
# `settings.motion_graphics`, exactly like its PIL path today.
_KIND_SETTINGS_GATE: Dict[str, str] = {"map": "maps", "stat_callout": "cards"}


# --- rendering --------------------------------------------------------------

def _cache_key(spec: Dict[str, Any]) -> str:
    payload = {
        "engine_version": ENGINE_VERSION,
        "engine_fingerprint": _engine_fingerprint(),
        "composition": spec["composition"],
        "props": spec["props"],
        "width": spec["width"],
        "height": spec["height"],
        "fps": spec["fps"],
        "frames": spec["frames"],
        "transparent": bool(spec.get("transparent")),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _asset_for(beat: Beat, out: Path, spec: Dict[str, Any], cache_hit: bool) -> Asset:
    duration_s = spec["frames"] / max(1.0, float(spec.get("fps") or DEFAULT_FPS))
    return Asset(beat_id=beat.id, kind="video", path=str(out), width=int(spec["width"]),
                height=int(spec["height"]), duration_s=duration_s, cache_hit=cache_hit,
                source="motion")


def _timeout_for(jobs: Sequence[Dict[str, Any]]) -> float:
    total_frames = sum(int(j.get("frames", 0)) for j in jobs)
    return max(MIN_TIMEOUT_S, total_frames * SECONDS_PER_FRAME)


def _run_batch(jobs: List[Dict[str, Any]], project_dir: Path) -> Dict[str, Dict[str, Any]]:
    """One Node subprocess for this batch; {job_id: result} on success, {} on
    any failure to even run it (a timeout, Node missing, a crash) -- every job
    in the batch is then simply absent from the caller's results and falls
    back like any other failed beat."""
    node = _node_path()
    if not node or not jobs:
        return {}
    work_dir = Path(project_dir) / "assets" / "motion" / ".jobs"
    work_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    jobs_path = work_dir / f"jobs_{stamp}.json"
    results_path = work_dir / f"results_{stamp}.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    cmd = [node, str(RENDER_SCRIPT), str(jobs_path), str(results_path)]
    timeout = _timeout_for(jobs)
    try:
        proc = subprocess.run(cmd, cwd=str(MOTION_DIR), capture_output=True, text=True,
                              timeout=timeout, creationflags=MOTION_PRIORITY_FLAGS)
        if proc.returncode != 0 and not results_path.exists():
            logger.warning("Motion render.mjs exited %s: %s", proc.returncode,
                           (proc.stderr or proc.stdout or "")[-800:])
    except subprocess.TimeoutExpired:
        logger.warning("Motion render.mjs timed out after %.0fs for %d job(s)", timeout, len(jobs))
        return {}
    except Exception as e:  # noqa: BLE001 - a motion failure must never raise
        logger.warning("Motion render.mjs failed to run: %s", e)
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for r in data:
            results[r.get("id")] = r
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read motion render results: %s", e)
    finally:
        for p in (jobs_path, results_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    return results


def render_motion_assets(beats: Sequence[Beat], project_dir: Path, width: int, height: int,
                         genre: str, settings: PresentationSettings
                         ) -> Tuple[List[Asset], List[Dict[str, str]], set]:
    """Motion-template renders for every beat BUILDERS knows how to make.

    Returns ``(assets, failures, handled_beat_ids)`` -- ``handled_beat_ids``
    holds only the beats a motion asset was actually produced for (cached or
    freshly rendered). ``director._draw_local_assets`` skips those before its
    PIL renderers run; a beat in ``failures`` is left for a fallback (or drop)
    instead.
    """
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []
    handled: set = set()
    # A beat whose kind's own setting is off (maps, cards) is skipped here
    # exactly as silently as its PIL path would skip it -- not reported as a
    # motion failure, since there was never anything to draw for it at all.
    wanted = [b for b in beats if b.kind in BUILDERS
             and getattr(settings, _KIND_SETTINGS_GATE.get(b.kind, ""), True)]
    if not wanted:
        return assets, failures, handled
    # Both of these still report a failure per beat rather than returning
    # quietly: the beat came from a `[kinetic: ...]` the user wrote, and
    # director._draw_local_assets only tries a beat's fallback for a beat it
    # finds in `failures` -- an empty list here would drop it with no asset
    # and no reason at all.
    if not settings.motion_graphics:
        return assets, [{"beat_id": b.id, "reason": "motion graphics are off"} for b in wanted], handled
    if not available():
        logger.info("Motion engine not available; %d beat(s) left for fallback", len(wanted))
        return assets, [{"beat_id": b.id, "reason": "motion engine not installed"} for b in wanted], handled

    from runtime.master_control import wait_if_paused

    theme = theme_for(genre, settings.art_style)
    ctx = MotionContext(width=width, height=height, fps=DEFAULT_FPS, genre=genre, theme=theme,
                        settings=settings)
    directory = Path(project_dir) / "assets" / "motion"
    directory.mkdir(parents=True, exist_ok=True)

    pending: List[Dict[str, Any]] = []
    by_job_id: Dict[str, Tuple[Beat, Path, Dict[str, Any]]] = {}
    for beat in wanted:
        builder = BUILDERS[beat.kind]
        try:
            spec = builder(beat, ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("Motion builder for %s (%s) raised: %s", beat.id, beat.kind, e)
            failures.append({"beat_id": beat.id, "reason": f"build failed: {e}"[:200]})
            continue
        if not spec:
            failures.append({"beat_id": beat.id, "reason": "no content for motion beat"})
            continue
        key = _cache_key(spec)
        ext = "mov" if spec.get("transparent") else "mp4"
        out = directory / f"motion_{key}.{ext}"
        if out.exists():
            assets.append(_asset_for(beat, out, spec, cache_hit=True))
            handled.add(beat.id)
            continue
        job = {"id": f"job_{beat.id}", "composition": spec["composition"], "props": spec["props"],
              "width": spec["width"], "height": spec["height"], "fps": spec["fps"],
              "frames": spec["frames"], "out": str(out), "transparent": bool(spec.get("transparent")),
              "concurrency": 2}
        pending.append(job)
        by_job_id[job["id"]] = (beat, out, spec)

    if not pending:
        return assets, failures, handled

    concurrency = max(1, int(getattr(settings, "motion_concurrency", 4) or 4))
    for start in range(0, len(pending), concurrency):
        batch = pending[start:start + concurrency]
        wait_if_paused("motion_render")
        results = _run_batch(batch, project_dir)
        for job in batch:
            beat, out, spec = by_job_id[job["id"]]
            result = results.get(job["id"])
            if result and result.get("ok") and out.exists():
                assets.append(_asset_for(beat, out, spec, cache_hit=False))
                handled.add(beat.id)
            else:
                reason = (result or {}).get("error") or "render failed"
                logger.warning("Motion render failed for beat %s: %s", beat.id, reason)
                failures.append({"beat_id": beat.id, "reason": f"motion render failed: {reason}"[:200]})
    return assets, failures, handled


# --- fallback ----------------------------------------------------------------

def render_fallback(beat: Optional[Beat], project_dir: Path, width: int, height: int) -> Optional[Asset]:
    """A static PIL card for a motion beat the engine could not render.

    kinetic_text's fallback is the same statement-card look graphics.py
    already draws for a `[statement:]` directive, so a failed motion render
    changes only that the beat holds still, never what kind of thing is on
    screen. stick_scene's fallback is a plain PIL stick figure (motion_stick.
    render_stick_still) -- never a ComfyUI generation, since by the time a
    motion failure is known the generation pass over the plan's beats has
    already run and skipped this one. paper_collage's fallback (M5) is one of
    its own already-generated sub-pictures, used as a plain still -- stashed
    on `beat.data["fallback_image"]` by motion_collage.annotate_collage_assets
    right after that same generation pass. flow_diagram/icon_list (M9) fall
    back to the closest existing PIL card graphics.py already has -- a
    vertical numbered-point list and a stack of pills, respectively
    (motion_diagrams.render_flow_fallback / render_icon_list_fallback).
    callout's fallback (M10) is its own already-generated still, placed as a
    plain `broll_image` rather than a motion card -- stashed on the same
    `beat.data["fallback_image"]` key by motion_ui.annotate_callout_assets.
    ui_mockup's fallback (M10) redraws its body text as a statement card
    (motion_ui.render_ui_mockup_fallback), the same way kinetic_text's own
    fallback does. Returns None (drop the beat, with its failure reason)
    when the kind has no fallback that fits -- today every BUILDERS kind but
    kinetic_text, stick_scene, paper_collage, flow_diagram, icon_list,
    callout and ui_mockup falls into that case.
    """
    if beat is None or beat.kind not in (
        "kinetic_text", "stick_scene", "paper_collage", "flow_diagram", "icon_list",
        "callout", "ui_mockup",
    ):
        return None
    if beat.kind in ("flow_diagram", "icon_list"):
        from . import motion_diagrams as motion_diagrams_stage
        directory = Path(project_dir) / "assets" / "motion_fallback"
        directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"{beat.kind}:{beat.id}:{width}x{height}".encode()).hexdigest()[:12]
        path = directory / f"{beat.kind}_fallback_{key}.png"
        if not path.exists():
            try:
                if beat.kind == "flow_diagram":
                    image = motion_diagrams_stage.render_flow_fallback(beat, width, height)
                else:
                    image = motion_diagrams_stage.render_icon_list_fallback(beat, width, height)
                image.save(path)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s fallback for %r failed: %s", beat.kind, beat.topic, e)
                return None
        return Asset(beat_id=beat.id, kind="image", path=str(path), width=width, height=height,
                    cache_hit=True, source="motion_fallback")
    if beat.kind == "paper_collage":
        fallback_image = (beat.data or {}).get("fallback_image")
        if fallback_image and Path(fallback_image).exists():
            return Asset(beat_id=beat.id, kind="image", path=str(fallback_image), width=width,
                        height=height, cache_hit=True, source="motion_fallback")
        return None
    if beat.kind == "callout":
        # A plain broll_image, not a motion card -- motion_ui.expand_callouts
        # already generated this exactly like any other B-roll still, so a
        # failed Callout render just shows it without the mark/magnifier.
        fallback_image = (beat.data or {}).get("fallback_image")
        if fallback_image and Path(fallback_image).exists():
            return Asset(beat_id=beat.id, kind="image", path=str(fallback_image), width=width,
                        height=height, cache_hit=True, source="generated")
        return None
    if beat.kind == "ui_mockup":
        from . import motion_ui as motion_ui_stage
        directory = Path(project_dir) / "assets" / "motion_fallback"
        directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"ui_mockup:{beat.id}:{width}x{height}".encode()).hexdigest()[:12]
        path = directory / f"ui_mockup_fallback_{key}.png"
        if not path.exists():
            try:
                image = motion_ui_stage.render_ui_mockup_fallback(beat, width, height)
                image.save(path)
            except Exception as e:  # noqa: BLE001
                logger.warning("UI-mockup fallback for %r failed: %s", beat.topic, e)
                return None
        return Asset(beat_id=beat.id, kind="image", path=str(path), width=width, height=height,
                    cache_hit=True, source="motion_fallback")
    if beat.kind == "stick_scene":
        from . import motion_stick as motion_stick_stage
        directory = Path(project_dir) / "assets" / "motion_fallback"
        directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"stick:{beat.id}:{width}x{height}".encode()).hexdigest()[:12]
        path = directory / f"stick_fallback_{key}.png"
        if not path.exists():
            try:
                image = motion_stick_stage.render_stick_still(beat, width, height)
                image.save(path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Stick-scene fallback for %r failed: %s", beat.topic, e)
                return None
        return Asset(beat_id=beat.id, kind="image", path=str(path), width=width, height=height,
                    cache_hit=True, source="motion_fallback")
    text = (beat.text or beat.topic or "").strip()
    if not text:
        return None
    from . import graphics as graphics_stage
    directory = Path(project_dir) / "assets" / "motion_fallback"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{text}:{width}x{height}".encode()).hexdigest()[:12]
    path = directory / f"kinetic_fallback_{key}.png"
    if not path.exists():
        try:
            image = graphics_stage.render_statement_card(text, width, height, with_speaker=False)
            image.save(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Kinetic-text fallback for %r failed: %s", beat.topic, e)
            return None
    return Asset(beat_id=beat.id, kind="image", path=str(path), width=width, height=height,
                cache_hit=True, source="motion_fallback")
