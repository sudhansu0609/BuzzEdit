"""Motion Gallery (MOTION_GRAPHICS_PLAN.md M11): render every registered
motion composition with its own sample props, once per theme, into a dated
folder with an index.html -- so the creator can see what the engine can do
without hunting through test output or needing a real project.

The composition list and each one's own ``defaultProps`` come straight from
the Node registry (``motion/src/registry.ts``) through a new ``--dump`` mode
on ``render.mjs`` -- never hand-copied here, so a new M-item composition
shows up in the gallery with no change to this file. Every job in the
gallery is one Node batch (``motion.py``'s own ``_run_batch``), at
below-normal priority, exactly like a real presentation pass's motion
renders.

The five transition stingers (``ShapeWipe``/``Iris``/``Ink``/``Bars``/
``PaperTear``) render transparent (ProRes 4444) -- on their own that is just
a black square in a plain ``<video>`` tag, so each is rendered to a scratch
file first and then composited over a neutral vertical gradient with ffmpeg,
the same way a real render composites it over footage, before being written
into the gallery folder.
"""

import hashlib
import html
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import BASE_DIR, FFMPEG_BIN

from .motion import MOTION_DIR, MOTION_PRIORITY_FLAGS, RENDER_SCRIPT, _node_path, _run_batch, theme_for

logger = logging.getLogger("presentation.motion_gallery")

# BASE_DIR is BuzzEdit's own root (backend's parent); its parent is the
# Buzcaf_Media folder that also holds BuzzcafStudio and Samples -- samples and
# tests live under Samples, never inside a project folder (creator rule).
SAMPLES_ROOT = BASE_DIR.parent / "Samples"
GALLERY_ROOT = SAMPLES_ROOT / "motion_gallery"

DEFAULT_THEMES: Tuple[str, ...] = ("clean", "noir")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# A gallery theme name maps to a genre whose own style_for() resolves to it
# (charts.style_for / maps.style_for: "noir" for horror/true_crime, "clean"
# otherwise) -- so theme_for() is reused exactly as every real render uses
# it, instead of a second theme table living here.
_THEME_GENRE: Dict[str, str] = {"noir": "horror"}

# Only the five M3 transition stingers render transparent (motion_transitions
# .build_transition_stinger is the only BUILDERS entry that sets
# "transparent": True); everything else, including Parallax, is opaque.
_TRANSPARENT_IDS = {"ShapeWipe", "Iris", "Ink", "Bars", "PaperTear"}

_DUMP_TIMEOUT_S = 180.0
_COMPOSITE_TIMEOUT_S = 120.0

# The directive string an agent actually writes to place each composition --
# copied from presentation/script.py's own directive grammar
# (directive_beats / its _*_options helpers), not invented here. Parallax has
# none: motion_parallax.py applies it automatically to a share of
# already-placed stills, with no beat kind or directive of its own.
_DIRECTIVES: Dict[str, str] = {
    "KineticText": "[kinetic: text with *marks* | style=slam|stack|reveal|typewriter]",
    "ShapeWipe": "[fx: shape_wipe dur=0.8]",
    "Iris": "[fx: iris dur=0.8]",
    "Ink": "[fx: ink dur=0.8]",
    "Bars": "[fx: bars dur=0.8]",
    "PaperTear": "[fx: paper_tear dur=0.8]",
    "MotionMap": "[map: Place Name]",
    "MotionChart": "[chart: bar: A=40, B=12]",
    "MotionStat": "[stat: 25% of viewers ...]",
    "MotionTimeline": "[timeline: 1975, 1980, *1985*, 1990 | Headline text]",
    "StickScene": "[stick: action=swim | scene=escaping a flooded house | prop=ship]",
    "PaperCollage": "[collage: prompt A ; prompt B ; prompt C | caption=Three places he ...]",
    "FlowDiagram": "[flow: step one -> step two -> step three]",
    "IconList": "[icons: item one, item two, item three]",
    "UiMockup": "[ui: post | text=... | name=... | handle=... | meta=...]",
    "Callout": "[callout: <image prompt> | mark=circle | at=center | label=...]",
    "Parallax": "(no directive -- applied automatically to a share of already-placed stills)",
}


def directive_for(comp_id: str) -> str:
    return _DIRECTIVES.get(comp_id, "")


def dump_compositions(timeout_s: float = _DUMP_TIMEOUT_S) -> List[Dict[str, Any]]:
    """Every registered composition's id/defaultProps/own default size, read
    straight out of the bundle through ``render.mjs --dump``."""
    node = _node_path()
    if not node:
        raise RuntimeError("Motion engine not available: node is not on PATH")
    if not RENDER_SCRIPT.exists():
        raise RuntimeError("Motion engine not available: render.mjs is missing")
    with tempfile.TemporaryDirectory(prefix="motion_dump_") as tmp:
        out_path = Path(tmp) / "compositions.json"
        cmd = [node, str(RENDER_SCRIPT), "--dump", str(out_path)]
        try:
            proc = subprocess.run(cmd, cwd=str(MOTION_DIR), capture_output=True, text=True,
                                  timeout=timeout_s, creationflags=MOTION_PRIORITY_FLAGS)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Listing motion compositions timed out after {timeout_s:.0f}s") from e
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"Could not list motion compositions: {(proc.stderr or proc.stdout or '')[-800:]}")
        return json.loads(out_path.read_text(encoding="utf-8"))


def _theme_dict(theme_name: str) -> Dict[str, Any]:
    genre = _THEME_GENRE.get(theme_name, "general")
    return theme_for(genre, "real")


def _today_folder() -> Path:
    return GALLERY_ROOT / time.strftime("%Y-%m-%d")


def _composite_over_gradient(overlay_path: Path, out_path: Path, width: int, height: int,
                             fps: float, frames: int, timeout_s: float = _COMPOSITE_TIMEOUT_S) -> None:
    """Composites a transparent render over a plain vertical gradient, so a
    transition stinger's own alpha window shows instead of reading as a
    black square in a plain ``<video>`` tag."""
    duration = max(0.1, frames / max(1.0, fps))
    gradient = (f"nullsrc=size={int(width)}x{int(height)}:rate={fps}:duration={duration:.3f},"
               "format=rgba,"
               "geq=r='86+42*(Y/H)':g='90+42*(Y/H)':b='104+46*(Y/H)':a=255")
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", gradient,
           "-i", str(overlay_path),
           "-filter_complex", "[0:v][1:v]overlay=format=auto[vout]",
           "-map", "[vout]", "-frames:v", str(int(frames)), "-r", str(fps),
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                          creationflags=MOTION_PRIORITY_FLAGS)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg could not composite {overlay_path.name}: {(proc.stderr or '')[-500:]}")


def _safe_name(comp_id: str) -> str:
    return "".join(c for c in comp_id if c.isalnum()) or "comp"


def build_jobs(comps: Sequence[Dict[str, Any]], themes: Sequence[str], width: int, height: int,
              folder: Path, scratch: Path
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, Path, Path]], List[Dict[str, str]]]:
    """Every (composition, theme) render job -- opaque jobs write straight
    into ``folder``, transparent ones (the stingers) render into ``scratch``
    first and are composited into ``folder`` afterwards.

    Returns ``(jobs, by_job_id, cards)`` -- ``by_job_id`` maps a job id to
    ``(theme, render_out, final_out)``; ``cards`` is every (composition,
    theme) pair for the index, whether or not its job later succeeds.
    """
    jobs: List[Dict[str, Any]] = []
    by_job: Dict[str, Tuple[str, Path, Path]] = {}
    cards: List[Dict[str, str]] = []
    for comp in comps:
        comp_id = str(comp.get("id") or "")
        if not comp_id:
            continue
        fps = float(comp.get("fps") or 30)
        frames = max(1, int(comp.get("durationInFrames") or 30))
        default_props = dict(comp.get("defaultProps") or {})
        transparent = comp_id in _TRANSPARENT_IDS
        safe = _safe_name(comp_id)
        for theme_name in themes:
            props = dict(default_props)
            if "theme" in default_props:
                props["theme"] = _theme_dict(theme_name)
            final_out = folder / f"{safe}_{theme_name}.mp4"
            render_out = (scratch / f"{safe}_{theme_name}.mov") if transparent else final_out
            job_id = f"{safe}_{theme_name}"
            job: Dict[str, Any] = {
                "id": job_id, "composition": comp_id, "props": props,
                "width": int(width), "height": int(height), "fps": fps, "frames": frames,
                "out": str(render_out), "transparent": transparent, "concurrency": 1,
            }
            if comp_id == "Parallax":
                job["gl"] = "angle"
            jobs.append(job)
            by_job[job_id] = (theme_name, render_out, final_out)
            cards.append({"id": comp_id, "theme": theme_name, "file": final_out.name,
                          "directive": directive_for(comp_id)})
    return jobs, by_job, cards


def _write_index(folder: Path, cards: List[Dict[str, str]], errors: List[Dict[str, str]]) -> None:
    rows: List[str] = []
    for card in sorted(cards, key=lambda c: (c["id"], c["theme"])):
        if not (folder / card["file"]).exists():
            continue
        rows.append(
            '<div class="card">'
            f'<video src="{html.escape(card["file"])}" controls loop muted playsinline></video>'
            '<div class="meta">'
            f'<strong>{html.escape(card["id"])}</strong> &middot; {html.escape(card["theme"])}'
            f'<code>{html.escape(card["directive"])}</code>'
            '</div></div>'
        )
    error_block = ""
    if errors:
        items = "".join(f"<li>{html.escape(str(e.get('id', '')))}: {html.escape(str(e.get('reason', '')))}</li>"
                        for e in errors)
        error_block = f'<h2>Failed ({len(errors)})</h2><ul>{items}</ul>'
    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Motion Gallery -- {html.escape(folder.name)}</title>
<style>
body {{ background:#15161a; color:#e8e6e0; font-family:"Segoe UI", Arial, sans-serif; margin:24px; }}
h1 {{ font-weight:600; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:20px; }}
.card {{ background:#1f2128; border-radius:10px; overflow:hidden; }}
.card video {{ width:100%; display:block; background:#000; }}
.card .meta {{ padding:10px 14px; font-size:13px; }}
.card code {{ display:block; margin-top:6px; color:#9ad3ff; word-break:break-word; white-space:pre-wrap; }}
</style>
</head>
<body>
<h1>Motion Gallery -- {html.escape(folder.name)}</h1>
<p>{len(rows)} render(s).</p>
<div class="grid">{''.join(rows)}</div>
{error_block}
</body>
</html>"""
    (folder / "index.html").write_text(doc, encoding="utf-8")


def run_gallery(themes: Optional[Sequence[str]] = None, width: int = DEFAULT_WIDTH,
                height: int = DEFAULT_HEIGHT, comps: Optional[Sequence[Dict[str, Any]]] = None,
                progress_cb: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Renders every registered composition's defaultProps, once per theme,
    into a dated folder under ``GALLERY_ROOT`` with an ``index.html``.

    ``comps`` lets a caller pass an already-dumped composition list (the
    route does, so the creator sees "Rendering N effects..." immediately
    instead of waiting on a second dump); None dumps it here.

    Returns ``{"folder": str, "count": int, "errors": [...], "rendered": N}``.
    """
    report = progress_cb or (lambda fraction, message="": None)
    theme_list = [str(t) for t in themes] if themes else list(DEFAULT_THEMES)
    width = int(width) if width else DEFAULT_WIDTH
    height = int(height) if height else DEFAULT_HEIGHT

    from runtime.master_control import wait_if_paused
    wait_if_paused("motion_gallery")

    report(0.05, "Listing motion compositions")
    comp_list = list(comps) if comps is not None else dump_compositions()
    if not comp_list:
        raise RuntimeError("Motion engine returned no registered compositions")

    folder = _today_folder()
    folder.mkdir(parents=True, exist_ok=True)

    scratch = Path(tempfile.mkdtemp(prefix="motion_gallery_scratch_"))
    try:
        jobs, by_job, cards = build_jobs(comp_list, theme_list, width, height, folder, scratch)
        total = len(jobs)
        if not total:
            _write_index(folder, cards, [])
            return {"folder": str(folder), "count": 0, "errors": [], "rendered": 0}

        report(0.1, f"Rendering {total} effect(s)")
        results = _run_batch(jobs, scratch)

        count = 0
        errors: List[Dict[str, str]] = []
        for job in jobs:
            theme_name, render_out, final_out = by_job[job["id"]]
            result = results.get(job["id"])
            if not (result and result.get("ok") and render_out.exists()):
                reason = (result or {}).get("error") or "render failed"
                errors.append({"id": job["id"], "reason": str(reason)[:200]})
                continue
            if render_out != final_out:
                try:
                    _composite_over_gradient(render_out, final_out, job["width"], job["height"],
                                             job["fps"], job["frames"])
                except Exception as e:  # noqa: BLE001 - one bad composite must not fail the gallery
                    logger.warning("Motion gallery composite failed for %s: %s", job["id"], e)
                    errors.append({"id": job["id"], "reason": f"composite failed: {e}"[:200]})
                    continue
                finally:
                    render_out.unlink(missing_ok=True)
            if final_out.exists():
                count += 1
            else:
                errors.append({"id": job["id"], "reason": "no output file"})

        report(0.95, "Writing the index")
        _write_index(folder, cards, errors)
        report(1.0, "Motion gallery ready")
        return {"folder": str(folder), "count": count, "errors": errors, "rendered": total}
    finally:
        import shutil as _shutil
        _shutil.rmtree(scratch, ignore_errors=True)
