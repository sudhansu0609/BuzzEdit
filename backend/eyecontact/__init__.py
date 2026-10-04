"""Teleprompter eye-contact correction.

Reading a prompter that sits beside (or above/below) the lens leaves the eyes aimed
a few degrees off-camera and sweeping along each line. This package re-aims them at
the lens by sliding the real iris pixels inside each eye — no generative model, so
nothing about the face is invented and nothing flickers.

Two GPU passes over the recording, both near NVENC/NVDEC speed:
  1. track  — face + iris landmarks for every frame (cached per source file),
  2. render — per-frame correction from an offline plan, warp, encode, mux.

    from eyecontact import correct_eye_contact, Settings
    report = correct_eye_contact("take1.mp4", settings=Settings(steadiness=0.7))
"""
import hashlib
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from config import DATA_DIR

from .plan import Settings, make_plan

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "eyecontact"
OUTPUT_SUFFIX = "_eyecontact"
TRACK_SHARE = 0.5     # progress split between the two passes (they run at about the same speed)

__all__ = ["Settings", "available", "correct_eye_contact", "default_output", "analyse",
           "preview_clip", "prune_previews"]


def available() -> Tuple[bool, str]:
    """Whether this machine can run the correction, and why not."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No CUDA GPU is available to torch"
    except Exception as e:
        return False, f"torch unavailable ({e})"
    for mod, pkg in (("PyNvVideoCodec", "PyNvVideoCodec"), ("av", "av")):
        try:
            __import__(mod)
        except Exception as e:
            return False, f"{pkg} is not installed ({e})"
    return True, ""


def default_output(src: str) -> str:
    p = Path(src)
    return str(p.with_name(f"{p.stem}{OUTPUT_SUFFIX}.mp4"))


def _cache_path(src: str) -> Path:
    st = os.stat(src)
    key = f"{os.path.abspath(src).lower()}|{st.st_size}|{int(st.st_mtime)}|v2"
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def _track(src, info, progress, cancelled):
    from .tracker import track_video
    cache = _cache_path(src)
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    data = track_video(src, info, progress, cancelled)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, cache)
    return data


PREVIEW_DIR = CACHE_DIR / "preview"
PREVIEW_KEEP = 24     # newest preview files kept; each 10 s 4K pair is ~100 MB


def preview_clip(src: str, start_s: float, duration_s: float) -> str:
    """Cut `duration_s` seconds of `src` into an 8-bit H.264 clip the browser can play,
    with any display rotation baked in. The before/after pair is both made from this clip,
    so they line up frame for frame whatever the source's keyframes are. Cached."""
    import subprocess
    st = os.stat(src)
    key = f"{os.path.abspath(src).lower()}|{st.st_size}|{int(st.st_mtime)}|{start_s:.3f}|{duration_s:.3f}"
    out = PREVIEW_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}_before.mp4"
    if out.exists():
        return str(out)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-ss", f"{start_s:.3f}", "-i", src,
                    "-t", f"{duration_s:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "constqp", "-qp", "14", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)],
                   check=True, capture_output=True, timeout=600)
    os.replace(tmp, out)
    return str(out)


def prune_previews():
    """Keep only the newest preview files."""
    files = sorted(PREVIEW_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[PREVIEW_KEEP:]:
        try:
            old.unlink()
        except OSError:
            pass   # still being streamed to the dialog


def analyse(src: str, settings: Settings = Settings(),
            progress: Optional[Callable[[float], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None, lens_e: Optional[dict] = None):
    """Pass 1 + plan only: (VideoInfo, plan). Cheap to repeat once the landmarks are cached.
    `lens_e`: each eye's lens position from the whole recording, for a clip of it."""
    from .gpu import probe
    info = probe(src)
    data = _track(src, info, progress, cancelled)
    return info, make_plan(data["pts"], data["score"], info.fps, settings, lens_e)


def lens_positions(src: str, settings: Settings = Settings()) -> Optional[dict]:
    """Where each eye sits when looking into the lens, over the whole of `src`, or None.

    Only from landmarks already cached (a full run or an earlier preview tracked them): a
    preview must stay quick, so it never tracks a whole recording just for this."""
    try:
        if not _cache_path(src).exists():
            return None
        _info, plan = analyse(src, settings)
        return plan.get("lens_e") or None
    except Exception:
        return None


def correct_eye_contact(src: str, dst: Optional[str] = None, settings: Settings = Settings(),
                        quality: str = "high", codec: str = "hevc",
                        progress: Optional[Callable[[float], None]] = None,
                        cancelled: Optional[Callable[[], bool]] = None,
                        lens_e: Optional[dict] = None) -> dict:
    """Write an eye-contact-corrected copy of `src`. Returns a small report. Pass `lens_e`
    (lens_positions of the whole recording) when `src` is a short clip of it."""
    ok, why = available()
    if not ok:
        raise RuntimeError(f"Eye contact correction needs an NVIDIA GPU setup: {why}")
    dst = dst or default_output(src)
    if os.path.abspath(dst).lower() == os.path.abspath(src).lower():
        raise ValueError("Output would overwrite the source recording")
    t0 = time.time()
    report = lambda f: progress(f) if progress else None   # noqa: E731

    info, plan = analyse(src, settings, lambda f: report(f * TRACK_SHARE), cancelled, lens_e)
    t_track = time.time() - t0
    from .render import render
    frames = render(src, dst, plan, info, quality, codec,
                    lambda f: report(TRACK_SHARE + f * (1 - TRACK_SHARE)), cancelled)
    report(1.0)

    moved = np.abs(plan["corr"]) > 1e-3
    w = float(np.median(plan["w_e"]["R"][plan["valid"]])) if plan["valid"].any() else 0.0
    out = {
        "output_path": dst,
        "frames": int(frames),
        "face_found_pct": round(100 * float(plan["valid"].mean()), 1) if len(plan["valid"]) else 0.0,
        "corrected_pct": round(100 * float(moved.mean()), 1) if len(moved) else 0.0,
        # Frames already looking into the lens, left alone (not pushed past it).
        "camera_look_pct": round(plan.get("camera_pct", 0.0), 1),
        "lens_separation_deg": round(plan.get("lens_sep_deg", 0.0), 1),
        "prompter_offset_deg": round(plan["offset_deg"], 1),
        "symmetry_offset_deg": round(plan["symmetry_offset_deg"], 1),
        "reading_sweep_deg": round(plan["sweep_deg"], 1),
        "median_shift_px": round(float(np.median(np.abs(plan["corr"][moved]))) * w, 2) if moved.any() else 0.0,
        # What each eye actually moved (per_eye_limits); median_shift_px is the shared push before it.
        "median_shift_px_per_eye": {k: round(float(np.median(np.abs(plan["corr_e"][k][moved]))) * w, 2)
                                    for k in plan.get("corr_e", {})} if moved.any() else {},
        "lens_per_eye": {k: round(v, 3) for k, v in (plan.get("lens_e") or {}).items()},
        "settings": asdict(settings),
        "quality": quality,
        "seconds": round(time.time() - t0, 1),
        "track_seconds": round(t_track, 1),
    }
    logger.info("Eye contact: %s", out)
    return out
