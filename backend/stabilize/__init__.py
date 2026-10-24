"""Video stabilization for handheld clips (libvidstab through ffmpeg).

Two ffmpeg passes over the recording:
  1. detect    -- `vidstabdetect` measures the frame-to-frame motion (CPU) into an ASCII
                  .trf file, cached per source file + shakiness + tripod;
  2. transform -- `vidstabtransform` smooths the camera path and moves each frame
                  against the shake, encoded on NVENC; audio copied untouched, the
                  source's timestamps passed through, colour tags and rotation kept.

    from stabilize import stabilize_video, Settings
    report = stabilize_video("take1.mp4", settings=Settings(strength=0.6))

The .trf file (ASCII, `fileformat=ascii`) as written by ffmpeg 8.1's libvidstab:

    VID.STAB 1
    #      accuracy = 15
    #     shakiness = 6
    #      stepsize = 6
    #   mincontrast = 0.250000
    Frame 1 (List 0 [])
    Frame 2 (List 161 [(LM 10 2 1456 216 112 0.476845 0.662070),(LM 11 2 1084 702 ...),...])

One line per frame. Each `LM` is one local motion field: `v.x v.y f.x f.y f.size contrast
match` -- the field's displacement from the previous frame (v, whole pixels of the frame
as decoded), its centre (f), size, contrast and match quality. It does not store a global
per-frame transform (vidstabtransform derives that itself), so the per-frame (dx, dy)
used here is the median of the frame's field vectors -- robust to fields on moving
subjects. Frame 1 and featureless frames have an empty list.
"""
import hashlib
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import DATA_DIR, FFMPEG_BIN
from utils.ffmpeg_utils import parse_frame_rate, run_ffprobe
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "stabilize"
PREVIEW_DIR = CACHE_DIR / "preview"
PREVIEW_KEEP = 24                  # newest preview files kept
CACHE_MAX_BYTES = 4 * 1024 ** 3    # .trf files are ~6 KB a frame (ASCII); oldest dropped past this
OUTPUT_SUFFIX = "_stable"
DETECT_SHARE = 0.4                 # progress split between the two passes
SHAKY_PX = 1.5                     # mean jitter (1080p-equivalent px) above which a clip reads as shaky
SHARPEN_ABOVE_PCT = 3.0            # a zoom past this softens the picture enough to sharpen back
UNSHARP = "unsharp=5:5:0.4:3:3:0.2"
CROPS = ("auto", "keep", "black")
# Constant-QP NVENC, the same table as eye contact (eyecontact/render.py QUALITY_QP).
QUALITY_QP = {"standard": 22, "high": 18, "max": 14}

__all__ = ["Settings", "available", "stabilize_video", "analyse", "default_output", "preview_clip",
           "prune_previews", "detect_filter", "transform_filter", "parse_trf", "frame_motion", "shake_stats"]


@dataclass(frozen=True)
class Settings:
    strength: float = 0.6          # 0..1 -> vidstabdetect shakiness 1..10
    crop: str = "auto"             # auto: optimal static zoom hides the borders; keep: borders
                                   # filled from earlier frames, no zoom; black: black borders, no zoom
    smoothing_s: float = 1.0       # half-window of the camera-path smoothing, in seconds (0.5 left 3.8 px of a synthetic wobble; 1.0 left 0.6 px)
    zoom_pct: Optional[float] = None   # a fixed zoom instead of the automatic one (crop mode still
                                       # picks the border fill: black for "black", else keep)
    tripod: bool = False           # locked-off camera: every frame is held to the first one
    quality: str = "high"          # standard | high | max (NVENC constant QP 22/18/14)

    def __post_init__(self):
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if self.crop not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        if not 0.0 <= float(self.smoothing_s) <= 10.0:
            raise ValueError("smoothing_s must be between 0 and 10 seconds")
        if self.zoom_pct is not None and not -50.0 <= float(self.zoom_pct) <= 50.0:
            raise ValueError("zoom_pct must be between -50 and 50")
        if self.quality not in QUALITY_QP:
            raise ValueError(f"quality must be one of {tuple(QUALITY_QP)}")


@lru_cache(maxsize=1)
def available() -> Tuple[bool, str]:
    """Whether this machine's ffmpeg can stabilize, and why not."""
    try:
        filters = subprocess.run([FFMPEG_BIN, "-hide_banner", "-filters"], capture_output=True, text=True,
                                 timeout=30, creationflags=NO_WINDOW).stdout
        encoders = subprocess.run([FFMPEG_BIN, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                  timeout=30, creationflags=NO_WINDOW).stdout
    except Exception as e:
        return False, f"ffmpeg unavailable ({e})"
    for name in ("vidstabdetect", "vidstabtransform"):
        if not re.search(rf"\b{name}\b", filters):
            return False, f"ffmpeg was built without libvidstab ({name} missing)"
    for name in ("hevc_nvenc", "h264_nvenc"):
        if not re.search(rf"\b{name}\b", encoders):
            return False, f"ffmpeg has no {name} encoder"
    return True, ""


def default_output(src: str) -> str:
    p = Path(src)
    return str(p.with_name(f"{p.stem}{OUTPUT_SUFFIX}.mp4"))


def shakiness(strength: float) -> int:
    return int(min(10, max(1, round(1 + 9 * float(strength)))))


def _cache_path(src: str, settings: Settings) -> Path:
    st = os.stat(src)
    key = (f"{os.path.abspath(src).lower()}|{st.st_size}|{int(st.st_mtime)}"
           f"|s{shakiness(settings.strength)}|t{int(bool(settings.tripod))}|v1")
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.trf"


# ffmpeg runs with its cwd in the cache folder, so the filters name the .trf by its bare
# (hex) file name: no Windows drive colons or backslashes to escape inside a filtergraph.
def detect_filter(settings: Settings, trf_name: str) -> str:
    parts = [f"shakiness={shakiness(settings.strength)}", "accuracy=15", "fileformat=ascii",
             f"result={trf_name}"]
    if settings.tripod:
        parts.append("tripod=1")      # reference frame 1; vidstabtransform's tripod needs it
    return "vidstabdetect=" + ":".join(parts)


def transform_filter(settings: Settings, trf_name: str, fps: float, zoom_pct: float = 0.0) -> str:
    """`zoom_pct`: the zoom this run applies (the probed optimal zoom for crop=auto, the
    fixed zoom_pct when set, else 0); past SHARPEN_ABOVE_PCT the picture is sharpened back."""
    parts = [f"input={trf_name}",
             f"smoothing={int(min(1000, max(0, round(float(settings.smoothing_s) * float(fps)))))}"]
    if settings.zoom_pct is not None:
        parts += ["optzoom=0", f"zoom={float(settings.zoom_pct):g}",
                  "crop=black" if settings.crop == "black" else "crop=keep"]
    elif settings.crop == "auto":
        parts += ["optzoom=1", "crop=keep"]
    elif settings.crop == "keep":
        parts += ["optzoom=0", "crop=keep"]
    else:
        parts += ["optzoom=0", "crop=black"]
    if settings.tripod:
        parts.append("tripod=1")
    parts.append("interpol=bicubic")
    vf = "vidstabtransform=" + ":".join(parts)
    if zoom_pct > SHARPEN_ABOVE_PCT:
        vf += "," + UNSHARP
    return vf


_LM = re.compile(r"\(LM (-?\d+) (-?\d+) ")


def parse_trf(path) -> np.ndarray:
    """Per-frame (dx, dy) from an ASCII .trf (see the module docstring): the median of the
    frame's local-motion vectors; NaN where the frame has none. Shape (frames, 2)."""
    rows: List[Tuple[float, float]] = []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.startswith("Frame "):
                continue
            v = _LM.findall(line)
            if v:
                a = np.asarray(v, dtype=np.float64)
                rows.append((float(np.median(a[:, 0])), float(np.median(a[:, 1]))))
            else:
                rows.append((np.nan, np.nan))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def frame_motion(trf, settings: Settings) -> np.ndarray:
    """Per-frame (dx, dy) motion from a .trf. A tripod-mode detect measures every frame
    against frame 1 (an offset, not a step), so its steps are the differences."""
    motion = parse_trf(trf)
    if settings.tripod and len(motion):
        motion = np.diff(motion, axis=0, prepend=motion[:1])
    return motion


def shake_stats(motion: np.ndarray, fps: float, short_side: int) -> Tuple[float, float]:
    """(mean, max) jitter in 1080p-equivalent pixels per frame: each frame's motion minus its
    centred 1 s moving average, so a deliberate pan or tilt does not count as shake."""
    if motion.size == 0:
        return 0.0, 0.0
    valid = ~np.isnan(motion).any(axis=1)
    if not valid.any():
        return 0.0, 0.0
    w = max(1, int(round(float(fps))))
    kernel = np.ones(w)
    filled = np.where(valid[:, None], motion, 0.0)
    counts = np.convolve(valid.astype(np.float64), kernel, mode="same")
    avg = np.stack([np.convolve(filled[:, k], kernel, mode="same") for k in range(2)], axis=1)
    avg /= np.maximum(counts, 1.0)[:, None]
    jitter = np.hypot(*(motion - avg)[valid].T) * (1080.0 / max(1, int(short_side)))
    return round(float(jitter.mean()), 2), round(float(jitter.max()), 2)


def probe(src: str) -> Dict[str, float]:
    """fps, duration, coded width/height and display rotation of the first video stream."""
    data = run_ffprobe(src)
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not vs:
        raise ValueError(f"No video stream in {src}")
    num, den = parse_frame_rate(vs.get("avg_frame_rate") or "", 0, 1)
    if not num:
        num, den = parse_frame_rate(vs.get("r_frame_rate") or "30/1")
    rotation = 0
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    duration = float((data.get("format") or {}).get("duration") or vs.get("duration") or 0.0)
    return {"fps": num / den, "duration": duration, "width": int(vs.get("width") or 0),
            "height": int(vs.get("height") or 0), "rotation": rotation}


def _kill_tree(proc: subprocess.Popen):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           timeout=30, creationflags=NO_WINDOW)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _ffmpeg(args: List[str], duration: float, progress: Optional[Callable[[float], None]],
            cancelled: Optional[Callable[[], bool]], cwd: Optional[Path] = None) -> str:
    """Run ffmpeg with `-progress pipe:1`; report 0..1 from its out_time, kill the process
    tree on cancel (InterruptedError). stderr is drained on its own thread (a chatty filter
    would otherwise fill the pipe and stall ffmpeg). Returns stderr."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-nostats", "-y", "-progress", "pipe:1"] + args
    logger.info("Stabilize ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None,
                            creationflags=NO_WINDOW)
    err: deque = deque(maxlen=400)
    stop = threading.Event()
    state = {"cancelled": False}

    def drain():
        for raw in proc.stderr:
            err.append(raw.decode("utf-8", "replace").rstrip())

    def watch():
        while not stop.wait(0.25):
            if cancelled and cancelled():
                state["cancelled"] = True
                _kill_tree(proc)
                return

    threads = [threading.Thread(target=drain, daemon=True), threading.Thread(target=watch, daemon=True)]
    for t in threads:
        t.start()
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if progress and duration > 0 and (line.startswith("out_time_us=") or line.startswith("out_time_ms=")):
                try:
                    progress(min(1.0, max(0.0, int(line.split("=", 1)[1]) / 1e6 / duration)))
                except ValueError:
                    pass
        proc.wait()
    finally:
        stop.set()
        if proc.poll() is None:
            _kill_tree(proc)
            proc.wait()
        threads[0].join(timeout=30)
    if state["cancelled"]:
        raise InterruptedError("cancelled")
    text = "\n".join(err)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {text[-800:]}")
    return text


def _prune_cache():
    """Drop the oldest .trf files once the cache passes CACHE_MAX_BYTES."""
    files = sorted(CACHE_DIR.glob("*.trf"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for p in files:
        total += p.stat().st_size
        if total > CACHE_MAX_BYTES:
            try:
                p.unlink()
            except OSError:
                pass


def detect(src: str, settings: Settings, info: Dict[str, float],
           progress: Optional[Callable[[float], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None) -> Path:
    """Pass 1: the .trf for `src` (cached)."""
    trf = _cache_path(src, settings)
    if trf.exists():
        os.utime(trf)       # recently used: pruned last
        if progress:
            progress(1.0)
        return trf
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = trf.with_name(trf.stem + ".tmp.trf")
    try:
        _ffmpeg(["-noautorotate", "-hwaccel", "cuda", "-i", src, "-map", "0:v:0",
                 "-vf", detect_filter(settings, tmp.name), "-f", "null", "-"],
                info["duration"], progress, cancelled, cwd=CACHE_DIR)
        os.replace(tmp, trf)
    finally:
        if tmp.exists():
            tmp.unlink()
    _prune_cache()
    return trf


def analyse(src: str, settings: Settings = Settings(),
            progress: Optional[Callable[[float], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Pass 1 only: how shaky `src` is, so the UI can suggest stabilizing. The detect
    result is cached, so a full run afterwards skips straight to the transform."""
    t0 = time.time()
    info = probe(src)
    trf = detect(src, settings, info, progress, cancelled)
    motion = frame_motion(trf, settings)
    mean, peak = shake_stats(motion, info["fps"], min(info["width"], info["height"]))
    return {"mean_shift_px": mean, "max_shift_px": peak, "shaky": bool(mean > SHAKY_PX),
            "frames": int(len(motion)), "seconds": round(time.time() - t0, 1)}


_FINAL_ZOOM = re.compile(r"Final zoom:\s*(-?[\d.]+)")


def _probe_zoom(src: str, trf: Path, settings: Settings, fps: float) -> float:
    """The optimal zoom vidstabtransform will pick (it logs it while reading the .trf, before
    the first frame), from a one-frame dry run, so the sharpening choice can follow it."""
    text = _ffmpeg(["-noautorotate", "-i", src, "-map", "0:v:0",
                    "-vf", transform_filter(settings, trf.name, fps), "-frames:v", "1", "-f", "null", "-"],
                   0.0, None, None, cwd=trf.parent)
    m = _FINAL_ZOOM.findall(text)
    return float(m[-1]) if m else 0.0


def _encoder_args(codec: str, quality: str, fps: float) -> List[str]:
    qp = str(QUALITY_QP.get(quality, QUALITY_QP["high"]))
    gop = str(max(1, int(round(fps * 2))))
    if codec == "h264":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "constqp", "-qp", qp,
                "-bf", "0", "-g", gop, "-pix_fmt", "yuv420p"]
    return ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "constqp", "-qp", qp,
            "-bf", "0", "-g", gop, "-tag:v", "hvc1"]


def stabilize_video(src: str, dst: Optional[str] = None, settings: Settings = Settings(),
                    progress: Optional[Callable[[float], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None, codec: str = "hevc") -> dict:
    """Write a stabilized copy of `src` (never over it). Returns a small report; the shift
    numbers describe the shake measured in `src`."""
    ok, why = available()
    if not ok:
        raise RuntimeError(f"Stabilization unavailable: {why}")
    dst = dst or default_output(src)
    if os.path.normcase(os.path.abspath(dst)) == os.path.normcase(os.path.abspath(src)):
        raise ValueError("Output would overwrite the source recording")
    t0 = time.time()
    report = (lambda f: progress(f)) if progress else (lambda f: None)
    info = probe(src)
    fps = info["fps"]

    trf = detect(src, settings, info, lambda f: report(f * DETECT_SHARE), cancelled)
    if cancelled and cancelled():
        raise InterruptedError("cancelled")
    if settings.zoom_pct is not None:
        zoom = float(settings.zoom_pct)
    elif settings.crop == "auto":
        zoom = _probe_zoom(src, trf, settings, fps)
    else:
        zoom = 0.0
    vf = transform_filter(settings, trf.name, fps, zoom)

    tmp = dst + ".tmp.mp4"
    rot = dst + ".rot.tmp.mp4"
    try:
        _ffmpeg(["-noautorotate", "-hwaccel", "cuda", "-i", src, "-map", "0:v:0", "-map", "0:a?",
                 "-vf", vf, "-fps_mode", "passthrough", *_encoder_args(codec, settings.quality, fps),
                 "-c:a", "copy", "-map_metadata", "0", "-movflags", "+faststart", tmp],
                info["duration"], lambda f: report(DETECT_SHARE + f * (1 - DETECT_SHARE)), cancelled,
                cwd=trf.parent)
        if info["rotation"]:
            # filtering with -noautorotate drops the display matrix; put it back without re-encoding
            _ffmpeg(["-noautorotate", "-display_rotation:v:0", str(info["rotation"]), "-i", tmp,
                     "-map", "0", "-c", "copy", "-movflags", "+faststart", rot], 0.0, None, cancelled)
            os.replace(rot, tmp)
        os.replace(tmp, dst)
    finally:
        for leftover in (tmp, rot):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
    report(1.0)

    motion = frame_motion(trf, settings)
    mean, peak = shake_stats(motion, fps, min(info["width"], info["height"]))
    out = {"output_path": dst, "frames": int(len(motion)), "seconds": round(time.time() - t0, 1),
           "zoom_pct": round(zoom, 2), "mean_shift_px": mean, "max_shift_px": peak,
           "settings": asdict(settings)}
    logger.info("Stabilize: %s", out)
    return out


def preview_clip(src: str, start_s: float, duration_s: float) -> str:
    """Cut `duration_s` seconds of `src` into an 8-bit H.264 clip the browser can play
    (rotation baked in), the "before" of a preview; the "after" is stabilized from this same
    clip so the pair lines up frame for frame. Cached (same scheme as eye contact's)."""
    st = os.stat(src)
    key = f"{os.path.abspath(src).lower()}|{st.st_size}|{int(st.st_mtime)}|{start_s:.3f}|{duration_s:.3f}"
    out = PREVIEW_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}_before.mp4"
    if out.exists():
        return str(out)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    subprocess.run([FFMPEG_BIN, "-v", "error", "-y", "-hwaccel", "cuda", "-ss", f"{start_s:.3f}", "-i", src,
                    "-t", f"{duration_s:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "constqp", "-qp", "14", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)],
                   check=True, capture_output=True, timeout=600, creationflags=NO_WINDOW)
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
