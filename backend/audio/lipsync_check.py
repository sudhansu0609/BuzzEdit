"""Lip-sync check for voice-converted spans (docs/LIPSYNC_CHECK_PLAN.md).

A converted span should still move with the speaker's mouth. Per span:

* `mouth_openness` -- the eye-contact module's GPU landmark net (`eyecontact.tracker.track_window`)
  over only the span's frames (+0.5 s context to lock on): |p13 - p14| / |p78 - p308| per frame,
  NaN where no face. Decode path: PyNvVideoCodec `SimpleDecoder.seek_to_index` (NVDEC), not the
  whole-file ThreadedDecoder pass, and nothing touches the eye-contact cache.
* `audio_envelope` -- mono PCM, 300-3000 Hz band-pass, rectified, 20 Hz low-pass, one mean value
  per video frame, log-compressed, scaled to 0..1.
* `sync_score` -- best Pearson correlation of the smoothed, mean-removed series over +-max_lag
  frames (12); lag > 0 = the audio comes later than the mouth.
* `check_spans` -- original vs converted audio against the same mouth: flagged when the converted
  correlation is weak (< max(0.15, 0.6 * original)) or its lag differs from the original's by more
  than 3 frames. Lag is judged relative to the original because real footage shows the sound
  trailing the mouth by a steady 3-4 frames (mouth opening leads speech, plus any camera offset).
  A face in under 60 % of frames is noted ("no_face"), never flagged; spans under 1 s are
  skipped (corr NaN, reason "short").

`check_project` / `start_check_job` re-check a project's applied voice map (route
`POST /api/audio/{project}/voice/lipsync_check`).
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audio.analysis import read_mono

logger = logging.getLogger("audio.lipsync_check")

Progress = Optional[Callable[[float, str], None]]

MOUTH_POINTS = (13, 14, 78, 308)       # inner upper lip, inner lower lip, mouth corners
CONTEXT_S = 0.5
BAND_HZ = (300.0, 3000.0)
LOWPASS_HZ = 20.0
LOG_FLOOR = 1e-3        # log(env + floor * max): keeps the pauses from dominating
MAX_LAG = 12
LAG_LIMIT = 3
MIN_SPAN_S = 1.0
MIN_FACE = 0.6
MIN_CORR = 0.15
REL_CORR = 0.6

_tracker_lock = threading.Lock()
_trackers: Dict[str, Any] = {}


# ------------------------------------------------------------------ series

def _frames(start_s: float, end_s: float, fps: float) -> int:
    return max(0, int(round((float(end_s) - float(start_s)) * float(fps))))


def video_fps(src_video: str) -> float:
    """The video stream's average frame rate (0.0 when there is no video stream)."""
    try:
        import av
        with av.open(str(src_video)) as container:
            if not container.streams.video:
                return 0.0
            rate = container.streams.video[0].average_rate
            return float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _tracker(colorspace: str) -> Any:
    from eyecontact.tracker import Tracker
    with _tracker_lock:
        if colorspace not in _trackers:
            _trackers[colorspace] = Tracker(colorspace)
        return _trackers[colorspace]


def release_tracker() -> None:
    """Drop the landmark net so its VRAM goes back (check_spans calls this when it is done)."""
    with _tracker_lock:
        had = bool(_trackers)
        _trackers.clear()
    if had:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def mouth_openness(src_video: str, start_s: float, end_s: float, fps: float) -> np.ndarray:
    """Per-frame mouth-open ratio |p13 - p14| / |p78 - p308| over [start_s, end_s) (one value
    per frame at `fps`; NaN where no face). Tracked on the GPU over the span + 0.5 s context."""
    from eyecontact.gpu import probe
    from eyecontact.tracker import track_window
    info = probe(str(src_video))
    fps = float(fps or info.fps)
    n = _frames(start_s, end_s, fps)
    out = np.full(n, np.nan, np.float64)
    if n == 0:
        return out
    got = track_window(str(src_video), info, max(0.0, start_s - CONTEXT_S), end_s + CONTEXT_S,
                       MOUTH_POINTS, _tracker(info.colorspace))
    tb = float(info.time_base)
    for pts, ts in zip(got["pts"], got["ts"]):
        k = int(round((int(ts) * tb - start_s) * fps))
        if not 0 <= k < n or not np.all(np.isfinite(pts[:, :2])):
            continue
        width = float(np.linalg.norm(pts[2, :2] - pts[3, :2]))
        if width > 1e-6:
            out[k] = float(np.linalg.norm(pts[0, :2] - pts[1, :2])) / width
    return out


def audio_envelope(audio_path: str, start_s: float, end_s: float, fps: float) -> np.ndarray:
    """Speech envelope, one value per video frame: 300-3000 Hz band-pass, rectified, 20 Hz
    low-pass, frame means, log-compressed, scaled to 0..1."""
    n = _frames(start_s, end_s, fps)
    if n == 0:
        return np.zeros(0)
    samples, rate = read_mono(str(audio_path), start_s, end_s)
    x = np.asarray(samples, np.float64)
    if len(x) > 64:
        try:
            if rate > 2 * BAND_HZ[1] * 1.05:
                x = sosfiltfilt(butter(4, BAND_HZ, btype="band", fs=rate, output="sos"), x)
            x = np.abs(sosfiltfilt(butter(4, min(LOWPASS_HZ, rate / 2 * 0.9), btype="low", fs=rate,
                                          output="sos"), np.abs(x)))
        except ValueError:            # too short for the filters' padding
            x = np.abs(x)
    else:
        x = np.abs(x)
    edges = np.round(np.arange(n + 1) * rate / float(fps)).astype(np.int64)
    env = np.zeros(n)
    for k in range(n):
        a, b = edges[k], min(edges[k + 1], len(x))
        if b > a:
            env[k] = x[a:b].mean()
    peak = env.max()
    if peak <= 0:
        return env
    env = np.log(env / peak + LOG_FLOOR)
    low, high = env.min(), env.max()
    return (env - low) / (high - low) if high > low else np.zeros(n)


def _smooth(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    padded = np.concatenate([x[:1], x, x[-1:]])
    return np.convolve(padded, np.ones(3) / 3.0, mode="valid")


def sync_score(mouth: Sequence[float], env: Sequence[float], max_lag_frames: int = MAX_LAG) -> Dict[str, Any]:
    """Best Pearson correlation of the smoothed, mean-removed series over +-max_lag frames.
    `lag_frames` > 0: the audio comes that many frames after the mouth. NaN mouth frames are
    bridged for the smoothing and left out of the correlation."""
    m = np.asarray(mouth, np.float64)
    e = np.asarray(env, np.float64)
    n = min(len(m), len(e))
    m, e = m[:n], e[:n]
    valid = np.isfinite(m) & np.isfinite(e)
    if valid.sum() < 8:
        return {"corr": float("nan"), "lag_frames": 0}
    idx = np.arange(n)
    m = np.interp(idx, idx[valid], m[valid])
    e = np.interp(idx, idx[valid], e[valid])
    m, e = _smooth(m), _smooth(e)
    m, e = m - m.mean(), e - e.mean()
    best_corr, best_lag = float("nan"), 0
    for lag in range(-int(max_lag_frames), int(max_lag_frames) + 1):
        i0, i1 = max(0, -lag), min(n, n - lag)          # pairs m[i] with e[i + lag]
        if i1 - i0 < 8:
            continue
        keep = valid[i0:i1] & valid[i0 + lag:i1 + lag]
        a, b = m[i0:i1][keep], e[i0 + lag:i1 + lag][keep]
        if len(a) < 8 or a.std() < 1e-12 or b.std() < 1e-12:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if math.isfinite(c) and (not math.isfinite(best_corr) or c > best_corr):
            best_corr, best_lag = c, lag
    return {"corr": best_corr, "lag_frames": int(best_lag)}


# ------------------------------------------------------------------ spans

def check_spans(src_video: str, original_audio: str, converted_audio: str,
                spans: Sequence[Sequence[float]], fps: Optional[float] = None,
                progress: Progress = None) -> List[Dict[str, Any]]:
    """Per span: {start_s, end_s, corr_original, corr_converted, lag_frames, flag, reason};
    lag_frames = the converted audio's lag minus the original's (> 0: later than it was).
    reason: "weak_match" / "lag" (flagged), "no_face" (face in < 60 % of frames, not flagged),
    "short" (< 1 s, skipped, corr NaN), else None. Spans are seconds of the source."""
    fps = float(fps or video_fps(src_video) or 25.0)
    results: List[Dict[str, Any]] = []
    try:
        for k, span in enumerate(spans):
            start_s, end_s = float(span[0]), float(span[1])
            row: Dict[str, Any] = {"start_s": round(start_s, 3), "end_s": round(end_s, 3),
                                   "corr_original": float("nan"), "corr_converted": float("nan"),
                                   "lag_frames": None, "flag": False, "reason": None}
            results.append(row)
            if progress is not None:
                progress(k / max(len(spans), 1), f"lip sync {k + 1}/{len(spans)}")
            if end_s - start_s < MIN_SPAN_S:
                row["reason"] = "short"
                continue
            mouth = mouth_openness(src_video, start_s, end_s, fps)
            face = float(np.isfinite(mouth).mean()) if len(mouth) else 0.0
            if face < MIN_FACE:
                row["reason"] = "no_face"
                continue
            original = sync_score(mouth, audio_envelope(original_audio, start_s, end_s, fps))
            converted = sync_score(mouth, audio_envelope(converted_audio, start_s, end_s, fps))
            c_orig, c_conv = original["corr"], converted["corr"]
            # relative to how the original sits against the same mouth
            lag = converted["lag_frames"] - (original["lag_frames"] if math.isfinite(c_orig) else 0)
            row.update(corr_original=round(c_orig, 4) if math.isfinite(c_orig) else c_orig,
                       corr_converted=round(c_conv, 4) if math.isfinite(c_conv) else c_conv,
                       lag_frames=lag)
            floor = max(MIN_CORR, REL_CORR * c_orig) if math.isfinite(c_orig) else MIN_CORR
            if not math.isfinite(c_conv) or c_conv < floor:
                row.update(flag=True, reason="weak_match")
            elif abs(lag) > LAG_LIMIT:
                row.update(flag=True, reason="lag")
    finally:
        release_tracker()
    return results


def summary(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One `lipsync` record for a voice-conversion entry: the worst span's numbers (a flagged one
    first, then the lowest converted correlation) plus every span's row under `spans`."""
    if not rows:
        return None

    def badness(row: Dict[str, Any]) -> Tuple[bool, float]:
        c = row.get("corr_converted")
        return bool(row.get("flag")), -(c if isinstance(c, float) and math.isfinite(c) else 2.0)

    worst = max(rows, key=badness)
    keys = ("corr_original", "corr_converted", "lag_frames", "flag", "reason")
    return {**{key: worst.get(key) for key in keys}, "spans": rows}


# ------------------------------------------------------------------ project

def _audios(source: Any, record: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """(original, converted) audio paths: the recorded voice-map version and what lies beneath
    it, else the source's current override and the version before it (or the file itself)."""
    history = list(source.audio_history or [])
    converted = None
    if record and any(str(e.get("path")) == str(record.get("version_path")) for e in history):
        converted = str(record["version_path"])
    elif source.audio_override:
        converted = str(source.audio_override)
    if converted is None:
        return None, None
    position = next((i for i, e in enumerate(history) if str(e.get("path")) == converted), len(history))
    below = history[:position]
    return (str(below[-1]["path"]) if below else str(source.path)), converted


def project_spans(p_data: Dict[str, Any]) -> List[List[float]]:
    conversions = (p_data.get("metadata") or {}).get("voice_conversions") or []
    return sorted([float(a), float(b)] for c in conversions for a, b in (c.get("spans") or []))


def check_project(project_id: str, spans: Optional[Sequence[Sequence[float]]] = None,
                  progress: Progress = None) -> Dict[str, Any]:
    """Re-check a project's converted audio against the primary recording's mouth, without
    converting again. Default spans: the applied voice map's (project metadata)."""
    from audio import noise
    from audio.noise import Conflict
    from audio.voice_conversion import APPLIED_KEY
    _pid, p_data, tl = noise.load_project(project_id)
    _sid, source = noise.resolve_source(tl, p_data, None)
    record = (p_data.get("settings") or {}).get(APPLIED_KEY)
    original, converted = _audios(source, record if isinstance(record, dict) else None)
    if converted is None:
        raise Conflict("The source has no converted audio version to check")
    spans = [list(s) for s in spans] if spans else project_spans(p_data)
    if not spans:
        raise Conflict("No spans to check: apply a voice map or pass spans")
    for span in spans:
        if len(span) != 2 or not float(span[1]) > float(span[0]) >= 0.0:
            raise ValueError(f"Bad span {span!r}: [start_s, end_s] seconds of the source")
    if not video_fps(source.path):
        raise Conflict("The primary source has no video: there is no mouth to check against")
    rows = check_spans(source.path, original, converted, spans, None, progress)
    return {"spans": rows, "lipsync_flags": sum(1 for r in rows if r["flag"]),
            "original_audio": original, "converted_audio": converted}


def start_check_job(project_id: str, spans: Optional[Sequence[Sequence[float]]] = None) -> str:
    """`check_project` as a job of the process lane: poll `GET /api/audio/{project}/process/{job_id}`."""
    from audio import processes
    job_id = f"audio_process_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat(timespec="seconds")
    with processes._jobs_guard:
        processes.JOBS[job_id] = {"id": job_id, "job_type": "audio_process", "project_id": project_id,
                                  "source_id": None, "effect": "voice_conversion", "op": "lipsync_check",
                                  "status": "pending", "progress": 0.0, "message": "queued",
                                  "result": None, "error": None, "created": now}

    def run() -> None:
        def progress(fraction: float, message: str) -> None:
            processes._update_job(job_id, progress=round(max(0.0, min(1.0, fraction)), 4), message=message)

        processes._update_job(job_id, status="running", message="starting")
        try:
            result = check_project(project_id, spans, progress)
        except Exception as exc:
            logger.warning("Lip-sync check %s failed: %s", job_id, exc, exc_info=True)
            processes._update_job(job_id, status="failed", error=str(exc) or exc.__class__.__name__,
                                  message="failed", finished=datetime.now().isoformat(timespec="seconds"))
            return
        processes._update_job(job_id, status="completed", progress=1.0, message="done", result=result,
                              finished=datetime.now().isoformat(timespec="seconds"))

    threading.Thread(target=run, name=job_id, daemon=True).start()
    return job_id
