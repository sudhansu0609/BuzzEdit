"""Audio-based sync between two recordings of the same event (MULTICAM_SYNC_PLAN.md §1).

Convention throughout: `other_time = ref_time + offset_s`. A second camera that
started rolling 2.3 s after the reference hears a given clap 2.3 s earlier on
its own clock, so its offset is -2.3; an external recorder started 1.75 s before
the camera has +1.75.

Coarse to fine, numpy/scipy only:
  1. onset envelope of each file (|x| -> 20 Hz low-pass -> 100 Hz, log-compressed,
     positive slope), FFT cross-correlated over +-max_offset_s. Onsets survive a
     different microphone, EQ and gain, where raw PCM correlation does not.
  2. the lag refined on 8 kHz PCM within +-60 ms of the coarse one, on several
     windows of the overlap, with parabolic sub-sample interpolation.
  3. drift: the refined lag on a window near the start and one near the end of
     the overlap; drift_ppm = d(lag)/dt * 1e6. Two free-running clocks drift
     tens of ppm apart; over a 40-minute take 50 ppm is 120 ms, a visible lip
     sync error, so `write_aligned_wav` time-stretches past 20 ppm.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal

from config import FFMPEG_BIN
from utils.proc import NO_WINDOW

logger = logging.getLogger("audio_sync")

PCM_RATE = 8000           # decode rate for both stages
ENV_RATE = 100            # onset envelope rate (10 ms lag resolution)
ENV_LOWPASS_HZ = 20.0
REFINE_RADIUS_S = 0.060   # refine search either side of the coarse lag
LOCAL_RADIUS_S = 1.0    # per-window envelope search either side of the whole-file lag
WINDOW_S = 10.0          # refine window length (shorter on short overlaps)
MAX_WINDOWS = 24
PEAK_EXCLUSION_S = 0.5    # a "distinct" second peak is at least this far from the main one
MIN_CONFIDENCE = 1.5      # below this the match is reported as unreliable
DRIFT_CORRECT_PPM = 20.0  # past this the aligned WAV is time-stretched
METHOD = "onset-xcorr+pcm-refine"


@dataclass
class SyncResult:
    offset_s: float          # other_time = ref_time + offset_s (at ref_time 0)
    confidence: float        # main peak / second distinct peak; < 1.5 is unreliable
    drift_ppm: float         # d(offset)/d(ref_time) * 1e6
    method: str = METHOD

    @property
    def reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def to_dict(self) -> dict:
        out = asdict(self)
        out["reliable"] = self.reliable
        return out


class SyncError(RuntimeError):
    """A file could not be decoded, or the two have no usable overlap."""


def decode_mono(path: str, sr: int = PCM_RATE, start_s: Optional[float] = None,
                dur_s: Optional[float] = None) -> np.ndarray:
    """Mono float32 PCM of `path` at `sr` Hz via ffmpeg (optionally a window of it)."""
    cmd = [FFMPEG_BIN, "-v", "error", "-nostdin"]
    if start_s is not None and start_s > 0:
        cmd += ["-ss", f"{float(start_s):.6f}"]
    if dur_s is not None:
        cmd += ["-t", f"{float(dur_s):.6f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(int(sr)),
            "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=1800, creationflags=NO_WINDOW)
    except Exception as exc:
        raise SyncError(f"ffmpeg could not decode {path}: {exc}") from exc
    if proc.returncode != 0:
        raise SyncError(f"ffmpeg could not decode {path}: "
                        f"{proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# ------------------------------------------------------------------ envelope

def onset_envelope(pcm: np.ndarray, sr: int = PCM_RATE) -> np.ndarray:
    """Rectify -> 20 Hz low-pass -> 100 Hz, then the positive slope of its log.

    The log makes it gain-invariant; the slope keeps onsets (what two microphones
    agree on) and drops the slowly varying level (what they disagree on)."""
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float64)
    block = max(1, sr // (ENV_RATE * 8))                  # 8 kHz -> 800 Hz box average
    n = (pcm.size // block) * block
    rect = np.abs(pcm[:n].astype(np.float64)).reshape(-1, block).mean(axis=1)
    mid_rate = sr / block
    if rect.size > 30:
        sos = signal.butter(4, ENV_LOWPASS_HZ, btype="low", fs=mid_rate, output="sos")
        rect = signal.sosfiltfilt(sos, rect)
    step = max(1, int(round(mid_rate / ENV_RATE)))
    env = np.maximum(rect[::step], 0.0)
    floor = 1e-3 * (np.percentile(env, 99) if env.size else 0.0) + 1e-9
    log_env = np.log(env + floor)
    onset = np.maximum(np.diff(log_env, prepend=log_env[:1]), 0.0)
    onset -= onset.mean()
    std = onset.std()
    return onset / std if std > 0 else onset


def _xcorr(other: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """c[lag] = sum ref[n] * other[n + lag]; peaks at lag = offset (in samples)."""
    corr = signal.correlate(other, ref, mode="full", method="fft")
    lags = signal.correlation_lags(other.size, ref.size, mode="full")
    return corr, lags


def _peak_ratio(corr: np.ndarray, main_idx: int, exclusion: int) -> float:
    """Main peak over the highest local maximum at least `exclusion` samples away."""
    main = float(corr[main_idx])
    peaks, _ = signal.find_peaks(corr)
    far = peaks[np.abs(peaks - main_idx) > exclusion]
    if far.size == 0:
        return 99.0
    second = float(corr[far].max())
    if second <= 0:
        return 99.0
    return min(99.0, main / second)


def coarse_lag(ref_env: np.ndarray, other_env: np.ndarray,
               max_offset_s: float) -> Tuple[float, float]:
    """(lag seconds, confidence) of the onset envelopes."""
    corr, lags = _xcorr(other_env, ref_env)
    max_lag = int(round(max_offset_s * ENV_RATE))
    keep = np.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    if corr.size == 0:
        raise SyncError("No lag inside the search window")
    idx = int(np.argmax(corr))
    conf = _peak_ratio(corr, idx, int(PEAK_EXCLUSION_S * ENV_RATE))
    return float(lags[idx]) / ENV_RATE, conf


def _parabolic(corr: np.ndarray, idx: int) -> float:
    if 0 < idx < corr.size - 1:
        a, b, c = corr[idx - 1], corr[idx], corr[idx + 1]
        den = a - 2 * b + c
        if den < 0:
            return idx + 0.5 * (a - c) / den
    return float(idx)


def refine_lag(ref_pcm: np.ndarray, other_pcm: np.ndarray, coarse_s: float,
               win_start_s: float, win_len_s: float, sr: int = PCM_RATE
               ) -> Optional[float]:
    """The lag (seconds) on PCM within +-60 ms of `coarse_s`, for one ref window."""
    radius = int(round(REFINE_RADIUS_S * sr))
    a = int(round(win_start_s * sr))
    length = int(round(win_len_s * sr))
    ref = ref_pcm[a:a + length].astype(np.float64)
    lo = a + int(round(coarse_s * sr)) - radius
    hi = lo + length + 2 * radius
    if ref.size < length or lo < 0 or hi > other_pcm.size or length < sr // 4:
        return None
    other = other_pcm[lo:hi].astype(np.float64)
    if not np.any(ref) or not np.any(other):
        return None
    corr = signal.correlate(other, ref - ref.mean(), mode="valid", method="fft")
    idx = int(np.argmax(corr))
    return (lo - a + _parabolic(corr, idx)) / sr


def local_lag(ref_env: np.ndarray, other_env: np.ndarray, around_s: float,
              win_start_s: float, win_len_s: float, radius_s: float = LOCAL_RADIUS_S
              ) -> Optional[float]:
    """The envelope lag of one ref window, searched within +-radius_s of `around_s`.

    Two clocks drift apart over a long take (50 ppm is 120 ms over 40 minutes), so
    the whole-file lag can sit outside a window's +-60 ms PCM search; this puts
    each window's own coarse lag back inside it."""
    a = int(round(win_start_s * ENV_RATE))
    n = int(round(win_len_s * ENV_RATE))
    r = int(round(radius_s * ENV_RATE))
    lo = a + int(round(around_s * ENV_RATE)) - r
    lo_clip = max(0, lo)
    hi = min(other_env.size, lo + n + 2 * r)
    ref = ref_env[a:a + n]
    if ref.size < n or hi - lo_clip < n:
        return None
    corr = signal.correlate(other_env[lo_clip:hi], ref, mode="valid", method="fft")
    if corr.size == 0:
        return None
    idx = int(np.argmax(corr))
    return (lo_clip - a + _parabolic(corr, idx)) / ENV_RATE


def _windows(overlap_lo: float, overlap_hi: float) -> List[Tuple[float, float]]:
    """(start, length) of the refine windows in ref seconds, start to end of the overlap."""
    span = overlap_hi - overlap_lo
    count = int(min(MAX_WINDOWS, max(3, span // 10)))
    length = max(0.5, min(WINDOW_S, span / count))
    if count == 1 or span <= length:
        return [(overlap_lo, min(length, span))]
    step = (span - length) / (count - 1)
    return [(overlap_lo + k * step, length) for k in range(count)]


def _fit_line(measured: List[Tuple[float, float]]) -> Tuple[float, float]:
    """(intercept at ref 0, slope) of lag over ref time, one outlier pass."""
    ts = np.array([t for t, _ in measured])
    ls = np.array([lag for _, lag in measured])
    slope, intercept = np.polyfit(ts, ls, 1)
    resid = np.abs(ls - (slope * ts + intercept))
    keep = resid <= max(0.002, 3 * np.median(resid))
    if keep.sum() >= 2 and not keep.all():
        slope, intercept = np.polyfit(ts[keep], ls[keep], 1)
    return float(intercept), float(slope)


def find_offset_pcm(ref_pcm: np.ndarray, other_pcm: np.ndarray,
                    max_offset_s: float = 900.0, sr: int = PCM_RATE) -> SyncResult:
    """`find_offset` on already-decoded mono PCM."""
    ref_dur, other_dur = ref_pcm.size / sr, other_pcm.size / sr
    if ref_dur < 1.0 or other_dur < 1.0:
        raise SyncError("Both recordings need at least a second of audio")
    max_offset_s = min(float(max_offset_s), max(ref_dur, other_dur))
    ref_env, other_env = onset_envelope(ref_pcm, sr), onset_envelope(other_pcm, sr)
    coarse, confidence = coarse_lag(ref_env, other_env, max_offset_s)

    overlap_lo = max(0.0, -coarse) + LOCAL_RADIUS_S
    overlap_hi = min(ref_dur, other_dur - coarse) - LOCAL_RADIUS_S
    if overlap_hi - overlap_lo < 1.0:
        return SyncResult(round(coarse, 4), round(confidence, 3), 0.0)

    measured: List[Tuple[float, float]] = []       # (window centre in ref s, lag s)
    for start, length in _windows(overlap_lo, overlap_hi):
        around = local_lag(ref_env, other_env, coarse, start, length)
        if around is None:
            around = coarse
        lag = refine_lag(ref_pcm, other_pcm, around, start, length, sr)
        if lag is not None:
            measured.append((start + length / 2.0, lag))
    if not measured:
        return SyncResult(round(coarse, 4), round(confidence, 3), 0.0)

    drift = 0.0
    if len(measured) >= 2 and measured[-1][0] - measured[0][0] > 1.0:
        intercept, drift = _fit_line(measured)
    if abs(drift) * 1e6 > DRIFT_CORRECT_PPM:
        offset = intercept                          # lag at ref time 0 on the fitted line
    else:
        offset = float(np.median([lag for _, lag in measured]))
    return SyncResult(round(float(offset), 5), round(float(confidence), 3),
                      round(float(drift) * 1e6, 2))


def find_offset(ref_path: str, other_path: str, max_offset_s: float = 900.0) -> SyncResult:
    """How `other_path` lines up against `ref_path`: other_time = ref_time + offset_s."""
    ref = decode_mono(ref_path, PCM_RATE)
    other = decode_mono(other_path, PCM_RATE)
    result = find_offset_pcm(ref, other, max_offset_s, PCM_RATE)
    logger.info("sync %s vs %s: %s", Path(other_path).name, Path(ref_path).name, result)
    return result


# ------------------------------------------------------------------ aligned WAV

def aligned_filter(result: SyncResult, duration_s: float, rate: int = 48000) -> str:
    """ffmpeg -af chain: aligned(t) = other(offset + t * (1 + drift)), exactly
    `duration_s` long, silence where the other recording has nothing."""
    chain = [f"aresample={rate}"]
    if result.offset_s >= 0:
        chain.append(f"atrim=start={result.offset_s:.6f}")
        chain.append("asetpts=PTS-STARTPTS")
    else:
        pad = int(round(-result.offset_s * rate))
        chain.append(f"adelay={pad}S:all=1")
    if abs(result.drift_ppm) > DRIFT_CORRECT_PPM:
        chain.append(f"atempo={1.0 + result.drift_ppm * 1e-6:.8f}")
    chain.append(f"apad=whole_dur={duration_s:.6f}")
    chain.append(f"atrim=end={duration_s:.6f}")
    return ",".join(chain)


def write_aligned_wav(other_path: str, result: SyncResult, duration_s: float,
                      out_path: str, rate: int = 48000) -> str:
    """The other recording laid on the reference clock as a `rate` Hz WAV."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG_BIN, "-v", "error", "-nostdin", "-y", "-i", str(other_path), "-vn",
           "-af", aligned_filter(result, duration_s, rate),
           "-ar", str(rate), "-c:a", "pcm_s16le", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          timeout=3600, creationflags=NO_WINDOW)
    if proc.returncode != 0 or not Path(out_path).exists():
        raise SyncError(f"Could not write the aligned audio: "
                        f"{proc.stderr.decode(errors='replace')[-300:]}")
    return str(out_path)
