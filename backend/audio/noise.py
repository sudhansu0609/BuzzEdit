"""Detect noise, capture a noise print, and the noise-reduction engines
(AUDIO_SUITE_PLAN.md §2.3, brief A2).

Audition's workflow: find a stretch of room tone (`detect_noise`), capture
it as a print (`capture_print`: `audio/noise_print.json` + `.wav` under the
project), then reduce by it -- live with `afftdn` fed the print as a custom
15-band profile (`afftdn_chain`), or offline with `noisereduce`'s stationary
gate (`noisereduce_process`). `rnnoise_chain` is the speech-trained RNNoise.

The project plumbing at the top (store, directories, which source is meant)
is shared with the process lane (`audio.processes`) and the routes.
"""

import json
import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

from asr.vad import analyse_speech
from audio.analysis import (
    SAMPLE_RATE,
    decode_wav,
    media_duration,
    read_mono,
    rms_dbfs,
    spectrum_of,
)
from config import PROJECTS_DIR, TEMP_DIR
from render.audio import _ffmpeg_filter_path, rnnoise_model_path
from store.project_store import ProjectStore
from timeline.schema import SourceFile, Timeline

logger = logging.getLogger("audio.noise")

project_store = ProjectStore(base_dir=str(PROJECTS_DIR))

PRINT_JSON = "noise_print.json"
PRINT_WAV = "noise_print.wav"
PRINT_NFFT = 2048
WINDOW_S = 0.8            # shortest noise window worth printing
MIN_PRINT_S = 0.1         # a capture shorter than this has no stable spectrum
GAP_MARGIN_S = 0.1        # kept clear of word / speech edges
DETECT_RATE = 16000       # level map rate: enough to rank windows, small on a long file
FRAME_S = 0.02


class NotFound(LookupError):
    """A project, source, print or file the request names does not exist."""


class Conflict(RuntimeError):
    """The request cannot run in the project's current state."""


# --------------------------------------------------------- project plumbing

def project_dir(project_id: str) -> Path:
    return Path(PROJECTS_DIR) / project_id


def audio_dir(project_id: str) -> Path:
    return project_dir(project_id) / "audio"


def load_project(project: Union[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Timeline]:
    """`(project_id, project dict, Timeline)` from a project id or a project dict."""
    if isinstance(project, dict):
        p_data: Optional[Dict[str, Any]] = project
        project_id = str(project.get("id") or "")
    else:
        project_id = str(project)
        p_data = project_store.get_project(project_id)
    if not p_data or not p_data.get("timeline"):
        raise NotFound("Project or timeline not found")
    if not project_id:
        raise ValueError("The project has no id")
    return project_id, p_data, Timeline.model_validate(p_data["timeline"])


def _same_path(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except Exception:
        return False


def main_source_id(tl: Timeline, p_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The programme voice: the first A1 clip's source, else the transcribed
    `source_video`, else the first source with audio."""
    a1 = sorted((i for i in tl.items if i.track == "A1" and i.source_id in tl.sources),
                key=lambda i: i.timeline_start_frame)
    if a1:
        return a1[0].source_id
    source_video = (p_data or {}).get("source_video")
    if source_video:
        for sid, src in tl.sources.items():
            if _same_path(src.path, source_video):
                return sid
    for sid, src in tl.sources.items():
        if src.has_audio and src.kind in ("video", "audio"):
            return sid
    return None


def resolve_source(tl: Timeline, p_data: Optional[Dict[str, Any]],
                   source_id: Optional[str]) -> Tuple[str, SourceFile]:
    sid = source_id or main_source_id(tl, p_data)
    if not sid or sid not in tl.sources:
        raise NotFound(f"No source {source_id!r} in the timeline" if source_id
                       else "The project has no source with audio")
    return sid, tl.sources[sid]


def source_audio(source: SourceFile) -> str:
    """What the source sounds like now: its processed version, else its file."""
    path = source.audio_override or source.path
    if not path or not Path(path).exists():
        raise NotFound(f"Source audio not found: {path}")
    return path


def transcript_words(p_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Speech spans (seconds) from the project transcript: word timings, or a
    segment's own span when it carries no words."""
    transcript = p_data.get("transcript")
    segments = transcript.get("segments") if isinstance(transcript, dict) else transcript
    spans: List[Tuple[float, float]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        words = seg.get("words") or []
        if not words and isinstance(seg.get("start"), (int, float)) and isinstance(seg.get("end"), (int, float)):
            words = [seg]
        for word in words:
            start, end = word.get("start"), word.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                spans.append((float(start), float(end)))
    return sorted(spans)


# ------------------------------------------------------------ detect noise

def _gaps_from_words(words: Sequence[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    gaps: List[Tuple[float, float]] = []
    previous = 0.0
    for start, end in words:
        if start > previous:
            gaps.append((previous, start))
        previous = max(previous, end)
    if duration > previous:
        gaps.append((previous, duration))
    return gaps


def _frame_power(samples: np.ndarray, rate: int) -> np.ndarray:
    hop = max(1, int(round(rate * FRAME_S)))
    count = len(samples) // hop
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    frames = samples[: count * hop].astype(np.float64).reshape(count, hop)
    power = np.mean(frames ** 2, axis=1)
    # Digital silence (an edit, a gated gap) is no room tone: never print it.
    power[power < 1e-12] = np.inf
    return power


def _quietest(power: np.ndarray, spans: Sequence[Tuple[float, float]], window_s: float,
              margin_s: float) -> Optional[Tuple[float, float, float]]:
    """`(mean power, start_s, end_s)` of the quietest `window_s` window inside
    any span (shrunk by `margin_s` each side), or None when none fits."""
    n = int(round(window_s / FRAME_S))
    if power.size < n or n <= 0:
        return None
    finite = np.where(np.isfinite(power), power, 1e30)
    csum = np.concatenate([[0.0], np.cumsum(finite)])
    best: Optional[Tuple[float, float, float]] = None
    for start, end in spans:
        first = int(math.ceil((start + margin_s) / FRAME_S - 1e-9))
        last = min(power.size, int(math.floor((end - margin_s) / FRAME_S + 1e-9)))
        if last - first < n:
            continue
        sums = csum[first + n:last + 1] - csum[first:last - n + 1]
        j = int(np.argmin(sums))
        mean = float(sums[j]) / n
        if mean >= 1e29:
            continue
        if best is None or mean < best[0]:
            best = (mean, (first + j) * FRAME_S, (first + j + n) * FRAME_S)
    return best


def detect_noise(project: Union[str, Dict[str, Any]], source_id: Optional[str] = None) -> Dict[str, Any]:
    """The quietest >= 0.8 s window inside the non-speech gaps of a source.

    Gaps come from the project transcript's word timings when the source is
    the transcribed programme voice, else from `asr/vad.py`; with no gap long
    enough it is the quietest 0.8 s of the whole file. `floor_dbfs` is that
    window's RMS at 48 kHz.
    """
    project_id, p_data, tl = load_project(project)
    sid, source = resolve_source(tl, p_data, source_id)
    audio_path = source_audio(source)
    words = transcript_words(p_data) if sid == main_source_id(tl, p_data) else []

    with tempfile.TemporaryDirectory(dir=str(TEMP_DIR)) as tmp:
        wav = decode_wav(audio_path, Path(tmp) / "detect.wav", sample_rate=DETECT_RATE)
        samples, rate = sf.read(str(wav), dtype="float32")
        duration = len(samples) / float(rate)
        if words:
            gaps, method = _gaps_from_words(words, duration), "transcript"
        else:
            speech_map = analyse_speech(str(wav))
            gaps = speech_map.silences(min_duration=WINDOW_S) if speech_map else []
            method = "vad"
    power = _frame_power(samples, rate)
    del samples

    found = _quietest(power, gaps, WINDOW_S, GAP_MARGIN_S)
    if found is None:
        found = _quietest(power, [(0.0, duration)], WINDOW_S, 0.0)
        method = "whole_file"
    if found is None:
        raise ValueError(f"The source is shorter than {WINDOW_S} s or holds no sound")
    _mean, start, end = found
    window, _rate = read_mono(audio_path, start, end, sample_rate=SAMPLE_RATE)
    return {
        "source_id": sid,
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "floor_dbfs": round(rms_dbfs(window), 2),
        "method": method,
    }


# ------------------------------------------------------------- noise print

def print_paths(project_id: str) -> Tuple[Path, Path]:
    folder = audio_dir(project_id)
    return folder / PRINT_JSON, folder / PRINT_WAV


def load_print(project_id: str) -> Optional[Dict[str, Any]]:
    json_path, _wav = print_paths(project_id)
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def capture_print(project: Union[str, Dict[str, Any]], source_id: Optional[str],
                  start_s: float, end_s: float) -> Dict[str, Any]:
    """Store a range of the source as the project's noise print and return it.

    `audio/noise_print.wav` is the range at 48 kHz mono; `noise_print.json`
    holds its mean-dB spectrum and the 15 afftdn band values derived from it.
    """
    project_id, p_data, tl = load_project(project)
    sid, source = resolve_source(tl, p_data, source_id)
    audio_path = source_audio(source)
    start = max(0.0, float(start_s))
    end = min(float(end_s), media_duration(audio_path))
    if end - start < MIN_PRINT_S:
        raise ValueError(f"A noise print needs at least {MIN_PRINT_S} s of audio "
                         f"(got {start:.3f}-{end:.3f} s)")

    json_path, wav_path = print_paths(project_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = wav_path.with_name("noise_print.tmp.wav")
    decode_wav(audio_path, tmp_wav, start, end, sample_rate=SAMPLE_RATE, mono=True)
    samples, rate = sf.read(str(tmp_wav), dtype="float32")
    floor = rms_dbfs(samples)
    if floor <= -140.0:
        tmp_wav.unlink(missing_ok=True)
        raise ValueError("The range is digital silence: pick one with the room noise in it")
    spectrum = spectrum_of(samples, rate, PRINT_NFFT)
    band_floor, bands = afftdn_profile(spectrum)

    noise_print = {
        "source_id": sid,
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "floor_dbfs": round(floor, 2),
        "nfft": PRINT_NFFT,
        "sample_rate": int(rate),
        "freqs_hz": spectrum["freqs_hz"],
        "mean_db": spectrum["mean_db"],
        "bands_db": bands,
        "band_floor_db": band_floor,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    os.replace(tmp_wav, wav_path)
    tmp_json = json_path.with_name("noise_print.tmp.json")
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(noise_print, fh)
    os.replace(tmp_json, json_path)
    return noise_print


# ------------------------------------------------------- afftdn with print

def _afftdn_band_centres() -> List[int]:
    """afftdn's 15 profile band centres (Hz), as `af_afftdn.c` builds them."""
    centres = [80]
    for _ in range(14):
        c = int(math.floor(1.5 * centres[-1] + 5.0 + 0.5))
        if c < 1000:
            c = 10 * (c // 10)
        elif c < 5000:
            c = 50 * ((c + 20) // 50)
        elif c < 15000:
            c = 100 * ((c + 45) // 100)
        else:
            c = 1000 * ((c + 495) // 1000)
        centres.append(c)
    return centres


AFFTDN_BAND_CENTRES = _afftdn_band_centres()   # 80, 120, 180 … 16000, 24000
# afftdn's internal level of white noise sits 10*log10(2) dB under its RMS in
# dBFS (af_afftdn.c: band power / floor, floor = 2^48 * 1e-10 * window weight).
_AFFTDN_LEVEL_OFFSET_DB = 10.0 * math.log10(2.0)


def afftdn_profile(noise_print: Dict[str, Any]) -> Tuple[float, List[float]]:
    """`(noise floor, 15 band values)` for afftdn's custom profile.

    Each band is the mean power of the print's spectrum between afftdn's band
    edges (centre / sqrt(1.5) up to the next centre / sqrt(1.5)), on afftdn's
    own level scale. The floor is their mean (afftdn's `nf`); the band values
    are each band relative to it, clipped to afftdn's +-24 dB (`bn`).
    """
    freqs = np.asarray(noise_print["freqs_hz"], dtype=np.float64)
    mean_db = np.asarray(noise_print["mean_db"], dtype=np.float64)
    nyquist = float(noise_print.get("sample_rate") or 2.0 * freqs[-1]) / 2.0
    power = 10.0 ** (mean_db / 10.0)
    centres = AFFTDN_BAND_CENTRES
    edges = [min(c / 1.224745, nyquist) for c in centres] + [min(centres[-1] * 1.224745, nyquist)]
    levels: List[float] = []
    for band in range(len(centres)):
        low, high = edges[band], edges[band + 1]
        chosen = (freqs >= low) & ((freqs < high) | ((band == len(centres) - 1) & (freqs <= high)))
        if chosen.any():
            levels.append(10.0 * math.log10(max(float(power[chosen].mean()), 1e-20))
                          - _AFFTDN_LEVEL_OFFSET_DB)
        elif levels:
            levels.append(levels[-1])        # band above Nyquist: afftdn repeats the last
        else:
            nearest = int(np.argmin(np.abs(freqs - centres[band])))
            levels.append(10.0 * math.log10(max(float(power[nearest]), 1e-20))
                          - _AFFTDN_LEVEL_OFFSET_DB)
    floor = float(np.mean(levels))
    bands = [round(max(-24.0, min(24.0, level - floor)), 2) for level in levels]
    return round(floor, 2), bands


def afftdn_chain(noise_print: Dict[str, Any], reduce_by_db: float = 24.0,
                 noise_floor_offset_db: float = 6.0, residual_floor_db: float = -38.0,
                 output_noise_only: bool = False, effect_id: Optional[str] = None) -> List[str]:
    """One live-capable afftdn instance reducing by the captured print.

    `nt=c:bn=` carries the print's 15-band shape and `nf` its level plus
    `noise_floor_offset_db` (above the measured mean is where the gain starts
    to bite: +6 dB takes white noise down ~28 dB at nr=40); `nr` is Reduce by,
    `om=noise` lets the user hear what is removed, `tn=0` keeps the print's
    floor fixed. `rf` is passed as given; afftdn only reads it with
    track_residual on. Named `afftdn@<effect_id>_0` for the live engine. No
    asendcmd and no in-stream noise region: the print is the profile.
    """
    band_floor, bands = afftdn_profile(noise_print)
    nf = max(-80.0, min(-20.0, band_floor + float(noise_floor_offset_db)))
    nr = max(0.01, min(97.0, float(reduce_by_db)))
    rf = max(-80.0, min(-20.0, float(residual_floor_db)))
    name = f"afftdn@{effect_id}_0" if effect_id else "afftdn"
    profile = "|".join(f"{b:.2f}" for b in bands)
    mode = "noise" if output_noise_only else "output"
    return [f"{name}=nr={nr:.2f}:nf={nf:.2f}:rf={rf:.2f}:nt=c:bn={profile}:om={mode}:tn=0"]


def rnnoise_chain(mix: float = 1.0) -> List[str]:
    """RNNoise (`arnndn`) with the model under data/models/rnnoise/."""
    model = rnnoise_model_path()
    if model is None:
        raise NotFound("No RNNoise model (.rnnn) under data/models/rnnoise/")
    mix = max(-1.0, min(1.0, float(mix)))
    return [f"arnndn=m='{_ffmpeg_filter_path(model)}':mix={mix:.3f}"]


# ------------------------------------------------- noisereduce with print

def noisereduce_process(in_wav: str, out_wav: str, print_wav: str, prop_decrease: float = 1.0,
                        n_std_thresh: float = 1.5, freq_smooth_hz: float = 500.0,
                        time_smooth_ms: float = 50.0, n_fft: int = 1024,
                        output_noise_only: bool = False) -> Path:
    """Noise Reduction (process): `noisereduce`'s stationary spectral gate with
    the print as `y_noise`, channel by channel. Writes a float32 WAV of the same
    length and rate; `output_noise_only` writes what was removed instead."""
    import noisereduce

    n_fft = int(n_fft)
    if n_fft < 256 or n_fft > 16384 or n_fft & (n_fft - 1):
        raise ValueError(f"FFT size must be a power of two in 256..16384, not {n_fft}")
    data, rate = sf.read(str(in_wav), dtype="float32", always_2d=True)
    noise, _ = read_mono(str(print_wav), sample_rate=rate)
    if noise.size < n_fft:
        raise ValueError(f"The noise print is shorter than one FFT frame ({n_fft} samples)")
    if not np.any(noise):
        raise ValueError("The noise print is digital silence")
    hop = n_fft // 4
    # noisereduce refuses smoothing narrower than one step of its grid.
    freq_smooth_hz = max(float(freq_smooth_hz), 1.001 * rate / (n_fft / 2))
    time_smooth_ms = max(float(time_smooth_ms), 1.001 * 1000.0 * hop / rate)
    prop_decrease = max(0.0, min(1.0, float(prop_decrease)))

    out = np.empty_like(data)
    for ch in range(data.shape[1]):
        signal = data[:, ch]
        if signal.size < n_fft:
            out[:, ch] = 0.0 if output_noise_only else signal
            continue
        clean = noisereduce.reduce_noise(
            y=signal, sr=rate, stationary=True, y_noise=noise, prop_decrease=prop_decrease,
            n_std_thresh_stationary=float(n_std_thresh), freq_mask_smooth_hz=freq_smooth_hz,
            time_mask_smooth_ms=time_smooth_ms, n_fft=n_fft, use_tqdm=False, n_jobs=1)
        clean = np.asarray(clean, dtype=np.float32).reshape(-1)[: signal.size]
        if clean.size < signal.size:
            clean = np.pad(clean, (0, signal.size - clean.size))
        out[:, ch] = signal - clean if output_noise_only else clean
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out[:, 0] if out.shape[1] == 1 else out, rate, subtype="FLOAT")
    return out_path
