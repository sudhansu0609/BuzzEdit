"""Audio analysis of a source range (AUDIO_SUITE_PLAN.md §2.3, brief A2).

`analyze()` measures a range of any media file the way Audition's Amplitude
Statistics and Loudness Radar panels do: `astats` (peak, RMS, DC offset,
noise floor), `ebur128` (integrated loudness, loudness range, true peak and
the momentary / short-term series every 100 ms), a `showspectrumpic` PNG and
a numpy mean-dB spectrum (`spectrum_db`, the curve a noise print is drawn
with). The shared decode helpers here are what the noise tools and the
process lane read audio through.
"""

import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from config import FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR
from utils.proc import NO_WINDOW

logger = logging.getLogger("audio.analysis")

SAMPLE_RATE = 48000
# Long enough for a full-length recording to decode or measure.
FFMPEG_TIMEOUT_S = 3600


class AudioToolError(RuntimeError):
    """An ffmpeg/ffprobe step failed; the message carries its stderr tail."""


# ------------------------------------------------------------------ ffmpeg

def run_ffmpeg(args: Sequence[str], timeout: float = FFMPEG_TIMEOUT_S,
               check: bool = True) -> subprocess.CompletedProcess:
    """ffmpeg with `args`; stderr is returned decoded (measurements log there)."""
    command = [FFMPEG_BIN, "-hide_banner", "-nostats", *[str(a) for a in args]]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired as exc:
        raise AudioToolError(f"ffmpeg timed out after {timeout:.0f} s") from exc
    stderr = (result.stderr or b"").decode("utf-8", "replace")
    result = subprocess.CompletedProcess(command, result.returncode, result.stdout, stderr)
    if check and result.returncode != 0:
        raise AudioToolError(f"ffmpeg failed ({result.returncode}): {stderr[-800:]}")
    return result


def range_args(start_s: Optional[float], end_s: Optional[float]) -> List[str]:
    """Input-side `-ss`/`-t` for a range; empty for the whole file."""
    args: List[str] = []
    start = max(0.0, float(start_s or 0.0))
    if start > 0.0:
        args += ["-ss", f"{start:.6f}"]
    if end_s is not None:
        duration = float(end_s) - start
        if duration <= 0.0:
            raise ValueError(f"Empty range: start {start:.3f} s, end {float(end_s):.3f} s")
        args += ["-t", f"{duration:.6f}"]
    return args


def media_duration(path: str) -> float:
    """Duration in seconds: soundfile for what it reads, else ffprobe."""
    try:
        info = sf.info(str(path))
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, creationflags=NO_WINDOW)
        return float((result.stdout or b"0").decode().strip() or 0.0)
    except Exception as exc:
        raise AudioToolError(f"Could not read the duration of {path}: {exc}") from exc


def decode_wav(path: str, out_wav: Path, start_s: Optional[float] = 0.0,
               end_s: Optional[float] = None, sample_rate: int = SAMPLE_RATE,
               mono: bool = True) -> Path:
    """The first audio stream of `path` (any container) as a float32 WAV."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    args: List[str] = ["-y", "-loglevel", "error", *range_args(start_s, end_s),
                       "-i", str(path), "-map", "0:a:0", "-vn"]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(int(sample_rate)), "-c:a", "pcm_f32le", str(out_wav)]
    run_ffmpeg(args)
    if not out_wav.exists():
        raise AudioToolError(f"ffmpeg wrote no audio for {path}")
    return out_wav


def read_mono(path: str, start_s: Optional[float] = 0.0, end_s: Optional[float] = None,
              sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """A range of `path` as mono float32 samples and their rate.

    Files soundfile reads (WAV, FLAC, …) are read directly when no resampling
    is asked for; anything else (a video, a rate change) is decoded by ffmpeg
    to a float32 WAV first.
    """
    try:
        info = sf.info(str(path))
    except Exception:
        info = None
    if info is not None and (sample_rate is None or int(sample_rate) == info.samplerate):
        rate = int(info.samplerate)
        first = int(round(max(0.0, float(start_s or 0.0)) * rate))
        last = info.frames if end_s is None else min(info.frames, int(round(float(end_s) * rate)))
        data, _ = sf.read(str(path), start=first, stop=max(first, last), dtype="float32",
                          always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), rate
    with tempfile.TemporaryDirectory(dir=str(TEMP_DIR)) as tmp:
        wav = decode_wav(path, Path(tmp) / "range.wav", start_s, end_s,
                         sample_rate=int(sample_rate or SAMPLE_RATE))
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(rate)


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of float samples in dBFS (-inf clamped to -150)."""
    if samples.size == 0:
        return -150.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(-150.0, 20.0 * math.log10(rms)) if rms > 0.0 else -150.0


# ---------------------------------------------------------------- measures

def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def astats(path: str, start_s: Optional[float] = 0.0, end_s: Optional[float] = None,
           filters: Optional[List[str]] = None) -> Dict[str, float]:
    """The "Overall" block of ffmpeg's `astats` for a range, keyed by its labels
    ("Peak level dB", "RMS level dB", "DC offset", "Noise floor dB", …).
    `filters` run before the measurement."""
    chain = ",".join([*(filters or []), "astats=measure_perchannel=none"])
    result = run_ffmpeg([*range_args(start_s, end_s), "-i", str(path), "-map", "0:a:0",
                         "-vn", "-af", chain, "-f", "null", "-"])
    stats: Dict[str, float] = {}
    in_overall = False
    for line in result.stderr.splitlines():
        if "Parsed_astats" not in line:
            continue
        body = line.split("]", 1)[-1].strip()
        if body == "Overall":
            in_overall = True
            continue
        if in_overall and ":" in body:
            key, value = body.split(":", 1)
            stats[key.strip()] = _number(value.strip().split()[0] if value.strip() else "nan")
    if not stats:
        raise AudioToolError(f"astats reported nothing for {path}")
    return stats


_EBU_FRAME = re.compile(r"\bt:\s*([0-9.]+)\s+TARGET:.*?\bM:\s*(-?[0-9.]+|-?inf)\s+S:\s*(-?[0-9.]+|-?inf)")
_EBU_I = re.compile(r"\bI:\s*(-?[0-9.]+|-?inf)\s*LUFS")
_EBU_LRA = re.compile(r"\bLRA:\s*(-?[0-9.]+|-?inf)\s*LU\b")
_EBU_PEAK = re.compile(r"\bPeak:\s*(-?[0-9.]+|-?inf)\s*dBFS")


def ebur128(path: str, start_s: Optional[float] = 0.0,
            end_s: Optional[float] = None) -> Dict[str, Any]:
    """Integrated loudness, loudness range and true peak of a range, plus the
    momentary and short-term loudness every 100 ms (ITU-R BS.1770 / EBU R128)."""
    result = run_ffmpeg([*range_args(start_s, end_s), "-i", str(path), "-map", "0:a:0", "-vn",
                         "-af", "ebur128=peak=true:framelog=info", "-f", "null", "-"])
    times: List[float] = []
    momentary: List[float] = []
    short_term: List[float] = []
    summary = ""
    lines = result.stderr.splitlines()
    for index, line in enumerate(lines):
        if "Summary:" in line:
            summary = "\n".join(lines[index:])
            break
        match = _EBU_FRAME.search(line)
        if match:
            times.append(round(_number(match.group(1)), 3))
            momentary.append(_number(match.group(2)))
            short_term.append(_number(match.group(3)))

    def pick(pattern: re.Pattern) -> float:
        found = pattern.search(summary)
        return _number(found.group(1)) if found else float("nan")

    offset = max(0.0, float(start_s or 0.0))
    return {
        "integrated_lufs": pick(_EBU_I),
        "lra_lu": pick(_EBU_LRA),
        "true_peak_dbtp": pick(_EBU_PEAK),
        "series": {
            "step_s": 0.1,
            # Source time, so the readout lines up with the waveform.
            "t": [round(offset + t, 3) for t in times],
            "momentary_lufs": momentary,
            "short_term_lufs": short_term,
        },
    }


def spectrum_png(path: str, png_path: Path, start_s: Optional[float] = 0.0,
                 end_s: Optional[float] = None) -> Path:
    """`showspectrumpic` of a range (log intensity, with legend)."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-loglevel", "error", *range_args(start_s, end_s), "-i", str(path),
                "-filter_complex", "[0:a:0]showspectrumpic=s=1024x512:legend=1[spectrum]",
                "-map", "[spectrum]", "-frames:v", "1", str(png_path)])
    if not png_path.exists():
        raise AudioToolError(f"showspectrumpic wrote nothing for {path}")
    return png_path


def spectrum_of(samples: np.ndarray, sample_rate: int, nfft: int = 2048) -> Dict[str, Any]:
    """Mean power spectrum of mono samples (Hann window, 75 % overlap).

    `mean_db` is the mean power per FFT bin normalised by the window energy,
    so white noise at X dBFS RMS reads X dB in every bin: the level scale the
    noise print and afftdn's band profile are both built on.
    """
    nfft = int(nfft)
    if nfft < 16 or nfft & (nfft - 1):
        raise ValueError(f"nfft must be a power of two >= 16, not {nfft}")
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nfft) / nfft)).astype(np.float64)
    hop = nfft // 4
    data = np.asarray(samples, dtype=np.float64)
    if data.size < nfft:
        data = np.pad(data, (0, nfft - data.size))
    count = 1 + (data.size - nfft) // hop
    power = np.zeros(nfft // 2 + 1, dtype=np.float64)
    # Blocks of frames keep memory flat on a long range.
    for first in range(0, count, 512):
        idx = np.arange(first, min(count, first + 512))[:, None] * hop + np.arange(nfft)[None, :]
        spectrum = np.fft.rfft(data[idx] * window, axis=1)
        power += np.sum(spectrum.real ** 2 + spectrum.imag ** 2, axis=0)
    power /= count * float(np.sum(window ** 2))
    mean_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    return {
        "nfft": nfft,
        "sample_rate": int(sample_rate),
        "freqs_hz": [round(float(f), 2) for f in freqs],
        "mean_db": [round(float(v), 2) for v in mean_db],
    }


def spectrum_db(path: str, start_s: Optional[float] = 0.0, end_s: Optional[float] = None,
                nfft: int = 2048) -> Dict[str, Any]:
    """`{freqs_hz, mean_db}` of a range of `path` (see `spectrum_of`)."""
    samples, rate = read_mono(path, start_s, end_s)
    return spectrum_of(samples, rate, nfft)


def analyze(path: str, start_s: Optional[float] = 0.0, end_s: Optional[float] = None,
            png_path: Optional[Path] = None) -> Dict[str, Any]:
    """Everything the Audio panel's meters and noise view show for a range.

    `png_path` is where the spectrum picture goes (a temp file when omitted).
    """
    start = max(0.0, float(start_s or 0.0))
    stats = astats(path, start, end_s)
    loudness = ebur128(path, start, end_s)
    if png_path is None:
        png_path = Path(tempfile.mkdtemp(dir=str(TEMP_DIR))) / "spectrum.png"
    picture = spectrum_png(path, png_path, start, end_s)
    end = float(end_s) if end_s is not None else media_duration(path)
    return {
        "path": str(path),
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "peak_dbfs": stats.get("Peak level dB", float("nan")),
        "rms_dbfs": stats.get("RMS level dB", float("nan")),
        "dc_offset": stats.get("DC offset", float("nan")),
        "noise_floor_dbfs": stats.get("Noise floor dB", float("nan")),
        "integrated_lufs": loudness["integrated_lufs"],
        "lra_lu": loudness["lra_lu"],
        "true_peak_dbtp": loudness["true_peak_dbtp"],
        "loudness_series": loudness["series"],
        "spectrum_png": str(picture),
        "spectrum": spectrum_db(path, start, end_s),
    }


def json_safe(value: Any) -> Any:
    """NaN/inf → None, recursively, so a result can go out as JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
