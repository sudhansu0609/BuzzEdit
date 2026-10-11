"""Voice Changer (AUDIO_SUITE_PLAN.md §3 "Voice Changer", brief A3).

`change_voice()` re-synthesises the voice with WORLD (pyworld): f0 from
harvest is scaled by 2^(semitones/12); the spectral envelope (cheaptrick) is
warped along the frequency axis by the formant ratio -- 1.3 moves every
formant 30 % up, independent of the pitch; the aperiodicity (d4c) passes
through. Speed other than 1.0 is applied afterwards by Rubber Band offline.
At speed 1.0 the output has exactly the input's length, rate and channel
count; its overall RMS matches the input's.

Long files are processed in ~20 s chunks cut at the quietest 20 ms near
each boundary, each analysed with 0.3 s of context on both sides and joined
with a 10 ms crossfade, so memory stays flat.

`VOICE_PRESETS` are the creator's "voice change" characters: pitch, formant
and speed for WORLD plus an `fx_rack` of AudioEffect-shaped dicts using the
ffmpeg catalogue (audio/catalogue.py), run after the WORLD pass by
`audio/processes_ext.py`.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from audio.catalogue import EFFECTS
from config import FFMPEG_BIN

Progress = Optional[Callable[[float, str], None]]

FRAME_PERIOD_MS = 5.0
F0_FLOOR, F0_CEIL = 50.0, 1000.0
CHUNK_S = 20.0          # nominal chunk length
SEARCH_S = 4.0          # look this far either side of a nominal boundary for a quiet spot
CONTEXT_S = 0.3         # analysis context on both sides of a chunk
XFADE_S = 0.010         # crossfade at a chunk boundary


# --- presets ------------------------------------------------------------------------

def _fx(effect: str, preset: Optional[str] = None, **params: Any) -> Dict[str, Any]:
    """An AudioEffect-shaped rack entry from a catalogue preset plus overrides."""
    base = dict(EFFECTS[effect]["presets"][preset]) if preset else {}
    base.update(params)
    return {"effect": effect, "params": base, "enabled": True, "preset": preset}


def _voice(pitch: float, formant: float, speed: float,
           *fx: Dict[str, Any]) -> Dict[str, Any]:
    rack = [{"id": f"vc{i}_{d['effect']}", **d} for i, d in enumerate(fx)]
    return {"pitch_semitones": pitch, "formant_ratio": formant, "speed": speed, "fx_rack": rack}


VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Deeper": _voice(-3.0, 0.92, 1.0),
    "Lighter": _voice(3.0, 1.08, 1.0),
    "Giant": _voice(-6.0, 0.78, 0.92, _fx("reverb", "Great Hall", wet=22.0)),
    "Child": _voice(6.0, 1.22, 1.06),
    # metallic: a static 4 ms comb with heavy feedback, a little grit, band-limited
    "Robot": _voice(0.0, 1.0, 1.0,
                    _fx("flanger", initial_delay=4.0, final_delay=4.0, feedback=85.0,
                        rate=0.1, mix=70.0),
                    _fx("distortion", "Warm Saturation"),
                    _fx("parametric_eq", hp_on=True, hp_frequency=200.0, band3_gain=3.0)),
    "Alien": _voice(5.0, 1.4, 1.0, _fx("phaser", "Deep Space"),
                    _fx("flanger", "Light Flange"), _fx("reverb", "Vocal Ambience (Small)")),
    "Telephone": _voice(0.0, 1.0, 1.0, _fx("scientific_filter", "Telephone (300-3400 Hz)"),
                        _fx("distortion", "Warm Saturation")),
    "Radio": _voice(0.0, 1.0, 1.0,
                    _fx("scientific_filter", filter_type="Butterworth", mode="Band Pass",
                        cutoff=200.0, high_cutoff=5000.0, order=2.0),
                    _fx("distortion", "Warm Saturation"), _fx("dynamics", "Broadcast")),
    "Megaphone": _voice(0.0, 1.05, 1.0,
                        _fx("parametric_eq", hp_on=True, hp_frequency=500.0, lp_on=True,
                            lp_frequency=4000.0, band3_frequency=1800.0, band3_gain=8.0),
                        _fx("distortion", "Soft Clip")),
    # approximate: WORLD keeps the voicing, so the whisper is an airy EQ + a small room
    "Whisper": _voice(0.0, 1.08, 1.0,
                      _fx("parametric_eq", hp_on=True, hp_frequency=350.0, ls_gain=-6.0,
                          band4_gain=4.0, hs_frequency=6000.0, hs_gain=8.0),
                      _fx("dynamics", "Gentle Compression"),
                      _fx("reverb", "Vocal Ambience (Small)", wet=12.0)),
    "Ghost": _voice(1.0, 1.05, 0.97, _fx("flanger", "Light Flange"),
                    _fx("parametric_eq", hp_on=True, hp_frequency=200.0),
                    _fx("reverb", "Great Hall", wet=55.0)),
    "Demon": _voice(-7.0, 0.78, 0.97, _fx("sub_octave", "Deep Double"),
                    _fx("distortion", "Warm Saturation"), _fx("reverb", "Horror Intimate")),
    "Monster": _voice(-10.0, 0.68, 0.92, _fx("sub_octave", "Monster Low"),
                      _fx("distortion", "Soft Clip"), _fx("reverb", "Cave", wet=30.0)),
}


# --- WORLD --------------------------------------------------------------------------

def _warp_envelope(sp: np.ndarray, ratio: float) -> np.ndarray:
    """Move every formant by `ratio` along the frequency axis: the new
    envelope at f is the old one at f / ratio (log-domain interpolation)."""
    if abs(ratio - 1.0) < 1e-6:
        return sp
    nb = sp.shape[1]
    src = np.clip(np.arange(nb) / ratio, 0.0, nb - 1.0)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, nb - 1)
    frac = src - i0
    logsp = np.log(np.maximum(sp, 1e-30))
    return np.ascontiguousarray(np.exp(logsp[:, i0] * (1.0 - frac) + logsp[:, i1] * frac))


def _world(x: np.ndarray, sr: int, pitch_semitones: float, formant_ratio: float) -> np.ndarray:
    """One mono segment through WORLD; returns exactly len(x) samples."""
    import pyworld as pw

    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD_MS)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    f0 = np.minimum(f0 * 2.0 ** (pitch_semitones / 12.0), 0.45 * sr)
    y = pw.synthesize(f0, _warp_envelope(sp, formant_ratio), ap, sr, FRAME_PERIOD_MS)
    if len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
    return y[:len(x)]


def _cut_points(x: np.ndarray, sr: int) -> List[int]:
    """Chunk boundaries at the quietest 20 ms near every CHUNK_S."""
    n = len(x)
    hop = max(1, int(0.02 * sr))
    if n <= (CHUNK_S + SEARCH_S) * sr:
        return []
    m = n // hop
    energy = (x[: m * hop].reshape(m, hop) ** 2).mean(axis=1)
    points: List[int] = []
    pos = int(CHUNK_S * sr)
    while pos < n - int(SEARCH_S * sr):
        lo = max(0, (pos - int(SEARCH_S * sr)) // hop)
        hi = min(m, (pos + int(SEARCH_S * sr)) // hop)
        cut = (lo + int(np.argmin(energy[lo:hi]))) * hop + hop // 2
        points.append(cut)
        pos = cut + int(CHUNK_S * sr)
    return points


def _process_channel(x: np.ndarray, sr: int, pitch: float, formant: float,
                     report: Callable[[float], None]) -> np.ndarray:
    n = len(x)
    bounds = [0] + _cut_points(x, sr) + [n]
    ctx, xf = int(CONTEXT_S * sr), max(1, int(XFADE_S * sr / 2))
    out = np.zeros(n)
    chunks = len(bounds) - 1
    for i in range(chunks):
        a, b = bounds[i], bounds[i + 1]
        lo, hi = max(0, a - ctx), min(n, b + ctx)
        y = _world(x[lo:hi], sr, pitch, formant)
        w = np.zeros(hi - lo)
        s, e = (a - xf if i > 0 else a) - lo, (b + xf if i < chunks - 1 else b) - lo
        w[s:e] = 1.0
        if i > 0:      # rise across [a - xf, a + xf), complementary to the previous fall
            w[s:s + 2 * xf] = np.linspace(0.0, 1.0, 2 * xf, endpoint=False) + 0.5 / (2 * xf)
        if i < chunks - 1:
            w[e - 2 * xf:e] = 1.0 - (np.linspace(0.0, 1.0, 2 * xf, endpoint=False)
                                     + 0.5 / (2 * xf))
        out[lo:hi] += w * y
        report((i + 1) / chunks)
    return out


def _rubberband_speed(src: Path, dst: Path, speed: float) -> None:
    r = subprocess.run([FFMPEG_BIN, "-hide_banner", "-nostats", "-v", "error", "-y",
                        "-i", str(src), "-filter_complex",
                        f"[0:a]rubberband=tempo={speed:.6f}:pitchq=quality[out]",
                        "-map", "[out]", "-c:a", "pcm_f32le", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"rubberband speed change failed: {r.stderr[-800:]}")


def change_voice(in_wav: Path, out_wav: Path, pitch_semitones: float, formant_ratio: float,
                 speed: float = 1.0, progress: Progress = None) -> Path:
    """Pitch (semitones), formant (ratio) and speed (1.0 = unchanged) of a voice."""
    import soundfile as sf

    in_wav, out_wav = Path(in_wav), Path(out_wav)
    pitch = float(min(max(pitch_semitones, -24.0), 24.0))
    formant = float(min(max(formant_ratio, 0.5), 2.0))
    speed = float(min(max(speed, 0.5), 2.0))
    x, sr = sf.read(str(in_wav), dtype="float64", always_2d=True)
    n, nch = x.shape
    if progress:
        progress(0.02, "Voice Changer: analysing")
    same = nch > 1 and float(np.max(np.abs(x - x[:, :1]))) < 1e-4
    todo = 1 if (nch == 1 or same) else nch
    y = np.zeros((n, nch))
    for c in range(todo):
        def report(frac: float, c: int = c) -> None:
            if progress:
                progress(0.05 + 0.8 * (c + frac) / todo, "Voice Changer: re-synthesising")
        y[:, c] = _process_channel(x[:, c], sr, pitch, formant, report) if n else x[:, c]
    if todo == 1 and nch > 1:
        y[:, 1:] = y[:, :1]
    rms_in = float(np.sqrt(np.mean(x ** 2))) if n else 0.0
    rms_out = float(np.sqrt(np.mean(y ** 2))) if n else 0.0
    if rms_out > 1e-9 and rms_in > 0:
        y *= rms_in / rms_out
    peak = float(np.max(np.abs(y))) if n else 0.0
    if peak > 0.999:
        y *= 0.999 / peak
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if abs(speed - 1.0) < 1e-6:
        sf.write(str(out_wav), y.astype(np.float32), sr, subtype="FLOAT")
    else:
        if progress:
            progress(0.88, f"Voice Changer: speed x{speed:g}")
        with tempfile.TemporaryDirectory(prefix="voice_changer_") as tmp:
            mid = Path(tmp) / "world.wav"
            sf.write(str(mid), y.astype(np.float32), sr, subtype="FLOAT")
            _rubberband_speed(mid, out_wav, speed)
    if progress:
        progress(1.0, "Voice Changer: done")
    return out_wav
