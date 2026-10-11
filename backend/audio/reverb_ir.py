"""Impulse responses for the reverbs (AUDIO_SUITE_PLAN.md §3, brief A3).

`synthetic_ir()` builds a stereo room response from the reverb knobs:
early reflections spread over a window set by the room size, then a tail of
noise whose echo density grows with time (diffusion), decaying exponentially
with a frequency-dependent decay time (high-frequency damping), band-limited
by the low/high cuts and decorrelated between the channels by the width.
Every IR is normalised to unit energy, so a reverb at 100 % wet is about as
loud as the dry sound. Generated files are cached as 32-bit float WAVs under
`data/ir/synthetic/<hash>.wav` (the hash covers every parameter, and the
noise is seeded from it, so the same knobs always give the same file).

`ir_library()` lists the creator's own IRs, `data/ir/user/*.wav`.
`prepared_ir_path()` turns any IR (synthetic or the creator's) into the file
the Convolution Reverb plays: resampled to 48 kHz, cut to the room-size
percentage, damped, pre-delayed, width-adjusted and normalised (cached too).
`audio/compile.py` loads the file in the graph with `amovie` into `afir`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import DATA_DIR

IR_DIR = Path(DATA_DIR) / "ir"
SYNTH_DIR = IR_DIR / "synthetic"
USER_DIR = IR_DIR / "user"

IR_RATE = 48000
MAX_IR_SECONDS = 30.0          # afir's default maxir
_VERSION = 1                    # bump when the generator changes
_LN1000 = math.log(1000.0)      # amplitude decay of 60 dB

SYNTHETIC_PREFIX = "Synthetic: "
USER_PREFIX = "User: "

# Synthetic impulses offered by the Convolution Reverb (synthetic_ir kwargs).
SYNTHETIC_IMPULSES: Dict[str, Dict[str, float]] = {
    "Small Room": dict(decay_s=0.45, pre_delay_ms=4.0, room_size=25.0, diffusion=70.0,
                       hf_damping=50.0, lf_cut_hz=80.0, hf_cut_hz=12000.0, width=60.0,
                       early_level=60.0),
    "Medium Room": dict(decay_s=0.9, pre_delay_ms=8.0, room_size=45.0, diffusion=75.0,
                        hf_damping=45.0, lf_cut_hz=60.0, hf_cut_hz=12000.0, width=80.0,
                        early_level=50.0),
    "Large Hall": dict(decay_s=2.4, pre_delay_ms=22.0, room_size=85.0, diffusion=85.0,
                       hf_damping=40.0, lf_cut_hz=50.0, hf_cut_hz=10000.0, width=100.0,
                       early_level=35.0),
    "Plate": dict(decay_s=1.6, pre_delay_ms=0.0, room_size=10.0, diffusion=100.0,
                  hf_damping=15.0, lf_cut_hz=120.0, hf_cut_hz=14000.0, width=100.0,
                  early_level=5.0),
    "Intimate Booth": dict(decay_s=0.25, pre_delay_ms=2.0, room_size=10.0, diffusion=60.0,
                           hf_damping=60.0, lf_cut_hz=120.0, hf_cut_hz=9000.0, width=40.0,
                           early_level=70.0),
    "Cave": dict(decay_s=4.5, pre_delay_ms=35.0, room_size=100.0, diffusion=40.0,
                 hf_damping=65.0, lf_cut_hz=40.0, hf_cut_hz=7000.0, width=100.0,
                 early_level=55.0),
}


def _key(payload: Dict[str, Any]) -> str:
    text = json.dumps({"v": _VERSION, **payload}, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.wav")
    sf.write(str(tmp), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")
    os.replace(tmp, path)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Unit energy per channel on average (white noise in = same RMS out)."""
    energy = float(np.mean(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=0)))
    return x / math.sqrt(energy) if energy > 0 else x


def _cuts(x: np.ndarray, sr: int, lf_cut_hz: float, hf_cut_hz: float) -> np.ndarray:
    from scipy import signal

    if 20.0 < lf_cut_hz < 0.45 * sr:
        x = signal.sosfilt(signal.butter(2, lf_cut_hz, "highpass", fs=sr, output="sos"), x, axis=0)
    if 20.0 < hf_cut_hz < 0.45 * sr:
        x = signal.sosfilt(signal.butter(2, hf_cut_hz, "lowpass", fs=sr, output="sos"), x, axis=0)
    return x


def _tail(rng: np.random.Generator, n: int, sr: int, decay_s: float, diffusion: float,
          hf_damping: float, onset_s: float) -> np.ndarray:
    """Noise tail: echo density rising to full by the mixing time (low diffusion
    = audible discrete echoes first), decaying 60 dB in `decay_s` at low
    frequencies and faster above 1 kHz by the damping."""
    from scipy import signal

    d = min(max(diffusion / 100.0, 0.0), 1.0)
    t = np.arange(n) / sr
    t_mix = 0.005 + 0.25 * (1.0 - d)
    rho0 = 300.0 + 1700.0 * d
    rho = rho0 + (4000.0 - rho0) * np.minimum(1.0, (t / t_mix) ** 2)
    prob = np.clip(rho / sr, 1e-6, 1.0)
    hits = rng.random(n) < prob
    sparse = np.where(hits, rng.choice([-1.0, 1.0], n) / np.sqrt(prob), 0.0)
    dense = rng.standard_normal(n)
    raw = math.sqrt(d) * dense + math.sqrt(1.0 - d) * sparse

    nper, hop = 1024, 256
    f, ft, z = signal.stft(raw, fs=sr, nperseg=nper, noverlap=nper - hop)
    h = min(max(hf_damping / 100.0, 0.0), 1.0)
    above = np.clip(np.log2(np.maximum(f, 1.0) / 1000.0) / math.log2(12.0), 0.0, 1.0)
    rt = np.maximum(decay_s * (1.0 - 0.85 * h * above), 0.02)
    z = z * np.exp(-_LN1000 * ft[np.newaxis, :] / rt[:, np.newaxis])
    _, x = signal.istft(z, fs=sr, nperseg=nper, noverlap=nper - hop)
    x = np.concatenate([x, np.zeros(max(0, n - len(x)))])[:n]
    x = x * np.minimum(1.0, t / max(onset_s, 0.002))
    fade = max(1, int(0.05 * n))
    x[-fade:] *= np.linspace(1.0, 0.0, fade)
    return x


def _early(rng: np.random.Generator, n: int, sr: int, decay_s: float, size: float,
           er_len: float) -> np.ndarray:
    count = int(round(6 + 14 * size))
    times = np.sort(rng.uniform(0.1, 1.0, count)) * er_len
    amps = (rng.choice([-1.0, 1.0], count) / (1.0 + 3.0 * times / er_len)
            * np.exp(-_LN1000 * times / decay_s))
    er = np.zeros(n)
    np.add.at(er, np.minimum((times * sr).astype(int), n - 1), amps)
    return er


def synthetic_ir(decay_s: float, pre_delay_ms: float, room_size: float, diffusion: float,
                 hf_damping: float, lf_cut_hz: float, hf_cut_hz: float, width: float,
                 sr: int = IR_RATE, *, early_level: float = 50.0,
                 seed: Optional[int] = None) -> np.ndarray:
    """A stereo room response, shape (samples, 2), float32, unit energy.

    decay_s      RT60 at low frequencies (s)
    pre_delay_ms silence before the first reflection
    room_size    0-100 %: early-reflection window 4-80 ms and their count
    diffusion    0-100 %: how fast the tail turns from discrete echoes to smooth
    hf_damping   0-100 %: how much faster than decay_s the highs die (above 1 kHz)
    lf_cut_hz / hf_cut_hz  band limits of the reverb
    width        0-100 %: 0 = the same response in both channels, 100 = independent
    early_level  0-100 %: early reflections' share of the energy (0-60 %)
    """
    decay_s = min(max(float(decay_s), 0.05), 20.0)
    size = min(max(room_size / 100.0, 0.0), 1.0)
    w = min(max(width / 100.0, 0.0), 1.0)
    share = 0.6 * min(max(early_level / 100.0, 0.0), 1.0)
    if seed is None:
        seed = int(_key(dict(decay_s=decay_s, pre_delay_ms=pre_delay_ms, room_size=room_size,
                             diffusion=diffusion, hf_damping=hf_damping, lf_cut_hz=lf_cut_hz,
                             hf_cut_hz=hf_cut_hz, width=width, early_level=early_level,
                             sr=sr))[:8], 16)
    rng = np.random.default_rng(seed)
    er_len = 0.004 + 0.076 * size
    n = int((er_len + decay_s * 1.15) * sr) + 1
    parts = []
    for _ in range(2):
        tail = _tail(rng, n, sr, decay_s, diffusion, hf_damping, er_len)
        tail /= max(float(np.sqrt(np.sum(tail ** 2))), 1e-12)
        er = _early(rng, n, sr, decay_s, size, er_len)
        er /= max(float(np.sqrt(np.sum(er ** 2))), 1e-12)
        parts.append(math.sqrt(1.0 - share) * tail + math.sqrt(share) * er)
    a, b = math.sqrt(1.0 - w / 2.0), math.sqrt(w / 2.0)
    ir = np.stack([a * parts[0] + b * parts[1], a * parts[0] - b * parts[1]], axis=1)
    pre = int(round(max(pre_delay_ms, 0.0) / 1000.0 * sr))
    if pre:
        ir = np.concatenate([np.zeros((pre, 2)), ir], axis=0)
    ir = _cuts(ir, sr, float(lf_cut_hz), float(hf_cut_hz))
    return _normalise(ir).astype(np.float32)


def synthetic_ir_path(decay_s: float, pre_delay_ms: float, room_size: float, diffusion: float,
                      hf_damping: float, lf_cut_hz: float, hf_cut_hz: float, width: float,
                      sr: int = IR_RATE, *, early_level: float = 50.0) -> Path:
    """synthetic_ir(...) cached as data/ir/synthetic/<hash>.wav."""
    args = dict(decay_s=round(float(decay_s), 4), pre_delay_ms=round(float(pre_delay_ms), 3),
                room_size=round(float(room_size), 3), diffusion=round(float(diffusion), 3),
                hf_damping=round(float(hf_damping), 3), lf_cut_hz=round(float(lf_cut_hz), 2),
                hf_cut_hz=round(float(hf_cut_hz), 2), width=round(float(width), 3),
                early_level=round(float(early_level), 3))
    path = SYNTH_DIR / f"{_key({'kind': 'synthetic', 'sr': int(sr), **args})}.wav"
    if not path.exists():
        early = args.pop("early_level")
        _write_wav(path, synthetic_ir(sr=int(sr), early_level=early, **args), int(sr))
    return path


def ir_library() -> List[Dict[str, Any]]:
    """The creator's own IRs: data/ir/user/*.wav as {name, path, seconds}
    (the folder is created when missing; unreadable files are skipped)."""
    import soundfile as sf

    USER_DIR.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []
    for path in sorted(USER_DIR.glob("*.wav"), key=lambda p: p.name.lower()):
        try:
            info = sf.info(str(path))
        except Exception:
            continue
        out.append({"name": path.stem, "path": str(path),
                    "seconds": round(float(info.frames) / float(info.samplerate), 3)})
    return out


def impulse_choices() -> List[str]:
    """The Convolution Reverb's Impulse menu: synthetic presets, then the library."""
    return ([SYNTHETIC_PREFIX + name for name in SYNTHETIC_IMPULSES]
            + [USER_PREFIX + item["name"] for item in ir_library()])


def impulse_source(choice: str) -> Optional[Path]:
    """The IR file behind an Impulse choice (a synthetic one is generated on
    first use); None when a library file has gone."""
    if choice.startswith(SYNTHETIC_PREFIX):
        kwargs = SYNTHETIC_IMPULSES.get(choice[len(SYNTHETIC_PREFIX):])
        return synthetic_ir_path(**kwargs) if kwargs else None
    if choice.startswith(USER_PREFIX):
        path = USER_DIR / f"{choice[len(USER_PREFIX):]}.wav"
        return path if path.exists() else None
    return None


def prepared_ir_path(source: Path, *, room_size: float = 100.0, damping_lf_hz: float = 20.0,
                     damping_hf_hz: float = 20000.0, pre_delay_ms: float = 0.0,
                     width: float = 100.0, sr: int = IR_RATE) -> Path:
    """The IR the Convolution Reverb plays, cached under data/ir/synthetic/.

    room_size     % of the impulse kept (Audition: "a percentage of the full
                  room defined by the impulse file"), faded out over its last quarter
    damping_lf_hz high-pass on the reverb (off at 20 Hz or below)
    damping_hf_hz low-pass on the reverb (off at 20 kHz or above)
    pre_delay_ms  silence added before the impulse
    width         % of the impulse's side signal (stereo IRs; 100 = as recorded)
    """
    import soundfile as sf
    from scipy import signal

    source = Path(source)
    st = source.stat()
    args = dict(room_size=round(float(room_size), 3),
                damping_lf_hz=round(float(damping_lf_hz), 2),
                damping_hf_hz=round(float(damping_hf_hz), 2),
                pre_delay_ms=round(float(pre_delay_ms), 3), width=round(float(width), 3))
    key = _key({"kind": "prepared", "src": str(source.resolve()), "size": st.st_size,
                "mtime": st.st_mtime_ns, "sr": int(sr), **args})
    path = SYNTH_DIR / f"{key}.wav"
    if path.exists():
        return path
    data, rate = sf.read(str(source), dtype="float64", always_2d=True)
    data = data[:, :2]
    if rate != sr:
        g = math.gcd(int(rate), int(sr))
        data = signal.resample_poly(data, int(sr) // g, int(rate) // g, axis=0)
    data = data[: int(MAX_IR_SECONDS * sr)]
    keep = max(64, int(round(len(data) * min(max(args["room_size"], 1.0), 100.0) / 100.0)))
    data = np.array(data[:keep], dtype=np.float64)
    fade = max(1, len(data) // 4)
    data[-fade:] *= (0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, fade)))[:, np.newaxis]
    hf = args["damping_hf_hz"] if args["damping_hf_hz"] < 20000.0 else float(sr)
    data = _cuts(data, int(sr), args["damping_lf_hz"], hf)
    if data.shape[1] == 2:
        mid, side = (data[:, 0] + data[:, 1]) / 2.0, (data[:, 0] - data[:, 1]) / 2.0
        side = side * args["width"] / 100.0
        data = np.stack([mid + side, mid - side], axis=1)
    pre = int(round(max(args["pre_delay_ms"], 0.0) / 1000.0 * sr))
    if pre:
        data = np.concatenate([np.zeros((pre, data.shape[1])), data], axis=0)
    _write_wav(path, _normalise(data), int(sr))
    return path
