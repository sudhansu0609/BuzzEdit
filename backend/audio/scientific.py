"""Scientific Filter design (AUDIO_SUITE_PLAN.md §3, brief A3).

Audition's Scientific Filter: Bessel, Butterworth, Chebyshev I, Chebyshev II
and Elliptic designs in low-pass, high-pass, band-pass and band-stop mode.
The filter is designed with scipy.signal at 48 kHz (the live and render rate)
as second-order sections; `audio/compile.py` turns each section into one
ffmpeg `biquad` instance. Biquad coefficients take runtime commands, so a
change of type, cutoff, ripple or attenuation is live; order and mode change
how many sections exist and therefore restart the stream.

Conventions (scipy's):
  * Butterworth and Bessel: the cutoff is the -3 dB point (Bessel is
    designed with norm="mag" so both mean the same thing);
  * Chebyshev I and Elliptic: the cutoff is the edge of the pass band, where
    the response leaves the ripple band (-ripple dB);
  * Chebyshev II: the cutoff is the start of the stop band, where the
    attenuation first reaches the stop-band attenuation.
A band mode of order N has 2N poles (N sections); a low/high pass has
ceil(N/2) sections.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

DESIGN_RATE = 48000

FILTER_TYPES = {"Bessel": "bessel", "Butterworth": "butter", "Chebyshev I": "cheby1",
                "Chebyshev II": "cheby2", "Elliptic": "ellip"}
FILTER_MODES = {"Low Pass": "lowpass", "High Pass": "highpass", "Band Pass": "bandpass",
                "Band Stop": "bandstop"}
BAND_MODES = ("Band Pass", "Band Stop")

MIN_HZ = 10.0
MAX_ORDER = 20


def max_hz(sr: int = DESIGN_RATE) -> float:
    """Highest usable edge: just under Nyquist."""
    return 0.475 * sr


def section_count(mode: str, order: float) -> int:
    """How many biquads a design of this mode and order needs."""
    n = int(max(1, min(MAX_ORDER, round(order))))
    return n if mode in BAND_MODES else (n + 1) // 2


def band_edges(mode: str, cutoff_hz: float, high_cutoff_hz: float,
               sr: int = DESIGN_RATE) -> Tuple[float, ...]:
    """The edge frequencies in Hz, kept inside (MIN_HZ, max_hz) and, for the
    band modes, ordered and at least 1 % apart."""
    top = max_hz(sr)
    lo = min(max(float(cutoff_hz), MIN_HZ), top)
    if mode not in BAND_MODES:
        return (lo,)
    hi = min(max(float(high_cutoff_hz), MIN_HZ), top)
    lo, hi = min(lo, hi), max(lo, hi)
    if hi < lo * 1.01:
        hi = min(lo * 1.01, top)
        lo = min(lo, hi / 1.01)
    return (lo, hi)


def _balance(sos: np.ndarray) -> np.ndarray:
    """Spread the overall gain evenly over the sections (scipy puts it all in
    the first one, which can be 1e-20 for a steep low cutoff): each section is
    scaled to the same peak magnitude, and the product of the scale factors is
    1, so the cascade is unchanged."""
    from scipy import signal

    sos = np.array(sos, dtype=np.float64)
    peaks = []
    for row in sos:
        _, h = signal.sosfreqz(row[np.newaxis, :], worN=4096)
        peaks.append(max(float(np.max(np.abs(h))), 1e-300))
    peaks = np.array(peaks)
    target = float(np.exp(np.mean(np.log(peaks))))
    for i, peak in enumerate(peaks):
        sos[i, :3] *= target / peak
    return sos


@lru_cache(maxsize=512)
def _design(filter_type: str, mode: str, edges: Tuple[float, ...], order: int, ripple_db: float,
            attenuation_db: float, sr: int) -> Tuple[Tuple[float, ...], ...]:
    from scipy import signal

    kind = FILTER_TYPES[filter_type]
    btype = FILTER_MODES[mode]
    nyq = sr / 2.0
    wn = [e / nyq for e in edges]
    wn = wn[0] if len(wn) == 1 else wn
    rp = max(float(ripple_db), 0.01)
    rs = max(float(attenuation_db), rp + 0.1)
    if kind == "bessel":
        sos = signal.bessel(order, wn, btype=btype, output="sos", norm="mag")
    elif kind == "butter":
        sos = signal.butter(order, wn, btype=btype, output="sos")
    elif kind == "cheby1":
        sos = signal.cheby1(order, rp, wn, btype=btype, output="sos")
    elif kind == "cheby2":
        sos = signal.cheby2(order, rs, wn, btype=btype, output="sos")
    else:
        sos = signal.ellip(order, rp, rs, wn, btype=btype, output="sos")
    sos = _balance(sos)
    assert len(sos) == section_count(mode, order), (filter_type, mode, order, len(sos))
    return tuple(tuple(float(c) for c in row) for row in sos)


def design_sos(filter_type: str, mode: str, cutoff_hz: float, high_cutoff_hz: float,
               order: float, ripple_db: float, attenuation_db: float,
               sr: int = DESIGN_RATE) -> np.ndarray:
    """Second-order sections, shape (sections, 6) as [b0, b1, b2, a0, a1, a2]
    with a0 = 1 -- the coefficient order of ffmpeg's `biquad`."""
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"unknown filter type {filter_type!r}")
    if mode not in FILTER_MODES:
        raise ValueError(f"unknown filter mode {mode!r}")
    n = int(max(1, min(MAX_ORDER, round(order))))
    edges = tuple(round(e, 6) for e in band_edges(mode, cutoff_hz, high_cutoff_hz, sr))
    rows = _design(filter_type, mode, edges, n, round(float(ripple_db), 6),
                   round(float(attenuation_db), 6), int(sr))
    return np.array(rows, dtype=np.float64)


def response_db(sos: np.ndarray, freqs_hz, sr: int = DESIGN_RATE) -> np.ndarray:
    """Magnitude response of a design in dB at the given frequencies."""
    from scipy import signal

    _, h = signal.sosfreqz(np.asarray(sos), worN=np.asarray(freqs_hz, dtype=np.float64), fs=sr)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))
