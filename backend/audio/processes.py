"""The process lane (AUDIO_SUITE_PLAN.md §2.1, brief A2).

Audition's "(process)" effects rewrite the audio instead of running in a
rack. A run decodes the source as it sounds now (its `audio_override`, else
its own file) to 48 kHz mono, processes the whole file or one range (spliced
back with 10 ms crossfades), and writes a new version
`audio/versions/<n>_<effect>.wav` under the project. The version is pushed on
`SourceFile.audio_history` and becomes `audio_override`, so the compiler, the
previews and the live engine hear it; `undo` pops it. The original file is
never touched and no version is ever overwritten.

A run may also take several source ranges (`ranges`, VOICE_CHANGER_PLAN.md
brief S2a: a programme range across a cut): each is processed exactly like a
single range and all are spliced into ONE version with the same crossfades;
the preview puts them back to back, as the programme plays them.

Runs are jobs in an in-process registry (`JOBS`), one thread each, progress
0..1. `PROCESSES` lists these effects in the EffectDef shape of §2.6
(`engine: "process"`; no parameter is live).

Also registered here: `voice_conversion` (VOICE_CHANGER_PLAN.md brief S1,
`audio/voice_conversion.py`, run by the seed-vc worker) and brief A3's
`audio/processes_ext.py` entries, which `_apply` hands to
`processes_ext.run_process_ext`. An entry whose def says `available: false`
stays registered but is not listed (`listed_processes`) and does not run
(`check_available` raises Conflict with the reason).
"""

import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from audio import deepfilter, noise
from audio.analysis import (
    SAMPLE_RATE,
    AudioToolError,
    astats,
    decode_wav,
    ebur128,
    media_duration,
    range_args,
    run_ffmpeg,
)
from audio.noise import Conflict, NotFound

logger = logging.getLogger("audio.processes")

Progress = Optional[Callable[[float, str], None]]

FADE_S = 0.010          # splice crossfade, centred on each range edge
PAD_S = 0.5             # context processed either side of a range
PREVIEW_MAX_S = 60.0
_BLOCK = 1 << 20        # samples per streamed block when splicing
_RNNOISE_DELAY = 480    # arnndn's output lags its input by one 10 ms frame at 48 kHz


# ----------------------------------------------------------------- catalogue

def _param(key: str, label: str, type_: str, default: Any, min_: Optional[float] = None,
           max_: Optional[float] = None, step: Optional[float] = None, unit: str = "",
           choices: Optional[List[Any]] = None) -> Dict[str, Any]:
    param: Dict[str, Any] = {"key": key, "label": label, "type": type_, "min": min_, "max": max_,
                             "step": step, "default": default, "unit": unit, "live": False}
    if choices is not None:
        param["choices"] = choices
    return param


PROCESSES: Dict[str, Dict[str, Any]] = {
    "noise_reduction": {
        "id": "noise_reduction",
        "name": "Noise Reduction (process)",
        "category": "Noise Reduction / Restoration",
        "engine": "process",
        "description": "Reduces the noise in the captured noise print (Capture Noise Print "
                       "first): noisereduce's stationary spectral gate. Noise Reduction = how "
                       "much of the gated noise goes (prop_decrease), Reduce by = how far above "
                       "the print's level counts as noise (20 dB = 1.5 sigma), Spectral Decay = "
                       "mask smoothing in time and frequency (65 % = 50 ms, 500 Hz).",
        "params": [
            _param("reduction_pct", "Noise Reduction", "number", 100.0, 0.0, 100.0, 1.0, "%"),
            _param("reduce_by_db", "Reduce by", "number", 20.0, 0.0, 60.0, 0.5, "dB"),
            _param("spectral_decay_pct", "Spectral Decay Rate", "number", 65.0, 0.0, 100.0, 1.0, "%"),
            _param("fft_size", "FFT Size", "choice", 1024, choices=[256, 512, 1024, 2048, 4096, 8192]),
            _param("output_noise_only", "Output Noise Only", "bool", False),
        ],
        "presets": {
            "Light Reduction": {"reduction_pct": 70.0, "reduce_by_db": 10.0, "spectral_decay_pct": 65.0},
            "Heavy Reduction": {"reduction_pct": 100.0, "reduce_by_db": 30.0, "spectral_decay_pct": 80.0},
        },
        "windowable": True,
    },
    "noise_reduction_ai": {
        "id": "noise_reduction_ai",
        "name": "Noise Reduction (AI)",
        "category": "Noise Reduction / Restoration",
        "engine": "process",
        "description": "DeepFilterNet 3 speech enhancement (CPU). Attenuation Limit caps how "
                       "far noise is pushed down (100 dB = no limit); Post-filter > 0 adds "
                       "DeepFilterNet's post-filter with that strength (0.02 is its default).",
        "params": [
            _param("attenuation_limit_db", "Attenuation Limit", "number", 100.0, 0.0, 100.0, 1.0, "dB"),
            _param("post_filter", "Post-filter", "number", 0.0, 0.0, 0.05, 0.005, ""),
        ],
        "presets": {
            "Light": {"attenuation_limit_db": 12.0},
            "Medium": {"attenuation_limit_db": 24.0},
            "Full": {"attenuation_limit_db": 100.0},
        },
        "windowable": True,
    },
    "noise_reduction_rnnoise_process": {
        "id": "noise_reduction_rnnoise_process",
        "name": "RNNoise Noise Reduction (process)",
        "category": "Noise Reduction / Restoration",
        "engine": "process",
        "description": "RNNoise (arnndn) rendered into a version. Mix = how much of the "
                       "denoised signal replaces the original.",
        "params": [
            _param("mix", "Mix", "number", 1.0, 0.0, 1.0, 0.01, ""),
        ],
        "presets": {},
        "windowable": True,
    },
    "normalize": {
        "id": "normalize",
        "name": "Normalize (process)",
        "category": "Amplitude and Compression",
        "engine": "process",
        "description": "Two passes: measure (astats; loudnorm for LUFS), then apply. Peak and "
                       "RMS set one gain; LUFS goes through loudnorm (a gain whenever the true "
                       "peak stays under -1 dBTP). DC Bias Adjust removes the measured DC offset.",
        "params": [
            _param("mode", "Normalize To", "choice", "peak", choices=["peak", "rms", "lufs"]),
            _param("peak_dbfs", "Peak", "number", -1.0, -60.0, 0.0, 0.1, "dBFS"),
            _param("rms_dbfs", "RMS", "number", -20.0, -60.0, 0.0, 0.1, "dBFS"),
            _param("lufs", "Loudness", "number", -16.0, -60.0, -5.0, 0.1, "LUFS"),
            _param("dc_remove", "DC Bias Adjust", "bool", False),
        ],
        "presets": {
            "Peak -1 dBFS": {"mode": "peak", "peak_dbfs": -1.0},
            "Peak -0.1 dBFS": {"mode": "peak", "peak_dbfs": -0.1},
            "Podcast -16 LUFS": {"mode": "lufs", "lufs": -16.0},
            "Broadcast -23 LUFS": {"mode": "lufs", "lufs": -23.0},
        },
        "windowable": True,
    },
    "match_loudness": {
        "id": "match_loudness",
        "name": "Match Loudness",
        "category": "Panels",
        "engine": "process",
        "description": "ITU-R BS.1770 loudness to a target with a true-peak ceiling: loudnorm "
                       "two-pass (a plain gain when the ceiling allows it, loudnorm's limiter "
                       "otherwise).",
        "params": [
            _param("target_lufs", "Target Loudness", "number", -14.0, -40.0, -5.0, 0.1, "LUFS"),
            _param("true_peak_db", "True-peak Ceiling", "number", -1.5, -9.0, 0.0, 0.1, "dBTP"),
        ],
        "presets": {
            "YouTube (-14 LUFS)": {"target_lufs": -14.0, "true_peak_db": -1.5},
            "Podcast (-16 LUFS)": {"target_lufs": -16.0, "true_peak_db": -1.0},
            "EBU R128 (-23 LUFS)": {"target_lufs": -23.0, "true_peak_db": -1.0},
            "ATSC A/85 (-24 LKFS)": {"target_lufs": -24.0, "true_peak_db": -2.0},
        },
        "windowable": True,
    },
}


# Voice conversion (VOICE_CHANGER_PLAN.md brief S1): its `voice` choices follow the library.
try:
    from audio import voice_conversion as _voice_conversion
    PROCESSES["voice_conversion"] = _voice_conversion.process_def()
except Exception as _exc:  # noqa: BLE001 -- the lane works without it
    _voice_conversion = None  # type: ignore[assignment]
    logger.warning("voice_conversion process not registered: %s", _exc)

# Brief A3's process effects (Voice Changer, Enhance Speech (AI)), run by their own runner.
try:
    from audio.processes_ext import PROCESSES_EXT as _PROCESSES_EXT
    from audio.processes_ext import run_process_ext as _run_process_ext
except ImportError as _exc:
    _PROCESSES_EXT, _run_process_ext = {}, None
    logger.info("processes_ext not merged: %s", _exc)
for _effect_id, _definition in _PROCESSES_EXT.items():
    PROCESSES.setdefault(_effect_id, _definition)
EXT_IDS = frozenset(_PROCESSES_EXT)

_UNAVAILABLE_HINTS = {
    "voice_changer": "pyworld is not installed in BuzzEdit's environment",
    "enhance_speech_ai": "no speech-enhancement engine is installed (run `python -m audio.enhance_ai` "
                         "from backend/)",
}


def _refresh_dynamic_choices(effect: str) -> None:
    if effect == "voice_conversion" and _voice_conversion is not None and effect in PROCESSES:
        _voice_conversion.refresh_choices(PROCESSES[effect])


def check_available(effect: str) -> None:
    """Conflict (409) with the reason when `effect`'s def says `available: false`."""
    definition = PROCESSES.get(effect)
    if definition is None or definition.get("available", True) is not False:
        return
    reason = definition.get("unavailable_reason") or _UNAVAILABLE_HINTS.get(effect) or \
        "its engine is not installed on this machine"
    if effect == "enhance_speech_ai":
        try:
            from audio import enhance_ai
            error = enhance_ai.status().get("error")
            if error:
                reason = f"{reason}; last check: {error}"
        except Exception:
            pass
    raise Conflict(f"{definition.get('name', effect)} is not available: {reason}")


def listed_processes() -> Dict[str, Dict[str, Any]]:
    """What `GET /processes` lists: every registered process except those whose def says
    `available: false` (they stay registered, so installing the engine lists them again)."""
    for effect in PROCESSES:
        _refresh_dynamic_choices(effect)
    return {key: definition for key, definition in PROCESSES.items()
            if definition.get("available", True) is not False}


def resolve_params(effect: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Every parameter of `effect`: given values coerced and clamped, the rest
    at their defaults. Unknown keys are dropped."""
    definition = PROCESSES.get(effect)
    if definition is None:
        raise ValueError(f"Unknown process effect {effect!r}; one of {sorted(PROCESSES)}")
    _refresh_dynamic_choices(effect)
    given = dict(params or {})
    values: Dict[str, Any] = {}
    for spec in definition["params"]:
        key, kind = spec["key"], spec["type"]
        value = given.get(key, spec["default"])
        if kind == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number, not {value!r}")
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
            values[key] = max(spec["min"], min(spec["max"], number))
        elif kind == "bool":
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            values[key] = bool(value)
        elif kind == "choice":
            choices = spec["choices"]
            if isinstance(choices[0], int) and not isinstance(value, bool):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            if value not in choices:
                raise ValueError(f"{key} must be one of {choices}, not {value!r}")
            values[key] = value
        else:
            values[key] = value
    return values


# ------------------------------------------------------------------- helpers

def _report(progress: Progress, fraction: float, message: str) -> None:
    if progress is not None:
        try:
            progress(max(0.0, min(1.0, fraction)), message)
        except Exception:
            logger.debug("progress callback failed", exc_info=True)


def _scaled(progress: Progress, low: float, high: float) -> Progress:
    if progress is None:
        return None
    return lambda fraction, message: _report(progress, low + (high - low) * fraction, message)


def _fit_length(path: Path, frames: int) -> None:
    """Trim or zero-pad a mono WAV to exactly `frames` samples."""
    info = sf.info(str(path))
    if info.frames == frames:
        return
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    data = data[:frames] if data.size >= frames else np.pad(data, (0, frames - data.size))
    sf.write(str(path), data, rate, subtype="FLOAT")


def _ffmpeg_filter(in_wav: Path, out_wav: Path, filters: List[str]) -> None:
    run_ffmpeg(["-y", "-loglevel", "error", "-i", str(in_wav), "-af", ",".join(filters),
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_f32le", str(out_wav)])


def _seconds(measure: Optional[Tuple[float, float]]) -> Tuple[float, Optional[float]]:
    return (0.0, None) if measure is None else (measure[0], measure[1])


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def _loudnorm(in_wav: Path, out_wav: Path, target_i: float, true_peak: float,
              measure: Optional[Tuple[float, float]], pre: Optional[List[str]] = None) -> Dict[str, Any]:
    """loudnorm two-pass. Pass 1 measures the range; pass 2 applies the gain
    as `volume` when the true peak stays under the ceiling (what loudnorm's
    linear mode does, without its refusal of a 0 LU loudness range), else
    loudnorm itself with the measured values, which limits the peaks. Its
    limiter works at 192 kHz; brought back to 48 kHz the peaks can land up to
    ~1 dB over the ceiling, so that excess is trimmed off with a gain."""
    pre = list(pre or [])
    start, end = _seconds(measure)
    first = run_ffmpeg([*range_args(start, end), "-i", str(in_wav), "-af",
                        ",".join(pre + [f"loudnorm=I={target_i:.2f}:TP={true_peak:.2f}:LRA=50"
                                        f":print_format=json"]), "-f", "null", "-"])
    found = _LOUDNORM_JSON.findall(first.stderr)
    if not found:
        raise AudioToolError("loudnorm printed no measurement")
    measured = json.loads(found[-1])
    input_i = float(measured["input_i"])
    input_tp = float(measured["input_tp"])
    if not math.isfinite(input_i) or input_i <= -70.0:
        raise ValueError("The audio is too quiet to measure its loudness")
    gain = target_i - input_i
    if math.isfinite(input_tp) and input_tp + gain <= true_peak:
        _ffmpeg_filter(in_wav, out_wav, pre + [f"volume={gain:.4f}dB"])
        mode = "linear"
    else:
        _ffmpeg_filter(in_wav, out_wav, pre + [
            f"loudnorm=I={target_i:.2f}:TP={true_peak:.2f}:LRA=50"
            f":measured_I={input_i:.2f}:measured_TP={float(measured['input_tp']):.2f}"
            f":measured_LRA={float(measured['input_lra']):.2f}"
            f":measured_thresh={float(measured['input_thresh']):.2f}"
            f":offset={float(measured['target_offset']):.2f}:linear=true:print_format=none",
            f"aresample={SAMPLE_RATE}"])
        mode = "dynamic"
        peak = ebur128(str(out_wav), start, end)["true_peak_dbtp"]
        if math.isfinite(peak) and peak > true_peak:
            trim = true_peak - peak - 0.1       # ebur128 reads true peak to 0.1 dB
            trimmed = out_wav.with_name(out_wav.stem + ".trim.wav")
            _ffmpeg_filter(out_wav, trimmed, [f"volume={trim:.4f}dB"])
            os.replace(trimmed, out_wav)
            gain += trim
    return {"mode": mode, "input_lufs": round(input_i, 2), "input_true_peak_dbtp": round(input_tp, 2),
            "gain_db": round(gain, 2)}


def _normalize(in_wav: Path, out_wav: Path, values: Dict[str, Any],
               measure: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    start, end = _seconds(measure)
    stats = astats(str(in_wav), start, end)
    dc = float(stats.get("DC offset", 0.0)) if values["dc_remove"] else 0.0
    if not math.isfinite(dc):
        dc = 0.0
    pre = [f"dcshift=shift={-dc:.8f}"] if dc else []
    mode = values["mode"]
    if mode == "lufs":
        info = _loudnorm(in_wav, out_wav, values["lufs"], -1.0, measure, pre)
        info["dc_removed"] = round(dc, 8)
        return info
    if mode == "peak":
        peak = max(abs(float(stats["Max level"]) - dc), abs(float(stats["Min level"]) - dc))
        if not peak > 0.0:
            raise ValueError("The audio is silent: nothing to normalize")
        gain = values["peak_dbfs"] - 20.0 * math.log10(peak)
    else:
        mean_square = 10.0 ** (float(stats["RMS level dB"]) / 10.0) - dc * dc
        if not mean_square > 0.0:
            raise ValueError("The audio is silent: nothing to normalize")
        gain = values["rms_dbfs"] - 10.0 * math.log10(mean_square)
    _ffmpeg_filter(in_wav, out_wav, pre + [f"volume={gain:.4f}dB"])
    return {"mode": mode, "gain_db": round(gain, 2), "dc_removed": round(dc, 8)}


def _apply(effect: str, values: Dict[str, Any], in_wav: Path, out_wav: Path, project_id: str,
           measure: Optional[Tuple[float, float]], progress: Progress) -> Dict[str, Any]:
    """Run one process over a 48 kHz mono WAV. `measure` is the range (seconds
    in `in_wav`) that loudness/level measurements read; None = all of it."""
    if effect == "noise_reduction":
        _json, print_wav = noise.print_paths(project_id)
        if noise.load_print(project_id) is None or not print_wav.exists():
            raise Conflict("Capture a noise print first (Detect noise, then Capture Noise Print)")
        decay = values["spectral_decay_pct"] / 65.0
        noise.noisereduce_process(
            str(in_wav), str(out_wav), str(print_wav),
            prop_decrease=values["reduction_pct"] / 100.0,
            n_std_thresh=0.5 + values["reduce_by_db"] / 20.0,
            freq_smooth_hz=500.0 * decay, time_smooth_ms=50.0 * decay,
            n_fft=int(values["fft_size"]), output_noise_only=values["output_noise_only"])
        return {}
    if effect == "noise_reduction_ai":
        deepfilter.run(str(in_wav), str(out_wav), values["attenuation_limit_db"],
                       values["post_filter"], progress)
        return {}
    if effect == "noise_reduction_rnnoise_process":
        # Pad by arnndn's delay and cut it off the front: output lines up with input.
        _ffmpeg_filter(in_wav, out_wav, [f"apad=pad_len={_RNNOISE_DELAY}",
                                         *noise.rnnoise_chain(values["mix"]),
                                         f"atrim=start_sample={_RNNOISE_DELAY}",
                                         "asetpts=PTS-STARTPTS"])
        return {}
    if effect == "normalize":
        return _normalize(in_wav, out_wav, values, measure)
    if effect == "match_loudness":
        return _loudnorm(in_wav, out_wav, values["target_lufs"], values["true_peak_db"], measure)
    if effect == "voice_conversion" and _voice_conversion is not None:
        return _voice_conversion.process_wav(in_wav, out_wav, values, progress)
    if effect in EXT_IDS and _run_process_ext is not None:
        _run_process_ext(effect, in_wav, out_wav, values, progress)
        return {}
    raise ValueError(f"Unknown process effect {effect!r}")


def _copy_frames(src: sf.SoundFile, dst: sf.SoundFile, start: int, end: int) -> None:
    src.seek(start)
    remaining = end - start
    while remaining > 0:
        block = src.read(min(_BLOCK, remaining), dtype="float32")
        if block.size == 0:
            break
        dst.write(block)
        remaining -= len(block)


def _splice(base_wav: Path, segment_wav: Path, segment_start: int, a: int, b: int,
            out_wav: Path) -> None:
    """`base` with samples [a, b) taken from the processed segment (which
    starts at `segment_start`), joined by linear crossfades of FADE_S centred
    on a and b. A range touching the file's start or end has no fade there."""
    with sf.SoundFile(str(base_wav)) as base, sf.SoundFile(str(segment_wav)) as segment:
        rate, total = base.samplerate, base.frames
        half = max(1, int(round(FADE_S * rate / 2.0)))
        segment_end = segment_start + segment.frames
        low = a if a <= 0 else max(segment_start, a - half)
        high = b if b >= total else min(segment_end, b + half)
        with sf.SoundFile(str(out_wav), "w", samplerate=rate, channels=1, subtype="FLOAT") as out:
            _copy_frames(base, out, 0, low)
            base.seek(low)
            segment.seek(low - segment_start)
            position = low
            while position < high:
                count = min(_BLOCK, high - position)
                dry = base.read(count, dtype="float32")
                wet = segment.read(count, dtype="float32")
                if wet.size < dry.size:
                    wet = np.pad(wet, (0, dry.size - wet.size))
                index = np.arange(position, position + dry.size, dtype=np.float64)
                gain = np.ones(dry.size, dtype=np.float64)
                if a > 0:
                    gain = np.minimum(gain, np.clip((index - (a - half)) / (2.0 * half), 0.0, 1.0))
                if b < total:
                    gain = np.minimum(gain, np.clip(((b + half) - index) / (2.0 * half), 0.0, 1.0))
                out.write((dry * (1.0 - gain) + wet[:dry.size] * gain).astype(np.float32))
                position += dry.size
                if dry.size == 0:
                    break
            _copy_frames(base, out, high, total)


def _splice_in_place(out_wav: Path, segment_wav: Path, segment_start: int, a: int, b: int) -> None:
    """`_splice` into a file being built (a float WAV libsndfile wrote): samples [a, b) from the
    processed segment (which starts at `segment_start`), with the same FADE_S crossfades
    centred on a and b (none at the file's start or end). Only that stretch is read and
    written back, so several ranges cost one copy of the file, not one each."""
    with sf.SoundFile(str(out_wav), "r+") as out, sf.SoundFile(str(segment_wav)) as segment:
        rate, total = out.samplerate, out.frames
        half = max(1, int(round(FADE_S * rate / 2.0)))
        low = a if a <= 0 else max(segment_start, a - half)
        high = b if b >= total else min(segment_start + segment.frames, b + half)
        position = low
        while position < high:
            count = min(_BLOCK, high - position)
            out.seek(position)
            dry = out.read(count, dtype="float32")
            if dry.size == 0:
                break
            segment.seek(position - segment_start)
            wet = segment.read(dry.size, dtype="float32")
            if wet.size < dry.size:
                wet = np.pad(wet, (0, dry.size - wet.size))
            index = np.arange(position, position + dry.size, dtype=np.float64)
            gain = np.ones(dry.size, dtype=np.float64)
            if a > 0:
                gain = np.minimum(gain, np.clip((index - (a - half)) / (2.0 * half), 0.0, 1.0))
            if b < total:
                gain = np.minimum(gain, np.clip(((b + half) - index) / (2.0 * half), 0.0, 1.0))
            out.seek(position)
            out.write((dry * (1.0 - gain) + wet[:dry.size] * gain).astype(np.float32))
            position += dry.size


def _check_ranges(ranges: Any) -> List[Tuple[float, float]]:
    """`ranges` as (start_s, end_s) float pairs, in the order given: each finite with its end
    after its start, and at least one. ValueError otherwise."""
    if isinstance(ranges, (str, bytes, dict)):
        raise ValueError(f"ranges must be a list of (start_s, end_s) pairs, not {ranges!r}")
    try:
        pairs = list(ranges)
    except TypeError:
        raise ValueError(f"ranges must be a list of (start_s, end_s) pairs, not {ranges!r}")
    checked: List[Tuple[float, float]] = []
    for k, pair in enumerate(pairs):
        try:
            start, end = (float(value) for value in pair)
        except (TypeError, ValueError):
            raise ValueError(f"ranges[{k}] must be a (start_s, end_s) pair, not {pair!r}")
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"ranges[{k}] must be finite, not {pair!r}")
        if end <= start:
            raise ValueError(f"Empty range: ranges[{k}] = {start!r}..{end!r} s")
        checked.append((start, end))
    if not checked:
        raise ValueError("ranges is empty: give at least one (start_s, end_s) pair")
    return checked


def _process_ranges(effect: str, values: Dict[str, Any], base: Path, total: int,
                    ranges: List[Tuple[float, float]], work: Path, project_id: str,
                    progress: Progress) -> Tuple[Path, Dict[str, Any]]:
    """Several source ranges of `base` into one file: each processed like a single range
    (PAD_S of context either side, measured on its own range), then spliced into a copy of
    `base` with the lane's crossfades. Returns (that file, {"ranges": [info per range]})."""
    spans = [_range_samples(start, end, total, SAMPLE_RATE) for start, end in ranges]
    result = work / "spliced.wav"
    # written by libsndfile, not copied: ffmpeg's INFO tag would keep it from opening r+
    with sf.SoundFile(str(base)) as src, sf.SoundFile(str(result), "w", samplerate=src.samplerate,
                                                      channels=1, subtype="FLOAT") as dst:
        _copy_frames(src, dst, 0, src.frames)
    pad = int(round(PAD_S * SAMPLE_RATE))
    whole = float(sum(b - a for a, b in spans)) or 1.0
    done = 0
    infos: List[Dict[str, Any]] = []
    for k, (a, b) in enumerate(spans):
        low, high = 0.1 + 0.78 * done / whole, 0.1 + 0.78 * (done + b - a) / whole
        _report(progress, low, f"range {k + 1}/{len(spans)}")
        pa, pb = max(0, a - pad), min(total, b + pad)
        segment, processed = work / f"segment_{k}.wav", work / f"segment_{k}_out.wav"
        with sf.SoundFile(str(base)) as src:
            src.seek(pa)
            sf.write(str(segment), src.read(pb - pa, dtype="float32"), SAMPLE_RATE, subtype="FLOAT")
        measure = ((a - pa) / SAMPLE_RATE, (b - pa) / SAMPLE_RATE)
        infos.append(_apply(effect, values, segment, processed, project_id, measure,
                            _scaled(progress, low, high)))
        _fit_length(processed, pb - pa)
        _splice_in_place(result, processed, pa, a, b)
        done += b - a
    return result, {"ranges": infos}


# ------------------------------------------------------------ project state

_locks: Dict[Tuple[str, str], threading.Lock] = {}
_locks_guard = threading.Lock()


def _source_lock(project_id: str, source_id: str) -> threading.Lock:
    """One process at a time per source: each builds on the version before."""
    with _locks_guard:
        return _locks.setdefault((project_id, source_id), threading.Lock())


def _next_version(versions_dir: Path) -> int:
    numbers = [int(m.group(1)) for p in versions_dir.glob("*.wav")
               if (m := re.match(r"^(\d+)_", p.name))]
    return max(numbers, default=0) + 1


def _timeline_changed(project_id: str) -> None:
    """What routes/audio.py does after a revision bump: drop cached frames."""
    try:
        from render import preview as frame_preview
        frame_preview.invalidate(project_id)
    except Exception:
        logger.debug("preview cache not invalidated", exc_info=True)


def _save_source(project_id: str, source_id: str, update: Callable[[Any], None]) -> Dict[str, Any]:
    """Re-read the project, change one source, bump the revision, save."""
    _pid, p_data, tl = noise.load_project(project_id)
    if source_id not in tl.sources:
        raise NotFound(f"Source {source_id!r} left the timeline while it was processed")
    source = tl.sources[source_id]
    update(source)
    tl.revision += 1
    p_data["timeline"] = tl.model_dump()
    noise.project_store.save_project(project_id, p_data)
    _timeline_changed(project_id)
    return {"source_id": source_id, "audio_override": source.audio_override,
            "history_len": len(source.audio_history), "revision": tl.revision}


def _range_samples(start_s: Optional[float], end_s: Optional[float], total: int,
                   rate: int) -> Tuple[int, int]:
    a = 0 if start_s is None else int(round(max(0.0, float(start_s)) * rate))
    b = total if end_s is None else int(round(max(0.0, float(end_s)) * rate))
    a, b = min(a, total), min(b, total)
    if b <= a:
        raise ValueError(f"Empty range: {start_s!r}..{end_s!r} s on a {total / rate:.3f} s source")
    return a, b


# ---------------------------------------------------------------- the lane

def run_process(project_id: str, source_id: Optional[str], effect: str,
                params: Optional[Dict[str, Any]], start_s: Optional[float] = None,
                end_s: Optional[float] = None, progress: Progress = None,
                ranges: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """Apply a process effect to a source (or one range of it) as a new version.

    `ranges` (source seconds, in place of start_s/end_s): several ranges, each processed like
    one range and all spliced into ONE version; outside them the samples are left as they
    are. A single range takes the one-range path."""
    values = resolve_params(effect, params)
    check_available(effect)
    if ranges is not None:
        ranges = _check_ranges(ranges)
        if len(ranges) == 1:
            (start_s, end_s), ranges = ranges[0], None
        else:
            start_s = end_s = None
    _pid, p_data, tl = noise.load_project(project_id)
    sid, _source = noise.resolve_source(tl, p_data, source_id)
    with _source_lock(project_id, sid):
        # Read again inside the lock: a run that just finished moved the override.
        _pid, p_data, tl = noise.load_project(project_id)
        sid, source = noise.resolve_source(tl, p_data, sid)
        audio_path = noise.source_audio(source)
        folder = noise.audio_dir(project_id)
        versions = folder / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="process_", dir=str(folder)))
        try:
            _report(progress, 0.02, "decoding the source")
            base = decode_wav(audio_path, work / "base.wav", sample_rate=SAMPLE_RATE, mono=True)
            total = sf.info(str(base)).frames
            a, b = _range_samples(start_s, end_s, total, SAMPLE_RATE)
            label = PROCESSES[effect]["name"]
            engine_progress = _scaled(progress, 0.1, 0.85)
            _report(progress, 0.1, label)
            if ranges is not None:
                result, info = _process_ranges(effect, values, base, total, ranges, work, project_id,
                                               progress)
            elif a == 0 and b == total:
                result = work / "processed.wav"
                info = _apply(effect, values, base, result, project_id, None, engine_progress)
                _fit_length(result, total)
            else:
                pad = int(round(PAD_S * SAMPLE_RATE))
                pa, pb = max(0, a - pad), min(total, b + pad)
                with sf.SoundFile(str(base)) as src:
                    src.seek(pa)
                    sf.write(str(work / "segment.wav"), src.read(pb - pa, dtype="float32"),
                             SAMPLE_RATE, subtype="FLOAT")
                processed = work / "segment_out.wav"
                measure = ((a - pa) / SAMPLE_RATE, (b - pa) / SAMPLE_RATE)
                info = _apply(effect, values, work / "segment.wav", processed, project_id,
                              measure, engine_progress)
                _fit_length(processed, pb - pa)
                _report(progress, 0.88, "splicing the range back")
                result = work / "spliced.wav"
                _splice(base, processed, pa, a, b, result)
            number = _next_version(versions)
            final = versions / f"{number}_{effect}.wav"
            os.replace(result, final)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        _report(progress, 0.95, f"saving version {number}")
        entry = {"path": str(final), "effect": effect, "params": values,
                 "start_s": None if start_s is None else float(start_s),
                 "end_s": None if end_s is None else float(end_s),
                 "created": datetime.now().isoformat(timespec="seconds")}
        if ranges is not None:
            entry.update(start_s=min(start for start, _end in ranges),
                         end_s=max(end for _start, end in ranges),
                         ranges=[[start, end] for start, end in ranges])

        def push(src: Any) -> None:
            src.audio_history.append(entry)
            src.audio_override = str(final)

        state = _save_source(project_id, sid, push)
    _report(progress, 1.0, "done")
    return {**state, "effect": effect, "version": entry, "info": info}


def preview_process(project_id: str, source_id: Optional[str], effect: str,
                    params: Optional[Dict[str, Any]], start_s: float, seconds: float,
                    progress: Progress = None,
                    ranges: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """Run a process over `seconds` from `start_s` into `audio/process_preview.wav`
    (48 kHz mono). Nothing is versioned and the project is not changed.

    `ranges` (source seconds, in place of start_s/seconds): each range processed with its
    context and cut to itself, the pieces back to back as the programme's concat plays them,
    PREVIEW_MAX_S in all. A single range takes the one-range path."""
    values = resolve_params(effect, params)
    check_available(effect)
    if ranges is not None:
        ranges = _check_ranges(ranges)
        if len(ranges) == 1:
            start_s, seconds = ranges[0][0], ranges[0][1] - ranges[0][0]
            ranges = None
    _pid, p_data, tl = noise.load_project(project_id)
    sid, source = noise.resolve_source(tl, p_data, source_id)
    audio_path = noise.source_audio(source)
    duration = media_duration(audio_path)
    if ranges is not None:
        return _preview_ranges(project_id, sid, effect, values, audio_path, duration, ranges, progress)
    length = max(1.0, min(PREVIEW_MAX_S, float(seconds or 20.0)))
    start = max(0.0, min(float(start_s or 0.0), duration))
    end = min(duration, start + length)
    if end - start < 0.05:
        raise ValueError(f"Nothing to preview at {start:.3f} s of a {duration:.3f} s source")
    folder = noise.audio_dir(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="preview_", dir=str(folder)))
    try:
        pa, pb = max(0.0, start - PAD_S), min(duration, end + PAD_S)
        segment = decode_wav(audio_path, work / "segment.wav", pa, pb,
                             sample_rate=SAMPLE_RATE, mono=True)
        frames = sf.info(str(segment)).frames
        processed = work / "segment_out.wav"
        info = _apply(effect, values, segment, processed, project_id, (start - pa, end - pa),
                      progress)
        _fit_length(processed, frames)
        first = int(round((start - pa) * SAMPLE_RATE))
        last = min(frames, int(round((end - pa) * SAMPLE_RATE)))
        with sf.SoundFile(str(processed)) as src:
            src.seek(first)
            clip = src.read(max(0, last - first), dtype="float32")
        staged = work / "process_preview.wav"
        sf.write(str(staged), clip, SAMPLE_RATE, subtype="FLOAT")
        target = folder / "process_preview.wav"
        os.replace(staged, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"path": str(target), "seconds": round(len(clip) / SAMPLE_RATE, 3),
            "start_s": round(start, 3), "source_id": sid, "effect": effect, "info": info}


def _preview_ranges(project_id: str, sid: str, effect: str, values: Dict[str, Any], audio_path: str,
                    duration: float, ranges: List[Tuple[float, float]],
                    progress: Progress) -> Dict[str, Any]:
    """`preview_process` over several source ranges: the processed pieces back to back."""
    folder = noise.audio_dir(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="preview_", dir=str(folder)))
    budget = PREVIEW_MAX_S
    clips: List[np.ndarray] = []
    infos: List[Dict[str, Any]] = []
    used: List[List[float]] = []
    try:
        for k, (range_start, range_end) in enumerate(ranges):
            start = max(0.0, min(range_start, duration))
            end = min(duration, range_end, start + budget)
            if end - start < 1.0 / SAMPLE_RATE:
                continue
            pa, pb = max(0.0, start - PAD_S), min(duration, end + PAD_S)
            segment = decode_wav(audio_path, work / f"segment_{k}.wav", pa, pb,
                                 sample_rate=SAMPLE_RATE, mono=True)
            frames = sf.info(str(segment)).frames
            processed = work / f"segment_{k}_out.wav"
            infos.append(_apply(effect, values, segment, processed, project_id, (start - pa, end - pa),
                                _scaled(progress, k / len(ranges), (k + 1) / len(ranges))))
            _fit_length(processed, frames)
            first = int(round((start - pa) * SAMPLE_RATE))
            last = min(frames, int(round((end - pa) * SAMPLE_RATE)))
            with sf.SoundFile(str(processed)) as src:
                src.seek(first)
                clips.append(src.read(max(0, last - first), dtype="float32"))
            used.append([start, end])
            budget -= end - start
            if budget <= 1e-9:
                break
        if not clips:
            raise ValueError(f"Nothing to preview in the ranges {ranges} of a {duration:.3f} s source")
        clip = np.concatenate(clips)
        staged = work / "process_preview.wav"
        sf.write(str(staged), clip, SAMPLE_RATE, subtype="FLOAT")
        target = folder / "process_preview.wav"
        os.replace(staged, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"path": str(target), "seconds": round(len(clip) / SAMPLE_RATE, 3),
            "start_s": round(used[0][0], 3), "source_id": sid, "effect": effect,
            "info": {"ranges": infos}, "ranges": [[round(a, 6), round(b, 6)] for a, b in used]}


def undo(project_id: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    """Pop the newest version; the override goes back to the one before it,
    or None when no version is left. Version files stay on disk."""
    _pid, p_data, tl = noise.load_project(project_id)
    sid, _source = noise.resolve_source(tl, p_data, source_id)
    with _source_lock(project_id, sid):
        _pid, p_data, tl = noise.load_project(project_id)
        source = tl.sources[sid]
        if not source.audio_history and source.audio_override is None:
            return {"source_id": sid, "audio_override": None, "history_len": 0,
                    "revision": tl.revision, "undone": None}
        popped: Dict[str, Any] = {}

        def pop(src: Any) -> None:
            if src.audio_history:
                popped.update(src.audio_history.pop())
            src.audio_override = src.audio_history[-1]["path"] if src.audio_history else None

        state = _save_source(project_id, sid, pop)
    return {**state, "undone": popped or None}


def versions(project_id: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    _pid, p_data, tl = noise.load_project(project_id)
    sid, source = noise.resolve_source(tl, p_data, source_id)
    return {"source_id": sid, "audio_override": source.audio_override,
            "versions": list(source.audio_history)}


# --------------------------------------------------------------------- jobs

JOBS: Dict[str, Dict[str, Any]] = {}
_jobs_guard = threading.Lock()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_guard:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_guard:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _run_job(job_id: str, project_id: str, source_id: Optional[str], effect: str,
             params: Optional[Dict[str, Any]], start_s: Optional[float],
             end_s: Optional[float], ranges: Optional[List[Tuple[float, float]]] = None) -> None:
    def progress(fraction: float, message: str) -> None:
        with _jobs_guard:
            job = JOBS.get(job_id)
            if job is not None:
                job["progress"] = round(max(job["progress"], fraction), 4)
                job["message"] = message

    _update_job(job_id, status="running", message="starting")
    try:
        result = run_process(project_id, source_id, effect, params, start_s, end_s, progress,
                             ranges=ranges)
    except Exception as exc:
        logger.warning("Audio process %s (%s) failed: %s", job_id, effect, exc, exc_info=True)
        _update_job(job_id, status="failed", error=str(exc) or exc.__class__.__name__,
                    message="failed", finished=datetime.now().isoformat(timespec="seconds"))
        return
    _update_job(job_id, status="completed", progress=1.0, message="done", result=result,
                finished=datetime.now().isoformat(timespec="seconds"))


def start_job(project_id: str, source_id: Optional[str], effect: str,
              params: Optional[Dict[str, Any]], start_s: Optional[float] = None,
              end_s: Optional[float] = None,
              ranges: Optional[List[Tuple[float, float]]] = None) -> str:
    """Queue `run_process` on its own thread; poll `get_job(job_id)`. `ranges`: several source
    ranges into one version (see `run_process`)."""
    resolve_params(effect, params)          # refuse a bad request before queueing it
    check_available(effect)
    if ranges is not None:
        ranges = _check_ranges(ranges)
    job_id = f"audio_process_{uuid.uuid4().hex[:12]}"
    with _jobs_guard:
        JOBS[job_id] = {"id": job_id, "job_type": "audio_process", "project_id": project_id,
                        "source_id": source_id, "effect": effect, "status": "pending",
                        "progress": 0.0, "message": "queued", "result": None, "error": None,
                        "created": datetime.now().isoformat(timespec="seconds")}
    threading.Thread(target=_run_job, name=job_id, daemon=True,
                     args=(job_id, project_id, source_id, effect, params, start_s, end_s, ranges)).start()
    return job_id
