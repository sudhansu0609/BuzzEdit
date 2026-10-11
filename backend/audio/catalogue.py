"""The audio effect catalogue -- data, not code (AUDIO_SUITE_PLAN.md §2.2, §2.6, §3).

One entry per effect, keyed by catalogue id, in the menu order of the plan's
§3. Every entry is plain JSON-safe data:

    EffectDef = {id, name (Audition's), category, engine: "ffmpeg"|"process",
                 description, params: [ParamDef], presets: {name: {param: value}},
                 windowable: bool}
    ParamDef  = {key, label, type: "number"|"choice"|"bool"|"curve"|"bands",
                 min, max, step, default, unit, choices?, live: bool}

Extra keys, only where the type needs them: a "curve" param carries its x axis
(`x_min`, `x_max`, `x_unit`, `x_log`; `min`/`max`/`unit` are the y axis) and a
"bands" param carries `fields` -- the ParamDefs of one band.

`live` is derived, never typed in: a parameter is live only when every ffmpeg
option it drives takes a runtime command (the "T" flag in
`ffmpeg -h filter=<name>`, measured on Gyan 8.1 full on 2026-10-09 and copied
into LIVE_OPTIONS below). Parameters that change which filters exist (section
on/off switches) drive no option and are therefore never live -- changing them
restarts the stream. The ffmpeg graph for each entry is built in
`audio/compile.py`, which refuses to register a live command on an option
that is not in LIVE_OPTIONS, so the two cannot drift apart.

Values are in Audition's UI units (dB, Hz, ms, %, semitones); compile.py does
the conversion to ffmpeg's units.
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Tuple

from audio import reverb_ir
from audio.scientific import FILTER_MODES as SCIENTIFIC_MODES
from audio.scientific import FILTER_TYPES as SCIENTIFIC_TYPES

# --- measured ffmpeg facts (ffmpeg 8.1 full_build, gyan.dev, 2026-10-09) ------

_BIQUAD_LIVE = frozenset({"frequency", "f", "width_type", "t", "width", "w", "mix", "m",
                          "channels", "c", "normalize", "n"})

# Options that carry the "T" (runtime command) flag, per filter. Only the
# filters this catalogue uses are listed; an option missing here is not live.
LIVE_OPTIONS: Dict[str, FrozenSet[str]] = {
    "volume": frozenset({"volume"}),
    "acompressor": frozenset({"level_in", "mode", "threshold", "ratio", "attack", "release",
                              "makeup", "knee", "link", "detection", "level_sc", "mix"}),
    "agate": frozenset({"level_in", "mode", "range", "threshold", "ratio", "attack", "release",
                        "makeup", "knee", "detection", "link", "level_sc"}),
    "alimiter": frozenset({"level_in", "level_out", "limit", "attack", "release", "asc",
                           "asc_level", "level", "latency"}),
    "adynamicequalizer": frozenset({"threshold", "dfrequency", "dqfactor", "tfrequency",
                                    "tqfactor", "attack", "release", "ratio", "makeup",
                                    "range", "mode", "dftype", "tftype", "auto"}),
    "equalizer": _BIQUAD_LIVE | {"gain", "g"},
    "lowshelf": _BIQUAD_LIVE | {"gain", "g"},
    "highshelf": _BIQUAD_LIVE | {"gain", "g"},
    "highpass": _BIQUAD_LIVE,
    "lowpass": _BIQUAD_LIVE,
    "bandreject": _BIQUAD_LIVE,
    "biquad": frozenset({"b0", "b1", "b2", "a0", "a1", "a2", "mix", "normalize"}),
    "firequalizer": frozenset({"gain", "gain_entry"}),
    "speechnorm": frozenset({"peak", "p", "expansion", "e", "compression", "c", "threshold",
                             "t", "raise", "r", "fall", "f", "channels", "h", "invert", "i",
                             "link", "l", "rms", "m"}),
    "afftdn": frozenset({"noise_reduction", "nr", "noise_floor", "nf", "residual_floor", "rf",
                         "track_noise", "tn", "track_residual", "tr", "output_mode", "om",
                         "adaptivity", "ad", "floor_offset", "fo", "noise_link", "nl",
                         "sample_noise", "sn", "gain_smooth", "gs"}),
    "arnndn": frozenset({"model", "m", "mix"}),
    "asoftclip": frozenset({"type", "threshold", "output", "param", "oversample"}),
    "acrusher": frozenset({"level_in", "level_out", "bits", "mix", "mode", "dc", "aa",
                           "samples", "lfo", "lforange", "lforate"}),
    "aexciter": frozenset({"level_in", "level_out", "amount", "drive", "blend", "freq",
                           "ceil", "listen"}),
    "stereotools": frozenset({"level_in", "level_out", "balance_in", "balance_out", "softclip",
                              "mutel", "muter", "phasel", "phaser", "mode", "slev", "sbal",
                              "mlev", "mpan", "base", "delay", "sclevel", "phase",
                              "bmode_in", "bmode_out"}),
    "extrastereo": frozenset({"m", "c"}),
    "aphaseshift": frozenset({"shift", "level", "order"}),
    "adelay": frozenset({"delays"}),
    "amix": frozenset({"weights", "normalize"}),
    "rubberband": frozenset({"tempo", "pitch"}),
}

# Filters whose controls never change live (plan §1/§2.3): a change restarts
# the stream. anequalizer's `change` command is deliberately not used.
NOT_LIVE_FILTERS: FrozenSet[str] = frozenset({
    "pan", "anequalizer", "compand", "mcompand", "acrossover", "deesser", "loudnorm",
    "adeclick", "adeclip", "aecho", "chorus", "flanger", "aphaser", "tremolo", "ladspa",
    "asplit",
})

# Filters that support the generic `enable` (timeline) option -- the ones a
# window (`enable='between(t,A,B)'`) and an "enable" bypass can reach.
TIMELINE_FILTERS: FrozenSet[str] = frozenset({
    "acrusher", "adeclick", "adeclip", "adelay", "adynamicequalizer", "aexciter", "afftdn",
    "agate", "alimiter", "anequalizer", "aphaseshift", "arnndn", "asoftclip", "bandreject",
    "dynaudnorm", "equalizer", "extrastereo", "highpass", "highshelf", "lowpass", "lowshelf",
    "speechnorm", "stereotools", "volume", "biquad",
})

# Filters bypassed by `mix 0` rather than `enable 0` (plan §2.3).
MIX_BYPASS_FILTERS: FrozenSet[str] = frozenset({"acompressor", "arnndn"})


def option_is_live(filter_name: str, option: str) -> bool:
    """Whether `option` of `filter_name` takes a runtime command (T flag)."""
    if filter_name in NOT_LIVE_FILTERS:
        return False
    return option in LIVE_OPTIONS.get(filter_name, frozenset())


def drives_live(drives: Iterable[Tuple[str, str]]) -> bool:
    """A parameter is live only if it drives options and every one is live."""
    drives = list(drives)
    return bool(drives) and all(option_is_live(f, o) for f, o in drives)


# --- definition helpers -------------------------------------------------------

Drives = Sequence[Tuple[str, str]]


def _num(key: str, label: str, default: float, lo: float, hi: float, step: float,
         unit: str = "", drives: Drives = ()) -> Dict[str, Any]:
    assert lo <= default <= hi, (key, default)
    return {"key": key, "label": label, "type": "number", "min": lo, "max": hi,
            "step": step, "default": default, "unit": unit, "live": drives_live(drives)}


def _choice(key: str, label: str, default: str, choices: Sequence[str],
            drives: Drives = ()) -> Dict[str, Any]:
    assert default in choices, (key, default)
    return {"key": key, "label": label, "type": "choice", "min": None, "max": None,
            "step": None, "default": default, "unit": "", "choices": list(choices),
            "live": drives_live(drives)}


def _bool(key: str, label: str, default: bool, drives: Drives = ()) -> Dict[str, Any]:
    return {"key": key, "label": label, "type": "bool", "min": None, "max": None,
            "step": None, "default": bool(default), "unit": "", "live": drives_live(drives)}


def _curve(key: str, label: str, default: List[List[float]], *, x_min: float, x_max: float,
           x_unit: str, x_log: bool, y_min: float, y_max: float, y_unit: str,
           drives: Drives = ()) -> Dict[str, Any]:
    return {"key": key, "label": label, "type": "curve", "min": y_min, "max": y_max,
            "step": None, "default": [list(p) for p in default], "unit": y_unit,
            "x_min": x_min, "x_max": x_max, "x_unit": x_unit, "x_log": x_log,
            "live": drives_live(drives)}


def _bands(key: str, label: str, default: List[Dict[str, Any]], fields: List[Dict[str, Any]],
           drives: Drives = ()) -> Dict[str, Any]:
    return {"key": key, "label": label, "type": "bands", "min": None, "max": None,
            "step": None, "default": [dict(b) for b in default], "unit": "",
            "fields": fields, "live": drives_live(drives)}


def _effect(eid: str, name: str, category: str, description: str,
            params: List[Dict[str, Any]], presets: Dict[str, Dict[str, Any]],
            filters_used: Sequence[str]) -> Dict[str, Any]:
    keys = [p["key"] for p in params]
    assert len(keys) == len(set(keys)), eid
    for pname, values in presets.items():
        unknown = set(values) - set(keys)
        assert not unknown, (eid, pname, unknown)
    return {"id": eid, "name": name, "category": category, "engine": "ffmpeg",
            "description": description, "params": params, "presets": presets,
            "windowable": all(f in TIMELINE_FILTERS for f in filters_used)}


# Shorthands for the drives lists below.
def _d(filter_name: str, *options: str) -> List[Tuple[str, str]]:
    return [(filter_name, o) for o in options]


CATEGORIES: List[str] = [
    "Amplitude and Compression",
    "Delay and Echo",
    "Diagnostics",
    "Filter and EQ",
    "Modulation",
    "Noise Reduction / Restoration",
    "Reverb",
    "Special",
    "Stereo Imagery",
    "Time and Pitch",
]

_AC, _DE, _DI, _FE, _MO, _NR, _SP, _ST, _TP = (
    "Amplitude and Compression", "Delay and Echo", "Diagnostics", "Filter and EQ",
    "Modulation", "Noise Reduction / Restoration", "Special", "Stereo Imagery",
    "Time and Pitch")

_DEFS: List[Dict[str, Any]] = []

# --- Amplitude and Compression --------------------------------------------------

_DEFS.append(_effect(
    "amplify", "Amplify", _AC,
    "Boosts or cuts the level by a fixed number of decibels.",
    [_num("gain", "Gain", 0.0, -48.0, 48.0, 0.1, "dB", _d("volume", "volume"))],
    {"+3 dB Boost": {"gain": 3.0}, "+6 dB Boost": {"gain": 6.0},
     "+10 dB Boost": {"gain": 10.0}, "-3 dB Cut": {"gain": -3.0},
     "-6 dB Cut": {"gain": -6.0}, "-10 dB Cut": {"gain": -10.0}},
    ["volume"]))

_DEFS.append(_effect(
    "channel_mixer", "Channel Mixer", _AC,
    "Rebuilds each output channel from a percentage of the left and right inputs.",
    [_num("l_from_l", "Left: Left Input", 100.0, -100.0, 100.0, 1.0, "%", _d("pan", "args")),
     _num("l_from_r", "Left: Right Input", 0.0, -100.0, 100.0, 1.0, "%", _d("pan", "args")),
     _bool("invert_l", "Invert Left", False, _d("pan", "args")),
     _num("r_from_l", "Right: Left Input", 0.0, -100.0, 100.0, 1.0, "%", _d("pan", "args")),
     _num("r_from_r", "Right: Right Input", 100.0, -100.0, 100.0, 1.0, "%", _d("pan", "args")),
     _bool("invert_r", "Invert Right", False, _d("pan", "args"))],
    {"Full Mono": {"l_from_l": 50.0, "l_from_r": 50.0, "r_from_l": 50.0, "r_from_r": 50.0},
     "Swap Channels": {"l_from_l": 0.0, "l_from_r": 100.0, "r_from_l": 100.0, "r_from_r": 0.0},
     "Both = Left": {"l_from_l": 100.0, "l_from_r": 0.0, "r_from_l": 100.0, "r_from_r": 0.0},
     "Both = Right": {"l_from_l": 0.0, "l_from_r": 100.0, "r_from_l": 0.0, "r_from_r": 100.0},
     "Vocal Remove": {"l_from_l": 100.0, "l_from_r": -100.0, "r_from_l": -100.0,
                      "r_from_r": 100.0}},
    ["pan"]))

_ADE = "adynamicequalizer"
_DEFS.append(_effect(
    "deesser", "DeEsser", _AC,
    "Turns down sibilance (s, sh, t) in a band around the centre frequency when it "
    "rises above the threshold.",
    [_choice("mode", "Mode", "Multiband", ["Broadband", "Multiband"],
             _d(_ADE, "tftype", "tfrequency", "tqfactor")),
     _num("threshold", "Threshold", -30.0, -60.0, 0.0, 0.5, "dB", _d(_ADE, "threshold")),
     _num("center", "Center Frequency", 6500.0, 2000.0, 16000.0, 10.0, "Hz",
          _d(_ADE, "dfrequency", "tfrequency")),
     _num("bandwidth", "Bandwidth", 3000.0, 100.0, 10000.0, 10.0, "Hz",
          _d(_ADE, "dqfactor", "tqfactor")),
     _bool("sibilance_only", "Output Sibilance Only", False, _d(_ADE, "mode"))],
    {"Male Voice": {"center": 5500.0, "bandwidth": 3000.0, "threshold": -32.0},
     "Female Voice": {"center": 7500.0, "bandwidth": 4000.0, "threshold": -32.0},
     "Heavy DeEss": {"center": 6500.0, "bandwidth": 5000.0, "threshold": -45.0,
                     "mode": "Broadband"}},
    [_ADE]))

_DEFS.append(_effect(
    "dynamics", "Dynamics", _AC,
    "AutoGate, Compressor, Expander and Limiter in one rack slot; each section has "
    "its own switch.",
    [_bool("gate_on", "AutoGate", False),
     _num("gate_threshold", "Gate Threshold", -50.0, -80.0, 0.0, 0.5, "dB",
          _d("agate", "threshold")),
     _num("gate_attack", "Gate Attack", 2.0, 0.1, 100.0, 0.1, "ms", _d("agate", "attack")),
     _num("gate_release", "Gate Release", 100.0, 1.0, 2000.0, 1.0, "ms",
          _d("agate", "release")),
     _bool("comp_on", "Compressor", True),
     _num("comp_threshold", "Compressor Threshold", -20.0, -60.0, 0.0, 0.5, "dB",
          _d("acompressor", "threshold")),
     _num("comp_ratio", "Compressor Ratio", 4.0, 1.0, 20.0, 0.1, ":1",
          _d("acompressor", "ratio")),
     _num("comp_attack", "Compressor Attack", 10.0, 0.01, 500.0, 0.1, "ms",
          _d("acompressor", "attack")),
     _num("comp_release", "Compressor Release", 100.0, 1.0, 5000.0, 1.0, "ms",
          _d("acompressor", "release")),
     _num("comp_makeup", "Make-Up Gain", 0.0, 0.0, 30.0, 0.1, "dB",
          _d("acompressor", "makeup")),
     _bool("exp_on", "Expander", False),
     _num("exp_threshold", "Expander Threshold", -40.0, -80.0, 0.0, 0.5, "dB",
          _d("compand", "points")),
     _num("exp_ratio", "Expander Ratio", 2.0, 1.0, 10.0, 0.1, ":1", _d("compand", "points")),
     _bool("lim_on", "Limiter", True),
     _num("lim_threshold", "Limiter Threshold", -1.0, -24.0, 0.0, 0.1, "dB",
          _d("alimiter", "limit")),
     _num("lim_attack", "Limiter Attack", 5.0, 0.1, 80.0, 0.1, "ms", _d("alimiter", "attack")),
     _num("lim_release", "Limiter Release", 50.0, 1.0, 2000.0, 1.0, "ms",
          _d("alimiter", "release"))],
    {"Podcast Voice": {"gate_on": True, "gate_threshold": -50.0, "comp_threshold": -20.0,
                       "comp_ratio": 4.0, "comp_makeup": 4.0, "lim_threshold": -1.0},
     "Gentle Compression": {"comp_threshold": -18.0, "comp_ratio": 2.0, "comp_attack": 20.0,
                            "comp_release": 200.0, "comp_makeup": 2.0},
     "Broadcast": {"comp_threshold": -24.0, "comp_ratio": 3.0, "comp_makeup": 6.0,
                   "lim_threshold": -1.0, "lim_release": 80.0},
     "Noise Gate": {"gate_on": True, "gate_threshold": -45.0, "comp_on": False,
                    "lim_on": False},
     "Expand Quiet Passages": {"exp_on": True, "exp_threshold": -45.0, "exp_ratio": 2.0,
                               "comp_on": False}},
    ["agate", "acompressor", "compand", "alimiter"]))

_DEFS.append(_effect(
    "dynamics_processing", "Dynamics Processing", _AC,
    "Draw the level transfer curve (input dB to output dB) to compress, expand, gate "
    "or limit.",
    [_curve("curve", "Transfer Curve", [[-100.0, -100.0], [0.0, 0.0]], x_min=-100.0,
            x_max=0.0, x_unit="dB", x_log=False, y_min=-100.0, y_max=0.0, y_unit="dB",
            drives=_d("compand", "points")),
     _num("attack", "Attack Time", 10.0, 0.0, 1000.0, 0.1, "ms", _d("compand", "attacks")),
     _num("release", "Release Time", 100.0, 0.0, 5000.0, 1.0, "ms", _d("compand", "decays")),
     _num("lookahead", "Look-Ahead Time", 0.0, 0.0, 200.0, 0.1, "ms",
          _d("compand", "delay"))],
    {"2:1 Compression above -20 dB": {"curve": [[-100.0, -100.0], [-20.0, -20.0],
                                                [0.0, -10.0]]},
     "4:1 Compression above -15 dB": {"curve": [[-100.0, -100.0], [-15.0, -15.0],
                                                [0.0, -11.25]]},
     "2:1 Expander below -40 dB": {"curve": [[-100.0, -100.0], [-70.0, -100.0],
                                             [-40.0, -40.0], [0.0, 0.0]]},
     "Noise Gate below -50 dB": {"curve": [[-100.0, -100.0], [-51.0, -100.0],
                                           [-50.0, -50.0], [0.0, 0.0]],
                                 "attack": 1.0, "release": 150.0},
     "Hard Limit at -3 dB": {"curve": [[-100.0, -100.0], [-3.0, -3.0], [0.0, -3.0]],
                             "attack": 1.0, "release": 50.0}},
    ["compand"]))

_DEFS.append(_effect(
    "hard_limiter", "Hard Limiter", _AC,
    "Keeps peaks under the maximum amplitude; Input Boost raises the level into it.",
    [_num("max_amplitude", "Maximum Amplitude", -0.1, -24.0, 0.0, 0.1, "dB",
          _d("alimiter", "limit")),
     _num("input_boost", "Input Boost", 0.0, 0.0, 36.0, 0.1, "dB", _d("alimiter", "level_in")),
     _num("lookahead", "Look-Ahead Time", 7.0, 0.1, 80.0, 0.1, "ms", _d("alimiter", "attack")),
     _num("release", "Release Time", 100.0, 1.0, 2000.0, 1.0, "ms",
          _d("alimiter", "release"))],
    {"Limit to -0.1 dB": {"max_amplitude": -0.1},
     "Limit to -3 dB": {"max_amplitude": -3.0},
     "Limit to -6 dB": {"max_amplitude": -6.0},
     "+3 dB Boost, Limit to -0.1 dB": {"max_amplitude": -0.1, "input_boost": 3.0},
     "+6 dB Boost, Limit to -0.1 dB": {"max_amplitude": -0.1, "input_boost": 6.0},
     "Broadcast Limiter": {"max_amplitude": -1.0, "input_boost": 3.0, "lookahead": 5.0,
                           "release": 80.0}},
    ["alimiter"]))

_BAND_FIELDS = [
    _num("threshold", "Threshold", -18.0, -60.0, 0.0, 0.5, "dB"),
    _num("gain", "Gain", 0.0, -18.0, 18.0, 0.1, "dB"),
    _num("ratio", "Ratio", 3.0, 1.0, 20.0, 0.1, ":1"),
    _num("attack", "Attack", 10.0, 0.01, 500.0, 0.1, "ms"),
    _num("release", "Release", 100.0, 1.0, 5000.0, 1.0, "ms"),
]


def _mb_bands(*rows: Tuple[float, float, float, float, float]) -> List[Dict[str, float]]:
    return [{"threshold": t, "gain": g, "ratio": r, "attack": a, "release": rel}
            for t, g, r, a, rel in rows]


_DEFS.append(_effect(
    "multiband_compressor", "Multiband Compressor", _AC,
    "Splits the sound into four bands at the crossovers, compresses each on its own, "
    "then limits the sum.",
    [_num("crossover_low", "Low Crossover", 120.0, 20.0, 1000.0, 1.0, "Hz",
          _d("acrossover", "split")),
     _num("crossover_mid", "Mid Crossover", 1000.0, 200.0, 8000.0, 1.0, "Hz",
          _d("acrossover", "split")),
     _num("crossover_high", "High Crossover", 8000.0, 1000.0, 18000.0, 1.0, "Hz",
          _d("acrossover", "split")),
     _bands("bands", "Bands", _mb_bands(*[(-18.0, 0.0, 3.0, 10.0, 100.0)] * 4), _BAND_FIELDS,
            _d("acompressor", "threshold", "ratio", "attack", "release")
            + _d("amix", "weights")),
     _num("output_gain", "Output Gain", 0.0, -18.0, 18.0, 0.1, "dB",
          _d("alimiter", "level_in")),
     _num("limiter_threshold", "Limiter Threshold", -0.1, -24.0, 0.0, 0.1, "dB",
          _d("alimiter", "limit"))],
    {"Broadcast": {"bands": _mb_bands((-24.0, 2.0, 3.0, 10.0, 150.0),
                                      (-24.0, 2.0, 3.0, 8.0, 120.0),
                                      (-22.0, 2.0, 3.0, 5.0, 100.0),
                                      (-20.0, 1.0, 2.5, 3.0, 80.0)),
                   "output_gain": 2.0, "limiter_threshold": -1.0},
     "Spoken Word": {"bands": _mb_bands((-30.0, -2.0, 4.0, 20.0, 200.0),
                                        (-22.0, 1.0, 2.5, 10.0, 150.0),
                                        (-20.0, 2.0, 2.0, 5.0, 100.0),
                                        (-26.0, 0.0, 3.0, 2.0, 60.0)),
                     "limiter_threshold": -1.0},
     "Pop Master": {"bands": _mb_bands((-20.0, 1.5, 3.0, 15.0, 150.0),
                                       (-18.0, 0.5, 2.0, 10.0, 120.0),
                                       (-18.0, 0.5, 2.0, 5.0, 100.0),
                                       (-20.0, 1.5, 2.5, 3.0, 80.0)),
                    "output_gain": 1.5, "limiter_threshold": -0.3},
     "Enhance Lows": {"bands": _mb_bands((-18.0, 4.0, 2.0, 15.0, 150.0),
                                         *[(-18.0, 0.0, 3.0, 10.0, 100.0)] * 3)},
     "Enhance Highs": {"bands": _mb_bands(*[(-18.0, 0.0, 3.0, 10.0, 100.0)] * 3,
                                          (-18.0, 4.0, 2.0, 3.0, 80.0))}},
    ["acrossover", "acompressor", "amix", "alimiter"]))

_DEFS.append(_effect(
    "single_band_compressor", "Single-band Compressor", _AC,
    "Classic compressor: reduces the level above the threshold by the ratio.",
    [_num("threshold", "Threshold", -20.0, -60.0, 0.0, 0.5, "dB",
          _d("acompressor", "threshold")),
     _num("ratio", "Ratio", 4.0, 1.0, 20.0, 0.1, ":1", _d("acompressor", "ratio")),
     _num("attack", "Attack", 10.0, 0.01, 500.0, 0.1, "ms", _d("acompressor", "attack")),
     _num("release", "Release", 100.0, 1.0, 5000.0, 1.0, "ms", _d("acompressor", "release")),
     _num("output_gain", "Output Gain", 0.0, -30.0, 30.0, 0.1, "dB", _d("volume", "volume"))],
    {"Light Vocal Compression": {"threshold": -18.0, "ratio": 2.0, "attack": 10.0,
                                 "release": 120.0, "output_gain": 2.0},
     "Heavy Vocal Compression": {"threshold": -28.0, "ratio": 6.0, "attack": 5.0,
                                 "release": 80.0, "output_gain": 6.0},
     "Voice Leveler": {"threshold": -24.0, "ratio": 3.0, "attack": 20.0, "release": 300.0,
                       "output_gain": 4.0},
     "Hard Limiting": {"threshold": -6.0, "ratio": 20.0, "attack": 0.1, "release": 50.0}},
    ["acompressor", "volume"]))

_DEFS.append(_effect(
    "speech_volume_leveler", "Speech Volume Leveler", _AC,
    "Evens out the level of speech toward a target, optionally raising quiet words and "
    "gating the noise between them.",
    [_num("amount", "Leveling Amount", 50.0, 0.0, 100.0, 1.0, "%",
          _d("speechnorm", "expansion", "compression")),
     _num("target", "Target Volume Level", -20.0, -40.0, 0.0, 0.5, "dB",
          _d("speechnorm", "rms")),
     _bool("boost_low", "Boost Low Signals", True, _d("speechnorm", "expansion")),
     _bool("noise_gate", "Noise Gate", False),
     _num("gate_threshold", "Noise Gate Threshold", -50.0, -80.0, -20.0, 0.5, "dB",
          _d("agate", "threshold"))],
    {"Light Leveling": {"amount": 30.0},
     "Strong Leveling": {"amount": 80.0, "target": -18.0},
     "Leveling with Noise Gate": {"amount": 50.0, "noise_gate": True,
                                  "gate_threshold": -50.0}},
    ["speechnorm", "agate"]))

_DEFS.append(_effect(
    "tube_compressor", "Tube-modeled Compressor", _AC,
    "A compressor followed by soft tube-style saturation and a touch of harmonic "
    "excitement.",
    [_num("threshold", "Threshold", -18.0, -60.0, 0.0, 0.5, "dB",
          _d("acompressor", "threshold")),
     _num("ratio", "Ratio", 3.0, 1.0, 20.0, 0.1, ":1", _d("acompressor", "ratio")),
     _num("attack", "Attack", 10.0, 0.01, 500.0, 0.1, "ms", _d("acompressor", "attack")),
     _num("release", "Release", 150.0, 1.0, 5000.0, 1.0, "ms", _d("acompressor", "release")),
     _num("output_gain", "Output Gain", 0.0, -30.0, 20.0, 0.1, "dB",
          _d("asoftclip", "output"))],
    {"Warm Vocal": {"threshold": -20.0, "ratio": 3.0, "attack": 10.0, "release": 150.0,
                    "output_gain": 3.0},
     "Gentle Glue": {"threshold": -15.0, "ratio": 2.0, "attack": 20.0, "release": 200.0,
                     "output_gain": 1.0},
     "Heavy Tube": {"threshold": -30.0, "ratio": 6.0, "attack": 5.0, "release": 100.0,
                    "output_gain": 6.0}},
    ["acompressor", "asoftclip", "aexciter"]))

# --- Delay and Echo ---------------------------------------------------------------

_DEFS.append(_effect(
    "delay", "Delay", _DE,
    "Mixes a delayed copy of each channel with the dry sound.",
    [_num("left_delay", "Left Delay Time", 250.0, 0.0, 2000.0, 1.0, "ms",
          _d("adelay", "delays")),
     _num("right_delay", "Right Delay Time", 250.0, 0.0, 2000.0, 1.0, "ms",
          _d("adelay", "delays")),
     _num("mix", "Mix", 30.0, 0.0, 100.0, 1.0, "%", _d("amix", "weights"))],
    {"Slapback": {"left_delay": 90.0, "right_delay": 90.0, "mix": 35.0},
     "Doubler": {"left_delay": 18.0, "right_delay": 28.0, "mix": 40.0},
     "Stereo Spread": {"left_delay": 0.0, "right_delay": 20.0, "mix": 50.0},
     "Quarter Note at 120 BPM": {"left_delay": 500.0, "right_delay": 500.0, "mix": 30.0}},
    ["asplit", "adelay", "amix"]))

_DEFS.append(_effect(
    "echo", "Echo", _DE,
    "Repeating echoes that fade by the feedback amount.",
    [_num("delay", "Delay Time", 300.0, 1.0, 2000.0, 1.0, "ms", _d("aecho", "delays")),
     _num("feedback", "Feedback", 40.0, 0.0, 95.0, 1.0, "%", _d("aecho", "decays")),
     _num("level", "Echo Level", 50.0, 1.0, 100.0, 1.0, "%", _d("aecho", "decays"))],
    {"Short Echo": {"delay": 120.0, "feedback": 30.0, "level": 40.0},
     "Long Echo": {"delay": 600.0, "feedback": 50.0, "level": 50.0},
     "Canyon": {"delay": 900.0, "feedback": 60.0, "level": 60.0},
     "Spooky": {"delay": 450.0, "feedback": 70.0, "level": 45.0}},
    ["aecho"]))

_DEFS.append(_effect(
    "analog_delay", "Analog Delay", _DE,
    "A darker, tape- or tube-coloured echo.",
    [_choice("mode", "Mode", "Tape", ["Tape", "Tape/Tube", "Analog"],
             _d("lowpass", "frequency")),
     _num("dry", "Dry Out", 80.0, 0.0, 100.0, 1.0, "%", _d("aecho", "in_gain")),
     _num("wet", "Wet Out", 50.0, 0.0, 100.0, 1.0, "%", _d("aecho", "decays")),
     _num("delay", "Delay", 200.0, 1.0, 2000.0, 1.0, "ms", _d("aecho", "delays")),
     _num("feedback", "Feedback", 40.0, 0.0, 95.0, 1.0, "%", _d("aecho", "decays"))],
    {"Tape Slap": {"mode": "Tape", "dry": 80.0, "wet": 40.0, "delay": 120.0,
                   "feedback": 20.0},
     "Warm Tube Echo": {"mode": "Tape/Tube", "dry": 80.0, "wet": 50.0, "delay": 350.0,
                        "feedback": 45.0},
     "Dark Analog": {"mode": "Analog", "dry": 75.0, "wet": 55.0, "delay": 450.0,
                     "feedback": 55.0}},
    ["aecho", "lowpass"]))

# --- Diagnostics --------------------------------------------------------------------

_DEFS.append(_effect(
    "declipper", "DeClipper", _DI,
    "Rebuilds clipped peaks from the surrounding waveform.",
    [_num("gain", "Gain", 0.0, -24.0, 0.0, 0.1, "dB", _d("volume", "volume")),
     _num("threshold", "Threshold", 10.0, 1.0, 100.0, 1.0, "", _d("adeclip", "threshold")),
     _num("window", "Window Size", 55.0, 10.0, 100.0, 1.0, "ms", _d("adeclip", "window")),
     _num("overlap", "Overlap", 75.0, 50.0, 95.0, 1.0, "%", _d("adeclip", "overlap")),
     _num("arorder", "Autoregression Order", 8.0, 0.0, 25.0, 1.0, "%",
          _d("adeclip", "arorder")),
     _num("hsize", "Histogram Size", 1000.0, 100.0, 9999.0, 1.0, "", _d("adeclip", "hsize"))],
    {"Restore Light Clipping": {"threshold": 20.0},
     "Restore Heavy Clipping": {"gain": -3.0, "threshold": 5.0, "hsize": 2000.0}},
    ["volume", "adeclip"]))

_DEFS.append(_effect(
    "dehummer", "DeHummer", _DI,
    "Notches out mains hum and its harmonics.",
    [_num("frequency", "Frequency", 50.0, 20.0, 500.0, 0.1, "Hz",
          _d("anequalizer", "params")),
     _num("q", "Q", 30.0, 1.0, 100.0, 0.5, "", _d("anequalizer", "params")),
     _num("gain", "Gain", -30.0, -60.0, 0.0, 0.5, "dB", _d("anequalizer", "params")),
     _num("harmonics", "Number of Harmonics", 5.0, 1.0, 10.0, 1.0, "",
          _d("anequalizer", "params"))],
    {"50 Hz Hum Removal": {"frequency": 50.0},
     "60 Hz Hum Removal": {"frequency": 60.0},
     "Heavy 50 Hz Hum Removal": {"frequency": 50.0, "gain": -50.0, "harmonics": 8.0}},
    ["anequalizer"]))

_DEFS.append(_effect(
    "declicker", "DeClicker", _DI,
    "Finds clicks and pops and interpolates over them.",
    [_num("threshold", "Threshold", 2.0, 1.0, 100.0, 0.5, "", _d("adeclick", "threshold")),
     _num("complexity", "Complexity", 10.0, 0.0, 100.0, 1.0, "%", _d("adeclick", "arorder")),
     _num("window", "Window Size", 55.0, 10.0, 100.0, 1.0, "ms", _d("adeclick", "window")),
     _num("burst", "Burst Fusion", 2.0, 0.0, 10.0, 0.5, "", _d("adeclick", "burst"))],
    {"Light Reduction": {"threshold": 6.0, "burst": 1.0, "complexity": 8.0},
     "Heavy Reduction": {"threshold": 1.0, "burst": 4.0, "complexity": 20.0}},
    ["adeclick"]))

# --- Filter and EQ ------------------------------------------------------------------

_DEFS.append(_effect(
    "fft_filter", "FFT Filter", _FE,
    "Draw any frequency response; the curve is applied as a minimum-phase FIR filter, "
    "so the sound gets no delay against the picture.",
    [_curve("curve", "Curve", [[20.0, 0.0], [20000.0, 0.0]], x_min=20.0, x_max=20000.0,
            x_unit="Hz", x_log=True, y_min=-60.0, y_max=20.0, y_unit="dB",
            drives=_d("firequalizer", "gain_entry")),
     # off: straight lines between the points (no overshoot on steep curves)
     _bool("spline", "Spline Curves", False, _d("firequalizer", "gain"))],
    {"Kill the Mic Rumble": {"curve": [[20.0, -40.0], [60.0, -20.0], [90.0, 0.0],
                                       [20000.0, 0.0]]},
     "Kill the Subsonic": {"curve": [[20.0, -40.0], [30.0, -12.0], [40.0, 0.0],
                                     [20000.0, 0.0]]},
     "Telephone": {"curve": [[20.0, -60.0], [250.0, -60.0], [300.0, 0.0], [3400.0, 0.0],
                             [3800.0, -60.0], [24000.0, -60.0]]},
     "Old Time Radio": {"curve": [[20.0, -40.0], [150.0, -18.0], [300.0, 0.0],
                                  [4000.0, 0.0], [5500.0, -20.0], [20000.0, -40.0]]}},
    ["firequalizer"]))

GRAPHIC_EQ_CENTRES: Dict[str, List[float]] = {
    # ISO 1-octave
    "graphic_eq_10": [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    # 1/2-octave, ending at 20 kHz
    "graphic_eq_20": [31.5, 44, 63, 88, 125, 177, 250, 355, 500, 710, 1000, 1400, 2000,
                      2800, 4000, 5600, 8000, 11200, 16000, 20000],
    # ISO 1/3-octave
    "graphic_eq_30": [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                      800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                      12500, 16000, 20000],
}

# Q of each band's peaking filter, by band spacing: octave, half-octave, third-octave.
GRAPHIC_EQ_Q: Dict[str, float] = {"graphic_eq_10": 1.41, "graphic_eq_20": 2.87,
                                  "graphic_eq_30": 4.32}


def _hz_label(f: float) -> str:
    if f >= 1000:
        k = f / 1000.0
        return f"{k:g} kHz"
    return f"{f:g} Hz"


def _octaves(f: float, ref: float) -> float:
    return math.log2(f / ref)


def _ramp(f: float, a: float, b: float) -> float:
    """0 at or below a, 1 at or above b, linear in log-frequency between."""
    if f <= a:
        return 0.0
    if f >= b:
        return 1.0
    return _octaves(f, a) / _octaves(b, a)


_GEQ_SHAPES = {
    "Bass Boost": lambda f: 6.0 * (1.0 - _ramp(f, 100.0, 400.0)),
    "Treble Boost": lambda f: 5.0 * _ramp(f, 2000.0, 6000.0),
    "Vocal Presence": lambda f: (3.0 * max(0.0, 1.0 - abs(_octaves(f, 3500.0))))
                                - 2.0 * (1.0 - _ramp(f, 60.0, 120.0)),
    "Telephone": lambda f: -24.0 * ((1.0 - _ramp(f, 150.0, 300.0))
                                    + _ramp(f, 3400.0, 6800.0)),
    "Loudness": lambda f: 4.0 * (1.0 - _ramp(f, 60.0, 250.0)) + 3.0 * _ramp(f, 4000.0, 10000.0),
}


def _graphic_eq(eid: str, nbands: int) -> Dict[str, Any]:
    centres = GRAPHIC_EQ_CENTRES[eid]
    assert len(centres) == nbands, eid
    eq_gain = _d("equalizer", "g")
    params = [_num(f"band_{i + 1}", _hz_label(c), 0.0, -60.0, 60.0, 0.1, "dB", eq_gain)
              for i, c in enumerate(centres)]
    params += [
        _num("range", "Range", 48.0, 1.5, 120.0, 0.5, "dB", eq_gain),
        _num("master_gain", "Master Gain", 0.0, -24.0, 24.0, 0.1, "dB", _d("volume", "volume")),
    ]
    presets = {name: {f"band_{i + 1}": round(shape(c) * 2.0) / 2.0
                      for i, c in enumerate(centres)}
               for name, shape in _GEQ_SHAPES.items()}
    return _effect(
        eid, f"Graphic Equalizer ({nbands} Bands)", _FE,
        f"{nbands} fixed bands at ISO centre frequencies. Range sets how far each "
        "slider can go (half up, half down).",
        params, presets, ["equalizer", "volume"])


_DEFS.append(_graphic_eq("graphic_eq_10", 10))
_DEFS.append(_graphic_eq("graphic_eq_20", 20))
_DEFS.append(_graphic_eq("graphic_eq_30", 30))

NOTCH_COUNT = 6
NOTCH_WIDTH_Q = {"Super Narrow": 60.0, "Narrow": 30.0, "Medium": 10.0, "Wide": 3.0}
_NOTCH_DEFAULT_HZ = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]


def _notch_params() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(NOTCH_COUNT):
        n = i + 1
        out += [
            _bool(f"notch{n}_on", f"Notch {n}", n == 1),
            _num(f"notch{n}_frequency", f"Notch {n} Frequency", _NOTCH_DEFAULT_HZ[i], 20.0,
                 20000.0, 0.1, "Hz", _d("bandreject", "frequency")),
            _num(f"notch{n}_gain", f"Notch {n} Gain", -30.0, -60.0, 0.0, 0.5, "dB",
                 _d("bandreject", "mix")),
        ]
    out.append(_choice("width", "Notch Width", "Narrow", list(NOTCH_WIDTH_Q),
                       _d("bandreject", "width")))
    return out


def _notch_preset(freqs: Sequence[float], gain: float, width: str) -> Dict[str, Any]:
    p: Dict[str, Any] = {"width": width}
    for i in range(NOTCH_COUNT):
        n = i + 1
        on = i < len(freqs)
        p[f"notch{n}_on"] = on
        if on:
            p[f"notch{n}_frequency"] = float(freqs[i])
            p[f"notch{n}_gain"] = gain
    return p


_DEFS.append(_effect(
    "notch_filter", "Notch Filter", _FE,
    "Up to six narrow notches, each with its own frequency and depth.",
    _notch_params(),
    {"DTMF": _notch_preset([697, 770, 852, 941, 1209, 1336], -40.0, "Narrow"),
     "50 Hz Hum": _notch_preset([50, 100, 150, 200, 250, 300], -30.0, "Super Narrow"),
     "60 Hz Hum": _notch_preset([60, 120, 180, 240, 300, 360], -30.0, "Super Narrow")},
    ["bandreject"]))

PEQ_BANDS = 5
_PEQ_DEFAULT_HZ = [125.0, 400.0, 1000.0, 3000.0, 8000.0]


def _peq_params() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [
        _bool("hp_on", "High-Pass", False),
        _num("hp_frequency", "High-Pass Frequency", 80.0, 10.0, 2000.0, 1.0, "Hz",
             _d("highpass", "frequency")),
        _num("hp_q", "High-Pass Q", 0.707, 0.1, 10.0, 0.01, "", _d("highpass", "width")),
        _bool("ls_on", "Low Shelf", True),
        _num("ls_frequency", "Low Shelf Frequency", 100.0, 20.0, 1000.0, 1.0, "Hz",
             _d("lowshelf", "frequency")),
        _num("ls_gain", "Low Shelf Gain", 0.0, -30.0, 30.0, 0.1, "dB", _d("lowshelf", "gain")),
        _num("ls_q", "Low Shelf Q", 0.707, 0.1, 10.0, 0.01, "", _d("lowshelf", "width")),
    ]
    for i in range(PEQ_BANDS):
        n = i + 1
        out += [
            _bool(f"band{n}_on", f"Band {n}", True),
            _num(f"band{n}_frequency", f"Band {n} Frequency", _PEQ_DEFAULT_HZ[i], 20.0,
                 20000.0, 1.0, "Hz", _d("equalizer", "frequency")),
            _num(f"band{n}_gain", f"Band {n} Gain", 0.0, -30.0, 30.0, 0.1, "dB",
                 _d("equalizer", "gain")),
            _num(f"band{n}_q", f"Band {n} Q", 1.0, 0.1, 100.0, 0.01, "",
                 _d("equalizer", "width")),
        ]
    out += [
        _bool("hs_on", "High Shelf", True),
        _num("hs_frequency", "High Shelf Frequency", 10000.0, 1000.0, 20000.0, 1.0, "Hz",
             _d("highshelf", "frequency")),
        _num("hs_gain", "High Shelf Gain", 0.0, -30.0, 30.0, 0.1, "dB",
             _d("highshelf", "gain")),
        _num("hs_q", "High Shelf Q", 0.707, 0.1, 10.0, 0.01, "", _d("highshelf", "width")),
        _bool("lp_on", "Low-Pass", False),
        _num("lp_frequency", "Low-Pass Frequency", 18000.0, 1000.0, 22000.0, 1.0, "Hz",
             _d("lowpass", "frequency")),
        _num("lp_q", "Low-Pass Q", 0.707, 0.1, 10.0, 0.01, "", _d("lowpass", "width")),
        _num("master_gain", "Master Gain", 0.0, -30.0, 30.0, 0.1, "dB", _d("volume", "volume")),
    ]
    return out


_DEFS.append(_effect(
    "parametric_eq", "Parametric Equalizer", _FE,
    "Five peaking bands plus high-pass, low-pass and two shelves, each with its own "
    "switch, and a master gain.",
    _peq_params(),
    {"Vocal Presence": {"band4_frequency": 3000.0, "band4_gain": 3.0, "band4_q": 1.0,
                        "ls_gain": -2.0},
     "Warm Voice": {"band1_frequency": 200.0, "band1_gain": 2.5, "band1_q": 0.8,
                    "hs_gain": -2.0},
     "Remove Rumble": {"hp_on": True, "hp_frequency": 80.0},
     "Telephone": {"hp_on": True, "hp_frequency": 400.0, "lp_on": True,
                   "lp_frequency": 3400.0, "band3_frequency": 1500.0, "band3_gain": 4.0},
     "Bass Boost": {"ls_frequency": 120.0, "ls_gain": 6.0},
     # Speech curves the Essential Sound card lists first (brief A5).
     "Podcast Voice": {"hp_on": True, "hp_frequency": 80.0,
                       "band2_frequency": 250.0, "band2_gain": -2.0, "band2_q": 1.0,
                       "band4_frequency": 3000.0, "band4_gain": 2.0, "band4_q": 1.0,
                       "hs_frequency": 8000.0, "hs_gain": 1.5},
     "Broadcast": {"hp_on": True, "hp_frequency": 90.0,
                   "band1_frequency": 180.0, "band1_gain": 1.0, "band1_q": 1.0,
                   "band4_frequency": 4000.0, "band4_gain": 3.0, "band4_q": 0.9,
                   "lp_on": True, "lp_frequency": 15000.0},
     "Warm": {"hp_on": True, "hp_frequency": 60.0, "ls_frequency": 150.0, "ls_gain": 2.0,
              "band4_frequency": 4000.0, "band4_gain": -1.0},
     "Clean": {"hp_on": True, "hp_frequency": 100.0,
               "band2_frequency": 300.0, "band2_gain": -1.5,
               "band4_frequency": 5000.0, "band4_gain": 1.0},
     # The Inspector's "Crisp & bassy" voice curve (render/audio.EQ_PRESETS
     # "crisp_bass") as a rack preset: body from the low shelf, the 350 Hz mud
     # cut under it, presence and air on top.
     "Crisp and Bassy": {"hp_on": True, "hp_frequency": 70.0,
                         "ls_frequency": 160.0, "ls_gain": 4.0, "ls_q": 0.707,
                         "band2_frequency": 350.0, "band2_gain": -2.5, "band2_q": 1.2,
                         "band4_frequency": 3500.0, "band4_gain": 3.0, "band4_q": 1.0,
                         "hs_frequency": 10000.0, "hs_gain": 3.5}},
    ["highpass", "lowshelf", "equalizer", "highshelf", "lowpass", "volume"]))

# Type, cutoffs, ripple and attenuation rewrite every section's coefficients
# live; Order and Mode change how many sections exist, so they restart.
_SCI_DRIVES = _d("biquad", "b0", "b1", "b2", "a0", "a1", "a2")

_DEFS.append(_effect(
    "scientific_filter", "Scientific Filter", _FE,
    "Textbook filter designs (Bessel, Butterworth, Chebyshev I and II, Elliptic) as "
    "low-pass, high-pass, band-pass or band-stop, up to order 20.",
    [_choice("filter_type", "Type", "Butterworth", list(SCIENTIFIC_TYPES), _SCI_DRIVES),
     _choice("mode", "Mode", "Low Pass", list(SCIENTIFIC_MODES)),
     _num("cutoff", "Cutoff", 1000.0, 10.0, 22000.0, 1.0, "Hz", _SCI_DRIVES),
     _num("high_cutoff", "High Cutoff", 4000.0, 10.0, 22000.0, 1.0, "Hz", _SCI_DRIVES),
     _num("order", "Order", 4.0, 1.0, 20.0, 1.0, ""),
     _num("ripple", "Pass-band Ripple", 1.0, 0.01, 12.0, 0.01, "dB", _SCI_DRIVES),
     _num("attenuation", "Stop-band Attenuation", 60.0, 10.0, 120.0, 0.5, "dB", _SCI_DRIVES),
     _num("master_gain", "Master Gain", 0.0, -30.0, 30.0, 0.1, "dB", _d("volume", "volume"))],
    {"Remove Rumble (High Pass 80 Hz)": {"filter_type": "Butterworth", "mode": "High Pass",
                                         "cutoff": 80.0, "order": 4.0},
     "Telephone (300-3400 Hz)": {"filter_type": "Butterworth", "mode": "Band Pass",
                                 "cutoff": 300.0, "high_cutoff": 3400.0, "order": 4.0},
     "Brickwall Low Pass 8 kHz": {"filter_type": "Elliptic", "mode": "Low Pass",
                                  "cutoff": 8000.0, "order": 8.0, "ripple": 0.5,
                                  "attenuation": 80.0},
     "Gentle Low Pass 5 kHz": {"filter_type": "Bessel", "mode": "Low Pass", "cutoff": 5000.0,
                               "order": 4.0},
     "Hum Band Stop 45-65 Hz": {"filter_type": "Chebyshev II", "mode": "Band Stop",
                                "cutoff": 45.0, "high_cutoff": 65.0, "order": 4.0,
                                "attenuation": 40.0}},
    ["biquad", "volume"]))

# --- Modulation ---------------------------------------------------------------------

_DEFS.append(_effect(
    "chorus", "Chorus", _MO,
    "Several slightly delayed, slowly wavering voices that thicken the sound.",
    [_num("voices", "Voices", 3.0, 1.0, 8.0, 1.0, "", _d("chorus", "delays")),
     _num("delay", "Delay Time", 20.0, 5.0, 50.0, 0.1, "ms", _d("chorus", "delays")),
     _num("spread", "Spread", 5.0, 0.0, 20.0, 0.1, "ms", _d("chorus", "delays")),
     _num("depth", "Modulation Depth", 2.0, 0.1, 10.0, 0.1, "ms", _d("chorus", "depths")),
     _num("rate", "Modulation Rate", 0.5, 0.1, 10.0, 0.01, "Hz", _d("chorus", "speeds")),
     _num("dry", "Dry Output", 70.0, 0.0, 100.0, 1.0, "%", _d("chorus", "in_gain")),
     _num("wet", "Wet Output", 50.0, 0.0, 100.0, 1.0, "%", _d("chorus", "decays"))],
    {"Classic Chorus": {"voices": 3.0, "delay": 20.0, "depth": 2.0, "rate": 0.5,
                        "spread": 5.0, "dry": 70.0, "wet": 50.0},
     "Thick Chorus": {"voices": 6.0, "delay": 25.0, "depth": 3.0, "rate": 0.8,
                      "spread": 10.0, "dry": 60.0, "wet": 60.0},
     "Subtle Doubling": {"voices": 2.0, "delay": 15.0, "depth": 1.0, "rate": 0.3,
                         "spread": 4.0, "dry": 80.0, "wet": 35.0}},
    ["chorus"]))

_DEFS.append(_effect(
    "flanger", "Flanger", _MO,
    "A short sweeping delay mixed with the original: the jet-plane swoosh.",
    [_num("initial_delay", "Initial Delay Time", 1.0, 0.0, 30.0, 0.1, "ms",
          _d("flanger", "delay", "depth")),
     _num("final_delay", "Final Delay Time", 5.0, 0.0, 40.0, 0.1, "ms",
          _d("flanger", "depth")),
     _num("stereo_phasing", "Stereo Phasing", 90.0, 0.0, 360.0, 1.0, "°",
          _d("flanger", "phase")),
     _num("feedback", "Feedback", 50.0, -95.0, 95.0, 1.0, "%", _d("flanger", "regen")),
     _num("rate", "Modulation Rate", 0.5, 0.1, 10.0, 0.01, "Hz", _d("flanger", "speed")),
     _choice("mode", "Mode", "Sinusoidal", ["Sinusoidal", "Triangular", "Inverted"],
             _d("flanger", "shape", "regen")),
     _num("mix", "Mix", 71.0, 0.0, 100.0, 1.0, "%", _d("flanger", "width"))],
    {"Heavy Flange": {"initial_delay": 0.0, "final_delay": 5.0, "stereo_phasing": 90.0,
                      "feedback": 80.0, "rate": 0.3, "mix": 80.0},
     "Light Flange": {"initial_delay": 1.0, "final_delay": 3.0, "stereo_phasing": 90.0,
                      "feedback": 20.0, "rate": 0.5, "mix": 50.0},
     "Jet": {"initial_delay": 0.0, "final_delay": 10.0, "stereo_phasing": 180.0,
             "feedback": 85.0, "rate": 0.15, "mix": 90.0}},
    ["flanger"]))

_DEFS.append(_effect(
    "phaser", "Phaser", _MO,
    "A sweeping comb of phase-shifted notches.",
    [_num("depth", "Depth", 60.0, 0.0, 100.0, 1.0, "%", _d("aphaser", "delay")),
     _num("rate", "Modulation Rate", 0.5, 0.1, 2.0, 0.01, "Hz", _d("aphaser", "speed")),
     _num("feedback", "Feedback", 40.0, 0.0, 99.0, 1.0, "%", _d("aphaser", "decay")),
     _choice("shape", "Shape", "Triangular", ["Triangular", "Sinusoidal"],
             _d("aphaser", "type")),
     _num("output_gain", "Output Gain", 0.0, -20.0, 10.0, 0.1, "dB",
          _d("aphaser", "out_gain"))],
    {"Slow Sweep": {"rate": 0.2, "depth": 70.0, "feedback": 50.0},
     "Fast Phaser": {"rate": 1.5, "depth": 50.0, "feedback": 40.0},
     "Deep Space": {"rate": 0.1, "depth": 100.0, "feedback": 80.0, "shape": "Sinusoidal"}},
    ["aphaser"]))

# --- Noise Reduction / Restoration -----------------------------------------------------

AFFTDN_NOISE_TYPES = {"White": "w", "Vinyl": "v", "Shellac": "s", "Custom": "c"}

_DEFS.append(_effect(
    "adaptive_noise_reduction", "Adaptive Noise Reduction", _NR,
    "Learns the noise as it plays and removes it continuously (FFmpeg afftdn with "
    "noise tracking).",
    [_num("reduce_by", "Reduce Noise By", 12.0, 0.0, 97.0, 0.5, "dB",
          _d("afftdn", "noise_reduction")),
     _num("noisiness", "Noisiness", 50.0, 0.0, 100.0, 1.0, "%", _d("afftdn", "adaptivity")),
     _num("noise_floor", "Fine Tune Noise Floor", -50.0, -80.0, -20.0, 0.5, "dB",
          _d("afftdn", "noise_floor")),
     _num("residual_floor", "Residual Floor", -38.0, -80.0, -20.0, 0.5, "dB",
          _d("afftdn", "residual_floor")),
     _choice("noise_type", "Noise Type", "White", list(AFFTDN_NOISE_TYPES),
             _d("afftdn", "noise_type")),
     _bool("noise_only", "Output Noise Only", False, _d("afftdn", "output_mode"))],
    {"Light Reduction": {"reduce_by": 6.0, "noisiness": 30.0},
     "Heavy Reduction": {"reduce_by": 24.0, "noisiness": 70.0, "residual_floor": -30.0},
     "Vinyl Noise": {"noise_type": "Vinyl", "reduce_by": 12.0}},
    ["afftdn"]))

_DEFS.append(_effect(
    "noise_reduction_rnnoise", "Noise Reduction (RNNoise)", _NR,
    "Speech-trained neural noise suppression (RNNoise); Mix blends it with the original.",
    [_num("mix", "Mix", 100.0, 0.0, 100.0, 1.0, "%", _d("arnndn", "mix"))],
    {"Light Reduction": {"mix": 50.0}, "Heavy Reduction": {"mix": 100.0}},
    ["arnndn"]))

_DEFS.append(_effect(
    "hiss_reduction", "Hiss Reduction", _NR,
    "Removes steady broadband hiss below the noise floor.",
    [_num("noise_floor", "Noise Floor", -45.0, -80.0, -20.0, 0.5, "dB",
          _d("afftdn", "noise_floor")),
     _num("reduce_by", "Reduce Noise By", 12.0, 0.0, 97.0, 0.5, "dB",
          _d("afftdn", "noise_reduction"))],
    {"Light Reduction": {"noise_floor": -50.0, "reduce_by": 6.0},
     "Heavy Reduction": {"noise_floor": -38.0, "reduce_by": 24.0}},
    ["afftdn"]))

# --- Reverb -----------------------------------------------------------------------
#
# Both reverbs convolve with an impulse response loaded in the graph by
# `amovie` (audio/reverb_ir.py builds or prepares the file). Only the mix
# levels and the gain change live; every knob that shapes the impulse
# restarts the stream with a new IR file.

_RV = "Reverb"
IMPULSE_DEFAULT = reverb_ir.SYNTHETIC_PREFIX + "Medium Room"

_DEFS.append(_effect(
    "convolution_reverb", "Convolution Reverb", _RV,
    "Places the sound in a space by convolving it with an impulse response: a synthetic "
    "room, or the creator's own IRs from data/ir/user (listed when the backend starts).",
    [_choice("impulse", "Impulse", IMPULSE_DEFAULT, reverb_ir.impulse_choices()),
     _num("mix", "Mix", 35.0, 0.0, 100.0, 1.0, "%", _d("amix", "weights")),
     _num("room_size", "Room Size", 100.0, 1.0, 100.0, 1.0, "%"),
     _num("damping_lf", "Damping LF", 20.0, 20.0, 1000.0, 1.0, "Hz"),
     _num("damping_hf", "Damping HF", 20000.0, 1000.0, 20000.0, 10.0, "Hz"),
     _num("pre_delay", "Pre-Delay", 0.0, 0.0, 200.0, 0.1, "ms"),
     _num("width", "Width", 100.0, 0.0, 200.0, 1.0, "%"),
     _num("gain", "Gain", 0.0, -24.0, 12.0, 0.1, "dB", _d("volume", "volume"))],
    {"Small Room": {"impulse": "Synthetic: Small Room", "mix": 25.0},
     "Large Hall": {"impulse": "Synthetic: Large Hall", "mix": 35.0, "pre_delay": 10.0},
     "Vocal Plate": {"impulse": "Synthetic: Plate", "mix": 25.0, "damping_lf": 150.0},
     "Intimate Booth": {"impulse": "Synthetic: Intimate Booth", "mix": 20.0},
     "Cave": {"impulse": "Synthetic: Cave", "mix": 45.0, "damping_hf": 6000.0}},
    ["asplit", "amovie", "afir", "amix", "volume"]))


def _studio_reverb(room: float, decay: float, er: float, width: float, hf: float, lf: float,
                   damping: float, diffusion: float, dry: float, wet: float) -> Dict[str, float]:
    return {"room_size": room, "decay": decay, "early_reflections": er, "width": width,
            "hf_cut": hf, "lf_cut": lf, "damping": damping, "diffusion": diffusion,
            "dry": dry, "wet": wet}


_DEFS.append(_effect(
    "reverb", "Studio Reverb", _RV,
    "A room reverb built from the knobs below (a synthetic impulse response, convolved). "
    "Dry and Wet change live; the room knobs restart the stream.",
    [_num("room_size", "Room Size", 50.0, 0.0, 100.0, 1.0, "%"),
     _num("decay", "Decay", 1200.0, 100.0, 10000.0, 10.0, "ms"),
     _num("early_reflections", "Early Reflections", 50.0, 0.0, 100.0, 1.0, "%"),
     _num("width", "Width", 80.0, 0.0, 100.0, 1.0, "%"),
     _num("hf_cut", "High Frequency Cut", 8000.0, 1000.0, 20000.0, 10.0, "Hz"),
     _num("lf_cut", "Low Frequency Cut", 100.0, 20.0, 1000.0, 1.0, "Hz"),
     _num("damping", "Damping", 50.0, 0.0, 100.0, 1.0, "%"),
     _num("diffusion", "Diffusion", 80.0, 0.0, 100.0, 1.0, "%"),
     _num("dry", "Dry", 100.0, 0.0, 100.0, 1.0, "%", _d("amix", "weights")),
     _num("wet", "Wet", 30.0, 0.0, 100.0, 1.0, "%", _d("amix", "weights"))],
    {"Vocal Ambience (Small)": _studio_reverb(15, 450, 60, 50, 9000, 120, 55, 70, 100, 18),
     "Vocal Ambience (Medium)": _studio_reverb(35, 900, 50, 70, 8500, 110, 50, 75, 100, 22),
     "Vocal Ambience (Large)": _studio_reverb(60, 1600, 40, 85, 8000, 100, 45, 80, 100, 25),
     "Drum Plate": _studio_reverb(20, 1400, 10, 100, 12000, 150, 20, 100, 100, 30),
     "Great Hall": _studio_reverb(95, 3800, 35, 100, 7000, 60, 40, 85, 90, 40),
     "Horror Intimate": _studio_reverb(12, 600, 70, 30, 5000, 150, 70, 55, 100, 28),
     "Cave": _studio_reverb(100, 5000, 60, 100, 6000, 50, 60, 45, 85, 45)},
    ["asplit", "amovie", "afir", "amix"]))

# --- Special ------------------------------------------------------------------------

DISTORTION_CURVES = {"Soft Clip": "tanh", "Hard Clip": "hard", "Arctangent": "atan",
                     "Cubic": "cubic", "Exponential": "exp", "Algebraic": "alg",
                     "Quintic": "quintic", "Sine": "sin", "Error Function": "erf"}

_DEFS.append(_effect(
    "distortion", "Distortion", _SP,
    "Drives the signal into a clipping curve.",
    [_choice("curve_type", "Curve Type", "Soft Clip", list(DISTORTION_CURVES),
             _d("asoftclip", "type")),
     _num("drive", "Drive", 12.0, 0.0, 36.0, 0.1, "dB", _d("acrusher", "level_in")),
     _num("output", "Output", -6.0, -24.0, 12.0, 0.1, "dB", _d("asoftclip", "output"))],
    {"Soft Clip": {"curve_type": "Soft Clip", "drive": 12.0, "output": -6.0},
     "Hard Clip": {"curve_type": "Hard Clip", "drive": 18.0, "output": -8.0},
     "Warm Saturation": {"curve_type": "Arctangent", "drive": 6.0, "output": -3.0},
     "Fuzz": {"curve_type": "Hard Clip", "drive": 30.0, "output": -12.0}},
    ["acrusher", "asoftclip"]))

EXCITER_MODES = {"Retro": (2500.0, 6.0), "Tape": (4500.0, 4.0), "Tube": (7500.0, 8.5)}

_DEFS.append(_effect(
    "mastering", "Mastering", _SP,
    "Final polish: three-band EQ, exciter, stereo widener and loudness maximizer.",
    [_num("low", "Low", 0.0, -12.0, 12.0, 0.1, "dB", _d("lowshelf", "g")),
     _num("mid", "Mid", 0.0, -12.0, 12.0, 0.1, "dB", _d("equalizer", "g")),
     _num("high", "High", 0.0, -12.0, 12.0, 0.1, "dB", _d("highshelf", "g")),
     _choice("exciter_mode", "Exciter Mode", "Tape", list(EXCITER_MODES),
             _d("aexciter", "freq", "drive")),
     _num("exciter", "Exciter Amount", 0.0, 0.0, 100.0, 1.0, "%", _d("aexciter", "amount")),
     _num("widener", "Widener", 0.0, 0.0, 100.0, 1.0, "%", _d("stereotools", "slev")),
     _num("maximizer", "Loudness Maximizer", 0.0, 0.0, 100.0, 1.0, "%",
          _d("alimiter", "level_in")),
     _num("output_gain", "Output Gain", 0.0, -24.0, 24.0, 0.1, "dB",
          _d("alimiter", "level_out"))],
    {"Subtle Clarity": {"high": 2.0, "exciter_mode": "Tape", "exciter": 20.0,
                        "maximizer": 20.0},
     "Warm Voice": {"low": 2.0, "high": -1.0, "exciter_mode": "Tube", "exciter": 15.0},
     "Bright Podcast": {"low": -1.0, "mid": 1.0, "high": 3.0, "exciter_mode": "Retro",
                        "exciter": 25.0, "maximizer": 40.0},
     "Loud and Wide": {"widener": 50.0, "maximizer": 70.0}},
    ["lowshelf", "equalizer", "highshelf", "aexciter", "stereotools", "alimiter"]))

# Per mode: high-pass Hz, rumble-shelf Hz, presence Hz, presence dB at 100 %,
# de-ess Hz, exciter Hz.
VOCAL_ENHANCER_MODES = {
    "Male": (80.0, 150.0, 3000.0, 3.5, 6000.0, 5000.0),
    "Female": (100.0, 200.0, 4500.0, 3.0, 7500.0, 6500.0),
    "Music": (40.0, 100.0, 3500.0, 2.0, 8000.0, 7500.0),
}
_VE_MODE_DRIVES = (_d("highpass", "frequency") + _d("lowshelf", "frequency", "gain")
                   + _d("equalizer", "frequency", "gain")
                   + _d("adynamicequalizer", "dfrequency", "tfrequency")
                   + _d("aexciter", "freq"))
_VE_AMOUNT_DRIVES = (_d("lowshelf", "gain") + _d("equalizer", "gain")
                     + _d("acompressor", "ratio") + _d("aexciter", "amount"))

_DEFS.append(_effect(
    "vocal_enhancer", "Vocal Enhancer", _SP,
    "One-knob voice polish: rumble cut, presence lift, de-essing, gentle compression, "
    "a touch of excitement and a -1 dB ceiling.",
    [_choice("mode", "Mode", "Male", list(VOCAL_ENHANCER_MODES), _VE_MODE_DRIVES),
     _num("amount", "Amount", 60.0, 0.0, 100.0, 1.0, "%", _VE_AMOUNT_DRIVES)],
    {"Vocal Enhancer: Male": {"mode": "Male", "amount": 60.0},
     "Vocal Enhancer: Female": {"mode": "Female", "amount": 60.0},
     "Vocal Enhancer: Music": {"mode": "Music", "amount": 60.0},
     "Podcast Voice": {"mode": "Male", "amount": 80.0}},
    ["highpass", "lowshelf", "equalizer", "adynamicequalizer", "acompressor", "aexciter",
     "alimiter"]))

_DEFS.append(_effect(
    "sub_octave", "Sub-Octave", _SP,
    "Adds a copy of the sound one octave lower under the original: weight, or a "
    "monster-like double.",
    [_num("mix", "Mix", 50.0, 0.0, 100.0, 1.0, "%", _d("volume", "volume"))],
    {"Subtle Weight": {"mix": 25.0}, "Deep Double": {"mix": 50.0},
     "Monster Low": {"mix": 85.0}},
    ["asplit", "rubberband", "volume", "amix"]))

# --- Stereo Imagery -------------------------------------------------------------------

_DEFS.append(_effect(
    "stereo_expander", "Stereo Expander", _ST,
    "Moves the centre of the stereo image and widens or narrows it.",
    [_num("center_pan", "Center Channel Pan", 0.0, -100.0, 100.0, 1.0, "%",
          _d("stereotools", "mpan")),
     _num("expand", "Stereo Expand", 100.0, 0.0, 300.0, 1.0, "%", _d("extrastereo", "m"))],
    {"Narrow": {"expand": 50.0}, "Wide": {"expand": 160.0},
     "Extra Wide": {"expand": 250.0}, "Mono": {"expand": 0.0}},
    ["stereotools", "extrastereo"]))

_DEFS.append(_effect(
    "graphic_phase_shifter", "Graphic Phase Shifter", _ST,
    "Shifts the phase of the whole signal by a set angle.",
    [_num("shift", "Phase Shift", 90.0, -180.0, 180.0, 1.0, "°", _d("aphaseshift", "shift")),
     _num("level", "Level", 0.0, -24.0, 0.0, 0.1, "dB", _d("aphaseshift", "level")),
     _num("order", "Filter Order", 8.0, 1.0, 16.0, 1.0, "", _d("aphaseshift", "order"))],
    {"+90 Degrees": {"shift": 90.0}, "-90 Degrees": {"shift": -90.0},
     "Invert (180 Degrees)": {"shift": 180.0}},
    ["aphaseshift"]))

# --- Time and Pitch -------------------------------------------------------------------

PITCH_PRECISION = {"Low": ("speed", "smooth"), "Medium": ("consistency", "mixed"),
                   "High": ("quality", "crisp")}

_DEFS.append(_effect(
    "pitch_shifter", "Pitch Shifter", _TP,
    "Changes pitch without changing duration (Rubber Band).",
    [_num("semitones", "Semi-tones", 0.0, -36.0, 36.0, 1.0, "st", _d("rubberband", "pitch")),
     _num("cents", "Cents", 0.0, -100.0, 100.0, 1.0, "cents", _d("rubberband", "pitch")),
     _choice("formant", "Formant", "Preserved", ["Preserved", "Shifted"],
             _d("rubberband", "formant")),
     _choice("precision", "Precision", "High", list(PITCH_PRECISION),
             _d("rubberband", "pitchq", "transients"))],
    {"Angry Gerbil": {"semitones": 9.0, "formant": "Shifted"},
     "Darth Vader": {"semitones": -8.0, "formant": "Shifted"},
     "Octave Up": {"semitones": 12.0}, "Octave Down": {"semitones": -12.0},
     "Deeper Voice": {"semitones": -3.0, "formant": "Preserved"}},
    ["rubberband"]))

_DEFS.append(_effect(
    "stretch_and_pitch", "Stretch and Pitch", _TP,
    "Changes duration and pitch independently, or locked together like tape speed "
    "(Rubber Band). A stretch changes the length of the scope.",
    [_num("stretch", "Stretch", 100.0, 10.0, 400.0, 0.1, "%", _d("rubberband", "tempo")
          + _d("rubberband", "pitch")),
     _num("pitch", "Pitch Shift", 0.0, -36.0, 36.0, 0.1, "st", _d("rubberband", "pitch")),
     _bool("lock", "Lock Stretch and Pitch", False, _d("rubberband", "pitch")),
     _bool("preserve_formants", "Preserve Formants", False, _d("rubberband", "formant"))],
    {"Speed Up 10%": {"stretch": 91.0}, "Slow Down 10%": {"stretch": 110.0},
     "Half Speed": {"stretch": 200.0}, "Double Speed": {"stretch": 50.0},
     "Tape Slow": {"stretch": 120.0, "lock": True}},
    ["rubberband"]))

_DEFS.append(_effect(
    "pitch_bender", "Pitch Bender", _TP,
    "Bends the pitch along a curve (semitones at each point, straight ramps between) "
    "without changing the duration. Times are seconds into the scope; a curve edit "
    "restarts the stream.",
    [_curve("curve", "Pitch Curve", [[0.0, 0.0], [1.0, 0.0]], x_min=0.0, x_max=3600.0,
            x_unit="s", x_log=False, y_min=-24.0, y_max=24.0, y_unit="st")],
    {"Rise One Octave": {"curve": [[0.0, 0.0], [2.0, 12.0]]},
     "Fall One Octave": {"curve": [[0.0, 0.0], [2.0, -12.0]]},
     "Up a Fifth and Back": {"curve": [[0.0, 0.0], [1.0, 7.0], [2.0, 0.0]]},
     "Wobble": {"curve": [[0.0, 0.0], [0.25, 1.0], [0.5, -1.0], [0.75, 1.0], [1.0, -1.0],
                          [1.25, 0.0]]}},
    ["asendcmd", "rubberband"]))

# --- the catalogue ----------------------------------------------------------------------

EFFECTS: Dict[str, Dict[str, Any]] = {d["id"]: d for d in _DEFS}
assert len(EFFECTS) == len(_DEFS), "duplicate effect id"
assert all(d["category"] in CATEGORIES for d in _DEFS)


def param_defs(effect_id: str) -> List[Dict[str, Any]]:
    """The ParamDefs of one catalogue entry (KeyError for an unknown id)."""
    return EFFECTS[effect_id]["params"]
