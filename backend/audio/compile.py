"""Compile Audition-style audio effects to ffmpeg (AUDIO_SUITE_PLAN.md §2.6).

    compile_effect(effect) -> CompiledEffect     one AudioEffect -> named filter instances
    compile_rack(effects, scope=...) -> CompiledRack   skips disabled effects, joins the chain
    commands_for(rack, effect_id, param, value, params) -> [(target, option, value)] | None
    public_catalogue() -> JSON-safe copy of catalogue.EFFECTS

Every filter instance is named `<filter>@<effect.id>_<k>` (k = position inside
the effect), which is also the target of a runtime command: ffmpeg 8.1 names
the instance `equalizer@fx_7f3a_0`, so `azmq`/`asendcmd` take exactly that
string. `CommandSpec.transform(value, all_params)` returns the option value
as a raw string -- no filtergraph quoting, ready for `azmq`.

An effect is anything with `.id .effect .params .enabled .start_s .end_s`
(the `AudioEffect` model of timeline/schema.py) or a dict with those keys.

Bypass strategies (one per compiled effect):
  "enable"  every instance supports the timeline: send `enable 0` to each
            (restore with `enable 1`, or the window's `between(t,A,B)`);
  "mix"     acompressor/arnndn instances get `mix 0`, every other instance of
            the effect `enable 0`;
  "flat"    firequalizer instances get the flat `gain_entry entry(0,0)`,
            acompressor/arnndn `mix 0`, every other instance `enable 0`;
  "restart" rebuild the stream without the effect.

Windows: `start_s`/`end_s` become `enable='between(t,A,B)'` on every instance
of a windowable effect; an effect that is not windowable compiles to the
whole scope and says so in `warnings`.

The chain string (the contract for every caller -- render compiler, preview,
live engine): `CompiledRack.chain` may contain `;`-separated graph segments
with internal labels, e.g.

    asplit@x_0[x_o0][x_o1];amovie@x_1=filename='B\\:/.../ir.wav'[x_o2];
    [x_o1][x_o2]afir@x_2=...[x_o3];[x_o0][x_o3]amix@x_3=...,volume@x_4=...

The first segment's head input and the last segment's tail output are left
unlabelled, so a caller prefixes its own input label and suffixes its own
output label (`[in]` + chain + `[out]`). A chain can hold source filters
(`amovie` loading an impulse response), so it must only ever be used inside
`-filter_complex`, never `-af`. Internal labels are `<stem>_o<n>`, derived
from the effect id, so they are unique per effect. `afir` takes the main
signal first and the impulse response second; file paths inside the graph
are escaped with `render/audio.py::_ffmpeg_filter_path`. Effects load such
files themselves, so a rack never needs extra `-i` inputs.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from audio.catalogue import (
    AFFTDN_NOISE_TYPES,
    DISTORTION_CURVES,
    EFFECTS,
    EXCITER_MODES,
    GRAPHIC_EQ_CENTRES,
    GRAPHIC_EQ_Q,
    MIX_BYPASS_FILTERS,
    NOTCH_COUNT,
    NOTCH_WIDTH_Q,
    PEQ_BANDS,
    PITCH_PRECISION,
    TIMELINE_FILTERS,
    VOCAL_ENHANCER_MODES,
    option_is_live,
)
from audio import reverb_ir
from audio.catalogue import IMPULSE_DEFAULT
from audio.scientific import design_sos, section_count

Bypass = Literal["enable", "mix", "flat", "restart"]

FLAT_GAIN_ENTRY = "entry(0,0)"
_FOREVER_S = 31_536_000.0  # an open-ended window runs to "a year"


@dataclass
class CommandSpec:
    target: str                                   # "<filter>@<effect.id>_<k>"
    option: str
    transform: Callable[[Any, Dict[str, Any]], str]   # (value, all_params) -> option value


@dataclass
class CompiledEffect:
    effect_id: str
    filters: List[str]
    commands: Dict[str, List[CommandSpec]]        # param key -> commands (live params only)
    bypass: Bypass
    warnings: List[str] = field(default_factory=list)


@dataclass
class CompiledRack:
    filters: List[str]
    chain: str                                    # "" for an empty rack
    effects: List[CompiledEffect]
    live_params: Dict[Tuple[str, str], List[CommandSpec]]
    restart_params: Set[Tuple[str, str]]


# --- small helpers -------------------------------------------------------------

def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _f(x: float) -> str:
    """A number as ffmpeg wants it: plain decimal, no exponent, no trailing zeros."""
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _lin(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


_NEEDS_QUOTE = re.compile(r"[\s,;\[\]\\:=']")


def _q(value: str) -> str:
    """Quote an option value for the filtergraph when it holds a separator."""
    return f"'{value}'" if _NEEDS_QUOTE.search(value) else value


def _stem(effect_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", str(effect_id or ""))
    return stem or "fx"


# --- parameter resolution (UI values -> clamped, typed values) ------------------

def _coerce(d: Dict[str, Any], v: Any) -> Any:
    t = d["type"]
    try:
        if t == "number":
            x = float(v)
            if not math.isfinite(x):
                raise ValueError(v)
            return _clamp(x, d["min"], d["max"])
        if t == "bool":
            if isinstance(v, str):
                return v.strip().lower() not in ("", "0", "false", "off", "no")
            return bool(v)
        if t == "choice":
            if v in d["choices"]:
                return v
            for c in d["choices"]:
                if str(v).strip().lower() == c.lower():
                    return c
            return d["default"]
        if t == "curve":
            pts: Dict[float, float] = {}
            for p in v:
                x, y = float(p[0]), float(p[1])
                if math.isfinite(x) and math.isfinite(y):
                    pts[_clamp(x, d["x_min"], d["x_max"])] = _clamp(y, d["min"], d["max"])
            if not pts:
                raise ValueError(v)
            return [[x, pts[x]] for x in sorted(pts)]
        if t == "bands":
            given = list(v) if isinstance(v, (list, tuple)) else []
            out = []
            for i, band_default in enumerate(d["default"]):
                src = given[i] if i < len(given) and isinstance(given[i], Mapping) else {}
                out.append({f["key"]: _coerce(f, src.get(f["key"], band_default.get(f["key"], f["default"])))
                            for f in d["fields"]})
            return out
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return copy.deepcopy(d["default"])


def _resolve_params(catalogue_id: str, params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Defaults merged with `params`, every value coerced and clamped to its ParamDef."""
    given = dict(params or {})
    return {d["key"]: _coerce(d, given.get(d["key"], d["default"]))
            for d in EFFECTS[catalogue_id]["params"]}


# --- graph building ---------------------------------------------------------------

class _L:
    """An option value computed from the params, live for the listed param keys."""

    __slots__ = ("fn", "keys")

    def __init__(self, fn: Callable[[Dict[str, Any]], str], *keys: str):
        self.fn = fn
        self.keys = keys


@dataclass
class _Node:
    filter: str
    target: str
    opts: List[Tuple[str, str]]
    ins: Sequence[str] = ()
    outs: Sequence[str] = ()

    def render(self) -> str:
        s = "".join(f"[{x}]" for x in self.ins) + self.target
        if self.opts:
            s += "=" + ":".join(f"{k}={_q(v)}" for k, v in self.opts)
        return s + "".join(f"[{x}]" for x in self.outs)


class _Graph:
    def __init__(self, stem: str, catalogue_id: str, params: Dict[str, Any]):
        self.stem = stem
        self.cat = catalogue_id
        self.p = params
        self.nodes: List[_Node] = []
        self.commands: Dict[str, List[CommandSpec]] = {}
        self.warnings: List[str] = []
        self._labels = 0
        self._live = {d["key"]: d["live"] for d in EFFECTS[catalogue_id]["params"]}

    def label(self) -> str:
        name = f"{self.stem}_o{self._labels}"
        self._labels += 1
        return name

    def add(self, filt: str, opts: Sequence[Tuple[str, Any]] = (), ins: Sequence[str] = (),
            outs: Sequence[str] = ()) -> str:
        target = f"{filt}@{self.stem}_{len(self.nodes)}"
        rendered: List[Tuple[str, str]] = []
        for opt, val in opts:
            if isinstance(val, _L):
                rendered.append((opt, val.fn(self.p)))
                for key in val.keys:
                    self._register(key, filt, target, opt, val.fn)
            else:
                rendered.append((opt, val if isinstance(val, str) else _f(val)))
        self.nodes.append(_Node(filt, target, rendered, tuple(ins), tuple(outs)))
        return target

    def _register(self, key: str, filt: str, target: str, opt: str,
                  fn: Callable[[Dict[str, Any]], str]) -> None:
        if key not in self._live:
            raise KeyError(f"{self.cat}: unknown param {key!r}")
        if not self._live[key]:
            return  # the catalogue says this param restarts the stream
        if not option_is_live(filt, opt):
            raise AssertionError(f"{self.cat}.{key} is marked live but {filt}:{opt} "
                                 "takes no runtime command")
        cat = self.cat

        def transform(value: Any, all_params: Dict[str, Any], _fn=fn, _key=key) -> str:
            merged = dict(all_params or {})
            merged[_key] = value
            return _fn(_resolve_params(cat, merged))

        self.commands.setdefault(key, []).append(CommandSpec(target, opt, transform))


def _num(key: str, conv: Callable[[float], float] = float) -> _L:
    """Live option = conv(param) formatted as a number."""
    return _L(lambda q: _f(conv(q[key])), key)


def _bypass_for(names: Sequence[str]) -> Bypass:
    if all(n in TIMELINE_FILTERS for n in names):
        return "enable"
    mixable = TIMELINE_FILTERS | MIX_BYPASS_FILTERS
    if all(n in mixable for n in names):
        return "mix"
    if "firequalizer" in names and all(n in mixable or n == "firequalizer" for n in names):
        return "flat"
    return "restart"


# --- builders, one per catalogue entry -------------------------------------------

_BUILDERS: Dict[str, Callable[[_Graph, Dict[str, Any]], None]] = {}


def _builder(eid: str):
    def deco(fn):
        _BUILDERS[eid] = fn
        return fn
    return deco


# ffmpeg option ranges the conversions must respect
_COMP_THR = (0.000976563, 1.0)       # acompressor threshold
_UNITY_64 = (0.015625, 64.0)         # level_in / level_out style gains


def _comp_threshold(db: float) -> float:
    return _clamp(_lin(db), *_COMP_THR)


# --- Amplitude and Compression

@_builder("amplify")
def _b_amplify(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("volume", [("volume", _num("gain", _lin))])


def _pan_terms(terms: Sequence[Tuple[float, str]]) -> str:
    out = ""
    for i, (gain, ch) in enumerate(terms):
        if i == 0:
            out += f"{_f(gain)}*{ch}"
        else:
            out += ("+" if gain >= 0 else "-") + f"{_f(abs(gain))}*{ch}"
    return out


@_builder("channel_mixer")
def _b_channel_mixer(g: _Graph, p: Dict[str, Any]) -> None:
    sl = -1.0 if p["invert_l"] else 1.0
    sr = -1.0 if p["invert_r"] else 1.0
    left = _pan_terms([(sl * p["l_from_l"] / 100.0, "c0"), (sl * p["l_from_r"] / 100.0, "c1")])
    right = _pan_terms([(sr * p["r_from_l"] / 100.0, "c0"), (sr * p["r_from_r"] / 100.0, "c1")])
    g.add("pan", [("args", f"stereo|c0={left}|c1={right}")])


def _deesser_multiband(q: Dict[str, Any]) -> bool:
    return q["mode"] == "Multiband"


@_builder("deesser")
def _b_deesser(g: _Graph, p: Dict[str, Any]) -> None:
    # Multiband: a bell at the sibilance band is turned down; Broadband: a
    # shelf from 20 Hz up, i.e. the whole signal ducks while the "s" lasts.
    g.add("adynamicequalizer", [
        ("threshold", _num("threshold", _lin)),
        ("dfrequency", _num("center")),
        ("dqfactor", _L(lambda q: _f(q["center"] / q["bandwidth"]), "center", "bandwidth")),
        ("tfrequency", _L(lambda q: _f(q["center"] if _deesser_multiband(q) else 20.0),
                          "center", "mode")),
        ("tqfactor", _L(lambda q: _f(q["center"] / q["bandwidth"]
                                     if _deesser_multiband(q) else 0.707),
                        "center", "bandwidth", "mode")),
        ("tftype", _L(lambda q: "bell" if _deesser_multiband(q) else "highshelf", "mode")),
        ("mode", _L(lambda q: "listen" if q["sibilance_only"] else "cutabove",
                    "sibilance_only")),
        ("dftype", "bandpass"), ("ratio", "0.5"), ("range", "12"),
        ("attack", "1"), ("release", "50"),
    ])


def _expander_points(threshold: float, ratio: float) -> str:
    low = -90.0
    pts: List[Tuple[float, float]] = []
    if threshold > low:
        pts.append((low, threshold + (low - threshold) * ratio))
    pts.append((threshold, threshold))
    if threshold < 0.0:
        pts.append((0.0, 0.0))
    return "|".join(f"{_f(x)}/{_f(y)}" for x, y in pts)


@_builder("dynamics")
def _b_dynamics(g: _Graph, p: Dict[str, Any]) -> None:
    if p["gate_on"]:
        g.add("agate", [
            ("threshold", _num("gate_threshold", lambda db: _clamp(_lin(db), 0.0, 1.0))),
            ("attack", _num("gate_attack")),
            ("release", _num("gate_release")),
            ("ratio", "10"), ("range", "0.001"),
        ])
    if p["comp_on"]:
        g.add("acompressor", [
            ("threshold", _num("comp_threshold", _comp_threshold)),
            ("ratio", _num("comp_ratio")),
            ("attack", _num("comp_attack")),
            ("release", _num("comp_release")),
            ("makeup", _num("comp_makeup", lambda db: _clamp(_lin(db), 1.0, 64.0))),
        ])
    if p["exp_on"]:
        g.add("compand", [
            ("attacks", "0.005"), ("decays", "0.1"),
            ("points", _expander_points(p["exp_threshold"], p["exp_ratio"])),
        ])
    if p["lim_on"]:
        g.add("alimiter", [
            ("limit", _num("lim_threshold", lambda db: _clamp(_lin(db), 0.0625, 1.0))),
            ("attack", _num("lim_attack")),
            ("release", _num("lim_release")),
            ("level", "0"),
        ])


@_builder("dynamics_processing")
def _b_dynamics_processing(g: _Graph, p: Dict[str, Any]) -> None:
    points = "|".join(f"{_f(x)}/{_f(y)}" for x, y in p["curve"])
    g.add("compand", [
        ("attacks", _f(p["attack"] / 1000.0)),
        ("decays", _f(p["release"] / 1000.0)),
        ("points", points),
        ("delay", _f(p["lookahead"] / 1000.0)),
    ])
    if p["lookahead"] > 0:
        g.warnings.append(f"Look-Ahead delays this effect's output by {_f(p['lookahead'])} ms")


@_builder("hard_limiter")
def _b_hard_limiter(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("alimiter", [
        ("limit", _num("max_amplitude", lambda db: _clamp(_lin(db), 0.0625, 1.0))),
        ("level_in", _num("input_boost", lambda db: _clamp(_lin(db), *_UNITY_64))),
        ("attack", _num("lookahead", lambda ms: _clamp(ms, 0.1, 80.0))),
        ("release", _num("release", lambda ms: _clamp(ms, 1.0, 8000.0))),
        ("level", "0"),
    ])


def _crossovers(p: Dict[str, Any]) -> List[float]:
    lo, mid, hi = sorted([p["crossover_low"], p["crossover_mid"], p["crossover_high"]])
    mid = max(mid, lo * 1.25)
    hi = min(max(hi, mid * 1.25), 20000.0)
    return [lo, mid, hi]


@_builder("multiband_compressor")
def _b_multiband_compressor(g: _Graph, p: Dict[str, Any]) -> None:
    split = [g.label() for _ in range(4)]
    g.add("acrossover", [("split", " ".join(_f(x) for x in _crossovers(p)))], outs=split)
    comped = []
    for i in range(4):
        out = g.label()
        g.add("acompressor", [
            ("threshold", _L(lambda q, i=i: _f(_comp_threshold(q["bands"][i]["threshold"])),
                             "bands")),
            ("ratio", _L(lambda q, i=i: _f(q["bands"][i]["ratio"]), "bands")),
            ("attack", _L(lambda q, i=i: _f(q["bands"][i]["attack"]), "bands")),
            ("release", _L(lambda q, i=i: _f(q["bands"][i]["release"]), "bands")),
        ], ins=[split[i]], outs=[out])
        comped.append(out)
    g.add("amix", [
        ("inputs", "4"), ("duration", "first"), ("normalize", "0"),
        ("weights", _L(lambda q: " ".join(_f(_lin(b["gain"])) for b in q["bands"]), "bands")),
    ], ins=comped)
    g.add("alimiter", [
        ("level_in", _num("output_gain", lambda db: _clamp(_lin(db), *_UNITY_64))),
        ("limit", _num("limiter_threshold", lambda db: _clamp(_lin(db), 0.0625, 1.0))),
        ("attack", "5"), ("release", "50"), ("level", "0"),
    ])


@_builder("single_band_compressor")
def _b_single_band_compressor(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("acompressor", [
        ("threshold", _num("threshold", _comp_threshold)),
        ("ratio", _num("ratio")),
        ("attack", _num("attack")),
        ("release", _num("release")),
    ])
    g.add("volume", [("volume", _num("output_gain", _lin))])


def _leveling_factor(q: Dict[str, Any]) -> float:
    return 1.0 + 9.0 * q["amount"] / 100.0


@_builder("speech_volume_leveler")
def _b_speech_volume_leveler(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("speechnorm", [
        ("expansion", _L(lambda q: _f(_leveling_factor(q) if q["boost_low"] else 1.0),
                         "amount", "boost_low")),
        ("compression", _L(lambda q: _f(_leveling_factor(q)), "amount")),
        ("rms", _num("target", lambda db: _clamp(_lin(db), 0.0, 1.0))),
        ("peak", "0.95"), ("link", "1"),
    ])
    if p["noise_gate"]:
        g.add("agate", [
            ("threshold", _num("gate_threshold", lambda db: _clamp(_lin(db), 0.0, 1.0))),
            ("ratio", "10"), ("range", "0.003"), ("attack", "5"), ("release", "150"),
        ])


@_builder("tube_compressor")
def _b_tube_compressor(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("acompressor", [
        ("threshold", _num("threshold", _comp_threshold)),
        ("ratio", _num("ratio")),
        ("attack", _num("attack")),
        ("release", _num("release")),
        ("knee", "4"),
    ])
    g.add("asoftclip", [
        ("type", "tanh"), ("threshold", "0.7"),
        ("output", _num("output_gain", lambda db: _clamp(_lin(db), 0.000001, 16.0))),
    ])
    g.add("aexciter", [("amount", "0.5"), ("drive", "4"), ("blend", "2"), ("freq", "4000"),
                       ("ceil", "16000")])


# --- Delay and Echo

@_builder("delay")
def _b_delay(g: _Graph, p: Dict[str, Any]) -> None:
    dry, wet = g.label(), g.label()
    g.add("asplit", [], outs=[dry, wet])
    delayed = g.label()
    g.add("adelay", [
        ("delays", _L(lambda q: f"{_f(q['left_delay'])}|{_f(q['right_delay'])}",
                      "left_delay", "right_delay")),
    ], ins=[wet], outs=[delayed])
    g.add("amix", [
        ("inputs", "2"), ("duration", "first"), ("normalize", "0"),
        ("weights", _L(lambda q: f"{_f(1.0 - q['mix'] / 100.0)} {_f(q['mix'] / 100.0)}",
                       "mix")),
    ], ins=[dry, delayed])


def _echo_taps(delay_ms: float, level: float, feedback: float) -> Tuple[str, str]:
    """aecho taps that emulate a feedback echo: tap k at k*delay, level*fb^(k-1)."""
    decays = [_clamp(level, 0.001, 1.0)]
    while len(decays) < 8:
        nxt = decays[-1] * feedback
        if nxt < 0.001:
            break
        decays.append(nxt)
    delays = [min(delay_ms * (k + 1), 90000.0) for k in range(len(decays))]
    return "|".join(_f(d) for d in delays), "|".join(_f(x) for x in decays)


@_builder("echo")
def _b_echo(g: _Graph, p: Dict[str, Any]) -> None:
    delays, decays = _echo_taps(p["delay"], p["level"] / 100.0, p["feedback"] / 100.0)
    g.add("aecho", [("in_gain", "1"), ("out_gain", "1"), ("delays", delays), ("decays", decays)])


ANALOG_DELAY_LOWPASS = {"Tape": 6000.0, "Tape/Tube": 4500.0, "Analog": 3000.0}


@_builder("analog_delay")
def _b_analog_delay(g: _Graph, p: Dict[str, Any]) -> None:
    delays, decays = _echo_taps(p["delay"], p["wet"] / 100.0, p["feedback"] / 100.0)
    g.add("aecho", [("in_gain", _f(p["dry"] / 100.0)), ("out_gain", "1"),
                    ("delays", delays), ("decays", decays)])
    g.add("lowpass", [("frequency", _L(lambda q: _f(ANALOG_DELAY_LOWPASS[q["mode"]]), "mode")),
                      ("poles", "2")])


# --- Diagnostics

@_builder("declipper")
def _b_declipper(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("volume", [("volume", _num("gain", _lin))])
    g.add("adeclip", [("window", _f(p["window"])), ("overlap", _f(p["overlap"])),
                      ("arorder", _f(round(p["arorder"]))), ("threshold", _f(p["threshold"])),
                      ("hsize", _f(round(p["hsize"])))])


@_builder("dehummer")
def _b_dehummer(g: _Graph, p: Dict[str, Any]) -> None:
    entries = []
    for k in range(1, int(round(p["harmonics"])) + 1):
        freq = p["frequency"] * k
        if freq >= 20000.0:
            break
        width = freq / p["q"]
        for ch in (0, 1):
            entries.append(f"c{ch} f={_f(freq)} w={_f(width)} g={_f(p['gain'])} t=0")
    g.add("anequalizer", [("params", "|".join(entries))])


@_builder("declicker")
def _b_declicker(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("adeclick", [("window", _f(p["window"])), ("overlap", "75"),
                       ("arorder", _f(round(p["complexity"] / 4.0))),
                       ("threshold", _f(p["threshold"])), ("burst", _f(p["burst"]))])


# --- Filter and EQ

def _gain_entry(points: Sequence[Tuple[float, float]]) -> str:
    return ";".join(f"entry({_f(f)},{_f(gain)})" for f, gain in points)


@_builder("fft_filter")
def _b_fft_filter(g: _Graph, p: Dict[str, Any]) -> None:
    # scale=linlog: entries in Hz and dB (loglog read the Hz as log2(f/20), so
    # every frequency got the first entry's gain). min_phase=1: no onset delay
    # against the picture (linear phase would delay the sound by `delay`). The
    # stream still ends 2 x delay (100 ms) later; render/compiler.py cuts the
    # finished audio back to the picture whenever a rack is present.
    g.add("firequalizer", [
        ("gain", _L(lambda q: "cubic_interpolate(f)" if q["spline"] else "gain_interpolate(f)",
                    "spline")),
        ("gain_entry", _L(lambda q: _gain_entry(q["curve"]), "curve")),
        ("scale", "linlog"),
        ("delay", "0.05"),
        ("min_phase", "1"),
    ])


def _graphic_eq_builder(eid: str):
    """One peaking `equalizer` per band (Q by the band spacing), then the
    master gain as a `volume`. Range clamps each band to +/- range/2."""
    centres = GRAPHIC_EQ_CENTRES[eid]
    width = _f(GRAPHIC_EQ_Q[eid])
    keys = [f"band_{i + 1}" for i in range(len(centres))]

    def band_gain(key: str) -> _L:
        return _L(lambda q: _f(_clamp(q[key], -q["range"] / 2.0, q["range"] / 2.0)),
                  key, "range")

    def build(g: _Graph, p: Dict[str, Any]) -> None:
        for centre, key in zip(centres, keys):
            g.add("equalizer", [("f", _f(centre)), ("width_type", "q"), ("w", width),
                                ("g", band_gain(key))])
        g.add("volume", [("volume", _num("master_gain", _lin))])
    return build


for _eid in GRAPHIC_EQ_CENTRES:
    _BUILDERS[_eid] = _graphic_eq_builder(_eid)


@_builder("notch_filter")
def _b_notch_filter(g: _Graph, p: Dict[str, Any]) -> None:
    for n in range(1, NOTCH_COUNT + 1):
        if not p[f"notch{n}_on"]:
            continue
        g.add("bandreject", [
            ("frequency", _num(f"notch{n}_frequency")),
            ("width_type", "q"),
            ("width", _L(lambda q: _f(NOTCH_WIDTH_Q[q["width"]]), "width")),
            # depth at the centre = 20*log10(1 - mix): mix carries the notch gain
            ("mix", _num(f"notch{n}_gain", lambda db: _clamp(1.0 - _lin(db), 0.0, 1.0))),
        ])


def _biquad(g: _Graph, filt: str, prefix: str, with_gain: bool) -> None:
    opts: List[Tuple[str, Any]] = [
        ("frequency", _num(f"{prefix}_frequency")),
        ("width_type", "q"),
        ("width", _num(f"{prefix}_q")),
    ]
    if with_gain:
        opts.append(("gain", _num(f"{prefix}_gain")))
    if filt in ("highpass", "lowpass"):
        opts.append(("poles", "2"))
    g.add(filt, opts)


@_builder("parametric_eq")
def _b_parametric_eq(g: _Graph, p: Dict[str, Any]) -> None:
    if p["hp_on"]:
        _biquad(g, "highpass", "hp", False)
    if p["ls_on"]:
        _biquad(g, "lowshelf", "ls", True)
    for n in range(1, PEQ_BANDS + 1):
        if p[f"band{n}_on"]:
            _biquad(g, "equalizer", f"band{n}", True)
    if p["hs_on"]:
        _biquad(g, "highshelf", "hs", True)
    if p["lp_on"]:
        _biquad(g, "lowpass", "lp", False)
    g.add("volume", [("volume", _num("master_gain", _lin))])


_SCI_COEFS = ("b0", "b1", "b2", "a0", "a1", "a2")
_SCI_LIVE_KEYS = ("filter_type", "cutoff", "high_cutoff", "ripple", "attenuation")


def _coef(x: float) -> str:
    """A biquad coefficient at full double precision (shortest repr that
    round-trips; ffmpeg reads the exponent form)."""
    s = repr(float(x))
    return "0" if s in ("0.0", "-0.0") else s


def _sci_sos(q: Dict[str, Any]) -> Any:
    return design_sos(q["filter_type"], q["mode"], q["cutoff"], q["high_cutoff"], q["order"],
                      q["ripple"], q["attenuation"])


@_builder("scientific_filter")
def _b_scientific_filter(g: _Graph, p: Dict[str, Any]) -> None:
    # one biquad per second-order section of the 48 kHz design, in double
    # precision; every coefficient follows type/cutoffs/ripple/attenuation live
    for k in range(section_count(p["mode"], p["order"])):
        opts: List[Tuple[str, Any]] = [
            (name, _L(lambda q, k=k, j=j: _coef(_sci_sos(q)[k][j]), *_SCI_LIVE_KEYS))
            for j, name in enumerate(_SCI_COEFS)]
        opts.append(("precision", "f64"))
        g.add("biquad", opts)
    g.add("volume", [("volume", _num("master_gain", _lin))])


# --- Modulation

@_builder("chorus")
def _b_chorus(g: _Graph, p: Dict[str, Any]) -> None:
    n = max(1, int(round(p["voices"])))
    delays = [p["delay"] + (p["spread"] * i / (n - 1) if n > 1 else 0.0) for i in range(n)]
    speeds = [_clamp(p["rate"] * (1.0 + 0.13 * i), 0.1, 20.0) for i in range(n)]
    decay = _clamp(p["wet"] / 100.0 / math.sqrt(n), 0.001, 1.0)
    g.add("chorus", [
        ("in_gain", _f(p["dry"] / 100.0)), ("out_gain", "1"),
        ("delays", "|".join(_f(d) for d in delays)),
        ("decays", "|".join(_f(decay) for _ in range(n))),
        ("speeds", "|".join(_f(s) for s in speeds)),
        ("depths", "|".join(_f(p["depth"]) for _ in range(n))),
    ])


@_builder("flanger")
def _b_flanger(g: _Graph, p: Dict[str, Any]) -> None:
    regen = p["feedback"] * (-1.0 if p["mode"] == "Inverted" else 1.0)
    g.add("flanger", [
        ("delay", _f(_clamp(p["initial_delay"], 0.0, 30.0))),
        ("depth", _f(_clamp(p["final_delay"] - p["initial_delay"], 0.0, 10.0))),
        ("regen", _f(_clamp(regen, -95.0, 95.0))),
        ("width", _f(p["mix"])),
        ("speed", _f(_clamp(p["rate"], 0.1, 10.0))),
        ("shape", "triangular" if p["mode"] == "Triangular" else "sinusoidal"),
        ("phase", _f(_clamp(p["stereo_phasing"] / 360.0 * 100.0, 0.0, 100.0))),
    ])


@_builder("phaser")
def _b_phaser(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("aphaser", [
        ("in_gain", "0.4"),
        ("out_gain", _f(_clamp(0.74 * _lin(p["output_gain"]), 0.0, 1e9))),
        ("delay", _f(_clamp(p["depth"] / 100.0 * 5.0, 0.0, 5.0))),
        ("decay", _f(_clamp(p["feedback"] / 100.0, 0.0, 0.99))),
        ("speed", _f(_clamp(p["rate"], 0.1, 2.0))),
        ("type", "sinusoidal" if p["shape"] == "Sinusoidal" else "triangular"),
    ])


# --- Noise Reduction / Restoration

@_builder("adaptive_noise_reduction")
def _b_adaptive_noise_reduction(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("afftdn", [
        ("nr", _num("reduce_by", lambda db: _clamp(db, 0.01, 97.0))),
        ("ad", _num("noisiness", lambda pct: pct / 100.0)),
        ("nf", _num("noise_floor")),
        ("rf", _num("residual_floor")),
        ("nt", AFFTDN_NOISE_TYPES[p["noise_type"]]),
        ("tn", "1"),
        ("om", _L(lambda q: "n" if q["noise_only"] else "o", "noise_only")),
    ])


@_builder("noise_reduction_rnnoise")
def _b_noise_reduction_rnnoise(g: _Graph, p: Dict[str, Any]) -> None:
    try:  # lazy: render.audio pulls in the timeline schema
        from render.audio import _ffmpeg_filter_path, rnnoise_model_path
        model = rnnoise_model_path()
    except Exception as exc:  # pragma: no cover - only on a broken install
        g.warnings.append(f"RNNoise unavailable ({exc}); effect skipped")
        return
    if model is None:
        g.warnings.append("No RNNoise model under data/models/rnnoise; effect skipped")
        return
    g.add("arnndn", [("m", _ffmpeg_filter_path(model)),
                     ("mix", _num("mix", lambda pct: pct / 100.0))])


@_builder("hiss_reduction")
def _b_hiss_reduction(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("afftdn", [
        ("nt", "w"),
        ("nf", _num("noise_floor")),
        ("nr", _num("reduce_by", lambda db: _clamp(db, 0.01, 97.0))),
    ])


# --- Reverb

def _ir_reverb(g: _Graph, ir_path: Path, weights: _L) -> None:
    """Dry/wet split around `afir`; the impulse response is loaded in the graph
    by `amovie` (main signal first, IR second). afir's own dry/wet are its
    input/output gains, not a mix, so the mix is the amix weights."""
    from render.audio import _ffmpeg_filter_path  # lazy: render.audio pulls in the schema

    dry, wet, ir, conv = g.label(), g.label(), g.label(), g.label()
    g.add("asplit", [], outs=[dry, wet])
    g.add("amovie", [("filename", _ffmpeg_filter_path(Path(ir_path)))], outs=[ir])
    g.add("afir", [("dry", "1"), ("wet", "1"), ("irnorm", "-1"), ("minp", "1024"),
                   ("maxp", "8192")], ins=[wet, ir], outs=[conv])
    g.add("amix", [("inputs", "2"), ("duration", "first"), ("normalize", "0"),
                   ("weights", weights)], ins=[dry, conv])


@_builder("convolution_reverb")
def _b_convolution_reverb(g: _Graph, p: Dict[str, Any]) -> None:
    source = reverb_ir.impulse_source(p["impulse"])
    if source is None:
        g.warnings.append(f"Impulse {p['impulse']!r} not found; using {IMPULSE_DEFAULT}")
        source = reverb_ir.impulse_source(IMPULSE_DEFAULT)
    ir = reverb_ir.prepared_ir_path(source, room_size=p["room_size"],
                                    damping_lf_hz=p["damping_lf"],
                                    damping_hf_hz=p["damping_hf"],
                                    pre_delay_ms=p["pre_delay"], width=p["width"])
    _ir_reverb(g, ir, _L(lambda q: f"{_f(1.0 - q['mix'] / 100.0)} {_f(q['mix'] / 100.0)}",
                         "mix"))
    g.add("volume", [("volume", _num("gain", _lin))])


@_builder("reverb")
def _b_reverb(g: _Graph, p: Dict[str, Any]) -> None:
    ir = reverb_ir.synthetic_ir_path(
        decay_s=p["decay"] / 1000.0, pre_delay_ms=2.0 + 0.4 * p["room_size"],
        room_size=p["room_size"], diffusion=p["diffusion"], hf_damping=p["damping"],
        lf_cut_hz=p["lf_cut"], hf_cut_hz=p["hf_cut"], width=p["width"],
        early_level=p["early_reflections"])
    _ir_reverb(g, ir, _L(lambda q: f"{_f(q['dry'] / 100.0)} {_f(q['wet'] / 100.0)}",
                         "dry", "wet"))


# --- Special

@_builder("distortion")
def _b_distortion(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("acrusher", [
        ("level_in", _num("drive", lambda db: _clamp(_lin(db), *_UNITY_64))),
        ("level_out", "1"), ("bits", "16"), ("mix", "1"), ("mode", "lin"), ("aa", "1"),
    ])
    g.add("asoftclip", [
        ("type", _L(lambda q: DISTORTION_CURVES[q["curve_type"]], "curve_type")),
        ("output", _num("output", lambda db: _clamp(_lin(db), 0.000001, 16.0))),
    ])


@_builder("mastering")
def _b_mastering(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("lowshelf", [("f", "150"), ("g", _num("low"))])
    g.add("equalizer", [("f", "1000"), ("width_type", "q"), ("w", "0.7"), ("g", _num("mid"))])
    g.add("highshelf", [("f", "5000"), ("g", _num("high"))])
    g.add("aexciter", [
        ("amount", _num("exciter", lambda pct: pct / 100.0 * 2.0)),
        ("drive", _L(lambda q: _f(EXCITER_MODES[q["exciter_mode"]][1]), "exciter_mode")),
        ("freq", _L(lambda q: _f(EXCITER_MODES[q["exciter_mode"]][0]), "exciter_mode")),
        ("ceil", "16000"),
    ])
    g.add("stereotools", [("slev", _num("widener", lambda pct: 1.0 + pct / 100.0))])
    g.add("alimiter", [
        ("level_in", _num("maximizer", lambda pct: _lin(pct / 100.0 * 12.0))),
        ("level_out", _num("output_gain", lambda db: _clamp(_lin(db), *_UNITY_64))),
        ("limit", _f(_lin(-1.0))), ("attack", "5"), ("release", "50"), ("level", "0"),
    ])


def _ve(q: Dict[str, Any], i: int) -> float:
    return VOCAL_ENHANCER_MODES[q["mode"]][i]


@_builder("vocal_enhancer")
def _b_vocal_enhancer(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("highpass", [("frequency", _L(lambda q: _f(_ve(q, 0)), "mode")),
                       ("width_type", "q"), ("width", "0.707"), ("poles", "2")])
    g.add("lowshelf", [
        ("frequency", _L(lambda q: _f(_ve(q, 1)), "mode")),
        ("width_type", "q"), ("width", "0.707"),
        ("gain", _L(lambda q: _f(-4.0 * q["amount"] / 100.0
                                 * (0.5 if q["mode"] == "Music" else 1.0)), "amount", "mode")),
    ])
    g.add("equalizer", [
        ("frequency", _L(lambda q: _f(_ve(q, 2)), "mode")),
        ("width_type", "q"), ("width", "1"),
        ("gain", _L(lambda q: _f(_ve(q, 3) * q["amount"] / 100.0), "amount", "mode")),
    ])
    g.add("adynamicequalizer", [
        ("threshold", _f(_lin(-30.0))),
        ("dfrequency", _L(lambda q: _f(_ve(q, 4)), "mode")), ("dqfactor", "2"),
        ("tfrequency", _L(lambda q: _f(_ve(q, 4)), "mode")), ("tqfactor", "2"),
        ("mode", "cutabove"), ("ratio", "0.5"), ("range", "12"),
        ("attack", "1"), ("release", "50"),
    ])
    g.add("acompressor", [
        ("threshold", _f(_comp_threshold(-18.0))),
        ("ratio", _num("amount", lambda pct: 1.0 + 1.5 * pct / 100.0)),
        ("attack", "10"), ("release", "120"), ("knee", "4"),
    ])
    g.add("aexciter", [
        ("amount", _num("amount", lambda pct: pct / 100.0)),
        ("drive", "4"),
        ("freq", _L(lambda q: _f(_ve(q, 5)), "mode")),
        ("ceil", "16000"),
    ])
    g.add("alimiter", [("limit", _f(_lin(-1.0))), ("attack", "5"), ("release", "50"),
                       ("level", "0")])


@_builder("sub_octave")
def _b_sub_octave(g: _Graph, p: Dict[str, Any]) -> None:
    dry, wet, low = g.label(), g.label(), g.label()
    g.add("asplit", [], outs=[dry, wet])
    g.add("rubberband", [("pitch", "0.5"), ("formant", "shifted")], ins=[wet])
    g.add("volume", [("volume", _num("mix", lambda pct: pct / 100.0))], outs=[low])
    g.add("amix", [("inputs", "2"), ("duration", "first"), ("normalize", "0"),
                   ("weights", "1 1")], ins=[dry, low])


# --- Stereo Imagery

@_builder("stereo_expander")
def _b_stereo_expander(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("stereotools", [("mpan", _num("center_pan", lambda pct: pct / 100.0))])
    g.add("extrastereo", [("m", _num("expand", lambda pct: pct / 100.0))])


@_builder("graphic_phase_shifter")
def _b_graphic_phase_shifter(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("aphaseshift", [
        ("shift", _num("shift", lambda deg: _clamp(deg / 180.0, -1.0, 1.0))),
        ("level", _num("level", lambda db: _clamp(_lin(db), 0.0, 1.0))),
        ("order", _num("order", lambda x: float(int(round(x))))),
    ])


# --- Time and Pitch

@_builder("pitch_shifter")
def _b_pitch_shifter(g: _Graph, p: Dict[str, Any]) -> None:
    pitchq, transients = PITCH_PRECISION[p["precision"]]
    g.add("rubberband", [
        ("pitch", _L(lambda q: _f(_clamp(2.0 ** ((q["semitones"] + q["cents"] / 100.0) / 12.0),
                                         0.01, 100.0)), "semitones", "cents")),
        ("formant", "preserved" if p["formant"] == "Preserved" else "shifted"),
        ("pitchq", pitchq),
        ("transients", transients),
    ])


def _stretch_pitch_ratio(q: Dict[str, Any]) -> float:
    if q["lock"]:
        return _clamp(100.0 / q["stretch"], 0.01, 100.0)
    return _clamp(2.0 ** (q["pitch"] / 12.0), 0.01, 100.0)


@_builder("stretch_and_pitch")
def _b_stretch_and_pitch(g: _Graph, p: Dict[str, Any]) -> None:
    g.add("rubberband", [
        ("tempo", _num("stretch", lambda pct: _clamp(100.0 / pct, 0.01, 100.0))),
        ("pitch", _L(lambda q: _f(_stretch_pitch_ratio(q)), "stretch", "pitch", "lock")),
        ("formant", "preserved" if p["preserve_formants"] else "shifted"),
    ])
    if abs(p["stretch"] - 100.0) > 1e-9:
        g.warnings.append(f"Stretch {_f(p['stretch'])}% changes the duration of this scope")


PITCH_BEND_STEP_S = 0.05          # a linear ramp is sent as steps this far apart
PITCH_BEND_MAX_COMMANDS = 20000   # beyond this the steps are spread further apart


def _bend_ratio(semitones: float) -> float:
    return _clamp(2.0 ** (semitones / 12.0), 0.01, 100.0)


def pitch_bend_commands(curve: Sequence[Sequence[float]],
                        target: str) -> Tuple[str, float, List[str]]:
    """asendcmd commands for a [time_s, semitones] curve, sent to `target`
    (`rubberband@<stem>_1`): the value is held before the first point and
    after the last, and each ramp between two points is sent as steps of
    PITCH_BEND_STEP_S. Returns (commands, initial ratio, warnings)."""
    pts = [(float(x), float(y)) for x, y in curve] or [(0.0, 0.0)]
    ramp_s = sum(x1 - x0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if y1 != y0)
    step = PITCH_BEND_STEP_S
    warnings: List[str] = []
    if ramp_s / step > PITCH_BEND_MAX_COMMANDS:
        step = ramp_s / PITCH_BEND_MAX_COMMANDS
        warnings.append(f"Pitch Bender: ramps sent every {_f(step * 1000.0)} ms "
                        f"(over {PITCH_BEND_MAX_COMMANDS} steps at 50 ms)")
    events: List[Tuple[float, float]] = [(0.0, pts[0][1])]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 == y0:
            events.append((x0, y0))
            continue
        n = max(1, int(math.ceil((x1 - x0) / step - 1e-9)))
        events += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]
    events.append((pts[-1][0], pts[-1][1]))
    cmds: List[str] = []
    last = None
    for t, st in events:
        ratio = _f(_bend_ratio(st))
        if ratio != last:
            cmds.append(f"{_f(t)} {target} pitch {ratio}")
            last = ratio
    return ";".join(cmds), _bend_ratio(pts[0][1]), warnings


@_builder("pitch_bender")
def _b_pitch_bender(g: _Graph, p: Dict[str, Any]) -> None:
    # asendcmd is instance 0 and fires on the input's time, ahead of
    # rubberband (instance 1), so each step applies where it belongs.
    # pitchq=consistency is Rubber Band's mode for pitch that changes while
    # it runs: measured 2026-10-09, audio after a 0 -> +12 st bend lands
    # 13 ms early (quality mode: 58 ms late), after 0 -> -6 st 18 ms late.
    cmds, initial, warnings = pitch_bend_commands(p["curve"], f"rubberband@{g.stem}_1")
    g.warnings.extend(warnings)
    g.add("asendcmd", [("c", cmds)])
    g.add("rubberband", [("pitch", _f(initial)), ("pitchq", "consistency")])


missing = set(EFFECTS) - set(_BUILDERS)
assert not missing, f"catalogue entries without a builder: {sorted(missing)}"
del missing


# --- public API --------------------------------------------------------------------

def _compile(effect: Any, stem: Optional[str] = None) -> CompiledEffect:
    effect_id = str(_field(effect, "id", "") or "")
    catalogue_id = _field(effect, "effect")
    edef = EFFECTS.get(catalogue_id)
    if edef is None:
        raise ValueError(f"unknown audio effect {catalogue_id!r}")
    if edef["engine"] != "ffmpeg":
        raise ValueError(f"{catalogue_id!r} is a process effect, not an ffmpeg one")
    params = _resolve_params(catalogue_id, _field(effect, "params") or {})
    g = _Graph(stem or _stem(effect_id), catalogue_id, params)
    _BUILDERS[catalogue_id](g, params)

    start, end = _field(effect, "start_s"), _field(effect, "end_s")
    if start is not None or end is not None:
        a = float(start) if start is not None else 0.0
        b = float(end) if end is not None else _FOREVER_S
        if b <= a:
            g.warnings.append(f"{edef['name']}: empty time range {_f(a)}-{_f(b)} s; skipped")
            return CompiledEffect(effect_id, [], {}, "enable", g.warnings)
        if edef["windowable"]:
            expr = f"between(t,{_f(a)},{_f(b)})"
            for node in g.nodes:
                node.opts.append(("enable", expr))
        else:
            g.warnings.append(f"{edef['name']} cannot be limited to a time range; "
                              "applied to the whole scope")

    return CompiledEffect(
        effect_id=effect_id,
        filters=[n.render() for n in g.nodes],
        commands=g.commands,
        bypass=_bypass_for([n.filter for n in g.nodes]),
        warnings=g.warnings,
    )


def compile_effect(effect: Any) -> CompiledEffect:
    """One AudioEffect (engine "ffmpeg") -> named filter instances + command map."""
    return _compile(effect)


def _join(filters: Sequence[str]) -> str:
    chain = ""
    for f in filters:
        if chain:
            # a filter that ends in an output label closes its chain segment
            chain += ";" if chain.endswith("]") else ","
        chain += f
    return chain


def compile_rack(effects: Sequence[Any], *, scope: str) -> CompiledRack:
    """A rack in order -> one filter chain. Disabled effects are skipped (their
    params need no command while they are off: `commands_for` returns [])."""
    compiled: List[CompiledEffect] = []
    live: Dict[Tuple[str, str], List[CommandSpec]] = {}
    restart: Set[Tuple[str, str]] = set()
    stems: Set[str] = set()
    for effect in effects or []:
        effect_id = str(_field(effect, "id", "") or "")
        catalogue_id = _field(effect, "effect")
        edef = EFFECTS.get(catalogue_id)
        if edef is None or edef["engine"] != "ffmpeg":
            compiled.append(CompiledEffect(effect_id, [], {}, "restart",
                                           [f"[{scope}] unknown audio effect "
                                            f"{catalogue_id!r}; skipped"]))
            continue
        if not _field(effect, "enabled", True):
            for d in edef["params"]:
                live[(effect_id, d["key"])] = []
            continue
        stem = _stem(effect_id)
        warnings: List[str] = []
        if stem in stems:
            n = 2
            while f"{stem}_dup{n}" in stems:
                n += 1
            stem = f"{stem}_dup{n}"
            warnings.append(f"[{scope}] duplicate effect id {effect_id!r}; renamed to {stem}")
        stems.add(stem)
        ce = _compile(effect, stem)
        ce.warnings[:0] = warnings
        compiled.append(ce)
        for d in edef["params"]:
            key = (effect_id, d["key"])
            if d["live"]:
                live[key] = list(ce.commands.get(d["key"], []))
            else:
                restart.add(key)
    filters = [f for ce in compiled for f in ce.filters]
    return CompiledRack(filters=filters, chain=_join(filters), effects=compiled,
                        live_params=live, restart_params=restart)


def commands_for(rack: CompiledRack, effect_id: str, param: str, value: Any,
                 params: Dict[str, Any]) -> Optional[List[Tuple[str, str, str]]]:
    """Runtime commands that apply `param = value` to the running graph, as
    [(target, option, value_str)]; None when the change needs a restart. An
    empty list means nothing audible changes (the param's section or the whole
    effect is switched off)."""
    specs = rack.live_params.get((effect_id, param))
    if specs is None:
        return None
    merged = dict(params or {})
    merged[param] = value
    return [(s.target, s.option, s.transform(value, merged)) for s in specs]


def public_catalogue() -> Dict[str, Any]:
    """catalogue.EFFECTS as plain JSON data (a deep copy; no callables)."""
    return json.loads(json.dumps(EFFECTS))
