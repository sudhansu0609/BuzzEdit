"""Audio effect catalogue + ffmpeg compiler (AUDIO_SUITE_PLAN.md brief A1b).

Every catalogue entry at its defaults, with every section switched on, and
with every shipped preset compiles and runs through a real ffmpeg on one
second of `anullsrc`; every live parameter yields runtime commands that ffmpeg
accepts and that reproduce a full recompile exactly; a Parametric EQ band, the
Graphic EQ bands, the FFT Filter's Telephone preset and the Mastering low band
are measured on tones, and the EQs' onset delay on a tone after silence.
Synthetic signals only -- never the creator's media.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import config
from audio.catalogue import CATEGORIES, EFFECTS, TIMELINE_FILTERS
from audio.compile import (
    CompiledEffect,
    CompiledRack,
    commands_for,
    compile_effect,
    compile_rack,
    public_catalogue,
)


@dataclass
class FX:
    """Duck-typed stand-in for timeline.schema.AudioEffect."""
    id: str
    effect: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    start_s: Optional[float] = None
    end_s: Optional[float] = None


def _ffmpeg(args: List[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([config.FFMPEG_BIN, "-hide_banner", "-nostats", *args],
                          capture_output=True, text=True, cwd=cwd, timeout=120)


def _run_chain(chain: str, source: str = "anullsrc=r=48000:cl=stereo", seconds: float = 1.0,
               verbose: bool = False, cwd=None) -> subprocess.CompletedProcess:
    # a rack chain may hold `;` segments and amovie sources: -filter_complex only
    return _ffmpeg(["-v", "verbose" if verbose else "error", "-f", "lavfi", "-i", source,
                    "-t", str(seconds), "-filter_complex", f"[0:a]{chain or 'anull'}[out]",
                    "-map", "[out]", "-f", "null", "-"], cwd=cwd)


def _all_on(eid: str) -> Dict[str, Any]:
    return {p["key"]: True for p in EFFECTS[eid]["params"] if p["type"] == "bool"}


def _variants():
    for eid, d in EFFECTS.items():
        yield pytest.param(eid, {}, id=f"{eid}-defaults")
        yield pytest.param(eid, _all_on(eid), id=f"{eid}-all-on")
        for name, values in d["presets"].items():
            yield pytest.param(eid, values, id=f"{eid}-{name}")


def _changed(pdef: Dict[str, Any], current: Any) -> Any:
    """A valid value different from `current`, for a live-parameter change."""
    t = pdef["type"]
    if t == "bool":
        return not current
    if t == "choice":
        return next(c for c in pdef["choices"] if c != current)
    if t == "number":
        lo, hi = pdef["min"], pdef["max"]
        for frac in (0.37, 0.61, 0.13):
            v = round(lo + (hi - lo) * frac, 2)
            if abs(v - current) > 1e-6:
                return v
    if t == "curve":
        return [[pdef["x_min"], -6.0], [1000.0, 3.0], [pdef["x_max"], -6.0]]
    if t == "bands":
        return [{"threshold": -30.0 + i, "gain": 2.0 - i, "ratio": 5.0, "attack": 5.0,
                 "release": 200.0} for i in range(len(pdef["default"]))]
    raise AssertionError(pdef)


def _defaults(eid: str) -> Dict[str, Any]:
    return {p["key"]: p["default"] for p in EFFECTS[eid]["params"]}


_LABELS_HEAD = re.compile(r"^(\[[^\]]*\])+")
_LABELS_TAIL = re.compile(r"(\[[^\]]*\])+$")


def _parse_filters(filters: List[str]) -> Dict[str, Dict[str, str]]:
    """{instance: {option: raw value}} from rendered filter strings."""
    out: Dict[str, Dict[str, str]] = {}
    for s in filters:
        s = _LABELS_TAIL.sub("", _LABELS_HEAD.sub("", s))
        name, _, args = s.partition("=")
        parts, cur, quoted = [], "", False
        for ch in args:
            if ch == "'":
                quoted = not quoted
            if ch == ":" and not quoted:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        if cur:
            parts.append(cur)
        opts = {}
        for part in parts:
            k, _, v = part.partition("=")
            opts[k] = v[1:-1] if len(v) >= 2 and v[0] == v[-1] == "'" else v
        out[name] = opts
    return out


_REPLY = re.compile(r"Command reply for command #\d+: ret:(.*?) res:")
_OK = ("Success", "Error number 0 occurred")


# --- the catalogue -----------------------------------------------------------------

def test_catalogue_shape():
    assert len(EFFECTS) == 40
    order = [CATEGORIES.index(d["category"]) for d in EFFECTS.values()]
    assert order == sorted(order), "EFFECTS must follow the menu order"
    for eid, d in EFFECTS.items():
        assert d["id"] == eid and d["engine"] == "ffmpeg" and d["name"] and d["description"]
        assert isinstance(d["windowable"], bool)
        keys = {p["key"] for p in d["params"]}
        for p in d["params"]:
            for k in ("key", "label", "type", "min", "max", "step", "default", "unit", "live"):
                assert k in p, (eid, p.get("key"), k)
            assert p["type"] in ("number", "choice", "bool", "curve", "bands")
            if p["type"] == "choice":
                assert p["default"] in p["choices"]
            if p["type"] == "number":
                assert p["min"] <= p["default"] <= p["max"]
        for name, values in d["presets"].items():
            assert set(values) <= keys, (eid, name)


def test_public_catalogue_is_json():
    data = public_catalogue()
    text = json.dumps(data)
    assert json.loads(text) == data
    assert list(data) == list(EFFECTS)
    assert data["parametric_eq"]["params"][0]["key"] == "hp_on"
    data["amplify"]["name"] = "changed"
    assert EFFECTS["amplify"]["name"] == "Amplify", "public_catalogue must be a copy"


# --- every entry compiles and runs ---------------------------------------------------

@pytest.mark.parametrize("eid,params", list(_variants()))
def test_effect_runs_through_ffmpeg(eid, params):
    ce = compile_effect(FX("fx_t1", eid, params))
    assert isinstance(ce, CompiledEffect) and ce.filters, ce.warnings
    rack = compile_rack([FX("fx_t1", eid, params)], scope="A1")
    r = _run_chain(rack.chain)
    assert r.returncode == 0, f"{rack.chain}\n{r.stderr[-800:]}"


@pytest.mark.parametrize("eid", list(EFFECTS))
def test_instance_names(eid):
    ce = compile_effect(FX("fx_7f3a", eid, _all_on(eid)))
    names = []
    for k, s in enumerate(ce.filters):
        m = re.match(r"^(?:\[[^\]]+\])*([a-z0-9_]+)@fx_7f3a_(\d+)(?:=|\[|$)", s)
        assert m, s
        assert int(m.group(2)) == k, s
        names.append(f"{m.group(1)}@fx_7f3a_{k}")
    for specs in ce.commands.values():
        for spec in specs:
            assert spec.target in names


# --- live parameters ------------------------------------------------------------------

@pytest.mark.parametrize("eid", list(EFFECTS))
def test_live_flags_match_commands(eid):
    for params, sections_on in (({}, False), (_all_on(eid), True)):
        rack = compile_rack([FX("fx_l", eid, params)], scope="A1")
        current = {**_defaults(eid), **params}
        for p in EFFECTS[eid]["params"]:
            cmds = commands_for(rack, "fx_l", p["key"], _changed(p, current[p["key"]]), current)
            if p["live"]:
                assert cmds is not None, (eid, p["key"])
                if sections_on:
                    assert cmds, (eid, p["key"], "live param without a command")
            else:
                assert cmds is None, (eid, p["key"])


def _live_changes(eid: str):
    params = {**_defaults(eid), **_all_on(eid)}
    return params, [(p["key"], _changed(p, params[p["key"]]))
                    for p in EFFECTS[eid]["params"] if p["live"]]


_LIVE_EFFECTS = [eid for eid in EFFECTS if any(p["live"] for p in EFFECTS[eid]["params"])]


@pytest.mark.parametrize("eid", _LIVE_EFFECTS)
def test_live_commands_reproduce_a_recompile(eid):
    params, changes = _live_changes(eid)
    rack = compile_rack([FX("fx_c", eid, params)], scope="A1")
    for key, value in changes:
        applied = _parse_filters(rack.filters)
        for target, option, val in commands_for(rack, "fx_c", key, value, params):
            applied[target][option] = val
        fresh = compile_rack([FX("fx_c", eid, {**params, key: value})], scope="A1")
        assert applied == _parse_filters(fresh.filters), (eid, key)


@pytest.mark.parametrize("eid", _LIVE_EFFECTS)
def test_live_commands_accepted_by_ffmpeg(eid, tmp_path):
    params, changes = _live_changes(eid)
    rack = compile_rack([FX("fx_c", eid, params)], scope="A1")
    cmds = []
    for key, value in changes:
        cmds += commands_for(rack, "fx_c", key, value, params)
    assert cmds
    (tmp_path / "cmds.txt").write_text(
        "0.2 " + ", ".join(f"{t} {o} '{v}'" for t, o, v in cmds) + ";\n", encoding="utf-8")
    r = _run_chain("asendcmd=f=cmds.txt," + rack.chain, verbose=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    replies = _REPLY.findall(r.stderr)
    assert len(replies) == len(cmds), (cmds, replies)
    bad = [(c, ret) for c, ret in zip(cmds, replies) if ret not in _OK]
    assert not bad, bad


# --- measurements, windows, bypass, racks ------------------------------------------------

def _rms_db(chain: str) -> float:
    r = _ffmpeg(["-v", "info", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000",
                 "-t", "2", "-af",
                 (chain + "," if chain else "")
                 + "astats=metadata=0:measure_perchannel=none:measure_overall=RMS_level",
                 "-f", "null", "-"])
    assert r.returncode == 0, r.stderr[-800:]
    return float(re.findall(r"RMS level dB: (-?[0-9.]+)", r.stderr)[-1])


def test_parametric_eq_band_measures_plus_6_db():
    fx = FX("fx_eq", "parametric_eq", {"band3_frequency": 1000.0, "band3_gain": 6.0,
                                       "band3_q": 1.0})
    rack = compile_rack([fx], scope="A1")
    gain = _rms_db(rack.chain) - _rms_db("")
    assert abs(gain - 6.0) <= 0.5, gain


def _tone_db(chain: str, frequency: float) -> float:
    """RMS level (dBFS) of a 2 s, 48 kHz stereo tone through `chain`. Stereo on
    both sides of a comparison: through a filter that forces stereo
    (stereotools), a mono tone would read ffmpeg's -3 dB upmix as gain."""
    r = _ffmpeg(["-v", "info", "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000",
                 "-t", "2", "-filter_complex",
                 f"[0:a]pan=stereo|c0=c0|c1=c0,{chain + ',' if chain else ''}"
                 "astats=metadata=0:measure_perchannel=none:measure_overall=RMS_level[out]",
                 "-map", "[out]", "-f", "null", "-"])
    assert r.returncode == 0, r.stderr[-800:]
    return float(re.findall(r"RMS level dB: (-?[0-9.]+)", r.stderr)[-1])


def _tone_gain_db(eid: str, params: Dict[str, Any], frequency: float) -> float:
    """What one effect does to a tone, in dB."""
    rack = compile_rack([FX("fx_m", eid, params)], scope="A1")
    return _tone_db(rack.chain, frequency) - _tone_db("", frequency)


def _onset_s(chain: str) -> float:
    """When a 1 kHz tone that starts after 0.5 s of silence first comes out of
    `chain` above -40 dBFS (silencedetect's first silence_end)."""
    r = _ffmpeg(["-v", "info", "-f", "lavfi", "-i",
                 "sine=frequency=1000:sample_rate=48000:duration=1.5", "-filter_complex",
                 f"[0:a]pan=stereo|c0=c0|c1=c0,adelay=500:all=1,{chain},"
                 "silencedetect=n=-40dB:d=0.005[out]", "-map", "[out]", "-f", "null", "-"])
    assert r.returncode == 0, r.stderr[-800:]
    return float(re.findall(r"silence_end: (-?[0-9.]+)", r.stderr)[0])


def test_graphic_eq_10_bands_measure_plus_6_db():
    """A band at +6 dB lifts its own tone by 6 dB and leaves a far one alone."""
    gain = _tone_gain_db("graphic_eq_10", {"band_6": 6.0}, 1000)      # 1 kHz band
    assert abs(gain - 6.0) <= 0.3, gain
    gain = _tone_gain_db("graphic_eq_10", {"band_6": 6.0}, 63)
    assert abs(gain) <= 0.3, gain
    gain = _tone_gain_db("graphic_eq_10", {"band_2": 6.0}, 63)        # 63 Hz band
    assert abs(gain - 6.0) <= 0.3, gain


def test_fft_filter_telephone_preset_is_a_band_pass():
    """scale=loglog cut 1 kHz by 48 dB here; linlog passes 300-3400 Hz."""
    tel = EFFECTS["fft_filter"]["presets"]["Telephone"]
    for frequency in (1000, 2000):
        gain = _tone_gain_db("fft_filter", tel, frequency)
        assert abs(gain) <= 1.0, (frequency, gain)
    gain = _tone_gain_db("fft_filter", tel, 8000)
    assert gain <= -40.0, gain


def test_mastering_low_band_measures_plus_6_db():
    gain = _tone_gain_db("mastering", {"low": 6.0}, 63)
    assert 5.0 <= gain <= 6.5, gain


@pytest.mark.parametrize("eid,params", [
    ("graphic_eq_10", {"band_6": 6.0}),
    ("fft_filter", EFFECTS["fft_filter"]["presets"]["Telephone"]),
])
def test_eq_adds_no_delay_against_the_picture(eid, params):
    rack = compile_rack([FX("fx_o", eid, params)], scope="A1")
    onset = _onset_s(rack.chain)
    assert abs(onset - 0.5) <= 0.002, onset


def test_graphic_eqs_fft_filter_and_mastering_are_live():
    """Every parameter of these effects is live and yields commands; that ffmpeg
    accepts them is test_live_commands_accepted_by_ffmpeg's job."""
    for eid in ("graphic_eq_10", "graphic_eq_20", "graphic_eq_30"):
        assert "accuracy" not in {p["key"] for p in EFFECTS[eid]["params"]}, eid
        assert EFFECTS[eid]["windowable"], eid
    for eid in ("graphic_eq_10", "graphic_eq_20", "graphic_eq_30", "fft_filter", "mastering"):
        rack = compile_rack([FX("fx_l", eid)], scope="A1")
        current = _defaults(eid)
        for p in EFFECTS[eid]["params"]:
            assert p["live"], (eid, p["key"])
            assert commands_for(rack, "fx_l", p["key"], _changed(p, current[p["key"]]),
                                current), (eid, p["key"])


def test_window_becomes_enable_between():
    rack = compile_rack([FX("fx_w", "parametric_eq", {"band3_gain": 4.0}, start_s=0.25,
                            end_s=0.75)], scope="A1")
    assert rack.filters and all("enable='between(t,0.25,0.75)'" in f for f in rack.filters)
    assert _run_chain(rack.chain).returncode == 0
    ce = compile_effect(FX("fx_n", "single_band_compressor", start_s=0.25, end_s=0.75))
    assert not EFFECTS["single_band_compressor"]["windowable"]
    assert not any("enable=" in f for f in ce.filters) and ce.warnings


def test_bypass_rules():
    expect = {"amplify": "enable", "parametric_eq": "enable", "deesser": "enable",
              "single_band_compressor": "mix", "vocal_enhancer": "mix",
              "graphic_eq_10": "enable", "fft_filter": "flat", "mastering": "enable",
              "channel_mixer": "restart", "echo": "restart", "pitch_shifter": "restart",
              "multiband_compressor": "restart"}
    for eid, bypass in expect.items():
        assert compile_effect(FX("fx_b", eid)).bypass == bypass, eid
    for eid, d in EFFECTS.items():
        ce = compile_effect(FX("fx_b", eid, _all_on(eid)))
        names = [re.sub(r"^(\[[^\]]+\])*", "", f).split("@")[0] for f in ce.filters]
        if d["windowable"]:
            assert all(n in TIMELINE_FILTERS for n in names)
            assert ce.bypass == "enable", eid


def test_rack_skips_disabled_and_joins_labelled_segments():
    rack = compile_rack([
        FX("fx_1", "amplify", {"gain": 3.0}),
        FX("fx_2", "delay"),
        FX("fx_3", "hard_limiter", enabled=False),
        {"id": "fx_4", "effect": "amplify", "params": {"gain": -3.0}},
    ], scope="MASTER")
    assert isinstance(rack, CompiledRack)
    assert [ce.effect_id for ce in rack.effects] == ["fx_1", "fx_2", "fx_4"]
    assert not any("fx_3" in f for f in rack.filters)
    assert "[fx_2_o1]adelay@fx_2_1" in rack.chain and ";[fx_2_o0][fx_2_o2]amix@fx_2_2" in rack.chain
    assert _run_chain(rack.chain).returncode == 0
    assert commands_for(rack, "fx_3", "max_amplitude", -3.0, {}) == []
    assert commands_for(rack, "fx_1", "gain", 6.0, {}) == [("volume@fx_1_0", "volume", "1.995262")]
    assert commands_for(rack, "nope", "gain", 6.0, {}) is None
    assert compile_rack([], scope="A1").chain == ""
