"""Brief A3 of AUDIO_SUITE_PLAN.md: Scientific Filter, reverbs, Pitch Bender,
Sub-Octave, Voice Changer and the Enhance Speech (AI) availability check.

Synthetic signals only (sines, a synthetic vowel, anullsrc) -- never the
creator's media. Rack chains run through `-filter_complex "[0:a]<chain>[out]"`
(the chain contract of audio/compile.py).
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import soundfile as sf

import config
from audio import enhance_ai, reverb_ir
from audio.catalogue import EFFECTS, LIVE_OPTIONS, TIMELINE_FILTERS
from audio.compile import commands_for, compile_effect, compile_rack, pitch_bend_commands
from audio.processes_ext import PROCESSES_EXT, run_process_ext
from audio.scientific import design_sos, response_db, section_count
from audio.voice_changer import VOICE_PRESETS, change_voice

SR = 48000
NEW_EFFECTS = ["scientific_filter", "convolution_reverb", "reverb", "sub_octave", "pitch_bender"]


def _fx(effect: str, params: Optional[Dict[str, Any]] = None, fid: str = "fx_a3") -> Dict[str, Any]:
    return {"id": fid, "effect": effect, "params": dict(params or {}), "enabled": True}


def _ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([config.FFMPEG_BIN, "-hide_banner", "-nostats", *args],
                          capture_output=True, text=True, timeout=180)


def _render(chain: str, src_args: List[str], out: Path) -> np.ndarray:
    """Input args (lavfi or file) -> chain -> float WAV -> samples (n, ch)."""
    r = _ffmpeg(["-v", "error", "-y", *src_args, "-filter_complex",
                 f"[0:a]{chain or 'anull'}[out]", "-map", "[out]", "-c:a", "pcm_f32le", str(out)])
    assert r.returncode == 0, f"{chain}\n{r.stderr[-800:]}"
    data, rate = sf.read(str(out), dtype="float64", always_2d=True)
    assert rate == SR
    return data


def _lavfi(spec: str) -> List[str]:
    return ["-f", "lavfi", "-i", spec]


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-30))


def _peak_hz(x: np.ndarray) -> float:
    """Spectral peak with parabolic interpolation on a Hann-windowed FFT."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(spec[1:-1])) + 1
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-30)
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    return (k + delta) * SR / len(x)


def _vowel(seconds: float, f0: float = 140.0) -> np.ndarray:
    """A steady synthetic vowel: harmonics of f0 shaped by three formants."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    k = 1
    while k * f0 < 9000.0:
        f = k * f0
        env = sum(1.0 / (1.0 + ((f - F) / B) ** 2)
                  for F, B in ((700.0, 90.0), (1200.0, 110.0), (2600.0, 160.0))) + 0.01
        x += env / np.sqrt(k) * np.sin(2 * np.pi * f * t + 0.3 * k * k)
        k += 1
    return 0.3 * x / np.abs(x).max()


def _f0_median(x: np.ndarray) -> float:
    import pyworld as pw

    f0, _ = pw.harvest(np.ascontiguousarray(x, dtype=np.float64), SR, f0_floor=50.0,
                       f0_ceil=1000.0, frame_period=5.0)
    return float(np.median(f0[f0 > 0]))


def _centroid(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float((freqs * spec).sum() / spec.sum())


# --- catalogue entries --------------------------------------------------------------

def test_new_entries_in_the_catalogue():
    assert len(EFFECTS) == 40
    for eid in NEW_EFFECTS:
        assert EFFECTS[eid]["engine"] == "ffmpeg"
    assert EFFECTS["scientific_filter"]["category"] == "Filter and EQ"
    assert EFFECTS["convolution_reverb"]["category"] == "Reverb"
    assert EFFECTS["reverb"]["category"] == "Reverb"
    assert EFFECTS["sub_octave"]["category"] == "Special"
    assert EFFECTS["pitch_bender"]["category"] == "Time and Pitch"
    assert {"Vocal Ambience (Small)", "Vocal Ambience (Medium)", "Vocal Ambience (Large)",
            "Drum Plate", "Great Hall", "Horror Intimate",
            "Cave"} <= set(EFFECTS["reverb"]["presets"])
    assert "biquad" in TIMELINE_FILTERS and {"a0", "b0"} <= LIVE_OPTIONS["biquad"]
    live = {p["key"]: p["live"] for p in EFFECTS["scientific_filter"]["params"]}
    assert live == {"filter_type": True, "mode": False, "cutoff": True, "high_cutoff": True,
                    "order": False, "ripple": True, "attenuation": True, "master_gain": True}
    assert not any(p["live"] for p in EFFECTS["pitch_bender"]["params"])
    assert EFFECTS["scientific_filter"]["windowable"]
    assert not EFFECTS["pitch_bender"]["windowable"]


def _new_variants():
    for eid in NEW_EFFECTS:
        yield pytest.param(eid, {}, id=f"{eid}-defaults")
        for name, values in EFFECTS[eid]["presets"].items():
            yield pytest.param(eid, values, id=f"{eid}-{name}")


@pytest.mark.parametrize("eid,params", list(_new_variants()))
def test_new_entry_runs_on_mono_anullsrc(eid, params):
    # the A1b parametrised test runs these on stereo; this covers mono
    ce = compile_effect(_fx(eid, params))
    assert ce.filters and not ce.warnings, ce.warnings
    rack = compile_rack([_fx(eid, params)], scope="A1")
    r = _ffmpeg(["-v", "error", *_lavfi("anullsrc=r=48000:cl=mono"), "-t", "1",
                 "-filter_complex", f"[0:a]{rack.chain}[out]", "-map", "[out]", "-f", "null", "-"])
    assert r.returncode == 0, f"{rack.chain}\n{r.stderr[-800:]}"


# --- Scientific Filter ------------------------------------------------------------------

def test_section_counts():
    for ftype in ("Bessel", "Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"):
        for mode in ("Low Pass", "High Pass", "Band Pass", "Band Stop"):
            for order in (1, 4, 7, 20):
                sos = design_sos(ftype, mode, 1000.0, 4000.0, order, 1.0, 60.0)
                assert len(sos) == section_count(mode, order)
                assert np.all(np.abs(np.roots([1.0, *sos[0, 4:]])) < 1.0)


def test_butterworth_4th_order_lowpass_slope(tmp_path):
    fx = _fx("scientific_filter", {"filter_type": "Butterworth", "mode": "Low Pass",
                                   "cutoff": 1000.0, "order": 4.0})
    chain = compile_rack([fx], scope="A1").chain
    assert chain.count("biquad@fx_a3_") == 2

    def level(freq: float, filtered: bool) -> float:
        src = _lavfi(f"sine=frequency={freq}:sample_rate=48000:duration=2")
        y = _render(chain if filtered else "", src, tmp_path / f"s{freq}{filtered}.wav")
        return _rms_db(y[SR // 2:])

    ref_1k = level(1000, False)                     # the 1 kHz reference sine
    out_1k, out_2k, out_4k = level(1000, True), level(2000, True), level(4000, True)
    attenuation_4k = ref_1k - out_4k                # same input level as the reference
    assert 46.0 <= attenuation_4k <= 50.0, attenuation_4k
    assert abs((ref_1k - out_1k) - 3.01) <= 0.2     # -3 dB at the cutoff
    assert abs((out_2k - out_4k) - 24.0) <= 1.0     # -24 dB/oct between 2 and 4 kHz
    design = design_sos("Butterworth", "Low Pass", 1000.0, 4000.0, 4, 1.0, 60.0)
    assert abs(attenuation_4k + float(response_db(design, [4000.0])[0])) <= 0.2  # = the design


def test_chebyshev_i_ripple_within_spec(tmp_path):
    ripple = 1.0
    tones = np.arange(100.0, 3800.0, 200.0)              # inside a 4 kHz pass band
    t = np.arange(2 * SR) / SR
    x = sum(0.03 * np.sin(2 * np.pi * f * t + i) for i, f in enumerate(tones))
    src = tmp_path / "tones.wav"
    sf.write(str(src), x.astype(np.float32), SR, subtype="FLOAT")
    fx = _fx("scientific_filter", {"filter_type": "Chebyshev I", "mode": "Low Pass",
                                   "cutoff": 4000.0, "order": 4.0, "ripple": ripple})
    y = _render(compile_rack([fx], scope="A1").chain, ["-i", str(src)], tmp_path / "cheb.wav")[:, 0]
    seg_x, seg_y = np.fft.rfft(x[SR:]), np.fft.rfft(y[SR:])   # 1 Hz bins, tones on bins
    bins = tones.astype(int)
    gains = 20.0 * np.log10(np.abs(seg_y[bins]) / np.abs(seg_x[bins]))
    assert gains.max() <= 0.2 and gains.min() >= -ripple - 0.2, gains
    spread = gains.max() - gains.min()
    assert 0.7 * ripple <= spread <= ripple + 0.2, gains


def test_scientific_filter_live_commands():
    fx = _fx("scientific_filter", {"mode": "Band Pass", "cutoff": 300.0, "high_cutoff": 3400.0})
    rack = compile_rack([fx], scope="A1")
    cmds = commands_for(rack, "fx_a3", "cutoff", 400.0,
                        {"mode": "Band Pass", "high_cutoff": 3400.0})
    assert len(cmds) == 6 * section_count("Band Pass", 4)
    assert all(t.startswith("biquad@fx_a3_") for t, _, _ in cmds)
    assert commands_for(rack, "fx_a3", "order", 6.0, {}) is None
    assert commands_for(rack, "fx_a3", "mode", "Low Pass", {}) is None


# --- reverbs --------------------------------------------------------------------------------

def test_synthetic_ir_shape_energy_width():
    ir = reverb_ir.synthetic_ir(1.0, 10.0, 50.0, 80.0, 50.0, 100.0, 8000.0, 100.0)
    assert ir.ndim == 2 and ir.shape[1] == 2 and ir.dtype == np.float32
    assert abs(float(np.mean(np.sum(ir.astype(np.float64) ** 2, axis=0))) - 1.0) < 1e-3
    assert not np.any(ir[: int(0.009 * SR)])                     # pre-delay is silence
    mono = reverb_ir.synthetic_ir(1.0, 10.0, 50.0, 80.0, 50.0, 100.0, 8000.0, 0.0)
    assert np.allclose(mono[:, 0], mono[:, 1])
    e = ir[:, 0].astype(np.float64) ** 2
    early, late = e[int(0.1 * SR):int(0.2 * SR)].sum(), e[int(0.6 * SR):int(0.7 * SR)].sum()
    assert 10.0 * np.log10(early / late) == pytest.approx(30.0, abs=8.0)   # ~60 dB per second


def test_ir_library_lists_user_wavs(tmp_path, monkeypatch):
    monkeypatch.setattr(reverb_ir, "USER_DIR", tmp_path / "user")
    assert reverb_ir.ir_library() == [] and (tmp_path / "user").is_dir()
    sf.write(str(tmp_path / "user" / "Church.wav"), np.zeros(SR // 2, dtype=np.float32), SR)
    (tmp_path / "user" / "broken.wav").write_bytes(b"not a wav")
    lib = reverb_ir.ir_library()
    assert [(i["name"], i["seconds"]) for i in lib] == [("Church", 0.5)]
    assert "User: Church" in reverb_ir.impulse_choices()


def test_convolution_reverb_leaves_a_tail(tmp_path):
    rack = compile_rack([_fx("convolution_reverb")], scope="A1")
    assert "amovie@fx_a3_1" in rack.chain and "afir@fx_a3_2" in rack.chain
    y = _render(rack.chain, _lavfi("sine=frequency=1000:sample_rate=48000:duration=0.5,"
                                   "apad=whole_dur=1.5"), tmp_path / "tail.wav")
    assert len(y) == int(1.5 * SR)
    tone = _rms_db(y[int(0.1 * SR):int(0.5 * SR)])
    tail = _rms_db(y[int(0.5 * SR):int(0.8 * SR)])
    assert -50.0 < tail < tone, (tone, tail)
    dry = _render("", _lavfi("sine=frequency=1000:sample_rate=48000:duration=0.5,"
                             "apad=whole_dur=1.5"), tmp_path / "dry.wav")
    assert _rms_db(dry[int(0.5 * SR):int(0.8 * SR)]) < -100.0


def test_reverb_dry_wet_are_live():
    rack = compile_rack([_fx("reverb")], scope="A1")
    cmds = commands_for(rack, "fx_a3", "wet", 50.0, {})
    assert cmds == [("amix@fx_a3_3", "weights", "1 0.5")]
    assert commands_for(rack, "fx_a3", "decay", 3000.0, {}) is None


# --- time and pitch ---------------------------------------------------------------------------

def test_pitch_shifter_plus_4_semitones(tmp_path):
    rack = compile_rack([_fx("pitch_shifter", {"semitones": 4.0})], scope="A1")
    y = _render(rack.chain, _lavfi("sine=frequency=440:sample_rate=48000:duration=2"),
                tmp_path / "p4.wav")[:, 0]
    assert abs(len(y) - 2 * SR) <= int(0.005 * SR)
    peak = _peak_hz(y[SR // 4: -SR // 4])
    expect = 440.0 * 2.0 ** (4.0 / 12.0)
    assert abs(peak / expect - 1.0) <= 0.02, peak


def test_pitch_bender_compiles_to_asendcmd_and_bends(tmp_path):
    curve = [[0.0, 0.0], [0.6, 0.0], [0.7, 12.0], [2.0, 12.0]]
    ce = compile_effect(_fx("pitch_bender", {"curve": curve}))
    assert ce.filters[0].startswith("asendcmd@fx_a3_0=c='0 rubberband@fx_a3_1 pitch 1;")
    assert ce.filters[1].startswith("rubberband@fx_a3_1=")
    assert ce.bypass == "restart" and not ce.commands
    cmds, initial, warnings = pitch_bend_commands(curve, "rubberband@x_1")
    times = [float(c.split()[0]) for c in cmds.split(";")]
    assert initial == 1.0 and not warnings
    assert times == pytest.approx([0.0, 0.65, 0.7]), cmds        # a 100 ms ramp in 50 ms steps
    chain = compile_rack([_fx("pitch_bender", {"curve": curve})], scope="A1").chain
    y = _render(chain, _lavfi("sine=frequency=440:sample_rate=48000:duration=2"),
                tmp_path / "bend.wav")[:, 0]
    # Rubber Band's latency follows the ratio, so a bend moves the stream end
    # (measured -74 ms after +12 st); the audio itself stays within ~20 ms
    assert abs(len(y) - 2 * SR) <= int(0.1 * SR)
    assert abs(_peak_hz(y[int(0.1 * SR):int(0.5 * SR)]) / 440.0 - 1.0) <= 0.02
    assert abs(_peak_hz(y[int(1.0 * SR):int(1.7 * SR)]) / 880.0 - 1.0) <= 0.02


def test_pitch_bender_keeps_the_timing(tmp_path):
    onsets = np.arange(0.1, 2.9, 0.2)
    x = np.zeros(3 * SR)
    tb = np.arange(int(0.02 * SR)) / SR
    for o in onsets:
        i = int(o * SR)
        x[i:i + len(tb)] = 0.5 * np.sin(2 * np.pi * 300.0 * tb) * np.hanning(len(tb))
    src = tmp_path / "bursts.wav"
    sf.write(str(src), x.astype(np.float32), SR, subtype="FLOAT")

    def centres(y: np.ndarray) -> np.ndarray:
        out = []
        for o in onsets:
            a, b = int((o - 0.05) * SR), int((o + 0.12) * SR)
            e = y[a:b] ** 2
            out.append((np.arange(a, b) * e).sum() / e.sum() / SR)
        return np.array(out)

    for curve in ([[0.0, 0.0], [1.0, 0.0], [1.1, 12.0]], [[0.0, 0.0], [1.0, 0.0], [1.1, -6.0]]):
        chain = compile_rack([_fx("pitch_bender", {"curve": curve})], scope="A1").chain
        y = _render(chain, ["-i", str(src)], tmp_path / "b.wav")[:, 0]
        drift_ms = (centres(y) - centres(x)) * 1000.0
        assert np.all(np.abs(drift_ms) <= 25.0), (curve, np.round(drift_ms, 1))


def test_sub_octave_adds_the_octave_below(tmp_path):
    rack = compile_rack([_fx("sub_octave", {"mix": 50.0})], scope="A1")
    assert commands_for(rack, "fx_a3", "mix", 80.0, {}) == [("volume@fx_a3_2", "volume", "0.8")]
    y = _render(rack.chain, _lavfi("sine=frequency=440:sample_rate=48000:duration=2"),
                tmp_path / "sub.wav")[:, 0]
    seg = y[SR // 2: SR // 2 + SR]
    power = np.abs(np.fft.rfft(seg)) ** 2                         # 1 Hz bins
    low, fundamental = power[200:241].sum(), power[420:461].sum()
    assert 10.0 * np.log10(low / fundamental) > -12.0             # the octave below is there


# --- Voice Changer ----------------------------------------------------------------------

def test_change_voice_formant_keeps_f0_and_raises_centroid(tmp_path):
    x = _vowel(1.5)
    src, out = tmp_path / "vowel.wav", tmp_path / "formant.wav"
    sf.write(str(src), x.astype(np.float32), SR, subtype="FLOAT")
    change_voice(src, out, 0.0, 1.3)
    y, rate = sf.read(str(out), dtype="float64")
    assert rate == SR and abs(len(y) - len(x)) <= int(0.001 * SR)
    f0_in, f0_out = _f0_median(x), _f0_median(y)
    assert abs(f0_out / f0_in - 1.0) <= 0.02, (f0_in, f0_out)
    assert _centroid(y) > _centroid(x) * 1.15, (_centroid(x), _centroid(y))


def test_change_voice_pitch_and_speed(tmp_path):
    x = _vowel(1.0)
    src = tmp_path / "vowel.wav"
    sf.write(str(src), x.astype(np.float32), SR, subtype="FLOAT")
    change_voice(src, tmp_path / "p.wav", 4.0, 1.0)
    y, _ = sf.read(str(tmp_path / "p.wav"), dtype="float64")
    assert len(y) == len(x)
    assert abs(_f0_median(y) / (140.0 * 2 ** (4 / 12)) - 1.0) <= 0.02
    change_voice(src, tmp_path / "s.wav", 0.0, 1.0, speed=1.25)
    y, _ = sf.read(str(tmp_path / "s.wav"), dtype="float64")
    assert abs(len(y) - len(x) / 1.25) <= int(0.005 * SR)


def test_voice_presets_shape_and_racks_run():
    assert {"Deeper", "Lighter", "Giant", "Child", "Robot", "Alien", "Telephone", "Radio",
            "Megaphone", "Whisper", "Ghost", "Demon", "Monster"} == set(VOICE_PRESETS)
    for name, preset in VOICE_PRESETS.items():
        assert set(preset) == {"pitch_semitones", "formant_ratio", "speed", "fx_rack"}
        for fx in preset["fx_rack"]:
            assert fx["effect"] in EFFECTS and set(fx) >= {"id", "effect", "params", "enabled"}
        rack = compile_rack(preset["fx_rack"], scope="process")
        assert not any(w for ce in rack.effects for w in ce.warnings), name
        r = _ffmpeg(["-v", "error", *_lavfi("anullsrc=r=48000:cl=mono"), "-t", "1",
                     "-filter_complex", f"[0:a]{rack.chain or 'anull'}[out]", "-map", "[out]",
                     "-f", "null", "-"])
        assert r.returncode == 0, (name, r.stderr[-600:])


def test_process_defs_and_runner(tmp_path):
    vc = PROCESSES_EXT["voice_changer"]
    assert vc["engine"] == "process" and vc["available"] is True and not vc["windowable"]
    assert [p["key"] for p in vc["params"]] == ["pitch", "formant", "speed", "preset"]
    assert set(vc["presets"]) == set(VOICE_PRESETS)
    json.dumps(PROCESSES_EXT)                                     # JSON-safe
    x = _vowel(1.0)
    src = tmp_path / "vowel.wav"
    sf.write(str(src), x.astype(np.float32), SR, subtype="FLOAT")
    seen = []
    out = run_process_ext("voice_changer", src, tmp_path / "monster.wav",
                          vc["presets"]["Monster"], lambda f, m: seen.append(f))
    y, rate = sf.read(str(out), dtype="float64")
    expect = len(x) / VOICE_PRESETS["Monster"]["speed"]
    assert rate == SR and abs(len(y) - expect) <= int(0.005 * SR)
    assert -40.0 < _rms_db(y) < 0.0 and seen and seen[-1] == 1.0
    with pytest.raises(ValueError):
        run_process_ext("nope", src, tmp_path / "x.wav", {})


# --- Enhance Speech (AI) ------------------------------------------------------------------

def test_enhance_status_matches_available():
    assert enhance_ai.STATUS_PATH.exists(), "run `python -m audio.enhance_ai` once"
    data = json.loads(enhance_ai.STATUS_PATH.read_text(encoding="utf-8"))
    assert data["engine"] in ("voicefixer", "resemble_enhance", "none") and "error" in data
    assert enhance_ai.available() == (data["engine"] != "none")
    assert PROCESSES_EXT["enhance_speech_ai"]["available"] == enhance_ai.available()
