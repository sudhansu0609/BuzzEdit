"""Analysis, noise detection, the noise print and the noise-reduction engines
(AUDIO_SUITE_PLAN.md §2.3, brief A2).

Every signal is synthetic: a 1 kHz `sine` (-21 dBFS RMS) gated off in three
2 s gaps, mixed with `anoisesrc` white noise at -50 dBFS RMS. Levels are
measured on the decoded samples, never on the creator's media.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio import analysis, deepfilter, noise
from config import FFMPEG_BIN
from store.project_store import ProjectStore
from timeline.schema import SourceFile, Timeline, TimelineItem

GAPS = [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]          # noise only
TONES = [(2.0, 4.0), (6.0, 8.0)]                      # tone + noise
NOISE_DBFS = -50.0
NOISE_AMPLITUDE = 10 ** (NOISE_DBFS / 20) * 3 ** 0.5   # anoisesrc white is uniform in +-amplitude


def _synth(path: Path) -> Path:
    gate = "+".join(f"between(t,{a:g},{b:g})" for a, b in GAPS)
    graph = (f"[0:a]volume=volume=0:enable='{gate}'[tone];"
             f"[tone][1:a]amix=inputs=2:normalize=0:duration=shortest")
    result = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=10",
         "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude={NOISE_AMPLITUDE:.6f}"
                             f":sample_rate=48000:duration=10:seed=11",
         "-filter_complex", graph, "-c:a", "pcm_f32le", str(path)],
        capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-600:]
    return path


def _level(path, start_s: float, end_s: float) -> float:
    samples, _rate = analysis.read_mono(str(path), start_s, end_s)
    return analysis.rms_dbfs(samples)


def _gap_levels(path):
    return [_level(path, a + 0.2, b - 0.2) for a, b in GAPS]


def _tone_levels(path):
    return [_level(path, a + 0.2, b - 0.2) for a, b in TONES]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return _synth(tmp_path_factory.mktemp("synth") / "synth.wav")


@pytest.fixture
def project(tmp_path, monkeypatch, synth):
    """A stub project whose A1 voice is the synthetic file, store in tmp_path."""
    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    tl = Timeline(
        sources={"main": SourceFile(id="main", path=str(synth), duration_seconds=10.0, kind="audio",
                                    width=0, height=0)},
        items=[TimelineItem(id="a1_0", track="A1", source_id="main", source_start_frame=0,
                            source_end_frame=300, timeline_start_frame=0, timeline_end_frame=300)],
        duration_frames=300, revision=2)
    data = {"id": "p1", "name": "Synthetic", "timeline": tl.model_dump()}
    store.save_project("p1", data)
    return {"id": "p1", "store": store, "dir": projects / "p1", "data": data}


def test_the_synthetic_signal_has_the_levels_the_tests_assume(synth):
    for level in _gap_levels(synth):
        assert level == pytest.approx(NOISE_DBFS, abs=0.5)
    for level in _tone_levels(synth):
        assert level == pytest.approx(-21.0, abs=0.5)


# --- analysis -------------------------------------------------------------------

def test_analyze_reports_levels_loudness_series_and_spectrum(synth, tmp_path):
    png = tmp_path / "spectrum.png"
    result = analysis.analyze(str(synth), 0.0, 10.0, png_path=png)
    assert result["peak_dbfs"] == pytest.approx(-18.0, abs=0.6)       # sine peaks at 1/8 FS
    assert -27.0 < result["rms_dbfs"] < -22.0
    assert abs(result["dc_offset"]) < 1e-3
    assert result["noise_floor_dbfs"] < -30.0
    assert -26.0 < result["integrated_lufs"] < -18.0
    assert result["true_peak_dbtp"] == pytest.approx(-18.0, abs=0.8)
    assert result["lra_lu"] >= 0.0
    series = result["loudness_series"]
    assert series["step_s"] == 0.1 and 95 <= len(series["t"]) <= 101
    assert len(series["momentary_lufs"]) == len(series["t"]) == len(series["short_term_lufs"])
    assert png.exists() and png.stat().st_size > 1000 and result["spectrum_png"] == str(png)
    freqs = np.array(result["spectrum"]["freqs_hz"])
    mean_db = np.array(result["spectrum"]["mean_db"])
    assert len(freqs) == len(mean_db) == 1025
    assert freqs[int(np.argmax(mean_db))] == pytest.approx(1000.0, abs=25.0)


def test_analyze_series_is_in_source_time(synth, tmp_path):
    # The gate switches the tone per 1024-sample frame, so stay 0.1 s inside the gap.
    result = analysis.analyze(str(synth), 4.1, 5.9, png_path=tmp_path / "s.png")
    assert result["start_s"] == 4.1 and result["end_s"] == 5.9
    assert result["loudness_series"]["t"][0] == pytest.approx(4.2, abs=0.01)
    assert result["rms_dbfs"] == pytest.approx(NOISE_DBFS, abs=0.5)


def test_spectrum_db_reads_white_noise_at_its_rms_level(synth):
    spectrum = analysis.spectrum_db(str(synth), 4.2, 5.8, nfft=2048)
    assert float(np.median(spectrum["mean_db"])) == pytest.approx(NOISE_DBFS, abs=1.0)
    assert spectrum["freqs_hz"][1] == pytest.approx(48000 / 2048, abs=0.01)


# --- detect noise ---------------------------------------------------------------

def _inside_a_gap(found) -> bool:
    return any(a <= found["start_s"] and found["end_s"] <= b for a, b in GAPS)


def test_detect_noise_picks_a_window_inside_a_gap_from_vad(project):
    found = noise.detect_noise("p1")
    assert found["method"] == "vad"
    assert found["source_id"] == "main"
    assert found["end_s"] - found["start_s"] >= 0.8 - 1e-6
    assert _inside_a_gap(found), found
    assert found["floor_dbfs"] == pytest.approx(NOISE_DBFS, abs=1.5)


def test_detect_noise_uses_the_transcript_word_gaps(project):
    words = [{"word": "tone", "start": a, "end": b} for a, b in TONES]
    data = dict(project["data"], transcript={"segments": [
        {"id": 0, "start": 2.0, "end": 8.0, "text": "tone tone", "words": words}]})
    project["store"].save_project("p1", data)
    found = noise.detect_noise("p1")
    assert found["method"] == "transcript"
    assert _inside_a_gap(found), found


def test_detect_noise_falls_back_to_the_whole_file(project):
    # Words over every second: no gap is left, so the quietest 0.8 s of the file.
    data = dict(project["data"], transcript={"segments": [
        {"id": 0, "start": 0.0, "end": 10.0, "text": "all", "words": [{"word": "all", "start": 0.0, "end": 10.0}]}]})
    project["store"].save_project("p1", data)
    found = noise.detect_noise("p1")
    assert found["method"] == "whole_file"
    assert _inside_a_gap(found), found


# --- noise print ----------------------------------------------------------------

def test_capture_print_returns_15_bands_and_writes_json_and_wav(project):
    found = noise.detect_noise("p1")
    stored = noise.capture_print("p1", None, found["start_s"], found["end_s"])
    assert len(stored["bands_db"]) == 15
    assert all(abs(b) < 2.0 for b in stored["bands_db"])               # white noise: flat
    assert stored["band_floor_db"] == pytest.approx(stored["floor_dbfs"] - 3.0, abs=1.0)
    assert stored["floor_dbfs"] == pytest.approx(NOISE_DBFS, abs=1.5)
    assert stored["nfft"] == 2048 and len(stored["freqs_hz"]) == len(stored["mean_db"]) == 1025
    for key in ("source_id", "start_s", "end_s", "created"):
        assert key in stored
    json_path = project["dir"] / "audio" / "noise_print.json"
    wav_path = project["dir"] / "audio" / "noise_print.wav"
    assert json.loads(json_path.read_text(encoding="utf-8")) == stored == noise.load_print("p1")
    info = sf.info(str(wav_path))
    assert info.samplerate == 48000 and info.channels == 1
    assert info.frames / 48000 == pytest.approx(found["end_s"] - found["start_s"], abs=0.002)


def test_capture_print_refuses_a_range_too_short(project):
    with pytest.raises(ValueError):
        noise.capture_print("p1", None, 5.0, 5.05)


def test_afftdn_band_centres_match_the_filter():
    assert noise.AFFTDN_BAND_CENTRES == [80, 120, 180, 270, 410, 620, 930, 1400, 2100, 3150,
                                         4750, 7100, 10700, 16000, 24000]


# --- afftdn with the print ------------------------------------------------------

def _run_chain(src: Path, out: Path, chain) -> Path:
    analysis.run_ffmpeg(["-y", "-loglevel", "error", "-i", str(src), "-af", ",".join(chain),
                         "-c:a", "pcm_f32le", str(out)])
    return out


def test_afftdn_chain_names_the_instance_and_sets_the_profile(project):
    stored = noise.capture_print("p1", None, 0.2, 1.8)
    chain = noise.afftdn_chain(stored, reduce_by_db=30, noise_floor_offset_db=6,
                               residual_floor_db=-40, output_noise_only=True, effect_id="fx_nr")
    assert len(chain) == 1
    text = chain[0]
    assert text.startswith("afftdn@fx_nr_0=")
    for part in ("nr=30.00", "rf=-40.00", "nt=c", "om=noise", "tn=0"):
        assert part in text
    profile = text.split("bn=")[1].split(":")[0].split("|")
    assert len(profile) == 15
    nf = float(text.split("nf=")[1].split(":")[0])
    assert nf == pytest.approx(stored["band_floor_db"] + 6.0, abs=0.01)
    assert noise.afftdn_chain(stored)[0].startswith("afftdn=")


def test_afftdn_with_the_print_lowers_the_gaps_20_db_and_keeps_the_tone(project, synth, tmp_path):
    found = noise.detect_noise("p1")
    stored = noise.capture_print("p1", None, found["start_s"], found["end_s"])
    chain = noise.afftdn_chain(stored, reduce_by_db=30, noise_floor_offset_db=6,
                               residual_floor_db=-38, output_noise_only=False, effect_id="fx_nr")
    out = _run_chain(synth, tmp_path / "afftdn.wav", chain)
    for before, after in zip(_gap_levels(synth), _gap_levels(out)):
        assert before - after >= 20.0, (before, after)
    for before, after in zip(_tone_levels(synth), _tone_levels(out)):
        assert after == pytest.approx(before, abs=2.0)


def test_afftdn_output_noise_only_is_what_it_removes(project, synth, tmp_path):
    stored = noise.capture_print("p1", None, 0.2, 1.8)
    chain = noise.afftdn_chain(stored, reduce_by_db=30, output_noise_only=True)
    out = _run_chain(synth, tmp_path / "noise_only.wav", chain)
    assert _gap_levels(out)[1] == pytest.approx(NOISE_DBFS, abs=1.5)    # the noise itself
    assert _tone_levels(out)[0] < -40.0                                # the tone is kept out


def test_rnnoise_chain_runs():
    chain = noise.rnnoise_chain(0.8)
    assert chain[0].startswith("arnndn=m='") and chain[0].endswith(":mix=0.800")
    analysis.run_ffmpeg(["-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                         "-t", "1", "-af", ",".join(chain), "-f", "null", "-"])


# --- noisereduce with the print -------------------------------------------------

def test_noisereduce_lowers_the_gaps_20_db_keeps_the_tone_and_the_length(project, synth, tmp_path):
    found = noise.detect_noise("p1")
    noise.capture_print("p1", None, found["start_s"], found["end_s"])
    print_wav = project["dir"] / "audio" / "noise_print.wav"
    out = noise.noisereduce_process(str(synth), str(tmp_path / "nr.wav"), str(print_wav),
                                    prop_decrease=1.0, n_std_thresh=1.5, freq_smooth_hz=500.0,
                                    time_smooth_ms=50.0, n_fft=1024)
    src_info, out_info = sf.info(str(synth)), sf.info(str(out))
    assert out_info.samplerate == src_info.samplerate
    assert out_info.subtype == "FLOAT"
    assert abs(out_info.frames - src_info.frames) / src_info.samplerate <= 0.001
    for before, after in zip(_gap_levels(synth), _gap_levels(out)):
        assert before - after >= 20.0, (before, after)
    for before, after in zip(_tone_levels(synth), _tone_levels(out)):
        assert after == pytest.approx(before, abs=2.0)


def test_noisereduce_output_noise_only(project, synth, tmp_path):
    noise.capture_print("p1", None, 0.2, 1.8)
    out = noise.noisereduce_process(str(synth), str(tmp_path / "removed.wav"),
                                    str(project["dir"] / "audio" / "noise_print.wav"),
                                    output_noise_only=True)
    assert _gap_levels(out)[1] == pytest.approx(NOISE_DBFS, abs=1.5)
    assert _tone_levels(out)[0] < _tone_levels(synth)[0] - 6.0


# --- DeepFilterNet --------------------------------------------------------------

@pytest.fixture(scope="module")
def deep_filter():
    try:
        return deepfilter.ensure_binary()
    except deepfilter.DeepFilterUnavailable as exc:
        pytest.skip(f"DeepFilterNet binary not available: {exc}")


def test_deepfilter_lowers_the_gaps_15_db_and_keeps_the_length(deep_filter, synth, tmp_path):
    out = deepfilter.run(str(synth), str(tmp_path / "df.wav"), attenuation_limit_db=100,
                         postfilter_beta=0.0)
    assert sf.info(str(out)).frames == sf.info(str(synth)).frames
    assert sf.info(str(out)).samplerate == 48000
    for before, after in zip(_gap_levels(synth), _gap_levels(out)):
        assert before - after >= 15.0, (before, after)


def test_deepfilter_restores_the_input_rate(deep_filter, synth, tmp_path):
    staged = analysis.decode_wav(str(synth), tmp_path / "in44.wav", 0.0, 3.0, sample_rate=44100)
    out = deepfilter.run(str(staged), str(tmp_path / "df44.wav"), attenuation_limit_db=24,
                         postfilter_beta=0.02)
    assert sf.info(str(out)).samplerate == 44100
    assert sf.info(str(out)).frames == sf.info(str(staged)).frames


def test_deepfilter_status_records_the_ladspa_probe(deep_filter):
    status = deepfilter.read_status()
    assert isinstance(status.get("ladspa_ok"), bool)
    assert deepfilter.available()
