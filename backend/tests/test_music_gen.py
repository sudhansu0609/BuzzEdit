from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from config import FFMPEG_BIN
from presentation import music_gen
from presentation.sound import ACT_MUSIC


def _burst_track(path: Path, rate_hz: float, duration: float = 20.0, sr: int = 22050,
                 freq: float = 1000.0, burst_len_s: float = 0.02, decay: float = 80.0) -> Path:
    """A periodic train of short decaying sine bursts at `rate_hz` events/s --
    clean enough for onset detection and a tempogram to read back reliably,
    used as a click track (rate_hz = bpm/60) and as a sparse "note" sequence
    (a low rate_hz, e.g. 0.8) alike."""
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    burst_len = int(burst_len_s * sr)
    t = np.arange(burst_len) / sr
    envelope = np.exp(-t * decay).astype(np.float32)
    burst = (np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)
    position = 0.0
    while position < duration:
        start = int(position * sr)
        end = min(start + burst_len, n)
        y[start:end] += burst[: end - start]
        position += 1.0 / rate_hz
    sf.write(str(path), y, sr)
    return path


def test_every_genre_has_a_style_and_every_style_is_instrumental():
    for genre, style in music_gen.DEFAULT_STYLE_FOR_GENRE.items():
        assert style in music_gen.MUSIC_STYLES, genre
    for name, spec in music_gen.MUSIC_STYLES.items():
        assert "instrumental" in spec["tags"], name
        assert spec["genres"] and spec["moods"], name


def test_style_for_follows_the_act_mood():
    moods = {m for m, _, _ in ACT_MUSIC.values()}
    for mood in moods:
        assert music_gen.style_for("horror", mood) in music_gen.MUSIC_STYLES
    assert music_gen.style_for("horror") == "horror_dark"
    assert music_gen.style_for("vlog", "upbeat") in ("happy_flute", "morning_raga_flute", "motivational")


def test_generated_tracks_are_licensed():
    assert music_gen.GENERATED_LICENSE in music_gen.ALLOWED_LICENSES


def test_every_builder_mood_maps_to_an_act_mood():
    act_moods = {m for m, _, _ in ACT_MUSIC.values()}
    assert set(music_gen.MOODS.values()) <= act_moods


def test_compose_tags_swaps_the_preset_tempo_and_dedupes():
    tags = music_gen.compose_tags("lofi_calm", instruments=["piano", "Tabla"],
                                  moods=["dreamy"], bpm=92, extra_tags="rain, lo-fi")
    parts = [t.strip() for t in tags.split(",")]
    assert "92 bpm" in parts and "80 bpm" not in parts
    assert parts.count("lo-fi") == 1 and "Tabla" in parts and "rain" in parts
    assert parts.count("instrumental") == 1 and parts[-1] == "instrumental"


def test_picked_instrument_replaces_the_presets_instruments():
    # The reported bug: Sad violin + violin + sad came out as piano music.
    prompt = music_gen.compose_prompt("sad_violin", instruments=["violin"], moods=["sad"])
    parts = [t.strip() for t in prompt["tags"].split(",")]
    assert parts[0] == "solo violin"
    assert not any(w in prompt["tags"] for w in ("piano", "cello", "strings"))
    assert {"soft piano", "cello", "strings"} <= set(prompt["dropped"])
    assert "melancholic" in parts   # same act mood as "sad": kept


def test_a_picked_mood_drops_conflicting_preset_moods_only():
    prompt = music_gen.compose_prompt("happy_flute", moods=["sad"])
    assert prompt["tags"].startswith("sad, ")
    assert "happy" in prompt["dropped"] and "flute melody" in prompt["tags"]


def test_picks_lead_the_prompt():
    tags = music_gen.compose_tags("horror_climax", music_genres=["orchestral"], bpm=120)
    assert tags.startswith("orchestral, horror")
    assert "120 bpm" in tags and "90 bpm" not in tags


def test_compose_tags_builds_a_custom_recipe_without_a_preset():
    tags = music_gen.compose_tags(None, ["sufi"], ["sarangi"], ["sad"], 400)
    assert tags == f"sufi, solo sarangi, sad, {music_gen.MAX_BPM} bpm, instrumental"


def test_engines_cover_known_workflow_roles_and_licences():
    for name, eng in music_gen.ENGINES.items():
        assert eng["role"] in ("music", "music15", "audio"), name
        assert eng["license"] in music_gen.ALLOWED_LICENSES, name
    assert music_gen.DEFAULT_ENGINE in music_gen.ENGINES
    assert music_gen.DEFAULT_ENGINE == "ace_step15"
    assert music_gen.ENGINES["stable_audio"]["max_seconds"] <= 47


def test_custom_meta_files_under_the_act_mood():
    meta = music_gen._generation_meta(None, ["lo-fi"], ["piano"], ["dreamy", "sad"], "vlog")
    assert meta["folder"] == "vlog" and meta["moods"] == ["calm", "soft"]
    assert meta["label"] == "Custom: lo-fi, piano"


def test_delete_track_removes_file_and_manifest_entry(tmp_path):
    import json
    folder = tmp_path / "vlog"
    folder.mkdir()
    (folder / "a.wav").write_bytes(b"x" * 10)
    (folder / "manifest.json").write_text(json.dumps({"a.wav": {"license": "generated"}}))
    assert not music_gen.delete_track("../outside.wav", library=tmp_path)
    assert music_gen.delete_track("vlog/a.wav", library=tmp_path)
    assert not (folder / "a.wav").exists()
    assert json.loads((folder / "manifest.json").read_text()) == {}


def test_stage_tracker_times_and_weights_a_job():
    from routes import music as route
    job = {"state": "running", "started_at": 0.0, "ended_at": None, "stages": route._stage_plan(1)}
    route._advance(job, "gpu", None)
    route._advance(job, "v1:sample", 0.5)       # load never reported -> skipped
    states = {s["key"]: s["state"] for s in job["stages"]}
    assert states["gpu"] == "done" and states["v1:load"] == "skipped"
    assert states["v1:sample"] == "running"
    view = route._view(job)
    assert 0.0 < view["progress"] < 1.0
    route._finish(job, True)
    assert all(s["state"] in ("done", "skipped") for s in job["stages"])
    assert job["stages"][0]["seconds"] is not None


def test_comfy_events_map_to_stages():
    seen = []
    clock = music_gen._StageClock("v1:", lambda key, frac: seen.append((key, frac)))
    graph = {"6": {"class_type": "KSampler"}, "7": {"class_type": "VAEDecodeAudio"}}
    listen = music_gen._comfy_listener(graph, clock)
    listen("executing", {"node": "6"})
    listen("progress", {"value": 25, "max": 50})
    listen("executing", {"node": "7"})
    assert seen == [("v1:sample", 0.0), ("v1:sample", 0.5), ("v1:decode", None)]
    assert "sample" in clock.finish()


def test_parse_music_cue_free_text_and_fields():
    r = music_gen.parse_music_cue("sad violin, 70 bpm")
    assert r["style"] == "sad_violin" and r["instruments"] == ["violin"]
    assert r["moods"] == ["sad"] and r["bpm"] == 70 and not r["off"]
    r = music_gen.parse_music_cue("dark eerie ambient with tanpura drone and rain")
    assert r["music_genres"] == ["ambient"] and r["instruments"] == ["tanpura drone"]
    assert set(r["moods"]) == {"dark", "eerie"} and r["extra_tags"] == "rain"
    r = music_gen.parse_music_cue("style=lofi_calm; instruments=piano,tabla; mood=dreamy; bpm=85; engine=stable_audio")
    assert (r["style"], r["instruments"], r["moods"], r["bpm"], r["engine"]) == \
        ("lofi_calm", ["piano", "tabla"], ["dreamy"], 85, "stable_audio")
    for off in ("off", "None", "silence", ""):
        assert music_gen.parse_music_cue(off)["off"]


def test_cue_key_is_order_insensitive():
    a = music_gen.parse_music_cue("violin, sad")
    b = music_gen.parse_music_cue("sad violin")
    b["style"] = None
    assert music_gen.cue_key(a, "ace_step") == music_gen.cue_key(b, "ace_step")
    assert music_gen.cue_key(a, "ace_step") != music_gen.cue_key(a, "stable_audio")


def test_import_track_normalises_records_the_licence_and_refuses_unknown_ones(tmp_path):
    import subprocess
    import pytest
    from config import FFMPEG_BIN
    source = tmp_path / "My Song (final).wav"
    subprocess.run([FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3", str(source)], check=True)
    lib = tmp_path / "music"
    with pytest.raises(ValueError, match="not one the music guard accepts"):
        music_gen.import_track(source, source.name, "ripped-from-youtube", library=lib)
    first = music_gen.import_track(source, source.name, "Own", "Theme", library=lib)
    second = music_gen.import_track(source, source.name, "cc0", library=lib)
    assert first["file"] == "uploads/My_Song_final.flac"
    assert second["file"] == "uploads/My_Song_final_2.flac"
    assert first["seconds"] == pytest.approx(3.0, abs=0.2)
    tracks = {t["file"]: t for t in music_gen.library_tracks(lib)}
    assert tracks[first["file"]]["license_ok"] and tracks[first["file"]]["label"] == "Theme"
    assert tracks[second["file"]]["license"] == "cc0"
    with pytest.raises(ValueError, match="Could not read"):
        bad = tmp_path / "bad.mp3"
        bad.write_bytes(b"not audio")
        music_gen.import_track(bad, bad.name, "own", library=lib)
    assert not (lib / "uploads" / "bad.flac").exists()


def test_compose_caption_for_a_raga_preset():
    prompt = music_gen.compose_prompt(None, instruments=["bansuri flute"], raga="yaman",
                                      engine="ace_step15")
    caption = prompt["caption"]
    assert "yaman" in caption.lower() and "evening" in caption.lower()
    assert "bansuri flute" in caption.lower() and "tanpura" in caption.lower()
    assert "slow" in caption.lower()
    assert caption.rstrip(".").endswith("Instrumental") or "instrumental" in caption.lower()
    assert prompt["tags"].startswith(caption)
    # v1/Stable Audio Open are untouched: no caption, no engine param needed.
    plain = music_gen.compose_prompt(None, instruments=["bansuri flute"], raga="yaman")
    assert "caption" not in plain
    assert "alap" not in plain["tags"].lower()


def test_workflow_build_writes_bpm_and_keyscale_into_the_1_5_graph():
    from presentation import workflows
    resolved = workflows.resolve("music15")
    assert resolved is not None, "ace_step15_music.json must resolve"
    built = resolved.build(positive="a slow evening raga", seed=42, length=90, prefix="music15_test",
                           extra={"bpm": 62, "keyscale": "C major", "timesignature": "4"})
    assert built["5"]["inputs"]["tags"] == "a slow evening raga"
    assert built["5"]["inputs"]["seed"] == 42 and built["8"]["inputs"]["seed"] == 42
    assert built["7"]["inputs"]["seconds"] == 90 and built["5"]["inputs"]["duration"] == 90
    assert built["10"]["inputs"]["filename_prefix"] == "music15_test"
    assert built["5"]["inputs"]["bpm"] == 62
    assert built["5"]["inputs"]["keyscale"] == "C major"
    assert built["5"]["inputs"]["timesignature"] == "4"
    # steps/cfg/model are protected from the global single-model overrides.
    built = resolved.build(positive="p", steps=30, cfg=7.5, model="foo.safetensors")
    assert built["8"]["inputs"]["steps"] == 8 and built["8"]["inputs"]["cfg"] == 1
    assert built["1"]["inputs"]["unet_name"] == "acestep_v1.5_turbo.safetensors"


async def test_verify_and_retry_keeps_best_and_stops_on_pass(tmp_path):
    # Attempt 1 fails (tempo and onset rate both off); attempt 2 passes.
    measurements = [
        {"felt_bpm": 120.0, "onset_rate": 4.0},
        {"felt_bpm": 61.0, "onset_rate": 1.0},
    ]
    paths = []
    for i, _ in enumerate(measurements):
        p = tmp_path / f"attempt_{i}.wav"
        p.write_bytes(b"x")
        paths.append(p)
    calls = {"n": 0}

    async def attempt(attempt_number):
        calls["n"] += 1
        return paths[attempt_number - 1]

    def measure(path):
        return measurements[paths.index(path)]

    winner, measurement, attempts_used, passed = await music_gen._generate_with_verification(
        attempt, measure, requested_bpm=60, calm_required=True, max_attempts=3)
    assert attempts_used == 2 and passed and calls["n"] == 2
    assert winner == paths[1] and measurement["felt_bpm"] == 61.0
    assert not paths[0].exists()   # the losing attempt's file is cleaned up
    assert paths[1].exists()


async def test_verify_and_retry_keeps_the_best_scoring_failure_when_none_pass(tmp_path):
    measurements = [
        {"felt_bpm": 90.0, "onset_rate": 5.0},   # far from 60 bpm
        {"felt_bpm": 58.0, "onset_rate": 4.0},    # close tempo, still too busy
    ]
    paths = []
    for i, _ in enumerate(measurements):
        p = tmp_path / f"fail_{i}.wav"
        p.write_bytes(b"x")
        paths.append(p)

    async def attempt(attempt_number):
        return paths[attempt_number - 1]

    def measure(path):
        return measurements[paths.index(path)]

    winner, measurement, attempts_used, passed = await music_gen._generate_with_verification(
        attempt, measure, requested_bpm=60, calm_required=True, max_attempts=2)
    assert attempts_used == 2 and not passed
    assert winner == paths[1] and measurement["felt_bpm"] == 58.0


def test_resolve_engine_falls_back_to_v1_when_1_5_is_not_ready(monkeypatch):
    monkeypatch.setattr(music_gen, "ace_step15_ready", lambda: False)
    engine, note = music_gen.resolve_engine(music_gen.DEFAULT_ENGINE)
    assert engine == "ace_step"
    assert isinstance(note, str) and note
    # An explicitly chosen engine is never silently swapped.
    engine, note = music_gen.resolve_engine("stable_audio")
    assert engine == "stable_audio" and note is None


def test_parse_music_cue_evening_raga_on_sitar():
    r = music_gen.parse_music_cue("evening raga yaman on sitar")
    assert r["raga"] == "yaman"
    assert r["style"] == "evening_raga_sitar"
    assert "sitar" in r["instruments"]
    assert not r["off"]
    # A bare time-of-day phrase (no named raga) still picks a raga preset.
    r2 = music_gen.parse_music_cue("morning raga on violin")
    assert r2["style"] == "morning_raga_violin" and r2["raga"] is None


def test_measure_speed_reads_click_tracks_within_8_percent(tmp_path):
    # A click track's tempogram legitimately reads either octave (120 bpm's
    # strongest reading here is its 60 bpm subharmonic) -- passing the
    # requested bpm, as `generate_track` always does, picks the nearby peak.
    for bpm in (60, 120):
        path = _burst_track(tmp_path / f"click_{bpm}.wav", bpm / 60.0)
        result = music_gen.measure_speed(path, requested_bpm=bpm)
        assert result["pulsed"]
        assert result["felt_bpm"] is not None
        assert abs(result["felt_bpm"] - bpm) <= bpm * 0.08


def test_measure_speed_slow_calm_uses_onset_rate_not_the_tempogram(tmp_path):
    path = _burst_track(tmp_path / "notes.wav", 0.8, freq=440.0)
    result = music_gen.measure_speed(path, slow_calm=True)
    expected = 0.8 * 60.0 / music_gen.EVENTS_PER_BEAT
    assert result["felt_bpm"] is not None
    assert abs(result["felt_bpm"] - expected) <= expected * 0.08
    # The same audio, not marked slow/calm and without a target, would have
    # reported its tempogram's own top peak instead -- a different number.
    unmarked = music_gen.measure_speed(path)
    assert unmarked["felt_bpm"] == result["pulse_bpm"]


def test_tempo_log_distance_prefers_the_closer_reading_and_allows_octaves():
    close = music_gen._tempo_log_distance(100, 95.0, calm_required=True)
    far = music_gen._tempo_log_distance(100, 150.0, calm_required=True)
    assert close < far
    # Non-calm requests get octave leniency: double tempo scores as close as exact.
    octave = music_gen._tempo_log_distance(100, 200.0, calm_required=False)
    assert octave == pytest.approx(0.0, abs=1e-9)
    # The same reading has no such leniency once the request is slow/calm.
    octave_calm = music_gen._tempo_log_distance(100, 200.0, calm_required=True)
    assert octave_calm > 0.5


def test_lock_tempo_invokes_rubberband_and_clamps_an_extreme_ratio(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"x" * 2000)

        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(music_gen.subprocess, "run", fake_run)
    monkeypatch.setattr(music_gen, "measure_speed",
                        lambda path, requested_bpm=None, slow_calm=False:
                        {"felt_bpm": 100.0, "onset_rate": 2.0})

    source = tmp_path / "winner.wav"
    source.write_bytes(b"x" * 2000)
    # requested 200, felt 50 -> ratio 4.0, clamped to TEMPO_STRETCH_MAX.
    locked_path, info = music_gen._lock_tempo(source, 200, False, {"felt_bpm": 50.0, "onset_rate": 1.0})

    assert len(calls) == 1
    assert any("rubberband=tempo=" in str(arg) for arg in calls[0])
    assert info["tempo_clamped"] is True
    assert info["tempo_ratio"] == pytest.approx(music_gen.TEMPO_STRETCH_MAX)
    assert info["felt_bpm_before"] == 50.0
    assert info["measured_bpm"] == 100.0 and info["onset_rate"] == 2.0
    assert locked_path != source


def test_lock_tempo_is_a_no_op_within_tolerance(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(music_gen.subprocess, "run", lambda *a, **k: calls.append(a))
    source = tmp_path / "winner.wav"
    source.write_bytes(b"x")
    locked_path, info = music_gen._lock_tempo(source, 100, False, {"felt_bpm": 98.0, "onset_rate": 1.0})
    assert not calls
    assert locked_path == source
    assert info["tempo_ratio"] == 1.0 and not info["tempo_clamped"]
    assert info["measured_bpm"] == 98.0


def test_stretch_tempo_with_real_ffmpeg_changes_duration_by_the_ratio(tmp_path):
    import subprocess
    filters = subprocess.run([FFMPEG_BIN, "-hide_banner", "-filters"],
                             capture_output=True, text=True)
    if "rubberband" not in filters.stdout:
        pytest.skip("ffmpeg build has no rubberband filter")
    source = tmp_path / "tone.wav"
    subprocess.run([FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                   "-i", "sine=frequency=440:duration=4:sample_rate=22050", str(source)],
                   check=True)
    destination = tmp_path / "stretched.wav"
    ratio = 1.25
    assert music_gen._stretch_tempo(source, destination, ratio)
    before = sf.info(str(source)).duration
    after = sf.info(str(destination)).duration
    assert after == pytest.approx(before / ratio, rel=0.05)


def test_a_slow_groove_is_read_by_its_beat_not_its_onsets(tmp_path):
    # A 70 bpm groove with eighth-note hats: onsets at 140/min, beat at 70.
    # Read by onsets it looked like 140 and the lock would halve a correct take.
    path = _burst_track(tmp_path / "groove70.wav", 140 / 60.0)
    groove = music_gen.measure_speed(path, requested_bpm=70)
    assert groove["pulsed"]
    assert abs(groove["felt_bpm"] - 70) <= 70 * 0.08 or abs(groove["felt_bpm"] - 140) <= 140 * 0.08
    assert music_gen._tempo_lock_ratio(70, groove["felt_bpm"]) in (None, 1.0) or \
        abs(music_gen._tempo_log_distance(70, groove["felt_bpm"], calm_required=False)) < 0.08


def test_held_notes_with_vibrato_are_counted_once(tmp_path):
    """librosa's default detector read 1.5 vibrato notes/s as ~2.9, which made
    slow ragas look twice as fast and the tempo lock over-slow them."""
    import numpy as np
    import soundfile as sf

    sr, dur = 48000, 30
    for rate in (0.5, 1.0, 1.5):
        y = np.zeros(sr * dur)
        t = np.arange(int(sr * min(2.5, 1.1 / rate))) / sr
        for i, start in enumerate(np.arange(0, dur - 2.6, 1 / rate)):
            f = 440 * 2 ** ((i % 5) / 12)
            env = np.exp(-t * 1.2) * (1 - np.exp(-t * 60))
            seg = 0.3 * np.sin(2 * np.pi * f * t + (0.02 * f / 5) * np.sin(2 * np.pi * 5 * t)) * env
            a = int(start * sr)
            y[a:a + len(seg)] += seg[: len(y) - a]
        path = tmp_path / f"vibrato_{rate}.wav"
        sf.write(str(path), y, sr)
        result = music_gen.measure_speed(path, requested_bpm=rate * 60, slow_calm=True)
        assert abs(result["onset_rate"] - rate) <= rate * 0.15, (rate, result)


def test_tempo_lock_corrects_against_the_delivered_audio(tmp_path, monkeypatch):
    """The whole take averages 109 but its delivered opening is busier: round
    one lands at 106, round two re-stretches the original by the combined
    ratio and lands on target."""
    raw = tmp_path / "take.wav"
    raw.write_bytes(b"x" * 2000)
    ratios = []

    def fake_stretch(src, dst, ratio):
        ratios.append(ratio)
        dst.write_bytes(b"s" * 2000)
        return True

    def fake_trim(src, dst, seconds):
        dst.write_bytes(b"t" * 2000)
        return True

    def fake_measure(path, bpm, free):
        felt = 129.0 * ratios[-1]  # the delivered opening's true pace
        return {"felt_bpm": round(felt, 1), "onset_rate": felt / 60, "pulsed": False, "pulse_bpm": None}

    monkeypatch.setattr(music_gen, "_stretch_tempo", fake_stretch)
    monkeypatch.setattr(music_gen, "_trim", fake_trim)
    monkeypatch.setattr(music_gen, "measure_speed", fake_measure)
    final, info = music_gen._lock_and_trim(raw, 90, True, {"felt_bpm": 109.2, "onset_rate": 1.8},
                                           seconds=60, generated_seconds=90)
    assert info["lock_rounds"] >= 2
    assert abs(info["measured_bpm"] - 90) <= 90 * music_gen.TEMPO_TOLERANCE
    assert final.exists()
    leftovers = [p for p in tmp_path.iterdir() if p not in (raw, final)]
    assert leftovers == []


def test_a_near_silent_take_never_passes_a_calm_request():
    assert not music_gen._onset_ok(0.09, calm_required=True)
    assert music_gen._onset_ok(0.7, calm_required=True)
