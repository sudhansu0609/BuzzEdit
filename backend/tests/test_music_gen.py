from presentation import music_gen
from presentation.sound import ACT_MUSIC


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
        assert eng["role"] in ("music", "audio"), name
        assert eng["license"] in music_gen.ALLOWED_LICENSES, name
    assert music_gen.DEFAULT_ENGINE in music_gen.ENGINES
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
