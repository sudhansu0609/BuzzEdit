"""The creator's effects library (presentation/sfx_library.py): the index,
its tags and categories, the resolver, the generation warm-up and the
normalised copies."""

import asyncio
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.presentation import sfx_library
from backend.presentation.sound_design import DiegeticCue


def _library(tmp_path, monkeypatch, files):
    """Files {relative path: seconds} under a stand-in creator folder; durations
    come from the table instead of ffprobe, and every probe is recorded."""
    root = tmp_path / "Audios"
    seconds = {}
    for rel, length in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\0" * 64)
        seconds[path.name] = length
    probed = []

    def probe(path):
        probed.append(Path(path).name)
        return seconds.get(Path(path).name, 2.0)

    monkeypatch.setattr(sfx_library, "_probe", probe)
    monkeypatch.setattr(sfx_library, "CREATOR_LIBRARY", root)
    monkeypatch.setattr(sfx_library, "LIBRARY_DIR", tmp_path / "data_sfx")
    monkeypatch.setattr(sfx_library, "INDEX_PATH", tmp_path / "sfx_index.json")
    monkeypatch.setattr(sfx_library, "_memo", {})
    return root, probed


def test_index_tags_folders_and_freesound_names_and_finds_the_music(tmp_path, monkeypatch):
    root, _ = _library(tmp_path, monkeypatch, {
        "footSteps/191102__bia87__foot-steps-on-gravel.wav": 4.0,
        "Dark Forest Walk - Some Artist.mp3": 180.0,
        "chill audios/Calm Night.mp3": 150.0,
        "rain/heavy rain on roof.wav": 45.0,
        "door knock/knock.wav": 1.0,
    })
    entries = {Path(e.path).name: e for e in sfx_library.index([root])}
    steps = entries["191102__bia87__foot-steps-on-gravel.wav"]
    assert {"footsteps", "gravel"} <= set(steps.tags)
    assert "bia87" not in steps.tags and "191102" not in steps.tags
    assert "footsteps" in steps.folder_tags
    assert steps.category == "effect" and steps.license == "own"
    assert entries["Dark Forest Walk - Some Artist.mp3"].category == "music"
    assert entries["Calm Night.mp3"].category == "music"
    assert entries["heavy rain on roof.wav"].category == "ambience"
    assert {"door", "knock"} <= set(entries["knock.wav"].folder_tags)


def test_a_rerun_probes_only_new_files(tmp_path, monkeypatch):
    root, probed = _library(tmp_path, monkeypatch, {"dog/bark.wav": 1.5})
    sfx_library.index([root])
    assert probed == ["bark.wav"]
    (root / "owl").mkdir()
    (root / "owl" / "hoot.wav").write_bytes(b"RIFF" + b"\0" * 64)
    probed.clear()
    sfx_library._memo.clear()
    entries = sfx_library.index([root])
    assert probed == ["hoot.wav"]
    assert {Path(e.path).name: e.duration_s for e in entries}["bark.wav"] == 1.5


def test_library_roots_follow_the_job_then_the_app_setting_then_the_creator(monkeypatch, tmp_path):
    from store.app_settings import AppSettings
    monkeypatch.setattr(sfx_library, "LIBRARY_DIR", tmp_path / "data_sfx")
    monkeypatch.setattr(sfx_library, "CREATOR_LIBRARY", tmp_path / "creator")
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: None)
    assert sfx_library.library_roots() == [tmp_path / "data_sfx", tmp_path / "creator"]
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: [str(tmp_path / "set")])
    assert sfx_library.library_roots() == [tmp_path / "data_sfx", tmp_path / "set"]
    assert sfx_library.library_roots([str(tmp_path / "job")]) == [tmp_path / "data_sfx",
                                                                  tmp_path / "job"]


def test_resolve_effect_picks_the_file_whose_tags_match_the_words(tmp_path, monkeypatch):
    root, _ = _library(tmp_path, monkeypatch, {
        "footSteps/191102__bia87__foot-steps-on-gravel.wav": 4.0,
        "footSteps/55555__someone__steps-on-wood.wav": 4.0,
        "screams/woman scream.wav": 2.0,
        "chill audios/Walking Home - Artist.mp3": 120.0,
    })
    monkeypatch.setattr(sfx_library, "cached_effect", lambda text, duration: None)
    path, score, source = sfx_library.resolve_effect("man walking on gravel", random.Random(1),
                                                     roots=[root])
    assert source == "library" and score > 0
    assert path.name == "191102__bia87__foot-steps-on-gravel.wav"


def test_resolve_effect_falls_back_to_a_cached_generation_then_nothing(tmp_path, monkeypatch):
    root, _ = _library(tmp_path, monkeypatch, {"dog/bark.wav": 1.5})
    made = tmp_path / "made.wav"
    monkeypatch.setattr(sfx_library, "cached_effect", lambda text, duration: made)
    assert sfx_library.resolve_effect("temple conch blowing", random.Random(1),
                                      roots=[root]) == (made, 0, "generated")
    monkeypatch.setattr(sfx_library, "cached_effect", lambda text, duration: None)
    assert sfx_library.resolve_effect("temple conch blowing", random.Random(1),
                                      roots=[root]) == (None, 0, "none")


def test_files_without_a_licence_are_never_chosen(tmp_path, monkeypatch):
    _library(tmp_path, monkeypatch, {"dog/bark.wav": 1.5})
    other = tmp_path / "Other" / "dog"
    other.mkdir(parents=True)
    (other / "bark.wav").write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(sfx_library, "cached_effect", lambda text, duration: None)
    assert sfx_library.resolve_effect("dog barking", random.Random(1),
                                      roots=[tmp_path / "Other"])[2] == "none"


def test_warm_effects_generates_only_what_nothing_matches(tmp_path, monkeypatch):
    root, _ = _library(tmp_path, monkeypatch, {"dog/bark.wav": 1.5})
    from backend.presentation import workflows
    monkeypatch.setattr(sfx_library, "library_roots", lambda extra=None: [root])
    monkeypatch.setattr(sfx_library, "cached_effect", lambda text, duration: None)
    monkeypatch.setattr(workflows, "resolve", lambda kind: object())
    asked = []

    async def fake_generate(text, duration, seed=0):
        asked.append((text, duration))
        return tmp_path / "made.wav"

    monkeypatch.setattr(sfx_library, "generated_effect", fake_generate)
    cues = [DiegeticCue(1.0, "dog barking"), DiegeticCue(5.0, "temple conch", kind="bed")]
    counts = asyncio.run(sfx_library.warm_effects(cues, 7, comfyui_online=True))
    assert asked == [("temple conch", sfx_library.GEN_BED_S)] and counts["generated"] == 1
    asked.clear()
    asyncio.run(sfx_library.warm_effects(cues, 7, comfyui_online=False))
    assert asked == []


def test_prepare_effect_keeps_the_original_when_ffmpeg_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sfx_library, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sfx_library, "FFMPEG_BIN", str(tmp_path / "no-ffmpeg-here"))
    source = tmp_path / "x.wav"
    source.write_bytes(b"RIFF" + b"\0" * 64)
    assert sfx_library.prepare_effect(source, "accent") == source


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_prepare_effect_writes_one_normalised_copy_per_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(sfx_library, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1.5", str(source)], check=True)
    accent = sfx_library.prepare_effect(source, "accent")
    assert accent.parent == tmp_path / "cache" and accent.name.endswith("_accent.wav")
    assert accent.stat().st_size > 1000
    assert sfx_library.prepare_effect(source, "accent") == accent
    bed = sfx_library.prepare_effect(source, "bed")
    assert bed != accent and bed.name.endswith("_bed.wav")
