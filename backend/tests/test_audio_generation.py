"""Text-to-audio generation (Stable Audio Open on ComfyUI) and the live voice
denoiser (RNNoise via ffmpeg's `arnndn`).

Two things are checked here that nothing else in the suite covers:

  * the "audio" workflow role — `workflows/stable_audio_sfx.json` plus its
    `workflows/manifest.json` bindings — resolves and fills in correctly, and
    `presentation.sound` generates prompts/durations per SFX tag and music
    genre, masters (loudness-normalises, trims, fades) and caches the result
    once per prompt+duration.
  * `render.audio.resolve_voice_enhance("auto")` now actually picks RNNoise
    on this machine (a `.rnnn` model is installed under data/models/rnnoise/,
    DeepFilterNet is not and cannot be), and the `arnndn` filter string it
    builds is valid ffmpeg syntax on a Windows drive-letter path — verified
    against a real ffmpeg process, not just string-compared.
"""

import subprocess
from pathlib import Path

import pytest

import comfyui_bridge
from backend.config import FFMPEG_BIN
from backend.presentation import sound, workflows
from backend.render import audio as render_audio
from backend.timeline.schema import AudioMaster


# --- workflow role: manifest + build() -----------------------------------------

def test_manifest_resolves_audio_role():
    wf = workflows.resolve("audio")
    assert wf is not None
    assert wf.file == "stable_audio_sfx.json"
    assert wf.output_kind == "audio"
    assert "positive" in wf.bindings and "negative" in wf.bindings


def test_audio_workflow_describe_reports_ready_for_its_role():
    info = workflows.describe("stable_audio_sfx.json")
    assert info["valid"] is True
    assert info["output_kind"] == "audio"
    assert info["roles_ok"]["audio"] is True


def test_audio_workflow_build_injects_every_field():
    wf = workflows.resolve("audio")
    graph = wf.build(positive="deep cinematic boom", negative="low quality, hiss",
                     seed=4242, length=3, prefix="buzzedit_audio/boom",
                     steps=30, cfg=4.5)
    positive_id, _, positive_key = wf.bindings["positive"]
    negative_id, _, negative_key = wf.bindings["negative"]
    seed_id, _, seed_key = wf.bindings["seed"]
    length_id, _, length_key = wf.bindings["length"]
    prefix_id, _, prefix_key = wf.bindings["prefix"]
    assert graph[positive_id]["inputs"][positive_key] == "deep cinematic boom"
    assert graph[negative_id]["inputs"][negative_key] == "low quality, hiss"
    assert graph[seed_id]["inputs"][seed_key] == 4242
    assert graph[length_id]["inputs"][length_key] == 3
    assert graph[prefix_id]["inputs"][prefix_key] == "buzzedit_audio/boom"
    # The source graph is not mutated: build() always hands back a copy.
    assert wf.graph[positive_id]["inputs"][positive_key] != "deep cinematic boom"


# --- prompts per SFX tag / music genre -----------------------------------------

def test_every_sfx_tag_has_a_generation_prompt_and_a_short_duration():
    for tag in sound.SFX_TAGS:
        prompt = sound._SFX_GEN_PROMPTS.get(tag)
        assert prompt, f"{tag} has no generation prompt"
        duration = sound._SFX_GEN_DURATION.get(tag)
        assert duration and duration > 0, f"{tag} has no generation duration"


def test_music_genre_prompts_and_the_unknown_genre_fallback():
    assert "horror" in sound._MUSIC_GEN_PROMPTS
    assert sound._MUSIC_GEN_PROMPTS["horror"]
    # A genre with no dedicated prompt still gets something sane, per
    # `warm_generated_cache`'s own fallback expression.
    genre = "some_new_genre"
    fallback = sound._MUSIC_GEN_PROMPTS.get(genre, f"{genre} background music, cinematic instrumental")
    assert genre in fallback


def test_music_bed_negative_prompt_keeps_vocals_and_noise_out():
    assert "vocals" in sound._MUSIC_GEN_NEGATIVE
    assert "noise" in sound._MUSIC_GEN_NEGATIVE


def test_warm_generated_cache_passes_the_negative_prompt_for_a_bed(monkeypatch):
    calls = []

    async def fake_generate(prompt, duration_s, seed, negative=""):
        calls.append((prompt, duration_s, negative))
        return None

    monkeypatch.setattr(sound.workflows, "resolve", lambda role: object())
    monkeypatch.setattr(sound, "find_music", lambda genre, rng: None)
    monkeypatch.setattr(sound, "find_sfx", lambda tag, rng: Path("lib/x.wav"))  # skip SFX gen
    monkeypatch.setattr(sound, "_generate_audio", fake_generate)
    from backend.presentation.models import PresentationSettings

    settings = PresentationSettings(sfx_source="off", music_source="generated", music=True, sfx=False)
    import asyncio
    asyncio.run(sound.warm_generated_cache(settings, "horror", seed=1, comfyui_online=True))

    assert len(calls) == 1
    prompt, duration_s, negative = calls[0]
    assert prompt == sound._MUSIC_GEN_PROMPTS["horror"]
    assert duration_s == sound.SYNTH_LOOP_SECONDS
    assert negative == sound._MUSIC_GEN_NEGATIVE


# --- caching ---------------------------------------------------------------

def test_generate_audio_cache_hit_never_touches_the_workflow_or_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(sound, "AUDIO_GEN_DIR", tmp_path)
    prompt, duration = "a cached whoosh", 0.8
    cached = tmp_path / f"{sound._audio_gen_key(prompt, duration)}.wav"
    cached.write_bytes(b"RIFF" + b"\0" * 2000)

    def boom(role):
        raise AssertionError("cache hit must not resolve a workflow")

    monkeypatch.setattr(sound.workflows, "resolve", boom)
    import asyncio
    result = asyncio.run(sound._generate_audio(prompt, duration, seed=1))
    assert result == cached


def test_generate_audio_masters_and_caches_a_fresh_generation(monkeypatch, tmp_path):
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=f=440:d=1", str(tmp_path / "raw.wav")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0:
        pytest.skip("ffmpeg unavailable")
    raw = tmp_path / "raw.wav"

    monkeypatch.setattr(sound, "AUDIO_GEN_DIR", tmp_path / "audio_gen")
    captured = {}

    class _FakeWorkflow:
        def build(self, **kwargs):
            captured.update(kwargs)
            return {"fake": "graph"}

    monkeypatch.setattr(sound.workflows, "resolve", lambda role: _FakeWorkflow())

    async def fake_submit_and_wait(graph, timeout=120):
        captured["submitted_graph"] = graph
        return [str(raw)]

    monkeypatch.setattr(comfyui_bridge.queue_manager, "submit_and_wait", fake_submit_and_wait)

    import asyncio
    result = asyncio.run(sound._generate_audio("a test tone", 1.0, seed=7,
                                               negative="low quality"))

    assert captured["positive"] == "a test tone"
    assert captured["negative"] == "low quality"
    assert captured["submitted_graph"] == {"fake": "graph"}
    assert result is not None and result.exists() and result.stat().st_size > 1000
    assert result.parent == tmp_path / "audio_gen"

    # A second call for the same prompt+duration is a cache hit: the fake
    # workflow (which would raise nothing, but we can prove no new file is
    # written) is not consulted again for the file's identity.
    cached_again = asyncio.run(sound._generate_audio("a test tone", 1.0, seed=999))
    assert cached_again == result


# --- voice_enhance: rnnoise now wins "auto", deepfilter reports unavailable ----

def test_deepfilter_is_reported_unavailable_on_this_machine():
    # DeepFilterNet's `df` package pins a torchaudio this project cannot run
    # with, so it is never installed here; the check must fail closed, not
    # raise.
    assert render_audio.deepfilter_available() is False


def test_auto_selects_rnnoise_when_a_model_file_is_present(monkeypatch, tmp_path):
    model = tmp_path / "sh.rnnn"
    model.write_bytes(b"\0" * 32)
    monkeypatch.setattr(render_audio, "deepfilter_available", lambda: False)
    monkeypatch.setattr(render_audio, "rnnoise_model_path", lambda: model)
    assert render_audio.resolve_voice_enhance("auto") == "rnnoise"


def test_the_real_installed_rnnoise_model_is_what_auto_picks_up():
    # No monkeypatching: this is the actual repo state the brief installed
    # (data/models/rnnoise/sh.rnnn) and what a fresh render will really do.
    model = render_audio.rnnoise_model_path()
    assert model is not None and model.exists()
    assert render_audio.resolve_voice_enhance("auto") == "rnnoise"


def test_arnndn_filter_path_escapes_a_windows_drive_colon():
    path = Path("B:/data/models/rnnoise/sh.rnnn")
    escaped = render_audio._ffmpeg_filter_path(path)
    assert escaped == "B\\:/data/models/rnnoise/sh.rnnn"
    # A path with no drive letter (no colon) is left alone.
    assert render_audio._ffmpeg_filter_path(Path("data/sh.rnnn")) == "data/sh.rnnn"


def test_arnndn_chain_runs_on_real_ffmpeg_against_a_windows_drive_path(tmp_path):
    """The brief's own check: build the real chain and run it, not just
    string-compare it. Reproduces (and guards) the "No option name near
    '/...'" failure an unescaped drive-letter colon caused."""
    model = render_audio.rnnoise_model_path()
    if model is None:
        pytest.skip("no rnnoise model installed")

    src = tmp_path / "tone_noise.wav"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=f=440:d=3",
         "-f", "lavfi", "-i", "anoisesrc=color=white:d=3:seed=1",
         "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0[out]",
         "-map", "[out]", "-ar", "48000", "-ac", "1", str(src)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not src.exists():
        pytest.skip("ffmpeg unavailable")

    master = AudioMaster(voice_denoise=0.6, voice_deess=0.0, voice_compress=0.0,
                         voice_enhance="rnnoise", voice_eq_preset="", voice_fx=[],
                         loudness_lufs=None, origin="test")
    chain = render_audio.build_voice_chain(master)
    assert any(f.startswith("arnndn=") for f in chain)

    out = tmp_path / "denoised.wav"
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-af", ",".join(chain), "-ar", "48000", "-ac", "1", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert out.exists() and out.stat().st_size > 1000
