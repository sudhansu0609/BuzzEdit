"""The vibe reel (REELS_DESK_PLAN.md §6, VR1): request validation, camera-word
stripping, the ASS text file, the video/audio filter graphs, and a job that
fails loudly rather than deliver a reel with a missing or black scene."""

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

from presentation import vibe_reel as vr
from presentation import workflows
from presentation.models import Asset, Beat, Character, PresentationSettings
from presentation.vibe_reel import Ambience, Line, Music, Shot, VibeReelRequest
from runtime import master_control


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, capture_output=True, timeout=30, creationflags=0)


def _make_clip(path: Path, color: str, seconds: float = 0.3, size: str = "64x64") -> Path:
    """A tiny, real, solid-colour clip -- for the brightness check and the
    duration-mismatch tests, with no GPU involved."""
    _run([vr.FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={color}:s={size}:d={seconds}:r=10",
         "-pix_fmt", "yuv420p", str(path)])
    return path


def _make_image(path: Path, color: str, size: str = "64x64") -> Path:
    _run([vr.FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={color}:s={size}", "-frames:v", "1", str(path)])
    return path


def _still_beat(beat_id: str = "shot0", seconds: float = 6.0) -> Beat:
    return Beat(id=beat_id, start_s=0.0, end_s=seconds, topic="vibe reel scene",
               kind="broll_image", image_prompt="a quiet lake", planned_duration_s=seconds)


def _video_beat(beat_id: str = "shot0", seconds: float = 6.0) -> Beat:
    return Beat(id=beat_id, start_s=0.0, end_s=seconds, topic="vibe reel scene",
               kind="broll_video", image_prompt="a quiet lake", video_prompt="mist drifts",
               planned_duration_s=seconds)


def _request(**overrides) -> dict:
    body = dict(
        out_dir=str(Path.cwd() / "out"),
        name="reel",
        art_style="real",
        world="a quiet hill town at night",
        shots=[{"prompt": "a moonlit lake", "motion": "mist drifts", "seconds": 6.0, "kind": "video"}] * 4,
        lines=[{"text": "line one", "start_s": 0.0, "end_s": 6.0},
              {"text": "line two", "start_s": 6.0, "end_s": 12.0}],
        credit="— poet",
        music={"brief": "soft piano, calm, 60 bpm"},
        ambience=[{"prompt": "light rain", "gain_db": -20.0}],
    )
    body.update(overrides)
    return body


# --- request validation -----------------------------------------------------

def test_valid_request_parses():
    req = VibeReelRequest(**_request())
    assert len(req.shots) == 4 and len(req.lines) == 2


@pytest.mark.parametrize("overrides", [
    {"shots": []},                                                # too few shots
    {"shots": [{"prompt": "a", "seconds": 6.0}] * 8},             # too many shots
    {"shots": [{"prompt": "a", "seconds": 3.0}] * 4},             # seconds too short
    {"shots": [{"prompt": "a", "seconds": 35.0}] * 4},            # seconds too long
    {"lines": []},                                                # too few lines
    {"lines": [{"text": "x", "start_s": 0.0, "end_s": 1.0}] * 5},  # too many lines
    {"out_dir": "relative/path"},                                 # not absolute
    {"music": {"brief": None, "path": None}},                     # no music source
    {"ambience": [{"prompt": "rain"}, {"prompt": "wind"}, {"prompt": "crickets"}]},  # too many
    {"art_style": "cartoon"},                                     # not real/painterly
])
def test_bad_requests_are_rejected(overrides):
    with pytest.raises(Exception):
        VibeReelRequest(**_request(**overrides))


def test_lines_reel_with_no_music_is_accepted():
    """2026-10-24, VR8: the tool adds no AI music to a lines reel when asked
    not to -- the creator adds his own song in Instagram over the ambience."""
    req = VibeReelRequest(**_request(music=None))
    assert req.music is None


def test_illustrated_art_style_is_accepted():
    req = VibeReelRequest(**_request(art_style="illustrated"))
    assert req.art_style == "illustrated"


def test_shot_kind_defaults_to_still():
    req = VibeReelRequest(**_request(shots=[{"prompt": "a lake", "seconds": 6.0}] * 4))
    assert all(s.kind == "still" for s in req.shots)


# --- camera words --------------------------------------------------------------

@pytest.mark.parametrize("word", ["camera", "DSLR", "tripod", "lens", "shot on a Canon 5D", "photo of"])
def test_camera_words_are_stripped(word):
    text = f"a moonlit lake, {word}, soft mist"
    cleaned = vr.strip_camera_words(text)
    assert word.lower() not in cleaned.lower()


def test_strip_camera_words_leaves_the_rest():
    cleaned = vr.strip_camera_words("a moonlit lake with mist drifting, warm light")
    assert cleaned == "a moonlit lake with mist drifting, warm light"


# --- script detection and fonts -------------------------------------------------

def test_detect_script():
    assert vr.detect_script("ग़ालिब") == "devanagari"
    assert vr.detect_script("Sun lete hain") == "roman"
    assert vr.font_for_script("devanagari") == vr.DEVANAGARI_FONT
    assert vr.font_for_script("roman") == vr.ROMAN_FONT


# --- the ASS text file ----------------------------------------------------------

def test_ass_script_has_devanagari_font_fades_position_and_credit():
    lines = [Line(text="ता हम को शिकायत की भी", start_s=0.0, end_s=6.0)]
    ass_text = vr.build_ass_script(lines, "— ग़ालिब")
    assert f"Kokila" in ass_text
    assert "ता हम को शिकायत की भी" in ass_text
    assert "— ग़ालिब" in ass_text
    assert ass_text.count("\\fad(800,800)") >= 2       # the card and the credit
    assert "\\pos(540," in ass_text                     # horizontally centred (1080 wide)


def test_ass_script_uses_roman_sans_for_a_roman_card():
    lines = [Line(text="Sun lete hain go zikr hamara nahin karte", start_s=0.0, end_s=6.0)]
    ass_text = vr.build_ass_script(lines, None)
    assert vr.ROMAN_FONT in ass_text
    assert "Kokila" not in ass_text
    # a long line wraps onto two even rows (ASS line break) instead of shrinking
    assert "Sun lete hain go zikr hamara nahin karte" in ass_text.replace(r"\N", " ")


def test_explicit_script_field_overrides_detection():
    # Latin text forced to the Devanagari font via the explicit field.
    lines = [Line(text="namaste", start_s=0.0, end_s=4.0, script="devanagari")]
    ass_text = vr.build_ass_script(lines, None)
    assert "Kokila" in ass_text


def test_long_card_gets_a_smaller_font_than_the_default():
    short = [Line(text="chhoti si baat", start_s=0.0, end_s=6.0)]
    long = [Line(text="ek bahut lambi panktiyon wali shayari ki line jo poori width le legi", start_s=0.0, end_s=6.0)]
    short_size = vr._card_font_size(short[0].text, vr.ROMAN_FONT)
    long_size = vr._card_font_size(long[0].text, vr.ROMAN_FONT)
    assert long_size < short_size
    assert long_size >= vr.MIN_CARD_FONT_PX


# --- video filter graph ------------------------------------------------------------

def test_still_shot_filter_has_a_slow_centred_push_and_no_other_motion():
    f = vr.still_shot_filter(0, 6.0)
    assert "zoompan" in f
    assert "1+0.04*" in f
    assert "iw/2-(iw/zoom)/2" in f and "ih/2-(ih/zoom)/2" in f   # centred, no pan
    assert "d=1" in f


def test_video_shot_filter_has_no_zoom():
    f = vr.video_shot_filter(0, 6.0)
    assert "zoompan" not in f
    assert "scale=1080:1920" in f and "crop=1080:1920" in f


def test_video_shot_filter_pads_a_short_clip_with_a_held_frame():
    f = vr.video_shot_filter(0, 6.0, pad_s=0.4)
    assert "tpad=stop_mode=clone:stop_duration=0.400" in f


def test_video_shot_filter_is_not_stepped_by_default():
    f = vr.video_shot_filter(0, 6.0)
    assert f"fps={vr.ANIMATION_HOLD_FPS}" not in f
    assert f"fps={vr.FPS}" in f


def test_video_shot_filter_steps_to_animation_hold_fps_when_stepped():
    f = vr.video_shot_filter(0, 6.0, stepped=True)
    assert f"fps={vr.ANIMATION_HOLD_FPS},fps={vr.FPS}" in f


def test_video_graph_xfade_offsets_are_cumulative_seconds_minus_one_each():
    durations = [6.0, 7.0, 5.0, 6.0]
    kinds = ["video"] * 4
    graph, total = vr.build_video_graph(durations, kinds, [0.0] * 4, Path("x.ass"))
    offsets = [float(m) for m in re.findall(r"offset=([0-9.]+)", graph)]
    # Every shot here is well over the 1s dissolve, so the dissolve is never
    # clamped down -- each offset is simply the running total so far, minus
    # one more second than the last.
    expected = []
    running = durations[0]
    for d in durations[1:]:
        expected.append(running - 1.0)
        running = running + d - 1.0
    assert offsets == pytest.approx(expected, abs=1e-3)
    assert total == pytest.approx(running, abs=1e-3)
    assert total == pytest.approx(sum(durations) - 3 * 1.0, abs=1e-3)
    assert graph.count("xfade=") == 3


def test_video_graph_fades_from_and_to_black_and_burns_in_the_ass():
    graph, total = vr.build_video_graph([6.0, 6.0], ["video", "video"], [0.0, 0.0], Path("lines.ass"))
    assert "fade=t=in:st=0:d=0.800" in graph
    fade_out_start = total - vr.FADE_OUT_S
    assert f"fade=t=out:st={fade_out_start:.3f}:d=1.500" in graph
    assert "ass=filename=" in graph and "lines.ass" in graph


def test_mixed_still_and_video_shots_use_the_right_filter_each():
    graph, _total = vr.build_video_graph([6.0, 6.0], ["still", "video"], [0.0, 0.0], Path("x.ass"))
    v0 = graph.split(";")[0]
    v1 = graph.split(";")[1]
    assert "zoompan" in v0
    assert "zoompan" not in v1


# --- audio filter graph -----------------------------------------------------------

def test_audio_chain_mixes_music_and_ambience_with_fades_and_loudnorm():
    graph = vr.build_audio_filter(0, [(1, -20.0), (2, -24.0)], 20.0, with_music=True)
    assert "[mus]" in graph
    assert "volume=-20.0dB" in graph and "volume=-24.0dB" in graph
    assert "amix=inputs=3" in graph
    assert "afade=t=in:st=0:d=1.500" in graph
    assert f"afade=t=out:st={20.0 - vr.AUDIO_FADE_OUT_S:.3f}:d=2.500" in graph
    assert f"loudnorm=I={vr.LOUDNESS_LUFS:.1f}:TP={vr.TRUE_PEAK_DBTP:.1f}" in graph


def test_nomusic_chain_drops_the_music_but_keeps_ambience_and_loudness():
    graph = vr.build_audio_filter(None, [(1, -20.0)], 20.0, with_music=False)
    assert "[mus]" not in graph
    assert "volume=-20.0dB" in graph
    assert f"loudnorm=I={vr.LOUDNESS_LUFS:.1f}" in graph
    # A single ambience layer needs no amix -- it is the bed by itself.
    assert "amix=" not in graph


def test_nomusic_chain_with_no_ambience_is_silence_not_a_crash():
    graph = vr.build_audio_filter(None, [], 20.0, with_music=False)
    assert "anullsrc" in graph
    assert "loudnorm" in graph


# --- shot duration rules --------------------------------------------------------

@pytest.mark.parametrize("actual,wanted,action", [
    (6.0, 6.0, "trim"),
    (6.3, 6.0, "trim"),
    (5.7, 6.0, "pad"),       # short by 0.3s -- held
    (5.5, 6.0, "pad"),       # short by exactly 0.5s -- still held
    (5.4, 6.0, "retry"),     # short by 0.6s -- too much, retried
])
def test_shot_duration_action(actual, wanted, action):
    assert vr.shot_duration_action(actual, wanted) == action


# --- black-frame rejection --------------------------------------------------------

def test_mean_luma_tells_black_from_bright(tmp_path):
    black = _make_clip(tmp_path / "black.mp4", "black")
    white = _make_clip(tmp_path / "white.mp4", "white")
    black_luma = vr.mean_luma(black)
    white_luma = vr.mean_luma(white)
    assert black_luma is not None and black_luma <= vr.BLACK_LUMA_MAX
    assert white_luma is not None and white_luma > vr.BLACK_LUMA_MAX + 100


@pytest.mark.asyncio
async def test_generate_shot_retries_and_fails_on_a_persistently_black_image(tmp_path, monkeypatch):
    black_png = _make_image(tmp_path / "black.png", "black")
    calls = {"n": 0}

    async def fake_generate_assets(beats, project_dir, settings, seed_base=0, canvas_size=None,
                                   stats=None, character=None, character_cache_dir=None,
                                   progress_cb=None, workflow_files=None, gen_size_override=None):
        calls["n"] += 1
        beat = beats[0]
        return [Asset(beat_id=beat.id, kind="image", path=str(black_png), width=64, height=64)], []

    monkeypatch.setattr(vr.assets_mod, "generate_assets", fake_generate_assets)
    beat = _still_beat()
    path, actual, pad, reason, _workflow_file = await vr._generate_shot(beat, tmp_path, PresentationSettings(), seed_base=1)
    assert path is None
    assert "black" in reason.lower() or "luma" in reason.lower()
    assert calls["n"] == vr.MAX_SHOT_ATTEMPTS


@pytest.mark.asyncio
async def test_generate_shot_succeeds_with_a_bright_still(tmp_path, monkeypatch):
    bright_png = _make_image(tmp_path / "bright.png", "white")

    async def fake_generate_assets(beats, project_dir, settings, seed_base=0, canvas_size=None,
                                   stats=None, character=None, character_cache_dir=None,
                                   progress_cb=None, workflow_files=None, gen_size_override=None):
        beat = beats[0]
        return [Asset(beat_id=beat.id, kind="image", path=str(bright_png), width=64, height=64)], []

    monkeypatch.setattr(vr.assets_mod, "generate_assets", fake_generate_assets)
    beat = _still_beat(seconds=6.0)
    path, actual, pad, reason, _workflow_file = await vr._generate_shot(beat, tmp_path, PresentationSettings(), seed_base=1)
    assert path == bright_png
    assert reason == ""
    assert actual == 6.0


# --- the whole job fails loudly on a missing shot -------------------------------

@pytest.mark.asyncio
async def test_render_vibe_reel_fails_when_one_shot_cannot_be_made(tmp_path, monkeypatch):
    bright_png = None

    async def fake_generate_assets(beats, project_dir, settings, seed_base=0, canvas_size=None,
                                   stats=None, character=None, character_cache_dir=None,
                                   progress_cb=None, workflow_files=None, gen_size_override=None):
        nonlocal bright_png
        beat = beats[0]
        if beat.id == "shot1":
            return [], [{"beat_id": beat.id, "reason": "ComfyUI offline"}]
        if bright_png is None:
            bright_png = _make_image(project_dir / "bright.png", "white")
        return [Asset(beat_id=beat.id, kind="image", path=str(bright_png), width=64, height=64)], []

    monkeypatch.setattr(vr.assets_mod, "generate_assets", fake_generate_assets)
    req = VibeReelRequest(**_request(
        out_dir=str(tmp_path), shots=[{"prompt": "a lake", "seconds": 6.0}] * 4))
    with pytest.raises(vr.VibeReelError) as exc:
        await vr.render_vibe_reel(req)
    assert "shot1" in str(exc.value)
    assert not (tmp_path / "reel.mp4").exists()


@pytest.mark.asyncio
async def test_render_vibe_reel_skips_the_nomusic_file_when_the_request_has_no_music(
        tmp_path, monkeypatch):
    """2026-10-24, VR8: a lines reel with `music: null` gets `<name>.mp4`
    (ambience only) but no `<name>_nomusic.mp4` -- there is nothing left to
    drop from it."""
    bright_png = _make_image(tmp_path / "bright.png", "white")

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        return bright_png, beat.planned_duration_s or beat.duration_s, 0.0, "", ""

    async def fake_resolve_ambience(items, total_seconds):
        return []

    main_calls = []
    nomusic_calls = []

    async def fake_render_main(shot_paths, shot_durations, shot_kinds, shot_pads, ass_path,
                               music_path, ambience, out_path, **kwargs):
        main_calls.append(music_path)
        out_path.write_bytes(b"video")
        return sum(shot_durations)

    async def fake_render_nomusic(main_path, ambience, total, out_path):
        nomusic_calls.append(out_path)

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    monkeypatch.setattr(vr, "_resolve_ambience", fake_resolve_ambience)
    monkeypatch.setattr(vr, "_render_main", fake_render_main)
    monkeypatch.setattr(vr, "_render_nomusic", fake_render_nomusic)
    monkeypatch.setattr(vr, "check_no_black_frames", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_extract_cover", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_measure_lufs", lambda *a, **k: None)

    req = VibeReelRequest(**_request(
        out_dir=str(tmp_path), music=None,
        shots=[{"prompt": "a lake", "seconds": 6.0}] * 4))
    result = await vr.render_vibe_reel(req)

    assert main_calls == [None]            # _render_main got music_path=None
    assert nomusic_calls == []             # _render_nomusic never ran
    assert result["video_nomusic"] is None
    assert result["video"] == str(Path(tmp_path) / "reel.mp4")


# --- master pause (BuzzcafStudio docs/STUDIO_DIRECTOR_PLAN.md Part 1) -----------

@pytest.fixture(autouse=True)
def _isolated_master_state(tmp_path, monkeypatch):
    monkeypatch.setattr(master_control, "STATE_PATH", tmp_path / "master_control_state.json")
    master_control._state_cache = None
    master_control._held.clear()
    yield
    if master_control.is_paused():
        master_control.resume()
    master_control._state_cache = None
    master_control._held.clear()


@pytest.mark.asyncio
async def test_render_vibe_reel_holds_between_shots_then_resumes(tmp_path, monkeypatch):
    """A reel paused between shots makes nothing new while held, and
    finishes (not failed) once resumed -- every shot already cached stays."""
    bright_png = _make_image(tmp_path / "bright.png", "white")
    made: List[str] = []

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        made.append(beat.id)
        return bright_png, beat.planned_duration_s or beat.duration_s, 0.0, "", ""

    async def fake_resolve_ambience(items, total_seconds):
        return []

    async def fake_render_main(shot_paths, shot_durations, shot_kinds, shot_pads, ass_path,
                               music_path, ambience, out_path, **kwargs):
        out_path.write_bytes(b"video")
        return sum(shot_durations)

    async def fake_render_nomusic(main_path, ambience, total, out_path):
        pass

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    monkeypatch.setattr(vr, "_resolve_ambience", fake_resolve_ambience)
    monkeypatch.setattr(vr, "_render_main", fake_render_main)
    monkeypatch.setattr(vr, "_render_nomusic", fake_render_nomusic)
    monkeypatch.setattr(vr, "check_no_black_frames", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_extract_cover", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_measure_lufs", lambda *a, **k: None)
    monkeypatch.setattr(master_control, "POLL_S", 0.02)

    req = VibeReelRequest(**_request(
        out_dir=str(tmp_path), music=None,
        shots=[{"prompt": "a lake", "seconds": 6.0}] * 4))

    master_control.pause(mode="soft", reason="test")
    task = asyncio.ensure_future(vr.render_vibe_reel(req))
    try:
        for _ in range(200):
            if any("vibe_reel" in h for h in master_control.held()):
                break
            await asyncio.sleep(0.02)
        assert any("vibe_reel" in h for h in master_control.held())
        assert made == []  # nothing generated while held
        master_control.resume()
        result = await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()

    assert made == ["shot0", "shot1", "shot2", "shot3"]
    assert result["video"] == str(Path(tmp_path) / "reel.mp4")


# --- VR4: one face for every shot that shows the character ----------------------

def _lead(gender: str = "male") -> Character:
    return Character(name="Inspector Raghav", gender=gender,
                     description="a 45-year-old Indian man with short salt-and-pepper hair, "
                                 "a thick black moustache and a khaki police uniform")


def test_shows_character_without_a_character_is_rejected():
    with pytest.raises(Exception):
        VibeReelRequest(**_request(
            shots=[{"prompt": "a lake", "seconds": 6.0, "shows_character": True}] * 4))


def test_shows_character_is_accepted_with_a_character():
    lead = _lead()
    req = VibeReelRequest(**_request(
        character=lead.model_dump(),
        shots=[{"prompt": "a lake", "seconds": 6.0, "shows_character": True}] * 4))
    assert req.character is not None and req.character.name == "Inspector Raghav"
    assert all(s.shows_character for s in req.shots)


def test_build_beats_tags_the_beat_and_adds_the_character_description():
    lead = _lead()
    req = VibeReelRequest(**_request(
        character=lead.model_dump(),
        shots=[{"prompt": "standing on a misty platform", "motion": "mist drifts",
               "seconds": 6.0, "kind": "video", "shows_character": True},
              {"prompt": "an empty corridor", "seconds": 6.0},
              {"prompt": "a quiet lake", "seconds": 6.0}]))
    beats = vr.build_beats(req)
    shown, hidden = beats[0], beats[1]
    assert shown.shows_character
    assert lead.description in (shown.image_prompt or "")
    assert lead.description in (shown.video_prompt or "")
    assert not hidden.shows_character
    assert lead.description not in (hidden.image_prompt or "")


def test_build_beats_without_a_character_leaves_every_shot_unchanged():
    req = VibeReelRequest(**_request())
    beats = vr.build_beats(req)
    assert not any(b.shows_character for b in beats)


@pytest.mark.asyncio
async def test_generate_shot_passes_the_character_and_cache_dir_through(tmp_path, monkeypatch):
    lead = _lead()
    seen: dict = {}

    async def fake_generate_assets(beats, project_dir, settings, seed_base=0, canvas_size=None,
                                   stats=None, character=None, character_cache_dir=None,
                                   progress_cb=None, workflow_files=None, gen_size_override=None):
        seen["character"] = character
        seen["character_cache_dir"] = character_cache_dir
        beat = beats[0]
        bright = _make_image(project_dir / "bright.png", "white")
        return [Asset(beat_id=beat.id, kind="image", path=str(bright), width=64, height=64)], []

    monkeypatch.setattr(vr.assets_mod, "generate_assets", fake_generate_assets)
    beat = _still_beat().model_copy(update={"shows_character": True})
    shot_dir = tmp_path / "shots" / "shot0"
    shot_dir.mkdir(parents=True)
    shared = tmp_path / "reel"
    path, _actual, _pad, reason, _workflow_file = await vr._generate_shot(
        beat, shot_dir, PresentationSettings(), seed_base=1,
        character=lead, character_cache_dir=shared)
    assert path is not None, reason
    assert seen["character"] is lead
    assert seen["character_cache_dir"] == shared


class _FakeFaceQueue:
    """A ComfyUI stand-in: every submission returns the same output file, and
    each graph is recorded so a test can tell the reference portrait's own
    call (no face swap node, per `assets._prepare_face`) from a shot's own
    still (face swap added, per `assets._generate_one`)."""

    def __init__(self, output_path: Path):
        self.client = self
        self.graphs: List[Dict] = []
        self._output = output_path

    def is_connected(self, timeout: float = 10.0) -> bool:
        return True

    def has_node(self, class_type, timeout: float = 15) -> bool:
        return True

    async def submit_and_wait(self, graph, timeout=0):
        self.graphs.append(graph)
        return [str(self._output)]

    def upload_image(self, path, timeout=60):
        return "bz_ref_uploaded.png"


def _wire_fake_comfy(monkeypatch, tmp_path) -> "_FakeFaceQueue":
    import comfyui_bridge
    out = tmp_path / "out.png"
    out.write_bytes(b"not really a png")
    queue = _FakeFaceQueue(out)
    monkeypatch.setattr(comfyui_bridge, "queue_manager", queue)
    still_graph = {"dec": {"class_type": "VAEDecode", "inputs": {}},
                   "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "x"}}}
    monkeypatch.setattr(vr.assets_mod.workflows, "resolve",
                        lambda role: workflows.ResolvedWorkflow("broll_image", "fake.json", still_graph, {}, "image")
                        if role == "broll_image" else None)
    monkeypatch.setattr(vr.assets_mod, "_setting", lambda key, default: default)
    return queue


@pytest.mark.asyncio
async def test_character_reference_portrait_is_made_once_across_shots(tmp_path, monkeypatch):
    """Three shots that each `shows_character`, each with their own per-shot
    cache directory (VR1's near-match fix) but the same `character_cache_dir`:
    the reference portrait is generated once and the other two calls reuse
    it (VR4) -- only the content still is generated fresh every time."""
    lead = _lead()
    queue = _wire_fake_comfy(monkeypatch, tmp_path)
    settings = PresentationSettings(art_style="real", character_consistency=True)
    shared = tmp_path / "reel"
    for i in range(3):
        beat = _still_beat(beat_id=f"shot{i}").model_copy(update={"shows_character": True})
        shot_dir = tmp_path / "shots" / beat.id
        shot_dir.mkdir(parents=True)
        path, _actual, _pad, reason, _workflow_file = await vr._generate_shot(
            beat, shot_dir, settings, seed_base=vr.CHARACTER_SEED_BASE,
            character=lead, character_cache_dir=shared)
        assert path is not None, reason
    references = [g for g in queue.graphs if "bz_face_swap" not in g]
    swapped = [g for g in queue.graphs if "bz_face_swap" in g]
    assert len(references) == 1, "the reference portrait must be generated only once per reel"
    assert len(swapped) == 3, "every shot's own still still gets generated and face-swapped"


@pytest.mark.asyncio
async def test_shot_without_the_flag_never_calls_the_face_path(tmp_path, monkeypatch):
    """A shot with no `shows_character` is untouched by any of this: no
    character is passed down, so `generate_assets` never prepares a face."""
    seen: dict = {}

    async def fake_generate_assets(beats, project_dir, settings, seed_base=0, canvas_size=None,
                                   stats=None, character=None, character_cache_dir=None,
                                   progress_cb=None, workflow_files=None, gen_size_override=None):
        seen["character"] = character
        beat = beats[0]
        bright = _make_image(project_dir / "bright.png", "white")
        return [Asset(beat_id=beat.id, kind="image", path=str(bright), width=64, height=64)], []

    monkeypatch.setattr(vr.assets_mod, "generate_assets", fake_generate_assets)
    beat = _still_beat()
    assert not beat.shows_character
    path, _actual, _pad, reason, _workflow_file = await vr._generate_shot(
        beat, tmp_path, PresentationSettings(), seed_base=1)
    assert path is not None, reason
    assert seen["character"] is None


def test_vibe_reel_json_would_record_the_character_and_which_shots_showed_them():
    """The result (and so `vibe_reel.json`, written from `req` + `result`)
    names the character and lists the beat ids that showed them."""
    lead = _lead()
    req = VibeReelRequest(**_request(
        character=lead.model_dump(),
        shots=[{"prompt": "standing on a misty platform", "seconds": 6.0, "shows_character": True},
              {"prompt": "an empty corridor", "seconds": 6.0},
              {"prompt": "a quiet lake", "seconds": 6.0}]))
    beats = vr.build_beats(req)
    character_shots = [b.id for b in beats if b.shows_character]
    assert character_shots == ["shot0"]
    dumped = req.model_dump()
    assert dumped["character"]["name"] == "Inspector Raghav"
    assert dumped["shots"][0]["shows_character"] is True
    assert dumped["shots"][1]["shows_character"] is False


# --- reel_hq quality (2026-10-23): hyper-realistic AI clips, not stock footage --

def test_quality_defaults_to_standard():
    req = VibeReelRequest(**_request())
    assert req.quality == "standard"


def test_quality_accepts_reel_hq():
    req = VibeReelRequest(**_request(quality="reel_hq"))
    assert req.quality == "reel_hq"


def test_quality_rejects_unknown_value():
    with pytest.raises(Exception):
        VibeReelRequest(**_request(quality="ultra"))


def test_reel_hq_mode_is_registered_without_touching_the_production_default():
    mode = workflows.CLIP_GENERATION_MODES["reel_hq"]
    assert mode["file"] == "zimage_wan22_i2v_reel_portrait.json"
    assert mode["fallback"] == "zimage_wan22_i2v_hq16.json"
    assert tuple(mode["size"]) == (576, 1024)
    # Never the default a production reads -- reel_hq is only ever reached by
    # an explicit caller (the vibe reel), not the global clip_generation_mode.
    assert workflows.DEFAULT_CLIP_GENERATION_MODE == "classic"
    assert workflows.clip_generation_mode() != "reel_hq"


def test_resolve_override_chains_the_given_files_then_todays_own_resolution(monkeypatch):
    sentinel = workflows.ResolvedWorkflow(
        "broll_video", "todays_default.json",
        {"x": {"class_type": "SaveVideo", "inputs": {}}}, {}, "video")
    monkeypatch.setattr(workflows, "resolve", lambda role: sentinel if role == "broll_video" else None)
    resolved = workflows.resolve_override(
        "broll_video", ["zimage_wan22_i2v_reel_portrait.json", "zimage_wan22_i2v_hq16.json"])
    assert resolved is not None and resolved.file == "zimage_wan22_i2v_reel_portrait.json"
    assert resolved.fallback is not None and resolved.fallback.file == "zimage_wan22_i2v_hq16.json"
    assert resolved.fallback.fallback is sentinel


def test_resolve_override_skips_a_missing_file_in_the_chain(monkeypatch):
    monkeypatch.setattr(workflows, "resolve", lambda role: None)
    resolved = workflows.resolve_override(
        "broll_video", ["does_not_exist.json", "zimage_wan22_i2v_hq16.json"])
    assert resolved is not None and resolved.file == "zimage_wan22_i2v_hq16.json"
    assert resolved.fallback is None


def test_build_beats_reel_hq_adds_realism_suffix_negative_and_motion_wording():
    req = VibeReelRequest(**_request(
        quality="reel_hq",
        shots=[{"prompt": "a lake at dusk", "motion": "mist drifts", "seconds": 6.0, "kind": "video"},
              {"prompt": "an old doorway", "seconds": 6.0, "kind": "still"},
              {"prompt": "a quiet courtyard", "seconds": 6.0, "kind": "still"}]))
    beats = vr.build_beats(req)
    video_beat, still_beat = beats[0], beats[1]
    assert vr.REALISM_SUFFIX in video_beat.video_prompt
    assert vr.REALISM_SUFFIX in video_beat.image_prompt
    assert vr.REALISM_MOTION_SUFFIX in video_beat.video_prompt
    assert vr.REALISM_NEGATIVE in video_beat.negative_prompt
    assert vr.REALISM_SUFFIX in still_beat.image_prompt
    assert vr.REALISM_NEGATIVE in still_beat.negative_prompt
    # The suffix/motion wording never smuggles a camera word into a positive prompt.
    for beat in beats:
        for text in (beat.image_prompt, beat.video_prompt):
            if text:
                assert not vr._CAMERA_WORDS_RE.search(text), text


def test_build_beats_standard_quality_has_no_realism_wording():
    req = VibeReelRequest(**_request(
        shots=[{"prompt": "a lake at dusk", "motion": "mist drifts", "seconds": 6.0, "kind": "video"}] * 4))
    beats = vr.build_beats(req)
    assert all(vr.REALISM_SUFFIX not in (b.image_prompt or "") for b in beats)
    assert all(vr.REALISM_NEGATIVE not in b.negative_prompt for b in beats)


def test_build_beats_reel_hq_skips_realism_wording_for_painterly():
    req = VibeReelRequest(**_request(
        quality="reel_hq", art_style="painterly",
        shots=[{"prompt": "a lake at dusk", "seconds": 6.0}] * 4))
    beats = vr.build_beats(req)
    assert all(vr.REALISM_SUFFIX not in (b.image_prompt or "") for b in beats)
    assert all(vr.REALISM_NEGATIVE not in b.negative_prompt for b in beats)


def test_build_beats_reel_hq_skips_realism_wording_for_illustrated():
    req = VibeReelRequest(**_request(
        quality="reel_hq", art_style="illustrated",
        shots=[{"prompt": "a lake at dusk", "motion": "snow falls gently", "seconds": 6.0,
               "kind": "video"}] * 4))
    beats = vr.build_beats(req)
    assert all(vr.REALISM_SUFFIX not in (b.image_prompt or "") for b in beats)
    assert all(vr.REALISM_NEGATIVE not in b.negative_prompt for b in beats)
    assert all("soothing hand-drawn" in (b.image_prompt or "") for b in beats)


def test_video_graph_adds_the_look_pass_before_the_ass_overlay_for_reel_hq(tmp_path):
    ass_path = tmp_path / "x.ass"
    ass_filter = vr.ass_mod.build_ass_filter(ass_path)

    graph_std, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path)
    assert vr.LOOK_PASS_FILTER not in graph_std

    graph_hq, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path, look_pass=True)
    assert vr.LOOK_PASS_FILTER in graph_hq
    assert graph_hq.index(vr.LOOK_PASS_FILTER) < graph_hq.index(ass_filter)


@pytest.mark.asyncio
async def test_reel_hq_quality_passes_the_portrait_workflow_and_size_to_every_shot(tmp_path, monkeypatch):
    """`render_vibe_reel` computes the reel_hq workflow/size override once and
    hands it to every shot (still or video) -- never the global
    `clip_generation_mode`, and recorded per shot via `workflow_file`."""
    seen_calls = []

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None):
        seen_calls.append((beat.kind, workflow_files, gen_size_override))
        return None, 0.0, 0.0, "stopped for the test", ""

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    req = VibeReelRequest(**_request(
        out_dir=str(tmp_path), quality="reel_hq",
        shots=[{"prompt": "a lake", "motion": "mist drifts", "seconds": 6.0, "kind": "video"},
              {"prompt": "a quiet room", "seconds": 6.0, "kind": "still"}] * 2))
    with pytest.raises(vr.VibeReelError):
        await vr.render_vibe_reel(req)

    assert len(seen_calls) == 4
    assert {kind for kind, _wf, _gs in seen_calls} == {"broll_video", "broll_image"}
    for _kind, workflow_files, gen_size_override in seen_calls:
        assert workflow_files["broll_image"] == [vr.REEL_HQ_STILL_FILE]
        assert workflow_files["broll_video"] == ["zimage_wan22_i2v_reel_portrait.json",
                                                 "zimage_wan22_i2v_hq16.json"]
        assert gen_size_override["broll_image"] == vr.REEL_HQ_STILL_SIZE
        assert gen_size_override["broll_video"] == (576, 1024)


@pytest.mark.asyncio
async def test_standard_quality_passes_no_workflow_override_to_a_shot(tmp_path, monkeypatch):
    seen_calls = []

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None):
        seen_calls.append((workflow_files, gen_size_override))
        return None, 0.0, 0.0, "stopped for the test", ""

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    req = VibeReelRequest(**_request(out_dir=str(tmp_path)))
    with pytest.raises(vr.VibeReelError):
        await vr.render_vibe_reel(req)

    assert seen_calls and all(wf is None and gs is None for wf, gs in seen_calls)


# --- continuous shots: chaining (2026-10-23 night rules, VR7) --------------------

def test_plan_chain_segments_short_shot_is_one_segment():
    assert vr._plan_chain_segments(6.0) == [6.0]
    assert vr._plan_chain_segments(vr.CHAIN_SEGMENT_SECONDS_MAX) == [vr.CHAIN_SEGMENT_SECONDS_MAX]


def test_plan_chain_segments_long_shot_splits_at_the_ceiling():
    segments = vr._plan_chain_segments(12.0)
    assert len(segments) == 2
    assert sum(segments) == pytest.approx(12.0)
    assert all(s <= vr.CHAIN_SEGMENT_SECONDS_MAX for s in segments)


def test_plan_chain_segments_caps_the_segment_count():
    segments = vr._plan_chain_segments(100.0)
    assert len(segments) == vr.CHAIN_SEGMENTS_MAX
    assert sum(segments) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_generate_chained_shot_seeds_later_segments_from_the_last_frame(tmp_path, monkeypatch):
    calls = []

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        calls.append(start_image_override)
        clip = _make_clip(project_dir / f"seg{len(calls)}.mp4", "white", seconds=6.0)
        return clip, 6.0, 0.0, "", "fake.json"

    async def fake_join(paths, durations, out_path):
        out_path.write_bytes(b"joined")
        return sum(durations) - (len(paths) - 1) * 0.1

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    monkeypatch.setattr(vr, "_join_chain_segments", fake_join)

    beat = _video_beat(seconds=12.0).model_copy(update={"planned_duration_s": 12.0})
    path, actual, pad, reason, workflow_file, segments = await vr._generate_chained_shot(
        beat, tmp_path, PresentationSettings(), seed_base=1)
    assert segments == 2
    assert reason == "" and path is not None
    assert calls[0] is None           # the first segment starts from a fresh still
    assert calls[1] is not None       # the second continues from the first's last frame


@pytest.mark.asyncio
async def test_generate_chained_shot_falls_back_to_the_segments_already_made(tmp_path, monkeypatch):
    """A later segment that cannot be made does not fail the whole shot --
    the chain made so far is kept, held/joined as one (shorter) shot, and
    the failure is recorded in the segment count, not raised."""
    calls = {"n": 0}

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        calls["n"] += 1
        if calls["n"] == 1:
            clip = _make_clip(project_dir / "seg0.mp4", "white", seconds=6.0)
            return clip, 6.0, 0.0, "", "fake.json"
        return None, 0.0, 0.0, "ComfyUI offline", ""

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    beat = _video_beat(seconds=12.0).model_copy(update={"planned_duration_s": 12.0})
    path, actual, pad, reason, workflow_file, segments = await vr._generate_chained_shot(
        beat, tmp_path, PresentationSettings(), seed_base=1)
    assert segments == 1
    assert path is not None and reason == ""


@pytest.mark.asyncio
async def test_generate_chained_shot_fails_only_when_the_first_segment_does(tmp_path, monkeypatch):
    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        return None, 0.0, 0.0, "ComfyUI offline", ""

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)
    beat = _video_beat(seconds=12.0).model_copy(update={"planned_duration_s": 12.0})
    path, actual, pad, reason, workflow_file, segments = await vr._generate_chained_shot(
        beat, tmp_path, PresentationSettings(), seed_base=1)
    assert path is None and segments == 0
    assert "first chain segment failed" in reason


# --- grade and the handheld drift (2026-10-23 night rules, VR7) -----------------

def test_grade_bw_adds_full_desaturation_only_for_bw():
    ass_path = Path("x.ass")
    graph_natural, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path, look_pass=True,
                                            grade=vr.GRADE_NATURAL)
    assert vr.BW_FILTER not in graph_natural

    graph_bw, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path, look_pass=True,
                                       grade=vr.GRADE_BW)
    assert vr.BW_FILTER in graph_bw


def test_handheld_drift_present_for_reel_hq_only():
    ass_path = Path("x.ass")
    graph_std, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path, look_pass=False)
    assert "sin(2*PI*t" not in graph_std

    graph_hq, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path, look_pass=True)
    assert vr.HANDHELD_DRIFT_FILTER in graph_hq


# --- illustrated art style: the "on twos" look pass, no handheld drift (VR8) ----

def test_illustrated_look_pass_has_no_handheld_drift_or_real_style_grain():
    ass_path = Path("x.ass")
    graph, _ = vr.build_video_graph([6.0], ["video"], [0.0], ass_path, look_pass=True,
                                    art_style="illustrated")
    assert vr.ILLUSTRATED_LOOK_PASS_FILTER in graph
    assert vr.HANDHELD_DRIFT_FILTER not in graph
    assert vr.LOOK_PASS_FILTER not in graph
    assert "sin(2*PI*t" not in graph


def test_illustrated_look_pass_steps_a_video_shot_to_the_animation_hold_fps():
    ass_path = Path("x.ass")
    graph, _ = vr.build_video_graph([6.0], ["video"], [0.0], ass_path, look_pass=True,
                                    art_style="illustrated")
    assert f"fps={vr.ANIMATION_HOLD_FPS},fps={vr.FPS}" in graph


def test_illustrated_without_look_pass_is_not_stepped():
    """`stepped` only applies when `look_pass` is on (quality="reel_hq") --
    the illustrated style itself does not force it."""
    ass_path = Path("x.ass")
    graph, _ = vr.build_video_graph([6.0], ["video"], [0.0], ass_path, look_pass=False,
                                    art_style="illustrated")
    assert f"fps={vr.ANIMATION_HOLD_FPS}" not in graph
    assert vr.ILLUSTRATED_LOOK_PASS_FILTER not in graph


def test_real_style_look_pass_is_unaffected_by_the_illustrated_addition():
    ass_path = Path("x.ass")
    graph, _ = vr.build_video_graph([6.0], ["video"], [0.0], ass_path, look_pass=True,
                                    art_style="real")
    assert vr.LOOK_PASS_FILTER in graph and vr.HANDHELD_DRIFT_FILTER in graph
    assert vr.ILLUSTRATED_LOOK_PASS_FILTER not in graph
    assert f"fps={vr.ANIMATION_HOLD_FPS}" not in graph


def test_grade_rejects_unknown_value():
    with pytest.raises(Exception):
        VibeReelRequest(**_request(grade="sepia"))


def test_grade_defaults_to_natural():
    assert VibeReelRequest(**_request()).grade == vr.GRADE_NATURAL


# --- text_style: minimal vs bold, and the yellow highlight ---------------------

def test_text_style_defaults_to_minimal():
    assert VibeReelRequest(**_request()).text_style == vr.TEXT_STYLE_MINIMAL


def test_text_style_rejects_unknown_value():
    with pytest.raises(Exception):
        VibeReelRequest(**_request(text_style="huge"))


def test_minimal_card_is_lower_middle_and_plain_white():
    lines = [Line(text="chupke chupke gham ka khaana", start_s=0.0, end_s=6.0)]
    ass_text = vr.build_ass_script(lines, None, text_style=vr.TEXT_STYLE_MINIMAL)
    y_minimal = round((vr.CARD_POS_Y_MINIMAL + 1.0) / 2.0 * vr.HEIGHT)
    assert f",{y_minimal})" in ass_text
    assert vr.HIGHLIGHT_COLOR.lstrip("#").upper() not in ass_text.upper()


def test_bold_card_is_in_the_upper_third_with_one_yellow_keyword():
    lines = [Line(text="chupke chupke gham ka khaana", start_s=0.0, end_s=6.0)]
    ass_text = vr.build_ass_script(lines, None, text_style=vr.TEXT_STYLE_BOLD, highlight="gham")
    y_bold = round((vr.CARD_POS_Y_BOLD + 1.0) / 2.0 * vr.HEIGHT)
    assert f",{y_bold})" in ass_text
    # yellow (BGR-ordered ASS colour, alpha stripped) for the highlighted word
    rgb, _ = ass_mod_module()._parse_colour(vr.HIGHLIGHT_COLOR)
    assert f"{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper() in ass_text.upper()


def ass_mod_module():
    return vr.ass_mod


def test_highlight_clips_split_into_three_when_the_word_is_present():
    clips = vr._highlight_clips("chupke chupke gham ka khaana", "gham", vr.ROMAN_FONT, 64, 0.0)
    assert len(clips) == 3
    assert clips[1].content == "gham"
    assert clips[1].style.color == vr.HIGHLIGHT_COLOR
    assert clips[0].style.color == "white" and clips[2].style.color == "white"


def test_highlight_clips_is_one_plain_clip_when_the_word_is_absent():
    clips = vr._highlight_clips("chupke chupke gham ka khaana", "nahin", vr.ROMAN_FONT, 64, 0.0)
    assert len(clips) == 1
    assert clips[0].style.color == "white"


# --- the Hinglish font: Poppins, with its named fallback -----------------------

def test_roman_font_is_poppins_when_the_bundled_files_are_there(monkeypatch, tmp_path):
    fake_bold = tmp_path / "Poppins-Bold.ttf"
    fake_bold.write_bytes(b"x")
    monkeypatch.setattr(vr, "_POPPINS_BOLD_FILE", fake_bold)
    assert vr._resolve_roman_font() == "Poppins"


def test_roman_font_falls_back_when_poppins_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, "_POPPINS_BOLD_FILE", tmp_path / "does_not_exist.ttf")
    monkeypatch.setattr(vr, "_MONTSERRAT_BOLD_CANDIDATES", (str(tmp_path / "nope.ttf"),))
    assert vr._resolve_roman_font() == "Segoe UI"


# --- the scrim: a soft dark band scaled by brightness ---------------------------

def test_scrim_filter_is_a_feathered_luma_only_band():
    expr = vr.scrim_filter(vr.CARD_POS_Y_MINIMAL, 0.4)
    assert expr.startswith("geq=lum=")
    assert "cb='cb(X,Y)'" in expr and "cr='cr(X,Y)'" in expr


def test_scrim_alpha_increases_with_measured_brightness():
    dark = vr._scrim_alpha_for(10.0)
    bright = vr._scrim_alpha_for(250.0)
    unknown = vr._scrim_alpha_for(None)
    assert vr.SCRIM_ALPHA_MIN <= dark < unknown < bright <= vr.SCRIM_ALPHA_MAX


def test_video_graph_inserts_the_scrim_before_the_ass_overlay():
    ass_path = Path("x.ass")
    ass_filter = vr.ass_mod.build_ass_filter(ass_path)
    graph, _ = vr.build_video_graph([6.0], ["still"], [0.0], ass_path,
                                    scrim=(vr.CARD_POS_Y_MINIMAL, 0.4))
    assert "geq=lum=" in graph
    assert graph.index("geq=lum=") < graph.index(ass_filter)


# --- Hinglish transliteration (2026-10-23 night rules, VR7) ---------------------

@pytest.mark.asyncio
async def test_transliterate_to_hinglish_converts_devanagari_with_matching_count(monkeypatch):
    import llm.client as llm_client_mod

    async def fake_ask(system_prompt, user_prompt, schema):
        lines = json.loads(user_prompt)["lines"]
        return json.dumps({"lines": [f"hinglish {i}" for i in range(len(lines))]})

    monkeypatch.setattr(llm_client_mod.lm_studio_client, "ask_with_schema", fake_ask)
    out, ok = await vr.transliterate_to_hinglish(["सुबह जल्दी", "घर से निकल गया"])
    assert ok is True
    assert out == ["hinglish 0", "hinglish 1"]


@pytest.mark.asyncio
async def test_transliterate_to_hinglish_falls_back_on_a_count_mismatch(monkeypatch):
    import llm.client as llm_client_mod

    async def fake_ask(system_prompt, user_prompt, schema):
        return '{"lines": ["only one"]}'

    monkeypatch.setattr(llm_client_mod.lm_studio_client, "ask_with_schema", fake_ask)
    lines = ["सुबह जल्दी", "घर से निकल गया"]
    out, ok = await vr.transliterate_to_hinglish(lines)
    assert ok is False
    assert out == lines


@pytest.mark.asyncio
async def test_transliterate_to_hinglish_falls_back_when_the_llm_is_unavailable(monkeypatch):
    import llm.client as llm_client_mod

    async def fake_ask(system_prompt, user_prompt, schema):
        return None

    monkeypatch.setattr(llm_client_mod.lm_studio_client, "ask_with_schema", fake_ask)
    lines = ["सुबह जल्दी"]
    out, ok = await vr.transliterate_to_hinglish(lines)
    assert ok is False
    assert out == lines


@pytest.mark.asyncio
async def test_transliterate_to_hinglish_skips_already_roman_lines():
    out, ok = await vr.transliterate_to_hinglish(["already roman", "also roman"])
    assert ok is True
    assert out == ["already roman", "also roman"]


# --- durable jobs: the per-shot cache survives a resubmit (STUDIO_DIRECTOR_PLAN.md
# enhancement 2, Fable review row 7) --------------------------------------------------

@pytest.mark.asyncio
async def test_resubmitting_the_same_shot_reuses_the_cache_on_disk(tmp_path, monkeypatch):
    """A vibe reel re-submitted after a BuzzEdit restart, same out_dir: a shot
    already made must not be regenerated -- `generate_assets`' own cache
    (keyed on the shot's prompt/seed/workflow) is checked before any ComfyUI
    call, so the second `_generate_shot` call for the same shot dir and beat
    never touches the fake ComfyUI queue a second time."""
    queue = _wire_fake_comfy(monkeypatch, tmp_path)
    settings = PresentationSettings(art_style="real")
    beat = _still_beat("shot0")
    shot_dir = tmp_path / "shots" / beat.id
    shot_dir.mkdir(parents=True)

    path1, _actual1, _pad1, reason1, _wf1 = await vr._generate_shot(
        beat, shot_dir, settings, seed_base=1000)
    assert path1 is not None, reason1
    assert len(queue.graphs) == 1

    path2, _actual2, _pad2, reason2, _wf2 = await vr._generate_shot(
        beat, shot_dir, settings, seed_base=1000)
    assert path2 is not None, reason2
    assert path2 == path1
    assert len(queue.graphs) == 1, "a resubmit with the same shot dir must hit the cache, not regenerate"
