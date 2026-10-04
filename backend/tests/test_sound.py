"""Sound design: the music bed, effects, ambience and voice master.

The checks here are about what lands on the timeline and what the compiler
makes of it. Every sound the pass can fall back on is synthesised with
FFmpeg, so a machine with an empty library must still get a soundtrack, and
the mix must never let the bed sit on top of the voice.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.presentation import sound
from backend.presentation.models import PresentationSettings, SfxCue
from backend.presentation.program import build_program
from backend.render import audio as render_audio
from backend.render.compiler import FilterGraphCompiler
from backend.timeline import build_timeline_from_transcript, clip_ops
from backend.timeline.schema import AudioMaster, SourceFile, Timeline, TimelineItem


def _timeline(seconds: float = 20.0) -> Timeline:
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i in range(int(seconds * 2) - 1)]
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", seconds, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, seconds)], pause_padding_seconds=0.0)


def _fake_synth(monkeypatch, tmp_path):
    """Stand in for FFmpeg synthesis: real files, no subprocess."""
    made = {}

    def fake(kind):
        recipe = sound._recipe(kind)
        if recipe is None:
            return None
        path = tmp_path / "synth" / f"{kind}_deadbeef.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\0" * 2000)
        made[kind] = path
        return path

    monkeypatch.setattr(sound, "synth_path", fake)
    monkeypatch.setattr(sound, "SYNTH_DIR", tmp_path / "synth")
    monkeypatch.setattr(sound, "MUSIC_DIR", tmp_path / "music")
    monkeypatch.setattr(sound, "SFX_DIR", tmp_path / "sfx")
    monkeypatch.setattr(sound, "AMBIENCE_DIR", tmp_path / "ambience")
    monkeypatch.setattr(sound, "find_music", lambda genre, rng, mood=None, library=None: None)
    monkeypatch.setattr(sound, "find_sfx", lambda tag, rng, library=None: None)
    monkeypatch.setattr(sound, "find_ambience", lambda kind, rng, library=None: None)
    return made


# --- library -----------------------------------------------------------------

def test_library_lookup_prefers_the_genre_folder_then_general_then_manifest(tmp_path):
    lib = tmp_path / "music"
    (lib / "horror").mkdir(parents=True)
    (lib / "general").mkdir()
    (lib / "horror" / "creep.mp3").write_bytes(b"x")
    (lib / "general" / "bed.mp3").write_bytes(b"x")
    (lib / "loose.wav").write_bytes(b"x")
    (lib / "manifest.json").write_text('{"loose.wav": {"genre": ["finance"], "license": "cc0"}}')
    (lib / "horror" / "manifest.json").write_text('{"creep.mp3": {"license": "own"}}')
    (lib / "general" / "manifest.json").write_text('{"bed.mp3": {"license": "royalty-free"}}')
    rng = random.Random(1)

    assert sound.find_music("horror", rng, library=lib).name == "creep.mp3"
    assert sound.find_music("comedy", rng, library=lib).name == "bed.mp3"
    shutil.rmtree(lib / "general")
    assert sound.find_music("finance", rng, library=lib).name == "loose.wav"
    assert sound.find_music("comedy", rng, library=lib) is None


def test_library_music_without_a_licence_on_record_is_never_used(tmp_path):
    lib = tmp_path / "music"
    (lib / "horror").mkdir(parents=True)
    (lib / "general").mkdir()
    (lib / "horror" / "ripped.mp3").write_bytes(b"x")          # no manifest at all
    (lib / "general" / "bed.mp3").write_bytes(b"x")
    (lib / "general" / "manifest.json").write_text('{"bed.mp3": {"license": "generated"}}')
    (lib / "vlog").mkdir()
    (lib / "vlog" / "ripped.mp3").write_bytes(b"x")
    # The unlicensed genre folder is skipped, falling through to general/.
    assert sound.find_music("vlog", random.Random(1), library=lib).name == "bed.mp3"
    # ...but never for horror: an untagged general track (a "sad violin") is
    # not a horror bed, so horror gets nothing and the drone fallback instead.
    assert sound.find_music("horror", random.Random(1), library=lib) is None
    (lib / "general" / "manifest.json").write_text('{"bed.mp3": {"license": "unknown"}}')
    assert sound.find_music("vlog", random.Random(1), library=lib) is None


def test_every_synth_recipe_is_a_known_kind():
    for kind in sound.SYNTH_KINDS:
        assert sound._recipe(kind) is not None, kind
        assert sound.synth_duration(kind) > 0


# --- planning ----------------------------------------------------------------

def test_horror_with_an_empty_library_gets_a_drone_and_wind(monkeypatch, tmp_path):
    made = _fake_synth(monkeypatch, tmp_path)
    timeline = _timeline()
    program = build_program(timeline)
    plan = sound.plan_sound(timeline, program, PresentationSettings(), "horror", 1,
                            broll_windows=[(3.0, 6.0), (9.0, 12.0)], popup_times=[7.5])
    assert plan.music and plan.music[0].synthesised
    assert "drone" in made
    assert plan.ambience and plan.ambience[0].kind == "wind"
    tags = [c.tag for c in plan.sfx]
    # Two cutaways 6 s apart get one whoosh (one per WHOOSH_MIN_GAP_S), none on return.
    assert tags.count("whoosh") == 1 and "whoosh_soft" not in tags
    assert "pop" in tags
    assert plan.music[0].duck > 0


def test_a_plain_genre_with_no_library_reports_missing_music_not_a_fake_bed(monkeypatch, tmp_path):
    _fake_synth(monkeypatch, tmp_path)
    timeline = _timeline()
    plan = sound.plan_sound(timeline, build_program(timeline), PresentationSettings(),
                            "science_education", 1)
    assert not plan.music
    assert "music_missing" in plan.notes
    assert not plan.ambience


def test_effects_too_close_together_collapse_to_the_weightier_one():
    cues = sound._space_sfx(
        [(5.0, "whoosh", 1.0), (5.4, "stinger", 1.0), (5.9, "pop", 1.0), (9.0, "whoosh", 1.0)],
        20.0, 0.5)
    tags = [(c.tag, c.at_s) for c in cues]
    assert ("stinger", 5.4) in tags
    assert not any(t == "whoosh" and at == 5.0 for t, at in tags)
    assert ("whoosh", 9.0) in tags
    assert all(c.gain == pytest.approx(0.5) for c in cues)


# --- placement ---------------------------------------------------------------

def test_placement_puts_each_lane_on_its_own_track_and_reruns_replace(monkeypatch, tmp_path):
    _fake_synth(monkeypatch, tmp_path)
    monkeypatch.setattr(sound, "_probe_duration", lambda path: 0.0)
    timeline = _timeline()
    program = build_program(timeline)
    # Placement mechanics (loop/duck/fade), not sourcing policy: sfx_source is
    # pinned to "synth" so the faked library-empty SFX still resolve to a file,
    # the way they always did before "auto" stopped reaching for synth by
    # default (see test_sfx_source_auto_never_synthesises below).
    settings = PresentationSettings(music_volume=0.2, music_duck=0.9, sfx_source="synth")
    plan = sound.plan_sound(timeline, program, settings, "horror", 1,
                            broll_windows=[(3.0, 6.0)], popup_times=[10.0])

    counts = sound.apply_sound(timeline, plan, settings)
    assert counts == {"music": 1, "sfx": 2, "ambience": 1}   # whoosh in, pop
    music = [i for i in timeline.items if i.origin == sound.MUSIC_ORIGIN]
    assert len(music) == 1 and music[0].track == "A2"
    assert music[0].loop and music[0].duck == pytest.approx(0.9)
    assert music[0].volume == pytest.approx(0.2)
    assert music[0].audio_fade_in > 0 and music[0].audio_fade_out > 0
    assert music[0].timeline_end_frame == timeline.duration_frames
    sfx = [i for i in timeline.items if i.origin == sound.SFX_ORIGIN]
    assert {i.track for i in sfx} == {"A3"}
    assert all(not i.loop and i.duck == 0 for i in sfx)
    ambience = [i for i in timeline.items if i.origin == sound.AMBIENCE_ORIGIN]
    assert ambience[0].track == "A4" and ambience[0].loop
    audio_sources = [s for s in timeline.sources.values() if s.kind == "audio"]
    assert audio_sources and all(s.has_audio for s in audio_sources)

    # Running again replaces the lanes rather than doubling them.
    sound.apply_sound(timeline, plan, settings)
    assert len([i for i in timeline.items if i.origin == sound.MUSIC_ORIGIN]) == 1
    assert len([i for i in timeline.items if i.origin == sound.SFX_ORIGIN]) == 2


def test_a_later_planner_can_add_effects_without_stacking_on_existing_ones(monkeypatch, tmp_path):
    _fake_synth(monkeypatch, tmp_path)
    timeline = _timeline()
    settings = PresentationSettings(sfx_source="synth")
    plan = sound.plan_sound(timeline, build_program(timeline), settings, "horror", 1,
                            broll_windows=[(3.0, 6.0)])
    sound.apply_sound(timeline, plan, settings)
    added = sound.add_sfx_events(timeline, [(3.3, "stinger"), (12.0, "thunder"),
                                            (12.5, "boom")], settings)
    # 3.3 collides with the whoosh at 3.0; 12.5 with the thunder at 12.0.
    assert added == 1
    labels = sorted(i.label for i in timeline.items if i.origin == sound.SFX_ORIGIN)
    assert "SFX: thunder" in labels


def test_voice_master_respects_a_hand_set_master():
    timeline = _timeline()
    settings = PresentationSettings(voice_preset="podcast", loudness_lufs=-16)
    assert sound.apply_voice_master(timeline, settings) == "podcast"
    assert timeline.audio_master.origin == sound.MASTER_ORIGIN
    assert timeline.audio_master.loudness_lufs == -16
    assert timeline.audio_master.voice_compress == pytest.approx(0.65)

    timeline.audio_master = AudioMaster(voice_compress=0.9)     # the user's own
    assert sound.apply_voice_master(timeline, settings) is None
    assert timeline.audio_master.voice_compress == pytest.approx(0.9)

    timeline.audio_master = AudioMaster(origin=sound.MASTER_ORIGIN, loudness_lufs=-14)
    assert sound.apply_voice_master(timeline, PresentationSettings(voice_preset="off")) is None
    assert timeline.audio_master is None


# --- compiler ----------------------------------------------------------------

def _mix_timeline(**item_fields) -> Timeline:
    src = SourceFile(id="s", path="C:/m.mp4", duration_seconds=20, width=640, height=360)
    bed = SourceFile(id="m", path="C:/bed.mp3", duration_seconds=5, kind="audio",
                     has_audio=True, width=0, height=0)
    items = [
        TimelineItem(id="v", track="V1", source_id="s", source_start_frame=0,
                     source_end_frame=300, timeline_start_frame=0, timeline_end_frame=300),
        TimelineItem(id="a", track="A1", source_id="s", source_start_frame=0,
                     source_end_frame=300, timeline_start_frame=0, timeline_end_frame=300),
        TimelineItem(id="mu", track="A2", source_id="m", source_start_frame=0,
                     source_end_frame=300, timeline_start_frame=0, timeline_end_frame=300,
                     **item_fields),
    ]
    timeline = Timeline(sources={"s": src, "m": bed}, items=items)
    timeline.recalculate_duration()
    return timeline


def test_compiler_loops_fades_and_ducks_a_mix_item_against_the_voice():
    timeline = _mix_timeline(loop=True, duck=1.0, audio_fade_in=1.0, audio_fade_out=2.0,
                             volume=0.2)
    _, graph, _, final_a = FilterGraphCompiler(timeline).compile()
    assert "aloop=loop=3:size=240000" in graph
    assert "afade=t=in:st=0:d=1.000" in graph
    assert "afade=t=out:st=8.000:d=2.000" in graph
    assert "asplit=2[a1_main][a1_sc_0]" in graph
    assert "[aduck_in_0][a1_sc_0]sidechaincompress=" in graph
    assert "[a1_main][amix_0]amix=inputs=2" in graph
    assert final_a == "[mix_a]"


def test_compiler_leaves_an_unducked_item_alone():
    timeline = _mix_timeline()
    _, graph, _, _ = FilterGraphCompiler(timeline).compile()
    assert "sidechaincompress" not in graph
    assert "asplit" not in graph
    assert "aloop" not in graph


def test_voice_chain_runs_before_the_mix_and_loudness_after():
    timeline = _mix_timeline(duck=0.5)
    # Pinned to the "ffmpeg" engine on purpose: this test is about filter
    # ORDER (denoise -> mix -> loudness), and "afftdn=" is its marker for the
    # denoise stage. Leaving voice_enhance at its "auto" default would pick
    # RNNoise (arnndn=) whenever data/models/rnnoise/*.rnnn is present, same
    # engine choice this suite is not testing here.
    timeline.audio_master = AudioMaster(voice_denoise=0.5, voice_deess=0.5,
                                        voice_compress=0.5, loudness_lufs=-14,
                                        voice_enhance="ffmpeg")
    _, graph, _, final_a = FilterGraphCompiler(timeline).compile()
    voice_at = graph.index("afftdn=")
    mix_at = graph.index("amix=")
    loud_at = graph.index("loudnorm=I=-14.0")
    assert voice_at < mix_at < loud_at
    assert "deesser=" in graph and "acompressor=" in graph
    assert "loudnorm=I=-14.0:TP=-1.5:LRA=11,aresample=48000" in graph
    assert final_a == "[master_a]"
    # The sidechain is taken AFTER the voice treatment, so ducking follows the
    # cleaned voice, and the treated voice is what reaches the mix.
    assert "[voice_a]asplit=2" in graph


def test_a_neutral_master_compiles_to_nothing():
    timeline = _mix_timeline()
    timeline.audio_master = AudioMaster()
    _, graph, _, _ = FilterGraphCompiler(timeline).compile()
    assert "loudnorm" not in graph and "afftdn" not in graph


# --- voice presets / EQ / denoise engine --------------------------------------

@pytest.mark.parametrize("preset,expected_filters", [
    ("studio_mic", ["highpass=f=80", "equalizer=f=300", "equalizer=f=4000",
                    "treble=g=2.0:f=11000", "acompressor=", "alimiter="]),
    ("broadcast", ["highpass=f=90", "equalizer=f=3000", "acompressor="]),
    ("warm_radio", ["highpass=f=100", "equalizer=f=200", "lowpass=f=9000"]),
    ("rap_vocal", ["highpass=f=100", "equalizer=f=350", "equalizer=f=5000",
                  "treble=g=3.0:f=12000", "aexciter=", "mix=0.35"]),
    ("horror_intimate", ["highpass=f=60", "aecho="]),
    ("clean", ["highpass=f=70", "equalizer=f=300", "equalizer=f=4000"]),
    ("podcast", ["highpass=f=70", "acompressor="]),
    ("light", ["highpass=f=60", "equalizer=f=300"]),
])
def test_voice_chain_per_preset_carries_its_named_eq(preset, expected_filters):
    denoise, deess, compress = sound.VOICE_PRESETS[preset]
    master = AudioMaster(voice_denoise=denoise, voice_deess=deess, voice_compress=compress,
                         voice_eq_preset=preset, voice_enhance="ffmpeg")
    chain = render_audio.build_voice_chain(master)
    joined = ",".join(chain)
    for expected in expected_filters:
        assert expected in joined, f"{preset}: missing {expected!r} in {joined!r}"


def test_no_preset_afftdn_ever_exceeds_6db_by_default():
    for preset, (denoise, deess, compress) in sound.VOICE_PRESETS.items():
        master = AudioMaster(voice_denoise=denoise, voice_deess=deess, voice_compress=compress,
                             voice_eq_preset=preset, voice_enhance="ffmpeg")
        chain = render_audio.build_voice_chain(master)
        afftdn = next((f for f in chain if f.startswith("afftdn=")), None)
        assert afftdn is not None
        nr = float(afftdn.split("nr=")[1].split(":")[0])
        assert nr <= 6.0, f"{preset}: afftdn nr={nr} exceeds the 6dB ceiling"


def test_resolve_voice_enhance_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(render_audio, "deepfilter_available", lambda: False)
    monkeypatch.setattr(render_audio, "rnnoise_model_path", lambda: None)
    assert render_audio.resolve_voice_enhance("auto") == "ffmpeg"
    assert render_audio.resolve_voice_enhance("deepfilter") == "ffmpeg"
    assert render_audio.resolve_voice_enhance("rnnoise") == "ffmpeg"
    assert render_audio.resolve_voice_enhance("ffmpeg") == "ffmpeg"
    assert render_audio.resolve_voice_enhance("off") == "off"

    monkeypatch.setattr(render_audio, "deepfilter_available", lambda: True)
    assert render_audio.resolve_voice_enhance("auto") == "deepfilter"
    assert render_audio.resolve_voice_enhance("rnnoise") == "ffmpeg"  # still no model


def test_voice_fx_only_applies_inside_its_own_window():
    chain = render_audio.build_voice_fx_chain(
        [{"start_s": 10.0, "end_s": 12.5, "effect": "phone"}])
    joined = ",".join(chain)
    assert "highpass=f=300" in joined and "lowpass=f=3400" in joined
    assert "enable='between(t,10.000,12.500)'" in joined
    # An unknown effect name is silently ignored, not raised.
    assert render_audio.build_voice_fx_chain([{"start_s": 0, "end_s": 1, "effect": "nope"}]) == []


def test_voice_fx_windows_are_threaded_through_the_compiler():
    timeline = _mix_timeline()
    timeline.audio_master = AudioMaster(
        voice_fx=[{"start_s": 1.0, "end_s": 2.0, "effect": "reverb_hall"}])
    _, graph, _, _ = FilterGraphCompiler(timeline).compile()
    assert "enable='between(t,1.000,2.000)'" in graph


# --- two-pass loudness ---------------------------------------------------------

def test_loudness_chain_uses_single_pass_when_not_measured():
    master = AudioMaster(loudness_lufs=-14)
    chain = render_audio.build_loudness_chain(master)
    assert chain == ["loudnorm=I=-14.0:TP=-1.5:LRA=11", "aresample=48000"]


def test_loudness_chain_goes_linear_two_pass_when_measured():
    master = AudioMaster(loudness_lufs=-14)
    measured = {"input_i": -23.1, "input_tp": -3.2, "input_lra": 4.5, "input_thresh": -33.0,
               "target_offset": 0.4}
    chain = render_audio.build_loudness_chain(master, measured=measured)
    joined = ",".join(chain)
    assert "measured_I=-23.10" in joined and "linear=true" in joined
    assert "loudnorm=I=-14.0:TP=-1.5:LRA=11:measured_I=" in joined


def test_loudness_measurement_failure_falls_back_to_none(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg is not installed")
    monkeypatch.setattr(render_audio.subprocess, "run", boom)
    result = render_audio.measure_loudness_stats(
        "ffmpeg", ["-i", "in.mp4"], "[a]anull[b]", "[b]", -14.0, -1.5, timeout=1)
    assert result is None


# --- sound source policy --------------------------------------------------------

def test_sfx_source_auto_never_synthesises(monkeypatch):
    monkeypatch.setattr(sound, "find_sfx", lambda tag, rng: None)
    monkeypatch.setattr(sound, "_generated_sfx_path", lambda tag: None)
    monkeypatch.setattr(sound, "synth_path", lambda tag: Path("should_not_be_used.wav"))
    rng = random.Random(0)
    path, label = sound._resolve_sfx("whoosh", rng, "auto")
    assert path is None and label == "off"

    # "synth" is the explicit opt-in that still reaches the old fallback.
    path, label = sound._resolve_sfx("whoosh", rng, "synth")
    assert path == Path("should_not_be_used.wav") and label == "synth"


def test_sfx_source_auto_prefers_the_library_then_a_generated_cache(monkeypatch):
    rng = random.Random(0)
    monkeypatch.setattr(sound, "find_sfx", lambda tag, r: Path("lib/whoosh.wav"))
    path, label = sound._resolve_sfx("whoosh", rng, "auto")
    assert path == Path("lib/whoosh.wav") and label == "library"

    monkeypatch.setattr(sound, "find_sfx", lambda tag, r: None)
    monkeypatch.setattr(sound, "_generated_sfx_path", lambda tag: Path("gen/whoosh.wav"))
    path, label = sound._resolve_sfx("whoosh", rng, "auto")
    assert path == Path("gen/whoosh.wav") and label == "generated"


# --- end to end ----------------------------------------------------------------

def test_a_ducked_bed_renders_under_the_voice(tmp_path):
    """Real FFmpeg: a loud bed ducked under a sine 'voice' must come out quieter
    while the voice plays than in the silence after it."""
    import asyncio
    from backend.config import FFMPEG_BIN
    from backend.render.runner import render_timeline_async

    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6,volume=0.5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not source.exists():
        pytest.skip("ffmpeg unavailable")
    bed = sound.synth_path("drone")
    if bed is None:
        pytest.skip("synthesis unavailable")

    # Voice for the first 3 s only, then silence — the bed keeps going.
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.45} for i in range(6)]
    timeline = build_timeline_from_transcript(str(source), 6.0, words, fps_num=30, fps_den=1,
                                              width=320, height=180, pause_padding_seconds=0.0)
    # Extend the programme with a muted clip so there is a stretch with no voice.
    tail = clip_ops.add_media_item(timeline, next(iter(timeline.sources)), "V2",
                                   timeline.duration_frames, 90, 180)
    tail.mute = True
    timeline.recalculate_duration()
    source_id = sound._register(timeline, str(bed), sound.synth_duration("drone"))
    music = clip_ops.add_media_item(timeline, source_id, "A2", 0, 0, timeline.duration_frames)
    music.loop = True
    music.duck = 1.0
    music.volume = 0.8

    out = str(tmp_path / "out.mp4")
    asyncio.run(render_timeline_async(timeline, out, prefer_nvenc=False))

    def rms_db(start: float, end: float) -> float:
        result = subprocess.run(
            [FFMPEG_BIN, "-v", "info", "-ss", str(start), "-t", str(end - start), "-i", out,
             "-af", "highpass=f=20,lowpass=f=120,astats=measure_overall=RMS_level:measure_perchannel=none",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stderr.splitlines():
            if "RMS level dB" in line:
                return float(line.split(":")[-1])
        return 0.0

    # The bed lives under 120 Hz where the 440 Hz voice does not; compare its
    # level while the voice plays (ducked) with the stretch after.
    ducked = rms_db(0.5, 2.5)
    free = rms_db(3.8, 5.6)
    assert free - ducked > 4.0, f"bed under voice {ducked} dB vs free {free} dB"


# --- the look (grade + chapter transitions) ----------------------------------

def test_genre_grade_is_applied_and_replaced_but_never_overwrites_the_users():
    from backend.presentation import look
    from backend.timeline.schema import ColorGrade
    timeline = _timeline()
    settings = PresentationSettings()
    assert look.apply_genre_grade(timeline, settings, "horror") == "moody"
    assert timeline.master_color.preset == "presentation:moody"
    assert timeline.master_color.vignette == pytest.approx(0.5)
    assert timeline.master_color.saturation < 1.0

    assert look.apply_genre_grade(timeline, settings, "comedy") == "vibrant"
    assert timeline.master_color.preset == "presentation:vibrant"

    assert look.apply_genre_grade(timeline, PresentationSettings(grade="off"), "comedy") is None
    assert timeline.master_color is None

    timeline.master_color = ColorGrade(preset="mine", contrast=1.4)
    assert look.apply_genre_grade(timeline, settings, "horror") is None
    assert timeline.master_color.contrast == pytest.approx(1.4)


def test_topic_transitions_land_only_on_joins_and_keep_the_users_choice():
    from backend.presentation import look
    from backend.presentation.models import Topic
    from backend.timeline.schema import Transition
    # Six 2s clips with cuts at 2, 4, 6, 8, 10 s.
    words = []
    for n in range(6):
        for k in range(4):
            words.append({"word": f"w{n}{k}", "start": n * 3 + k * 0.5,
                          "end": n * 3 + k * 0.5 + 0.4})
    timeline = build_timeline_from_transcript(
        "C:/media/talk.mp4", 20.0, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, 20.0)], pause_padding_seconds=0.0, max_pause_seconds=0.3)
    v1 = sorted([i for i in timeline.items if i.track == "V1"],
                key=lambda i: i.timeline_start_frame)
    assert len(v1) == 6
    joins = [i.timeline_start_frame / 30 for i in v1[1:]]
    # The user already chose a transition on the third join.
    v1[3].transition = Transition(type="pixelize", duration=0.3)

    topics = [Topic(start_s=0.0, end_s=joins[0], topic="a"),
              Topic(start_s=joins[1] + 0.2, end_s=joins[2], topic="b"),   # near a join
              Topic(start_s=joins[2] - 1.0, end_s=joins[3], topic="c"),   # mid-take
              Topic(start_s=joins[2], end_s=joins[4], topic="d"),         # user's join
              Topic(start_s=joins[4] - 0.1, end_s=20.0, topic="e")]
    count = look.apply_topic_transitions(timeline, topics, "horror", PresentationSettings())
    assert count == 2
    assert v1[2].transition.type == "fadeblack" and v1[2].transition.origin == "presentation"
    assert v1[1].transition is None
    assert v1[3].transition.type == "pixelize" and v1[3].transition.origin is None
    assert v1[5].transition.type == "fadeblack"

    # A re-run for another genre replaces only its own.
    look.apply_topic_transitions(timeline, topics, "tech", PresentationSettings())
    assert v1[2].transition.type == "hlwind"
    assert v1[3].transition.type == "pixelize"
    assert look.clear_topic_transitions(timeline) == 2
    assert v1[3].transition is not None


def test_generated_static_is_rejected_but_a_tonal_sweep_is_kept(tmp_path):
    """The hiss in a real render was two Stable Audio "whoosh"/"pop" clips that
    were really broadband static; the gate must refuse those and keep sound."""
    from config import FFMPEG_BIN
    import subprocess as sp
    noise, sweep = tmp_path / "noise.wav", tmp_path / "sweep.wav"
    made_noise = sp.run([FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
                         "anoisesrc=color=white:duration=1:amplitude=0.5,highpass=f=4000",
                         str(noise)])
    made_sweep = sp.run([FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
                         "aevalsrc='0.5*sin(2*PI*(200*t+600*t*t))':d=1:s=44100", str(sweep)])
    if made_noise.returncode or made_sweep.returncode:
        pytest.skip("ffmpeg unavailable")
    assert sound.sfx_is_noise(noise) is True
    assert sound.sfx_is_noise(sweep) is False


def test_sfx_are_capped_per_minute_of_programme():
    events = [(i * 1.5, "whoosh", 1.0) for i in range(400)]   # one every 1.5s for 10 min
    kept = sound._space_sfx(events, 600.0, 1.0)
    assert len(kept) == int(sound.MAX_SFX_PER_MINUTE * 10)
    gaps = [b.at_s - a.at_s for a, b in zip(kept, kept[1:])]
    assert min(gaps) >= sound.MIN_SFX_GAP_S



def _lib(tmp_path, folder, name, entry):
    import json
    d = tmp_path / folder
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(b"RIFF" + b"0" * 2000)
    mf = d / "manifest.json"
    data = json.loads(mf.read_text()) if mf.exists() else {}
    data[name] = entry
    mf.write_text(json.dumps(data))
    return d / name


def test_general_tracks_made_for_the_genre_join_its_pool(tmp_path):
    import random
    from presentation import sound
    _lib(tmp_path, "vlog", "calm.wav", {"license": "generated", "mood": ["calm"]})
    sad = _lib(tmp_path, "general", "sad.wav",
               {"license": "generated", "mood": ["soft"], "genre": ["general", "vlog"]})
    _lib(tmp_path, "general", "horror.wav",
         {"license": "generated", "mood": ["soft"], "genre": ["horror"]})
    assert sound.find_music("vlog", random.Random(1), mood="soft", library=tmp_path) == sad
    assert sound.has_music_for("vlog", "soft", library=tmp_path)
    assert not sound.has_music_for("vlog", "intense", library=tmp_path)


def test_music_cue_sections_follow_script_cues_and_brief():
    from presentation import sound
    from presentation.models import PresentationSettings
    settings = PresentationSettings(music_brief="calm lo-fi")
    cues = [(60.0, "sad violin"), (150.0, "off"), (160.0, "epic orchestral")]
    sections = sound.music_cue_sections(settings, cues, 300.0)
    assert [(s, e) for s, e, _, _ in sections] == [(0.0, 60.0), (60.0, 150.0), (150.0, 300.0)]
    assert sections[0][2]["style"] == "lofi_calm"
    # "off" at 150 and a cue 10 s later merge: the later one wins.
    assert sections[2][2]["music_genres"] == ["orchestral"]
    assert sound.music_cue_sections(PresentationSettings(music_cues=False), cues, 300.0) == []
    assert sound.music_cue_sections(PresentationSettings(), [], 300.0) == []


def test_named_music_is_placed_per_cue(tmp_path, monkeypatch):
    import random
    from presentation import music_gen, sound
    from presentation.models import PresentationSettings
    recipe = music_gen.parse_music_cue("sad violin")
    track = _lib(tmp_path, "vlog", "cue.wav", {"license": "generated",
                                               "cue_key": music_gen.cue_key(recipe, "ace_step")})
    monkeypatch.setattr(music_gen, "MUSIC_DIR", tmp_path)
    monkeypatch.setattr(music_gen, "find_cue_track",
                        lambda r, e, library=tmp_path: (track if music_gen.cue_key(r, e) ==
                                                        music_gen.cue_key(recipe, "ace_step") else None))
    settings = PresentationSettings()
    sections = sound.music_cue_sections(settings, [(0.0, "sad violin"), (100.0, "off")], 200.0)
    cues = sound._plan_cue_music("vlog", settings, random.Random(1), 200.0, sections)
    assert len(cues) == 1 and cues[0].path == str(track) and cues[0].source == "cue"
    assert cues[0].end_s <= 101.0


def test_whooshes_mark_scene_changes_not_every_cutaway():
    timeline = _timeline()
    windows = [(float(t), float(t) + 3.0) for t in range(2, 200, 8)]   # a cutaway every 8 s
    plan = sound.plan_sound(timeline, build_program(timeline), PresentationSettings(sfx_source="synth"),
                            "horror", 1, broll_windows=windows)
    whooshes = sorted(c.at_s for c in plan.sfx if c.tag == "whoosh")
    assert all(b - a >= sound.WHOOSH_MIN_GAP_S for a, b in zip(whooshes, whooshes[1:]))
    assert not any(c.tag == "whoosh_soft" for c in plan.sfx)


def _chosen_library(tmp_path):
    lib = tmp_path / "music"
    (lib / "uploads").mkdir(parents=True)
    for name in ("rain.flac", "piano.flac", "ripped.mp3"):
        (lib / "uploads" / name).write_bytes(b"x")
    (lib / "uploads" / "manifest.json").write_text(
        '{"rain.flac": {"license": "own"}, "piano.flac": {"license": "cc0"}}')
    return lib


def test_chosen_tracks_keep_their_levels_and_skip_unlicensed_or_outside_files(tmp_path):
    lib = _chosen_library(tmp_path)
    settings = PresentationSettings(music_tracks=[
        {"file": "uploads/rain.flac", "volume_db": -20, "duck": True},
        {"file": "uploads/piano.flac", "volume_db": -99},          # clamped to -60
        {"file": "uploads/ripped.mp3", "volume_db": -10},          # no licence on record
        {"file": "../outside.flac"}, {"file": ""}])
    chosen = sound.chosen_tracks(settings, library=lib)
    assert [p.name for p, _, _ in chosen] == ["rain.flac", "piano.flac"]
    assert chosen[0][1] == pytest.approx(0.1) and chosen[0][2] is True
    assert chosen[1][1] == pytest.approx(10 ** (-60 / 20)) and chosen[1][2] is False


def test_chosen_tracks_together_layer_under_the_whole_video_at_fixed_levels(monkeypatch, tmp_path):
    lib = _chosen_library(tmp_path)
    monkeypatch.setattr(sound, "MUSIC_DIR", lib)
    monkeypatch.setattr(sound, "_probe_duration", lambda path: 30.0)
    timeline = _timeline(60.0)
    program = build_program(timeline)
    settings = PresentationSettings(music_duck=0.8, sfx=False, ambience=False, music_tracks=[
        {"file": "uploads/rain.flac", "volume_db": -20, "duck": False},
        {"file": "uploads/piano.flac", "volume_db": -6, "duck": True}])
    # A brief/cues would otherwise pick or generate music: the picks win.
    settings.music_brief = "sad violin"
    plan = sound.plan_sound(timeline, program, settings, "horror", 1,
                            music_cues=[(10.0, "tense drums")])
    assert [Path(c.path).name for c in plan.music] == ["rain.flac", "piano.flac"]
    assert all(c.start_s == 0.0 and c.end_s == pytest.approx(program.duration_s) for c in plan.music)
    assert plan.music[0].gain == pytest.approx(0.1) and plan.music[0].duck == 0.0
    assert plan.music[1].gain == pytest.approx(10 ** (-6 / 20)) and plan.music[1].duck == pytest.approx(0.8)
    sound.apply_sound(timeline, plan, settings)
    items = [i for i in timeline.items if i.origin == sound.MUSIC_ORIGIN]
    assert len(items) == 2 and {i.track for i in items} == {"A2"}
    assert sorted(round(i.volume, 3) for i in items) == [0.1, round(10 ** (-6 / 20), 3)]


def test_chosen_tracks_in_sequence_play_in_turn_and_loop_the_list(monkeypatch, tmp_path):
    lib = _chosen_library(tmp_path)
    monkeypatch.setattr(sound, "_probe_duration", lambda path: 20.0)
    settings = PresentationSettings(music_tracks_mode="sequence", music_tracks=[
        {"file": "uploads/rain.flac", "volume_db": -20}, {"file": "uploads/piano.flac", "volume_db": -14}])
    chosen = sound.chosen_tracks(settings, library=lib)
    cues = sound._plan_chosen_music(settings, 50.0, chosen)
    assert [Path(c.path).name for c in cues] == ["rain.flac", "piano.flac", "rain.flac"]
    # Each plays its own length, overlapping the next by the crossfade.
    assert cues[0].start_s == 0.0 and cues[0].end_s == pytest.approx(20.0)
    assert cues[1].start_s == pytest.approx(20.0 - sound.MUSIC_CROSSFADE_S)
    assert cues[-1].end_s == pytest.approx(50.0)
    assert cues[1].gain == pytest.approx(10 ** (-14 / 20))


def test_picked_tracks_that_cannot_be_used_leave_silence_not_a_guess(monkeypatch, tmp_path):
    lib = _chosen_library(tmp_path)
    monkeypatch.setattr(sound, "MUSIC_DIR", lib)
    timeline = _timeline()
    settings = PresentationSettings(sfx=False, ambience=False,
                                    music_tracks=[{"file": "uploads/ripped.mp3", "volume_db": -20}])
    plan = sound.plan_sound(timeline, build_program(timeline), settings, "horror", 1)
    assert plan.music == [] and "music_missing" in plan.notes
