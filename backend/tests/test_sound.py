"""Sound design: the music bed, effects, ambience and voice master.

The checks here are about what lands on the timeline and what the compiler
makes of it. Every sound the pass can fall back on is synthesised with
FFmpeg, so a machine with an empty library must still get a soundtrack, and
the mix must never let the bed sit on top of the voice.
"""

import random
import shutil
import subprocess

import pytest

from backend.presentation import sound
from backend.presentation.models import PresentationSettings, SfxCue
from backend.presentation.program import build_program
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
    (lib / "manifest.json").write_text('{"loose.wav": {"genre": ["finance"]}}')
    rng = random.Random(1)

    assert sound.find_music("horror", rng, library=lib).name == "creep.mp3"
    assert sound.find_music("comedy", rng, library=lib).name == "bed.mp3"
    shutil.rmtree(lib / "general")
    assert sound.find_music("finance", rng, library=lib).name == "loose.wav"
    assert sound.find_music("comedy", rng, library=lib) is None


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
    assert tags.count("whoosh") == 2
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
    settings = PresentationSettings(music_volume=0.2, music_duck=0.9)
    plan = sound.plan_sound(timeline, program, settings, "horror", 1,
                            broll_windows=[(3.0, 6.0)], popup_times=[10.0])

    counts = sound.apply_sound(timeline, plan, settings)
    assert counts == {"music": 1, "sfx": 3, "ambience": 1}
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
    assert len([i for i in timeline.items if i.origin == sound.SFX_ORIGIN]) == 3


def test_a_later_planner_can_add_effects_without_stacking_on_existing_ones(monkeypatch, tmp_path):
    _fake_synth(monkeypatch, tmp_path)
    timeline = _timeline()
    settings = PresentationSettings()
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
    timeline.audio_master = AudioMaster(voice_denoise=0.5, voice_deess=0.5,
                                        voice_compress=0.5, loudness_lufs=-14)
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
