"""Audio racks in the render compiler (AUDIO_SUITE_PLAN.md §2.6, brief A1a).

The A1 rack follows the voice chain, a clip rack sits in its item's own path
before the volume and fades, the MASTER rack goes after the mix and before the
loudness stage, and a source with `audio_override` is heard from that WAV.
A project without any of it must compile to exactly the graph it did before
racks existed: GOLDEN below was captured from the pre-A1a compiler
(2026-10-09). If a deliberate compiler change alters these graphs, recapture
them from the changed compiler with racks empty.

Rack assertions rely only on the §2.6 contract (instances are named
`<filter>@<effect.id>_<k>`), so they hold with the real `audio.compile` or
with the compiler's amplify-only fallback; the measured gains force the
fallback, whose `amplify` is a plain `volume` in dB.
"""

import array
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from config import FFMPEG_BIN
from render.compiler import FilterGraphCompiler, _compile_rack_fallback, rack_filters
from timeline.schema import AudioEffect, AudioMaster, SourceFile, Timeline, TimelineItem

FPS = 30


def _source(sid: str, path: str, *, kind: str = "video", duration: float = 60.0,
            width: int = 1920, height: int = 1080) -> SourceFile:
    return SourceFile(id=sid, path=path, duration_seconds=duration, width=width,
                      height=height, kind=kind, has_audio=True)


def old_project(segments: int = 2, *, mix: bool = True, master: bool = True) -> Timeline:
    """A project as it was saved before racks existed: no racks, no overrides."""
    sources = {"main": _source("main", "C:/media/talk.mp4", duration=120.0)}
    items = []
    for k in range(segments):
        src_start, tl_start = k * 60, k * 45
        for track in ("V1", "A1"):
            items.append(TimelineItem(
                id=f"{track.lower()}_{k}", track=track, source_id="main",
                source_start_frame=src_start, source_end_frame=src_start + 45,
                timeline_start_frame=tl_start, timeline_end_frame=tl_start + 45,
                volume=0.8 if (track == "A1" and k == 0) else 1.0))
    total = segments * 45
    if mix:
        sources["bed"] = _source("bed", "C:/media/bed.mp3", kind="audio",
                                 duration=30.0, width=0, height=0)
        sources["whoosh"] = _source("whoosh", "C:/media/whoosh.wav", kind="audio",
                                    duration=2.0, width=0, height=0)
        items.append(TimelineItem(
            id="bed_0", track="A2", source_id="bed", source_start_frame=0,
            source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total,
            volume=0.35, loop=True, duck=0.5, audio_fade_in=1.0, audio_fade_out=1.5))
        items.append(TimelineItem(
            id="whoosh_0", track="A3", source_id="whoosh", source_start_frame=0,
            source_end_frame=30, timeline_start_frame=40, timeline_end_frame=70))
    tl = Timeline(fps_num=FPS, fps_den=1, sources=sources, items=items,
                  duration_frames=total)
    if master:
        tl.audio_master = AudioMaster(voice_gain_db=2.0, voice_deess=0.3,
                                      voice_compress=0.5, loudness_lufs=-14.0)
    return tl


GOLDEN_BUILDERS = {
    "mixed": lambda: old_project(2),
    "seeked": lambda: old_project(12),
    "plain": lambda: old_project(2, mix=False, master=False),
}

# Captured from the pre-A1a compiler on 2026-10-09 (racks did not exist yet).
GOLDEN = {
    'mixed': {
        'inputs': [
            '-i', 'C:/media/talk.mp4', '-threads', '1',
            '-i', 'C:/media/bed.mp3', '-threads', '1',
            '-i', 'C:/media/whoosh.wav',
        ],
        'graph': [
            '[0:v]trim=start=0.000:end=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_0]',
            '[0:v]trim=start=2.000:end=3.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_1]',
            '[v1_0][v1_1]concat=n=2:v=1:a=0[vgrp_0]',
            '[0:a]atrim=start=0.000:end=1.500,asetpts=PTS-STARTPTS,volume=0.800,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_0]',
            '[0:a]atrim=start=2.000:end=3.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_1]',
            '[a1_0][a1_1]concat=n=2:v=0:a=1[agrp_0]',
            '[agrp_0]volume=2.00dB,deesser=i=0.33:m=0.5:f=0.5,acompressor=threshold=-18.0dB:ratio=3.25:attack=8:release=140:knee=4:makeup=1.60[voice_a]',
            '[voice_a]asplit=2[a1_main][a1_sc_0]',
            '[1:a]aresample=48000,aloop=loop=1:size=1440000,atrim=start=0.000:end=3.000,asetpts=PTS-STARTPTS,volume=0.350,afade=t=in:st=0:d=1.000,afade=t=out:st=1.500:d=1.500,adelay=0:all=1[aduck_in_0]',
            '[aduck_in_0][a1_sc_0]sidechaincompress=threshold=0.02:ratio=6.50:attack=25:release=450:knee=6:level_sc=1:makeup=1[amix_0]',
            '[2:a]atrim=start=0.000:end=1.000,asetpts=PTS-STARTPTS,adelay=1333:all=1[amix_1]',
            '[a1_main][amix_0][amix_1]amix=inputs=3:normalize=0:dropout_transition=0[mix_a]',
            '[mix_a]loudnorm=I=-14.0:TP=-1.5:LRA=11,aresample=48000[master_a]',
        ],
        'v': '[vgrp_0]',
        'a': '[master_a]',
    },
    'plain': {
        'inputs': [
            '-i', 'C:/media/talk.mp4',
        ],
        'graph': [
            '[0:v]trim=start=0.000:end=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_0]',
            '[0:v]trim=start=2.000:end=3.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_1]',
            '[v1_0][v1_1]concat=n=2:v=1:a=0[vgrp_0]',
            '[0:a]atrim=start=0.000:end=1.500,asetpts=PTS-STARTPTS,volume=0.800,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_0]',
            '[0:a]atrim=start=2.000:end=3.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_1]',
            '[a1_0][a1_1]concat=n=2:v=0:a=1[agrp_0]',
        ],
        'v': '[vgrp_0]',
        'a': '[agrp_0]',
    },
    'seeked': {
        'inputs': [
            '-i', 'C:/media/talk.mp4', '-threads', '1',
            '-i', 'C:/media/bed.mp3', '-threads', '1',
            '-i', 'C:/media/whoosh.wav', '-ss', '0.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '2.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '4.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '6.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '8.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '10.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '12.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '14.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '16.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '18.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '20.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4', '-ss', '22.000',
            '-t', '1.600', '-threads', '1',
            '-i', 'C:/media/talk.mp4',
        ],
        'graph': [
            '[3:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_0]',
            '[4:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_1]',
            '[5:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_2]',
            '[6:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_3]',
            '[7:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_4]',
            '[8:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_5]',
            '[9:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_6]',
            '[10:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_7]',
            '[11:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_8]',
            '[12:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_9]',
            '[13:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_10]',
            '[14:v]trim=duration=1.500,setpts=PTS-STARTPTS,fps=30.00000,setsar=1,format=yuv420p[v1_11]',
            '[v1_0][v1_1][v1_2][v1_3][v1_4][v1_5][v1_6][v1_7][v1_8][v1_9][v1_10][v1_11]concat=n=12:v=1:a=0[vgrp_0]',
            '[3:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,volume=0.800,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_0]',
            '[4:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_1]',
            '[5:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_2]',
            '[6:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_3]',
            '[7:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_4]',
            '[8:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_5]',
            '[9:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_6]',
            '[10:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_7]',
            '[11:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_8]',
            '[12:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_9]',
            '[13:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_10]',
            '[14:a]atrim=duration=1.500,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=1.492:d=0.008:curve=tri[a1_11]',
            '[a1_0][a1_1][a1_2][a1_3][a1_4][a1_5][a1_6][a1_7][a1_8][a1_9][a1_10][a1_11]concat=n=12:v=0:a=1[agrp_0]',
            '[agrp_0]volume=2.00dB,deesser=i=0.33:m=0.5:f=0.5,acompressor=threshold=-18.0dB:ratio=3.25:attack=8:release=140:knee=4:makeup=1.60[voice_a]',
            '[voice_a]asplit=2[a1_main][a1_sc_0]',
            '[1:a]aresample=48000,aloop=loop=1:size=1440000,atrim=start=0.000:end=18.000,asetpts=PTS-STARTPTS,volume=0.350,afade=t=in:st=0:d=1.000,afade=t=out:st=16.500:d=1.500,adelay=0:all=1[aduck_in_0]',
            '[aduck_in_0][a1_sc_0]sidechaincompress=threshold=0.02:ratio=6.50:attack=25:release=450:knee=6:level_sc=1:makeup=1[amix_0]',
            '[2:a]atrim=start=0.000:end=1.000,asetpts=PTS-STARTPTS,adelay=1333:all=1[amix_1]',
            '[a1_main][amix_0][amix_1]amix=inputs=3:normalize=0:dropout_transition=0[mix_a]',
            '[mix_a]loudnorm=I=-14.0:TP=-1.5:LRA=11,aresample=48000[master_a]',
        ],
        'v': '[vgrp_0]',
        'a': '[master_a]',
    },
}


def _golden(name: str):
    g = GOLDEN[name]
    return g["inputs"], ";".join(g["graph"]), g["v"], g["a"]


def _amp(effect_id: str, gain_db: float = 3.0, **extra) -> AudioEffect:
    return AudioEffect(id=effect_id, effect="amplify", params={"gain_db": gain_db}, **extra)


def _item(tl: Timeline, item_id: str) -> TimelineItem:
    return next(i for i in tl.items if i.id == item_id)


def _statement(graph: str, needle: str) -> str:
    found = [s for s in graph.split(";") if needle in s]
    assert len(found) == 1, (needle, found)
    return found[0]


def _producing(graph: str, label: str) -> str:
    """The one statement whose output is `label`."""
    found = [s for s in graph.split(";") if s.endswith(label)]
    assert len(found) == 1, (label, found)
    return found[0]


@pytest.fixture
def no_audio_compile(monkeypatch):
    """Make `from audio.compile import compile_rack` fail, as before A1b."""
    monkeypatch.setitem(sys.modules, "audio.compile", None)


# --- golden: old projects compile byte-identically --------------------------

@pytest.mark.parametrize("name", sorted(GOLDEN_BUILDERS))
def test_old_project_compiles_to_the_pre_rack_graph(name):
    compiled = FilterGraphCompiler(GOLDEN_BUILDERS[name]()).compile()
    assert compiled == _golden(name)
    assert "@" not in compiled[1]


@pytest.mark.parametrize("name", sorted(GOLDEN_BUILDERS))
def test_empty_racks_and_no_override_compile_to_the_same_graph(name):
    tl = GOLDEN_BUILDERS[name]()
    tl.audio_racks = {"A1": [], "MASTER": [], "A2": []}
    for item in tl.items:
        item.audio_effects = []
    for src in tl.sources.values():
        src.audio_override = None
        src.audio_history = []
    assert FilterGraphCompiler(tl).compile() == _golden(name)


def test_disabled_effects_compile_to_the_same_graph():
    tl = old_project(2)
    tl.audio_racks = {"A1": [_amp("fx_off1", enabled=False)],
                      "MASTER": [_amp("fx_off2", enabled=False)]}
    _item(tl, "bed_0").audio_effects = [_amp("fx_off3", enabled=False)]
    assert FilterGraphCompiler(tl).compile() == _golden("mixed")


def test_a1_clip_rack_is_ignored():
    """A1 clips are rebuilt from the transcript; their racks never compile."""
    tl = old_project(2)
    _item(tl, "a1_0").audio_effects = [_amp("fx_a1clip")]
    assert FilterGraphCompiler(tl).compile() == _golden("mixed")


# --- each rack in its place ----------------------------------------------------

def test_a1_rack_follows_the_voice_chain():
    tl = old_project(2)
    tl.audio_racks = {"A1": [_amp("fx_a1")]}
    _inputs, graph, _v, a = FilterGraphCompiler(tl).compile()
    voice = _statement(graph, "@fx_a1_0")
    assert voice.startswith("[agrp_0]") and voice.endswith("[voice_a]")
    assert voice.index("acompressor=") < voice.index("@fx_a1_0")
    # The voice (with its rack) is what gets split for ducking and mixed.
    assert "[voice_a]asplit=2[a1_main][a1_sc_0]" in graph.split(";")
    assert a == "[master_a]"


def test_a1_rack_alone_without_a_voice_master():
    tl = old_project(2, mix=False, master=False)
    tl.audio_racks = {"A1": [_amp("fx_a1")]}
    _inputs, graph, _v, a = FilterGraphCompiler(tl).compile()
    statement = _statement(graph, "@fx_a1_0")
    assert statement.startswith("[agrp_0]") and statement.endswith("[voice_a]")
    assert a == "[voice_a]"


def test_clip_rack_sits_before_the_items_volume_and_fades():
    tl = old_project(2)
    _item(tl, "bed_0").audio_effects = [_amp("fx_bed", -6.0)]
    _inputs, graph, _v, _a = FilterGraphCompiler(tl).compile()
    bed = _statement(graph, "@fx_bed_0")
    assert bed.endswith("[aduck_in_0]")
    order = [bed.index(s) for s in ("asetpts=PTS-STARTPTS", "@fx_bed_0", "volume=0.350",
                                    "afade=t=in", "afade=t=out", "adelay=")]
    assert order == sorted(order)
    # Only that item carries it.
    assert "@fx_bed_0" not in _producing(graph, "[amix_1]")


def test_master_rack_goes_after_the_mix_and_before_loudness():
    tl = old_project(2)
    tl.audio_racks = {"MASTER": [_amp("fx_master", -1.5)]}
    _inputs, graph, _v, a = FilterGraphCompiler(tl).compile()
    statements = graph.split(";")
    rack = _statement(graph, "@fx_master_0")
    assert rack.startswith("[mix_a]") and rack.endswith("[rack_master_a]")
    loud = _statement(graph, "loudnorm=")
    assert loud.startswith("[rack_master_a]") and loud.endswith("[master_a]")
    assert statements.index(rack) == statements.index(loud) - 1
    assert a == "[master_a]"


def test_master_rack_without_a_mix_takes_the_voice():
    tl = old_project(2, mix=False, master=True)
    tl.audio_racks = {"MASTER": [_amp("fx_master")]}
    _inputs, graph, _v, a = FilterGraphCompiler(tl).compile()
    rack = _statement(graph, "@fx_master_0")
    assert rack.startswith("[voice_a]") and rack.endswith("[rack_master_a]")
    assert _statement(graph, "loudnorm=").startswith("[rack_master_a]")
    assert a == "[master_a]"


def test_all_three_racks_each_compile_once_in_their_own_place():
    tl = old_project(2)
    tl.audio_racks = {"A1": [_amp("fx_a1")], "MASTER": [_amp("fx_master")]}
    _item(tl, "bed_0").audio_effects = [_amp("fx_bed")]
    _inputs, graph, _v, _a = FilterGraphCompiler(tl).compile()
    assert _statement(graph, "@fx_a1_0").endswith("[voice_a]")
    assert _statement(graph, "@fx_bed_0").endswith("[aduck_in_0]")
    assert _statement(graph, "@fx_master_0").endswith("[rack_master_a]")


# --- audio_override ------------------------------------------------------------

def test_audio_override_replaces_the_source_audio_in_every_path():
    tl = old_project(2)
    tl.sources["main"].audio_override = "C:/proj/audio/versions/1_noise_reduction.wav"
    tl.sources["bed"].audio_override = "C:/proj/audio/versions/2_noise_reduction.wav"
    inputs, graph, _v, _a = FilterGraphCompiler(tl).compile()
    golden_inputs = _golden("mixed")[0]
    assert inputs[:len(golden_inputs)] == golden_inputs
    assert inputs[len(golden_inputs):] == [
        "-threads", "1", "-i", "C:/proj/audio/versions/1_noise_reduction.wav",
        "-threads", "1", "-i", "C:/proj/audio/versions/2_noise_reduction.wav"]
    # Inputs 0..2 are the sources; 3 and 4 the two overrides.
    assert "[0:a]" not in graph and "[1:a]" not in graph
    assert _producing(graph, "[a1_0]").startswith("[3:a]atrim=start=0.000:end=1.500")
    assert _producing(graph, "[a1_1]").startswith("[3:a]atrim=start=2.000:end=3.500")
    assert _producing(graph, "[aduck_in_0]").startswith("[4:a]")
    assert _producing(graph, "[amix_1]").startswith("[2:a]")      # no override
    assert _producing(graph, "[v1_0]").startswith("[0:v]")        # picture untouched


def test_audio_override_with_seeked_segment_inputs():
    plain_inputs = FilterGraphCompiler(old_project(12, mix=False, master=False)).compile()[0]
    tl = old_project(12, mix=False, master=False)
    tl.sources["main"].audio_override = "C:/proj/audio/versions/1_nr.wav"
    inputs, graph, _v, _a = FilterGraphCompiler(tl).compile()
    # One extra input in all: the override. No seeked inputs for the A1 trims.
    assert inputs.count("-i") == plain_inputs.count("-i") + 1
    a1 = [s for s in graph.split(";") if "[a1_" in s and "concat" not in s]
    assert len(a1) == 12
    assert all(s.startswith("[1:a]atrim=start=") for s in a1)
    assert "atrim=duration=" not in graph
    assert "trim=duration=" in graph                               # V1 still seeked


# --- the rack compile seam -----------------------------------------------------

def test_fallback_compiles_amplify_only():
    rack = _compile_rack_fallback([
        _amp("fx_1", 3.0),
        _amp("fx_2", -1.5, enabled=False),
        AudioEffect(id="fx_3", effect="parametric_eq", params={}),
        _amp("fx_4", 0.5),
    ], scope="A1")
    assert rack.filters == ["volume@fx_1_0=volume=3dB", "volume@fx_4_0=volume=0.5dB"]
    assert rack.chain == "volume@fx_1_0=volume=3dB,volume@fx_4_0=volume=0.5dB"


def test_rack_filters_falls_back_without_audio_compile(no_audio_compile):
    assert rack_filters([_amp("fx_1", 6.0)], scope="A1") == ["volume@fx_1_0=volume=6dB"]
    assert rack_filters([], scope="A1") == []
    assert rack_filters(None, scope="MASTER") == []


# --- through a real ffmpeg -------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    result = subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *args],
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-800:]


def _tone(path: Path, frequency: int, seconds: float, gain: float) -> Path:
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
            "-af", f"volume={gain}", "-c:a", "pcm_s16le", str(path))
    return path


def _peak(path: Path, start_s: float, end_s: float) -> float:
    with wave.open(str(path), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        samples = array.array("h", w.readframes(w.getnframes()))
    window = samples[int(start_s * rate) * channels:int(end_s * rate) * channels]
    return max(abs(s) for s in window) / 32768.0


@pytest.fixture
def media(tmp_path):
    """3 s of picture with a 1 kHz tone, a processed version of that audio at
    twice the level, and a 2 s music bed; `peaks` holds each tone's peak."""
    tone = _tone(tmp_path / "main_tone.wav", 1000, 3, 1.0)
    main = tmp_path / "main.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3", "-i", str(tone),
            "-c:v", "mpeg4", "-q:v", "5", "-c:a", "pcm_s16le", "-shortest", str(main))
    override = _tone(tmp_path / "1_processed.wav", 440, 3, 2.0)
    bed = _tone(tmp_path / "bed.wav", 660, 2, 1.0)
    return {
        "main": main, "override": override, "bed": bed, "dir": tmp_path,
        "peaks": {"main": _peak(tone, 0, 3), "override": _peak(override, 0, 3),
                  "bed": _peak(bed, 0, 2)},
    }


def _rack_project(media) -> Timeline:
    """A1 plays programme 0-3 s; the bed plays 3-5 s, after the voice. The
    picture runs 0-5 s: with a rack, the sound is cut at the picture's end."""
    sources = {
        "main": SourceFile(id="main", path=str(media["main"]), duration_seconds=3.0,
                           width=320, height=180),
        "bed": SourceFile(id="bed", path=str(media["bed"]), duration_seconds=2.0,
                          width=0, height=0, kind="audio"),
    }
    items = []
    for k, (a, b) in enumerate([(0, 45), (45, 90)]):
        for track in ("V1", "A1"):
            items.append(TimelineItem(id=f"{track.lower()}_{k}", track=track, source_id="main",
                                      source_start_frame=a, source_end_frame=b,
                                      timeline_start_frame=a, timeline_end_frame=b))
    items.append(TimelineItem(id="v1_2", track="V1", source_id="main", source_start_frame=0,
                              source_end_frame=60, timeline_start_frame=90,
                              timeline_end_frame=150))
    items.append(TimelineItem(id="bed_0", track="A2", source_id="bed", source_start_frame=0,
                              source_end_frame=60, timeline_start_frame=90,
                              timeline_end_frame=150))
    return Timeline(fps_num=FPS, fps_den=1, width=320, height=180, sources=sources,
                    items=items, duration_frames=150)


def _render_audio(tl: Timeline, out: Path) -> Path:
    inputs, graph, v, a = FilterGraphCompiler(tl).compile()
    _ffmpeg(*inputs, "-filter_complex", graph, "-map", v, "-f", "null", "-",
            "-map", a, "-c:a", "pcm_s16le", str(out))
    return out


def test_racks_and_override_are_heard_in_the_render(media, no_audio_compile):
    tl = _rack_project(media)
    tl.sources["main"].audio_override = str(media["override"])
    tl.audio_racks = {"A1": [_amp("fx_a1", 9.5424)],          # x3
                      "MASTER": [_amp("fx_master", -6.0206)]}  # x0.5
    _item(tl, "bed_0").audio_effects = [_amp("fx_bed", -12.0412)]  # x0.25
    out = _render_audio(tl, media["dir"] / "racks.wav")
    peaks = media["peaks"]
    assert peaks["override"] == pytest.approx(2 * peaks["main"], rel=0.02)
    # Voice: the override, not the source's own tone, x3 by A1, x0.5 by MASTER.
    assert _peak(out, 0.2, 2.8) == pytest.approx(peaks["override"] * 1.5, rel=0.05)
    # Bed: x0.25 by its clip rack, x0.5 by the MASTER rack; the A1 rack not at all.
    assert _peak(out, 3.2, 4.8) == pytest.approx(peaks["bed"] * 0.125, rel=0.08)


def test_a_project_without_racks_renders_as_before(media):
    out = _render_audio(_rack_project(media), media["dir"] / "plain.wav")
    assert _peak(out, 0.2, 2.8) == pytest.approx(media["peaks"]["main"], rel=0.05)
    assert _peak(out, 3.2, 4.8) == pytest.approx(media["peaks"]["bed"], rel=0.05)


def test_rack_graph_runs_with_the_available_rack_compiler(media):
    """Whatever `rack_filters` resolves to (real module or fallback), the
    compiled graph is one ffmpeg accepts."""
    tl = _rack_project(media)
    tl.audio_racks = {"A1": [_amp("fx_a1")], "MASTER": [_amp("fx_master", -3.0)]}
    _item(tl, "bed_0").audio_effects = [_amp("fx_bed", -3.0)]
    out = _render_audio(tl, media["dir"] / "any.wav")
    assert out.exists() and out.stat().st_size > 1000


# --- a rack cannot change the programme's length -------------------------------

@pytest.mark.parametrize("scope", ["A1", "clip", "MASTER"])
def test_a_rack_ends_the_audio_at_the_picture_length(scope):
    """With any rack, the statement that makes the final audio ends padded and
    cut to the picture (old_project(2): 3 s); without racks there is no bound."""
    assert "apad=" not in FilterGraphCompiler(old_project(2)).compile()[1]
    tl = old_project(2)
    if scope == "clip":
        _item(tl, "bed_0").audio_effects = [_amp("fx_r")]
    else:
        tl.audio_racks = {scope: [_amp("fx_r")]}
    _inputs, graph, _v, a = FilterGraphCompiler(tl).compile()
    assert a == "[master_a]"
    assert _producing(graph, a).endswith(",apad=whole_dur=3.000000,atrim=end=3.000000[master_a]")
    assert graph.count("apad=") == 1


def test_fft_filter_racks_do_not_lengthen_the_render(media):
    """firequalizer flushes 2 x its delay (100 ms) of tail at the end of the
    stream; with FFT Filters in all three racks the sound still ends with the
    5 s picture, to the sample."""
    tl = _rack_project(media)
    tl.audio_racks = {"A1": [AudioEffect(id="fx_fa", effect="fft_filter")],
                      "MASTER": [AudioEffect(id="fx_fm", effect="fft_filter")]}
    _item(tl, "bed_0").audio_effects = [AudioEffect(id="fx_fb", effect="fft_filter")]
    graph = FilterGraphCompiler(tl).compile()[1]
    assert all(f"firequalizer@{i}_0" in graph for i in ("fx_fa", "fx_fb", "fx_fm"))
    out = _render_audio(tl, media["dir"] / "fft.wav")
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 48000
        assert w.getnframes() == 5 * 48000
