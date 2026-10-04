import pytest
from backend.timeline import build_timeline_from_transcript
from backend.render.compiler import FilterGraphCompiler

def test_filter_graph_compiler_single_segment():
    words = [
        {"word": "Hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
    ]
    # Padding off so the assertions can pin exact trim times.
    tl = build_timeline_from_transcript("C:/video.mp4", 5.0, words, pause_padding_seconds=0.0)
    compiler = FilterGraphCompiler(tl)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    assert inputs == ["-i", "C:/video.mp4"]
    # Each V1 segment is normalised (fps/sar/format) after the trim so mixed
    # sources concat cleanly; the assertion pins the trim and the label only.
    assert "[0:v]trim=start=0.000:end=2.000,setpts=PTS-STARTPTS" in filter_complex
    assert "[v1_0]" in filter_complex
    # Every A1 segment carries 8ms declick fades on its edges.
    assert ("[0:a]atrim=start=0.000:end=2.000,asetpts=PTS-STARTPTS,"
            "afade=t=in:st=0:d=0.008:curve=tri,"
            "afade=t=out:st=1.992:d=0.008:curve=tri[a1_0]") in filter_complex
    assert final_v == "[v1_0]"
    assert final_a == "[a1_0]"

def test_filter_graph_compiler_multiple_cuts():
    words = [
        {"word": "First", "start": 0.0, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 2.0, "disfluency": True},
        {"word": "Second", "start": 2.0, "end": 3.0},
    ]
    # Padding off so the assertions can pin exact trim times.
    tl = build_timeline_from_transcript("C:/video.mp4", 5.0, words, pause_padding_seconds=0.0)
    compiler = FilterGraphCompiler(tl)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    # Should create 2 cuts (0s-1s and 2s-3s) and concat them. With no transition
    # set, both cuts belong to one hard-cut run and are joined in a single concat.
    assert "[v1_0]" in filter_complex
    assert "[v1_1]" in filter_complex
    assert "[v1_0][v1_1]concat=n=2:v=1:a=0" in filter_complex
    assert "[a1_0][a1_1]concat=n=2:v=0:a=1" in filter_complex
    assert final_v.startswith("[") and final_v.endswith("]")
    assert final_a.startswith("[") and final_a.endswith("]")
    # The final labels are produced by the join, not by the per-clip trims.
    assert final_v not in ("[v1_0]", "[v1_1]")
    assert final_a not in ("[a1_0]", "[a1_1]")


def test_every_programme_audio_join_is_declicked():
    """A hard concat at an arbitrary sample is an audible tick at every cut;
    8ms edge fades on each A1 segment remove it without sounding like a fade."""
    from backend.timeline.schema import Timeline, SourceFile, TimelineItem
    from backend.render.compiler import FilterGraphCompiler

    src = SourceFile(id="s", path="C:/m.mp4", duration_seconds=20, width=640, height=360)
    items = []
    for n in range(2):
        for track in ("V1", "A1"):
            items.append(TimelineItem(
                id=f"{track.lower()}_{n}", track=track, source_id="s",
                source_start_frame=n * 90, source_end_frame=n * 90 + 60,
                timeline_start_frame=n * 60, timeline_end_frame=n * 60 + 60))
    tl = Timeline(fps_num=30, fps_den=1, sources={"s": src}, items=items)
    tl.recalculate_duration()

    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    assert fc.count("afade=t=in:st=0:d=0.008") == 2
    assert fc.count("afade=t=out:st=1.992:d=0.008") == 2


def test_overlay_cut_ahead_of_its_slot_gets_a_seeked_input():
    """The cold open: a hook from deep inside the programme shown at t=0. Cut
    from the shared decoded stream, the overlay could not produce a frame until
    the decoder reached the hook, and everything V1 made meanwhile piled up in
    RAM (gigabytes within a minute) until the memory guardian killed FFmpeg. So
    a video overlay whose source time runs ahead of its slot decodes from its
    own `-ss` input. So does one shown LATER than its source time (a replay,
    every B-roll clip): on the shared stream ffmpeg read it first -- lowest
    timestamp -- and held every frame until its slot (a real render peaked at
    14.7 GB). Each seeked input is stamped with its slot via `-itsoffset`."""
    from backend.timeline.schema import Timeline, SourceFile, TimelineItem
    from backend.render.compiler import FilterGraphCompiler

    src = SourceFile(id="s", path="C:/m.mp4", duration_seconds=120, width=1920, height=1080)
    items = [
        TimelineItem(id="v1", track="V1", source_id="s",
                     source_start_frame=0, source_end_frame=3000,
                     timeline_start_frame=0, timeline_end_frame=3000),
        TimelineItem(id="a1", track="A1", source_id="s",
                     source_start_frame=0, source_end_frame=3000,
                     timeline_start_frame=0, timeline_end_frame=3000),
        # Cold open: source 39.12s-42.16s shown at 0s -- 39s ahead of its slot.
        TimelineItem(id="hook", track="V2", source_id="s",
                     source_start_frame=978, source_end_frame=1054,
                     timeline_start_frame=0, timeline_end_frame=76),
        # Replay of an earlier moment: source 10s shown at 60s -- behind, fine.
        TimelineItem(id="replay", track="V2", source_id="s",
                     source_start_frame=250, source_end_frame=300,
                     timeline_start_frame=1500, timeline_end_frame=1550),
    ]
    tl = Timeline(fps_num=25, fps_den=1, sources={"s": src}, items=items)
    tl.recalculate_duration()

    inputs, fc, _, _ = FilterGraphCompiler(tl).compile()

    assert inputs[:2] == ["-i", "C:/m.mp4"]
    assert inputs[2:12] == ["-itsoffset", "0.000", "-ss", "39.120", "-t", "3.140",
                            "-threads", "1", "-i", "C:/m.mp4"]
    assert inputs[12:] == ["-itsoffset", "60.000", "-ss", "10.000", "-t", "2.100",
                           "-threads", "1", "-i", "C:/m.mp4"]
    assert "[1:v]trim=duration=3.040,setpts=PTS-STARTPTS" in fc
    assert "[2:v]trim=duration=2.000,setpts=PTS-STARTPTS" in fc
    assert "trim=start=39.120" not in fc and "trim=start=10.000" not in fc


def test_adjustment_copy_is_padded_so_the_base_never_waits_on_it():
    """An adjustment's treated copy is trimmed to its window. Unpadded, overlay
    could not pass a single base frame until that copy's first frame existed,
    so the whole programme before the window sat in RAM. The copy is padded
    with black from t=0 (never composited: the overlay is disabled there)."""
    from backend.timeline.schema import Timeline, SourceFile, TimelineItem, Transform
    from backend.render.compiler import FilterGraphCompiler

    src = SourceFile(id="s", path="C:/m.mp4", duration_seconds=200, width=1920, height=1080)
    items = [
        TimelineItem(id="v1", track="V1", source_id="s",
                     source_start_frame=0, source_end_frame=5000,
                     timeline_start_frame=0, timeline_end_frame=5000),
        TimelineItem(id="a1", track="A1", source_id="s",
                     source_start_frame=0, source_end_frame=5000,
                     timeline_start_frame=0, timeline_end_frame=5000),
        TimelineItem(id="punch", kind="adjustment", track="V2",
                     timeline_start_frame=2723, timeline_end_frame=2746,
                     transform=Transform(scale=1.1)),
    ]
    tl = Timeline(fps_num=25, fps_den=1, sources={"s": src}, items=items)
    tl.recalculate_duration()

    _, fc, _, _ = FilterGraphCompiler(tl).compile()

    assert "[adjfx_0]setpts=PTS-STARTPTS,tpad=start_mode=add:start_duration=108.920:color=black[adjpad_0]" in fc
    assert "[adjbase_0][adjpad_0]overlay=x=0:y=0:eof_action=pass:enable='between(t,108.920,109.840)'[adj_0]" in fc
