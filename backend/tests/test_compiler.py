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
