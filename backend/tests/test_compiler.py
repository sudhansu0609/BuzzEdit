import pytest
from backend.timeline import build_timeline_from_transcript
from backend.render.compiler import FilterGraphCompiler

def test_filter_graph_compiler_single_segment():
    words = [
        {"word": "Hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
    ]
    tl = build_timeline_from_transcript("C:/video.mp4", 5.0, words)
    compiler = FilterGraphCompiler(tl)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    assert inputs == ["-i", "C:/video.mp4"]
    assert "[0:v]trim=start=0.000:end=2.000,setpts=PTS-STARTPTS[v1_0]" in filter_complex
    assert "[0:a]atrim=start=0.000:end=2.000,asetpts=PTS-STARTPTS[a1_0]" in filter_complex
    assert final_v == "[v1_0]"
    assert final_a == "[a1_0]"

def test_filter_graph_compiler_multiple_cuts():
    words = [
        {"word": "First", "start": 0.0, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 2.0, "disfluency": True},
        {"word": "Second", "start": 2.0, "end": 3.0},
    ]
    tl = build_timeline_from_transcript("C:/video.mp4", 5.0, words)
    compiler = FilterGraphCompiler(tl)
    inputs, filter_complex, final_v, final_a = compiler.compile()

    # Should create 2 cuts (0s-1s and 2s-3s) and concat them
    assert "[v1_0]" in filter_complex
    assert "[v1_1]" in filter_complex
    assert "concat=n=2:v=1:a=0[base_v]" in filter_complex
    assert "concat=n=2:v=0:a=1[base_a]" in filter_complex
    assert final_v == "[base_v]"
    assert final_a == "[base_a]"
