from backend.timeline import (
    Timeline,
    build_timeline_from_transcript,
    toggle_word,
    add_broll_item,
    time_to_frame,
    frame_to_time
)

def test_time_frame_conversion():
    assert time_to_frame(1.0, 30, 1) == 30
    assert time_to_frame(1.5, 30, 1) == 45
    assert abs(frame_to_time(45, 30, 1) - 1.5) < 1e-5

def test_build_and_rebuild_timeline():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 1.5, "disfluency": True},
        {"word": "welcome", "start": 1.5, "end": 2.0},
    ]

    tl = build_timeline_from_transcript(
        source_path="video.mp4",
        duration_seconds=5.0,
        transcript_words=words,
        fps_num=30,
        fps_den=1
    )

    # Initial disfluency "um" should be disabled
    assert len(tl.words) == 4
    assert tl.words[2].enabled is False

    # V1 and A1 should have 2 continuous segments (Hello world & welcome), omitting "um"
    v1_items = [item for item in tl.items if item.track == "V1"]
    assert len(v1_items) == 2

    # First segment: 0 to 30 frames (0.0s to 1.0s) -> timeline 0 to 30
    assert v1_items[0].source_start_frame == 0
    assert v1_items[0].source_end_frame == 30
    assert v1_items[0].timeline_start_frame == 0
    assert v1_items[0].timeline_end_frame == 30

    # Second segment: 45 to 60 frames (1.5s to 2.0s) -> timeline 30 to 45 (rippled!)
    assert v1_items[1].source_start_frame == 45
    assert v1_items[1].source_end_frame == 60
    assert v1_items[1].timeline_start_frame == 30
    assert v1_items[1].timeline_end_frame == 45

    assert tl.duration_frames == 45

def test_toggle_word_updates_tracks():
    words = [
        {"word": "One", "start": 0.0, "end": 1.0},
        {"word": "Two", "start": 1.0, "end": 2.0},
        {"word": "Three", "start": 2.0, "end": 3.0},
    ]
    tl = build_timeline_from_transcript("video.mp4", 5.0, words)
    primary_src = list(tl.sources.keys())[0]

    v1_items = [item for item in tl.items if item.track == "V1"]
    assert len(v1_items) == 1
    assert tl.duration_frames == 90

    # Toggle middle word off
    word_id = tl.words[1].id
    toggle_word(tl, word_id, False, primary_src)

    v1_items = [item for item in tl.items if item.track == "V1"]
    assert len(v1_items) == 2
    assert tl.duration_frames == 60
