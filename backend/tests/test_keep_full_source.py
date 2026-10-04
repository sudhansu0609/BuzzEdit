"""A precut recording (Timeline.keep_full_source) is dressed, never re-cut."""

from timeline import build_timeline_from_transcript
from timeline.ops import rebuild_primary_tracks


def _words():
    # Long pauses and a flagged fumble: exactly what the normal rebuild trims.
    return [
        {"word": "pehla", "start": 1.0, "end": 1.4},
        {"word": "umm", "start": 1.5, "end": 1.9, "disfluency": True},
        {"word": "step", "start": 4.0, "end": 4.4},
        {"word": "mushkil", "start": 8.0, "end": 8.6},
    ]


def _build(keep):
    return build_timeline_from_transcript(
        source_path="take.mp4", duration_seconds=10.0, transcript_words=_words(),
        fps_num=30, fps_den=1, keep_full_source=keep)


def test_precut_keeps_every_frame_as_one_clip():
    tl = _build(True)
    v1 = [i for i in tl.items if i.track == "V1"]
    a1 = [i for i in tl.items if i.track == "A1"]
    assert len(v1) == len(a1) == 1
    assert (v1[0].source_start_frame, v1[0].source_end_frame) == (0, 300)
    assert all(w.enabled for w in tl.words)


def test_precut_survives_a_rebuild_after_a_word_is_struck():
    tl = _build(True)
    tl.words[2].enabled = False
    rebuild_primary_tracks(tl, next(iter(tl.sources)))
    v1 = [i for i in tl.items if i.track == "V1"]
    assert len(v1) == 1 and v1[0].source_end_frame == 300


def test_normal_build_still_cuts():
    tl = _build(False)
    v1 = [i for i in tl.items if i.track == "V1"]
    covered = sum(i.source_end_frame - i.source_start_frame for i in v1)
    assert covered < 300
