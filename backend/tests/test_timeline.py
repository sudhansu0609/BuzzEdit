from backend.timeline import (
    Timeline,
    build_timeline_from_transcript,
    toggle_word,
    add_broll_item,
    time_to_frame,
    frame_to_time
)
from backend.timeline.ops import rebuild_primary_tracks

def test_time_frame_conversion():
    assert time_to_frame(1.0, 30, 1) == 30
    assert time_to_frame(1.5, 30, 1) == 45
    assert abs(frame_to_time(45, 30, 1) - 1.5) < 1e-5


def _four_words():
    return [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 1.5, "disfluency": True},
        {"word": "welcome", "start": 1.5, "end": 2.0},
    ]


def test_build_and_rebuild_timeline():
    """Cutting on exact word boundaries, with padding switched off."""
    tl = build_timeline_from_transcript(
        source_path="video.mp4",
        duration_seconds=5.0,
        transcript_words=_four_words(),
        fps_num=30,
        fps_den=1,
        pause_padding_seconds=0.0,
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


def test_segments_keep_a_breath_either_side_of_a_cut():
    """Padding is on by default: slicing exactly on the words clips their onsets."""
    tl = build_timeline_from_transcript("video.mp4", 5.0, _four_words(), fps_num=30, fps_den=1)
    v1 = [i for i in tl.items if i.track == "V1"]

    padding = int(round(tl.pause_padding_seconds * 30))
    assert padding > 0
    assert v1[0].source_start_frame == 0                    # clamped at the media start
    # The gap here is entirely the removed "um", and padding must not replay a
    # removed word's onset — so both edges sit exactly on the cut. Padding still
    # breathes into gaps that are silence (see the trailing edge below).
    assert v1[0].source_end_frame == 30
    assert v1[1].source_start_frame == 45
    assert v1[1].source_end_frame == 60 + padding           # room after the last word


def test_padding_never_reaches_into_the_next_segment():
    words = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.55, "end": 1.0, "disfluency": True},
        {"word": "three", "start": 1.05, "end": 1.5},
    ]
    tl = build_timeline_from_transcript("video.mp4", 5.0, words, fps_num=30, fps_den=1)
    v1 = sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.source_start_frame)
    for earlier, later in zip(v1, v1[1:]):
        assert earlier.source_end_frame <= later.source_start_frame


def test_natural_pauses_are_kept_but_long_ones_are_trimmed():
    """A short beat is delivery; a long one is dead air."""
    words = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.7, "end": 1.2},      # 0.2s pause — keep
        {"word": "three", "start": 4.0, "end": 4.5},    # 2.8s pause — trim
    ]
    tl = build_timeline_from_transcript("video.mp4", 8.0, words, fps_num=30, fps_den=1,
                                        pause_padding_seconds=0.0)
    v1 = sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.source_start_frame)
    assert len(v1) == 2
    assert v1[0].source_start_frame == 0 and v1[0].source_end_frame == 36   # spans the short pause
    assert v1[1].source_start_frame == 120
    # The long pause is gone from the programme: segments ripple together.
    assert v1[1].timeline_start_frame == v1[0].timeline_end_frame


def test_silence_inside_a_word_is_cut_out():
    """A word whose timestamp swallowed a pause must not drag it into the edit.

    This is the failure that made auto-edit useless on real footage: Whisper
    handed back words spanning several seconds, so the dead air inside them was
    invisible to any gap-based rule. 76 seconds of one 195-second recording was
    hidden this way.
    """
    words = [
        {"word": "hello", "start": 0.0, "end": 1.0},
        {"word": "stretched", "start": 1.0, "end": 5.0},   # only 1..1.5 is speech
        {"word": "world", "start": 5.0, "end": 5.5},
    ]
    tl = build_timeline_from_transcript(
        "video.mp4", 8.0, words, fps_num=30, fps_den=1, pause_padding_seconds=0.0,
        speech_regions=[(0.0, 1.5), (5.0, 5.5)],
    )
    v1 = sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.source_start_frame)
    kept = sum(i.source_end_frame - i.source_start_frame for i in v1)
    # 1.5s of speech at the head plus 0.5s at the tail = 2.0s, not the 5.5s span.
    assert kept == 60
    assert v1[0].source_end_frame == 45        # trimmed to where speech stops


def test_without_a_speech_map_words_are_taken_at_face_value():
    words = [{"word": "hello", "start": 0.0, "end": 2.0}]
    tl = build_timeline_from_transcript("video.mp4", 5.0, words, fps_num=30, fps_den=1,
                                        pause_padding_seconds=0.0)
    v1 = [i for i in tl.items if i.track == "V1"]
    assert v1[0].source_end_frame == 60


def test_toggle_word_updates_tracks():
    words = [
        {"word": "One", "start": 0.0, "end": 1.0},
        {"word": "Two", "start": 1.0, "end": 2.0},
        {"word": "Three", "start": 2.0, "end": 3.0},
    ]
    tl = build_timeline_from_transcript("video.mp4", 5.0, words, pause_padding_seconds=0.0)
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


# --- removed words must actually leave the render --------------------------

def _covered(tl, frame):
    return any(i.source_start_frame <= frame < i.source_end_frame
               for i in tl.items if i.track == "V1")


def test_a_short_cut_word_is_not_merged_back_into_the_render():
    """The bug behind "the fumbles are still in the video": a removed word
    shorter than the natural-pause threshold left a gap the segment merge then
    happily bridged — so the "cut" filler played anyway. Measured on the real
    project: 13 cut words, five of them "[uh]", still audible in the render."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 0.8, "disfluency": True},   # 0.3s < 0.4s pause
        {"word": "today", "start": 0.8, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    v1 = [i for i in tl.items if i.track == "V1"]
    assert len(v1) == 2, "the cut must split the segment, not be bridged"
    assert not _covered(tl, 19)          # middle of "uh" (0.65s) is gone
    assert _covered(tl, 10) and _covered(tl, 33)   # both real words play


def test_a_natural_short_pause_is_still_bridged():
    """The same short gap with nothing removed in it is a breath, and stays."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "today", "start": 0.8, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 1


def test_padding_does_not_replay_removed_material():
    """The 0.12s breath either side of a cut used to expand back into the span
    that was just removed, re-covering most of a short fumble."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 0.8, "disfluency": True},
        {"word": "today", "start": 0.8, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.12)
    assert not _covered(tl, 19), "padding re-covered the removed fumble"


def test_a_one_frame_removal_is_left_alone():
    """Cutting a single frame is an inaudible pop and a visible video jump —
    worse than the thing being removed."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "x", "start": 0.5, "end": 0.533, "disfluency": True},
        {"word": "today", "start": 0.533, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 1


def test_cuts_snap_to_the_quietest_nearby_frame():
    """With an energy envelope stored, a cut lands on the energy dip next to the
    word boundary rather than on the ASR's ±50ms guess."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 1.1, "disfluency": True},
        {"word": "today", "start": 1.1, "end": 1.7},
    ]
    # 50 envelope frames/sec; loud everywhere except a dip at 0.54-0.56s.
    db = [-20] * 100
    db[27] = -60
    tl = build_timeline_from_transcript(
        "v.mp4", 5.0, words, 30, 1, pause_padding_seconds=0.0,
        energy_envelope={"rate": 50.0, "db": db},
    )
    v1 = [i for i in tl.items if i.track == "V1"]
    # The end of "so" (frame 15 = 0.50s) moves to the dip at ~0.54s (frame 16).
    assert v1[0].source_end_frame == 16


def test_the_end_snap_never_reaches_into_the_word_when_padding_was_clamped():
    """With no padding to trade (clamped by the removal right after the word),
    the segment end IS the word's own end. The dip hunt used to be allowed two
    frames inside anyway and shaved 66ms off final consonants — with nothing
    gained, since the frames it moved to were speech."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 1.1, "disfluency": True},
        {"word": "today", "start": 1.1, "end": 1.7},
    ]
    db = [-20] * 100
    db[23] = -60          # a deep dip at 0.46s — inside "so"'s final consonant
    tl = build_timeline_from_transcript(
        "v.mp4", 5.0, words, 30, 1, pause_padding_seconds=0.0,
        energy_envelope={"rate": 50.0, "db": db},
    )
    v1 = [i for i in tl.items if i.track == "V1"]
    assert v1[0].source_end_frame == 15   # exactly the word's end, no shave


# --- pacing: a cut has to be worth the jump it costs ------------------------

def test_a_removal_too_short_to_hear_is_not_cut():
    """Removing 3 frames of speech is inaudible; the jump cut it would create is
    very audible and visible. Below the threshold the material stays."""
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "x", "start": 0.5, "end": 0.60, "disfluency": True},   # 0.1s
        {"word": "today", "start": 0.60, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 1


def test_a_short_manual_strike_is_always_cut():
    """A word the user struck by hand (no disfluency flag, no filler reason) is an
    intentional cut and must be honoured even below the auto-filler floor — the
    render has to match the struck-out transcript, not glue the word back in."""
    from backend.timeline.ops import toggle_word
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "actually", "start": 0.5, "end": 0.60},   # 0.1s, no disfluency
        {"word": "today", "start": 0.60, "end": 1.4},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 1   # nothing cut yet
    src = tl.items[0].source_id
    struck = tl.words[1].id
    toggle_word(tl, struck, enabled=False, primary_source_id=src)
    # The strike splits the programme even though the word is only 3 frames long.
    assert len([i for i in tl.items if i.track == "V1"]) == 2


def test_a_removal_worth_making_is_still_cut():
    words = [
        {"word": "so", "start": 0.0, "end": 0.5},
        {"word": "uhh", "start": 0.5, "end": 1.1, "disfluency": True},  # 0.6s
        {"word": "today", "start": 1.1, "end": 1.9},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 2


def test_a_sliver_between_two_cuts_is_dropped():
    """A 0.2s survivor bracketed by removals is a flash of a different head
    position on screen; dropping it turns two cuts into one."""
    words = [
        {"word": "one", "start": 0.0, "end": 1.0},
        {"word": "aa", "start": 1.0, "end": 1.8, "disfluency": True},
        {"word": "ka", "start": 1.8, "end": 2.0},                        # sliver
        {"word": "bb", "start": 2.0, "end": 2.8, "disfluency": True},
        {"word": "two", "start": 2.8, "end": 3.8},
    ]
    tl = build_timeline_from_transcript("v.mp4", 6.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    v1 = [i for i in tl.items if i.track == "V1"]
    assert len(v1) == 2, "the sliver should have merged the two cuts into one"
    assert not any(i.source_start_frame <= 57 < i.source_end_frame for i in v1)


def test_a_short_segment_at_the_edges_is_kept():
    """Only slivers *surrounded* by removals go — a short first or last segment
    is content, not a flash."""
    words = [
        {"word": "hi", "start": 0.0, "end": 0.2},
        {"word": "aa", "start": 0.2, "end": 1.0, "disfluency": True},
        {"word": "there", "start": 1.0, "end": 2.0},
    ]
    tl = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                        pause_padding_seconds=0.0)
    assert len([i for i in tl.items if i.track == "V1"]) == 2


def test_the_spoken_script_survives_the_timeline_round_trip():
    """`word_native` must reach the timeline and come back out again.

    Every grammar and fluency prompt states that its input is Devanagari, and
    verify.PROTECTED_WORDS lists the Hindi negations in that script. The native
    form was being dropped at each dict-to-WordItem conversion, so a re-cut of an
    existing timeline judged romanized Hinglish against those prompts — the
    negations could not match and good Hindi read as broken English.
    """
    tl = build_timeline_from_transcript(
        source_path="video.mp4",
        duration_seconds=5.0,
        transcript_words=[
            {"word": "dosto", "word_native": "दोस्तों", "start": 0.0, "end": 0.5},
            {"word": "naheen", "word_native": "नहीं", "start": 0.5, "end": 1.0},
            {"word": "amazing", "start": 1.0, "end": 1.5},
        ],
        fps_num=30, fps_den=1,
    )
    assert [w.text for w in tl.words] == ["dosto", "naheen", "amazing"]
    assert [w.word_native for w in tl.words] == ["दोस्तों", "नहीं", None]


def test_the_planner_judges_the_native_script_not_its_romanization():
    """The token every text pass reads is the spoken script when there is one."""
    from backend.asr.fluency import _token as fluency_token
    from backend.asr.verify import _token as verify_token

    hindi = {"word": "naheen", "word_native": "नहीं"}
    english = {"word": "amazing"}
    for token_of in (fluency_token, verify_token):
        assert token_of(hindi) == "नहीं"
        assert token_of(english) == "amazing"
