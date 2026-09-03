"""Tests for the cut verifier's deterministic half and the alignment fallback.

The re-transcription half needs Whisper (and, for alignment, torch), so it is not
exercised here; these cover the logic that decides leaked/dropped/coverage and
the guarantee that forced alignment never breaks the pipeline when absent.
"""

from asr.cut_verify import (
    _collapse_repeats, _cut_spans, cut_layout, cut_time_to_source_frames,
    diff_transcripts, find_leaked_regions, repair_leaks, struck_audio_leak,
)
from asr.forced_align import align_words, align_available, map_stamps_to_words
from timeline.ops import _subtract_regions, trim_source_regions
from timeline.schema import SourceFile, Timeline, TimelineItem, WordItem


def _timeline(words, spans, fps_num=30, fps_den=1):
    """A timeline with the given (text, start_f, end_f, enabled) words and the
    given (src_start_f, src_end_f) V1/A1 kept spans."""
    src = SourceFile(id="s", path="x.mp4", duration_seconds=100.0,
                     fps_num=fps_num, fps_den=fps_den)
    wi = [WordItem(id=f"w{i}", text=t, start_frame=a, end_frame=b, enabled=en)
          for i, (t, a, b, en) in enumerate(words)]
    items = []
    for i, (a, b) in enumerate(spans):
        for track in ("V1", "A1"):
            items.append(TimelineItem(
                id=f"{track}_{i}", track=track, source_id="s",
                source_start_frame=a, source_end_frame=b,
                timeline_start_frame=0, timeline_end_frame=b - a, enabled=True))
    return Timeline(fps_num=fps_num, fps_den=fps_den, sources={"s": src},
                    words=wi, items=items)


def test_cut_spans_reads_kept_ranges_in_seconds():
    tl = _timeline([], [(30, 60), (90, 150)])
    assert _cut_spans(tl) == [(1.0, 2.0), (3.0, 5.0)]


def test_struck_audio_leak_flags_a_struck_word_inside_a_kept_span():
    # A struck word whose audio (frames 30-60) sits inside the kept span 0-90.
    tl = _timeline(
        [("keep", 0, 30, True), ("fumble", 30, 60, False), ("keep2", 60, 90, True)],
        [(0, 90)])
    leak = struck_audio_leak(tl)
    assert leak["leaked_struck_seconds"] == 1.0        # frames 30-60 at 30fps
    assert "fumble" in leak["leaked_struck_words"]
    assert leak["dropped_kept_count"] == 0


def test_struck_audio_leak_ignores_breath_padding():
    # A struck word barely clipped by a span (0.1s < 0.2s tolerance) is breath,
    # not a leak, and must not be flagged or trimmed.
    tl = _timeline(
        [("keep", 0, 30, True), ("uh", 30, 36, False)],   # 'uh' spans 30-36
        [(0, 33)])                                          # span clips 3 frames (0.1s)
    leak = struck_audio_leak(tl)
    assert leak["leaked_struck_words"] == []
    assert leak["leaked_regions"] == []


def test_repair_leaks_trims_a_leaked_word_and_reaches_clean():
    # A struck retake (frames 30-90, 2s) fully inside the kept span leaks; repair
    # trims it and a re-check finds nothing left.
    tl = _timeline(
        [("keep", 0, 30, True), ("retake", 30, 90, False), ("keep2", 90, 120, True)],
        [(0, 120)])
    assert struck_audio_leak(tl)["leaked_struck_words"] == ["retake"]
    result = repair_leaks(tl, "s")
    assert result["repaired_regions"] >= 1
    assert struck_audio_leak(tl)["leaked_struck_words"] == []
    # The kept audio is preserved: two segments (0-30 and 90-120) survive.
    v1 = sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.timeline_start_frame)
    assert [(i.source_start_frame, i.source_end_frame) for i in v1] == [(0, 30), (90, 120)]


def test_struck_audio_leak_flags_a_dropped_kept_word():
    # An enabled word (frames 200-230) that no kept span covers.
    tl = _timeline(
        [("keep", 0, 30, True), ("orphan", 200, 230, True)],
        [(0, 30)])
    leak = struck_audio_leak(tl)
    assert leak["dropped_kept_count"] == 1
    assert "orphan" in leak["dropped_kept_words"]
    assert leak["leaked_struck_seconds"] == 0.0


def test_collapse_repeats_folds_hallucinated_runs_but_keeps_doubles():
    # Three+ in a row -> one (Whisper loop); a double stays (real emphasis).
    assert _collapse_repeats(["a", "a", "a", "a", "b"]) == ["a", "b"]
    assert _collapse_repeats(["bahut", "bahut", "accha"]) == ["bahut", "bahut", "accha"]


def test_diff_transcripts_coverage_ignores_hallucinated_inserts():
    expected = ["dosto", "kya", "aap", "theek", "hain"]
    # The cut says the same, but Whisper looped "hain" at the seam.
    actual = ["dosto", "kya", "aap", "theek", "hain", "hain", "hain", "hain"]
    report = diff_transcripts(expected, actual)
    assert report["coverage"] == 1.0
    assert report["missing_count"] == 0


def test_diff_transcripts_reports_dropped_planned_words():
    expected = ["dosto", "kya", "aap", "theek", "hain"]
    actual = ["dosto", "aap", "hain"]           # "kya" and "theek" never said
    report = diff_transcripts(expected, actual)
    assert report["coverage"] < 1.0
    assert "kya" in report["missing_examples"]
    assert "theek" in report["missing_examples"]


def test_forced_alignment_absent_is_a_no_op():
    # These deps are not in the test env; alignment must degrade, never raise.
    if align_available():
        return
    words = [{"word": "dosto", "word_native": "दोस्तो", "start": 11.5, "end": 11.7}]
    out = align_words("nofile.wav", words, "hi")
    assert out[0]["start"] == 11.5 and "timing_aligned" not in out[0]


def test_forced_alignment_skips_unsupported_language():
    words = [{"word": "hello", "start": 1.0, "end": 1.2}]
    out = align_words("x.wav", words, "xx")
    assert out[0]["start"] == 1.0


def test_map_stamps_to_words_one_to_one():
    words = [{"word_native": "a"}, {"word_native": "b"}]
    stamps = [{"text": "a", "start": 0.0, "end": 0.5}, {"text": "b", "start": 0.5, "end": 1.0}]
    assert map_stamps_to_words(words, stamps) == {0: (0.0, 0.5), 1: (0.5, 1.0)}


def test_map_stamps_to_words_tolerates_a_dropped_token():
    # The aligner dropped the middle token; the outer two must still be re-timed.
    words = [{"word_native": "a"}, {"word_native": "b"}, {"word_native": "c"}]
    stamps = [{"text": "a", "start": 0.0, "end": 0.5}, {"text": "c", "start": 1.0, "end": 1.5}]
    mapping = map_stamps_to_words(words, stamps)
    assert mapping.get(0) == (0.0, 0.5)
    assert mapping.get(2) == (1.0, 1.5)
    assert 1 not in mapping                       # the unmatched word keeps its timing


def test_subtract_regions_splits_and_swallows():
    assert _subtract_regions([0, 100], [[40, 60]]) == [[0, 40], [60, 100]]
    assert _subtract_regions([0, 100], [[0, 100]]) == []
    assert _subtract_regions([0, 100], [[80, 200]]) == [[0, 80]]
    assert _subtract_regions([0, 100], [[200, 300]]) == [[0, 100]]


def test_cut_layout_and_time_mapping():
    # Two spans: source 1-2s and 3-5s -> cut 0-1s and 1-3s.
    layout = cut_layout([(1.0, 2.0), (3.0, 5.0)])
    assert layout == [(0.0, 1.0, 1.0, 2.0), (1.0, 3.0, 3.0, 5.0)]
    # A leak at cut 1.5-2.0s came from source 3.5-4.0s -> frames 105-120 at 30fps.
    assert cut_time_to_source_frames(layout, 30.0, 1.5, 2.0) == [[105, 120]]


def _repair_timeline():
    src = SourceFile(id="s", path="x.mp4", duration_seconds=100.0)
    items = []
    for i, (a, b) in enumerate([(0, 60), (90, 150)]):
        for track in ("V1", "A1"):
            items.append(TimelineItem(
                id=f"{track}{i}", track=track, source_id="s",
                source_start_frame=a, source_end_frame=b,
                timeline_start_frame=0, timeline_end_frame=b - a, enabled=True))
    return Timeline(sources={"s": src}, items=items)


def test_trim_source_regions_removes_a_leak_and_reripples():
    tl = _repair_timeline()
    # Leak: source frames 30-45 (inside the first span 0-60).
    applied = trim_source_regions(tl, [[30, 45]], "s")
    assert applied == 1
    v1 = sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.timeline_start_frame)
    # First span split into 0-30 and 45-60, second span 90-150 intact -> 3 items.
    assert [(i.source_start_frame, i.source_end_frame) for i in v1] == [(0, 30), (45, 60), (90, 150)]
    # Timeline is gap-free after the ripple.
    assert v1[0].timeline_start_frame == 0
    assert v1[1].timeline_start_frame == 30
    assert v1[2].timeline_start_frame == 45
    # A1 mirrors V1.
    assert len([i for i in tl.items if i.track == "A1"]) == 3


def test_find_leaked_regions_maps_extra_audio_to_source():
    src = SourceFile(id="s", path="x.mp4", duration_seconds=100.0)
    words = [WordItem(id="w0", text="dosto", start_frame=0, end_frame=15, enabled=True),
             WordItem(id="w1", text="hain", start_frame=15, end_frame=30, enabled=True)]
    items = [TimelineItem(id="V1", track="V1", source_id="s",
                          source_start_frame=0, source_end_frame=60,
                          timeline_start_frame=0, timeline_end_frame=60, enabled=True)]
    tl = Timeline(sources={"s": src}, words=words, items=items)
    # The cut re-transcribes as: dosto, <leaked fumble 0.5-1.2s>, hain.
    actual = [
        {"hinglish": "dosto", "start": 0.0, "end": 0.4},
        {"hinglish": "phir", "start": 0.5, "end": 0.9},
        {"hinglish": "baahar", "start": 0.9, "end": 1.3},
        {"hinglish": "hain", "start": 1.4, "end": 1.7},
    ]
    regions = find_leaked_regions(tl, actual)
    # The leaked run (0.5-1.3s) maps straight to source frames (30fps): 15-39.
    assert regions == [[15, 39]]
