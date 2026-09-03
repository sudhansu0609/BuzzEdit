"""Tests for checking the render against the plan, and for the last-read guard.

The word list is the plan; the V1 items are what will actually be played. Those
two disagreed for a long time and nothing noticed, because the transcript panel
reads the plan: the rebuild bridged any gap under `max_pause_seconds`, and a
removed filler is such a gap, so short cuts were glued back into the render while
the UI showed them struck out. `audit_cut_coverage` is what makes that visible.
"""

import pytest

from asr.fluency import admissible_repair_cuts
from timeline import build_timeline_from_transcript
from timeline.ops import audit_cut_coverage


def _timeline(words, **kwargs):
    return build_timeline_from_transcript(
        source_path="video.mp4",
        duration_seconds=10.0,
        transcript_words=words,
        fps_num=30,
        fps_den=1,
        pause_padding_seconds=0.0,
        **kwargs,
    )


# --- the render against the plan -------------------------------------------

def test_a_clean_edit_reports_nothing_still_audible():
    timeline = _timeline([
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "um", "start": 0.5, "end": 1.4, "disfluency": True},
        {"word": "world", "start": 1.4, "end": 2.0},
    ])

    coverage = audit_cut_coverage(timeline)

    assert coverage["checked"] == 1
    assert coverage["still_audible"] == 0


def test_a_cut_word_the_rebuild_bridged_back_in_is_reported():
    """A ~2-frame removal is deliberately not made — an inaudible pop is not
    worth a visible jump — so the material stays, the segments join, and the
    report has to carry the fact that the struck-out word still plays."""
    timeline = _timeline([
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 0.56, "disfluency": True},
        {"word": "world", "start": 0.56, "end": 1.2},
    ])

    coverage = audit_cut_coverage(timeline)

    assert coverage["still_audible"] == 1
    assert coverage["examples"] == ["uh"]


def test_a_filler_at_the_admission_floor_is_actually_cut():
    """The admission floor and the removal floor are one constant now: any
    filler long enough for the detector to mark (>= 0.18s) must be long enough
    for the rebuild to remove. Under the old 0.25s removal floor this exact
    word was marked cut and bridged straight back into the render."""
    timeline = _timeline([
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "uh", "start": 0.5, "end": 0.70, "disfluency": True},
        {"word": "world", "start": 0.70, "end": 1.2},
    ])

    v1 = sorted((i.source_start_frame, i.source_end_frame)
                for i in timeline.items if i.track == "V1")
    assert len(v1) == 2                                   # a real cut, not a bridge
    midpoint = int(round(0.60 * 30))                      # middle of "uh"
    assert not any(a <= midpoint < b for a, b in v1)
    assert audit_cut_coverage(timeline)["still_audible"] == 0


def test_a_dropped_sliver_disables_its_words():
    """A kept word isolated between two removals and shorter than
    `min_segment_seconds` is dropped from the render as a sliver. The plan must
    record that: the word flips to disabled/`sliver` so the transcript panel
    and the render agree, and the reverse audit stays clean."""
    timeline = _timeline([
        {"word": "hello", "start": 0.0, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 1.5, "disfluency": True},
        {"word": "oops", "start": 1.5, "end": 1.7},
        {"word": "uh", "start": 1.7, "end": 2.2, "disfluency": True},
        {"word": "world", "start": 2.2, "end": 3.0},
    ])

    sliver = next(w for w in timeline.words if w.text == "oops")
    assert sliver.enabled is False
    assert sliver.disfluency is True
    assert sliver.reason == "sliver"

    coverage = audit_cut_coverage(timeline)
    assert coverage["still_audible"] == 0
    assert coverage["kept_but_dropped"] == 0


def test_a_word_with_no_speech_under_it_is_not_counted():
    """A disabled word floating in silence carries no audio to leak, so counting
    it would report a problem that cannot be heard."""
    timeline = _timeline(
        [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "uh", "start": 2.0, "end": 2.4, "disfluency": True},
            {"word": "world", "start": 4.0, "end": 4.5},
        ],
        speech_regions=[(0.0, 0.5), (4.0, 4.5)],
    )

    assert audit_cut_coverage(timeline)["checked"] == 0


def test_an_edit_with_no_cuts_checks_nothing():
    timeline = _timeline([{"word": "hello", "start": 0.0, "end": 0.5}])
    coverage = audit_cut_coverage(timeline)
    assert coverage["checked"] == 0
    assert coverage["still_audible"] == 0
    assert coverage["examples"] == []
    assert coverage["kept_but_dropped"] == 0


# --- what a re-read of a finished edit is allowed to remove ----------------

def test_a_short_run_is_repair():
    tokens = "dosto aapake saath kabhee aisaa ha aap vah hai ki aap kamare".split()
    assert admissible_repair_cuts(tokens, {5, 6, 7}) == {5, 6, 7}


def test_a_long_run_that_repeats_nothing_is_refused():
    """This is the guard that saved the greeting: a re-read with nothing left to
    do starts improving the writing, and it cut the opening off the video."""
    tokens = "dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men".split()
    assert admissible_repair_cuts(tokens, set(range(0, 8))) == set()


def test_a_long_run_of_words_that_recur_nearby_is_the_leftover_of_a_repeat():
    tokens = ("sab aapako jaj kar rahe hain sab aapako jaj kar rahe hain").split()
    assert admissible_repair_cuts(tokens, set(range(0, 6))) == set(range(0, 6))


def test_nothing_proposed_removes_nothing():
    assert admissible_repair_cuts(["a", "b", "c"], set()) == set()


@pytest.mark.asyncio
async def test_the_final_read_never_cuts_the_sign_off():
    """The end of the video is the sign-off and nothing follows to supersede it
    — this pass once deleted the speaker's "बाय बाय" as debris."""
    from asr.fluency import final_read
    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3}
             for i in range(24)]
    words[-1]["word"] = "babai"

    async def ask(_system, user):
        text = user.split("Transcript:\n")[1].split("\n\nCleaned transcript:")[0]
        tokens = [t for t in text.split() if t != "|"]
        return " ".join(tokens[:-1])

    assert await final_read(words, ask) == set()


@pytest.mark.asyncio
async def test_a_word_doubled_at_the_tail_is_still_cut():
    """The tail guard protects a unique sign-off, not the remains of a doubled
    phrase — a stutter on the very last word still goes."""
    from asr.fluency import final_read
    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3}
             for i in range(22)]
    words.extend([{"word": "bye", "start": 8.8, "end": 9.1},
                  {"word": "bye", "start": 9.2, "end": 9.5}])

    async def ask(_system, user):
        text = user.split("Transcript:\n")[1].split("\n\nCleaned transcript:")[0]
        tokens = [t for t in text.split() if t != "|"]
        del tokens[22]
        return " ".join(tokens)

    # Either copy may be the one the diff lands on; both are the same sound.
    cut = await final_read(words, ask)
    assert len(cut) == 1 and cut <= {22, 23}


def test_a_short_run_carrying_a_number_is_not_debris():
    """"50" is data and "%" is the spoken word "percent" wearing punctuation —
    a live pass cut both out of "more than 50% 60%". The short-run shape may
    not touch them; only the repeat shape can, when the number is doubled."""
    assert admissible_repair_cuts(["more", "den", "50", "log"], {1, 2}) == set()
    assert admissible_repair_cuts(["more", "", "log"], {1}) == set()
    assert admissible_repair_cuts(["50", "log", "50", "log"], {0, 1}) == {0, 1}
