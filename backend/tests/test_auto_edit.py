"""Tests for the audio-driven auto-edit.

The old pipeline looked for filler *words* in the transcript. Measured on a real
195-second recording in this repo it cut 4 words out of 351, because Whisper
emits clean text and had already deleted every "um" before we saw it — and
because 76 seconds of dead air was sealed inside over-long word timestamps where
no gap-based rule could reach it.

These tests pin the two things that fix: silence found in the audio rather than
in the text, and word timings reconciled against real speech.
"""

import asyncio
import subprocess
import wave

import numpy as np
import pytest

from backend.asr import align, vad
from backend.asr.fumble_engine import last_report, refine_disfluencies


def _write_wave(path, segments, sample_rate=16000, duration=None):
    """Render a tone/silence pattern to a wav file.

    `segments` is a list of (start, end) speech spans in seconds; everything
    outside them is near-silent room tone rather than digital zero, so the tests
    exercise the same noise-floor estimation real recordings do.
    """
    total = duration if duration is not None else max(e for _s, e in segments) + 0.5
    t = np.arange(int(total * sample_rate)) / sample_rate
    rng = np.random.default_rng(7)
    audio = rng.normal(0.0, 0.0015, t.shape)          # room tone
    for start, end in segments:
        mask = (t >= start) & (t < end)
        # Voice-like: a low fundamental plus a harmonic, amplitude-modulated.
        audio[mask] += 0.35 * (np.sin(2 * np.pi * 140 * t[mask])
                               + 0.5 * np.sin(2 * np.pi * 280 * t[mask]))
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return str(path)


@pytest.fixture
def two_phrases(tmp_path):
    """Speech 0.5-2.0s and 5.0-6.5s, with a three-second hole between."""
    return _write_wave(tmp_path / "two_phrases.wav", [(0.5, 2.0), (5.0, 6.5)], duration=7.0)


@pytest.fixture
def with_filler(tmp_path):
    """Two phrases with a short "uh" between them at 3.0-3.35s.

    The blip is deliberately brief: a longer unclaimed region is speech the ASR
    missed, and the planner must leave that alone.
    """
    return _write_wave(tmp_path / "with_filler.wav",
                       [(0.5, 2.0), (3.0, 3.35), (5.0, 6.5)], duration=7.0)


# --- VAD ------------------------------------------------------------------

def test_speech_regions_are_found(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    assert speech_map is not None
    assert len(speech_map.speech) == 2
    first, second = speech_map.speech
    assert first[0] == pytest.approx(0.5, abs=0.1)
    assert first[1] == pytest.approx(2.0, abs=0.1)
    assert second[0] == pytest.approx(5.0, abs=0.1)


def test_threshold_adapts_to_a_quiet_recording(tmp_path):
    """A fixed -35 dB threshold called a quiet recording pure silence."""
    quiet = _write_wave(tmp_path / "quiet.wav", [(0.5, 2.0)], duration=3.0)
    with wave.open(quiet, "rb") as fh:
        frames = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
        rate = fh.getframerate()
    with wave.open(str(tmp_path / "quieter.wav"), "wb") as fh:
        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(rate)
        fh.writeframes((frames // 12).astype(np.int16).tobytes())   # ~22 dB down

    speech_map = vad.analyse_speech(str(tmp_path / "quieter.wav"))
    assert speech_map is not None
    assert speech_map.speech, "quiet speech was read as silence"
    assert speech_map.speech[0][0] == pytest.approx(0.5, abs=0.15)


def test_silences_include_the_head_and_tail(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    silences = speech_map.silences(min_duration=0.3)
    assert any(s < 0.1 for s, _e in silences), "lead-in silence missing"
    assert any(e > 6.4 for _s, e in silences), "trailing silence missing"


def test_speech_fraction_and_clamping(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    assert speech_map.speech_fraction(0.5, 2.0) > 0.9
    assert speech_map.speech_fraction(2.5, 4.5) < 0.1
    clamped = speech_map.clamp_to_speech(0.0, 3.0)
    assert clamped is not None and clamped[0] == pytest.approx(0.5, abs=0.1)
    assert speech_map.clamp_to_speech(2.5, 4.5) is None


def test_a_file_with_no_dynamic_range_is_left_alone(tmp_path):
    """Pure tone or pure silence must not be shredded into cuts."""
    flat = _write_wave(tmp_path / "flat.wav", [(0.0, 3.0)], duration=3.0)
    speech_map = vad.analyse_speech(flat)
    assert speech_map is not None
    assert len(speech_map.speech) == 1


# --- word timing repair ---------------------------------------------------

def test_a_word_that_swallowed_a_pause_is_clamped(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    words = [
        {"word": "hello", "start": 0.5, "end": 2.0},
        {"word": "stretched", "start": 2.0, "end": 6.5},   # swallows the 3s hole
    ]
    repaired, report = align.repair_word_timings(words, speech_map)

    assert report.repaired == 1
    assert repaired[0]["end"] == pytest.approx(2.0, abs=0.05)     # untouched
    assert repaired[1]["start"] == pytest.approx(5.0, abs=0.15)   # snapped to speech
    assert report.seconds_recovered > 2.5


def test_well_aligned_words_are_not_touched(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    words = [{"word": "hello", "start": 0.6, "end": 1.9}]
    repaired, report = align.repair_word_timings(words, speech_map)
    assert report.repaired == 0
    assert repaired[0]["start"] == 0.6 and repaired[0]["end"] == 1.9


def test_a_word_over_pure_silence_is_kept_but_shrunk(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    words = [{"word": "ghost", "start": 3.0, "end": 4.5}]
    repaired, report = align.repair_word_timings(words, speech_map)
    assert report.dropped == 1
    assert len(repaired) == 1, "a word must never silently vanish from the transcript"
    assert repaired[0]["end"] - repaired[0]["start"] < 0.2


def test_repair_keeps_words_in_order(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    words = [
        {"word": "a", "start": 0.5, "end": 4.0},
        {"word": "b", "start": 1.0, "end": 6.5},
    ]
    repaired, _ = align.repair_word_timings(words, speech_map)
    assert repaired[1]["start"] >= repaired[0]["end"]


def test_repair_without_audio_is_a_no_op():
    words = [{"word": "hello", "start": 0.0, "end": 9.0}]
    repaired, report = align.repair_word_timings(words, None)
    assert repaired == words and report.repaired == 0


# --- fillers the transcript never had -------------------------------------

def test_unvoiced_speech_is_found(two_phrases):
    """The "um" Whisper deleted is still in the audio, unclaimed by any word."""
    speech_map = vad.analyse_speech(two_phrases)
    words = [{"word": "hello", "start": 0.5, "end": 2.0}]      # nothing for 5.0-6.5
    unvoiced = align.find_unvoiced_speech(words, speech_map, max_duration=2.5)
    assert unvoiced
    assert unvoiced[0][0] == pytest.approx(5.0, abs=0.15)


def test_long_unclaimed_speech_is_left_alone(with_filler):
    """Speech the ASR missed is content, not a filler — cutting it loses the user's words."""
    speech_map = vad.analyse_speech(with_filler)
    found = align.find_unvoiced_speech([], speech_map, max_duration=1.2)
    # Only the short blip qualifies; the two 1.5s phrases are content.
    assert len(found) == 1
    assert found[0][0] == pytest.approx(3.0, abs=0.15)


def test_fully_transcribed_audio_yields_no_fillers(two_phrases):
    speech_map = vad.analyse_speech(two_phrases)
    words = [
        {"word": "hello", "start": 0.5, "end": 2.0},
        {"word": "again", "start": 5.0, "end": 6.5},
    ]
    assert align.find_unvoiced_speech(words, speech_map) == []


# --- the planner end to end -----------------------------------------------

def _plan(words, audio_path, **kwargs):
    return asyncio.run(refine_disfluencies(words, use_llm=False, audio_path=audio_path, **kwargs))


def test_planner_cuts_the_filler_the_transcript_never_had(with_filler):
    words = [
        {"word": "hello", "start": 0.5, "end": 2.0},
        {"word": "again", "start": 5.0, "end": 6.5},
    ]
    planned = _plan(words, with_filler)
    report = last_report(planned)

    assert report["used_audio"] is True
    assert report["fillers_found"] >= 1
    fillers = [w for w in planned if w.get("reason") == "filler_sound"]
    assert fillers and all(w["enabled"] is False for w in fillers)
    # The real word survives.
    assert any(w["word"] == "hello" and w["enabled"] for w in planned)


def test_planner_reports_what_it_recovered(two_phrases):
    words = [
        {"word": "hello", "start": 0.5, "end": 2.0},
        {"word": "stretched", "start": 2.0, "end": 6.5},
    ]
    report = last_report(_plan(words, two_phrases))
    assert report["timing"]["repaired"] >= 1
    assert report["seconds_recovered"] > 1.0
    assert 0.0 < report["speech_fraction"] < 1.0
    assert report["speech"], "speech map must be published for the timeline rebuild"


def test_planner_without_audio_falls_back_to_text(two_phrases):
    words = [
        {"word": "um", "start": 0.0, "end": 0.3},
        {"word": "hello", "start": 0.5, "end": 2.0},
    ]
    planned = _plan(words, None)
    report = last_report(planned)
    assert report["used_audio"] is False
    # The text layer still catches an explicit hard filler.
    assert planned[0]["enabled"] is False


def test_planner_leaves_speech_alone(two_phrases):
    """Nothing but the gap should be cut when every region is transcribed."""
    words = [
        {"word": "hello", "start": 0.5, "end": 2.0},
        {"word": "again", "start": 5.0, "end": 6.5},
    ]
    planned = _plan(words, two_phrases)
    assert all(w["enabled"] for w in planned if not w.get("detected"))
    assert last_report(planned)["fillers_found"] == 0


def test_detected_fillers_can_be_overruled_by_the_user(with_filler):
    """Fillers are words in the transcript, so re-enabling one puts it back."""
    planned = _plan([{"word": "hello", "start": 0.5, "end": 2.0},
                     {"word": "again", "start": 5.0, "end": 6.5}], with_filler)
    filler = next(w for w in planned if w.get("detected"))
    assert filler["word"] == "[uh]"
    assert "start" in filler and "end" in filler


async def test_a_cut_creates_no_new_stutter():
    """"kamare men hol men gae" with "hol" cut leaves "men men" back to back —
    a repeat manufactured by the edit itself. The final sweep collapses it."""
    from backend.asr.fumble_engine import refine_disfluencies
    words = []
    clock = 0.0
    for token in "kisee kamare men uh men gae ho".split():
        words.append({"word": token, "start": round(clock, 3),
                      "end": round(clock + 0.3, 3), "probability": 0.9})
        clock += 0.35
    out = await refine_disfluencies(words, use_llm=False, audio_path=None,
                                    detect_fillers=False)
    kept = " ".join(w["word"] for w in out if w.get("enabled"))
    assert kept == "kisee kamare men gae ho"
