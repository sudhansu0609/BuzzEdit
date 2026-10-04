"""Gap recovery: speech the main pass left without words is decoded again, on its own."""

from types import SimpleNamespace

import numpy as np

from asr import gap_recovery as gr


def _w(start, end, word="x"):
    return {"word": word, "start": start, "end": end}


def _run(start, end, word="x"):
    """Consecutive 0.5 s words from start to end — what real speech looks like."""
    out, t = [], start
    while t < end - 1e-9:
        out.append(_w(round(t, 3), round(min(end, t + 0.5), 3), word))
        t += 0.5
    return out


def test_uncovered_speech_is_found_and_short_slivers_are_not():
    regions = [(0.0, 10.0)]
    words = _run(0.0, 2.0) + _run(5.0, 6.0) + _run(6.2, 9.8)
    # 2.0-5.0 has no word; 6.0-6.2 and 9.8-10.0 are word edges, too short to be a phrase.
    assert gr.uncovered(regions, words) == [(2.0, 5.0)]


def test_a_stretched_word_only_covers_its_first_moments():
    # Whisper (and an aligner fitting around a hole) stretches a word over speech it
    # skipped: a 9 s "word" covers only its first 1.5 s.
    gaps = gr.uncovered([(0.0, 10.0)], [_w(0.0, 9.0)])
    assert gaps == [(gr.STRETCH_CAP_SECONDS, 10.0)]


def test_clips_carry_context_and_merge():
    clips = gr._clips([(2.0, 3.0), (3.3, 4.0), (8.0, 9.0)], duration=8.9)
    assert clips == [(1.75, 4.25), (7.75, 8.9)]


class _FakeModel:
    """Answers `transcribe(clip_timestamps=...)` with fixed words, recording the call."""

    def __init__(self, words):
        self.words = words
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        seg = SimpleNamespace(words=[SimpleNamespace(word=t, start=s, end=e, probability=p)
                                     for t, s, e, p in self.words])
        return iter([seg]), SimpleNamespace(language="hi")


def test_recovered_words_fill_the_hole_and_nothing_else(monkeypatch):
    monkeypatch.setattr(gr, "speech_regions", lambda audio, sample_rate=16000: [(0.0, 10.0)])
    words = _run(0.0, 2.0, "pehle") + _run(5.0, 10.0, "baad")
    model = _FakeModel([
        ("dosto", 2.2, 2.6, 0.9),      # inside the 2.0-5.0 hole: recovered
        ("kya", 2.7, 3.0, 0.9),        # inside: recovered
        ("hmm", 3.2, 3.4, 0.1),        # inside but too unsure: noise, dropped
        ("pehle", 0.5, 1.0, 0.9),      # outside every hole: already transcribed
        ("baad", 5.1, 5.4, 0.9),       # outside: already transcribed
    ])
    merged, report = gr.recover_gaps(model, "a.wav", words, "hi",
                                     make_word=lambda w: {"word": w.word, "start": w.start, "end": w.end},
                                     audio=np.zeros(16000 * 10, dtype=np.float32))
    recovered = [w for w in merged if w.get("recovered")]
    assert [w["word"] for w in recovered] == ["dosto", "kya"]
    assert len(merged) == len(words) + 2
    assert [w["start"] for w in merged] == sorted(w["start"] for w in merged)
    assert report["gaps"] == 1 and report["recovered_words"] == 2
    # Only the hole (plus context) was decoded, with the main pass's settings.
    call = model.calls[0]
    assert call["clip_timestamps"] == [1.75, 5.25]
    assert call["vad_filter"] is False and call["condition_on_previous_text"] is False


def test_recovery_never_breaks_the_main_transcript(monkeypatch):
    def boom(audio, sample_rate=16000):
        raise RuntimeError("no VAD model")

    monkeypatch.setattr(gr, "speech_regions", boom)
    words = [_w(0.0, 1.0)]
    merged, report = gr.recover_gaps(_FakeModel([]), "a.wav", words, "hi", make_word=dict,
                                     audio=np.zeros(16000, dtype=np.float32))
    assert merged is words and "error" in report
