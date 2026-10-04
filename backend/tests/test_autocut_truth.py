"""The answer-key matcher: recover which source spans a processed, reordered export plays.

Synthetic, so it runs anywhere: a "recording" of speech-like noise bursts and pauses, and an
"export" that keeps three of its stretches (one moved earlier, as a cut back to an earlier
take would be), filtered and turned up the way a mastering pass would, after a silent intro.
"""

import numpy as np

from tools import autocut_truth as at

SR = at.SR


def _recording(seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    t = 0.0
    while t < seconds - 1.0:
        burst = rng.uniform(0.4, 1.6)                       # a "word"
        start, end = int(t * SR), int(min(seconds, t + burst) * SR)
        envelope = np.hanning(end - start).astype(np.float32)
        audio[start:end] = rng.normal(0, 0.3, end - start).astype(np.float32) * envelope
        t += burst + rng.uniform(0.15, 0.6)                 # the pause after it
    audio += rng.normal(0, 0.002, audio.size).astype(np.float32)  # room tone
    return audio


def _master(audio: np.ndarray) -> np.ndarray:
    """A crude EQ (two-tap low-pass) plus +6 dB: the export is not the source's samples."""
    out = audio.copy()
    out[1:] = 0.7 * audio[1:] + 0.3 * audio[:-1]
    return (out * 2.0).astype(np.float32)


def test_matcher_recovers_kept_spans_through_filtering_and_reordering():
    src = _recording(60.0)
    kept = [(5.0, 15.0), (30.0, 40.0), (20.0, 25.0)]       # the third moves back in time
    intro = np.zeros(int(2.0 * SR), dtype=np.float32)       # silent title card
    cut = np.concatenate([intro] + [src[int(a * SR):int(b * SR)] for a, b in kept])
    result = at.align_edit(src, _master(cut), device=at._device("cpu"), log=lambda *_: None)

    matched = [s for s in result["segments"] if not s.get("unmatched") and s["speech_s"] > 0.5]
    assert len(matched) == 3
    for seg, (a, b) in zip(matched, kept):
        # Edges may sit anywhere inside the pause around a cut; the speech must line up.
        assert abs(seg["src_start"] - a) < 0.7 and abs(seg["src_end"] - b) < 0.7
        assert seg["match"] > 0.5 and seg["margin"] > 0.3


def test_removed_spans_are_the_complement_of_what_was_kept():
    spans = at.removed_spans([(5.0, 15.0), (30.0, 40.0), (20.0, 25.0)], 60.0)
    assert spans == [(0.0, 5.0), (15.0, 20.0), (25.0, 30.0), (40.0, 60.0)]
