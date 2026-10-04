"""Which Whisper segments are re-decoded (asr/faster_whisper_engine.py).

Normal Hinglish in Devanagari must never count as "broken": Whisper's own
compression-ratio test flagged 26 of 36 fine segments of a real take, and
re-decoding them is what made a 13:50 recording take 20 minutes."""

from types import SimpleNamespace

from asr.faster_whisper_engine import _is_broken, _retry_broken_segments


def _seg(text, start=0.0, end=5.0, ratio=3.1):
    return SimpleNamespace(text=text, start=start, end=end, compression_ratio=ratio,
                           avg_logprob=-1.3, no_speech_prob=0.1, words=[])


def test_normal_devanagari_is_not_broken_even_with_a_high_compression_ratio():
    assert not _is_broken(_seg("तो सुन हम सबके पास बड़े सपने होते हैं हम सब सोचते हैं कि अपने एक ब्रांड बनाएंगे"))


def test_a_hallucination_loop_is_broken():
    assert _is_broken(_seg(" ".join(["धन्यवाद"] * 14)))


def test_only_broken_segments_are_retried():
    calls = []

    class Model:
        def transcribe(self, *a, **k):
            calls.append(k["clip_timestamps"])
            return iter([_seg("फिर से साफ़ शब्द", 10.0, 15.0)]), None

    good, bad = _seg("एक दो तीन चार पांच छह सात आठ नौ दस ग्यारह बारह"), _seg(" ".join(["हाँ"] * 20), 10.0, 15.0)
    out = _retry_broken_segments(Model(), "a.wav", [good, bad], "hi", "transcribe", None)
    assert calls == [[10.0, 15.0]]
    assert out[0] is good and out[1].text == "फिर से साफ़ शब्द"
