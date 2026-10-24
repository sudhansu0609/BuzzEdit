"""asr/faster_whisper_engine.py: batched Whisper decoding is a setting the
A/B check turns on; the decoding options are the same either way."""
import types

from asr import faster_whisper_engine as engine


class _Seg:
    def __init__(self, start, end, words=None):
        self.start, self.end, self.words = start, end, words or []
        self.text = " ".join(w.word for w in self.words)
        self.avg_logprob, self.compression_ratio, self.no_speech_prob = -0.2, 1.1, 0.05


def _calls(monkeypatch, batched):
    seen = {}

    class Model:
        def transcribe(self, audio, **kw):
            seen["sequential"] = kw
            return iter([_Seg(0.0, 1.0)]), types.SimpleNamespace(language="hi")

    class Batched:
        def __init__(self, model):
            seen["wrapped"] = model

        def transcribe(self, audio, **kw):
            seen["batched"] = kw
            return iter([_Seg(0.0, 1.0)]), types.SimpleNamespace(language="hi")

    import faster_whisper
    import faster_whisper.audio
    import numpy as np
    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", Batched)
    # The batched path decodes the take itself to hand the pipeline one
    # whole-take clip in samples (faster-whisper 1.2 needs clips without VAD).
    monkeypatch.setattr(faster_whisper.audio, "decode_audio", lambda path, sampling_rate=16000: np.zeros(16000 * 7, dtype="float32"))
    monkeypatch.setattr(engine, "_batched_enabled", lambda: batched)
    monkeypatch.setattr(engine, "_retry_broken_segments", lambda model, path, segs, lang, task, prompt: segs)
    monkeypatch.setattr(engine, "_forced_alignment_enabled", lambda: False)
    monkeypatch.setattr(engine, "_gap_recovery_enabled", lambda: False)
    e = engine.FasterWhisperEngine()
    e._do_transcribe(Model(), "take.wav", "hi")
    return seen


def test_batched_decoding_is_off_until_the_setting_turns_it_on(monkeypatch):
    seen = _calls(monkeypatch, batched=False)
    assert "sequential" in seen and "batched" not in seen
    assert seen["sequential"]["temperature"] == 0.0 and seen["sequential"]["vad_filter"] is False


def test_batched_decoding_keeps_the_same_options(monkeypatch):
    seen = _calls(monkeypatch, batched=True)
    assert "batched" in seen and "sequential" not in seen
    kw = seen["batched"]
    assert kw["batch_size"] == engine.ASR_BATCH_SIZE
    assert kw["temperature"] == 0.0 and kw["vad_filter"] is False and kw["word_timestamps"] is True
    assert kw["condition_on_previous_text"] is False and kw["language"] == "hi"
    assert kw["clip_timestamps"] == [{"start": 0.0, "end": 7.0}], "30 s windows in seconds over the whole take"
