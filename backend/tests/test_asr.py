from backend.asr.disfluency import analyze_disfluencies
from backend.asr.fumble_engine import refine_disfluencies


def _w(word, i, prob=1.0):
    return {"word": word, "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": prob}


def test_hard_filler_and_stutter():
    words = [_w("Hello", 0), _w("um", 1), _w("we", 2), _w("we", 3), _w("are", 4), _w("here", 5)]
    a = analyze_disfluencies(words)
    assert a[0]["disfluency"] is False          # Hello
    assert a[1]["disfluency"] is True           # um (hard filler)
    assert a[2]["disfluency"] is True           # first "we" -> cut the earlier stutter
    assert a[3]["disfluency"] is False          # keep the clean final "we"
    assert a[4]["disfluency"] is False
    assert a[5]["disfluency"] is False


def test_soft_filler_is_candidate_not_cut():
    # "actually" and "matlab" (Hinglish) must NOT be auto-cut deterministically.
    words = [_w("this", 0), _w("actually", 1), _w("matlab", 2), _w("works", 3)]
    a = analyze_disfluencies(words)
    assert a[1]["disfluency"] is False and a[1]["candidate"] is True
    assert a[2]["disfluency"] is False and a[2]["candidate"] is True
    assert a[2]["reason"] == "soft_filler"


def test_low_confidence_microtoken_is_candidate():
    words = [_w("go", 0), _w("aa", 1, prob=0.1), _w("there", 2)]
    a = analyze_disfluencies(words)
    # "aa" is also a hard filler here, so it is cut; use a non-lexical short token instead
    words2 = [_w("go", 0), _w("zz", 1, prob=0.1), _w("there", 2)]
    a2 = analyze_disfluencies(words2)
    assert a2[1]["candidate"] is True and a2[1]["reason"] == "low_confidence"
    assert a[1]["disfluency"] is True  # "aa" hard filler


def test_phrase_repeat_flags_earlier_run():
    words = [_w("we", 0), _w("went", 1), _w("we", 2), _w("went", 3), _w("home", 4)]
    a = analyze_disfluencies(words)
    # earlier "we went" run flagged as false_start candidate
    assert a[0]["candidate"] is True or a[0]["disfluency"] is True
    assert a[1]["candidate"] is True or a[1]["disfluency"] is True


async def test_transcription_runs_off_the_event_loop(monkeypatch):
    """The model load and decode are blocking C work; if they run inline they
    freeze the server (and the auto-edit job's own /status polls) so the UI sits
    frozen 'not doing anything'. transcribe_words_async must run them in a thread,
    which a concurrent coroutine advancing during a slow decode proves."""
    import asyncio
    import time
    from backend.asr.faster_whisper_engine import whisper_engine

    def blocking_decode(audio_path, language, task, vocabulary=None, glossary=None):
        time.sleep(0.3)   # stand-in for a real decode
        return ([{"word": "hi", "word_native": "hi", "hinglish": "hi",
                  "start": 0.0, "end": 0.5, "probability": 1.0}], "en")

    monkeypatch.setattr(whisper_engine, "_transcribe_with_fallback", blocking_decode)

    ticks = 0
    async def ticker():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    words, lang = await whisper_engine.transcribe_words_async("x.wav")
    await task

    assert lang == "en" and words
    # A blocked loop would have frozen the ticker for the whole 0.3s decode.
    assert ticks >= 5


async def test_refine_failsafe_keeps_soft_fillers_when_conservative():
    words = [_w("this", 0), _w("actually", 1), _w("works", 2)]
    refined = await refine_disfluencies(words, aggressiveness=0.3, use_llm=False)
    assert refined[1]["disfluency"] is False   # conservative: keep ambiguous word
    assert refined[1]["enabled"] is True


async def test_refine_aggressive_cuts_soft_fillers_without_llm():
    words = [_w("this", 0), _w("actually", 1), _w("works", 2)]
    refined = await refine_disfluencies(words, aggressiveness=0.7, use_llm=False)
    assert refined[1]["disfluency"] is True    # aggressive: cut crutch word
    assert refined[1]["enabled"] is False


async def test_refine_always_cuts_hard_fillers():
    words = [_w("hello", 0), _w("um", 1), _w("world", 2)]
    refined = await refine_disfluencies(words, aggressiveness=0.0, use_llm=False)
    assert refined[1]["disfluency"] is True    # hard filler cut even at min aggressiveness


# --- language choice (pooled detection) -------------------------------------

def test_language_choice_prefers_indic_over_a_narrow_english_win():
    """An `en` pass over Hindi paraphrases instead of transcribing — the whole
    transcript becomes translation tokens with made-up timings. A `hi` pass over
    English still transcribes it. So a torn detection goes to the Indic side."""
    from backend.asr.faster_whisper_engine import choose_language
    assert choose_language({"en": 0.48, "hi": 0.41}) == "hi"
    assert choose_language({"en": 0.55, "hi": 0.30, "ur": 0.05}) == "hi"


def test_language_choice_keeps_a_clear_english_or_other_win():
    from backend.asr.faster_whisper_engine import choose_language
    assert choose_language({"en": 0.9, "hi": 0.05}) == "en"
    assert choose_language({"de": 0.7, "en": 0.2}) == "de"
    assert choose_language({"hi": 0.8, "en": 0.15}) == "hi"
    assert choose_language({}) == "en"


def test_hindi_reduplication_is_not_a_stutter():
    """"अपने-अपने काम" doubles the word on purpose — cutting one copy turned
    "वो सब अपने-अपने काम पर हैं" into "वो सब काम हो". The ASR's own hyphen and
    a vocabulary of words Hindi routinely doubles both protect the pair."""
    from asr.disfluency import analyze_disfluencies
    words = [
        {"word": "sab", "word_native": "सब", "start": 0.0, "end": 0.3, "probability": 0.99},
        {"word": "apne", "word_native": "अपने", "start": 0.35, "end": 0.6, "probability": 0.99},
        {"word": "-apne", "word_native": "-अपने", "start": 0.62, "end": 0.9, "probability": 0.99},
        {"word": "kaam", "word_native": "काम", "start": 0.95, "end": 1.2, "probability": 0.99},
        {"word": "par", "word_native": "पर", "start": 1.25, "end": 1.5, "probability": 0.99},
    ]
    out = analyze_disfluencies(words)
    assert not any(w.get("disfluency") for w in out)


def test_a_plain_doubled_word_is_still_a_stutter():
    from asr.disfluency import analyze_disfluencies
    words = [
        {"word": "dosto", "word_native": "दोस्तो", "start": 0.0, "end": 0.3, "probability": 0.99},
        {"word": "dosto", "word_native": "दोस्तो", "start": 0.35, "end": 0.6, "probability": 0.99},
        {"word": "kya", "word_native": "क्या", "start": 0.65, "end": 0.9, "probability": 0.99},
    ]
    out = analyze_disfluencies(words)
    assert out[0].get("disfluency") and out[0].get("reason") == "stutter"
    assert not out[1].get("disfluency")


def test_the_sweep_leaves_reduplication_alone_too():
    from asr.fumble_engine import _sweep_stutters
    words = [
        {"word": "dheere", "word_native": "धीरे", "start": 0.0, "end": 0.3},
        {"word": "dheere", "word_native": "धीरे", "start": 0.35, "end": 0.6},
        {"word": "chalo", "word_native": "चलो", "start": 0.65, "end": 0.9},
    ]
    _sweep_stutters(words)
    assert not any(w.get("disfluency") for w in words)
