"""Tests for who wins when the planner's layers disagree.

`refine_disfluencies` runs structure (filler vocabulary, repeated runs), then the
fluency model, then a second opinion, then verification. Each layer can overrule
the one before it, and the order they do it in is the whole behaviour of the
auto-edit — but every existing test ran with `use_llm=False`, so none of it was
covered. These pin the precedence down.

The `ask` handed to the pipeline is a stand-in: the point is what the pipeline
does with an answer, not what a model would say.
"""

import pytest

import asr.fumble_engine as fumble_engine
from asr.fumble_engine import refine_disfluencies


def _words(sentence: str, spacing: float = 0.35):
    words = []
    clock = 0.0
    for token in sentence.split():
        words.append({"word": token, "start": round(clock, 3),
                      "end": round(clock + spacing * 0.85, 3), "probability": 0.99})
        clock += spacing
    return words


class _Model:
    """Stands in for `lm_studio_client`, answering each pass in turn.

    The pipeline calls the model for four different jobs through one method, so
    the stand-in routes on the system prompt rather than on call order.
    """

    def __init__(self, cleaned=None, take=None, audit="1: OK", final=None):
        self.cleaned = cleaned
        self.take = take
        self.audit = audit
        self.final = final
        self.seen = []

    async def is_responsive(self, *args, **kwargs):
        # The engine probes the model before the fluency passes; a stand-in is
        # always "responsive" so tests exercise the LLM path.
        return True

    async def clean_transcript(self, system, user):
        self.seen.append(system)
        if system.startswith("A speaker recording to camera"):
            return self.take
        if system.startswith("You are proof-reading"):
            return self.audit
        if system.startswith("You are repairing"):
            return None
        if system.startswith("You are checking a video edit"):
            return None
        if system.startswith("You are doing the last read"):
            return self.final
        return self.cleaned


@pytest.fixture
def model(monkeypatch):
    """Install a stand-in model and return it, so a test can script the answers."""
    stand_in = _Model()
    import llm.client as llm_client
    monkeypatch.setattr(llm_client, "lm_studio_client", stand_in)
    return stand_in


def _cut(words):
    return [w["word"] for w in words if w.get("disfluency")]


def _kept(words):
    return [w["word"] for w in words if not w.get("disfluency")]


# --- the fluency model overrules structure ---------------------------------

@pytest.mark.asyncio
async def test_a_word_the_model_deletes_is_cut(model):
    text = "dosto aapako pataa hai ki india men ek aisee jagah hai jo bahut amazing hai"
    model.cleaned = text.replace(" bahut", "")

    words = await refine_disfluencies(_words(text), use_llm=True)

    assert "bahut" in _cut(words)
    assert next(w for w in words if w["word"] == "bahut")["reason"] == "not_fluent"


# The pile-up structure cuts on its own, verbatim from the reference recording.
# `apply_retakes` removes the first four attempts and keeps the fifth.
PILE_UP = ("dosto kya ap dosto kya dosto kya a dosto "
           "dosto kya apako pataa hai india men ek aisee jagah hai")


@pytest.mark.asyncio
async def test_the_model_can_put_back_words_structure_cut_as_a_retake(model):
    """Structure asks "does this run reappear?" and has no notion of meaning.
    Where the model has *engaged* — its answer is a real edit, not an echo — and
    kept the words, it wins; that is the whole reason the fluency pass exists."""
    text = PILE_UP + " matlab bahut hi amazing hai"
    # A real edit signal: one word deleted, every attempt deliberately kept.
    model.cleaned = text.replace(" matlab", "")

    result = await refine_disfluencies(_words(text), use_llm=True)

    # The retake cuts are given back, and the reason goes with them — a word in
    # the edit must not still be labelled with why it was removed.
    assert "retake" not in {w.get("reason") for w in result if w.get("disfluency")}
    assert "retake" not in {w.get("reason") for w in result if not w.get("disfluency")}
    # The stutter is not given back: a doubled word is non-lexical debris and
    # never the model's call.
    assert "stutter" in {w.get("reason") for w in result if w.get("disfluency")}


@pytest.mark.asyncio
async def test_an_echoing_model_cannot_put_back_structural_cuts(model):
    """A model that parrots the transcript back scores a perfect match and used
    to be read as "keep everything", un-cutting every retake structure found.
    An echo is no opinion: the structural cuts must stand."""
    model.cleaned = PILE_UP     # verbatim echo — no edit signal at all

    result = await refine_disfluencies(_words(PILE_UP), use_llm=True)

    assert "retake" in {w.get("reason") for w in result if w.get("disfluency")}


@pytest.mark.asyncio
async def test_a_filler_sound_is_never_the_models_call(model):
    """"[uh]" is not a word the model should have to reason about. It was found
    in the audio and it is always cut, whatever comes back."""
    words = _words("dosto aapako pataa hai ki india men ek aisee jagah hai bahut")
    words[3].update({"word": "[uh]", "disfluency": True, "reason": "filler_sound",
                     "detected": True})
    model.cleaned = " ".join(w["word"] for w in words)   # the model keeps it

    result = await refine_disfluencies(words, use_llm=True, detect_fillers=False)

    assert "[uh]" in _cut(result)


@pytest.mark.asyncio
async def test_a_window_the_model_fumbled_leaves_structure_alone(model):
    """An untrusted answer must be absent from the decision, not read as "keep
    everything" — reporting keeps once resurrected 81 structural cuts."""
    model.cleaned = "this is a completely different English sentence entirely"

    result = await refine_disfluencies(_words(PILE_UP), use_llm=True)

    assert _kept(result) == ["dosto", "kya", "apako", "pataa", "hai", "india",
                             "men", "ek", "aisee", "jagah", "hai"], \
        "an untrusted answer must not undo the structural cuts"


# --- structure keeps the last word on long verbatim repeats ----------------

@pytest.mark.asyncio
async def test_a_long_phrase_said_twice_loses_its_first_copy(model):
    """The model reads a doubled phrase as rhetoric and keeps both. In an
    unscripted monologue a four-word run coming back within a breath is the
    speaker repeating themselves, and the viewer hears a fumble."""
    text = ("sab aapako jaj kar rahe hain sab aapako jaj kar rahe hain "
            "lekin aisaa hotaa naheen hai")
    model.cleaned = text        # the model keeps both copies

    words = await refine_disfluencies(_words(text), use_llm=True)

    cut = _cut(words)
    assert cut[:5] == ["sab", "aapako", "jaj", "kar", "rahe"]
    assert _kept(words)[-4:] == ["aisaa", "hotaa", "naheen", "hai"]


# --- the last read ---------------------------------------------------------

@pytest.mark.asyncio
async def test_the_final_read_may_remove_a_short_leftover_stumble(model):
    text = ("dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men gae hoon "
            "jahaan par bahut saare log baithe hue the aur sab aapako dekh rahe the")
    model.cleaned = text
    model.final = text.replace(" kisee", "")

    words = await refine_disfluencies(_words(text), use_llm=True)

    assert "kisee" in _cut(words)


@pytest.mark.asyncio
async def test_the_final_read_may_not_rewrite_the_edit(model):
    """Left unbounded, a pass re-reading an already-clean edit starts improving
    the writing: one such pass cut the greeting off the top of the video. Only
    short runs and nearby repetitions are admissible."""
    text = ("dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men gae hoon "
            "jahaan par bahut saare log baithe hue the aur sab aapako dekh rahe the")
    model.cleaned = text
    model.final = " ".join(text.split()[8:])        # drops the opening eight words

    words = await refine_disfluencies(_words(text), use_llm=True)

    assert _kept(words)[:2] == ["dosto", "aapake"]


# --- the pipeline without a model -----------------------------------------

@pytest.mark.asyncio
async def test_without_a_model_structure_decides_alone(model):
    """Every LLM pass returning nothing must still produce a usable edit, not an
    empty one — the deterministic path is the fail-safe."""
    text = "dosto dosto aapako pataa hai ki india men uh ek aisee jagah hai"
    model.cleaned = None
    model.audit = None

    words = await refine_disfluencies(_words(text), use_llm=True)

    assert "uh" in _cut(words), "hard fillers go with or without a model"
    assert "india" in _kept(words)
