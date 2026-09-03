"""Tests for "will this model actually serve?" — the check that was missing.

`lms load` reports success for a model larger than the GPU: llama.cpp offloads
part of it and carries on. The failure arrives much later, on a real prompt, as
`{"error":"terminated"}` — and by then the fluency windows are gone and the edit
that ships is the structural fallback.

Measured on the reference machine: the configured `qwen/qwen3.8-27b` is 17.7GB of
weights on a 16.3GB card, so every model-driven layer had been silently doing
nothing. A model too big for the card is not a preference to honour.
"""

from backend.llm.lm_launcher import _fits
from backend.asr.auto_edit import public_report


GB = 1024.0


def test_a_model_larger_than_the_card_does_not_fit():
    """The reference case: 17.7GB of weights, 16.3GB of VRAM."""
    assert not _fits(17742039110 / (1024.0 * 1024.0), 16311.0)


def test_a_model_with_room_for_its_context_fits():
    """7.56GB of weights on the same card, with the KV cache and compute buffers
    still comfortably inside it."""
    assert _fits(7556572366 / (1024.0 * 1024.0), 16311.0)


def test_weights_that_only_just_fit_are_refused():
    """Fitting the weights is not enough — the KV cache and compute buffers are
    allocated on top, and a model with no room for them dies mid-request."""
    assert not _fits(15.5 * GB, 16.0 * GB)


def test_an_unmeasurable_model_is_allowed_through():
    """Refusing what cannot be measured would be worse than trying it: a failed
    load is already handled, and an unknown size is usually a model LM Studio
    listed in a shape this code did not expect."""
    assert _fits(None, 4.0 * GB)
    assert _fits(0, 4.0 * GB)


def test_a_structural_only_edit_says_so_in_the_report():
    """`used_llm: False` in a log is how this hid for weeks. The report has to
    say it in a sentence the user reads."""
    warnings = public_report({"used_fluency": False}).get("warnings") or []
    assert any("language model never ran" in w for w in warnings)


def test_a_normal_edit_carries_no_such_warning():
    report = public_report({
        "used_fluency": True,
        "fluency_windows": {"answered": 3, "failed": 0, "spans": 3,
                            "rewrites": 0, "none_answers": 0},
        "quality": {"cut_words_still_audible": 0, "verdict": "clean"},
    })
    assert "warnings" not in report


def test_a_model_that_would_not_name_cuts_is_reported():
    """The edit was made by the older, guessier contract; the user should know
    which one produced their cut."""
    warnings = public_report({
        "used_fluency": True,
        "fluency_windows": {"answered": 3, "failed": 0, "spans": 0, "rewrites": 3},
    }).get("warnings") or []
    assert any("would not name the cuts" in w for w in warnings)
