"""Claim/stat/quote beats becoming newspaper cutaways
(presentation.shotplan.convert_claims_to_newspapers)."""

from backend.presentation.models import Beat, PresentationSettings, Program
from backend.presentation.shotplan import convert_claims_to_newspapers


def _program(duration_s: float = 600.0) -> Program:
    return Program(duration_s=duration_s, fps=30.0)


def _stat(start_s: float, text: str = "25%", kind: str = "stat_callout", **kw) -> Beat:
    return Beat(kind=kind, start_s=start_s, end_s=start_s + 3.5, topic="a",
               text=text, origin="entity", **kw)


# --- genre eligibility ---------------------------------------------------------

def test_ineligible_genre_leaves_beats_untouched():
    beats = [_stat(10.0)]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "vlog", None)
    assert out[0].kind == "stat_callout"


def test_eligible_primary_genre_converts_a_stat():
    beats = [_stat(10.0)]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "true_crime", None)
    assert out[0].kind == "newspaper"
    assert out[0].data["headline"]
    assert out[0].data["highlight"] in out[0].data["headline"]
    # The highlight is drawn from the beat's own spoken text.
    assert "25" in out[0].data["highlight"]
    assert out[0].end_s == out[0].start_s + 4.5


def test_secondary_genre_alone_is_enough_to_qualify():
    """vlog does not reach for newspaper_highlight on its own, but a
    true_crime secondary blends it in (text_fx.palette_for)."""
    beats = [_stat(10.0)]
    out_no_secondary = convert_claims_to_newspapers(
        beats, _program(), PresentationSettings(), "vlog", None)
    out_with_secondary = convert_claims_to_newspapers(
        beats, _program(), PresentationSettings(), "vlog", "true_crime")
    assert out_no_secondary[0].kind == "stat_callout"
    assert out_with_secondary[0].kind == "newspaper"


# --- which beats are suitable ---------------------------------------------------

def test_quote_beats_always_qualify_even_with_no_numbers_or_cues():
    beats = [_stat(10.0, text="“nothing lasts forever”", kind="quote_card")]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "documentary", None)
    assert out[0].kind == "newspaper"


def test_a_number_makes_a_definition_card_qualify():
    beats = [_stat(10.0, text="the 1987 Cornell experiment", kind="definition_card")]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "documentary", None)
    assert out[0].kind == "newspaper"


def test_a_research_cue_makes_a_definition_card_qualify():
    beats = [_stat(10.0, text="according to a new study", kind="definition_card")]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "documentary", None)
    assert out[0].kind == "newspaper"


def test_plain_prose_with_no_number_or_cue_does_not_qualify():
    beats = [_stat(10.0, text="spotlight effect", kind="definition_card")]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "documentary", None)
    assert out[0].kind == "definition_card"


def test_kinds_outside_the_claim_family_are_never_converted():
    beats = [Beat(kind="chapter_title", start_s=10.0, end_s=12.0, topic="a",
                  text="Chapter 1987", origin="structure")]
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "documentary", None)
    assert out[0].kind == "chapter_title"


# --- density cap and crowding ---------------------------------------------------

def test_density_cap_keeps_only_one_conversion_within_the_balanced_window():
    beats = [_stat(10.0), _stat(60.0, text="3.5 crore")]  # 50s apart, < 90s
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "true_crime", None)
    kinds = sorted(b.kind for b in out)
    assert kinds == ["newspaper", "stat_callout"]


def test_density_cap_is_tighter_at_max_density():
    beats = [_stat(10.0), _stat(60.0, text="3.5 crore")]  # 50s apart
    out = convert_claims_to_newspapers(
        beats, _program(), PresentationSettings(density="max"), "true_crime", None)
    kinds = sorted(b.kind for b in out)
    assert kinds == ["newspaper", "newspaper"]


def test_a_beat_within_the_gap_of_an_existing_newspaper_beat_is_not_converted():
    existing = Beat(kind="newspaper", start_s=5.0, end_s=9.5, topic="a", text="OLD NEWS",
                    data={"headline": "OLD NEWS", "highlight": "OLD"}, origin="entity")
    beats = [existing, _stat(20.0)]  # 15s from the existing newspaper beat
    out = convert_claims_to_newspapers(beats, _program(), PresentationSettings(),
                                       "true_crime", None)
    # The new stat stayed a card; only the pre-existing newspaper beat is one.
    assert sum(1 for b in out if b.kind == "newspaper") == 1
    stat_beats = [b for b in out if b.kind == "stat_callout"]
    assert len(stat_beats) == 1 and stat_beats[0].start_s == 20.0


# --- settings and no-ops ---------------------------------------------------------

def test_newspapers_off_is_a_no_op():
    beats = [_stat(10.0)]
    out = convert_claims_to_newspapers(
        beats, _program(), PresentationSettings(newspapers=False), "true_crime", None)
    assert out[0].kind == "stat_callout"


def test_empty_beats_is_a_no_op():
    assert convert_claims_to_newspapers([], _program(), PresentationSettings(),
                                        "true_crime", None) == []
