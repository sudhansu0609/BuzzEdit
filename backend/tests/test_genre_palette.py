"""documentary/mystery/geopolitics genres and the per-genre effect palette."""

from presentation import genre as g


def test_new_genre_aliases_normalise_correctly():
    assert g.normalise("documentary") == "documentary"
    assert g.normalise("geopolitical") == "geopolitics"
    assert g.normalise("financial documentary") == "finance"


def test_new_genres_have_a_look():
    for name in ("documentary", "mystery", "geopolitics"):
        assert name in g.GENRE_STYLES
        assert g.GENRE_STYLES[name].look


def test_new_genres_have_a_grade_and_atmosphere():
    assert g.grade_for("mystery")
    assert g.atmosphere_for("mystery")


def test_fx_palette_covers_the_expected_genres():
    assert "flicker" in g.fx_palette_for("horror")
    assert "redaction" in g.fx_palette_for("mystery")
