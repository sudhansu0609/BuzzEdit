"""Maps and charts drawn offline, and their path into the B-roll placement."""

import pytest

from backend.presentation import charts, maps
from backend.presentation.models import Beat, PresentationSettings
from backend.presentation.placement import BROLL_ORIGIN, place_broll
from backend.presentation.program import build_program
from backend.timeline import build_timeline_from_transcript

needs_data = pytest.mark.skipif(not maps.data_available(), reason="Natural Earth data not downloaded")


def _timeline(seconds: float = 30.0):
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i in range(int(seconds * 2) - 1)]
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", seconds, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, seconds)], pause_padding_seconds=0.0)


# --- geocoding ------------------------------------------------------------------

@needs_data
def test_cities_countries_states_and_aliases_geocode_and_nonsense_does_not():
    jaipur = maps.geocode("Jaipur")
    assert jaipur and jaipur.kind == "city" and jaipur.country == "India"
    assert 26 < jaipur.lat < 27 and 75 < jaipur.lon < 76
    assert maps.geocode("Bombay").name == "Mumbai"
    assert maps.geocode("India").kind == "country"
    assert maps.geocode("Rajasthan").kind == "state"
    assert maps.geocode("Kolkata, West Bengal").name == "Kolkata"
    assert maps.geocode("Cornell University") is None
    assert maps.geocode("") is None
    # The biggest of several same-named places wins.
    assert maps.geocode("Hyderabad").country == "India"


def test_geocoding_without_data_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(maps, "COUNTRIES_FILE", tmp_path / "none.geojson")
    monkeypatch.setattr(maps, "PLACES_FILE", tmp_path / "none2.geojson")
    assert maps.geocode("Jaipur") is None
    assert not maps.data_available()


# --- rendering ----------------------------------------------------------------------

@needs_data
def test_maps_render_at_the_canvas_size_in_both_looks(tmp_path):
    place = maps.geocode("Jaipur")
    noir = maps.render_map([place], 640, 360, "noir", label="Jaipur")
    assert noir.size == (640, 360)
    clean = maps.render_map([place, maps.geocode("Delhi")], 640, 360, "clean")
    assert clean.size == (640, 360)
    # The two looks differ in their water colour at a corner.
    assert noir.getpixel((5, 5)) != clean.getpixel((5, 5))
    # A portrait canvas still frames the place (no crash, right size).
    country = maps.render_map([maps.geocode("India")], 360, 640, "clean")
    assert country.size == (360, 640)


@needs_data
def test_map_assets_are_cached_and_unknown_places_are_reported(tmp_path):
    beats = [Beat(id="m1", kind="map", start_s=1, end_s=5, topic="a", place="Jaipur", text="Jaipur"),
             Beat(id="m2", kind="map", start_s=6, end_s=10, topic="b", place="Nowhere Land"),
             Beat(id="x", kind="broll_image", start_s=1, end_s=5, topic="c", image_prompt="p")]
    assets, failures = maps.render_map_assets(beats, tmp_path, 320, 180, "horror")
    assert [a.beat_id for a in assets] == ["m1"]
    assert assets[0].kind == "image" and assets[0].path.endswith(".png")
    assert failures == [{"beat_id": "m2", "reason": "unknown place 'Nowhere Land'"}]
    again, _ = maps.render_map_assets(beats, tmp_path, 320, 180, "horror")
    assert again[0].path == assets[0].path
    assert maps.style_for("horror") == "noir" and maps.style_for("tech") == "clean"
    assert maps.style_for("horror", "clean") == "clean"


def test_charts_render_bars_and_timelines(tmp_path):
    bars = charts.render_bar_chart(["Students", "Guess"], [25, 50], 640, 360, unit="%",
                                   title="Who noticed", style="clean")
    assert bars.size == (640, 360)
    strip = charts.render_timeline_strip(["1987", "1992", "2001"], 640, 360, title="The years",
                                         style="noir", active="1992")
    assert strip.size == (640, 360)
    beats = [Beat(id="c1", kind="chart", start_s=1, end_s=6, topic="a", text="Who noticed",
                  data={"labels": ["A", "B"], "values": [25, 50], "unit": "%"}),
             Beat(id="c2", kind="chart", start_s=8, end_s=12, topic="b", text="Years",
                  data={"years": ["1987", "2001"], "active": "2001"})]
    assets, failures = charts.render_chart_assets(beats, tmp_path, 320, 180, "horror")
    assert [a.beat_id for a in assets] == ["c1", "c2"] and not failures


# --- placement ---------------------------------------------------------------------

@needs_data
def test_a_map_takes_the_b_roll_placement_path(tmp_path):
    timeline = _timeline()
    program = build_program(timeline)
    beat = Beat(id="m1", kind="map", start_s=4.0, end_s=9.0, topic="a", place="Jaipur",
                text="Jaipur", planned_duration_s=3.0)
    assets, _ = maps.render_map_assets([beat], tmp_path, 320, 180, "horror")
    placed = place_broll(timeline, [beat], assets, program, PresentationSettings())
    assert placed == 1
    item = next(i for i in timeline.items if i.origin == BROLL_ORIGIN)
    assert timeline.sources[item.source_id].kind == "image"
    assert item.transform is not None and item.transform.is_animated()
