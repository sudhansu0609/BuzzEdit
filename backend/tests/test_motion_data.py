"""presentation/motion_data.py (MOTION_GRAPHICS_PLAN.md M2): the ``map``,
``chart``, ``stat_callout`` and ``timeline`` builders that turn a beat into a
MotionMap/MotionChart/MotionStat/MotionTimeline job spec. The map tests need
the Natural Earth data (``needs_data``); chart/stat/timeline do not touch
disk at all. A handful render the real engine, skipped like test_motion.py's
when it is not installed."""

import pytest

from presentation import maps as maps_stage
from presentation import motion
from presentation.models import Beat, PresentationSettings
from presentation.motion_data import build_chart, build_map, build_stat, build_timeline

needs_data = pytest.mark.skipif(not maps_stage.data_available(),
                                reason="Natural Earth data not downloaded")
needs_engine = pytest.mark.skipif(not motion.available(),
                                  reason="motion engine not installed (npm install in motion/)")


def _beat(**kwargs) -> Beat:
    base = dict(start_s=0.0, end_s=6.0, topic="a topic", kind="map", priority=1.0)
    base.update(kwargs)
    return Beat(**base)


def _ctx(**kwargs) -> motion.MotionContext:
    base = dict(width=320, height=180, fps=30.0, genre="general",
               theme=motion.theme_for("general", "real"), settings=PresentationSettings())
    base.update(kwargs)
    return motion.MotionContext(**base)


# --- map --------------------------------------------------------------------

def test_build_map_returns_none_without_a_place():
    assert build_map(_beat(place=None), _ctx()) is None


def test_build_map_returns_none_when_data_unavailable(monkeypatch):
    monkeypatch.setattr(maps_stage, "data_available", lambda: False)
    assert build_map(_beat(place="Jaipur"), _ctx()) is None


@needs_data
def test_build_map_builds_a_job_with_rounded_places():
    beat = _beat(place="Jaipur", text="Jaipur")
    job = build_map(beat, _ctx())
    assert job["composition"] == "MotionMap"
    assert job["props"]["theme"] == _ctx().theme
    assert job["props"]["route"] is False
    assert job["props"]["label"] == "Jaipur"
    places = job["props"]["places"]
    assert len(places) == 1 and places[0]["name"] == "Jaipur"
    # Rounded to 2 decimals (Architecture: "coordinates rounded to 2 dp").
    assert places[0]["lat"] == round(places[0]["lat"], 2)
    assert places[0]["lon"] == round(places[0]["lon"], 2)
    assert job["transparent"] is False


@needs_data
def test_build_map_marks_a_route_for_several_places_with_the_route_flag():
    beat = _beat(place="Jaipur -> Delhi -> Agra", data={"route": True})
    job = build_map(beat, _ctx())
    assert job["props"]["route"] is True
    assert len(job["props"]["places"]) == 3


@needs_data
def test_build_map_is_not_a_route_without_the_flag():
    # places_for_beat only splits `place` on "->" when `data["route"]` is set
    # (maps.places_for_beat) -- without it a single real place still builds a
    # non-route job.
    beat = _beat(place="Jaipur", data={})
    job = build_map(beat, _ctx())
    assert job["props"]["route"] is False
    assert len(job["props"]["places"]) == 1


@needs_data
def test_build_map_highlights_a_country_but_not_a_city():
    country_job = build_map(_beat(place="India"), _ctx())
    assert country_job["props"]["highlight"] is not None
    assert country_job["props"]["highlight"]["rings"]

    city_job = build_map(_beat(place="Jaipur"), _ctx())
    assert city_job["props"]["highlight"] is None


@needs_data
def test_build_map_returns_none_for_an_ungeocodable_place():
    assert build_map(_beat(place="Not A Real Place Xyz123"), _ctx()) is None


@needs_data
@needs_engine
def test_build_map_renders_a_real_video_through_render_motion_assets(tmp_path):
    beat = _beat(id="m00", place="Jaipur", planned_duration_s=2.5)
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "general", PresentationSettings())
    assert failures == [] and handled == {"m00"}
    assert len(assets) == 1
    from pathlib import Path
    asset = assets[0]
    assert asset.kind == "video" and asset.source == "motion"
    assert Path(asset.path).exists() and Path(asset.path).stat().st_size > 0


# --- chart --------------------------------------------------------------------

def _chart_beat(**kwargs) -> Beat:
    return _beat(kind="chart", place=None, **kwargs)


def test_build_chart_returns_none_without_data():
    assert build_chart(_chart_beat(data={}), _ctx()) is None


def test_build_chart_years_mode():
    beat = _chart_beat(data={"years": [2020, 2021, 2022], "active": 2022})
    job = build_chart(beat, _ctx())
    assert job["composition"] == "MotionChart"
    assert job["props"]["mode"] == "years"
    assert job["props"]["years"] == ["2020", "2021", "2022"]
    assert job["props"]["active"] == "2022"


def test_build_chart_years_mode_defaults_active_to_the_last_year():
    beat = _chart_beat(data={"years": [2020, 2021]})
    job = build_chart(beat, _ctx())
    assert job["props"]["active"] == "2021"


def test_build_chart_bar_is_the_default_mode():
    beat = _chart_beat(data={"labels": ["A", "B"], "values": [1, 2]})
    job = build_chart(beat, _ctx())
    assert job["props"]["mode"] == "bar"
    assert job["props"]["values"] == [1.0, 2.0]


@pytest.mark.parametrize("chart_type", ["line", "graph"])
def test_build_chart_line_mode_for_line_or_graph_chart_type(chart_type):
    beat = _chart_beat(data={"labels": ["A", "B"], "values": [1, 2], "chart_type": chart_type})
    job = build_chart(beat, _ctx())
    assert job["props"]["mode"] == "line"


def test_build_chart_pie_mode():
    beat = _chart_beat(data={"labels": ["A", "B"], "values": [1, 2], "chart_type": "pie"})
    job = build_chart(beat, _ctx())
    assert job["props"]["mode"] == "pie"


def test_build_chart_returns_none_for_non_numeric_values():
    beat = _chart_beat(data={"labels": ["A"], "values": ["not-a-number"]})
    assert build_chart(beat, _ctx()) is None


def test_build_chart_returns_none_without_labels_or_values():
    assert build_chart(_chart_beat(data={"values": [1, 2]}), _ctx()) is None
    assert build_chart(_chart_beat(data={"labels": ["A"]}), _ctx()) is None


@needs_engine
def test_build_chart_renders_a_real_video_through_render_motion_assets(tmp_path):
    beat = _chart_beat(id="c00", planned_duration_s=2.5,
                       data={"labels": ["A", "B"], "values": [10, 20]})
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "general", PresentationSettings())
    assert failures == [] and handled == {"c00"}
    assert assets[0].kind == "video" and assets[0].source == "motion"


# --- stat callout -------------------------------------------------------------

def test_build_stat_returns_none_without_a_value():
    beat = _beat(kind="stat_callout", place=None, data={})
    assert build_stat(beat, _ctx()) is None


def test_build_stat_builds_a_job_with_prefix_suffix_decimals_and_label():
    beat = _beat(kind="stat_callout", place=None, subtext="since 1990",
                data={"value": 2847, "prefix": "", "suffix": "+", "decimals": 1})
    job = build_stat(beat, _ctx())
    assert job["composition"] == "MotionStat"
    assert job["props"]["value"] == 2847.0
    assert job["props"]["suffix"] == "+"
    assert job["props"]["decimals"] == 1
    assert job["props"]["label"] == "since 1990"


# --- timeline -------------------------------------------------------------------

def test_build_timeline_builds_a_job_with_ticks_and_default_active():
    beat = _beat(kind="timeline", place=None, text="Headline", subtext="Caption",
                data={"ticks": ["2001", "2016", "2024"]})
    job = build_timeline(beat, _ctx())
    assert job["composition"] == "MotionTimeline"
    assert job["props"]["ticks"] == ["2001", "2016", "2024"]
    assert job["props"]["active"] == 2  # len(ticks) - 1
    assert job["props"]["headline"] == "Headline"
    assert job["props"]["caption"] == "Caption"


def test_build_timeline_never_reports_no_content_even_with_no_ticks():
    # graphics.render_timeline draws unconditionally (DRAWN_CUTAWAY_KINDS);
    # motion_data.build_timeline mirrors that by always returning a job.
    beat = _beat(kind="timeline", place=None, data={})
    job = build_timeline(beat, _ctx())
    assert job is not None
    assert job["props"]["ticks"] == ["", ""]


@needs_engine
def test_build_timeline_renders_a_real_video_through_render_motion_assets(tmp_path):
    beat = _beat(id="t00", kind="timeline", place=None, planned_duration_s=2.5,
                text="Headline", data={"ticks": ["2001", "2024"]})
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "general", PresentationSettings())
    assert failures == [] and handled == {"t00"}
    assert assets[0].kind == "video" and assets[0].source == "motion"


@needs_engine
def test_build_stat_renders_a_real_video_through_render_motion_assets(tmp_path):
    beat = _beat(id="s00", kind="stat_callout", place=None, planned_duration_s=2.5,
                subtext="label", data={"value": 100})
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "general", PresentationSettings())
    assert failures == [] and handled == {"s00"}
    assert assets[0].kind == "video" and assets[0].source == "motion"


# --- fallback: the PIL stage still draws when motion cannot ------------------
# map/chart/timeline fall back through the existing PIL renderers because
# director._draw_local_assets only removes a beat from its own `beats` list
# once motion actually handles it (MOTION_GRAPHICS_PLAN.md M2 Architecture:
# "a beat it handled are skipped ... a failed beat stays for PIL").

@needs_data
def test_draw_local_assets_falls_back_to_the_pil_map_when_motion_is_unavailable(
        tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "available", lambda: False)
    from presentation.director import _draw_local_assets
    beat = _beat(id="b00", kind="map", place="Jaipur")
    assets, failures = _draw_local_assets([beat], tmp_path, 320, 180, "general",
                                          PresentationSettings())
    assert len(assets) == 1
    assert assets[0].beat_id == "b00" and assets[0].kind == "image"
    assert any("motion" in f.get("reason", "") for f in failures)


def test_draw_local_assets_falls_back_to_the_pil_chart_when_motion_is_unavailable(
        tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "available", lambda: False)
    from presentation.director import _draw_local_assets
    beat = _chart_beat(id="b00", data={"labels": ["A", "B"], "values": [1, 2]})
    assets, failures = _draw_local_assets([beat], tmp_path, 320, 180, "general",
                                          PresentationSettings())
    assert len(assets) == 1
    assert assets[0].beat_id == "b00" and assets[0].kind == "image"
    assert any("motion" in f.get("reason", "") for f in failures)


def test_draw_local_assets_falls_back_to_the_pil_timeline_when_motion_is_unavailable(
        tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "available", lambda: False)
    from presentation.director import _draw_local_assets
    beat = _beat(id="b00", kind="timeline", place=None, text="Headline",
                data={"ticks": ["2001", "2024"]})
    assets, failures = _draw_local_assets([beat], tmp_path, 320, 180, "general",
                                          PresentationSettings())
    assert len(assets) == 1
    assert assets[0].beat_id == "b00" and assets[0].kind == "image"
    assert any("motion" in f.get("reason", "") for f in failures)
