"""Tests for the BuzzcafStudio contract: density presets, the secondary genre
blend, density-aware zoom/crowding, and BuzzEdit-side overrides that beat
whatever Studio sent.
"""

import pytest
from fastapi.testclient import TestClient

from presentation import genre as genre_mod
from presentation.facezoom import _plan_punches, plan_zooms
from presentation.models import (DENSITY_PRESETS, Beat, PresentationSettings, Program,
                                 ProgramSegment, ProgramWord, WindowZoom,
                                 settings_sources_for)
from presentation.shotplan import _ensure_coverage_candidates, budget_beats


# --- PresentationSettings.resolve_density / effective_* / crowd_gap_s ------

def test_resolve_density_fills_only_unset_fields():
    settings = PresentationSettings(density="busy")
    resolved = settings.resolve_density()
    assert resolved.target_coverage == DENSITY_PRESETS["busy"]["target_coverage"]
    assert resolved.min_oncamera_gap_s == DENSITY_PRESETS["busy"]["min_oncamera_gap_s"]
    assert resolved.effective_punch_rate_per_minute == DENSITY_PRESETS["busy"]["punch_rate_per_minute"]


def test_resolve_density_leaves_an_explicit_value_alone():
    settings = PresentationSettings(density="busy", target_coverage=0.42)
    resolved = settings.resolve_density()
    assert resolved.target_coverage == 0.42


def test_resolve_density_with_no_density_is_a_no_op():
    settings = PresentationSettings()
    resolved = settings.resolve_density()
    assert resolved.target_coverage == settings.target_coverage
    assert resolved.min_oncamera_gap_s == settings.min_oncamera_gap_s


def test_resolve_density_with_an_unknown_name_is_a_no_op():
    settings = PresentationSettings(density="extreme")
    resolved = settings.resolve_density()
    assert resolved.target_coverage == settings.target_coverage


def test_effective_punch_rate_falls_back_to_1_5_with_no_density():
    assert PresentationSettings().effective_punch_rate_per_minute == 1.5
    assert PresentationSettings(punch_rate_per_minute=4.0).effective_punch_rate_per_minute == 4.0
    assert (PresentationSettings(density="max").effective_punch_rate_per_minute
            == DENSITY_PRESETS["max"]["punch_rate_per_minute"])


def test_effective_text_fx_falls_back_to_2_0_with_no_density():
    assert PresentationSettings().effective_text_fx_per_minute == 2.0
    assert (PresentationSettings(density="calm").effective_text_fx_per_minute
            == DENSITY_PRESETS["calm"]["text_fx_per_minute"])


def test_crowd_gap_s_is_zero_at_busy_and_max_only():
    assert PresentationSettings(density="busy").crowd_gap_s == 0.0
    assert PresentationSettings(density="max").crowd_gap_s == 0.0
    calm = PresentationSettings(density="calm", min_oncamera_gap_s=1.5)
    assert calm.crowd_gap_s == 1.5
    assert PresentationSettings(min_oncamera_gap_s=0.8).crowd_gap_s == 0.8


def test_settings_sources_for_reports_studio_override_density_and_default():
    settings = PresentationSettings(density="busy").resolve_density()
    sources = settings_sources_for(
        incoming={"target_coverage": 0.5},
        override_fields={"zoom_depth": 0.3},
        resolved=settings,
    )
    assert sources["target_coverage"] == "studio"
    assert sources["zoom_depth"] == "buzzedit_override"
    assert sources["video_broll_share"] == "density"
    assert sources["genre"] == "default"


# --- secondary genre blend ---------------------------------------------------

def test_blended_fx_palette_keeps_primary_and_adds_secondary():
    primary_only = genre_mod.fx_palette_for("horror")
    blended = genre_mod.blended_fx_palette("horror", "comedy")
    assert blended[:len(primary_only)] == primary_only
    assert any(name not in primary_only for name in blended)


def test_blended_fx_palette_with_no_secondary_is_just_the_primary():
    assert genre_mod.blended_fx_palette("horror", None) == genre_mod.fx_palette_for("horror")
    assert genre_mod.blended_fx_palette("horror", "horror") == genre_mod.fx_palette_for("horror")
    assert genre_mod.blended_fx_palette("horror", "general") == genre_mod.fx_palette_for("horror")


def test_apply_look_blends_secondary_style_tags():
    primary = genre_mod.apply_look("a scene", "horror")
    blended = genre_mod.apply_look("a scene", "horror", "comedy")
    assert primary in blended or primary == blended
    # comedy's own look words should show up somewhere the primary's did not.
    comedy_tags = genre_mod._style_tags("comedy")
    assert any(tag in blended for tag in comedy_tags)


def test_apply_negative_blends_secondary_negative():
    negative = genre_mod.apply_negative("blurry", "horror", "comedy")
    assert genre_mod.style_for("horror").negative in negative
    assert genre_mod.style_for("comedy").negative in negative


# --- facezoom: punch allowance, adaptive threshold, direction, whip --------

def _program(segment_count: int = 4, segment_seconds: float = 30.0, z: float = 3.0) -> Program:
    """Every word maximally emphatic, so allowance (not peak scarcity) governs
    how many punches land."""
    words = []
    segments = []
    for index in range(segment_count):
        start = index * segment_seconds
        segments.append(ProgramSegment(
            item_id=f"v1_{index}", tl_start_s=start, tl_end_s=start + segment_seconds,
            source_start_frame=int(start * 30), source_end_frame=int((start + segment_seconds) * 30)))
        clock = start
        while clock < start + segment_seconds - 0.5:
            words.append(ProgramWord(text=f"w{len(words)}", tl_start_s=clock, tl_end_s=clock + 0.4,
                                     source_start_frame=int(clock * 30), emphasis_z=z))
            clock += 0.5
    return Program(duration_s=segment_count * segment_seconds, fps=30.0,
                  words=words, segments=segments, has_energy=True)


def test_punch_allowance_scales_with_density():
    program = _program()
    calm_punches = _plan_punches(program, [], PresentationSettings(density="calm"), seed=0)
    max_punches = _plan_punches(program, [], PresentationSettings(density="max"), seed=0)
    assert len(max_punches) > len(calm_punches)


def test_adaptive_threshold_finds_peaks_when_scarce():
    """One word above the historical threshold, but a busy/max allowance asks
    for more punches than that one peak — the threshold must ease down rather
    than under-filling."""
    program = _program(segment_count=1, segment_seconds=90.0, z=0.0)
    # A single stand-out word, everything else silent (z=0.0, below the floor).
    program.words[10].emphasis_z = 1.3
    settings = PresentationSettings(density="max")
    punches = _plan_punches(program, [], settings, seed=0)
    assert len(punches) == 1


def test_punches_alternate_push_in_and_pull_back():
    program = _program(segment_count=6, segment_seconds=30.0)
    settings = PresentationSettings(density="max")
    punches = _plan_punches(program, [], settings, seed=0)
    assert len(punches) >= 2
    directions = [p.push_in for p in punches]
    assert True in directions and False in directions


def test_whip_only_appears_at_busy_or_max_density():
    program = _program(segment_count=10, segment_seconds=30.0)
    calm_punches = _plan_punches(program, [], PresentationSettings(density="calm"), seed=0)
    max_punches = _plan_punches(program, [], PresentationSettings(density="max"), seed=0)
    assert not any(p.reason.startswith("whip") for p in calm_punches)
    # A long enough programme at max density should produce at least one.
    assert any(p.reason.startswith("whip") for p in max_punches)


def test_plan_zooms_reads_punch_rate_from_settings():
    program = _program(segment_count=2, segment_seconds=60.0)
    _, few = plan_zooms(program, PresentationSettings(punch_rate_per_minute=0.5), [], seed=0)
    _, many = plan_zooms(program, PresentationSettings(punch_rate_per_minute=5.0), [], seed=0)
    assert len(many) > len(few)


# --- shotplan: density-scaled crowding and the coverage top-up -------------

def _cutaway(start_s, end_s, priority=0.8):
    return Beat(start_s=start_s, end_s=end_s, topic=f"t{start_s}", kind="broll_image",
               priority=priority, image_prompt="a photograph")


def test_budget_beats_rejects_back_to_back_at_default_density():
    settings = PresentationSettings(broll_seconds_min=3.0, broll_seconds_max=3.0,
                                    target_coverage=0.75)
    beats = [_cutaway(0.0, 3.0), _cutaway(3.0, 6.0)]
    program = Program(duration_s=60.0, fps=30.0)
    kept, dropped = budget_beats(beats, program, settings)
    assert len(kept) == 1
    assert dropped and dropped[0]["reason"] == "too close to another cutaway"


def test_budget_beats_allows_back_to_back_at_busy_density():
    settings = PresentationSettings(broll_seconds_min=3.0, broll_seconds_max=3.0,
                                    target_coverage=0.75, density="busy")
    beats = [_cutaway(0.0, 3.0), _cutaway(3.0, 6.0)]
    program = Program(duration_s=60.0, fps=30.0)
    kept, dropped = budget_beats(beats, program, settings)
    assert len(kept) == 2
    assert not dropped


def _words(count: int, spacing: float = 0.4):
    out = []
    clock = 0.0
    for i in range(count):
        out.append(ProgramWord(text=f"w{i}", tl_start_s=clock, tl_end_s=clock + spacing * 0.8,
                               source_start_frame=int(clock * 30), emphasis_z=0.0))
        clock += spacing
    return out


def test_ensure_coverage_candidates_tops_up_a_short_plan():
    words = _words(400, spacing=0.5)          # 200s of speech to draw filler from
    program = Program(duration_s=words[-1].tl_end_s + 1.0, fps=30.0, words=words, has_energy=False)
    settings = PresentationSettings(target_coverage=0.5, broll_seconds_min=2.0, broll_seconds_max=3.0)
    # One tiny beat: nowhere near the coverage budget on its own.
    unique = [_cutaway(0.0, 2.5)]
    topped_up = _ensure_coverage_candidates(unique, program, settings, "general", None)
    assert len(topped_up) > len(unique)
    have = sum((b.planned_duration_s or min(settings.broll_seconds_max, b.duration_s))
              for b in topped_up if b.is_cutaway)
    original = sum((b.planned_duration_s or min(settings.broll_seconds_max, b.duration_s))
                   for b in unique if b.is_cutaway)
    assert have > original


def test_ensure_coverage_candidates_is_a_no_op_once_the_budget_is_met():
    program = Program(duration_s=60.0, fps=30.0, words=_words(20))
    settings = PresentationSettings(target_coverage=0.1, broll_seconds_min=2.0, broll_seconds_max=3.0)
    unique = [_cutaway(0.0, 10.0)]     # already well past a 6s (0.1 * 60) budget
    topped_up = _ensure_coverage_candidates(unique, program, settings, "general", None)
    assert topped_up == unique


# --- store/app_settings: overrides beat Studio ------------------------------

@pytest.fixture
def isolated_app_settings(tmp_path, monkeypatch):
    from store.app_settings import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", tmp_path)
    monkeypatch.setattr(app_settings, "settings_file", tmp_path / "app_settings.json")
    monkeypatch.setattr(app_settings, "_cache", {
        "last_project_id": None, "auto_save_enabled": True,
        "auto_save_interval_seconds": 30, "last_save_timestamp": None,
    })
    yield app_settings


def test_apply_overrides_is_a_no_op_when_mode_is_off(isolated_app_settings):
    from store.app_settings import apply_presentation_overrides
    merged, fields = apply_presentation_overrides({"genre": "horror"})
    assert merged == {"genre": "horror"}
    assert fields == {}


def test_apply_overrides_beats_studio_when_a_field_is_locked(isolated_app_settings):
    from store.app_settings import apply_presentation_overrides, set_presentation_overrides
    set_presentation_overrides({"zoom_depth": 0.3}, "fields")
    merged, fields = apply_presentation_overrides({"zoom_depth": 0.1, "genre": "horror"})
    assert merged["zoom_depth"] == 0.3
    assert merged["genre"] == "horror"
    assert fields == {"zoom_depth": 0.3}


def test_set_and_get_presentation_overrides_roundtrip(isolated_app_settings):
    from store.app_settings import get_presentation_overrides_state, set_presentation_overrides
    set_presentation_overrides({"density": "max"}, "all")
    state = get_presentation_overrides_state()
    assert state == {"presentation_overrides": {"density": "max"}, "override_mode": "all"}


def test_bad_override_mode_falls_back_to_off(isolated_app_settings):
    from store.app_settings import get_override_mode, set_presentation_overrides
    set_presentation_overrides({"density": "max"}, "nonsense")
    assert get_override_mode() == "off"


# --- routes/settings.py: GET/PUT /api/settings/presentation_overrides ------

@pytest.fixture
def client(isolated_app_settings):
    from main import app
    with TestClient(app) as test_client:
        yield test_client


def test_presentation_overrides_endpoint_roundtrip(client):
    empty = client.get("/api/settings/presentation_overrides")
    assert empty.status_code == 200
    assert empty.json() == {"presentation_overrides": {}, "override_mode": "off"}

    put = client.put("/api/settings/presentation_overrides", json={
        "presentation_overrides": {"voice_preset": "broadcast"}, "override_mode": "fields",
    })
    assert put.status_code == 200
    assert put.json() == {"presentation_overrides": {"voice_preset": "broadcast"},
                          "override_mode": "fields"}

    again = client.get("/api/settings/presentation_overrides")
    assert again.json() == put.json()


def test_presentation_overrides_endpoint_rejects_a_bad_mode(client):
    res = client.put("/api/settings/presentation_overrides", json={
        "presentation_overrides": {}, "override_mode": "bogus",
    })
    assert res.status_code == 422
