"""The on-screen progress bar: filter construction, chunk offsets, the pass's layer."""
import re

import pytest

from presentation import structure
from presentation.director import (PROGRESS_ORIGIN, _apply_progress_bar,
                                   _progress_bar_wanted, _progress_placement)
from presentation.models import PresentationReport, PresentationSettings, Topic
from render.atmosphere import PROGRESS_BAR_DEFAULTS, available, build_effect
from render.chunked import sub_timeline
from timeline.schema import AtmosphereEffect, TextClip, TextStyle, Timeline, TimelineItem

CHAPTERS = [{"at_s": 0.0, "name": "Intro"}, {"at_s": 15.0, "name": "The House"},
            {"at_s": 30.0, "name": "रात तीन बजे"}]


def _bar(tmp_path, programme_s=40.0, **extra):
    effect = AtmosphereEffect(type="progress_bar", intensity=extra.pop("intensity", 1.0),
                              color=extra.pop("color", None), extra=extra)
    return build_effect(effect, "[v]", "[o]", 1920, 1080, 30.0, programme_s + 1.0, 0,
                        assets_dir=tmp_path, programme_s=programme_s)


# --- the filter --------------------------------------------------------------

def test_bar_without_chapters_is_track_and_sliding_fill_only(tmp_path):
    out = _bar(tmp_path)
    joined = ";".join(out)
    assert out[0].startswith("[v]drawbox=x=0:y=1074:w=1920:h=6:")
    assert "overlay=x='-1920+1920*min(1,t/40)'" in joined
    assert "s=1920x6" in joined, "the fill is a full-width box slid in"
    assert "drawtext" not in joined and joined.count("drawbox") == 1
    assert out[-1].endswith("[o]")


def test_chapters_give_ticks_and_one_label_each(tmp_path):
    out = _bar(tmp_path, chapters=CHAPTERS)
    joined = ";".join(out)
    # ticks at 15/40 and 30/40 of the width, none at 0
    assert "drawbox=x=719:y=1074:w=3:h=6" in joined
    assert "drawbox=x=1439:y=1074:w=3:h=6" in joined
    assert joined.count("drawbox") == 3
    assert "enable='between(t,0,15)'" in joined
    assert "enable='between(t,15,30)'" in joined
    assert "enable='between(t,30,41)'" in joined
    assert "alpha='0.7*clip((t-15)/1.2,0,1)'" in joined
    assert "fontsize=28" in joined and "y='1066-text_h'" in joined


def test_every_label_pins_a_font_and_devanagari_gets_nirmala(tmp_path):
    out = _bar(tmp_path, chapters=CHAPTERS)
    texts = [s for s in ";".join(out).split("drawtext=")[1:]]
    assert len(texts) == 3
    for t in texts:
        assert "fontfile='" in t
    devanagari = texts[2]
    assert "nirmala" in devanagari.lower()


def test_position_top_puts_bar_at_margin_and_label_below(tmp_path):
    out = _bar(tmp_path, chapters=CHAPTERS, position="top", margin_px=10)
    joined = ";".join(out)
    assert out[0].startswith("[v]drawbox=x=0:y=10:w=1920:h=6:")
    assert "y='24'" in joined   # 10 + 6 + 8 gap


def test_bottom_margin_lifts_the_bar(tmp_path):
    out = _bar(tmp_path, margin_px=232)
    assert out[0].startswith("[v]drawbox=x=0:y=842:")


def test_label_none_and_ticks_false(tmp_path):
    joined = ";".join(_bar(tmp_path, chapters=CHAPTERS, label="none", ticks=False))
    assert "drawtext" not in joined and joined.count("drawbox") == 1


def test_intensity_scales_opacity_and_color_overrides_fill(tmp_path):
    full = ";".join(_bar(tmp_path))
    half = ";".join(_bar(tmp_path, intensity=0.5, color="#00AAFF"))
    assert "color=c=0xFFD400@1:" in full
    assert "color=c=0x00AAFF@0.7:" in half
    track_full = float(re.search(r"color=0xFFFFFF@([\d.]+)", full).group(1))
    track_half = float(re.search(r"color=0xFFFFFF@([\d.]+)", half).group(1))
    assert track_half == pytest.approx(track_full * 0.7, abs=0.002)


def test_scales_with_canvas_height(tmp_path):
    effect = AtmosphereEffect(type="progress_bar", intensity=1.0, extra={"chapters": CHAPTERS})
    joined = ";".join(build_effect(effect, "[v]", "[o]", 3840, 2160, 30.0, 41.0, 0,
                                   assets_dir=tmp_path, programme_s=40.0))
    assert "y=2148:w=3840:h=12" in joined and "fontsize=56" in joined


def test_chunk_offset_puts_the_bar_on_programme_time(tmp_path):
    out = ";".join(_bar(tmp_path, programme_s=10.0, chapters=CHAPTERS,
                        _offset_s=20.0, _programme_s=40.0))
    assert "min(1,(t+20)/40)" in out
    assert "between((t+20),15,30)" in out


def test_sub_timeline_stamps_offset_and_programme_length():
    tl = Timeline(duration_frames=1200)
    tl.effects.append(AtmosphereEffect(type="progress_bar", extra={"chapters": CHAPTERS}))
    tl.effects.append(AtmosphereEffect(type="grain"))
    chunk = sub_timeline(tl, 600, 900)
    bar = chunk.effects[0]
    assert bar.extra["_offset_s"] == pytest.approx(20.0)
    assert bar.extra["_programme_s"] == pytest.approx(40.0)
    assert "_offset_s" not in chunk.effects[1].extra
    assert "_offset_s" not in tl.effects[0].extra, "the original is untouched"


def test_catalogue_lists_it_with_defaults():
    entry = next(e for e in available() if e["id"] == "progress_bar")
    assert entry["label"] == "Progress bar" and entry["description"]
    assert entry["extra"] == PROGRESS_BAR_DEFAULTS and entry["extra"]["chapters"] == []


# --- the presentation pass ---------------------------------------------------

def _topics():
    return [Topic(start_s=0.0, end_s=20.0, topic="intro"),
            Topic(start_s=20.0, end_s=60.0, topic="the house", heading="The House"),
            Topic(start_s=60.0, end_s=90.0, topic="the end")]


def _caption(pos_y=0.74, size=80, second=True):
    return TimelineItem(id=f"cap{pos_y}", track="CAP", kind="text", origin="caption",
                        timeline_start_frame=0, timeline_end_frame=30,
                        text=TextClip(content="short line",
                                      second_line="translation" if second else None,
                                      style=TextStyle(pos_y=pos_y, font_size=size,
                                                      stroke_width=6)))


def test_chapter_seconds_matches_the_stamps():
    secs = structure.chapter_seconds(_topics(), offset_s=5.0)
    assert secs == [(0.0, "intro"), (25.0, "The House"), (65.0, "the end")]
    assert structure.chapters_for(_topics(), offset_s=5.0) == [
        ("0:00", "intro"), ("0:25", "The House"), ("1:05", "the end")]


def test_director_adds_one_bar_and_replaces_it_on_rerun():
    tl = Timeline(duration_frames=2700, cold_open_frames=150)   # 5 s cold open
    tl.effects.append(AtmosphereEffect(type="progress_bar", extra={}))   # the user's
    settings = PresentationSettings(progress_bar=True)
    first = _apply_progress_bar(tl, settings, "tech", _topics())
    second = _apply_progress_bar(tl, settings, "tech", _topics())
    ours = [e for e in tl.effects if e.origin == PROGRESS_ORIGIN]
    assert len(ours) == 1 and len(tl.effects) == 2
    assert first == second == {"chapters": 3}
    assert [c["at_s"] for c in ours[0].extra["chapters"]] == [0.0, 25.0, 65.0]
    assert ours[0].intensity == 1.0 and ours[0].extra["fill_color"].startswith("#")

    _apply_progress_bar(tl, PresentationSettings(progress_bar=False), "tech", _topics())
    assert not any(e.origin == PROGRESS_ORIGIN for e in tl.effects)
    assert len(tl.effects) == 1, "the user's bar stays"


def test_genre_auto_rule():
    on = PresentationSettings(progress_bar=True)
    assert _progress_bar_wanted(on, "science_education")
    assert _progress_bar_wanted(on, "finance")
    assert not _progress_bar_wanted(on, "horror")
    assert not _progress_bar_wanted(on, "true_crime")
    assert not _progress_bar_wanted(on, "general")
    assert not _progress_bar_wanted(PresentationSettings(), "tech"), "off by default"
    assert _progress_bar_wanted(PresentationSettings(progress_bar=True,
                                                     progress_bar_style="on"), "horror")
    assert not _progress_bar_wanted(PresentationSettings(progress_bar=True,
                                                         progress_bar_style="off"), "tech")
    tl = Timeline(duration_frames=900)
    assert _apply_progress_bar(tl, on, "horror", _topics()) is None and not tl.effects


def test_placement_keeps_clear_of_captions():
    tl = Timeline(duration_frames=900)
    assert _progress_placement(tl) == {"position": "bottom", "margin_px": 0, "label": "name"}

    tl.items = [_caption(pos_y=-0.8)]                      # captions at the top
    assert _progress_placement(tl)["label"] == "name"

    tl.items = [_caption(pos_y=0.6, size=60, second=False)]  # room under them
    assert _progress_placement(tl) == {"position": "bottom", "margin_px": 0, "label": "none"}

    tl.items = [_caption(pos_y=0.9)]                       # they reach the edge
    placed = _progress_placement(tl)
    band_top = (0.9 + 1) / 2 * 1080 - (80 * 1.25 + 80 * 0.55 * 1.25 + 12) / 2
    assert placed["label"] == "name"
    assert 1080 - placed["margin_px"] <= band_top, "the bar sits above the band"


def test_placement_lifts_the_bar_off_a_letterbox():
    tl = Timeline(duration_frames=900, aspect_bars=2.333)
    assert _progress_placement(tl)["margin_px"] == 129


def test_report_field_is_optional():
    assert PresentationReport(project_id="p").progress_bar is None
    assert PresentationReport(project_id="p", progress_bar={"chapters": 4}).progress_bar == {"chapters": 4}


def test_sub_timeline_rebases_text_fx_windows_into_the_chunk():
    """A text-fx window at 100 s in a 2-chunk (0-60 s, 60-120 s) timeline."""
    tl = Timeline(duration_frames=3600)
    tl.effects.append(AtmosphereEffect(type="annotation",
                                       extra={"start_s": 100.0, "end_s": 101.5}))
    adj = TimelineItem(id="fx1", track="V3", kind="adjustment", origin="text_fx",
                       timeline_start_frame=3000, timeline_end_frame=3045,
                       atmosphere=[AtmosphereEffect(type="knockout", extra={
                           "text": "WOW", "start_s": 100.0, "end_s": 101.5,
                           "matte_offset": 12.0})])
    tl.items.append(adj)

    first = sub_timeline(tl, 0, 1800)
    assert first.effects[0].enabled is False, "outside chunk 1: never fires there"
    assert not first.items, "the adjustment clip is not in chunk 1"

    second = sub_timeline(tl, 1800, 3600)
    prog = second.effects[0]
    assert prog.enabled and prog.extra["start_s"] == pytest.approx(40.0)
    assert prog.extra["end_s"] == pytest.approx(41.5)
    fx = second.items[0].atmosphere[0]
    assert fx.enabled and fx.extra["start_s"] == pytest.approx(40.0)
    assert fx.extra["end_s"] == pytest.approx(41.5)
    assert fx.extra["matte_offset"] == 12.0, "a seek into the matte file, not programme time"
    assert second.items[0].timeline_start_frame == 1200
    assert tl.effects[0].extra["start_s"] == 100.0, "the original is untouched"
