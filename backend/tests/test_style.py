"""Tests for reference-video style matching.

Every reference here is generated with ffmpeg so the ground truth is known
exactly: a montage cut every 1.5s, a still pushed in by 30%, a track with a click
every 0.5s. That makes it possible to assert what the analyser *recovers* rather
than merely that it returns something.

The false-negative cases matter as much as the positives. An analyser that
reports a zoom on static footage, or captions on a textured background, produces
a profile that actively damages the edit it is applied to.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.style import color as color_mod
from backend.style import frames as frames_mod
from backend.style import motion as motion_mod
from backend.style import profile as profile_mod
from backend.style import rhythm as rhythm_mod
from backend.style import shots as shots_mod
from backend.style import text_regions
from backend.style.apply import ApplyOptions, apply_profile
from backend.timeline import build_timeline_from_transcript


def _ffmpeg(args, out: Path):
    from backend.config import FFMPEG_BIN
    result = subprocess.run([FFMPEG_BIN, "-y", "-v", "error", *args, str(out)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"ffmpeg could not build {out.name}")
    return out


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    """Reference clips with known, exact editing properties."""
    root = tmp_path_factory.mktemp("style_refs")

    # Eight visually distinct shots, 1.5s each -> 12s, 7 cuts, 34.5 cuts/min.
    colours = ["0x2E4A6B", "0xD9C7A0", "0x1A1A1A", "0x7B3F00",
               "0x0F5132", "0x4B0082", "0x8B0000", "0xE0E0E0"]
    shot_files = []
    for index, colour in enumerate(colours):
        shot = _ffmpeg(["-f", "lavfi",
                        "-i", f"color=c={colour}:s=320x180,"
                              f"drawbox=x={20 + index * 15}:y=30:w=120:h=90:color=white@0.85:t=fill",
                        "-t", "1.5", "-r", "25", "-pix_fmt", "yuv420p"],
                       root / f"shot{index}.mp4")
        shot_files.append(shot)
    listing = root / "shots.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in shot_files), encoding="utf-8")
    montage = _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"],
                      root / "montage.mp4")

    photo = _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1280x720", "-frames:v", "1"],
                    root / "photo.png")
    zoomed = _ffmpeg(["-loop", "1", "-i", str(photo), "-t", "6",
                      "-vf", "zoompan=z='1+0.30*on/149':x='(iw-iw/zoom)/2':"
                             "y='(ih-ih/zoom)/2':d=1:s=640x360:fps=25",
                      "-pix_fmt", "yuv420p"], root / "zoom.mp4")
    static = _ffmpeg(["-loop", "1", "-i", str(photo), "-t", "6",
                      "-vf", "scale=640:360", "-r", "25", "-pix_fmt", "yuv420p"],
                     root / "static.mp4")
    panned = _ffmpeg(["-loop", "1", "-i", str(photo), "-t", "6",
                      "-vf", "zoompan=z='1.4':x='(iw-iw/zoom)*(on/149)':"
                             "y='(ih-ih/zoom)/2':d=1:s=640x360:fps=25",
                      "-pix_fmt", "yuv420p"], root / "pan.mp4")

    gradient = ("gradients=s=640x360:c0=0x203040:c1=0x806040:d=6:speed=0.05")
    plain_bg = _ffmpeg(["-f", "lavfi", "-i", gradient, "-t", "6",
                        "-r", "25", "-pix_fmt", "yuv420p"], root / "plain_bg.mp4")

    # drawtext needs an explicit fontfile here: without one it falls back to
    # fontconfig, which has no default config on Windows and hard-crashes ffmpeg.
    from backend.render.effects import escape_filter_path
    from backend.utils.fonts import resolve_font_file

    font = resolve_font_file("Arial", bold=True)
    captioned = None
    if font:
        captioned = _ffmpeg(
            ["-f", "lavfi", "-i", gradient, "-t", "6",
             "-vf", f"drawtext=fontfile='{escape_filter_path(font)}':"
                    "text='THIS IS A CAPTION':fontsize=34:fontcolor=white:"
                    "borderw=3:bordercolor=black:x=(w-text_w)/2:y=h*0.82-text_h/2",
             "-r", "25", "-pix_fmt", "yuv420p"], root / "captioned.mp4")

    # A grade that genuinely moves the statistics. Contrast alone pivots around
    # mid-grey and leaves the mean where it was, which makes for a target the
    # fitter has nothing to correct towards.
    graded = _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=6",
                      "-vf", "eq=contrast=1.35:brightness=-0.18:saturation=0.45:"
                             "gamma_r=1.3:gamma_b=0.75,vignette",
                      "-pix_fmt", "yuv420p"], root / "graded.mp4")
    ungraded = _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=6",
                        "-pix_fmt", "yuv420p"], root / "ungraded.mp4")

    return {"root": root, "montage": montage, "zoom": zoomed, "static": static,
            "pan": panned, "captioned": captioned, "plain_bg": plain_bg,
            "graded": graded, "ungraded": ungraded}


def _gray(path, width=160, height=90, fps=6.0):
    return frames_mod.sample(str(path), width, height, fps=fps, gray=True)


# --- shots ----------------------------------------------------------------

def test_montage_cuts_are_found_exactly(refs):
    frames, fps = frames_mod.sample(str(refs["montage"]), 64, 36, fps=12.0, gray=False)
    analysis = shots_mod.detect_boundaries(frames, fps, 12.16)

    assert len(analysis.boundaries) == 7          # 8 shots -> 7 boundaries
    assert analysis.shot_lengths == pytest.approx([1.5] * 7 + [1.66], abs=0.12)
    assert analysis.cuts_per_minute == pytest.approx(34.5, abs=1.5)


def test_every_montage_boundary_is_a_hard_cut(refs):
    frames, fps = frames_mod.sample(str(refs["montage"]), 64, 36, fps=12.0, gray=False)
    mix = shots_mod.transition_mix(shots_mod.detect_boundaries(frames, fps, 12.16))
    assert mix["cut"]["count"] == 7
    assert mix["dissolve"]["count"] == 0 and mix["fade"]["count"] == 0


def test_static_footage_has_no_cuts(refs):
    frames, fps = frames_mod.sample(str(refs["static"]), 64, 36, fps=12.0, gray=False)
    analysis = shots_mod.detect_boundaries(frames, fps, 6.0)
    assert analysis.boundaries == []


def test_threshold_tracks_local_motion_not_a_global_constant():
    # Calm first half, busy second half. A single global threshold would either
    # miss the quiet cut or invent cuts throughout the busy stretch.
    calm = np.full(120, 2.0, dtype=np.float32)
    busy = np.full(120, 30.0, dtype=np.float32)
    diff = np.concatenate([calm, busy])
    thresholds = shots_mod.local_thresholds(diff, fps=12.0)
    assert thresholds[10] < 15.0        # a modest jump in the calm half counts
    assert thresholds[200] > 50.0       # the same jump in the busy half does not


def test_sustained_motion_is_not_mistaken_for_an_edit():
    # A long plateau is a whip pan or busy action, not a cut.
    diff = np.concatenate([np.full(40, 2.0), np.full(80, 40.0), np.full(40, 2.0)]).astype(np.float32)
    frames = np.zeros((diff.size + 1, 4, 4), dtype=np.uint8)
    analysis = shots_mod.detect_boundaries(frames, 12.0, 13.0)
    assert analysis.boundaries == []


# --- motion ---------------------------------------------------------------

def test_static_footage_reports_no_movement(refs):
    """The original integrating estimator invented 0.156 of zoom here."""
    frames, fps = _gray(refs["static"])
    result = motion_mod.analyse_motion(frames, fps, [(0.0, frames.shape[0] / fps)])
    assert result.mean_zoom_ratio == 0.0
    assert result.mean_pan_fraction == 0.0
    assert result.zoom_share == 0.0


def test_push_in_is_recovered(refs):
    frames, fps = _gray(refs["zoom"])
    result = motion_mod.analyse_motion(frames, fps, [(0.0, frames.shape[0] / fps)])
    assert result.zoom_share == 1.0
    assert result.mean_zoom_ratio == pytest.approx(0.30, abs=0.06)


def test_a_centred_zoom_is_not_also_reported_as_a_pan(refs):
    frames, fps = _gray(refs["zoom"])
    result = motion_mod.analyse_motion(frames, fps, [(0.0, frames.shape[0] / fps)])
    assert result.pan_share == 0.0


def test_pan_is_recovered_without_inventing_zoom(refs):
    frames, fps = _gray(refs["pan"])
    result = motion_mod.analyse_motion(frames, fps, [(0.0, frames.shape[0] / fps)])
    assert result.pan_share == 1.0
    assert result.mean_pan_fraction > 0.2
    assert result.zoom_share == 0.0


def test_untrackable_frames_report_nothing_rather_than_guessing():
    flat = np.full((12, 90, 160), 128, dtype=np.uint8)
    assert motion_mod.estimate_step(flat[0], flat[1]) is None


# --- colour ---------------------------------------------------------------

def _stats(path):
    frames, _ = frames_mod.sample(str(path), 128, 72, fps=0.5, gray=False, max_frames=300)
    return color_mod.measure(frames)


def test_fitting_a_grade_moves_footage_towards_the_reference(refs, tmp_path):
    from backend.render.effects import build_color_chain
    from backend.config import FFMPEG_BIN

    source = _stats(refs["ungraded"])
    target = _stats(refs["graded"])
    grade = color_mod.fit_grade(source, target)

    out = tmp_path / "graded_out.mp4"
    chain = ",".join(build_color_chain(grade))
    subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-i", str(refs["ungraded"]),
                    "-vf", chain, "-pix_fmt", "yuv420p", str(out)], check=True)
    result = _stats(out)

    def error(stats):
        return (abs(stats.mean - target.mean)
                + abs(stats.saturation - target.saturation)
                + abs(stats.std - target.std))

    assert error(result) < error(source) * 0.5, (
        f"grading moved error from {error(source):.3f} to {error(result):.3f}")
    assert abs(result.mean - target.mean) < abs(source.mean - target.mean)


def test_matching_a_look_to_itself_is_a_no_op(refs):
    stats = _stats(refs["graded"])
    grade = color_mod.fit_grade(stats, stats)
    assert grade.contrast == pytest.approx(1.0, abs=0.02)
    assert grade.brightness == pytest.approx(0.0, abs=0.02)
    assert grade.saturation == pytest.approx(1.0, abs=0.02)
    assert grade.temperature == pytest.approx(0.0, abs=0.02)


def test_temperature_is_not_inverted(refs, tmp_path):
    """A warm grade must render warmer. A sign slip here would be invisible in
    the numbers and obvious on screen."""
    from backend.render.effects import build_color_chain
    from backend.config import FFMPEG_BIN
    from backend.timeline.schema import ColorGrade

    def cast(path):
        stats = _stats(path)
        return stats.channel_mean["r"] - stats.channel_mean["b"]

    base = cast(refs["ungraded"])
    for temperature, warmer in ((1.0, True), (-1.0, False)):
        out = tmp_path / f"temp_{temperature}.mp4"
        chain = ",".join(build_color_chain(ColorGrade(temperature=temperature)))
        subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-i", str(refs["ungraded"]),
                        "-vf", chain, "-pix_fmt", "yuv420p", str(out)], check=True)
        assert (cast(out) > base) is warmer


# --- rhythm ---------------------------------------------------------------

def _click_track(bpm: float, seconds: float, sample_rate: int = 22050) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    period = 60.0 / bpm
    envelope = np.exp(-14.0 * (t % period))
    return (0.9 * np.sin(2 * np.pi * 180 * t) * envelope).astype(np.float32)


def test_tempo_is_recovered(refs):
    audio = _click_track(120.0, 16.0)
    envelope = rhythm_mod.onset_envelope(audio, 22050)
    bpm, confidence = rhythm_mod.estimate_tempo(envelope, 22050)
    assert bpm == pytest.approx(120.0, abs=6.0)
    assert confidence > 0.2


def test_tempo_does_not_halve_into_an_octave_error():
    """Autocorrelation peaks as hard at two beats as one; 120 must not read 60."""
    for true_bpm in (100.0, 120.0, 140.0):
        envelope = rhythm_mod.onset_envelope(_click_track(true_bpm, 16.0), 22050)
        bpm, _ = rhythm_mod.estimate_tempo(envelope, 22050)
        assert bpm == pytest.approx(true_bpm, rel=0.08), f"{true_bpm} read as {bpm}"


def test_cuts_on_the_beat_score_high():
    cuts = [i * 0.5 for i in range(1, 30)]          # every beat at 120 BPM
    assert rhythm_mod.beat_alignment(cuts, 120.0) > 0.9


def test_alignment_survives_a_small_tempo_error():
    """A grid-based measure collapses here; an interval-based one must not."""
    cuts = [i * 0.5 for i in range(1, 30)]
    assert rhythm_mod.beat_alignment(cuts, 122.0) > 0.8


def test_cuts_ignoring_the_beat_score_low():
    cuts = [0.31, 0.93, 1.12, 2.07, 2.61, 3.44, 4.02, 5.19, 6.33, 7.05]
    assert rhythm_mod.beat_alignment(cuts, 120.0) < 0.6


def test_silence_yields_no_rhythm():
    result = rhythm_mod.analyse_rhythm(np.zeros(0, dtype=np.float32), 22050, [1.0, 2.0])
    assert result.has_audio is False and result.bpm == 0.0


# --- captions -------------------------------------------------------------

def test_caption_band_position_is_recovered(refs):
    if not refs["captioned"]:
        pytest.skip("no system font available to render a caption reference")
    frames, _ = frames_mod.sample(str(refs["captioned"]), 192, 108, fps=6.0, gray=True)
    result = text_regions.analyse_captions(frames)
    assert result.present is True
    # Drawn at 82% of frame height -> +0.64 in -1..1 coordinates.
    assert result.pos_y == pytest.approx(0.64, abs=0.08)
    assert 0.03 < result.size_fraction < 0.18


def test_footage_without_captions_reports_none(refs):
    frames, _ = frames_mod.sample(str(refs["plain_bg"]), 192, 108, fps=6.0, gray=True)
    assert text_regions.analyse_captions(frames).present is False


def test_texture_and_graphics_are_not_mistaken_for_captions(refs):
    for key in ("static", "montage", "zoom", "pan"):
        frames, _ = frames_mod.sample(str(refs[key]), 192, 108, fps=6.0, gray=True)
        assert text_regions.analyse_captions(frames).present is False, key


# --- profile --------------------------------------------------------------

def test_full_profile_of_a_montage(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    assert built.pacing.shots == 8
    assert built.pacing.median_shot_seconds == pytest.approx(1.5, abs=0.15)
    assert built.pacing.cuts_per_minute == pytest.approx(34.5, abs=2.0)
    assert built.transitions["cut"]["count"] == 7
    assert built.look.description
    assert built.confidence["pacing"] == "high"


def test_profile_of_silent_footage_says_so(refs):
    built = profile_mod.analyze_video(str(refs["static"]), "Static")
    assert built.rhythm.cuts_to_music is False
    assert any("no audio" in note.lower() for note in built.notes)


def test_profiles_round_trip_through_disk(refs, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "STYLES_DIR", tmp_path / "styles")
    built = profile_mod.analyze_video(str(refs["static"]), "Round trip")
    profile_mod.save(built)

    loaded = profile_mod.load(built.id)
    assert loaded is not None and loaded.name == "Round trip"
    assert [p.id for p in profile_mod.list_profiles()] == [built.id]
    assert profile_mod.delete(built.id) is True
    assert profile_mod.load(built.id) is None


# --- applying -------------------------------------------------------------

def _program(source: str, segments: int = 40):
    # Gaps of 0.6s, comfortably over the 0.4s pause threshold, so each word
    # really does become its own programme segment for the mover to work on.
    words = [{"word": f"w{i}", "start": i * 1.4, "end": i * 1.4 + 0.8}
             for i in range(segments)]
    return build_timeline_from_transcript(source, segments * 1.4 + 1.0, words,
                                          width=640, height=360)


def test_applying_a_profile_grades_and_moves_the_programme(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    timeline = _program(str(refs["ungraded"]))

    report = apply_profile(timeline, built, source_path=str(refs["ungraded"]))

    assert report["look"]["applied"] is True
    assert timeline.master_color is not None
    assert report["motion"]["applied"] is True

    moved = [i for i in timeline.items if i.track == "V1" and i.transform]
    # The reference cuts 34.5 times a minute; a 36s programme wants ~21 moves.
    assert 12 <= len(moved) <= 26
    assert all(m.transform.scale_end is not None for m in moved)


def test_moves_alternate_direction(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    timeline = _program(str(refs["ungraded"]))
    apply_profile(timeline, built, source_path=str(refs["ungraded"]))

    moved = sorted([i for i in timeline.items if i.track == "V1" and i.transform],
                   key=lambda i: i.timeline_start_frame)
    directions = [m.transform.scale_end > m.transform.scale for m in moved]
    assert len(set(directions)) == 2, "every move pushes the same way"


def test_applying_twice_gives_the_same_edit(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")

    def moves_for():
        timeline = _program(str(refs["ungraded"]))
        apply_profile(timeline, built, source_path=str(refs["ungraded"]))
        return [(i.timeline_start_frame, i.transform.scale, i.transform.scale_end)
                for i in timeline.items if i.track == "V1" and i.transform]

    assert moves_for() == moves_for()


def test_deselected_parts_are_left_alone(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    timeline = _program(str(refs["ungraded"]))

    apply_profile(timeline, built, source_path=str(refs["ungraded"]),
                  options=ApplyOptions(look=False, motion=True,
                                       captions=False, transitions=False))
    assert timeline.master_color is None
    assert any(i.transform for i in timeline.items if i.track == "V1")


def test_a_styled_timeline_still_compiles(refs, tmp_path):
    from backend.render.compiler import FilterGraphCompiler

    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    timeline = _program(str(refs["ungraded"]))
    apply_profile(timeline, built, source_path=str(refs["ungraded"]))

    _inputs, graph, video_label, _audio = FilterGraphCompiler(
        timeline, assets_dir=tmp_path).compile()
    assert "zoompan=" in graph
    assert video_label == "[graded_v]"


def test_applying_a_profile_without_footage_reports_why(refs):
    built = profile_mod.analyze_video(str(refs["montage"]), "Montage")
    timeline = _program(str(refs["ungraded"]))
    report = apply_profile(timeline, built, source_path=None)
    assert report["look"]["applied"] is False
    assert "source" in report["look"]["reason"]


def test_silent_footage_still_renders(refs, tmp_path):
    """A source with no audio stream must not poison the filter graph.

    build_timeline_from_transcript creates A1 items regardless, and trimming
    [n:a] on a video with no audio makes ffmpeg reject the entire graph with
    "matches no streams".
    """
    from backend.render.compiler import FilterGraphCompiler

    timeline = _program(str(refs["static"]), segments=6)
    for source in timeline.sources.values():
        source.has_audio = False

    _inputs, graph, _video, audio_label = FilterGraphCompiler(
        timeline, assets_dir=tmp_path).compile()
    assert ":a]" not in graph
    assert audio_label == "[anull]"        # silent track instead of a broken map
