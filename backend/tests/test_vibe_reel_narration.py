"""The narrated vibe reel (REELS_DESK_PLAN.md VR1b): request validation,
coverage maths (weights -> durations), caption alignment/phrasing, the
narration-aware ASS style and audio chains, and the blackdetect gate."""

import subprocess
from pathlib import Path
from typing import List

import pytest

from presentation import caption_align as ca
from presentation import vibe_reel as vr
from presentation.vibe_reel import Shot, VibeReelRequest


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, capture_output=True, timeout=30, creationflags=0)


def _make_silence(path: Path, seconds: float = 2.0) -> Path:
    _run([vr.FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", f"{seconds:.3f}", str(path)])
    return path


def _base_request(tmp_path: Path, **overrides) -> dict:
    body = dict(
        out_dir=str(tmp_path),
        name="reel",
        art_style="real",
        world="an old haveli at night",
        shots=[{"prompt": "a dark lane", "seconds": 6.0, "kind": "still"}] * 4,
    )
    body.update(overrides)
    return body


# --- request validation -----------------------------------------------------

def test_narration_and_lines_together_is_rejected(tmp_path):
    wav = _make_silence(tmp_path / "n.wav")
    with pytest.raises(Exception):
        VibeReelRequest(**_base_request(
            tmp_path, narration={"path": str(wav), "text": "kahani"},
            lines=[{"text": "x", "start_s": 0.0, "end_s": 1.0}]))


def test_narration_missing_file_is_rejected(tmp_path):
    with pytest.raises(Exception):
        VibeReelRequest(**_base_request(
            tmp_path, narration={"path": str(tmp_path / "missing.wav"), "text": "kahani"}))


def test_narration_without_lines_or_music_is_allowed(tmp_path):
    wav = _make_silence(tmp_path / "n.wav")
    req = VibeReelRequest(**_base_request(tmp_path, narration={"path": str(wav), "text": "kahani"}))
    assert req.music is None
    assert req.lines == []


def test_lines_reel_without_narration_still_needs_a_line_card(tmp_path):
    with pytest.raises(Exception):
        VibeReelRequest(**_base_request(tmp_path, lines=[], music={"brief": "calm"}))


def test_lines_reel_without_narration_is_accepted_with_no_music(tmp_path):
    """2026-10-24, VR8: a lines reel's `music` may be null -- the creator
    adds his own song in Instagram over the ambience -- while a narrated
    reel is unchanged by this (it already allowed going without one)."""
    req = VibeReelRequest(**_base_request(
        tmp_path, lines=[{"text": "x", "start_s": 0.0, "end_s": 1.0}]))
    assert req.music is None


def test_captions_off_allows_narration_with_no_lines(tmp_path):
    wav = _make_silence(tmp_path / "n.wav")
    req = VibeReelRequest(**_base_request(
        tmp_path, narration={"path": str(wav), "text": "kahani"}, captions={"style": "off"}))
    assert req.captions.style == "off"


@pytest.mark.parametrize("field,value", [("push", 0.2), ("push", 0.01),
                                         ("dissolve_s", 2.0), ("dissolve_s", 0.1)])
def test_push_and_dissolve_out_of_range_are_rejected(tmp_path, field, value):
    with pytest.raises(Exception):
        VibeReelRequest(**_base_request(
            tmp_path, lines=[{"text": "x", "start_s": 0.0, "end_s": 1.0}],
            music={"brief": "calm"}, **{field: value}))


# --- coverage maths -----------------------------------------------------------

def _shots(video_n: int, still_n: int, video_w: float = 6.0, still_w: float = 6.0) -> List[Shot]:
    return ([Shot(prompt="v", seconds=video_w, kind="video")] * video_n
           + [Shot(prompt="s", seconds=still_w, kind="still")] * still_n)


def test_video_weight_becomes_its_duration_unchanged():
    shots = _shots(2, 4, video_w=6.0, still_w=6.0)
    durations = vr.compute_shot_durations(shots, 26.0, 0.4)
    assert durations[0] == 6.0 and durations[1] == 6.0


def test_stills_share_the_remaining_time_by_weight():
    shots = _shots(2, 4, video_w=6.0, still_w=6.0)
    durations = vr.compute_shot_durations(shots, 26.0, 0.4)
    # target_sum = 26 + 5*0.4 = 28; remaining after 2*6s video = 16; split
    # evenly over 4 equally-weighted stills = 4.0s each.
    assert durations[2:] == pytest.approx([4.0, 4.0, 4.0, 4.0])
    assert sum(durations) == pytest.approx(28.0)


def test_uneven_weights_split_the_remaining_time_proportionally():
    shots = [Shot(prompt="v", seconds=6.0, kind="video"),
            Shot(prompt="s1", seconds=4.0, kind="still"),
            Shot(prompt="s2", seconds=8.0, kind="still")]
    durations = vr.compute_shot_durations(shots, 15.0, 0.4)
    target_sum = 15.0 + 2 * 0.4
    remaining = target_sum - 6.0
    assert durations[1] == pytest.approx(remaining * (4.0 / 12.0))
    assert durations[2] == pytest.approx(remaining * (8.0 / 12.0))


def test_all_video_shots_need_no_stills_to_share_anything():
    shots = _shots(4, 0, video_w=6.0)
    durations = vr.compute_shot_durations(shots, 20.0, 0.4)
    assert durations == [6.0, 6.0, 6.0, 6.0]


def test_impossible_coverage_fails_with_a_shot_count_range():
    # Two video shots alone (12s) already exceed what a short narration needs.
    shots = _shots(2, 2, video_w=6.0, still_w=6.0)
    with pytest.raises(ValueError, match=r"need \d+-\d+ shots for"):
        vr.compute_shot_durations(shots, 4.0, 0.4)


def test_too_many_stills_for_too_little_remaining_time_fails():
    # 6 stills must share only 10s -- below STILL_SECONDS_MIN_NARR (2.5s) each.
    shots = _shots(0, 6, still_w=6.0)
    with pytest.raises(ValueError, match=r"need \d+-\d+ shots for"):
        vr.compute_shot_durations(shots, 10.0, 0.4)


# --- caption alignment: given spelling, interpolation, phrase breaks ------------

def test_given_spelling_is_shown_not_the_heard_spelling():
    given = "सुबह जल्दी घर से निकल गया"
    heard = [{"word_native": "सुबह", "start": 0.0, "end": 0.3},
            {"word_native": "jaldi", "start": 0.3, "end": 0.6},   # misheard/garbled
            {"word_native": "घर", "start": 0.6, "end": 0.8},
            {"word_native": "से", "start": 0.8, "end": 1.0},
            {"word_native": "निकल", "start": 1.0, "end": 1.3},
            {"word_native": "गया", "start": 1.3, "end": 1.6}]
    phrases = ca.build_caption_phrases(given, heard, 1.6)
    full_text = " ".join(p["text"] for p in phrases)
    assert "जल्दी" in full_text          # the GIVEN word, not "jaldi"
    assert "jaldi" not in full_text


def test_unmatched_word_is_interpolated_between_its_matched_neighbours():
    given_words = ["eka", "do", "teen", "char"]
    pairs = [(0, 0), (1, None), (2, None), (3, 1)]
    heard = [{"start": 0.0, "end": 1.0}, {"start": 4.0, "end": 5.0}]
    timed = ca.interpolate_times(given_words, pairs, heard, 5.0)
    assert timed[0]["start"] == 0.0 and timed[0]["end"] == 1.0
    assert timed[3]["start"] == 4.0 and timed[3]["end"] == 5.0
    # "do" and "teen" fill the 1.0..4.0 gap evenly, in order.
    assert timed[1]["start"] == pytest.approx(2.0)
    assert timed[2]["start"] == pytest.approx(3.0)
    assert timed[1]["end"] <= timed[2]["start"] + 1e-6


def test_phrase_breaks_after_punctuation():
    words = [{"text": "यह", "start": 0.0, "end": 0.2},
            {"text": "सच", "start": 0.2, "end": 0.4},
            {"text": "है।", "start": 0.4, "end": 0.6},
            {"text": "फिर", "start": 0.6, "end": 0.8},
            {"text": "क्या", "start": 0.8, "end": 1.0}]
    phrases = ca.break_phrases(words)
    assert phrases[0]["text"] == "यह सच है।"
    assert phrases[1]["text"] == "फिर क्या"


def test_phrase_breaks_at_a_long_pause():
    words = [{"text": "ek", "start": 0.0, "end": 0.3},
            {"text": "do", "start": 0.3, "end": 0.6},
            {"text": "teen", "start": 1.2, "end": 1.5}]   # 0.6s pause before this
    phrases = ca.break_phrases(words)
    assert len(phrases) == 2
    assert phrases[0]["text"] == "ek do"
    assert phrases[1]["text"] == "teen"


def test_phrase_breaks_at_four_words():
    words = [{"text": str(i), "start": i * 0.2, "end": i * 0.2 + 0.1} for i in range(5)]
    phrases = ca.break_phrases(words)
    assert len(phrases[0]["text"].split()) <= ca.PHRASE_MAX_WORDS


def test_phrase_breaks_at_max_seconds():
    # Four short words spread past 1.8s even though none of them pause or end
    # on punctuation.
    words = [{"text": "a", "start": 0.0, "end": 0.2},
            {"text": "b", "start": 0.3, "end": 0.5},
            {"text": "c", "start": 1.0, "end": 1.2},
            {"text": "d", "start": 1.9, "end": 2.1}]
    phrases = ca.break_phrases(words)
    assert all(p["end"] - p["start"] <= ca.PHRASE_MAX_S + 1e-6 for p in phrases)


def test_phrase_ends_are_padded_but_never_overlap_the_next_start():
    phrases = [{"text": "a", "start": 0.0, "end": 1.0}, {"text": "b", "start": 1.05, "end": 2.0}]
    padded = ca.pad_phrase_ends(phrases)
    assert padded[0]["end"] <= padded[1]["start"]
    assert padded[-1]["end"] == pytest.approx(2.0 + ca.CAPTION_END_PAD_S)


# --- the ASS caption style ---------------------------------------------------

def test_caption_ass_is_bold_outlined_white_and_at_70_percent_height():
    phrases = [{"text": "ghar mein andhera tha", "start": 0.0, "end": 1.5}]
    ass_text = vr.build_caption_ass(phrases)
    assert "ghar mein andhera tha" in ass_text
    assert ",-1,0," in ass_text               # Bold=-1 (True), Italic=0
    assert ",4," in ass_text                  # a ~4px outline
    assert "&H00FFFFFF" in ass_text            # opaque white primary colour
    # pos_y 0.4 on the -1..1 scale -> 0.7 * 1920 = 1344px down the frame.
    assert "\\pos(540,1344)" in ass_text


def test_caption_ass_uses_devanagari_font_for_devanagari_text():
    phrases = [{"text": "घर में अंधेरा था", "start": 0.0, "end": 1.5}]
    ass_text = vr.build_caption_ass(phrases)
    assert "Kokila" in ass_text


def test_long_caption_phrase_shrinks_to_fit_88pct_width():
    short = vr._caption_font_size("ek", vr.ROMAN_FONT)
    long = vr._caption_font_size("ek lambi panktiyon wali caption jo poori width le legi", vr.ROMAN_FONT)
    assert long < short
    assert long >= vr.CAPTION_MIN_FONT_PX


# --- narration audio chain ---------------------------------------------------

def test_narration_chain_is_full_level_music_ducked_ambience_at_own_gain():
    graph = vr.build_narration_audio_filter(0, 1, [(2, -26.0)], 20.0, 0.5, with_music=True)
    assert "[0:a]adelay=delays=500:all=1" in graph         # the 0.5s lead-in
    assert f"volume={vr.NARRATION_MUSIC_GAIN_DB:.1f}dB" in graph
    assert "volume=-26.0dB" in graph
    assert f"loudnorm=I={vr.LOUDNESS_LUFS:.1f}:TP={vr.TRUE_PEAK_DBTP:.1f}" in graph
    # the bed (music+ambience) is faded and mixed with the narration separately
    assert "[narr][bedf]amix=inputs=2" in graph


def test_narration_nomusic_chain_has_narration_and_ambience_but_no_music():
    graph = vr.build_narration_audio_filter(0, None, [(1, -26.0)], 20.0, 0.5, with_music=False)
    assert f"volume={vr.NARRATION_MUSIC_GAIN_DB:.1f}dB" not in graph
    assert "volume=-26.0dB" in graph
    assert "[0:a]adelay=delays=500:all=1" in graph


def test_narration_chain_with_no_bed_at_all_is_still_full_level_narration():
    graph = vr.build_narration_audio_filter(0, None, [], 20.0, 0.5, with_music=False)
    assert "[mixed]" not in graph
    assert "[narr]loudnorm" in graph


# --- the blackdetect gate -------------------------------------------------------

def _blackdetect_line(start: float, end: float) -> str:
    return (f"[blackdetect @ 0x0] black_start:{start:.3f} black_end:{end:.3f} "
           f"black_duration:{end - start:.3f}\n")


def test_parse_blackdetect_ignores_the_opening_fade_in():
    stderr = _blackdetect_line(0.0, vr.FADE_IN_S - 0.1)
    assert vr.parse_blackdetect(stderr, total_duration=30.0) == []


def test_parse_blackdetect_ignores_the_closing_fade_out():
    stderr = _blackdetect_line(30.0 - vr.FADE_OUT_S + 0.1, 30.0)
    assert vr.parse_blackdetect(stderr, total_duration=30.0) == []


def test_parse_blackdetect_flags_a_hit_in_the_middle():
    stderr = _blackdetect_line(10.0, 15.2)
    hits = vr.parse_blackdetect(stderr, total_duration=30.0)
    assert len(hits) == 1
    assert hits[0]["start"] == pytest.approx(10.0)
    assert hits[0]["end"] == pytest.approx(15.2)


def test_check_no_black_frames_raises_on_a_mocked_black_span(tmp_path, monkeypatch):
    fake_result = type("R", (), {"stderr": _blackdetect_line(10.0, 15.2), "returncode": 0})()
    monkeypatch.setattr(vr.subprocess, "run", lambda *a, **k: fake_result)
    with pytest.raises(vr.VibeReelError, match=r"10\.00-15\.20s"):
        vr.check_no_black_frames(tmp_path / "video.mp4", total_duration=30.0)


def test_check_no_black_frames_passes_on_a_mocked_clean_output(tmp_path, monkeypatch):
    fake_result = type("R", (), {"stderr": "", "returncode": 0})()
    monkeypatch.setattr(vr.subprocess, "run", lambda *a, **k: fake_result)
    vr.check_no_black_frames(tmp_path / "video.mp4", total_duration=30.0)   # no raise


# --- captions built, transliterated and the LLM freed before any picture (VR7) --

@pytest.mark.asyncio
async def test_narrated_reel_transliterates_captions_and_ejects_the_llm_before_shots(
        tmp_path, monkeypatch):
    """The whole point of VR7's reordering: the narration is transcribed and
    its captions transliterated to Hinglish, the LLM is ejected, and only
    THEN does the first picture get generated."""
    # Long enough that the base request's 4 still shots (weight 6.0 each)
    # cover it within STILL_SECONDS_MIN_NARR-MAX_NARR -- short enough and
    # the coverage maths (unrelated to this test) fails before any of this
    # code even runs.
    wav = _make_silence(tmp_path / "n.wav", seconds=8.0)
    order: List[str] = []

    async def fake_transcribe(path, language="hi"):
        return [], language

    import asr.faster_whisper_engine as whisper_mod
    monkeypatch.setattr(whisper_mod.whisper_engine, "transcribe_words_async", fake_transcribe)
    monkeypatch.setattr(vr.caption_align, "build_caption_phrases",
                        lambda text, heard, total: [{"text": "ghar mein andhera tha",
                                                     "start": 0.0, "end": 1.0}])

    async def fake_translit(lines):
        order.append("translit")
        return [f"hinglish {t}" for t in lines], True

    monkeypatch.setattr(vr, "transliterate_to_hinglish", fake_translit)

    async def fake_eject():
        order.append("eject")
        return True

    monkeypatch.setattr(vr.gpu_handover, "eject_lm_studio", fake_eject)
    monkeypatch.setattr(vr.gpu_handover, "release_in_process_models", lambda: None)

    async def fake_generate_shot(beat, project_dir, settings, seed_base, character=None,
                                 character_cache_dir=None, workflow_files=None,
                                 gen_size_override=None, start_image_override=None):
        order.append("shot")
        return None, 0.0, 0.0, "stopped for the test", ""

    monkeypatch.setattr(vr, "_generate_shot", fake_generate_shot)

    req = VibeReelRequest(**_base_request(
        tmp_path, narration={"path": str(wav), "text": "ghar mein andhera tha"}))
    with pytest.raises(vr.VibeReelError):
        await vr.render_vibe_reel(req)

    assert order[:2] == ["translit", "eject"]
    assert all(step == "shot" for step in order[2:]) and order[2:]
