"""Voice delivery (VOICE_DELIVERY_PLAN.md, E1): the TTS CLI's style/mood resolution, the inline
mood cues, sentence pauses and the word/phrase spacing post-process -- all pure Python/numpy, so
these import `cli.py` the same way `audio/narration.py` does, with no model loaded."""
import numpy as np
import pytest

from audio import narration, silence_gate

cli = narration.cli_module()


# --------------------------------------------------------------------------- resolve_style / moods

def test_resolve_style_layers_defaults_mood_and_explicit():
    steady = cli.resolve_style(None)
    assert steady == cli.resolve_style({}) == cli.resolve_style({"mood": "steady"})
    assert steady["mood"] == "steady" and steady["pace"] == 0.95 and steady["word_gap_ms"] == 40

    dramatic = cli.resolve_style({"mood": "dramatic"})
    assert dramatic["exaggeration"] == 0.8 and dramatic["cfg_weight"] == 0.4
    assert dramatic["sentence_pause_s"] == 0.3   # halved: an addition to the model's own pause, not a replacement

    overridden = cli.resolve_style({"mood": "steady", "pace": 1.2, "word_gap_ms": 10})
    assert overridden["pace"] == 1.2 and overridden["word_gap_ms"] == 10
    assert overridden["exaggeration"] == 0.5          # untouched key keeps the mood's value


def test_resolve_style_unknown_mood_raises():
    with pytest.raises(ValueError):
        cli.resolve_style({"mood": "furious"})


def test_resolve_style_ignores_unknown_keys():
    resolved = cli.resolve_style({"mood": "steady", "nonsense": 99})
    assert "nonsense" not in resolved


def test_list_moods_has_all_seven_presets_in_table_order_with_style():
    moods = cli.list_moods()
    assert [m["id"] for m in moods] == \
        ["steady", "calm", "warm", "eerie", "dramatic", "intense", "horror_narrator"]
    for mood in moods:
        assert set(("id", "label", "description", "style")) <= set(mood)
        assert set(cli.STYLE_DEFAULTS) <= set(mood["style"])


def test_horror_narrator_mood_is_slow_grave_and_low_key():
    horror = cli.resolve_style({"mood": "horror_narrator"})
    assert horror["exaggeration"] == 0.3 and horror["cfg_weight"] == 0.25
    assert horror["temperature"] == 0.6 and horror["pace"] == 0.85
    assert horror["word_gap_ms"] == 70 and horror["phrase_pause_s"] == 0.3
    assert horror["sentence_pause_s"] == 0.6


# --------------------------------------------------------------------------- inline mood cues

def test_mood_cues_switch_and_are_never_spoken():
    plan = cli.plan_batches("Ek. (dramatic) Do. (normal) Teen.")
    texts = [p for p in plan if "text" in p]
    assert [p["mood"] for p in texts] == [None, "dramatic", None]
    spoken = " ".join(p["text"] for p in texts)
    assert "dramatic" not in spoken and "normal" not in spoken


def test_dramatic_and_soft_add_their_before_silence():
    plan = cli.plan_batches("Ek. (dramatic) Do.")
    assert plan[1] == {"silence": pytest.approx(0.3)}
    plan = cli.plan_batches("Ek. (soft) Do.")
    assert plan[1] == {"silence": pytest.approx(0.2)}


def test_intense_and_calm_add_no_before_silence():
    for cue in ("intense", "calm"):
        plan = cli.plan_batches(f"Ek. ({cue}) Do.")
        assert all("silence" not in p for p in plan)


def test_normal_ends_a_mood_cue_same_as_slow():
    plan = cli.plan_batches("(slow) (dramatic) Ek. (normal) Do.")
    texts = [p for p in plan if "text" in p]
    assert texts[0]["slow"] is True and texts[0]["mood"] == "dramatic"
    assert texts[1]["slow"] is False and texts[1]["mood"] is None


def test_batches_never_span_a_mood_change():
    long_sentence = "shabd " * 60 + "aur " + "lafz " * 60 + "."
    plan = cli.plan_batches(f"({'dramatic'}) {long_sentence}")
    texts = [p for p in plan if "text" in p]
    assert len(texts) > 1
    assert all(p["mood"] == "dramatic" for p in texts)


# --------------------------------------------------------------------------- plan_batches packing

def test_plan_batches_never_splits_per_sentence():
    # Defect 1: short batches hallucinate babble. plan_batches packs regardless of how the
    # caller means to use sentence_pause_s -- that pause is now inserted afterwards.
    plan = cli.plan_batches("Ek. Do. Teen.")
    assert len([p for p in plan if "text" in p]) == 1


def test_plan_batches_takes_no_sentence_pause_s_argument():
    import inspect
    assert "sentence_pause_s" not in inspect.signature(cli.plan_batches).parameters


# --------------------------------------------------------------------------- insert_sentence_pauses

def test_insert_sentence_pauses_between_consecutive_text_batches():
    plan = [{"text": "Ek.", "slow": False, "mood": None}, {"text": "Do.", "slow": False, "mood": None}]
    out = cli.insert_sentence_pauses(plan, 0.45)
    assert out == [plan[0], {"silence": pytest.approx(0.45)}, plan[1]]


def test_insert_sentence_pauses_not_before_the_first_batch():
    plan = [{"text": "Ek.", "slow": False, "mood": None}]
    assert cli.insert_sentence_pauses(plan, 0.45) == plan


def test_insert_sentence_pauses_leaves_a_leading_cue_silence_alone():
    plan = [{"silence": 0.3}, {"text": "Ek.", "slow": False, "mood": "dramatic"}]
    assert cli.insert_sentence_pauses(plan, 0.6) == plan


def test_insert_sentence_pauses_takes_the_max_with_an_explicit_cue_silence():
    plan = [{"text": "Ek.", "slow": False, "mood": None}, {"silence": 2.0},
           {"text": "Do.", "slow": False, "mood": None}]
    assert cli.insert_sentence_pauses(plan, 0.45)[1] == {"silence": 2.0}
    plan = [{"text": "Ek.", "slow": False, "mood": None}, {"silence": 0.3},     # beat is 0.3s < 1.0
           {"text": "Do.", "slow": False, "mood": None}]
    assert cli.insert_sentence_pauses(plan, 1.0)[1] == {"silence": 1.0}


def test_insert_sentence_pauses_zero_is_a_no_op():
    plan = [{"text": "Ek.", "slow": False, "mood": None}, {"text": "Do.", "slow": False, "mood": None}]
    assert cli.insert_sentence_pauses(plan, 0.0) == plan


# --------------------------------------------------------------------------- space_out

def _tone_with_gaps(sr, gap_s_list):
    """A 0.3 s tone, then (gap, 0.3 s tone) for each gap in `gap_s_list`; none touching an edge."""
    tone = np.full(int(round(0.3 * sr)), 0.5, dtype=np.float32)
    chunks = [tone]
    for gap_s in gap_s_list:
        chunks.append(np.zeros(int(round(gap_s * sr)), dtype=np.float32))
        chunks.append(tone)
    return np.concatenate(chunks)


def test_space_out_zero_and_zero_returns_the_input_unchanged():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [0.1])
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(audio, sr, 0, 0)
    assert out is audio and word_gaps == 0 and phrase_gaps == 0 and sentence_gaps == 0


def test_space_out_grows_a_word_gap_by_the_requested_amount():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [0.1])                 # 100 ms: inside the 60-150 ms word window
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(audio, sr, word_gap_ms=40, phrase_pause_s=0.0)
    assert word_gaps == 1 and phrase_gaps == 0 and sentence_gaps == 0
    assert out.size == pytest.approx(audio.size + int(round(0.04 * sr)), abs=2)


def test_space_out_grows_a_phrase_pause_by_the_requested_amount():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [0.3])                 # 300 ms: in the 150-350 ms comma window
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(audio, sr, word_gap_ms=0, phrase_pause_s=0.25)
    assert word_gaps == 0 and phrase_gaps == 1 and sentence_gaps == 0
    assert out.size == pytest.approx(audio.size + int(round(0.25 * sr)), abs=2)


def test_space_out_grows_a_sentence_pause_past_350ms_by_the_requested_amount():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [0.4])                 # 400 ms: past the 350 ms comma ceiling
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(
        audio, sr, word_gap_ms=0, phrase_pause_s=0, sentence_pause_s=0.3)
    assert word_gaps == 0 and phrase_gaps == 0 and sentence_gaps == 1
    assert out.size == pytest.approx(audio.size + int(round(0.3 * sr)), abs=2)


def test_space_out_caps_the_resulting_gap_at_1_6s():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [1.5])                 # already close to the cap
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(
        audio, sr, word_gap_ms=0, phrase_pause_s=0, sentence_pause_s=1.0)
    assert sentence_gaps == 1
    assert out.size == pytest.approx(audio.size + int(round(0.1 * sr)), abs=2)   # capped to 1.6s total

    audio2 = _tone_with_gaps(sr, [1.6])                # already at the cap: nothing more fits
    out2, word_gaps2, phrase_gaps2, sentence_gaps2 = cli.space_out(
        audio2, sr, word_gap_ms=0, phrase_pause_s=0, sentence_pause_s=1.0)
    assert out2.size == audio2.size and sentence_gaps2 == 0


def test_space_out_leaves_edge_silence_untouched():
    sr = cli.SAMPLE_RATE
    tone = np.full(int(round(0.3 * sr)), 0.5, dtype=np.float32)
    lead = np.zeros(int(round(0.1 * sr)), dtype=np.float32)
    audio = np.concatenate([lead, tone])               # the quiet run touches the start
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(audio, sr, word_gap_ms=40, phrase_pause_s=0.25)
    assert word_gaps == 0 and phrase_gaps == 0 and sentence_gaps == 0
    assert out.size == audio.size


def test_space_out_handles_several_gaps_independently():
    sr = cli.SAMPLE_RATE
    audio = _tone_with_gaps(sr, [0.1, 0.3, 0.08])
    out, word_gaps, phrase_gaps, sentence_gaps = cli.space_out(audio, sr, word_gap_ms=40, phrase_pause_s=0.25)
    assert word_gaps == 2 and phrase_gaps == 1 and sentence_gaps == 0
    expect = audio.size + 2 * int(round(0.04 * sr)) + int(round(0.25 * sr))
    assert out.size == pytest.approx(expect, abs=4)


# --------------------------------------------------------------------------- trim_edges

def test_trim_edges_cuts_leading_and_trailing_quiet_to_the_target():
    sr = cli.SAMPLE_RATE
    tone = np.full(int(round(0.5 * sr)), 0.5, dtype=np.float32)
    lead = np.zeros(int(round(0.8 * sr)), dtype=np.float32)
    trail = np.zeros(int(round(0.6 * sr)), dtype=np.float32)
    audio = np.concatenate([lead, tone, trail])
    out = cli.trim_edges(audio, sr, target_s=0.06)
    expect = tone.size + 2 * int(round(0.06 * sr))
    assert out.size == pytest.approx(expect, abs=int(round(0.01 * sr)) + 1)


def test_trim_edges_leaves_short_edges_alone():
    sr = cli.SAMPLE_RATE
    tone = np.full(int(round(0.3 * sr)), 0.5, dtype=np.float32)
    lead = np.zeros(int(round(0.02 * sr)), dtype=np.float32)   # already under the 60 ms target
    audio = np.concatenate([lead, tone])
    out = cli.trim_edges(audio, sr, target_s=0.06)
    assert out.size == audio.size


# --------------------------------------------------------------------------- spoken_char_count

def test_spoken_char_count_excludes_whitespace_and_punctuation():
    assert cli.spoken_char_count("Ek, Do.") == 4            # E k D o
    assert cli.spoken_char_count("रात के तीन बजे थे।") == cli.spoken_char_count("रात के तीन बजे थे।".replace(" ", "").replace("।", ""))


# --------------------------------------------------------------------------- trim_babble

def _silence_then_tone(sr, lead_s, quiet_s, tail_s):
    lead = np.full(int(round(lead_s * sr)), 0.5, dtype=np.float32)
    quiet = np.zeros(int(round(quiet_s * sr)), dtype=np.float32)
    tail = np.full(int(round(tail_s * sr)), 0.5, dtype=np.float32)
    return np.concatenate([lead, quiet, tail])


def test_trim_babble_leaves_audio_within_the_expected_length_alone():
    sr = cli.SAMPLE_RATE
    audio = np.full(sr, 0.5, dtype=np.float32)          # 1.0 s
    out, cut = cli.trim_babble(audio, sr, expected_s=1.0)
    assert cut is False and out is audio


def test_trim_babble_cuts_after_a_late_sentence_length_pause():
    sr = cli.SAMPLE_RATE
    # expected ~1.0s; real speech ends around 1.0s, a 0.5s pause, then ~1.0s of babble: total 2.5s
    # is over 1.5 * 1.0 + 1.0 = 2.5 -> just over, make it clearly over:
    audio = _silence_then_tone(sr, lead_s=1.0, quiet_s=0.5, tail_s=1.2)
    out, cut = cli.trim_babble(audio, sr, expected_s=1.0)
    assert cut is True
    assert out.size < audio.size
    assert out.size / sr == pytest.approx(1.5, abs=0.05)    # kept up to the end of the pause


def test_trim_babble_no_qualifying_quiet_run_leaves_audio_unchanged():
    sr = cli.SAMPLE_RATE
    audio = np.full(int(round(3.5 * sr)), 0.5, dtype=np.float32)   # solid tone, no pause at all
    out, cut = cli.trim_babble(audio, sr, expected_s=1.0)
    assert cut is False and out is audio


def test_trim_babble_ignores_an_early_pause_before_the_cut_fraction():
    sr = cli.SAMPLE_RATE
    # a 400 ms pause right at the start (well before 0.85 * expected_s), then a long tail: the
    # early pause must not be used as the cut point.
    audio = _silence_then_tone(sr, lead_s=0.05, quiet_s=0.4, tail_s=3.0)
    out, cut = cli.trim_babble(audio, sr, expected_s=1.0)
    assert cut is False and out is audio


# --------------------------------------------------------------------------- Whisper verification (E4)
# Pure scoring/decision functions on fake transcripts (`{"word", "start", "end"}`); no model loaded.

def _heard(words, step_s=0.3, dur_s=0.25):
    return [{"word": w, "start": round(i * step_s, 3), "end": round(i * step_s + dur_s, 3)}
            for i, w in enumerate(words)]


def test_verify_normalize_drops_punctuation_and_nukta_folds_chandrabindu():
    assert cli._verify_normalize("Raat,") == "raat"
    assert cli._verify_normalize("खामोशी।") == cli._verify_normalize("खामोशी")
    assert cli._verify_normalize("ख़ामोशी") == cli._verify_normalize("खामोशी")   # nukta dropped
    assert cli._verify_normalize("साँस") == cli._verify_normalize("सांस")        # chandrabindu -> anusvara


def test_verify_words_splits_and_drops_pure_punctuation():
    assert cli.verify_words("मैंने सोचा, इतनी रात को कौन होगा?") == \
        ["मैंने", "सोचा", "इतनी", "रात", "को", "कौन", "होगा"]


def test_words_fuzzy_equal_allows_whisper_misspellings():
    assert cli._words_fuzzy_equal("तीन", "तीम") is True          # one char off, short word
    assert cli._words_fuzzy_equal("खामोशी", "खामुशी") is True
    assert cli._words_fuzzy_equal("राम", "श्याम") is False
    assert cli._words_fuzzy_equal("", "राम") is False


def test_score_transcript_exact_match_is_full_coverage_no_extra():
    script = cli.verify_words("मैंने सोचा, इतनी रात को कौन होगा?")
    heard = _heard(["मैंने", "सोचा", "इतनी", "रात", "को", "कौन", "होगा"])
    score = cli.score_transcript(script, heard)
    assert score["coverage"] == 1.0 and score["extra"] == 0
    assert score["cut_at_s"] == heard[-1]["end"]
    assert cli.verify_decision(score) == "accept"


def test_score_transcript_tolerates_a_whisper_misspelling():
    script = cli.verify_words("आज रात तीन बजे थे")
    heard = _heard(["आज", "रात", "तीम", "बजे", "थे"])    # तीन heard as तीम
    score = cli.score_transcript(script, heard)
    assert score["coverage"] == 1.0
    assert cli.verify_decision(score) == "accept"


def test_score_transcript_babble_after_the_real_sentence_is_cut():
    script = cli.verify_words("मैंने सोचा, इतनी रात को कौन होगा?")
    heard = _heard(["मैंने", "सोचा", "इतनी", "रात", "को", "कौन", "होगा", "बकबक", "बकबक", "बकबक"])
    score = cli.score_transcript(script, heard)
    assert score["coverage"] == 1.0 and score["extra"] == 3
    assert score["cut_at_s"] == heard[6]["end"]            # the last matched word, "होगा"
    assert cli.verify_decision(score) == "cut"


def test_score_transcript_one_extra_word_is_still_accepted():
    script = cli.verify_words("Raat hui.")
    heard = _heard(["raat", "hui", "ok"])
    score = cli.score_transcript(script, heard)
    assert score["extra"] == 1
    assert cli.verify_decision(score) == "accept"


def test_score_transcript_dropped_sentence_is_low_coverage_and_regenerates():
    script = cli.verify_words("Pehli line. Doosri line. Teesri line poori lambi wali.")
    heard = _heard(["pehli", "line", "teesri", "line", "poori", "lambi", "wali"])  # "doosri line" missing
    score = cli.score_transcript(script, heard)
    assert score["coverage"] < cli.VERIFY_MIN_COVERAGE
    assert cli.verify_decision(score) == "regenerate"


def test_score_transcript_empty_transcript_has_zero_coverage_and_no_cut_point():
    script = cli.verify_words("Kuch bhi nahi suna gaya.")
    score = cli.score_transcript(script, [])
    assert score["coverage"] == 0.0 and score["cut_at_s"] is None
    assert cli.verify_decision(score) == "regenerate"


def test_score_transcript_out_of_order_words_do_not_count_as_matched():
    # Whisper hearing the right words in the wrong order is not the script said in order.
    script = cli.verify_words("Ek do teen.")
    heard = _heard(["teen", "do", "ek"])
    score = cli.score_transcript(script, heard)
    assert score["coverage"] < 1.0


def test_verify_decision_boundaries():
    assert cli.verify_decision({"coverage": cli.VERIFY_MIN_COVERAGE, "extra": 0, "cut_at_s": 1.0}) == "accept"
    assert cli.verify_decision(
        {"coverage": cli.VERIFY_MIN_COVERAGE - 0.01, "extra": 0, "cut_at_s": 1.0}) == "regenerate"
    assert cli.verify_decision(
        {"coverage": 1.0, "extra": cli.VERIFY_MAX_EXTRA_WORDS + 1, "cut_at_s": 1.0}) == "cut"
    assert cli.verify_decision(
        {"coverage": 1.0, "extra": cli.VERIFY_MAX_EXTRA_WORDS + 1, "cut_at_s": None}) == "regenerate"


def test_cut_at_time_truncates_at_the_given_second():
    sr = cli.SAMPLE_RATE
    audio = np.arange(sr * 2, dtype=np.float32)   # 2.0 s
    out = cli.cut_at_time(audio, sr, 0.5)
    assert out.size == int(round(0.5 * sr))
    assert np.array_equal(out, audio[:out.size])


def test_fade_edges_starts_and_ends_on_silence_without_touching_the_middle():
    sr = cli.SAMPLE_RATE
    audio = np.ones(sr, dtype=np.float32)
    out = cli.fade_edges(audio, sr)
    assert out[0] == 0.0 and out[-1] == 0.0
    assert out[sr // 2] == 1.0
    assert out.size == audio.size and audio[0] == 1.0   # the input is not modified in place


def test_fade_edges_caps_the_fades_on_a_short_batch():
    sr = cli.SAMPLE_RATE
    audio = np.ones(int(0.1 * sr), dtype=np.float32)    # shorter than the 0.12 s fade-out
    out = cli.fade_edges(audio, sr)
    assert out[audio.size // 2] == 1.0


def test_cut_at_time_clamps_to_the_audios_own_length():
    sr = cli.SAMPLE_RATE
    audio = np.ones(sr, dtype=np.float32)
    assert cli.cut_at_time(audio, sr, 5.0).size == audio.size
    assert cli.cut_at_time(audio, sr, -1.0).size == 0


def test_align_words_never_matches_the_same_word_twice():
    alignment = cli._align_words(["ek", "do", "ek"], ["ek", "ek"])
    matched = [i for i in alignment if i is not None]
    assert len(matched) == len(set(matched))


# --------------------------------------------------------------------------- silence_gate (F8)
# seed-vc fills a true silence with a soft floor of its own target voice, never real quiet:
# `gate()` re-imposes the Chatterbox source's own silences on the converted audio.

def _speech_silence_speech(sr, speech_s=0.5, quiet_s=0.3):
    speech = np.full(int(round(speech_s * sr)), 0.5, dtype=np.float32)
    quiet = np.zeros(int(round(quiet_s * sr)), dtype=np.float32)
    return np.concatenate([speech, quiet, speech])


def test_gate_zeroes_the_converted_silence_and_keeps_speech():
    sr = 16000
    source = _speech_silence_speech(sr)
    converted = np.full(source.size, 0.3, dtype=np.float32)   # seed-vc's floor: never truly silent
    out = silence_gate.gate(source, sr, converted, sr, length_adjust=1.0)
    speech_n, quiet_n = int(round(0.5 * sr)), int(round(0.3 * sr))
    assert out[speech_n // 2] == pytest.approx(0.3)                          # inside the first speech
    assert out[speech_n + quiet_n // 2] == 0.0                               # deep inside the silence
    assert out[speech_n + quiet_n + speech_n // 2] == pytest.approx(0.3)     # inside the second speech


def test_gate_scales_the_zeroed_span_by_length_adjust():
    sr = 16000
    source = _speech_silence_speech(sr)
    la = 2.0
    converted = np.full(int(round(source.size * la)), 0.3, dtype=np.float32)
    out = silence_gate.gate(source, sr, converted, sr, length_adjust=la)
    speech_n, quiet_n = int(round(0.5 * sr)), int(round(0.3 * sr))
    mid_quiet = int(round((speech_n + quiet_n / 2) * la))
    tail_speech = int(round((speech_n + quiet_n + speech_n / 2) * la))
    assert out[mid_quiet] == 0.0
    assert out[tail_speech] == pytest.approx(0.3)


def test_gate_leaves_a_short_quiet_run_alone():
    sr = 16000
    speech = np.full(int(round(0.5 * sr)), 0.5, dtype=np.float32)
    short_quiet = np.zeros(int(round(0.05 * sr)), dtype=np.float32)   # under MIN_RUN_S (0.12 s)
    source = np.concatenate([speech, short_quiet, speech])
    converted = np.full(source.size, 0.3, dtype=np.float32)
    out = silence_gate.gate(source, sr, converted, sr, length_adjust=1.0)
    assert np.all(out == pytest.approx(0.3))


def test_gate_never_modifies_converted_in_place():
    sr = 16000
    source = _speech_silence_speech(sr)
    converted = np.full(source.size, 0.3, dtype=np.float32)
    original = converted.copy()
    silence_gate.gate(source, sr, converted, sr, length_adjust=1.0)
    assert np.array_equal(converted, original)
