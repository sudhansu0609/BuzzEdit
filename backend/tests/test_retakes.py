"""Tests for retake / false-start collapsing.

The case that drove this, from a real recording:

    dosto kya ap · dosto kya · dosto kya a · dosto · dosto kya apko pata hai india…

Four abandoned run-ups before the sentence lands. The old detector looked for
exact adjacent n-gram repeats and cut one word out of nine, because a restart is
a *prefix* of the good take at varying lengths with clipped words, which never
matches exactly.

Half of these tests are negative. The dangerous failure is not missing a fumble —
it is deleting a phrase the speaker repeated on purpose.
"""

import pytest

from backend.asr.disfluency import analyze_disfluencies
from backend.asr.retakes import apply_retakes, find_retakes, token_similarity


def _words(sentence: str, spacing: float = 0.35, pauses: dict = None):
    """Evenly spoken words, optionally with a pause before given indices."""
    tokens = sentence.split()
    pauses = pauses or {}
    words = []
    clock = 0.0
    for index, token in enumerate(tokens):
        clock += pauses.get(index, 0.0)
        words.append({"word": token, "start": round(clock, 3),
                      "end": round(clock + spacing * 0.85, 3), "probability": 0.9})
        clock += spacing
    return words


def _kept(sentence: str, **kwargs) -> str:
    planned = apply_retakes(_words(sentence, **kwargs))
    return " ".join(w["word"] for w in planned if not w.get("disfluency"))


def _kept_full(sentence: str, **kwargs) -> str:
    """Through the whole deterministic detector, not just the retake pass."""
    planned = analyze_disfluencies(_words(sentence, **kwargs))
    return " ".join(w["word"] for w in planned if not w.get("disfluency"))


# --- the real case --------------------------------------------------------

def test_four_abandoned_attempts_collapse_to_the_final_take():
    sentence = ("dosto kya ap dosto kya dosto kya a dosto dosto kya apko pata hai "
                "india me ek aisi jagah hai jo amazing hai")
    assert _kept(sentence) == (
        "dosto kya apko pata hai india me ek aisi jagah hai jo amazing hai")


def test_the_same_case_through_the_full_detector():
    sentence = ("dosto kya ap dosto kya dosto kya a dosto dosto kya apko pata hai "
                "india me ek aisi jagah hai jo amazing hai")
    assert _kept_full(sentence) == (
        "dosto kya apko pata hai india me ek aisi jagah hai jo amazing hai")


def test_the_whole_pile_up_is_reported_as_cut_material():
    """However the run-ups are grouped, everything before the good take goes."""
    sentence = "dosto kya ap dosto kya dosto kya a dosto dosto kya apko pata hai"
    retakes = find_retakes(_words(sentence))
    assert retakes
    covered = {i for r in retakes for i in range(r.start, r.end)}
    assert covered == set(range(0, 9))          # the four abandoned attempts
    # Any retakes that are reported chain end-to-start, never overlapping.
    for earlier, later in zip(retakes, retakes[1:]):
        assert earlier.end <= later.start


def test_a_speaker_who_gets_most_of_the_way_in_before_restarting():
    """Real restarts are not always quick — this one ran 16 words first.

    An earlier version capped the abandoned tail at 3 words and missed a
    seven-word verbatim match because of it.
    """
    sentence = ("dosto aapake saath kabhee aisaa huaa hai ki aap kisee aise kamare "
                "men gayaa hai jisamen dosto aapake saath kabhee aisaa huaa hai ka vo")
    assert _kept(sentence) == "dosto aapake saath kabhee aisaa huaa hai ka vo"


# --- other shapes of restart ---------------------------------------------

def test_single_restart():
    assert _kept("main aaj aapko main aaj aapko ek kahani sunata hun") == \
        "main aaj aapko ek kahani sunata hun"


def test_english_restart():
    assert _kept("so today I want to so today I want to talk about cameras") == \
        "so today I want to talk about cameras"


def test_word_cut_off_mid_syllable():
    """'prob' is the start of 'probably' — the commonest shape of a break-off."""
    assert _kept("we should prob we should probably go now") == "we should probably go now"


def test_restart_after_a_hesitation_pause():
    kept = _kept("chaliye aaj hum chaliye aaj hum baat karte hain",
                 pauses={3: 0.5})
    assert kept == "chaliye aaj hum baat karte hain"


def test_the_last_take_is_the_one_kept():
    """It is the later copy that survives, so the kept audio is the good take."""
    planned = apply_retakes(_words("main aaj aapko main aaj aapko kahani sunata hun"))
    survivors = [w for w in planned if not w.get("disfluency")]
    assert [w["word"] for w in survivors] == ["main", "aaj", "aapko", "kahani", "sunata", "hun"]
    assert survivors[0]["start"] > planned[0]["start"]


def test_a_bare_two_word_repeat_is_not_cut_on_its_own():
    """Without a chain or a hesitation this is as likely to be deliberate.

    "ek do ek do" is counting, not a fumble. Two matching words is the level at
    which structure stops being evidence, so it is offered to the LLM instead.
    """
    planned = apply_retakes(_words("ek do ek do teen char"))
    assert not any(w.get("disfluency") for w in planned)
    assert any(w.get("candidate") for w in planned)


def test_an_exact_double_after_a_hesitation_is_cut():
    """Said it, stopped, said exactly the same thing again — that is a restart.

    The same words without the hesitation stay a candidate (see the test above):
    it is the pause that separates a restart from a counting rhythm.
    """
    planned = apply_retakes(_words("ek do ek do teen char", pauses={2: 0.5}))
    survivors = [w["word"] for w in planned if not w.get("disfluency")]
    assert survivors == ["ek", "do", "teen", "char"]


def test_a_common_three_word_phrase_is_not_cut_on_structure_alone():
    """This once deleted a person's name from the video.

    "unakaa naam thaa" ("his name was") recurs naturally; the attempt that
    followed it carried the name and the retry did not.
    """
    planned = apply_retakes(_words("unakaa naam thaa thommel gilovich unakaa naam thaa ek experiment"))
    assert not any(w.get("disfluency") for w in planned)
    assert any(w.get("candidate") for w in planned)


def test_a_long_verbatim_match_is_cut_without_asking():
    """Nobody repeats ten words by accident."""
    sentence = ("aapako lagataa hogaa ki sab kee najare aap par hai aur "
                "aapako lagataa hogaa ki sab kee najare aap par hai bilkul")
    assert _kept(sentence) == "aapako lagataa hogaa ki sab kee najare aap par hai bilkul"


# --- what must NOT be cut -------------------------------------------------

def test_a_clean_sentence_is_untouched():
    sentence = "yeh jagah bahut hi khoobsurat hai aur log yahan aate hain"
    assert _kept(sentence) == sentence


def test_a_common_word_recurring_is_not_a_restart():
    """Hindi sentences are full of 'hai … hai'; treating that as a repeat would
    eat everything between the two."""
    sentence = "jagah hai jo amazing hai aur sabse alag hai"
    assert _kept(sentence) == sentence


def test_a_set_phrase_is_not_cut_outright():
    """'it is what it is' is a rough copy by every structural measure."""
    sentence = "it is what it is and that is that"
    assert _kept(sentence) == sentence


def test_rhetorical_repetition_is_not_cut_outright():
    sentence = "this is big this is huge for everyone"
    assert _kept(sentence) == sentence


def test_ambiguous_repetition_is_offered_to_the_llm():
    """Not cut, but not ignored either — the model decides in context."""
    planned = apply_retakes(_words("it is what it is and that is that"))
    assert any(w.get("candidate") for w in planned)
    assert not any(w.get("disfluency") for w in planned)


def test_repetition_across_a_long_pause_is_left_alone():
    """A pause that long means a new sentence, not a break-off."""
    sentence = "yeh accha hai yeh accha hai bilkul"
    assert _kept(sentence, pauses={3: 4.0}) == sentence


def test_far_apart_repetition_is_left_alone():
    sentence = ("camera achha hai aur iski quality bhi kaafi acchi lagti hai mujhe "
                "hamesha se camera achha hai")
    assert _kept(sentence) == sentence


# --- token matching -------------------------------------------------------

def test_prefix_counts_as_a_break_off():
    assert token_similarity("prob", "probably") >= 0.8
    assert token_similarity("ap", "apko") >= 0.8


def test_a_one_letter_stub_is_not_trusted_alone():
    assert token_similarity("a", "apko") < 0.8


def test_small_spelling_drift_still_matches():
    """The ASR romanises the same sound differently between attempts."""
    assert token_similarity("chaliye", "chalie") >= 0.8
    assert token_similarity("kahani", "kahaani") >= 0.8


def test_different_words_do_not_match():
    assert token_similarity("dosto", "india") == 0.0
    assert token_similarity("hai", "the") == 0.0


def test_empty_input_is_safe():
    assert find_retakes([]) == []
    assert apply_retakes([]) == []


# --- the interregnum ------------------------------------------------------

def test_a_filler_between_the_two_takes_does_not_hide_the_repeat():
    """The shape that survived on the real recording: "you must feel that
    everyone is watching you · uh · you must feel that everyone is watching you".

    A repair is reparandum → interregnum → repair, and the interregnum is made of
    the very words already marked for cutting. Matching straight through the word
    list stops dead at the "uh" and sees a one-word match, so the whole attempt
    was kept and the sentence played twice.
    """
    kept = _kept_full("and you must feel that everyone is watching you uh "
                      "and you must feel that everyone is watching you that how is it")
    assert kept == "and you must feel that everyone is watching you that how is it"


def test_a_hesitation_can_run_long_without_breaking_the_repair():
    """5.3s from the abandoned attempt to the retry, with an "uh" in the middle —
    measured from the recording. A silence that long is normally a new sentence,
    but one the speaker spent floundering in is the opposite."""
    kept = _kept_full(
        "and you must feel that everyone is watching you uh must feel that "
        "everyone is watching you and judging you",
        pauses={10: 2.2, 11: 2.8},
    )
    # "and you" led into the abandoned attempt and was never repeated, so it is
    # not part of the match and stays — the repair itself starts at "must".
    assert kept == "and you must feel that everyone is watching you and judging you"


def test_a_short_echo_across_a_long_silence_is_not_a_retake():
    """A long clean silence raises the evidence bar rather than ending the
    search: a short phrase recurring after a real pause is two sentences, and
    only a substantial reproduction (4+ words) counts as a second take."""
    kept = _kept_full("everyone is watching everyone is watching",
                      pauses={3: 4.0})
    assert kept == "everyone is watching everyone is watching"


def test_a_full_restart_after_a_long_silence_is_collapsed():
    """The on-camera pattern: flub the sentence, stop, breathe for four
    seconds, say it again. The silence is an edit point, not a boundary —
    measured from the real recording, where three such takes played in a row."""
    kept = _kept_full("ek experiment kiyaa gayaa thaa 2000 men "
                      "ek experiment kiyaa gayaa thaa 2000 men cornwell university men",
                      pauses={7: 4.5})
    assert kept == "ek experiment kiyaa gayaa thaa 2000 men cornwell university men"


def test_a_stutter_through_a_filler_is_still_a_stutter():
    """"the uh the" collapses: adjacency is judged on the words that survive."""
    kept = _kept_full("i went to the uh the shop")
    assert kept == "i went to the shop"


def test_seeing_through_a_filler_does_not_make_two_sentences_a_retake():
    """The cost of looking through the interregnum: two ordinary sentences that
    share a couple of words come within reach of each other. From the same
    recording — "they didn't notice it · uh · very few people noticed it" is a
    two-word rough copy, and both halves are things the speaker meant."""
    kept = _kept_full("but they said that they didn't notice it uh "
                      "very few people noticed it")
    assert "very few people noticed it" in kept
    assert "didn't notice it" in kept


# --- romanization variance ---------------------------------------------------

def test_the_same_word_romanised_differently_still_matches():
    """The ASR writes one take as "woh kyaa hai" and the retry as "vo kya hai".
    Token-identical to a listener, string-distinct to the matcher — the phonetic
    fold closes exactly that gap."""
    kept = _kept("woh kyaa bahut a vo kya bahut acha tha")
    assert kept == "vo kya bahut acha tha"


def test_the_phonetic_fold_does_not_invent_matches():
    assert token_similarity("cat", "dog") == 0.0
    assert token_similarity("baat", "boot") == 0.0


def test_a_long_flounder_between_take_and_retry_stays_in_range():
    """From the real recording: a 10-word take, then ~20 words of floundering
    (part of it an inner retake), then the retry — 31 words start to restart.
    The old 30-word lookahead missed the doubled sentence by exactly one."""
    take = "najare aapake hain aur sab aapako jaz kar rahe hain"
    flounder = ("par psychology aur science isake par aur aapako lagataa hogaa "
                "ki sab ke najare aur aapako lagataa hai ki sab ke")
    kept = _kept_full(f"{take} {flounder} {take} ki ye kaisaa hai")
    assert kept.count("jaz kar rahe hain") == 1, "the doubled sentence survived"
    assert kept.endswith("ki ye kaisaa hai")


# ---------------------------------------------------------------------------
# The spoken script, not its romanization
#
# Whisper decodes Hindi as Devanagari and a lossy post-process romanizes it. The
# romanization drifts between two takes of the SAME sentence — the reference
# recording has "jaj"/"jaz" and "dil"/"deel"/"reel" — which is exactly the case
# the matcher exists to catch, so it read the one spelling it could not rely on.
# ---------------------------------------------------------------------------


def _bilingual(pairs, spacing: float = 0.35):
    """Words carrying both spellings, as the ASR emits them."""
    words = []
    clock = 0.0
    for native, roman in pairs:
        words.append({"word": roman, "hinglish": roman, "word_native": native,
                      "start": round(clock, 3),
                      "end": round(clock + spacing * 0.85, 3), "probability": 0.9})
        clock += spacing
    return words


def test_two_unrelated_hindi_words_are_not_a_match():
    """The romanization-repair folds must not run on native script.

    `phonetic()` keeps only ASCII, so on two Devanagari tokens it compares "" to
    "" — which scored every pair of unrelated Hindi words 0.85, above the run
    threshold. Reading the native script without this guard would have made the
    matcher cut on nothing at all.
    """
    from backend.asr.retakes import normalise
    assert token_similarity(normalise("दोस्तों"), normalise("कमरे")) == 0.0
    assert token_similarity(normalise("नहीं"), normalise("नहीं")) == 1.0


def test_a_matra_is_not_stripped_from_a_hindi_token():
    r"""`\w` excludes combining marks, so the old normaliser turned नहीं into नह
    — merging a negation with unrelated words the matcher must keep apart."""
    from backend.asr.retakes import normalise
    assert normalise("नहीं") == "नहीं"


def test_a_repeat_the_romanization_spells_two_ways_is_still_caught():
    """One sentence said twice, romanized inconsistently the second time.

    Reading `word`, the two takes share too few tokens to match and the repeat
    plays twice in the finished edit. Reading `word_native`, they are identical.
    """
    said = [("सब", "sab"), ("आपको", "aapako"), ("जज", "jaj"), ("कर", "kar"),
            ("रहे", "rahe"), ("हैं", "hain")]
    # the retry: same words, a romanizer that spelled three of them differently
    retry = [("सब", "sub"), ("आपको", "apko"), ("जज", "jaz"), ("कर", "karr"),
             ("रहे", "rahen"), ("हैं", "hai")]
    planned = apply_retakes(_bilingual(said + retry))
    cut = [w for w in planned if w.get("disfluency")]
    assert len(cut) == len(said), "the first take should have been cut whole"
    assert all(w.get("reason") in ("retake", "false_start") for w in cut)


def test_the_filler_vocabulary_still_reads_the_romanization():
    """HARD_FILLERS is written in Latin letters, so the filler rule must keep
    reading the romanized spelling even though the repeat rules do not."""
    words = _bilingual([("उम", "um"), ("दोस्तों", "dosto")])
    out = analyze_disfluencies(words)
    assert out[0]["disfluency"] and out[0]["reason"] == "filler"
    assert not out[1]["disfluency"]


def test_a_repeated_grammatical_frame_is_not_a_retake():
    """Hindi lists repeat their frame — "…गया हो या फिर…" after every item. A
    short match made only of such frame words is grammar recurring, not the
    speaker restarting: cutting one copy tore "gaya ho ya phir" out of the
    middle of a list and left "daag lag joota fat"."""
    words = _words("shirt pe kuch daag lag gaya ho ya phir joota fat gaya ho "
                   "ya phir baal kharab ho ya phir aisa kuch bhi")
    apply_retakes(words)
    assert not any(w.get("disfluency") for w in words)
    assert not any(w.get("candidate") for w in words)


def test_a_short_match_with_a_content_word_is_still_a_retake():
    """One content word is what separates a restart from a recurring frame."""
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah hai")
    apply_retakes(words)
    assert any(w.get("disfluency") or w.get("candidate") for w in words[:3])


def test_a_retake_is_seen_across_a_script_flip():
    """Whisper writes the same name in Devanagari in one take and Latin in the
    next ("थॉमल गिल्गोविच" / "Thommel Gilgovich"); on native spellings those
    score zero and the doubled telling survived every structural pass. When the
    scripts differ, the romanized forms are the common ground."""
    words = []
    clock = 0.0
    for token, hinglish in [
        ("उनका", "unka"), ("नाम", "naam"), ("था", "tha"),
        ("थॉमल", "thomal"), ("गिल्गोविच", "gilgovich"),
        ("उनका", "unka"), ("नाम", "naam"), ("था", "tha"),
        ("Thommel", "Thommel"), ("Gilgovich", "Gilgovich"),
        ("Kenneth", "Kenneth"), ("Savitsky", "Savitsky"),
    ]:
        words.append({"word": hinglish, "word_native": token, "hinglish": hinglish,
                      "start": round(clock, 3), "end": round(clock + 0.3, 3),
                      "probability": 0.99})
        clock += 0.35
    # "उनका नाम था थॉमल गिल्गोविच" should match its Latin retry "Thommel naam
    # tha Thommel Gilgovich ..." well enough that the earlier copy is at least
    # a candidate.
    retakes = find_retakes(words)
    assert retakes, "the cross-script doubled naming must be seen"
    assert retakes[0].start == 0
