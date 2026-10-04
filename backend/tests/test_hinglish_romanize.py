"""Tests for the natural-Hinglish romanizer (`asr/transliterate.py`).

Whisper decodes Hindi (and code-switched English) into Devanagari; this module
turns that into readable Hinglish: English loanwords spelled in English
("life", not "laaiph"), Hindi in the spelling people actually type ("nahi",
"hoga", "hoon" -- not "naheen", "hogaa", "hoo.n"), and never a stray
Devanagari character, however imperfect the underlying transliteration.
"""

import re

import pytest

from backend.asr.transliterate import to_hinglish

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def hin(word, **kw):
    return to_hinglish(word, "hi", **kw)


# --- colloquial rules -------------------------------------------------------

class TestColloquialRules:
    def test_final_schwa_deletion(self):
        # "aaj" ("today") keeps its real vowel ending; only the inherent
        # consonant-final schwa drops.
        assert hin("आज") == "aaj"

    def test_aa_contracts_to_a(self):
        assert hin("होगा") == "hoga"          # not "hogaa"
        assert hin("वैसा") == "waisa"          # not "vaisaa"

    def test_ee_contracts_to_i(self):
        assert hin("नहीं") == "nahi"
        assert hin("सही") == "sahi"

    def test_oo_contracts_to_u(self):
        assert hin("करूँगा") == "karunga"       # not "karoo.ngaa"

    def test_anusvara_and_chandrabindu_become_n(self):
        assert hin("हूँ") == "hoon"             # chandrabindu
        assert hin("हूं") == "hoon"             # anusvara

    def test_nukta_z_f_kh(self):
        assert hin("ज़्यादा") == "zyada"        # nukta ja -> z
        assert hin("फ़ोन") == "fon"             # nukta pha -> f
        assert hin("ख़ास") == "khas"            # nukta kha -> kh, not "kas"

    def test_nukta_retroflex_flap_reads_as_d(self):
        assert hin("बड़े") == "bade"            # not "ba.de"
        assert hin("थोड़ा") == "thoda"          # not "tho.daa"

    def test_candra_o_never_leaks_devanagari(self):
        # The literal bug report: "कॉफी" -> "kaॉphee" (stray Devanagari char).
        out = hin("कॉफी")
        assert not DEVANAGARI_RE.search(out)
        assert out == "coffee"          # curated loanword table wins outright

    def test_no_devanagari_ever_survives(self):
        # Fuzz a spread of native words (including the ones that exercise the
        # candra-O / nukta leftovers) and assert the hard guarantee holds.
        words = ["कॉफी", "डॉक्टर", "ऑफ", "जॉब", "शॉप", "कॉम्प्लेक्स",
                 "हॉरर", "नहीं", "स्टार्ट", "ज़्यादा"]
        for w in words:
            out = hin(w)
            assert not DEVANAGARI_RE.search(out), f"{w!r} -> {out!r}"


# --- common-word table -------------------------------------------------------

class TestCommonWordTable:
    @pytest.mark.parametrize("native,expected", [
        ("है", "hai"), ("हैं", "hain"), ("नहीं", "nahi"), ("होगा", "hoga"),
        ("हूं", "hoon"), ("मैं", "main"), ("मेरा", "mera"), ("क्या", "kya"),
        ("क्यों", "kyun"), ("और", "aur"), ("भी", "bhi"), ("सब", "sab"),
        ("कुछ", "kuch"), ("अभी", "abhi"), ("ज़्यादा", "zyada"),
        ("थोड़ा", "thoda"), ("अच्छा", "achha"), ("करूंगा", "karunga"),
        ("सही", "sahi"), ("यही", "yahi"), ("वही", "wahi"), ("लेकिन", "lekin"),
        ("मतलब", "matlab"),
    ])
    def test_brief_named_common_words(self, native, expected):
        assert hin(native) == expected

    def test_table_overrides_the_general_rule(self):
        # The general rule alone would produce "hogaa" (long-vowel expansion,
        # not yet contracted the way the table spells it); the table wins.
        assert hin("होगा") != "hogaa"
        assert hin("होगा") == "hoga"


# --- loanword restoration ----------------------------------------------------

class TestLoanwordRestore:
    @pytest.mark.parametrize("native,expected", [
        ("लाइफ", "life"), ("स्टार्ट", "start"), ("ब्रांड", "brand"),
        ("प्रोसेस", "process"), ("फोकस", "focus"), ("चैनल", "channel"),
        ("कॉफी", "coffee"), ("मास्टरपीस", "masterpiece"), ("बिल्ट", "built"),
        ("टाइम", "time"), ("एक्चुअली", "actually"), ("वीडियो", "video"),
        ("आइडिया", "idea"), ("क्रिएटिव", "creative"),
    ])
    def test_curated_loanword_table(self, native, expected):
        assert hin(native) == expected

    def test_skeleton_fallback_restores_an_uncurated_loanword(self):
        # "डिफरेंट" ("different") is not in the curated table; the bundled
        # wordlist + phonetic-skeleton match should still recover it.
        assert hin("डिफरेंट") == "different"

    def test_vocabulary_takes_priority_and_recovers_proper_nouns(self):
        # A word from the user's own script (e.g. a proper noun with no
        # dictionary entry) should be restored via vocabulary, not left as a
        # phonetic guess.
        assert hin("कॉर्नवाल", vocabulary=["Cornwall"]) == "Cornwall"

    def test_latin_typo_is_repaired(self):
        # Whisper sometimes writes a code-switched word directly in (garbled)
        # Latin script even inside an "hi" pass.
        assert hin("aactually") == "actually"

    def test_rule_based_spelling_wins_over_a_weak_wordlist_collision(self):
        # Real false positives measured on a sample transcript: the bundled
        # wordlist's bare skeleton match used to turn these Hindi words into
        # unrelated English ones. The vowel-pattern confidence gate must keep
        # the correct (already-plausible) rule-based spelling instead.
        assert hin("सकते") == "sakate"          # not "society"
        assert hin("शौकिन") == "shaukin"         # not "skin"


# --- glossary precedence -----------------------------------------------------

class TestGlossaryPrecedence:
    def test_glossary_overrides_the_common_word_table_by_native_key(self):
        assert hin("होगा", glossary={"होगा": "hoga-custom"}) == "hoga-custom"

    def test_glossary_overrides_by_romanized_key(self):
        assert hin("लाइफ", glossary={"life": "LIFE"}) == "LIFE"

    def test_glossary_wins_over_vocabulary_restore(self):
        out = hin("कॉर्नवाल", vocabulary=["Cornwall"],
                   glossary={"Cornwall": "Cornwal"})
        assert out == "Cornwal"

    def test_no_glossary_entry_is_a_no_op(self):
        assert hin("होगा", glossary={"unrelated": "x"}) == "hoga"


# --- false-positive guards ---------------------------------------------------

class TestFalsePositiveGuards:
    @pytest.mark.parametrize("native,must_equal", [
        ("मैं", "main"), ("है", "hai"), ("पर", "par"), ("सब", "sab"),
    ])
    def test_real_hindi_words_stay_hindi(self, native, must_equal):
        # These are the explicit collision risks: "main" already looks
        # English, "par"/"sab" share consonant skeletons with "per"/"sub".
        # The common-word table must resolve them before any loanword
        # fallback ever runs.
        assert hin(native) == must_equal

    def test_short_word_below_the_confidence_floor_is_untouched(self):
        # A 2-letter skeleton against the generic wordlist is not enough
        # signal on its own ("kal" == "kl" collides with "call").
        assert hin("कल") == "kal"
