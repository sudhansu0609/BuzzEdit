"""Auto-edit planner: decides which parts of a recording to keep.

The original pipeline looked for filler *words* in the transcript. That cannot
work, and measuring it on a real 195-second project from this repo shows why:

  * Whisper is trained to emit clean, readable text, so it silently deletes
    "um", "uh", stutters and false starts before we ever see them. Matching
    filler vocabulary against its output flagged 4 words out of 351 — 1%.
  * Word timestamps absorb the pauses around them. 23 words claimed over a
    second each, one claimed 10.13 seconds, and 76 seconds of runtime — 39% of
    the video — sat sealed inside "words" where the gap-based cutter could not
    see it.

So the transcript is the wrong place to look. This planner decides from the
**audio**, which is reliable regardless of language, transcript quality or
timestamp drift, and uses the text only where text is genuinely better (spotting
a repeated take, judging a crutch word in context).

Four passes:
  1. Map speech and silence in the audio.
  2. Clamp word timings back onto the speech they actually cover.
  3. Insert the filler sounds Whisper dropped, as disabled words, so they are
     cut *and* visible in the transcript panel for the user to overrule.
  4. Apply the text rules — hard fillers, stutters, repeated takes — with LLM
     adjudication for the ambiguous ones.

The remaining pauses are handled downstream: `rebuild_primary_tracks` trims any
gap longer than the timeline's `max_pause_seconds` down to a natural beat rather
than deleting it outright, which is the difference between a tightened edit and
a breathless one.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .align import find_unvoiced_speech, repair_word_timings
from .disfluency import analyze_disfluencies
from .retakes import apply_retakes
from .vad import SpeechMap, analyse_speech

logger = logging.getLogger("fumble_engine")

# Candidate reasons we're willing to auto-cut WITHOUT the LLM, gated by
# aggressiveness (higher = cut more of the ambiguous ones).
_DETERMINISTIC_CUT_BY_AGGRESSION = {
    "soft_filler": 0.6,      # crutch words: cut when fairly aggressive
    "low_confidence": 0.8,   # risky: only when very aggressive
    # A weak rough copy — one restart, no chain, fluent delivery. It might be an
    # idiom ("it is what it is"), so without an LLM to read the context only a
    # deliberately aggressive setting will drop it.
    "false_start": 0.75,
}

# An un-transcribed speech blip this long or shorter is a filler sound or a
# breath. Anything longer is speech the ASR missed, and cutting it would delete
# the user's content — the one failure this must never make.
_FILLER_MAX_SECONDS = 1.2
# Shared with the rebuild's removal floor (timeline.schema) so a filler this
# pass admits can never be too short for the rebuild to actually cut.
from timeline.schema import FILLER_MIN_SECONDS as _FILLER_MIN_SECONDS


# Cuts the model is never asked about: non-lexical noise, already decided.
_NON_LEXICAL = {"filler", "filler_sound", "stutter"}

# A phrase this long, recurring within a breath, is not rhetoric in unscripted
# speech — it is the speaker saying it twice. The model kept "sab aapako jaz kar
# rahe hain" both times, reading it as deliberate restatement; on a monologue it
# reads as a fumble, so structure gets the last word on long verbatim repeats.
_REPEAT_PHRASE_WORDS = 4


def _long_repeats(words: List[Dict[str, Any]]) -> Set[int]:
    """Indices of an earlier copy of a phrase the speaker said twice.

    Deliberately narrow: only a run of `_REPEAT_PHRASE_WORDS` or more, matching
    verbatim, close enough together to be one breath. Short echoes and spaced
    restatements are left alone — those are the ones that turn out to be
    rhetoric, and cutting them takes meaning with them.
    """
    live = [i for i, w in enumerate(words) if not w.get("disfluency")]
    tokens = [_norm_token(words[i]) for i in live]
    doubled: Set[int] = set()
    window = 25

    position = 0
    while position < len(tokens):
        best = 0
        best_start = -1
        for other in range(position + _REPEAT_PHRASE_WORDS,
                           min(len(tokens), position + window + 1)):
            length = 0
            while (other + length < len(tokens)
                   and position + length < other
                   and tokens[position + length]
                   and tokens[position + length] == tokens[other + length]):
                length += 1
            if length >= _REPEAT_PHRASE_WORDS and length > best:
                best, best_start = length, other
        if best:
            # Drop the FIRST copy and keep the last, the same rule the rest of
            # the pipeline follows: the later take is the completed one.
            doubled.update(live[position + offset] for offset in range(best))
            position = best_start
        else:
            position += 1
    return doubled


async def _second_fluency_pass(words: List[Dict[str, Any]]) -> Set[int]:
    """Re-read the surviving words and report any further cuts, in original indices."""
    try:
        from llm.client import lm_studio_client
        from .fluency import plan_fluent_cuts
    except Exception:
        return set()

    live = [i for i, w in enumerate(words) if not w.get("disfluency")]
    if len(live) < 20:
        return set()
    try:
        decided = await plan_fluent_cuts([words[i] for i in live],
                                         lm_studio_client.clean_transcript)
    except Exception as e:
        logger.warning(f"Second fluency pass failed ({e}); keeping the first answer.")
        return set()
    if not decided:
        return set()

    proposed = {position for position, cut in decided.items() if cut}
    if not proposed:
        return set()

    # Only the two shapes that are repair rather than rewriting — see
    # fluency.admissible_repair_cuts, which the final read applies as well.
    from .fluency import admissible_repair_cuts
    tokens = [_norm_token(words[i]) for i in live]
    return {live[position] for position in admissible_repair_cuts(tokens, proposed)}


def _norm_token(w: Dict[str, Any]) -> str:
    return re.sub(r"[^\w]", "", str(w.get("word", w.get("text", ""))), flags=re.UNICODE).lower()


def _sweep_stutters(words: List[Dict[str, Any]]) -> None:
    """Cut a word that a removal has left sitting next to its own twin.

    Cuts *create* stutters: "kamare men [hol] men gae" reads fine until the cut
    removes "hol" and leaves "men men" playing back to back. Every pass that
    removes words has to be followed by this, because the stutter rule that ran
    earlier could not see a repetition the edit had not made yet.
    """
    previous: Optional[Dict[str, Any]] = None
    for word in words:
        if word.get("disfluency"):
            continue
        text = _norm_token(word)
        if text and previous is not None and _norm_token(previous) == text:
            previous["disfluency"] = True
            previous["reason"] = "stutter"
        previous = word


@dataclass
class EditReport:
    words_total: int = 0
    words_cut: int = 0
    fillers_found: int = 0
    seconds_recovered: float = 0.0
    timing: Dict[str, Any] = field(default_factory=dict)
    speech_fraction: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)
    used_audio: bool = False
    used_llm: bool = False
    used_fluency: bool = False
    # Times an earlier attempt was kept over the final one, and words the last
    # read of the finished edit still found to remove.
    take_swaps: int = 0
    final_read_cuts: int = 0
    # Grammar audit of the finished edit: how many sentences, how many broken,
    # what was wrong and what got repaired.
    quality: Dict[str, Any] = field(default_factory=dict)
    # Fluency-pass reliability: how many windows the model answered, failed to
    # answer (after a retry), or answered untrustably. A high failed count means
    # the edit quietly fell back to structural decisions — worth surfacing.
    fluency_windows: Dict[str, int] = field(default_factory=dict)
    # Speech regions in seconds, for the timeline to intersect kept words with.
    speech: List[Any] = field(default_factory=list)
    # Loudness envelope for cut placement: {"rate": frames/sec, "db": [ints]}.
    energy: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "words_total": self.words_total,
            "words_cut": self.words_cut,
            "fillers_found": self.fillers_found,
            "seconds_recovered": round(self.seconds_recovered, 2),
            "timing": self.timing,
            "speech_fraction": round(self.speech_fraction, 3),
            "reasons": self.reasons,
            "used_audio": self.used_audio,
            "used_llm": self.used_llm,
            "used_fluency": self.used_fluency,
            "take_swaps": self.take_swaps,
            "final_read_cuts": self.final_read_cuts,
            "quality": self.quality,
            "fluency_windows": self.fluency_windows,
            "speech": [[round(s, 3), round(e, 3)] for s, e in self.speech],
            "energy": self.energy,
        }


def _filler_word(start: float, end: float) -> Dict[str, Any]:
    """A detected non-lexical sound, represented as a disabled word.

    Modelling it as a word rather than a separate cut list means it flows through
    the existing timeline rebuild untouched, and shows up in the transcript panel
    where the user can re-enable it if the detector was wrong.
    """
    return {
        "word": "[uh]",
        "word_native": "[uh]",
        "hinglish": "[uh]",
        "start": round(start, 3),
        "end": round(end, 3),
        "probability": 0.0,
        "disfluency": True,
        "candidate": False,
        "reason": "filler_sound",
        "detected": True,
    }


def plan_audio_cuts(
    words: List[Dict[str, Any]],
    audio_path: Optional[str],
    report: EditReport,
    detect_fillers: bool = True,
) -> List[Dict[str, Any]]:
    """Passes 1-3: repair timings against the audio and add the missing fillers."""
    if not audio_path:
        return words

    speech_map: Optional[SpeechMap] = analyse_speech(audio_path)
    if speech_map is None:
        logger.warning("Auto-edit: no usable audio at %s; falling back to text only", audio_path)
        return words

    report.used_audio = True
    report.speech_fraction = (speech_map.speech_seconds / speech_map.duration
                              if speech_map.duration else 0.0)
    # Hand the map to the timeline rebuild so it can cut silence out of the
    # middle of a word, not just between words.
    report.speech = list(speech_map.speech)
    report.energy = speech_map.envelope()

    words, timing = repair_word_timings(words, speech_map)
    report.timing = timing.as_dict()
    report.seconds_recovered += timing.seconds_recovered

    if detect_fillers:
        unvoiced = find_unvoiced_speech(
            words, speech_map,
            min_duration=_FILLER_MIN_SECONDS,
            max_duration=_FILLER_MAX_SECONDS,
        )
        if unvoiced:
            words = sorted(words + [_filler_word(s, e) for s, e in unvoiced],
                           key=lambda w: float(w.get("start", 0.0)))
            report.fillers_found = len(unvoiced)
            report.seconds_recovered += sum(e - s for s, e in unvoiced)
            logger.info("Auto-edit: %d filler sounds recovered from the audio (%.1fs)",
                        len(unvoiced), sum(e - s for s, e in unvoiced))

    return words


async def refine_disfluencies(
    words: List[Dict[str, Any]],
    aggressiveness: float = 0.5,
    use_llm: bool = True,
    audio_path: Optional[str] = None,
    detect_fillers: bool = True,
) -> List[Dict[str, Any]]:
    """Annotate words with the final `disfluency` / `enabled` decision.

    Pass `audio_path` to get the audio-driven passes; without it this degrades to
    the old text-only behaviour, which barely cuts anything on real footage.
    """
    report = EditReport(words_total=len(words))
    words = plan_audio_cuts(words, audio_path, report, detect_fillers=detect_fillers)
    if not words:
        return words

    # Text rules run after the audio passes so they see the repaired timings.
    detected = [w for w in words if w.get("detected")]
    words = analyze_disfluencies(words)
    # analyze_disfluencies rebuilds the dicts; restore the audio-detected verdicts.
    detected_keys = {(round(float(w["start"]), 3)) for w in detected}
    for word in words:
        if round(float(word.get("start", 0.0)), 3) in detected_keys and word.get("word") == "[uh]":
            word["disfluency"] = True
            word["reason"] = "filler_sound"
            word["detected"] = True

    # Gate the whole LLM path on a fast responsiveness probe. The passes below
    # make many model calls in a row; on a model that is too slow, or a "thinking"
    # model that answers empty, that turns a ~30s edit into a 10-minute hang and
    # the UI sits at 0%. If the model can't answer a trivial prompt quickly, drop
    # to the deterministic edit (audio + structure), which always finishes fast.
    if use_llm:
        try:
            from llm.client import lm_studio_client
            if not await lm_studio_client.is_responsive():
                logger.warning("Auto-edit: LLM is not responsive enough for the fluency "
                               "passes; using the deterministic edit instead.")
                use_llm = False
        except Exception as e:
            logger.warning(f"Auto-edit: LLM probe unavailable ({e}); deterministic edit.")
            use_llm = False

    # --- the fluency pass: the model decides what the edit should say --------
    #
    # Everything above is structure: filler vocabulary, repeated runs of words.
    # Structure cannot answer the question that actually matters — "is what is
    # left a complete sentence?" — and on a real recording it cut *inside* the
    # surviving take and left a stitched-together fragment nobody said. So the
    # model is asked directly, and where it answers, it wins.
    fluent_cuts = None
    if use_llm:
        try:
            from llm.client import lm_studio_client
            from .fluency import plan_fluent_cuts
            fluent_cuts = await plan_fluent_cuts(words, lm_studio_client.clean_transcript,
                                                 stats=report.fluency_windows)
        except Exception as e:
            logger.warning(f"Fluency pass unavailable ({e}); structure decides alone.")
            fluent_cuts = None

    if fluent_cuts is not None:
        report.used_fluency = True
        restored = 0
        for index, word in enumerate(words):
            # Absent from the map = the model never ruled on this word (its
            # window was discarded). Structure keeps the last word there.
            if index not in fluent_cuts:
                continue
            # Non-lexical noise is not the model's call: "uh" is not a word it
            # should have to reason about, and it is always cut.
            if word.get("reason") in _NON_LEXICAL and word.get("disfluency"):
                continue
            if fluent_cuts[index]:
                word["disfluency"] = True
                word["candidate"] = False
                if word.get("reason") not in ("retake", "false_start"):
                    word["reason"] = "not_fluent"
            elif word.get("disfluency") and word.get("reason") in ("retake", "false_start"):
                # Structure wanted this gone; the model read the sentence and
                # kept it. The model is the one that can see meaning.
                word["disfluency"] = False
                word["candidate"] = False
                # And the reason goes with it. A kept word carrying "retake"
                # tells the transcript panel to explain why it was removed, on a
                # word that is in the edit.
                word["reason"] = None
                restored += 1
        logger.info("Fluency: ruled on %d words, %d cut, %d structural cuts overruled",
                    len(fluent_cuts), sum(1 for v in fluent_cuts.values() if v), restored)

        # Second pass over what survived. The first pass reads a transcript full
        # of abandoned attempts and filler; with those gone the remaining text is
        # clean enough that a repetition still sitting in it stands out. Cuts
        # only — a second opinion may tighten the edit, never reopen it.
        extra = await _second_fluency_pass(words)
        if extra:
            for index in extra:
                words[index]["disfluency"] = True
                words[index]["candidate"] = False
                if words[index].get("reason") not in ("retake", "false_start"):
                    words[index]["reason"] = "not_fluent"
            logger.info("Fluency: second pass tightened %d more words", len(extra))

        # Structure gets the last word on long verbatim repeats. The model reads
        # a phrase said twice as rhetoric and keeps both; in an unscripted
        # monologue a four-word run coming back within a breath is the speaker
        # repeating themselves, and the viewer hears it as a fumble.
        doubled = _long_repeats(words)
        if doubled:
            for index in doubled:
                words[index]["disfluency"] = True
                words[index]["candidate"] = False
                words[index]["reason"] = "retake"
            logger.info("Fluency: %d words removed as a long repeated phrase", len(doubled))

    candidate_indices = [i for i, w in enumerate(words) if w.get("candidate")]

    to_cut = None
    if use_llm and fluent_cuts is None and candidate_indices:
        try:
            from llm.client import lm_studio_client
            to_cut = await lm_studio_client.adjudicate_disfluencies(
                words, candidate_indices, aggressiveness
            )
        except Exception as e:
            logger.warning(f"LLM adjudication unavailable ({e}); using deterministic fallback.")
            to_cut = None

    if to_cut is not None:
        report.used_llm = True
        for idx in to_cut:
            if 0 <= idx < len(words):
                words[idx]["disfluency"] = True
                if not words[idx].get("reason"):
                    words[idx]["reason"] = "llm_filler"
        logger.info(f"Fumble pipeline: LLM confirmed {len(to_cut)}/{len(candidate_indices)} candidates.")
    else:
        cut = 0
        for i in candidate_indices:
            reason = words[i].get("reason", "")
            threshold = _DETERMINISTIC_CUT_BY_AGGRESSION.get(reason, 1.1)
            if aggressiveness >= threshold:
                words[i]["disfluency"] = True
                cut += 1
        logger.info(
            f"Fumble pipeline: no LLM; deterministically cut {cut}/{len(candidate_indices)} "
            f"candidates at aggressiveness={aggressiveness}."
        )

    # Second retake pass, now that the candidate cuts have landed. The matcher
    # walks the words that survive, so removing an *inner* flounder shortens the
    # distance between an outer abandoned attempt and its retry — a doubled
    # sentence that was out of matching range on the first pass becomes a plain
    # rough copy on the second. One extra pass converges on real recordings.
    # Skipped when the model has read the transcript: re-running structure over
    # its answer would put back exactly the mid-sentence cuts it was brought in
    # to prevent.
    if fluent_cuts is None:
        apply_retakes(words)

    # Cuts create new stutters: "kamare men [hol] men gae" reads fine until the
    # cut removes "hol" and leaves "men men" playing back to back. The stutter
    # rule ran before that cut existed, so sweep the survivors once more.
    _sweep_stutters(words)

    # --- which attempt to keep ----------------------------------------------
    #
    # Structure and the fluency pass both assume the last attempt is the good
    # one. Where an earlier attempt is a genuine full-length alternative reading
    # of the same sentence, ask which one to keep — see retakes.choose_best_takes
    # for how narrowly that case is defined.
    if use_llm:
        try:
            from llm.client import lm_studio_client
            from .retakes import choose_best_takes
            swaps = await choose_best_takes(words, lm_studio_client.clean_transcript)
            if swaps:
                report.take_swaps = swaps
                _sweep_stutters(words)
        except Exception as e:
            logger.warning(f"Best-take pass unavailable ({e}); keeping the last take.")

    # --- verification: is what is left actually a sentence? -----------------
    #
    # Everything above proposes cuts; nothing above checks the result. Read the
    # edit back, repair the sentences leftover debris has broken, drop the
    # abandoned starts that cannot be repaired, and repeat until it stops
    # improving — one pass was never enough, because removing debris from one
    # sentence changes where the next one begins.
    if use_llm:
        try:
            from llm.client import lm_studio_client
            from .verify import verify_until_clean

            quality, cuts = await verify_until_clean(words, lm_studio_client.clean_transcript)
            for index in cuts:
                words[index]["disfluency"] = True
                words[index]["candidate"] = False
                words[index]["reason"] = "not_grammatical"
            report.quality = quality
        except Exception as e:
            logger.warning(f"Verification pass unavailable ({e}); edit not checked.")

    # --- the last read: does it play as one take? ---------------------------
    #
    # The single thing no earlier pass can do: read the finished edit end to end.
    # Everything upstream judged a transcript still full of debris, in windows,
    # against rules. This reads what the viewer will actually hear, and may
    # remove only what still looks like debris.
    if use_llm:
        try:
            from llm.client import lm_studio_client
            from .fluency import final_read

            live = [i for i, w in enumerate(words) if not w.get("disfluency")]
            leftovers = await final_read([words[i] for i in live],
                                         lm_studio_client.clean_transcript)
            for position in leftovers:
                words[live[position]]["disfluency"] = True
                words[live[position]]["candidate"] = False
                words[live[position]]["reason"] = "not_fluent"
            if leftovers:
                report.final_read_cuts = len(leftovers)
                _sweep_stutters(words)
                logger.info("Final read: removed %d leftover words", len(leftovers))
        except Exception as e:
            logger.warning(f"Final read unavailable ({e}); shipping the edit as planned.")

    reasons: Dict[str, int] = {}
    for w in words:
        w["enabled"] = not w.get("disfluency", False)
        if w.get("disfluency"):
            reasons[w.get("reason") or "unspecified"] = reasons.get(w.get("reason") or "unspecified", 0) + 1

    report.words_total = len(words)
    report.words_cut = sum(1 for w in words if not w["enabled"])
    report.reasons = reasons
    if words:
        words[0]["_edit_report"] = report.as_dict()
    logger.info("Auto-edit plan: %s",
                {k: v for k, v in report.as_dict().items() if k not in ("speech", "energy")})
    return words


def last_report(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The report `refine_disfluencies` stashed on the first word, if present."""
    if words and isinstance(words[0], dict):
        return words[0].get("_edit_report") or {}
    return {}
