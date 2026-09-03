"""Deterministic forced alignment — put each word where its audio actually is.

Whisper's word timestamps come from decoder cross-attention and are unreliable
on hard audio: on this project's Hindi footage they misplace words by ten
seconds or more (the greeting transcribed at 1.4s was stamped at 11.5s). The cut
keeps source frames by those timestamps, so the video plays the wrong moment —
a fumbled retake where the transcript promises the clean take. The text is right;
the *timing* is not, and nothing downstream can recover from a bad timestamp.

Forced alignment fixes it at the source. It aligns the transcribed WORDS to the
audio acoustically with torchaudio's built-in MMS aligner
(`torchaudio.pipelines.MMS_FA` + `torchaudio.functional.forced_align`), which is
deterministic and accurate regardless of clip length — and ships precompiled in
the torchaudio wheel, so there is nothing to build. We keep Whisper's *words*
(what was said) and replace only its *times* (when).

The whole thing is optional and never fatal: if torch/torchaudio are not
installed, or alignment fails, the original Whisper timings are returned
untouched and the caller carries on. `align_available()` reports whether the real
thing is present.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forced_align")

# Lazily-loaded singletons: the MMS alignment model (~300M params) is loaded once
# and reused across every transcription for the life of the process.
_model = None
_star_dict = None
_load_failed = False
_SAMPLE_RATE = 16000


def align_available() -> bool:
    """True when torch + torchaudio's forced-alignment API import. Cheap — no
    model load. torchaudio ships the MMS aligner precompiled, so nothing to build."""
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from torchaudio.functional import forced_align, merge_tokens  # noqa: F401
        return hasattr(torchaudio.pipelines, "MMS_FA")
    except Exception:
        return False


def _ensure_model(device: str):
    """Load (once) the MMS_FA model and its char dictionary, or (None, None)."""
    global _model, _star_dict, _load_failed
    if _model is not None:
        return _model, _star_dict
    if _load_failed:
        return None, None
    try:
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        _model = bundle.get_model().to(device).eval()
        _star_dict = bundle.get_dict()          # char -> index; blank is '-'=0
        logger.info("MMS forced-alignment model loaded on %s", device)
        return _model, _star_dict
    except Exception as e:
        _load_failed = True
        logger.warning("MMS alignment model unavailable (%s); keeping Whisper timings.", e)
        return None, None


def release_model() -> None:
    """Drop the MMS aligner and hand its VRAM back.

    ~1.2GB that otherwise sits resident for the life of the process, through
    every language-model pass that follows. `_load_failed` is deliberately NOT
    reset: a model that could not load will not load now either, and retrying it
    on every release would cost a minute each time.
    """
    global _model, _star_dict
    if _model is None:
        return
    _model = None
    _star_dict = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("MMS forced-alignment model released.")


def _text_of(word: Dict[str, Any]) -> str:
    """Native script, for logging/mapping context (alignment itself romanizes)."""
    return str(word.get("word_native") or word.get("word") or word.get("text") or "").strip()


def _roman(word: Dict[str, Any], dictionary: Dict[str, int]) -> str:
    """The word as lowercase chars the MMS aligner knows (a-z + apostrophe).

    Uses the romanized 'hinglish'/'word' form — MMS aligns on romanization — and
    drops anything not in the model's alphabet (digits, punctuation, the blank
    '-'). A word left empty by that is un-alignable and keeps its Whisper time.
    """
    text = str(word.get("hinglish") or word.get("word") or word.get("text") or "").lower()
    return "".join(c for c in text if c in dictionary and c != "-")


def _load_waveform(audio_path: str):
    """16kHz mono waveform as a [1, samples] tensor, decoding video if need be."""
    import torch
    from faster_whisper.audio import decode_audio
    audio = decode_audio(audio_path, sampling_rate=_SAMPLE_RATE)   # float32 mono
    return torch.from_numpy(audio).unsqueeze(0)


def align_words(
    audio_path: str,
    words: List[Dict[str, Any]],
    language: Optional[str],
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Return `words` with `start`/`end` replaced by acoustically-aligned times.

    Aligns the romanized transcript to the audio with torchaudio's MMS forced
    aligner (deterministic, accurate at any length). Falls back to the input
    unchanged — same objects, same timings — whenever the aligner is missing or
    anything goes wrong. Never raises. `language` is advisory: MMS aligns on the
    romanization and is language-agnostic.
    """
    if not words or not audio_path or not align_available():
        return words

    model, dictionary = _ensure_model(device)
    if model is None:
        return words

    # The words we can align: those whose romanization has a char the model
    # knows. Fillers like "[uh]" and empty tokens keep their Whisper time.
    alignable = [w for w in words if _roman(w, dictionary)]
    if not alignable:
        return words

    try:
        import torch
        from torchaudio.functional import forced_align, merge_tokens

        romanized = [_roman(w, dictionary) for w in alignable]
        tokens: List[int] = []
        lengths: List[int] = []
        for text in romanized:
            ids = [dictionary[c] for c in text]
            tokens.extend(ids)
            lengths.append(len(ids))

        waveform = _load_waveform(audio_path)
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            if emission.size(1) < len(tokens):
                logger.warning("Forced alignment skipped: %d tokens for %d frames.",
                               len(tokens), emission.size(1))
                return words
            targets = torch.tensor([tokens], dtype=torch.int32, device=device)
            aligned, scores = forced_align(emission, targets, blank=0)
            spans = merge_tokens(aligned[0], scores[0].exp())

        if len(spans) != len(tokens):
            logger.warning("Forced alignment token span mismatch (%d vs %d); "
                           "keeping Whisper timings.", len(spans), len(tokens))
            return words

        seconds_per_frame = waveform.size(1) / emission.size(1) / _SAMPLE_RATE
    except Exception as e:
        logger.warning("Forced alignment failed (%s); keeping Whisper timings.", e)
        return words

    # Walk the per-char spans back into per-word times.
    moved = 0
    cursor = 0
    for word, length in zip(alignable, lengths):
        group = spans[cursor:cursor + length]
        cursor += length
        if not group:
            continue
        start = group[0].start * seconds_per_frame
        end = group[-1].end * seconds_per_frame
        if end <= start:
            end = start + 0.02
        if abs(start - float(word.get("start", 0.0))) > 0.25:
            moved += 1
        word["start"], word["end"] = round(start, 3), round(end, 3)
        word["timing_aligned"] = True

    logger.info("Forced alignment: %d/%d words re-timed (%d moved >0.25s).",
                len(alignable), len(words), moved)
    return words


def _norm(text: str) -> str:
    import re
    return re.sub(r"[^\w]", "", str(text or ""), flags=re.UNICODE).lower()


def map_stamps_to_words(alignable: List[Dict[str, Any]],
                        stamps: List[Dict[str, Any]]) -> Dict[int, tuple]:
    """Match the aligner's word stamps back onto our word list, tolerantly.

    The happy path is a one-to-one, same-length list. But the aligner may split
    or drop a token (punctuation, a joined compound), leaving the counts off by a
    few — and a strict length check would then throw away a perfectly good
    alignment. So when lengths differ we align the two token streams by text with
    difflib and only re-time the words that match a stamp unambiguously; the rest
    keep their Whisper timing. Returns {word_index: (start, end)}.
    """
    if not stamps:
        return {}
    if len(stamps) == len(alignable):
        return {i: (float(s.get("start", 0.0)), float(s.get("end", 0.0)))
                for i, s in enumerate(stamps)}

    import difflib
    left = [_norm(_text_of(w)) for w in alignable]
    right = [_norm(s.get("text", "")) for s in stamps]
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    out: Dict[int, tuple] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            stamp = stamps[j1 + offset]
            out[i1 + offset] = (float(stamp.get("start", 0.0)),
                                float(stamp.get("end", 0.0)))
    return out
