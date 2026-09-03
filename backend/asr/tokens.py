"""Which of a word's two spellings each pass should read.

Whisper hears Hindi and writes it in Devanagari; `transliterate.to_hinglish`
then makes a romanized copy for the captions, the timeline and the UI. So every
word carries two spellings of the same sound, and which one a pass reads decides
whether it works.

**The romanization is not stable.** Re-transcribing the finished cut of the
reference project and diffing it against the plan scored 0.55 coverage on audio
that had not changed at all: the same sounds came back as `gayaa` where the plan
said `gae`, `hai` for `hain`, `par` for `pe`, `doston` for `dosto`. Inside one
transcript it is no better — the recording's `jaj` and `jaz` are one word said
twice, and `dil`/`deel`/`reel` are one word said three times. Every pass that
decides a cut by comparing tokens — the retake matcher, the stutter sweep, the
long-repeat override, the model's delete-diff — was comparing those, so a
sentence the speaker plainly said twice read as two different sentences and
survived into the edit.

The native script does not drift like that: it is what the model actually
decoded, not a lossy post-process of it. So:

- `spoken(word)`  — the native script. What every pass that JUDGES or COMPARES
  should read: fluency, the grammar audit, retakes, stutters, repeats.
- `romanized(word)` — the Latin form. What belongs where a human or a
  Latin-alphabet tool reads it: the caption text, the timeline's `text`, the
  forced aligner (MMS aligns on romanization), and the romanized filler
  vocabularies in `disfluency.py`.

Both fall back to the other spelling when a word has only one — English speech
carries no native form, and words added by hand (`[uh]`) carry the same string
in both.
"""

from typing import Any, Dict


def spoken(word: Dict[str, Any]) -> str:
    """The word as the speaker said it — native script when there is one.

    Falls back through every spelling the word does carry rather than returning
    nothing: a romanized token still compares usefully against another romanized
    token, whereas an empty one silently drops out of every match.
    """
    return str(word.get("word_native") or word.get("word")
               or word.get("text") or word.get("hinglish") or "")


def romanized(word: Dict[str, Any]) -> str:
    """The word in Latin letters — for captions, the aligner, and vocabularies."""
    return str(word.get("hinglish") or word.get("word")
               or word.get("text") or "")


def is_latin(text: str) -> bool:
    """True when `text` is written in the Latin alphabet.

    The romanization-repair folds in `retakes.py` — the digraph phonetic fold and
    the vowel-stripped consonant skeleton — describe Latin spelling and mean
    nothing anywhere else. Worse than nothing: `phonetic()` keeps only ASCII, so
    on two Devanagari tokens it compares "" against "" and calls every pair of
    unrelated Hindi words an 0.85 match. Passes guard those folds with this.
    """
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in str(text or ""))
