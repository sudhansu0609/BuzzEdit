"""The span-selection planner: the model names what to delete, not what to keep.

The older contract asked the model for the cleaned transcript and then worked out
what it had removed by diffing that text back onto the tokens. Everything awkward
in `fluency.py` exists to survive that guess:

- `align_deletions` reverses both sequences before diffing, because every attempt
  at a sentence opens with the same words and a left-anchored diff splices
  attempt one's opening onto attempt five's ending;
- `snap_runs_to_the_final_take` then drags surviving blocks forward, because even
  a right-anchored diff can assemble the best *text* out of two different takes,
  which reads perfectly and plays as an eight-second jump mid-sentence;
- `admissible_repair_cuts` has to guess from shape alone whether a proposed cut
  was repair or rewriting, because the answer carries no reason.

None of that is inherent to the job. It is all reconstruction of a decision the
model already made and was never asked to state. So here it states it: the tokens
are numbered, and the answer is a list of spans to delete, each with the words at
its ends quoted back and a reason.

    DELETE 12-27 | दोस्तों ... आप | retake
    DELETE 44-44 | उम | filler
    NONE

The quoted ends are what makes this safe with a local model. A model that
miscounts indices -- the standard objection to index-based output -- writes a
span whose quoted words do not match the tokens it points at, and the mismatch is
detectable: the span is relocated by searching for the quoted pair, or refused.
An index scheme without that check is a guess wearing a number.

Delete-only, as everywhere else in this pipeline: the model can name a span to
remove, and nothing else. It cannot reword, reorder or add, so a model that
paraphrases or hallucinates can only fail to cut.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("spans")

# Reasons the planner understands, folded onto the tags the rest of the pipeline
# already uses. Anything else the model writes becomes the generic "not_fluent".
REASONS = {
    "false_start": "retake",
    "abandoned": "retake",
    "attempt": "retake",
    "retake": "retake",
    "repetition": "retake",
    "repeat": "retake",
    "filler": "filler",
    "stutter": "stutter",
}

# MULTILINE is load-bearing for `looks_like_spans`, which searches the whole
# answer: without it a two-line DELETE answer matches nothing (the first line's
# `$` is mid-string, the second line's `^` never anchors) and a perfectly good
# span answer is misread as a rewrite. `parse_spans` splits lines first, so it
# never noticed.
_DELETE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?DELETE\s+(\d+)\s*(?:-|–|—|to|\.\.+)\s*(\d+)(.*)$",
    re.IGNORECASE | re.MULTILINE)
_SINGLE_RE = re.compile(r"^\s*(?:[-*]\s*)?DELETE\s+(\d+)\s*(?!\d)(.*)$",
                        re.IGNORECASE | re.MULTILINE)
_NONE_RE = re.compile(r"^\s*(NONE|NO\s+CUTS?|NOTHING)\b", re.IGNORECASE)


@dataclass
class Span:
    """One deletion the model asked for."""
    start: int               # window-relative, inclusive
    end: int                 # window-relative, inclusive
    first: str = ""          # the word the model says is at `start`
    last: str = ""           # the word the model says is at `end`
    reason: str = "not_fluent"
    relocated: bool = False  # the quoted ends did not sit where the model said

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def indices(self) -> Set[int]:
        return set(range(self.start, self.end + 1))


def _reason_of(text: str) -> str:
    lowered = str(text or "").strip().lower()
    for key, canonical in REASONS.items():
        if key in lowered:
            return canonical
    return "not_fluent"


def _quoted_ends(tail: str) -> Tuple[str, str]:
    """The `| first ... last |` the model quoted back, if it quoted anything.

    Tolerant on purpose: models write the separator as an ellipsis, a dash, or
    just a run of words. The first and last word of whatever sits in the quote
    field is all this needs, so a model that quotes the entire span verifies just
    as well as one that quotes only its ends.
    """
    parts = [p.strip() for p in tail.split("|") if p.strip()]
    if not parts:
        return "", ""
    quote = re.sub(r"\s*(?:…|\.\.\.+|—|–)\s*", " ", parts[0])
    tokens = quote.split()
    if not tokens:
        return "", ""
    return tokens[0], tokens[-1]


def parse_spans(answer: str, total: int) -> List[Span]:
    """Every well-formed DELETE line in `answer`, clamped to the window.

    Lines that are not DELETE instructions are ignored rather than treated as a
    failure -- a model that prefaces its answer with a sentence of commentary has
    still answered. An answer with no DELETE line at all yields an empty list,
    which the caller must tell apart from a failure to answer: see
    `looks_like_spans`.
    """
    spans: List[Span] = []
    for line in str(answer or "").splitlines():
        match = _DELETE_RE.match(line)
        if match:
            start, end, tail = int(match.group(1)), int(match.group(2)), match.group(3)
        else:
            match = _SINGLE_RE.match(line)
            if not match:
                continue
            start = end = int(match.group(1))
            tail = match.group(2)
        if start > end:
            start, end = end, start
        if start >= total:
            continue                        # entirely past the window
        first, last = _quoted_ends(tail)
        spans.append(Span(start=start, end=min(end, total - 1),
                          first=first, last=last, reason=_reason_of(tail)))
    return spans


def parse_json_spans(answer: str, total: int) -> Optional[List[Span]]:
    """Spans from the JSON contract: {"deletions": [{first_index, last_index,
    first_word, last_word, reason}]}.

    The schema-constrained request is what makes a small local model answer in
    span form at all — the same model asked for DELETE lines ignores the format
    and rewrites the transcript instead. Verification is unchanged: the quoted
    words still have to match the tokens the indices point at, so constrained
    output is no more trusted than free-form, it is just reliably parseable.

    None means the answer was not this shape at all (the caller should fall
    back); an empty list is a real "nothing to delete".
    """
    import json as _json

    text = str(answer or "").strip()
    if not text:
        return None
    try:
        parsed = _json.loads(text)
    except Exception:
        # Tolerate prose around the object, the same latitude the line parser
        # gives commentary around DELETE lines.
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index < 0 or end_index <= start_index:
            return None
        try:
            parsed = _json.loads(text[start_index:end_index + 1])
        except Exception:
            return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("deletions"), list):
        return None

    spans: List[Span] = []
    for raw in parsed["deletions"]:
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw.get("first_index"))
            end = int(raw.get("last_index"))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        if start >= total or end < 0:
            continue
        spans.append(Span(
            start=max(0, start), end=min(end, total - 1),
            first=str(raw.get("first_word") or "").strip(),
            last=str(raw.get("last_word") or "").strip(),
            reason=_reason_of(str(raw.get("reason") or "")),
        ))
    return spans


def looks_like_spans(answer: str) -> bool:
    """Whether the model answered in this format at all.

    `NONE` is a real answer -- "nothing to cut here" -- and must not be mistaken
    for a model that ignored the format and rewrote the transcript instead. Those
    two need opposite handling: the first is a decision to respect, the second is
    the signal to fall back to the older rewrite-and-diff contract.
    """
    text = str(answer or "").strip()
    if not text:
        return False
    return bool(_DELETE_RE.search(text) or _SINGLE_RE.search(text)
                or _NONE_RE.match(text))


# How far a span's un-verified end may be moved to reach the word the model
# quoted for it. Small on purpose: one end has already verified in place, so the
# other being a token or two off is a counting slip, while a longer hunt would
# let a mis-quoted word drag the span onto unrelated text.
_STRETCH_LIMIT = 3


def _stretch(span: Span, norms: Sequence[str]) -> Optional[Span]:
    """Repair a span whose quoted words match at ONE end but not the other.

    The dominant miscount in practice is off-by-one-or-two on a single end —
    the model counts the run right, anchors one end right, and slips on the
    other. With the good end verified in place, the bad end is moved to the
    nearest occurrence of its quoted word within `_STRETCH_LIMIT` tokens of
    where the model said it was. Both ends then match the transcript, which is
    the same guarantee a span that verified outright carries.
    """
    first_ok = span.first and span.start < len(norms) and norms[span.start] == span.first
    last_ok = span.last and span.end < len(norms) and norms[span.end] == span.last
    if first_ok == last_ok:
        return None                      # both fine or both wrong; not this repair
    if first_ok:
        candidates = [p for p in range(max(span.start, span.end - _STRETCH_LIMIT),
                                       min(len(norms), span.end + _STRETCH_LIMIT + 1))
                      if norms[p] == span.last]
        if not candidates:
            return None
        end = min(candidates, key=lambda p: abs(p - span.end))
        return Span(start=span.start, end=end, first=span.first, last=span.last,
                    reason=span.reason, relocated=True)
    candidates = [p for p in range(max(0, span.start - _STRETCH_LIMIT),
                                   min(span.end + 1, span.start + _STRETCH_LIMIT + 1))
                  if norms[p] == span.first]
    if not candidates:
        return None
    start = min(candidates, key=lambda p: abs(p - span.start))
    return Span(start=start, end=span.end, first=span.first, last=span.last,
                reason=span.reason, relocated=True)


def _relocate(span: Span, norms: Sequence[str]) -> Optional[Span]:
    """Find where the model's quoted words actually are, or give up.

    Searches for a span of the same length whose ends carry the quoted words,
    nearest to where the model said it was. The length is held fixed: a model
    that miscounted the offset still counted the span itself, and letting the
    length float would turn a small numbering slip into an arbitrary cut.
    """
    first, last = span.first, span.last
    if not first or not last:
        return None
    length = span.length
    candidates = [
        position for position in range(0, len(norms) - length + 1)
        if norms[position] == first and norms[position + length - 1] == last
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(p - span.start))
    return Span(start=best, end=best + length - 1, first=span.first,
                last=span.last, reason=span.reason, relocated=True)


def verify_spans(spans: Sequence[Span], tokens: Sequence[str], normalise
                 ) -> Tuple[List[Span], List[Span]]:
    """Keep the spans whose quoted ends match the tokens they point at.

    Returns (kept, refused) — both as spans, so the caller can log the refusals
    AND mark their regions as ruled-on-but-unusable. Three outcomes per span: it
    lands where the model said (kept), it lands elsewhere but its quoted words
    are findable (relocated or end-stretched, kept), or those words are nowhere
    near where it points (refused -- the model was describing a transcript that
    is not the one it was given).

    A span that quotes nothing is kept as-is. Quoting is what makes an index
    checkable, so an unquoted span is unverified rather than wrong; the trust
    limits in `judge_spans` are what stand behind it.
    """
    norms = [normalise(t) for t in tokens]
    kept: List[Span] = []
    refused: List[Span] = []
    for span in spans:
        if span.start >= len(norms) or span.end >= len(norms):
            refused.append(span)
            continue
        if not span.first and not span.last:
            kept.append(span)
            continue
        quoted_first, quoted_last = normalise(span.first), normalise(span.last)
        first_ok = not quoted_first or norms[span.start] == quoted_first
        last_ok = not quoted_last or norms[span.end] == quoted_last
        if first_ok and last_ok:
            kept.append(span)
            continue
        normalised_span = Span(start=span.start, end=span.end, first=quoted_first,
                               last=quoted_last, reason=span.reason)
        moved = _stretch(normalised_span, norms) or _relocate(normalised_span, norms)
        if moved is not None:
            moved.first, moved.last = span.first, span.last
            kept.append(moved)
            continue
        refused.append(span)
    return kept, refused


def merge(spans: Sequence[Span]) -> List[Span]:
    """Overlapping or touching spans folded into one, in order."""
    out: List[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if out and span.start <= out[-1].end + 1:
            previous = out[-1]
            previous.end = max(previous.end, span.end)
            previous.last = span.last or previous.last
            previous.relocated = previous.relocated or span.relocated
        else:
            out.append(Span(start=span.start, end=span.end, first=span.first,
                            last=span.last, reason=span.reason,
                            relocated=span.relocated))
    return out
