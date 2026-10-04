"""The creator's taste, shown to the editor as worked examples from their own cuts.

AUTO_CUT_EDITOR_PLAN.md §4.7. Rules cannot hold taste: which of two complete takes to keep,
whether an aside about the channel stays, how much of a recording-start ramble goes. What
the editor got wrong in the first measured runs was exactly that, content the creator kept
(the setup of a story, an aside about the channel). So every recording the creator cut by
hand, once matched to its raw (tools/autocut_truth.py), is turned into excerpts: a stretch of
the utterance table, and what the creator did with each line. The editor reads a few of those
before it reads a new recording.

An example is only ever taken from a *different* recording than the one being edited, so a
score on the answer keys measures what carries over to a recording the editor has never seen.
"""

from typing import Any, Dict, List, Sequence, Tuple

EXCERPT_LINES = 10            # utterances per excerpt
EXCERPTS_PER_SOURCE = 4
REMOVED_SHARE = 0.6           # an utterance at least this much inside a removed span was removed
TRIMMED_SHARE = 0.15          # ...between this and REMOVED_SHARE it was trimmed


def labels(utterances: Sequence[Any], removed_spans: Sequence[Tuple[float, float]],
           editorial: Sequence[Tuple[float, float]] = ()) -> Dict[str, str]:
    """uid -> "removed" | "trimmed" | "kept" | "section" (inside an editorial section drop)."""
    def overlap(a: float, b: float, spans) -> float:
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)

    out = {}
    for u in utterances:
        length = max(1e-3, u.end - u.start)
        if overlap(u.start, u.end, editorial) / length >= 0.5:
            out[u.uid] = "section"
            continue
        share = overlap(u.start, u.end, removed_spans) / length
        out[u.uid] = "removed" if share >= REMOVED_SHARE else "trimmed" if share >= TRIMMED_SHARE else "kept"
    return out


def _line(u: Any) -> str:
    from .utterances import table
    return table([u])


def excerpts(utterances: Sequence[Any], decided: Dict[str, str], name: str,
             per_source: int = EXCERPTS_PER_SOURCE) -> List[str]:
    """Worked examples from one cut recording: windows where the creator removed something
    next to something kept, retakes first (a removed line pointing at a similar kept one)."""
    index = {u.uid: n for n, u in enumerate(utterances)}

    def interest(n: int) -> float:
        window = utterances[n:n + EXCERPT_LINES]
        states = [decided.get(u.uid) for u in window]
        if "section" in states or not any(s in ("removed", "trimmed") for s in states) or "kept" not in states:
            return 0.0
        retakes = sum(1 for u in window if decided.get(u.uid) == "removed"
                      and any(decided.get(uid) == "kept" for uid, _ in u.similar))
        return retakes * 2.0 + sum(s in ("removed", "trimmed") for s in states)

    picked: List[int] = []
    for n in sorted(range(max(1, len(utterances) - EXCERPT_LINES + 1)), key=lambda n: -interest(n)):
        if interest(n) <= 0 or len(picked) >= per_source:
            break
        if all(abs(n - p) >= EXCERPT_LINES for p in picked):
            picked.append(n)

    out = []
    for n in sorted(picked):
        window = utterances[n:n + EXCERPT_LINES]
        lines = [_line(u) for u in window]
        verdicts = []
        for u in window:
            state = decided.get(u.uid)
            if state == "removed":
                kept_twin = next((uid for uid, _ in u.similar
                                  if decided.get(uid) == "kept" and index.get(uid, -1) > index[u.uid]), None)
                verdicts.append(f"{u.uid}: removed" + (f" (an earlier attempt; {kept_twin} kept)" if kept_twin else ""))
            elif state == "trimmed":
                verdicts.append(f"{u.uid}: partly removed")
            else:
                verdicts.append(f"{u.uid}: kept")
        out.append(f"From {name}:\n" + "\n".join(lines) + "\nWhat the creator did: " + "; ".join(verdicts))
    return out


def style_block(sources: Sequence[Tuple[str, Sequence[Any], Dict[str, str]]], notes: str = "") -> str:
    """The text placed before a new recording: optional channel notes, then the excerpts.
    `sources` are (name, utterances, labels) of OTHER recordings than the one being edited."""
    parts = [notes.strip()] if notes.strip() else []
    for name, utts, decided in sources:
        parts.extend(excerpts(utts, decided, name))
    return "\n\n".join(parts)
