"""Answering the AI editor's review list, and learning from the answer (AUTO_CUT_EDITOR_PLAN.md
Phase 6, §4.7).

The list holds the calls the editor made under doubt (its reads disagreed, so the take was
kept) and the cut points the listening check could not settle. An answer is applied to the
whole take at once, and a keep/cut answer is stored as a "correction" example, which every later
edit of the same channel reads before the creator's hand-cut excerpts. No training: the editor
simply sees what the creator decided last time.
"""

from typing import Any, Dict, List, Optional

ANSWERS = ("keep", "cut", "dismiss")


def answer(timeline, take: str, verdict: str, primary_source_id: str,
           channel: str = "", source: str = "") -> Dict[str, Any]:
    """Apply `verdict` to every review item and word of `take`; returns {changed, example}.

    keep    — every word of the take plays;
    cut     — every word of the take is removed (reason "review");
    dismiss — nothing changes (a listening item the user heard and is happy with).
    Raises KeyError when the take is not on the review list, ValueError on a bad verdict.
    """
    from timeline.ops import rebuild_primary_tracks

    if verdict not in ANSWERS:
        raise ValueError(f"answer must be one of {ANSWERS}")
    items = [r for r in timeline.review if r.get("take") == take]
    if not items:
        raise KeyError(take)
    changed = 0
    if verdict in ("keep", "cut"):
        enable = verdict == "keep"
        for w in timeline.words:
            if w.take != take:
                continue
            if w.enabled != enable:
                changed += 1
            w.enabled = enable
            w.disfluency = not enable
            w.candidate = False
            w.reason = None if enable else "review"
        if changed:
            rebuild_primary_tracks(timeline, primary_source_id)
            timeline.revision += 1
    for r in items:
        r["answer"] = verdict

    example = None
    doubt = next((r for r in items if r.get("kind") == "doubt"), None)
    if doubt and verdict in ("keep", "cut"):
        from .style import add_examples
        did = "kept" if verdict == "keep" else "removed"
        twin = f" (it sounds like {doubt['twin']})" if doubt.get("twin") else ""
        example = {
            "kind": "correction", "channel": channel, "source": source,
            "text": (f"From {source or 'a past recording'} (the creator's own answer):\n"
                     f"{doubt.get('context', '')}\nWhat the creator did: {take}: {did}{twin}. "
                     f"The editor had been unsure: {doubt.get('why') or doubt.get('votes', '')}"),
        }
        add_examples([example])
    return {"changed": changed, "example": bool(example)}


def open_items(timeline) -> List[Dict[str, Any]]:
    return [r for r in timeline.review if r.get("answer") is None]
