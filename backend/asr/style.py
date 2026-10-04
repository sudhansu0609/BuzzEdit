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


# --- the stored examples (§4.7) ------------------------------------------------------
#
# One JSON object per line in data/style/examples.jsonl:
#   {"id", "channel", "source", "kind": "hand_cut" | "correction", "text", "created"}
# "hand_cut" excerpts come from recordings the creator cut by hand (tools/build_style.py);
# "correction" ones from answers in the Cuts view's review list. Corrections go first: they
# are the creator overruling this editor, the most direct statement of taste there is.

MAX_EXAMPLES = 8


def store_path():
    from config import DATA_DIR
    return DATA_DIR / "style" / "examples.jsonl"


def load_examples() -> List[Dict[str, Any]]:
    import json
    path = store_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def add_examples(entries: Sequence[Dict[str, Any]], replace_source: str = "") -> int:
    """Append examples; with `replace_source`, first drop that source's hand-cut excerpts so a
    rebuilt key does not pile up duplicates. Returns how many are stored."""
    import json, time, uuid
    kept = [e for e in load_examples()
            if not (replace_source and e.get("source") == replace_source and e.get("kind") == "hand_cut")]
    for e in entries:
        kept.append({"id": uuid.uuid4().hex[:10], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **e})
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
    tmp.replace(path)
    return len(kept)


def stored_style(channel: str = "", exclude_source: str = "", notes: str = "",
                 limit: int = MAX_EXAMPLES) -> str:
    """The style block for a new recording, from the stored examples.

    Corrections before hand cuts, this channel's before other channels', newest first; never
    from `exclude_source` (leave-one-out when the recording being cut is itself an answer key).
    """
    channel = (channel or "").lower()
    pool = [e for e in load_examples() if e.get("text") and e.get("source") != exclude_source]
    bands: Dict[Tuple[bool, bool], List[Dict[str, Any]]] = {}
    for e in pool:
        bands.setdefault((e.get("kind") != "correction",
                          (e.get("channel") or "").lower() != channel), []).append(e)
    ordered = [e for key in sorted(bands) for e in sorted(bands[key], key=lambda e: e.get("created") or "",
                                                          reverse=True)]
    parts = [notes.strip()] if notes.strip() else []
    parts.extend(e["text"] for e in ordered[:limit])
    return "\n\n".join(parts)
