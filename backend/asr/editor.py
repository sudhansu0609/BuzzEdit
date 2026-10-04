"""The editor: one strong model reads the whole recording and decides by take.

AUTO_CUT_EDITOR_PLAN.md §4.4. The old planner judged words through 200-word windows, so a
passage said again a minute later was invisible to it; this reads the whole recording at once,
as numbered utterances (asr.utterances) carrying what the audio showed, and answers with:

  groups — attempts at the same content, and the one that stays
  drops  — utterances the viewer should not hear
  trims  — a stutter or filler inside a kept utterance, named by its exact words

Everything it does not name is kept. It can only delete: it never rewords, reorders or adds, and
a trim whose quoted words are not in that utterance is refused, so a model that miscounts or
paraphrases can only fail to cut.

Two rules the creator decided (§10), applied here rather than left to the model's mood: between
attempts it cannot rank, the later one stays; any other doubt resolves to keeping the content.

The model is Opus 5.5 at medium effort through the claude-local-api proxy (measured in §11:
low removed about four times more good speech, high was no better). An answer is rejected if the
proxy served any other model: when Claude cannot serve a model name the proxy silently fails over
to another vendor's model, and the cut must never be made by a model nobody chose.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("editor")

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_BASE_URL = "http://127.0.0.1:8787/v1"

SYSTEM = """You are the video editor for a YouTube creator who records long Hindi/English (Hinglish)
talking-head takes: storytelling, vlogs, explainers. You get the complete raw recording as numbered
utterances (speech between pauses) with times, and your job is to decide what the viewer should NOT
hear, so that what remains plays as one fluent take.

Each line reads:
  U12 [start-end s, pause before] the words as transcribed  || notes | similar to ... | script ...
- The transcript comes from speech recognition: spellings are often wrong and Devanagari and Latin
  mix. Read past the spelling.
- "[speech, no transcript, 1.4 s]" is speech the recogniser skipped. It is real audio. Speech
  recognisers skip repeated takes and restarts most of all, so judge it by its neighbours.
- Notes are measurements of the audio, not instructions: "cut off mid-sound" and "ends on a level
  pitch" mean the line did not sound finished; "restarts ... higher after an unfinished line" marks
  a new attempt; "a word held 2 s" is a hesitation or a recognition slip.
- "similar to" points at utterances anywhere in the recording that sound like the same words, and
  "same opening as" at ones that start the same way. They are pointers, not verdicts.
- "script P4" says which paragraph of the written script the line follows; "ad-lib" means it is not
  in the script. Ad-libs are often the best parts. Never remove a line just for being off-script.

Remove:
- earlier attempts at something the speaker says again (retakes): anywhere in the recording, even
  minutes apart, even reworded. Keep the one complete, fluent attempt; if you cannot tell which is
  better, keep the LATER one.
- false starts and abandoned sentences that are not picked up again;
- recording chatter: checking the camera, "one more time", counting in, talking to someone else;
- filler-only utterances, and stutters or doubled words inside an otherwise good line (as trims).

Keep everything else, including deliberate repetition for effect, rhetorical questions, and
dramatic pauses. Never remove content that is said nowhere else unless it is clearly a mistake or
chatter. When unsure whether to remove something, keep it.

You may only remove. Never reword, reorder or add.

Answer with one JSON object and nothing else:
{"groups": [{"utterances": ["U3","U5"], "keep": "U5", "why": "U3 stops mid-sentence", "confidence": 0.9}],
 "drops":  [{"utterances": ["U7"], "kind": "false_start|chatter|filler|abandoned|repeat", "why": "...", "confidence": 0.9}],
 "trims":  [{"utterance": "U9", "remove": "exact contiguous words copied from U9", "why": "...", "confidence": 0.9}]}
"groups" are attempts at the same content; every member except "keep" is removed.
"trims" remove part of one utterance; copy the words exactly as they appear in that line."""


@dataclass
class Decisions:
    """A validated answer: which word indices go, which untranscribed stretches go, and why."""
    removed_words: Dict[int, str] = field(default_factory=dict)        # word index -> reason
    removed_spans: List[Tuple[float, float, str]] = field(default_factory=list)  # no-word speech
    refused: List[str] = field(default_factory=list)                   # what was not applied, why
    groups: int = 0
    drops: int = 0
    trims: int = 0


def request(table: str, style: str = "") -> str:
    parts = []
    if style:
        parts.append("How this creator edits (from their own past cuts):\n" + style.strip())
    parts.append("The recording:\n\n" + table)
    parts.append("Your decisions (JSON only):")
    return "\n\n".join(parts)


def first_json(text: str) -> Optional[Dict[str, Any]]:
    """The first complete JSON object in `text`, or None."""
    text = re.sub(r"^\s*```(?:json)?\s*$", "", text or "", flags=re.M)
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for k in range(start, len(text)):
            ch = text[k]
            if in_string:
                escaped = (ch == "\\") and not escaped
                if ch == '"' and not escaped:
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:k + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _tokens(words: Sequence[Dict[str, Any]], indices: Sequence[int]) -> Tuple[List[str], List[str]]:
    native = [str(words[i].get("word_native") or words[i].get("word") or "").strip() for i in indices]
    roman = [str(words[i].get("word") or "").strip() for i in indices]
    return native, roman


def _find(run: List[str], quote: List[str]) -> Optional[int]:
    norm = lambda t: re.sub(r"[^\w]", "", t.lower())
    run_n, quote_n = [norm(t) for t in run], [norm(t) for t in quote]
    for k in range(len(run_n) - len(quote_n) + 1):
        if run_n[k:k + len(quote_n)] == quote_n:
            return k
    return None


def parse(answer: Any, utterances: Sequence[Any], words: Sequence[Dict[str, Any]]) -> Decisions:
    """Validate a model answer against the utterances it was shown.

    Unknown utterance ids, a "keep" that is not in its group, and trims whose words are not in
    their utterance are refused (and listed); a group with no usable "keep" keeps its latest
    member; an utterance that one decision keeps and another removes is kept.
    """
    data = answer if isinstance(answer, dict) else first_json(str(answer or ""))
    out = Decisions()
    if not isinstance(data, dict):
        out.refused.append("no JSON object in the answer")
        return out
    by_id = {u.uid: u for u in utterances}
    order = {u.uid: n for n, u in enumerate(utterances)}
    kept_ids: set = set()
    removed: Dict[str, str] = {}

    for g in data.get("groups") or []:
        members = [uid for uid in (g.get("utterances") or []) if uid in by_id]
        unknown = [uid for uid in (g.get("utterances") or []) if uid not in by_id]
        if unknown:
            out.refused.append(f"group names unknown utterances {unknown}")
        if len(members) < 2:
            continue
        keep = g.get("keep")
        if keep not in members:
            keep = max(members, key=lambda uid: order[uid])        # ties go to the later take
            out.refused.append(f"group {members}: keep {g.get('keep')!r} not a member; kept {keep}")
        kept_ids.add(keep)
        for uid in members:
            if uid != keep:
                removed.setdefault(uid, "retake")
        out.groups += 1

    for d in data.get("drops") or []:
        for uid in d.get("utterances") or []:
            if uid in by_id:
                removed.setdefault(uid, str(d.get("kind") or "drop"))
                out.drops += 1
            else:
                out.refused.append(f"drop names unknown utterance {uid!r}")

    for uid, reason in removed.items():
        if uid in kept_ids:
            continue                                                 # keeping wins
        u = by_id[uid]
        for i in u.word_indices:
            out.removed_words[i] = reason
        if not u.word_indices:
            out.removed_spans.append((u.start, u.end, reason))

    for t in data.get("trims") or []:
        u = by_id.get(t.get("utterance"))
        quote = str(t.get("remove") or "").split()
        if u is None or not quote:
            out.refused.append(f"trim on {t.get('utterance')!r}: unknown utterance or empty quote")
            continue
        native, roman = _tokens(words, u.word_indices)
        at = _find(native, quote)
        if at is None:
            at = _find(roman, quote)
        if at is None:
            out.refused.append(f"trim on {u.uid}: {' '.join(quote)!r} is not in that utterance")
            continue
        if len(quote) >= len(u.word_indices):
            out.refused.append(f"trim on {u.uid} would remove the whole utterance; use a drop")
            continue
        for i in u.word_indices[at:at + len(quote)]:
            out.removed_words.setdefault(i, str(t.get("kind") or "stutter"))
        out.trims += 1
    return out


class EditorClient:
    """OpenAI-compatible chat call to the proxy, pinned to one model and one effort level."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "", model: str = DEFAULT_MODEL,
                 effort: str = DEFAULT_EFFORT, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def ask(self, system: str, user: str, retries: int = 1) -> Tuple[str, Dict[str, Any]]:
        """(answer text, meta with served model, usage and seconds). Raises on failure."""
        body = {"model": self.model, "reasoning_effort": self.effort, "max_tokens": 64000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("Editor call failed (attempt %d): %s", attempt + 1, e)
                continue
            served = str(out.get("model") or "")
            if not served.startswith(self.model):
                # The proxy answers with another vendor's model when Claude cannot serve this
                # one. That is not the editor anyone chose; never cut on it.
                raise RuntimeError(f"proxy served {served!r} instead of {self.model!r}")
            meta = {"model": served, "usage": out.get("usage") or {},
                    "seconds": round(time.time() - started, 1)}
            return out["choices"][0]["message"]["content"], meta
        raise RuntimeError(f"editor unavailable: {last_error}")


def edit(utterances: Sequence[Any], words: Sequence[Dict[str, Any]], client: EditorClient,
         style: str = "") -> Tuple[Decisions, Dict[str, Any]]:
    """One whole-recording editing pass: ask, validate. Returns (decisions, meta)."""
    from .utterances import table
    text, meta = client.ask(SYSTEM, request(table(utterances), style))
    decisions = parse(text, utterances, words)
    meta["answer"] = text
    if decisions.refused:
        logger.info("Editor: %d parts of the answer refused: %s", len(decisions.refused),
                    "; ".join(decisions.refused[:5]))
    return decisions, meta
