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

The speaker, dialect and style:
- Regional and dialect Hindi forms are style, never errors: Bihari, Awadhi, Bhojpuri, Haryanvi or
  Punjabi-influenced verb forms, "हम" for "मैं", regional vocabulary. So are Hinglish
  code-switching and the speaker's habitual phrases.
- Never remove or "correct" grammar: unusual grammar is how this speaker talks, not a mistake.
- The fixed greeting line is said once: only an extra copy of it goes.
- Deliberate repetition for effect in storytelling stays.
- A retake is removed only when a kept line carries its content.
- A false start is a line that breaks off (cut off, ends on a connective, restarts) and whose
  content the next attempt picks up. Such a line goes.

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
    trim_ranges: List[Tuple[str, int, int]] = field(default_factory=list)  # (uid, first, last word index)


def request(table: str, style: str = "", speaker_style: str = "") -> str:
    parts = []
    if speaker_style and speaker_style.strip():
        parts.append("About this speaker (from their own channel notes):\n" + speaker_style.strip())
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


def _find(run: List[str], quote: List[str], fuzzy: bool = False) -> Optional[int]:
    """Where `quote` starts in `run`. Tokens compare NFC-normalised and lowercased with
    punctuation dropped; Devanagari vowel marks are kept ("के" is not "की"). `fuzzy` also
    accepts a token of 4+ characters one edit away from its word, for at most a third of
    the quote's tokens (rounded up)."""
    import unicodedata

    norm = lambda t: re.sub(r"[^\w\u0900-\u097F]", "", unicodedata.normalize("NFC", t).lower())
    run_n, quote_n = [norm(t) for t in run], [norm(t) for t in quote]

    def near(a: str, b: str) -> bool:                  # both 4+ characters, Levenshtein <= 1
        if len(a) < 4 or len(b) < 4 or abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) <= 1
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        k = 0
        while k < len(short) and short[k] == long_[k]:
            k += 1
        return short[k:] == long_[k + 1:]

    allowed = (len(quote_n) + 2) // 3 if fuzzy else 0   # ceil(len / 3) slipped tokens
    for k in range(len(run_n) - len(quote_n) + 1):
        pairs = list(zip(run_n[k:k + len(quote_n)], quote_n))
        slips = sum(a != b for a, b in pairs)
        if slips == 0 or (slips <= allowed and all(a == b or near(a, b) for a, b in pairs)):
            return k
    return None


def _twin(run: List[str], at: int, n: int) -> Optional[int]:
    """Where `run` says its n tokens at `at` again: the first n-token window not overlapping them
    whose tokens match theirs at 60 % or more of the positions (equal, or one slip apart as
    `_find` allows). None when the phrase is said only once."""
    for k in range(len(run) - n + 1):
        if at - n < k < at + n:
            continue                                   # overlaps the quoted run
        same = sum(_find([run[k + j]], [run[at + j]], fuzzy=True) is not None for j in range(n))
        if same * 5 >= n * 3:                          # 60 % of the positions
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
        # The table writes speech no word covers as "[speech, no transcript, 1.2 s]"; a quote
        # that copies it (or "[…]") means the words around it, so it is not matched as a word.
        remove = re.sub(r"\[speech, no transcript, [\d.]+ s\]", " ", str(t.get("remove") or ""))
        quote = remove.replace("[…]", " ").split()
        if u is None or not quote:
            out.refused.append(f"trim on {t.get('utterance')!r}: unknown utterance or empty quote")
            continue
        native, roman = _tokens(words, u.word_indices)
        at = _find(native, quote)
        if at is None:
            at = _find(roman, quote)
        if at is None and len(quote) >= 3:                 # a slip of one letter, a third of the tokens
            at = _find(native, quote, fuzzy=True)
            if at is None:
                at = _find(roman, quote, fuzzy=True)
        if at is None:
            out.refused.append(f"trim on {u.uid}: {' '.join(quote)!r} is not in that utterance")
            continue
        if len(quote) >= len(u.word_indices) or (len(quote) >= 4 and len(u.word_indices) - len(quote) < 3):
            out.refused.append(f"trim on {u.uid} would remove the whole utterance; use a drop")
            continue
        if len(quote) >= 4:                                # a long trim is one copy of a doubled phrase
            twin = _twin(native, at, len(quote))
            if twin is None:
                twin = _twin(roman, at, len(quote))
            if twin is None:
                out.refused.append(f"trim on {u.uid}: {' '.join(quote)!r} ({len(quote)} words) is not a "
                                   "repeated phrase; a trim removes only a stutter or a doubled phrase")
                continue
            at = min(at, twin)                             # the earlier copy goes, whichever was quoted
        for i in u.word_indices[at:at + len(quote)]:
            out.removed_words.setdefault(i, str(t.get("kind") or "stutter"))
        out.trim_ranges.append((u.uid, u.word_indices[at], u.word_indices[at + len(quote) - 1]))
        out.trims += 1
    return out


def proxy_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """{base_url, api_key, model, effort} for the claude-local-api proxy; api_key "" when none
    is configured.

    1. the job's own `llm_*` settings, when the caller routed the job to an OpenAI-compatible
       endpoint (BuzzcafStudio does that for presentation jobs when production runs on Claude);
    2. BuzzEdit's app setting `auto_cut_editor` ({"base_url", "api_key"});
    3. the `CLAUDE_API_KEY` (and optional `CLAUDE_URL`) environment variables.

    The model and effort are never taken from the job's settings, whatever model the caller uses
    for its own planning: `auto_cut_editor` {"model", "effort"} when set, else the
    `CLAUDE_EDITOR_MODEL` / `CLAUDE_EDITOR_EFFORT` environment variables, else Opus 5.5 at
    medium (§10 item 4).
    """
    import os
    settings = settings or {}
    try:
        from store.app_settings import AppSettings
        own = AppSettings().get("auto_cut_editor") or {}
    except Exception:
        own = {}
    pinned = {"model": str(own.get("model") or os.environ.get("CLAUDE_EDITOR_MODEL") or DEFAULT_MODEL),
              "effort": str(own.get("effort") or os.environ.get("CLAUDE_EDITOR_EFFORT") or DEFAULT_EFFORT)}
    if settings.get("llm_api_key") and str(settings.get("llm_provider") or "") == "openai_compat":
        return {"base_url": str(settings.get("llm_base_url") or DEFAULT_BASE_URL),
                "api_key": str(settings["llm_api_key"]), **pinned}
    if own.get("api_key"):
        return {"base_url": str(own.get("base_url") or DEFAULT_BASE_URL), "api_key": str(own["api_key"]),
                **pinned}
    return {"base_url": os.environ.get("CLAUDE_URL", DEFAULT_BASE_URL),
            "api_key": os.environ.get("CLAUDE_API_KEY", ""), **pinned}


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
                # one. That is not the editor anyone chose; never cut on it. The failover is
                # often transient ("Couldn't confirm model ... Try again", 2026-10-09), so
                # one more attempt is made before giving up.
                last_error = RuntimeError(f"proxy served {served!r} instead of {self.model!r}")
                logger.warning("Editor call (attempt %d): %s", attempt + 1, last_error)
                if attempt < retries:
                    time.sleep(3.0)
                continue
            meta = {"model": served, "usage": out.get("usage") or {},
                    "seconds": round(time.time() - started, 1)}
            return out["choices"][0]["message"]["content"], meta
        if isinstance(last_error, RuntimeError):
            raise last_error
        raise RuntimeError(f"editor unavailable: {last_error}")


def edit(utterances: Sequence[Any], words: Sequence[Dict[str, Any]], client: EditorClient,
         style: str = "", speaker_style: str = "") -> Tuple[Decisions, Dict[str, Any]]:
    """One whole-recording editing pass: ask, validate. Returns (decisions, meta)."""
    from .utterances import table
    text, meta = client.ask(SYSTEM, request(table(utterances), style, speaker_style))
    decisions = parse(text, utterances, words)
    meta["answer"] = text
    if decisions.refused:
        logger.info("Editor: %d parts of the answer refused: %s", len(decisions.refused),
                    "; ".join(decisions.refused[:5]))
    return decisions, meta
