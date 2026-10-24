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


# --- the local editor --------------------------------------------------------
#
# 2026-10-21: the creator's rule is a local model by default, Claude only when
# the local one cannot answer (and the other way round when Claude is the
# default and hits its usage limit -- that fallback used to be the classic
# planner, which cut 674aefab to the bone). One model in memory at a time: a
# copy already loaded, in Ollama or LM Studio, is used as it is; otherwise
# everything is ejected and the model is loaded into LM Studio at the context
# the whole-recording reads need.
DEFAULT_PROVIDER = "local"                # "local" | "claude": which editor is tried first
DEFAULT_LOCAL_MODEL = "google/gemma-4-12b"
LM_STUDIO_URL = "http://127.0.0.1:1234"
OLLAMA_URL = "http://127.0.0.1:11434"
LOCAL_CONTEXT = 32768                     # a 40-minute take's utterance table is ~20K tokens
LOCAL_MIN_CONTEXT = 16384                 # an Ollama copy loaded smaller than this is not reused
LOCAL_ANSWER_TOKENS = 8000                # room kept for the decisions JSON of one part
PROXY_INPUT_TOKENS = 150_000              # Opus reads a whole take in one go
PROBE_TIMEOUT_S = 240.0                   # a cold model's first short answer, prompt cache empty


def _own_settings() -> Dict[str, Any]:
    try:
        from store.app_settings import AppSettings
        return dict(AppSettings().get("auto_cut_editor") or {})
    except Exception:
        return {}


def editor_provider(settings: Optional[Dict[str, Any]] = None) -> str:
    import os
    value = str((settings or {}).get("editor_provider") or _own_settings().get("provider")
                or os.environ.get("CLAUDE_EDITOR_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    return value if value in ("local", "claude") else DEFAULT_PROVIDER


def model_key(name: str) -> str:
    """'google/gemma-4-12b', 'gemma-4-12b:latest' and 'gemma4:12b' are one model."""
    import re
    s = str(name or "").strip().lower()
    s = s.split("/")[-1]
    s = re.sub(r":latest$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _get_json(url: str, timeout: float = 2.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_loaded(base_url: str = OLLAMA_URL) -> List[Dict[str, Any]]:
    """[{name, context_length}] for every model Ollama holds in memory."""
    try:
        data = _get_json(f"{base_url}/api/ps")
    except Exception:
        return []
    return [{"name": str(m.get("name") or m.get("model") or ""),
             "context_length": int(m.get("context_length") or 0)}
            for m in (data.get("models") or [])]


def lm_studio_loaded(base_url: str = LM_STUDIO_URL) -> List[Dict[str, Any]]:
    """[{id, context_length}] for the models LM Studio holds in memory."""
    try:
        data = _get_json(f"{base_url}/api/v0/models")
    except Exception:
        return []
    return [{"id": str(m.get("id") or ""), "context_length": int(m.get("loaded_context_length") or 0)}
            for m in (data.get("data") or []) if m.get("state") == "loaded"]


def local_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Where the local editor runs: {backend, base_url, model, context, load}.

    A copy of the wanted model already in memory is used as it is when it was loaded
    with enough context -- Ollama's (LOCAL_MIN_CONTEXT) or LM Studio's (LOCAL_CONTEXT).
    Otherwise `load` is True and `ensure_local_loaded` ejects everything and loads it
    into LM Studio at LOCAL_CONTEXT. LM Studio's own just-in-time load is 16k, and an
    editor reading 24k-token parts on it got "Context size has been exceeded" for every
    read (674aefab, 2026-10-21), so a small copy is reloaded, not reused.
    """
    import os
    own = _own_settings()
    wanted = str((settings or {}).get("editor_local_model") or own.get("local_model")
                 or os.environ.get("LOCAL_EDITOR_MODEL") or DEFAULT_LOCAL_MODEL)
    key = model_key(wanted)
    for m in ollama_loaded():
        if model_key(m["name"]) == key and m["context_length"] >= LOCAL_MIN_CONTEXT:
            return {"settings": settings, "backend": "ollama", "base_url": OLLAMA_URL, "model": m["name"],
                    "context": m["context_length"], "load": False}
    for m in lm_studio_loaded():
        if model_key(m["id"]) == key and m["context_length"] >= LOCAL_CONTEXT:
            return {"settings": settings, "backend": "lm_studio", "base_url": LM_STUDIO_URL + "/v1", "model": m["id"],
                    "context": m["context_length"], "load": False}
    return {"settings": settings, "backend": "lm_studio", "base_url": LM_STUDIO_URL + "/v1", "model": wanted,
            "context": LOCAL_CONTEXT, "load": True}


def _unload_ollama(base_url: str = OLLAMA_URL) -> None:
    for m in ollama_loaded(base_url):
        try:
            req = urllib.request.Request(f"{base_url}/api/generate", method="POST",
                                         data=json.dumps({"model": m["name"], "keep_alive": 0}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:  # noqa: BLE001 - best effort; the load below reports the real failure
            logger.info("Could not unload Ollama model %s (%s)", m["name"], e)


def _free_card_for_the_editor() -> None:
    """Everything else off the card before the editor model loads: this
    process's Whisper and aligner weights (resident after the transcription
    that always precedes a cut), and an idle ComfyUI's cached models (the
    previous video's Wan experts stay in VRAM after its render). A 12B model
    with a 32k KV cache beside those is streamed from RAM and reads slowly."""
    try:
        from runtime import gpu_handover
        gpu_handover.release_in_process_models()
    except Exception as e:  # noqa: BLE001 - best effort
        logger.debug("Could not release the in-process ASR models: %s", e)
    try:
        from config import COMFYUI_URL
        base = COMFYUI_URL.rstrip("/")
        with urllib.request.urlopen(f"{base}/queue", timeout=3) as r:
            q = json.loads(r.read().decode("utf-8"))
        if q.get("queue_running") or q.get("queue_pending"):
            logger.info("ComfyUI is generating; its models stay while the editor loads")
            return
        req = urllib.request.Request(f"{base}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10).close()
        logger.info("Asked ComfyUI to unload its models before the editor model loads")
    except Exception as e:  # noqa: BLE001 - ComfyUI down or absent is fine
        logger.debug("ComfyUI free skipped: %s", e)


def ensure_local_loaded(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Make the local model answer: reuse the loaded copy, or eject every other model and
    load it into LM Studio at LOCAL_CONTEXT. Raises RuntimeError when it cannot."""
    import subprocess
    if not cfg.get("load"):
        return cfg
    from llm.lm_launcher import LOAD_LOCK, find_lms_cli
    lms = find_lms_cli()
    if not lms:
        raise RuntimeError("LM Studio's lms CLI was not found, so the local editor model cannot be loaded")
    with LOAD_LOCK:
        # Another loader (the genre pass, the Studio's planner) may have put the
        # model up while we waited for the lock: reuse it rather than eject it.
        again = local_config(cfg.get("settings"))
        if not again.get("load") and int(again.get("context") or 0) >= LOCAL_MIN_CONTEXT:
            logger.info("Local editor: %s is already loaded (by another loader); reusing it", again.get("model"))
            return {**cfg, **again}
        _free_card_for_the_editor()
        _unload_ollama()
        flags = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        subprocess.run([lms, "unload", "--all"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, **flags)
        proc = subprocess.run([lms, "load", cfg["model"], "-c", str(LOCAL_CONTEXT), "-y", "--gpu", "max"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, **flags)
        loaded = [m for m in lm_studio_loaded() if model_key(m["id"]) == model_key(cfg["model"])]
    if not loaded:
        raise RuntimeError(f"lms load {cfg['model']} failed: {(proc.stderr or proc.stdout or '').strip()[:300]}")
    context = loaded[0]["context_length"] or LOCAL_CONTEXT
    logger.info("Local editor: loaded %s into LM Studio at %d context", loaded[0]["id"], context)
    return {**cfg, "model": loaded[0]["id"], "context": context, "load": False}


def editor_clients(settings: Optional[Dict[str, Any]] = None) -> List[Tuple[str, "EditorClient"]]:
    """The editors to try, in order: the default provider, then the other one.

    Claude is only offered when a proxy key exists; the local editor is always offered
    (its model is loaded on first use, see `EditorClient.prepare`)."""
    proxy = proxy_config(settings)
    claude = ("claude", EditorClient(base_url=proxy["base_url"], api_key=proxy["api_key"],
                                     model=proxy["model"], effort=proxy["effort"])) if proxy["api_key"] else None
    local = ("local", EditorClient.local(settings))
    order = [local, claude] if editor_provider(settings) == "local" else [claude, local]
    return [c for c in order if c is not None]


class EditorClient:
    """OpenAI-compatible chat call to the proxy, pinned to one model and one effort level."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "", model: str = DEFAULT_MODEL,
                 effort: str = DEFAULT_EFFORT, timeout: float = 1800.0, backend: str = "proxy",
                 local_settings: Optional[Dict[str, Any]] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.backend = backend            # "proxy" | "lm_studio" | "ollama"
        self.context = LOCAL_CONTEXT      # the local model's loaded context; `prepare` sets it
        self._local_settings = local_settings
        self._prepared = backend == "proxy"

    @classmethod
    def local(cls, settings: Optional[Dict[str, Any]] = None) -> "EditorClient":
        """The local editor; which copy it talks to is decided in `prepare()`, at call time."""
        cfg = local_config(settings)
        client = cls(base_url=cfg["base_url"], model=cfg["model"], effort="", backend=cfg["backend"],
                     local_settings=settings)
        client.context = int(cfg.get("context") or LOCAL_CONTEXT)
        return client

    def prepare(self) -> None:
        """Have the model answering before the reads: a loaded copy with enough context as
        it is, else (re)loaded now."""
        if self._prepared:
            return
        cfg = ensure_local_loaded(local_config(self._local_settings))
        self.base_url, self.model, self.backend = cfg["base_url"].rstrip("/"), cfg["model"], cfg["backend"]
        self.context = int(cfg.get("context") or LOCAL_CONTEXT)
        if self.context < LOCAL_MIN_CONTEXT:
            raise RuntimeError(f"{self.backend} loaded {self.model} with {self.context} tokens of context; "
                               f"the editor needs {LOCAL_MIN_CONTEXT}")
        self.probe()
        self._prepared = True

    def probe(self) -> None:
        """One tiny read, made exactly as a job makes its reads, before any take is cut
        on this model. A copy that answers empty (thinking left on), without JSON, at
        the wrong context or not at all is found here and the planner moves on to the
        next editor -- the gate `tools/editor_smoke.py` used to be run for by hand
        (2026-10-21: a model loaded at 16k and a thinking model answering nothing had
        each passed every unit test). Raises with the reason on failure."""
        table = ("1  0.0-2.1  hello doston mera naam hai\n"
                 "2  2.1-4.4  hello doston mera naam hai sudhansu aur aap dekh rahe hain raat 3 baje\n"
                 "3  4.4-6.9  aaj ki kahani shuru karte hain\n")
        timeout, self.timeout = self.timeout, min(self.timeout, PROBE_TIMEOUT_S)
        try:
            answer, meta = self.ask(SYSTEM, request(table), retries=0)
        finally:
            self.timeout = timeout
        if first_json(answer) is None:
            raise RuntimeError(f"{self.backend} {self.model} answered the probe without JSON "
                               f"({answer.strip()[:120]!r})")
        logger.info("Editor probe: %s %s (context %d) answered in %ss", self.backend,
                    meta.get("model") or self.model, self.context, meta.get("seconds"))

    def _request(self, system: str, user: str) -> urllib.request.Request:
        if self.backend != "proxy":
            # A local thinking model (Gemma 4, Qwen 3) left to reason spends its whole
            # answer on it and returns empty content (the smoke check found exactly that,
            # 2026-10-21). The editing read wants the decisions, not a chain of thought:
            # the same switches llm/client.py uses for BuzzEdit's other local passes.
            system = "/no_think\n" + system
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        headers = {"Content-Type": "application/json"}
        if self.backend == "ollama":
            # Ollama's OpenAI endpoint ignores the context size; its own API takes it.
            body: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "think": False,
                                    "options": {"num_ctx": int(self.context), "num_predict": LOCAL_ANSWER_TOKENS}}
            return urllib.request.Request(self.base_url + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                          method="POST", headers=headers)
        body = {"model": self.model, "messages": messages,
                "max_tokens": 64000 if self.backend == "proxy" else LOCAL_ANSWER_TOKENS,
                "reasoning_effort": self.effort if self.backend == "proxy" else "none"}
        if not body["reasoning_effort"]:
            del body["reasoning_effort"]
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                      method="POST", headers=headers)

    def input_budget(self) -> int:
        """Tokens one read's prompt may take; `edit` splits the table into parts above it."""
        if self.backend == "proxy":
            return PROXY_INPUT_TOKENS
        return max(2048, int(self.context) - LOCAL_ANSWER_TOKENS - 512)

    def ask(self, system: str, user: str, retries: int = 1) -> Tuple[str, Dict[str, Any]]:
        """(answer text, meta with served model, usage and seconds). Raises on failure."""
        last_error: Optional[Exception] = None
        reloaded = False
        attempt = -1
        while attempt < retries:
            attempt += 1
            req = self._request(system, user)
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # The body says why ("exceeds the available context size", a bad key): keep it.
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    detail = ""
                last_error = RuntimeError(f"HTTP {e.code}: {detail or e.reason}")
                logger.warning("Editor call failed (attempt %d): %s", attempt + 1, last_error)
                if "unloaded" in detail.lower() and self.backend != "proxy" and not reloaded:
                    # Something ejected the model under us (2026-10-21: a second
                    # loader ran `lms unload --all`). Put it back and read again.
                    reloaded = True
                    try:
                        self._prepared = False
                        self.prepare()
                        retries += 1
                    except Exception as prep:  # noqa: BLE001
                        logger.warning("Editor: could not reload the model after it was unloaded: %s", prep)
                continue
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("Editor call failed (attempt %d): %s", attempt + 1, e)
                continue
            served = str(out.get("model") or "")
            if not model_key(served).startswith(model_key(self.model)):
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
                    "seconds": round(time.time() - started, 1), "backend": self.backend}
            if self.backend == "ollama":
                return str((out.get("message") or {}).get("content") or ""), meta
            message = out["choices"][0]["message"]
            content = str(message.get("content") or "")
            if not content.strip() and message.get("reasoning_content"):
                # The answer ended up in the thinking channel: take it from there.
                content = str(message["reasoning_content"])
            return content, meta
        if isinstance(last_error, RuntimeError):
            raise last_error
        raise RuntimeError(f"editor unavailable: {last_error}")


# --- reading a long recording in parts ---------------------------------------
#
# A whole-recording read is one call. A local model has LOCAL_CONTEXT tokens and
# a 42-minute take's utterance table is more than that, so the table is read in
# parts that fit, each shown the last PART_OVERLAP lines of the one before it (a
# retake straddling the cut is still seen whole), and the decisions are merged.
# Retakes further apart than a part are the retake sweep's job (it reads
# similarity groups, not the table). A part that still does not fit -- the token
# estimate is only an estimate -- is read in two halves.
PART_OVERLAP = 24
MIN_PART = 12
CHARS_PER_TOKEN = 2.2      # Hinglish with Devanagari on a Gemma/Qwen tokenizer, conservative


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def parts_for(utterances: Sequence[Any], budget: int, style: str = "",
              speaker_style: str = "") -> List[Tuple[int, int, int]]:
    """(shown_from, own_from, to) per part, `to` exclusive: the part shows
    utterances[shown_from:to], of which [own_from:to] are new. One part when it all fits."""
    from .utterances import table
    room = budget - estimate_tokens(SYSTEM) - estimate_tokens(request("", style, speaker_style))
    sizes = [estimate_tokens(table([u])) + 1 for u in utterances]
    n = len(sizes)
    if n == 0 or sum(sizes) <= room:
        return [(0, 0, n)]
    parts: List[Tuple[int, int, int]] = []
    start = 0
    while start < n:
        shown_from = max(0, start - PART_OVERLAP) if parts else 0
        total = sum(sizes[shown_from:start])
        end = start
        while end < n and total + sizes[end] <= room:
            total += sizes[end]
            end += 1
        if end - start < MIN_PART:         # too little to judge anything: take MIN_PART anyway
            end = min(n, start + MIN_PART)
        parts.append((shown_from, start, end))
        start = end
    return parts


def merge_decisions(into: Decisions, part: Decisions) -> Decisions:
    into.removed_words.update(part.removed_words)
    into.removed_spans.extend(s for s in part.removed_spans if s not in into.removed_spans)
    into.refused.extend(part.refused)
    into.groups += part.groups
    into.drops += part.drops
    into.trims += part.trims
    into.trim_ranges.extend(t for t in part.trim_ranges if t not in into.trim_ranges)
    return into


def _context_overflow(error: BaseException) -> bool:
    return re.search(r"context|too many tokens|exceeds", str(error), re.IGNORECASE) is not None


MAX_HALVINGS = 2   # a part that still overflows after two halvings is not a size problem


def _edit_part(shown: Sequence[Any], words: Sequence[Dict[str, Any]], client: Any,
               style: str, speaker_style: str, halvings: int = 0) -> Tuple[Decisions, Dict[str, Any]]:
    from .utterances import table
    try:
        text, meta = client.ask(SYSTEM, request(table(shown), style, speaker_style))
    except Exception as e:
        if len(shown) < 2 * MIN_PART or halvings >= MAX_HALVINGS or not _context_overflow(e):
            # Not the part's size: the model's loaded context is too small or shared
            # (674aefab: 672 failed calls halving 219 lines down to 24). Let the read fail
            # so the next editor takes over instead of grinding on.
            raise
        logger.warning("Editor: a part of %d utterances did not fit the context (%s); reading it in two",
                       len(shown), str(e)[:120])
        mid = len(shown) // 2
        first, meta = _edit_part(shown[:mid + PART_OVERLAP // 2], words, client, style, speaker_style,
                                 halvings + 1)
        second, more = _edit_part(shown[max(0, mid - PART_OVERLAP // 2):], words, client, style,
                                  speaker_style, halvings + 1)
        meta["answer"] = meta.get("answer", "") + "\n\n" + more.get("answer", "")
        meta["seconds"] = round(float(meta.get("seconds") or 0) + float(more.get("seconds") or 0), 1)
        return merge_decisions(first, second), meta
    decisions = parse(text, shown, words)
    meta["answer"] = text
    return decisions, meta


def edit(utterances: Sequence[Any], words: Sequence[Dict[str, Any]], client: EditorClient,
         style: str = "", speaker_style: str = "") -> Tuple[Decisions, Dict[str, Any]]:
    """One whole-recording editing pass: ask, validate. Returns (decisions, meta).

    Read in parts (`parts_for`) when the table does not fit the model's context."""
    budget = getattr(client, "input_budget", lambda: PROXY_INPUT_TOKENS)()
    parts = parts_for(utterances, budget, style, speaker_style)
    if len(parts) == 1:
        decisions, meta = _edit_part(utterances, words, client, style, speaker_style)
    else:
        logger.info("Editor: %d utterances read in %d parts (context budget %d tokens, %s)",
                    len(utterances), len(parts), budget, getattr(client, "model", "?"))
        decisions, meta = Decisions(), {"usage": {}, "seconds": 0.0, "parts": len(parts)}
        answers: List[str] = []
        for shown_from, _own_from, to in parts:
            part, m = _edit_part(list(utterances[shown_from:to]), words, client, style, speaker_style)
            merge_decisions(decisions, part)
            answers.append(m.pop("answer", ""))
            meta.setdefault("model", m.get("model"))
            if m.get("backend"):
                meta["backend"] = m["backend"]
            meta["seconds"] = round(meta["seconds"] + float(m.get("seconds") or 0), 1)
            for k, v in (m.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    meta["usage"][k] = meta["usage"].get(k, 0) + v
        meta["answer"] = "\n\n".join(answers)
    if decisions.refused:
        logger.info("Editor: %d parts of the answer refused: %s", len(decisions.refused),
                    "; ".join(decisions.refused[:5]))
    return decisions, meta
