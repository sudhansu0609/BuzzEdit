import json
import logging
import re
import time
import httpx
from typing import List, Dict, Any, Optional
from runtime.gpu_broker import gpu_broker
from .lm_launcher import ensure_ready as ensure_lm_ready, forget_resolution

logger = logging.getLogger("lm_studio_client")

# `/no_think` turns off the step-by-step "reasoning" of thinking models
# (Qwen3, etc.) for this request. Prepended to the system prompt of the editing
# passes: those need a direct answer, not a chain of thought, and a thinking
# model can otherwise spend its whole token budget reasoning and return an empty
# answer — see is_responsive() for how we detect and route around that.
_NO_THINK = "/no_think\n"

# Cache of the last responsiveness probe, so the many passes in one auto-edit
# don't each pay for it. Short TTL: a model can be swapped out under us.
_probe_ok: Optional[bool] = None
_probe_until = 0.0
_PROBE_TTL = 45.0


def _reasoning_effort() -> str:
    """How hard a 'thinking' model should reason on the editing passes.

    Default "none": these are mechanical text edits (remove fumbles, judge a
    sentence), and a thinking model left to reason freely burns thousands of
    tokens — minutes — per call and often overruns into an empty answer. With
    "none" it answers directly, so a 27B thinking model runs as fast as an
    instruct model. Raise to "low"/"medium"/"high" via the `llm_reasoning_effort`
    setting to trade speed for deeper reasoning.
    """
    try:
        from store.app_settings import AppSettings
        value = AppSettings().get("llm_reasoning_effort")
        return str(value) if value else "none"
    except Exception:
        return "none"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_PREAMBLE_RE = re.compile(
    r"^\s*(here\s+is|here's|cleaned\s+transcript|sure[,!]?)[^\n:]*:\s*",
    re.IGNORECASE)


def _strip_wrapping(content: str) -> str:
    """Peel off a code fence or a "Here is the cleaned transcript:" preamble.

    Alignment would survive these — stray words simply fail to match — but they
    inflate the window and push the real text out of shape, so they go first.
    """
    text = _FENCE_RE.sub("", str(content or "")).strip()
    text = _PREAMBLE_RE.sub("", text, count=1)
    return text.strip()


def _first_json_object(content: str) -> Optional[Dict[str, Any]]:
    """Parse the model's answer, tolerating chat-model padding around the JSON.

    A model told to answer in JSON still opens with "Sure!" or wraps the object
    in a ```json fence often enough that a strict json.loads throws the whole
    adjudication away.
    """
    if not content:
        return None
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None

# The default is "local auto": use whichever model the local LM Studio already
# has loaded (or the first one it can load), so the app never depends on a
# specific model being downloaded. A user who wants a particular model — the
# fluency pass genuinely benefits from a larger one; a 7.5B tends to echo the
# transcript back where a 26B removes real fumbles — can pin it in Preferences.
AUTO_MODEL = "auto"
# Kept only as a hint for users choosing a capable model; no longer forced.
DEFAULT_MODEL = "gemma-4-26b-a4b-it-ultra-uncensored-heretic"


class LMStudioClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1",
                 model_name: Optional[str] = None):
        self.base_url = base_url
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """The model to send requests with, re-read each time so a settings
        change lands without a restart.

        An empty string means "local auto": the launcher then uses whichever
        model is already loaded in LM Studio, or loads the first available one.
        This is the default so nothing depends on a specific model being present.
        """
        if self._model_name:
            return self._model_name
        try:
            from store.app_settings import AppSettings
            choice = AppSettings().get("llm_model")
        except Exception:
            choice = None
        if not choice or choice == AUTO_MODEL:
            return ""   # auto: resolver picks the loaded/available local model
        return choice

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        self._model_name = value

    async def is_responsive(self, timeout: float = 120.0, probe_tokens: int = 512) -> bool:
        """Confirm the configured model is loaded and actually answering.

        The gate exists so a genuinely dead/misconfigured model falls back to the
        deterministic edit instead of hanging forever. It is NOT meant to reject a
        slow *thinking* model (Qwen3 etc.) — those are a deliberate choice for
        better cuts, and they answer through a `reasoning_content` field before the
        real `content`. So the model counts as alive if it returns EITHER content
        or reasoning within the (generous) timeout; only a timeout or an outright
        error/empty response drops us to the deterministic path. Cached briefly so
        one edit probes once, not once per pass.
        """
        global _probe_ok, _probe_until
        if _probe_ok is not None and time.monotonic() < _probe_until:
            return _probe_ok

        ok = False
        try:
            model = await ensure_lm_ready(self.base_url, self.model_name)
            if model:
                # A miniature version of the real job, not "Say READY": a model
                # can greet perfectly and still be unable to edit a transcript,
                # and that failure only used to surface after minutes of burned
                # windows. Ten words, one obvious filler.
                probe_text = "dosto aap uh uh sab log mujhe hi dekh rahe hain"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": (
                                    "Remove filler sounds like 'uh' from the transcript. "
                                    "Reply with the cleaned transcript only, otherwise "
                                    "unchanged.")},
                                {"role": "user", "content": probe_text},
                            ],
                            "temperature": 0.0,
                            "max_tokens": probe_tokens,
                            "reasoning_effort": _reasoning_effort(),
                        },
                    )
                if resp.status_code == 200:
                    msg = resp.json()["choices"][0]["message"]
                    content = (msg.get("content") or "").strip().lower()
                    reasoning = (msg.get("reasoning_content") or "").strip()
                    if content:
                        # Engaged = the answer is still the transcript (not a
                        # refusal, a translation or chatter about the task).
                        ok = sum(w in content for w in ("dosto", "log", "dekh", "rahe")) >= 2
                        if not ok:
                            logger.warning("LLM probe answered off-task (%r); "
                                           "skipping LLM passes.", content[:120])
                    else:
                        # A thinking model may spend the probe budget reasoning and
                        # return no content yet. That is alive, not incompetent —
                        # rejecting it here would throw away the better editor.
                        ok = bool(reasoning)
                        if not ok:
                            logger.warning("LLM probe returned nothing at all; skipping LLM passes.")
                else:
                    logger.warning("LLM probe got HTTP %s; skipping LLM passes.", resp.status_code)
        except Exception as e:
            logger.warning("LLM probe failed/timed out (%s); using the deterministic edit.", e)
            ok = False

        _probe_ok = ok
        _probe_until = time.monotonic() + _PROBE_TTL
        return ok

    async def adjudicate_cut_decisions(
        self,
        words: List[Dict[str, Any]],
        aggressiveness: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Send transcript words to LM Studio for intelligent cut adjudication.
        Returns list of decisions: [{"word_id": "w_0", "keep": bool, "reason": str}]
        """
        if not words:
            return []

        await gpu_broker.acquire_lease("lm_studio", required_vram_mb=7000.0)
        try:
            if not await ensure_lm_ready(self.base_url, self.model_name):
                logger.warning("LM Studio not ready; using default disfluency rule.")
                return [{"word_id": w["id"], "keep": not w.get("disfluency", False)} for w in words]

            # Prepare compact representation for LLM prompt
            prompt_items = [
                {"id": w.get("id"), "text": w.get("text"), "disfluency": w.get("disfluency", False)}
                for w in words
            ]

            system_prompt = (
                "You are an expert video editor AI. Analyze transcript words and decide which filler, "
                "repeated, or redundant words should be cut (keep=false) or preserved (keep=true). "
                "Return a valid JSON array of objects with fields: 'word_id', 'keep', 'reason'."
            )

            user_prompt = f"Transcript items:\n{json.dumps(prompt_items[:100])}\nAggressiveness: {aggressiveness}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"}
                    }
                )

                if resp.status_code == 200:
                    result = resp.json()
                    content = result["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "decisions" in parsed:
                        return parsed["decisions"]

            logger.warning("LM Studio call returned non-200 or unexpected structure. Fallback to default disfluency rule.")
            return [{"word_id": w["id"], "keep": not w.get("disfluency", False)} for w in words]
        except Exception as e:
            logger.warning(f"LM Studio connection/inference error: {e}. Fallback active.")
            return [{"word_id": w["id"], "keep": not w.get("disfluency", False)} for w in words]
        finally:
            await gpu_broker.release_lease("lm_studio")

    async def adjudicate_disfluencies(
        self,
        words: List[Dict[str, Any]],
        candidate_indices: List[int],
        aggressiveness: float = 0.5,
    ) -> Optional[set]:
        """Context-aware filler/false-start adjudication.

        Given the full transcript and the indices flagged as ambiguous cut
        candidates, return the SUBSET of indices that should actually be cut.
        Returns None if the LLM is unavailable/invalid so the caller can apply
        its deterministic fail-safe (never guess destructively).
        """
        if not words or not candidate_indices:
            return set()

        await gpu_broker.acquire_lease("lm_studio", required_vram_mb=7000.0)
        try:
            # Auto-start LM Studio + load a model if needed; bail to fail-safe if
            # nothing will load. `ensure_ready` answers with the model that is
            # actually serving, which is not always the preferred one.
            model = await ensure_lm_ready(self.base_url, self.model_name)
            if not model:
                logger.warning("LM Studio not ready; skipping LLM adjudication (deterministic fallback).")
                return None

            to_cut: set = set()
            # Process in windows so long transcripts stay within context limits,
            # with overlap on each side for context.
            WINDOW, PAD = 220, 30
            cand_set = set(candidate_indices)
            i = 0
            n = len(words)
            any_success = False
            while i < n:
                lo = max(0, i - PAD)
                hi = min(n, i + WINDOW + PAD)
                window = words[lo:hi]
                win_cands = [idx for idx in range(lo, hi) if idx in cand_set]
                if win_cands:
                    numbered = " ".join(
                        f"[{idx}]{'*' if idx in cand_set else ''}{self._safe(words[idx])}"
                        for idx in range(lo, hi)
                    )
                    decided = await self._adjudicate_window(numbered, win_cands, aggressiveness, model)
                    if decided is not None:
                        any_success = True
                        to_cut |= (decided & cand_set)
                i += WINDOW

            if not any_success:
                # Every window failed: whatever is wrong will still be wrong in
                # 30 seconds, so drop the cached model and re-resolve next time.
                forget_resolution()
                logger.warning("LLM adjudication produced nothing usable with %r; "
                               "keeping the deterministic edit.", model)
                return None
            return to_cut
        except Exception as e:
            logger.warning(f"LLM disfluency adjudication failed: {e}")
            return None
        finally:
            await gpu_broker.release_lease("lm_studio")

    async def clean_transcript(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """One fluency pass: hand the model spoken text, get back how it reads.

        Plain text, not JSON — the answer *is* the transcript, and forcing a
        schema on a few hundred words of Hinglish only gives a small model more
        ways to fail. The caller aligns the answer back onto the real tokens, so
        anything the model invents simply fails to match and cuts nothing.
        """
        await gpu_broker.acquire_lease("lm_studio", required_vram_mb=7000.0)
        try:
            model = await ensure_lm_ready(self.base_url, self.model_name)
            if not model:
                logger.warning("LM Studio not ready; skipping the fluency pass.")
                return None
            # The answer is a few hundred words at most, so 4k is ample. A bigger
            # cap is actively harmful: a "thinking" model treats the whole budget
            # as room to reason and will burn 8k+ tokens (minutes) per call, then
            # often overrun and return empty `content` — which is exactly the
            # "no answer, 30-40 min" failure. A non-thinking instruct model answers
            # directly well under this. Keep it tight.
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 4096,
                        "reasoning_effort": _reasoning_effort(),
                    },
                )
            if resp.status_code != 200:
                logger.warning("LM Studio %s during the fluency pass: %s",
                               resp.status_code, resp.text[:200])
                forget_resolution()
                return None
            # A thinking model puts its chain-of-thought in `reasoning_content`;
            # `content` is the actual answer, so this already excludes the thinking.
            content = resp.json()["choices"][0]["message"].get("content") or ""
            return _strip_wrapping(content)
        except Exception as e:
            logger.warning(f"Fluency pass failed: {e}")
            return None
        finally:
            await gpu_broker.release_lease("lm_studio")

    async def ask_with_schema(self, system_prompt: str, user_prompt: str,
                              schema: Dict[str, Any]) -> Optional[str]:
        """One completion constrained to `schema` (LM Studio json_schema format).

        This is what makes the span contract land on a small local model: asked
        in prose for DELETE lines, the available 12B ignored the format on every
        real window and rewrote the transcript instead; constrained, it answers
        in span form every time. `_chat_json` already falls back to a plain
        request when the server refuses the schema, and the caller treats a
        non-JSON answer as "ask in prose instead", so nothing here is fatal.
        """
        await gpu_broker.acquire_lease("lm_studio", required_vram_mb=7000.0)
        try:
            model = await ensure_lm_ready(self.base_url, self.model_name)
            if not model:
                logger.warning("LM Studio not ready; skipping the schema-constrained ask.")
                return None
            return await self._chat_json(model, system_prompt, user_prompt, schema=schema)
        except Exception as e:
            logger.warning(f"Schema-constrained ask failed: {e}")
            return None
        finally:
            await gpu_broker.release_lease("lm_studio")

    @staticmethod
    def _safe(w: Dict[str, Any]) -> str:
        # Judge on the native Devanagari when the word carries one, so the model
        # reads real Hindi rather than its romanization. See asr.fluency._token.
        text = w.get("word_native") or w.get("word") or w.get("text", "")
        return str(text).strip().replace(" ", "_") or "_"

    async def _adjudicate_window(self, numbered_text: str, candidates: List[int],
                                 aggressiveness: float, model: Optional[str] = None):
        # The hard judgement here is a restart versus a deliberate repetition.
        # Structure alone cannot separate "dosto kya ap · dosto kya apko pata hai"
        # (an abandoned attempt) from "it is what it is" (a set phrase) — both are
        # a phrase that recurs a moment later. Reading the meaning is exactly what
        # the model is for, so the prompt spells that distinction out and gives it
        # the rule for which copy survives.
        system_prompt = (
            "You are a precise video editor cleaning a speech transcript. Tokens are "
            "shown as [index]word; candidates for removal are marked with '*'. The speech "
            "is Hindi in its native Devanagari script, with English words mixed in (or, "
            "occasionally, wholly English). Judge it as Hindi, applying Hindi grammar. "
            "Decide which CANDIDATE indices are true disfluencies to delete.\n"
            "Delete: filler words (um, uh, like, matlab, yaani, basically, you know), "
            "stutters, and ABANDONED ATTEMPTS.\n"
            "An abandoned attempt is where the speaker starts a sentence, breaks off, and "
            "starts it again — often several times — before finally completing it. Example: "
            "'dosto kya ap | dosto kya | dosto kya a | dosto | dosto kya apko pata hai india "
            "me...' Here every attempt before the last is abandoned; delete them all and keep "
            "ONLY the final complete run. Always keep the LAST, most complete take.\n"
            "Do NOT delete a phrase the speaker repeated on purpose: set phrases ('it is what "
            "it is'), emphasis ('bahut bahut accha', 'this is big, this is huge'), or a point "
            "restated for effect. If the sentence reads as complete and fluent, it is not a "
            "fumble.\n"
            "NEVER delete meaningful content words even if marked. Only return indices from the "
            "candidate list. Respond ONLY as JSON: {\"cut\": [<indices>]}."
        )
        user_prompt = (
            f"Aggressiveness (0=minimal, 1=aggressive): {aggressiveness}\n"
            f"Candidate indices: {candidates}\n"
            f"Transcript:\n{numbered_text}"
        )
        content = await self._chat_json(
            model or self.model_name, system_prompt, user_prompt,
            schema=self.CUT_SCHEMA)
        if content is None:
            return None
        parsed = _first_json_object(content)
        if parsed is None:
            logger.warning("LLM answer was not JSON: %s", content[:160])
            return None
        cut = parsed.get("cut", []) if isinstance(parsed, dict) else []
        return {int(x) for x in cut if isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit()}

    # The shape the disfluency adjudicator asks for. It used to be hardcoded
    # inside `_chat_json`, which meant every *other* caller silently got their
    # answer forced into `{"cut": [...]}` — a 200 response of entirely the wrong
    # shape, which looks like the model failing rather than the client.
    CUT_SCHEMA = {
        "name": "cut_decisions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"cut": {"type": "array", "items": {"type": "integer"}}},
            "required": ["cut"],
            "additionalProperties": False,
        },
    }

    async def _chat_json(self, model: str, system_prompt: str,
                         user_prompt: str,
                         schema: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """One completion asking for JSON, tolerant of what the server supports.

        LM Studio's accepted `response_format` has moved: this build rejects
        `{"type": "json_object"}` outright with a 400, which — being swallowed —
        was the second reason the adjudication layer silently never ran. Ask for a
        schema, and fall back to a plain request if the server will not take one.

        `schema` is the answer's shape. Pass `None` to ask in the prompt alone,
        which is the right choice when the shape is too rich to express strictly.
        """
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            # Short answer; a tight cap keeps a thinking model from burning minutes
            # on a runaway reasoning chain (see clean_transcript).
            "max_tokens": 4096,
            "reasoning_effort": _reasoning_effort(),
        }
        formats = [
            {"type": "json_schema", "json_schema": schema} if schema else None,
            None,        # last resort: the prompt already specifies the shape
        ]
        async with httpx.AsyncClient(timeout=180.0) as client:
            for response_format in formats:
                payload = dict(body)
                if response_format:
                    payload["response_format"] = response_format
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"].get("content") or ""
                # Silence here is what hid all of this: a 400 saying the model
                # will not load, or that the response format is wrong, used to
                # end the layer without a word in the log.
                logger.warning("LM Studio %s for model %r (%s): %s",
                               resp.status_code, model,
                               "schema" if response_format else "plain",
                               resp.text[:200])
                if resp.status_code != 400:
                    return None
        return None


lm_studio_client = LMStudioClient()
