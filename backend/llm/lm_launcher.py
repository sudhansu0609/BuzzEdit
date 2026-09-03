"""Auto-start LM Studio and load the model on demand.

Uses the `lms` CLI (bundled with LM Studio; run `lms bootstrap` once to put it on
PATH). Every step is best-effort and non-fatal: if the CLI is missing or a step
fails, callers fall back to the deterministic path — we never hard-depend on the
LLM being present.

Flow (`ensure_ready`):
  1. If the OpenAI-compatible server (127.0.0.1:1234) already answers, skip.
  2. Otherwise `lms server start` and poll until it's up.
  3. Load the preferred model if it is not loaded already.
  4. If it will not load, load one that will and return that instead.

`ensure_ready` returns **the model id to send requests with**, or None. That is the
point of step 4: `/v1/models` lists every model that has been *downloaded*, loaded
or not, so a model that could never load looked "ready" — the readiness check
passed, every completion came back 400 "Failed to load model", the failure was
swallowed, and the whole LLM layer quietly did nothing for weeks. Readiness is now
judged on `/api/v0/models`, which reports a real `state`, and is only ever reported
for a model that is actually loaded.
"""

import os
import asyncio
import shutil
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("lm_launcher")

# Auto-start can be disabled via env for users who manage LM Studio themselves.
_AUTOSTART = os.environ.get("LM_STUDIO_AUTOSTART", "1") not in ("0", "false", "False")

_lock = asyncio.Lock()
_ready_until = 0.0            # cache "ready" for a short window to avoid repeated CLI probes
_READY_TTL = 30.0
_resolved: Optional[str] = None   # the model id last confirmed to actually serve
# Models that failed to load this run. A broken model takes a minute to fail, and
# retrying it on every re-resolve would spend that minute over and over.
_unusable: set = set()


def find_lms_cli() -> Optional[str]:
    """Locate the `lms` CLI on PATH or in common LM Studio install locations."""
    found = shutil.which("lms") or shutil.which("lms.exe")
    if found:
        return found
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(home) / ".lmstudio" / "bin" / "lms.exe",
        Path(home) / ".lmstudio" / "bin" / "lms",
        Path(home) / ".cache" / "lm-studio" / "bin" / "lms.exe",
        Path(local) / "LM Studio" / "lms.exe",
        Path(local) / "Programs" / "lm-studio" / "lms.exe",
    ]
    for c in candidates:
        try:
            if c and c.exists():
                return str(c)
        except Exception:
            pass
    return None


async def _run(cmd: List[str], timeout: float = 60.0):
    """Run a command, returning (rc, stdout, stderr); (-1, '', reason) on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return -1, "", f"spawn failed: {e}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return proc.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -1, "", "timeout"


async def is_server_up(base_url: str, timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code == 200
    except Exception:
        return False


def _rest_root(base_url: str) -> str:
    """LM Studio's own REST API, alongside the OpenAI-compatible one."""
    return base_url.rstrip("/").rsplit("/v1", 1)[0] + "/api/v0"


async def catalogue(base_url: str) -> List[Dict[str, Any]]:
    """Every model LM Studio knows about, with its real load state.

    `/v1/models` cannot answer this: it lists what has been downloaded, not what
    is loaded. `/api/v0/models` carries `state` ("loaded" / "not-loaded") and
    `type` ("llm" / "vlm" / "embeddings").
    """
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{_rest_root(base_url)}/models")
            if r.status_code == 200:
                return [m for m in r.json().get("data", []) if isinstance(m, dict)]
    except Exception:
        pass
    return []


async def _loaded_models(base_url: str) -> List[str]:
    """Ids of models that are loaded and ready to answer right now."""
    entries = await catalogue(base_url)
    if entries:
        return [m.get("id", "") for m in entries
                if m.get("state") == "loaded" and m.get("type") != "embeddings"]
    # Older builds without /api/v0 — fall back to the OpenAI listing.
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url}/models")
            if r.status_code == 200:
                return [m.get("id", "") for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


def _model_matches(loaded: List[str], model_name: str) -> bool:
    base = model_name.split("/")[-1].lower()
    for m in loaded:
        ml = (m or "").lower()
        if model_name.lower() == ml or base in ml:
            return True
    return False


def _resolve_id(entries: List[Dict[str, Any]], model_name: str) -> Optional[str]:
    """The catalogue's own id for a model, so `lms load` and the API agree."""
    wanted = (model_name or "").lower()
    base = wanted.split("/")[-1]
    for entry in entries:
        if (entry.get("id") or "").lower() == wanted:
            return entry.get("id")
    for entry in entries:
        if base and base in (entry.get("id") or "").lower():
            return entry.get("id")
    return None


async def _unload_all(base_url: str) -> None:
    """Eject every currently-loaded model so the next load has the whole GPU.

    LM Studio keeps models resident (JIT) and will happily hold several at once,
    which on a single card is wasted VRAM and a common cause of the next load
    failing out-of-memory. Ejecting first is the memory-efficient default: one
    model in at a time.
    """
    lms = find_lms_cli()
    if not lms:
        return
    if not await _loaded_models(base_url):
        return  # nothing resident, nothing to do
    rc, out, err = await _run([lms, "unload", "--all"], timeout=30)
    if rc == 0:
        logger.info("LM Studio: ejected all resident models before loading the target.")
    else:
        logger.debug("lms unload --all rc=%s: %s", rc, (err or out or "").strip()[:120])


# What a model needs beyond its own weights: the KV cache, the compute buffers
# and the context llama.cpp allocates up front. A flat 1GB plus 15% of the
# weights matches what these GGUFs actually take at the context this pipeline
# uses, and is deliberately generous — the cost of refusing a model that would
# just have fitted is a smaller model, while the cost of accepting one that does
# not fit is the entire language-model layer silently doing nothing.
_VRAM_OVERHEAD_FRACTION = 0.15
_VRAM_OVERHEAD_MB = 1024.0


async def model_sizes() -> Dict[str, float]:
    """Every downloaded model's weight size in MB, by id. Empty if unknown.

    `/api/v0/models` does not carry size — it reports `state`, `type`, `arch` and
    context length, but nothing about how much VRAM the thing needs. `lms ls
    --json` does, under `sizeBytes`.
    """
    lms = find_lms_cli()
    if not lms:
        return {}
    rc, out, _err = await _run([lms, "ls", "--json"], timeout=20)
    if rc != 0 or not out.strip():
        return {}
    try:
        import json
        entries = json.loads(out)
    except Exception:
        return {}
    sizes: Dict[str, float] = {}
    for entry in entries if isinstance(entries, list) else []:
        key = entry.get("modelKey") or entry.get("indexedModelIdentifier")
        size = entry.get("sizeBytes")
        if key and isinstance(size, (int, float)) and size > 0:
            sizes[str(key)] = float(size) / (1024.0 * 1024.0)
    return sizes


def _fits(size_mb: Optional[float], free_mb: float) -> bool:
    """Whether a model of this weight size can actually serve in `free_mb`.

    Unknown size is treated as fitting: refusing to try a model we cannot measure
    would be worse than trying it, since a failed load is already handled.
    """
    if not size_mb:
        return True
    return size_mb * (1.0 + _VRAM_OVERHEAD_FRACTION) + _VRAM_OVERHEAD_MB <= free_mb


def _free_vram_mb() -> Optional[float]:
    try:
        from runtime.gpu_broker import gpu_broker
        return gpu_broker.get_free_vram_mb()
    except Exception:
        return None


def _total_vram_mb() -> Optional[float]:
    try:
        from runtime.gpu_broker import gpu_broker
        return gpu_broker.get_total_vram_mb()
    except Exception:
        return None


async def _try_load(base_url: str, model_id: str, load_timeout: float) -> bool:
    """Load one model, reporting whether it can actually serve afterwards."""
    if model_id in _unusable:
        return False
    lms = find_lms_cli()
    if not lms:
        return False
    rc, _out, err = await _run([lms, "load", model_id, "-y", "--gpu", "max"], timeout=load_timeout)
    if rc == 0:
        return True
    _unusable.add(model_id)
    logger.warning("LM Studio could not load %r (rc=%s): %s",
                   model_id, rc, (err or "").strip()[:160])
    return False


async def ensure_ready(base_url: str, model_name: str,
                       load_timeout: float = 240.0) -> Optional[str]:
    """Get LM Studio running with a model that works, and return that model's id.

    None means there is nothing usable, and the caller should take its
    deterministic path. A returned id is a model that is *loaded* — never one
    that merely appears in a listing.
    """
    global _ready_until, _resolved
    if _resolved and time.monotonic() < _ready_until:
        return _resolved
    if not _AUTOSTART:
        loaded = await _loaded_models(base_url) if await is_server_up(base_url) else []
        return _resolved_from(loaded, model_name)

    async with _lock:
        if _resolved and time.monotonic() < _ready_until:
            return _resolved

        # 1. Server up?
        if not await is_server_up(base_url):
            lms = find_lms_cli()
            if not lms:
                logger.warning("LM Studio not running and `lms` CLI not found; skipping auto-start. "
                               "Run `lms bootstrap` once to enable it.")
                return None
            logger.info("LM Studio server not up — starting via `lms server start`...")
            await _run([lms, "server", "start"], timeout=30)
            for _ in range(25):
                if await is_server_up(base_url):
                    break
                await asyncio.sleep(1.0)
            else:
                logger.warning("LM Studio server did not come up after start.")
                return None

        entries = await catalogue(base_url)
        sizes = await model_sizes()
        total_mb = _total_vram_mb()

        # 2. Preferred model already loaded? Only an actual match short-circuits
        #    here. Accepting whatever happened to be resident used to look like a
        #    free win, but model capability is not interchangeable for this work:
        #    a 7.5B echoes the transcript back unchanged where a 26B removes a
        #    hundred words of fumbles. Loading the right one is worth 15 seconds.
        #
        #    "Loaded" is still not "usable", though. A model bigger than the card
        #    loads with part of itself offloaded and reports success, then dies
        #    on the first real prompt — so a resident model that cannot fit in
        #    TOTAL VRAM is rejected here and replaced below.
        loaded = await _loaded_models(base_url)
        if _model_matches(loaded, model_name):
            resident = _resolved_from(loaded, model_name)
            if total_mb is None or _fits(sizes.get(resident or ""), total_mb):
                return _remember(resident)
            logger.warning(
                "Model %r is loaded but needs ~%.1fGB against a %.1fGB card; it "
                "is running partly offloaded and will fail mid-request. "
                "Replacing it with one that fits.",
                resident, sizes.get(resident or "", 0.0) / 1024.0, total_mb / 1024.0)
            _unusable.add(resident or "")

        # 3. Load the preferred model. Eject anything else first so it has the
        #    whole GPU — a different model left resident is wasted VRAM and can
        #    tip the load into an out-of-memory failure. (The already-loaded case
        #    returned at step 2, so nothing we want is being ejected here.)
        wanted = _resolve_id(entries, model_name) or model_name
        await _unload_all(base_url)

        # 3a. Will it even fit? `lms load` reports success for a model that does
        #     not fit — llama.cpp simply offloads part of it — and the failure
        #     then arrives much later, as `{"error":"terminated"}` on a real
        #     prompt, by which time the fluency windows have already been lost.
        #     Measured: the configured qwen3.8-27b is 17.7GB of weights on a
        #     16.3GB card, so every model pass on this machine was dying and the
        #     shipped edit was the structural fallback. A model too big for the
        #     GPU is not a preference to honour; it is a model that cannot serve.
        # Let the ejection actually land before reading free VRAM. `lms unload`
        # returns before the driver has reclaimed the memory, and measuring too
        # early reports the GPU as still full — which would reject every model
        # that fits and settle for the smallest one in the catalogue.
        await asyncio.sleep(1.5)
        free_mb = _free_vram_mb()
        if free_mb is not None and not _fits(sizes.get(wanted), free_mb):
            # The card may be held by the OTHER tenant: a warm ComfyUI keeps its
            # generation models resident, and measuring against that reads every
            # loadable language model as too big — which is how the second
            # presentation pass of a night silently lost its planner. Ask it to
            # let go (the mirror of the pass ejecting the LLM before ComfyUI
            # runs), then measure again.
            try:
                from runtime.gpu_broker import gpu_broker
                await gpu_broker.release_comfyui_vram()
                await asyncio.sleep(1.5)
                free_mb = _free_vram_mb()
            except Exception:
                pass
        if free_mb is not None and not _fits(sizes.get(wanted), free_mb):
            logger.warning(
                "Preferred model %r needs ~%.1fGB of weights and only %.1fGB of "
                "VRAM is free; it would load partially offloaded and then die "
                "mid-request. Choosing the largest model that fits instead.",
                wanted, sizes.get(wanted, 0.0) / 1024.0, free_mb / 1024.0)
            wanted = None
        else:
            logger.info("Loading model %r into LM Studio (may take a while)...", wanted)
            if await _try_load(base_url, wanted, load_timeout):
                return _remember(wanted)

        # 4. Try the rest of the catalogue (all ejected now, so each really is
        #    loaded fresh rather than assumed resident). Largest first, because
        #    capability on this task scales hard with size — a 7.5B echoes the
        #    transcript back where a 12B removes real fumbles — but only among
        #    the ones that actually fit.
        alternatives = [entry.get("id") for entry in entries
                        if entry.get("type") != "embeddings"
                        and entry.get("id") != wanted]
        alternatives = [a for a in alternatives if a
                        and (free_mb is None or _fits(sizes.get(a), free_mb))]
        alternatives.sort(key=lambda a: sizes.get(a, 0.0), reverse=True)
        for alternative in alternatives[:3]:
            if await _try_load(base_url, alternative, load_timeout):
                logger.warning("Model %r is unusable; falling back to %r — expect a "
                               "weaker edit if it is a much smaller model.",
                               wanted, alternative)
                return _remember(alternative)

        logger.warning("LM Studio is up but no model could be loaded; "
                       "falling back to the deterministic edit.")
        return None


def _resolved_from(loaded: List[str], model_name: str) -> Optional[str]:
    """The loaded id matching the preference, or any loaded model as a stand-in."""
    if not loaded:
        return None
    base = (model_name or "").split("/")[-1].lower()
    for candidate in loaded:
        if (candidate or "").lower() == (model_name or "").lower():
            return candidate
    for candidate in loaded:
        if base and base in (candidate or "").lower():
            return candidate
    return loaded[0]


def _remember(model_id: str) -> str:
    global _ready_until, _resolved
    _resolved = model_id
    _ready_until = time.monotonic() + _READY_TTL
    return model_id


def forget_resolution() -> None:
    """Drop the cached model after a request fails, so the next call re-resolves."""
    global _ready_until, _resolved
    _resolved = None
    _ready_until = 0.0


async def unload_all_models(base_url: str) -> bool:
    """Eject every resident LM Studio model to hand the GPU to another tenant.

    Called before a ComfyUI generation pass: the language model has done its work
    (all prompts and titles are computed up front) and a multi-GB model left
    loaded is exactly what tips image generation into an out-of-memory error. Also
    clears the resolver cache so the next LLM call loads cleanly.
    """
    await _unload_all(base_url)
    forget_resolution()
    return True


async def get_status(base_url: str, model_name: str) -> dict:
    """Lightweight status for a UI indicator (does not auto-start)."""
    up = await is_server_up(base_url)
    loaded = await _loaded_models(base_url) if up else []
    return {
        "server_up": up,
        "model_loaded": bool(loaded),
        "preferred_loaded": _model_matches(loaded, model_name),
        "active_model": _resolved or _resolved_from(loaded, model_name),
        "loaded_models": loaded,
        "unusable_models": sorted(_unusable),
        "cli_available": find_lms_cli() is not None,
        "autostart_enabled": _AUTOSTART,
    }
