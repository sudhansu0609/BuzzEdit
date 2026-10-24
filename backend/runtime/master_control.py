"""
The BuzzEdit side of the master pause / resume (Buzzcaf Studio's
docs/STUDIO_DIRECTOR_PLAN.md Part 1). The Studio calls `POST /api/master/pause`
/ `/resume` (routes/master.py) on this process whenever its own master pause
is switched; this module is what those routes, the scheduler and the vibe
reel / narration jobs actually check.

State lives in `DATA_DIR/master_control.json` (gitignored, like
`data/jobs.json`) so it survives a restart: a BuzzEdit that comes back up
mid-pause stays paused until the Studio resumes it.

`pause(mode)`:
- "soft" -- the scheduler holds new jobs (`JobScheduler.is_paused`, its
  existing flag; a job already running finishes); a vibe reel or narration
  in progress holds at its next shot/job boundary.
- "now" -- soft, plus ComfyUI's current prompt is interrupted right away (a
  shot not yet cached regenerates on resume; a finished one is reused).

Either way every local model this process holds (LM Studio, Ollama, the
in-process ASR/aligner) is released through `runtime.gpu_handover.offload_all`.

`wait_if_paused` (sync, for a thread-run job like narration) and
`wait_if_paused_async` (for an `async def` loop like the vibe reel's per-shot
pass) block while paused and return the moment `resume()` runs, so the
caller just carries on from where it was -- no job is marked failed for
being held.
"""
import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import DATA_DIR

logger = logging.getLogger("runtime.master_control")

STATE_PATH = DATA_DIR / "master_control.json"
POLL_S = 2.0
MODES = ("soft", "now")

_lock = threading.Lock()
_held: Dict[str, str] = {}
_state_cache: Optional[Dict[str, Any]] = None


def _default() -> Dict[str, Any]:
    return {"paused": False, "since": None, "reason": "", "mode": "soft", "by": "",
            "prior_scheduler_paused": False}


def _load() -> Dict[str, Any]:
    global _state_cache
    if _state_cache is not None:
        return _state_cache
    data: Dict[str, Any] = {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:  # noqa: BLE001 - no file yet, or unreadable: start fresh
        data = {}
    state = _default()
    state.update({k: v for k, v in data.items() if k in state})
    _state_cache = state
    return state


def _save(state: Dict[str, Any]) -> None:
    global _state_cache
    _state_cache = dict(state)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = str(STATE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(STATE_PATH))
    except Exception as exc:  # noqa: BLE001 - the in-memory state still holds
        logger.warning("master_control: could not write %s (%s)", STATE_PATH, exc)


def is_paused() -> bool:
    with _lock:
        return bool(_load().get("paused"))


def current_mode() -> str:
    with _lock:
        return str(_load().get("mode") or "soft")


def held() -> List[str]:
    with _lock:
        return sorted(_held.keys())


def state() -> Dict[str, Any]:
    with _lock:
        data = dict(_load())
    data["held"] = held()
    data.pop("prior_scheduler_paused", None)
    return data


def _mark(where: str) -> None:
    with _lock:
        _held[where] = datetime.now().isoformat(timespec="seconds")


def _unmark(where: str) -> None:
    with _lock:
        _held.pop(where, None)


def wait_if_paused(where: str, say: Optional[Callable[[str], None]] = None) -> bool:
    """Blocking (thread) version, for a job run on its own thread (narration)."""
    if not is_paused():
        return False
    _mark(where)
    told = False
    try:
        while is_paused():
            if not told:
                logger.info("master pause: holding at %s", where)
                if say:
                    try:
                        say(f"holding for the master pause ({where})")
                    except Exception:  # noqa: BLE001
                        pass
                told = True
            time.sleep(POLL_S)
    finally:
        _unmark(where)
    return told


async def wait_if_paused_async(where: str, say: Optional[Callable[[float, str], None]] = None) -> bool:
    """Non-blocking version, for an `async def` loop (the vibe reel's per-shot
    pass): awaits instead of sleeping the event loop's own thread."""
    if not is_paused():
        return False
    _mark(where)
    told = False
    try:
        while is_paused():
            if not told:
                logger.info("master pause: holding at %s", where)
                if say:
                    try:
                        say(0.0, f"holding for the master pause ({where})")
                    except Exception:  # noqa: BLE001
                        pass
                told = True
            await asyncio.sleep(POLL_S)
    finally:
        _unmark(where)
    return told


def _interrupt_comfyui_now() -> None:
    """"now" mode: stop whatever ComfyUI is making right now. Best effort --
    a shot not yet cached simply regenerates on resume."""
    try:
        import requests
        from config import COMFYUI_URL
        base = COMFYUI_URL.rstrip("/")
        requests.post(f"{base}/interrupt", timeout=5)
    except Exception as exc:  # noqa: BLE001 - ComfyUI down/idle: nothing to interrupt
        logger.debug("master pause (now): ComfyUI interrupt skipped (%s)", exc)


def _pause_scheduler() -> bool:
    """The scheduler's own pause flag: new jobs hold, the running one finishes."""
    try:
        from routes.scheduler import scheduler_instance
        already = bool(scheduler_instance.is_paused)
        scheduler_instance.is_paused = True
        return already
    except Exception as exc:  # noqa: BLE001
        logger.warning("master pause: scheduler not paused (%s)", exc)
        return False


def _resume_scheduler(prior_paused: bool) -> None:
    try:
        from routes.scheduler import scheduler_instance
        if not prior_paused:
            scheduler_instance.is_paused = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("master resume: scheduler not resumed (%s)", exc)


def _release_memory(reason: str) -> None:
    try:
        from runtime import gpu_handover

        async def _go() -> None:
            await gpu_handover.offload_all(reason)

        try:
            asyncio.get_running_loop()
            # Called from inside the route's own event loop: fire-and-forget,
            # the pause itself must not block on it.
            asyncio.create_task(_go())
        except RuntimeError:
            asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001 - never fail the pause over the release
        logger.warning("master pause: release skipped (%s)", exc)


def pause(mode: str = "soft", reason: str = "", by: str = "") -> Dict[str, Any]:
    """Never quits BuzzEdit -- only holds new jobs and releases models. Only
    ever lifted by an explicit `resume()`; nothing here auto-resumes."""
    mode = mode if mode in MODES else "soft"
    with _lock:
        data = _load()
        already = bool(data.get("paused"))
        if already:
            data["mode"] = mode
            if reason:
                data["reason"] = reason
            if by:
                data["by"] = by
            _save(data)
        else:
            prior_sched_paused = _pause_scheduler()
            data.update(paused=True, since=datetime.now().isoformat(timespec="seconds"),
                       reason=reason, mode=mode, by=by, prior_scheduler_paused=prior_sched_paused)
            _save(data)
    logger.warning("BuzzEdit master pause (%s) by %s: %s", mode, by or "unknown", reason or "no reason given")
    if mode == "now":
        _interrupt_comfyui_now()
    _release_memory(f"master pause ({mode})")
    return state()


def resume() -> Dict[str, Any]:
    with _lock:
        data = _load()
        was_paused = bool(data.get("paused"))
        if was_paused:
            prior_sched_paused = bool(data.get("prior_scheduler_paused"))
            data.update(paused=False, since=None, reason="", mode="soft", by="", prior_scheduler_paused=False)
            _save(data)
    # `state()` re-acquires `_lock` -- never call it while still holding it
    # above (a plain Lock is not reentrant; it deadlocked here once).
    if not was_paused:
        return state()
    _resume_scheduler(prior_sched_paused)
    logger.warning("BuzzEdit master resume: held work continues.")
    return state()
