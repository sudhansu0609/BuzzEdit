"""Move the backend onto new code by itself, the first time it is idle.

A fix to the backend did nothing until someone restarted BuzzEdit, and BuzzEdit
must never be killed while it works. On 2026-10-21 the render fix for 674aefab
sat on disk while the running process failed the same render again. Now the
backend notices that its own source changed since it started and, once it has
been idle for QUIET_S, exits cleanly; Electron's supervisor (main.cjs) starts
it again on the new code within seconds. The scheduler keeps its queue on disk,
so nothing queued is lost.

Idle means all of: no scheduler job running or due, no HTTP request or
websocket open (transcription runs inside its request), no background task
spawned by a route (presentation, eye contact, music, SFX, auto-edit...), and
an empty ComfyUI queue. Anything unknown counts as busy. Only a supervised
backend (started by Electron) ever exits; a hand-started one just logs.
"""
import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
_SKIP_DIRS = {"tests", "data", "__pycache__", ".venv", "node_modules"}
CHECK_S = 10.0
QUIET_S = 30.0
EXIT_GRACE_S = 60.0
# Long-lived reads that are not work: media byte-range streams for the UI; and
# the health probe, which the Studio polls while it waits for this very reload.
_IGNORED_PATHS = ("/api/media", "/api/health")

#: True once the source has changed under a running backend: the next idle
#: moment restarts it. `/api/health` reports it so the Studio's queue waits for
#: the restart before starting its next item instead of keeping it busy forever.
_pending = False

_inflight = 0
_lock = threading.Lock()


def reload_pending() -> bool:
    return _pending


def supervised() -> bool:
    return os.environ.get("BUZZEDIT_SUPERVISED") == "1" or bool(os.environ.get("BUZZEDIT_PORT"))


def code_mtime(root: Path = BACKEND_DIR) -> float:
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                try:
                    newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
                except OSError:
                    continue
    return newest


class InflightMiddleware:
    """Counts open HTTP requests and websockets (pure ASGI, so a response's
    background tasks and a long transcription both count until they end)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _inflight
        counted = scope.get("type") in ("http", "websocket") and \
            not str(scope.get("path", "")).startswith(_IGNORED_PATHS)
        if counted:
            with _lock:
                _inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            if counted:
                with _lock:
                    _inflight -= 1


def _route_tasks() -> int:
    routes = str(BACKEND_DIR / "routes")
    count = 0
    for task in asyncio.all_tasks():
        code = getattr(task.get_coro(), "cr_code", None)
        if code is not None and code.co_filename.startswith(routes) and not task.done():
            count += 1
    return count


def _comfyui_busy() -> bool:
    from config import COMFYUI_URL
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL.rstrip('/')}/queue", timeout=3) as r:
            q = json.loads(r.read().decode("utf-8"))
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:  # noqa: BLE001 - ComfyUI down is not ComfyUI busy
        return False


def busy_reason(route_tasks: int = 0) -> Optional[str]:
    """Why the backend must not exit now, or None. `route_tasks` is counted on the
    event loop's own thread (`_route_tasks`): this runs in a worker thread, where
    asyncio has no running loop to ask."""
    from agents.scheduler import _is_due
    from routes.scheduler import scheduler_instance as sched
    # Master pause (docs/STUDIO_DIRECTOR_PLAN.md Part 1): anything actually
    # held right now (a vibe reel between shots, a narration between jobs)
    # must survive a reload -- the checks below already catch most of these
    # (the job stays "running" while held), this is the explicit backstop.
    try:
        from runtime import master_control
        mc_held = master_control.held() if master_control.is_paused() else []
        if mc_held:
            return f"master paused with {len(mc_held)} held: {', '.join(mc_held[:3])}"
    except Exception:  # noqa: BLE001
        pass
    if sched.current_job_id:
        return f"scheduler job {sched.current_job_id}"
    if any(j.get("status") in ("pending", "running") and _is_due(j) for j in sched.jobs):
        return "a scheduler job is due"
    with _lock:
        open_requests = _inflight
    if open_requests:
        return f"{open_requests} request(s) open"
    if route_tasks:
        return f"{route_tasks} background task(s)"
    # Thread-run jobs hold no request and no route task: a narration (a Story Desk
    # preview) and a vibe reel would otherwise die with the reload.
    try:
        from audio import narration
        if narration.running_jobs():
            return "a narration is running"
    except Exception:  # noqa: BLE001 - an import problem must not block reloads forever
        pass
    try:
        from routes import reels
        if reels._busy():
            return "a reel is rendering"
    except Exception:  # noqa: BLE001
        pass
    if _comfyui_busy():
        return "ComfyUI is generating"
    return None


async def watch(request_exit: Callable[[], None]) -> None:
    global _pending
    started = code_mtime()
    idle_since: Optional[float] = None
    announced = False
    while True:
        await asyncio.sleep(CHECK_S)
        try:
            changed = await asyncio.to_thread(code_mtime) > started
            if not changed:
                continue
            if not announced:
                _pending = True
                logger.info("Code reload: backend source changed since start; reloading at the first idle moment.")
                announced = True
            reason = await asyncio.to_thread(busy_reason, _route_tasks())
            if reason:
                idle_since = None
                continue
            now = time.monotonic()
            idle_since = idle_since or now
            if now - idle_since < QUIET_S:
                continue
            if not supervised():
                logger.info("Code reload: idle on new code, but not started by BuzzEdit's supervisor; restart by hand.")
                return
            logger.warning("Code reload: idle for %.0f s on changed code; exiting so the supervisor restarts it.",
                           now - idle_since)
            _exit_backstop()
            request_exit()
            return
        except Exception as e:  # noqa: BLE001 - never take the backend down over this
            logger.warning("Code reload check failed: %s", e)


def _exit_backstop() -> None:
    """A clean shutdown that hangs would leave a live, deaf process the
    supervisor never restarts; end it hard after EXIT_GRACE_S."""
    def kill() -> None:
        time.sleep(EXIT_GRACE_S)
        logger.warning("Code reload: shutdown did not finish in %.0f s; exiting hard.", EXIT_GRACE_S)
        os._exit(0)
    threading.Thread(target=kill, daemon=True).start()
