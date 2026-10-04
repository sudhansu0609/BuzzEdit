"""Tell Sentinel that BuzzEdit's work outranks background jobs while it runs.

Sentinel picks what to freeze or kill under memory pressure by class first:
every `batch` process, lowest priority first, then the unclassified ones,
biggest first, and `interactive`/`assistant` only when the machine is really
out of memory. Its config had `ffmpeg.exe` as batch priority 10 -- the first
victim on the machine -- and the processes that actually do BuzzEdit's work
match no rule at all: the backend runs as the base `python.exe` the venv
launcher hands off to, and ComfyUI as a uv-managed `python.exe`, so both were
unclassified. A render killed three hours in ("exit code 1", no message) was
the result.

A process a client registers takes the registered class over the config. So
while BuzzEdit works a job, its backend, ComfyUI and every FFmpeg pass are
registered `interactive`; when the job ends they are dropped again, so an idle
ComfyUI holding a cache stays reclaimable. Registrations drop after 120 s of
silence, so a heartbeat re-sends them.

Best-effort throughout: Sentinel absent, a v1 service, or a refused pipe means
nothing changes. Pipe I/O runs on its own thread and never blocks a render.
"""
import json
import logging
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("sentinel_priority")

PIPE = os.environ.get("SENTINEL_PIPE", r"\\.\pipe\Sentinel")
#: Sentinel drops a registration silent for 120 s (`registration_ttl_secs`).
HEARTBEAT_SECS = 45.0
PIPE_TIMEOUT_SECS = 1.0
CLASS = "interactive"
BACKEND = "buzzedit-backend"
COMFYUI = "buzzedit-comfyui"

_lock = threading.Lock()
_held: Dict[str, int] = {}          # client name -> pid
_wake = threading.Event()
_thread: Optional[threading.Thread] = None
_active_jobs = 0


def _round_trip(command: Any) -> Any:
    line = (json.dumps(command) + "\n").encode("utf-8")
    with open(PIPE, "r+b", buffering=0) as handle:
        handle.write(line)
        reply = bytearray()
        while not reply.endswith(b"\n"):
            chunk = handle.read(4096)
            if not chunk:
                break
            reply.extend(chunk)
    return json.loads(reply.decode("utf-8")) if reply.strip() else None


def _send(command: Any) -> Any:
    """One command with a deadline; None on any failure or timeout."""
    if sys.platform != "win32":
        return None
    result: Dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = _round_trip(command)
        except Exception as exc:  # pipe missing, ACL, service restarting
            result["error"] = exc

    worker = threading.Thread(target=run, name="sentinel-priority-io", daemon=True)
    worker.start()
    worker.join(PIPE_TIMEOUT_SECS)
    return None if worker.is_alive() or "error" in result else result.get("value")


def _register(client: str, pid: int) -> bool:
    # v1-compatible payload: a v1 service rejects keys it does not know.
    reply = _send({"Register": {"client": client, "pid": pid, "class": CLASS, "budget_mib": None}})
    return reply == "Ok" or (isinstance(reply, dict) and "Registered" in reply)


def _unregister(client: str) -> None:
    _send({"Unregister": {"client": client}})


def _loop() -> None:
    announced = set()
    while True:
        _wake.wait(HEARTBEAT_SECS)
        _wake.clear()
        with _lock:
            in_job = _active_jobs > 0
        if in_job:
            # Looked up here, off the event loop, and on every beat: ComfyUI
            # can be restarted mid-job under a new pid.
            port = _comfyui_port()
            comfy_pid = listening_pid(port) if port else None
            with _lock:
                if _active_jobs > 0 and comfy_pid:
                    _held[COMFYUI] = comfy_pid
        with _lock:
            held = dict(_held)
        for client, pid in held.items():
            if _register(client, pid) and client not in announced:
                announced.add(client)
                logger.info("Sentinel: %s (pid %s) registered %s for the job.", client, pid, CLASS)
        announced &= set(held)


def _ensure_thread() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, name="sentinel-priority", daemon=True)
        _thread.start()


def hold(client: str, pid: Optional[int]) -> None:
    """Keep `pid` registered as `client` until `drop(client)`."""
    if not pid:
        return
    with _lock:
        _held[client] = int(pid)
    _ensure_thread()
    _wake.set()


def drop(client: str) -> None:
    with _lock:
        known = _held.pop(client, None)
    if known is not None:
        threading.Thread(target=_unregister, args=(client,), name="sentinel-priority-drop", daemon=True).start()


@contextmanager
def ffmpeg(pid: Optional[int]) -> Iterator[None]:
    """An FFmpeg pass: registered for as long as it runs."""
    client = f"buzzedit-ffmpeg-{pid}"
    hold(client, pid)
    try:
        yield
    finally:
        drop(client)


def listening_pid(port: int) -> Optional[int]:
    """The pid listening on 127.0.0.1:`port`, from `netstat` (no psutil here)."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                             timeout=10, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except Exception:
        return None
    pattern = re.compile(rf"^\s*TCP\s+\S+:{int(port)}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.MULTILINE)
    found = pattern.search(out or "")
    return int(found.group(1)) if found else None


def _comfyui_port() -> Optional[int]:
    try:
        from config import COMFYUI_URL
        found = re.search(r":(\d+)", COMFYUI_URL.split("//", 1)[-1])
        return int(found.group(1)) if found else None
    except Exception:
        return None


@contextmanager
def job() -> Iterator[None]:
    """A scheduler job: the backend and ComfyUI outrank background work until
    it ends. Nested or overlapping jobs keep the registrations until the last
    one finishes."""
    global _active_jobs
    with _lock:
        _active_jobs += 1
        first = _active_jobs == 1
    if first:
        hold(BACKEND, os.getpid())   # the heartbeat adds ComfyUI
    try:
        yield
    finally:
        with _lock:
            _active_jobs -= 1
            last = _active_jobs == 0
        if last:
            drop(BACKEND)
            drop(COMFYUI)
