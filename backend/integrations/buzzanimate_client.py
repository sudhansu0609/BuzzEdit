"""
Talk to BuzzAnimate, the native animation app, through its local API.

This is the BuzzEdit -> BuzzAnimate side of the ecosystem bridge (step E4 of
`BuzzAnimate/docs/ECOSYSTEM_BRIDGE_PLAN.md`). BuzzAnimate serves a small JSON
API on 127.0.0.1 (`crates/buzz-app/src/api.rs`): `GET /health`,
`GET /api/state`, `GET /api/catalog`, `POST /api/call {tool, args}`,
`POST /api/open {path}`, `POST /api/save {path?}`,
`POST /api/export {path, format, from?, to?}` -> `{job_id}`,
`GET /api/jobs[/{id}]` and `POST /api/episode {...}` -> `{job_id}`.

Discovery goes through the shared port ledger
(`buzzcaf_ports.discover("buzzanimate", 8840, health_path="/health")`), never
a hardcoded port. Requests never carry an `Origin` header: BuzzAnimate refuses
any request that looks like it came from a web page.

Best-effort by design: BuzzAnimate not running or not answering is one clear
exception (`BuzzAnimateUnavailable`), an answer it refused is
`BuzzAnimateError` (with BuzzAnimate's own status and message), and a job it
reported failed is `BuzzAnimateJobFailed`.

`start()` launches `<estate>/BuzzAnimate/BuzzAnimate.vbs` and waits for the
health route. That launcher runs a cargo build first, so a cold start can take
several minutes; the wait is bounded by `START_TIMEOUT_SECONDS`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import buzzcaf_ports  # backend_dir is already on sys.path (see main.py)
import requests

APP = "buzzanimate"
PREFERRED_PORT = 8840
HEALTH_PATH = "/health"

#: `<estate>/BuzzAnimate/BuzzAnimate.vbs`, the estate root being the folder
#: that holds BuzzEdit.
LAUNCHER = Path(__file__).resolve().parents[3] / "BuzzAnimate" / "BuzzAnimate.vbs"

#: The launcher builds with cargo before the window opens.
START_TIMEOUT_SECONDS = 600.0
#: BuzzAnimate gives each request 60 s on its UI thread before it answers with
#: an error, so a read timeout a little above that never cuts it short.
REQUEST_TIMEOUT = (3.0, 70.0)
#: `/api/open` reads the file in the background; how long to wait for
#: `/api/state` to show it.
OPEN_TIMEOUT_SECONDS = 180.0
#: An export of a long scene is minutes of rendering.
EXPORT_TIMEOUT_SECONDS = 2 * 60 * 60
POLL_SECONDS = 1.0

FORMATS = ("mp4", "mov", "gif", "webp", "png")


class BuzzAnimateUnavailable(Exception):
    """BuzzAnimate could not be found, or did not answer."""


class BuzzAnimateError(RuntimeError):
    """BuzzAnimate answered, but refused or failed the request."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = message


class BuzzAnimateJobFailed(RuntimeError):
    """A BuzzAnimate export or episode job ended `failed`."""


NOT_RUNNING = (
    "BuzzAnimate is not running (not found on the shared port ledger, and nothing on "
    "ports 8840-8860 says it is BuzzAnimate). Pass start: true to launch it; its launcher "
    "builds with cargo first, so that can take a few minutes."
)


def base_url(ttl: float = buzzcaf_ports.DEFAULT_TTL) -> Optional[str]:
    """Where BuzzAnimate is listening, or None. `BUZZANIMATE_URL` overrides."""
    return buzzcaf_ports.discover(APP, PREFERRED_PORT, health_path=HEALTH_PATH, ttl=ttl)


def _base() -> str:
    url = base_url()
    if not url:
        raise BuzzAnimateUnavailable(NOT_RUNNING)
    return url


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None,
             timeout: Any = REQUEST_TIMEOUT) -> Any:
    """One request; BuzzAnimate's JSON answer, or a clear exception."""
    base = _base()
    try:
        resp = requests.request(method, f"{base}{path}", json=body if method != "GET" else None,
                                timeout=timeout)
    except requests.RequestException as exc:
        buzzcaf_ports._forget(APP)
        raise BuzzAnimateUnavailable(f"BuzzAnimate did not answer {method} {path}: {exc}") from exc
    try:
        answer = resp.json()
    except ValueError:
        answer = None
    if resp.status_code != 200:
        message = (answer or {}).get("error") if isinstance(answer, dict) else None
        raise BuzzAnimateError(resp.status_code, message or resp.text[:300] or f"HTTP {resp.status_code}")
    return answer


# ------------------------------------------------------------------ reading


def health() -> Optional[Dict[str, Any]]:
    """`GET /health`, or None -- never raises."""
    try:
        return _request("GET", "/health", timeout=(2.0, 5.0))
    except Exception:
        return None


def state() -> Dict[str, Any]:
    """`GET /api/state`: path, title, dirty, frame, frames, fps, scenes, selection."""
    return _request("GET", "/api/state")


def catalog() -> Any:
    return _request("GET", "/api/catalog")


def status() -> Dict[str, Any]:
    """Whether BuzzAnimate is up, where, and what it has open. Never raises."""
    url = base_url()
    if not url:
        return {"running": False, "url": None, "message": NOT_RUNNING,
                "launcher": str(LAUNCHER), "launcher_found": LAUNCHER.is_file()}
    out: Dict[str, Any] = {"running": True, "url": url, "health": health()}
    try:
        out["state"] = state()
    except Exception as exc:  # the window is busy; health is still the answer
        out["state"] = None
        out["state_error"] = str(exc)
    return out


# ----------------------------------------------------------------- driving


def call(tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`POST /api/call {tool, args}` -> `{ok, value, error}`."""
    return _request("POST", "/api/call", {"tool": tool, "args": args or {}})


def open_document(path: str) -> Dict[str, Any]:
    """`POST /api/open {path}`. The file is read in the background."""
    return _request("POST", "/api/open", {"path": str(path)})


def save(path: Optional[str] = None) -> Dict[str, Any]:
    return _request("POST", "/api/save", {"path": str(path)} if path else {})


def export(path: str, fmt: str = "mp4", from_frame: Optional[int] = None,
           to_frame: Optional[int] = None) -> int:
    """`POST /api/export` -> the job id."""
    body: Dict[str, Any] = {"path": str(path), "format": fmt}
    if from_frame is not None:
        body["from"] = int(from_frame)
    if to_frame is not None:
        body["to"] = int(to_frame)
    return int(_request("POST", "/api/export", body)["job_id"])


def episode(out_dir: str, script_path: Optional[str] = None, script_text: Optional[str] = None,
            casting_path: Optional[str] = None, scenes: Optional[list] = None,
            render: bool = False) -> int:
    """`POST /api/episode` -> the job id."""
    body: Dict[str, Any] = {"out_dir": str(out_dir), "render": bool(render)}
    if script_path:
        body["script_path"] = str(script_path)
    if script_text:
        body["script_text"] = script_text
    if casting_path:
        body["casting_path"] = str(casting_path)
    if scenes:
        body["scenes"] = list(scenes)
    return int(_request("POST", "/api/episode", body)["job_id"])


def job(job_id: int) -> Dict[str, Any]:
    """`GET /api/jobs/{id}`: `{id, kind, state, progress, output, error, result}`."""
    return _request("GET", f"/api/jobs/{int(job_id)}")


def jobs() -> Dict[str, Any]:
    return _request("GET", "/api/jobs")


def wait_job(job_id: int, timeout: float = EXPORT_TIMEOUT_SECONDS,
             on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
             poll: float = POLL_SECONDS) -> Dict[str, Any]:
    """Poll a job until it is `done` (returned) or `failed` (raised)."""
    deadline = time.monotonic() + timeout
    while True:
        current = job(job_id)
        if on_progress:
            on_progress(current)
        if current.get("state") == "done":
            return current
        if current.get("state") == "failed":
            raise BuzzAnimateJobFailed(current.get("error") or f"BuzzAnimate job {job_id} failed")
        if time.monotonic() >= deadline:
            raise BuzzAnimateJobFailed(f"BuzzAnimate job {job_id} had not finished after {int(timeout)} s")
        time.sleep(poll)


def _same_path(a: Optional[str], b: str) -> bool:
    if not a:
        return False
    norm = lambda p: os.path.normcase(os.path.normpath(os.path.abspath(p)))  # noqa: E731
    return norm(a) == norm(b)


def wait_open(path: str, timeout: float = OPEN_TIMEOUT_SECONDS, poll: float = POLL_SECONDS) -> Dict[str, Any]:
    """Wait until `/api/state` says `path` is the open document."""
    deadline = time.monotonic() + timeout
    while True:
        current = state()
        if _same_path(current.get("path"), path):
            return current
        if time.monotonic() >= deadline:
            raise BuzzAnimateError(504, f"BuzzAnimate had not opened {path} after {int(timeout)} s")
        time.sleep(poll)


def render_file(buzz_path: str, out_path: Optional[str] = None, fmt: str = "mp4",
                on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                timeout: float = EXPORT_TIMEOUT_SECONDS) -> str:
    """Open a `.buzz`, export it, wait for the export: the output path.

    The default output is beside the `.buzz`, with the format's extension.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format is one of {', '.join(FORMATS)}")
    out = str(out_path or Path(buzz_path).with_suffix(f".{fmt}"))
    open_document(buzz_path)
    wait_open(buzz_path)
    finished = wait_job(export(out, fmt), timeout=timeout, on_progress=on_progress)
    return finished.get("output") or out


# ----------------------------------------------------------------- starting


def _launch(launcher: Path) -> None:
    """Run the `.vbs` launcher, detached and windowless."""
    if sys.platform != "win32":
        raise BuzzAnimateUnavailable(f"BuzzAnimate's launcher {launcher.name} only runs on Windows.")
    subprocess.Popen(["wscript.exe", str(launcher)], cwd=str(launcher.parent),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def start(timeout: float = START_TIMEOUT_SECONDS, poll: float = 2.0) -> str:
    """Make sure BuzzAnimate is running: its base URL, or `BuzzAnimateUnavailable`."""
    found = base_url(ttl=0)
    if found:
        return found
    launcher = LAUNCHER
    if not launcher.is_file():
        raise BuzzAnimateUnavailable(f"BuzzAnimate is not running and its launcher was not found at {launcher}.")
    _launch(launcher)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll)
        found = base_url(ttl=0)
        if found:
            return found
    raise BuzzAnimateUnavailable(
        f"BuzzAnimate was launched but its /health had not answered after {int(timeout)} s "
        "(its launcher builds with cargo first; try again in a minute)."
    )


def ensure(start_it: bool = False, timeout: float = START_TIMEOUT_SECONDS) -> str:
    """BuzzAnimate's base URL; with `start_it`, launch it first when it is down."""
    if start_it:
        return start(timeout)
    return _base()
