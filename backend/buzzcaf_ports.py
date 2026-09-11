"""
The Buzzcaf port ledger — pick a port, publish it, find someone else's.

GUARDIAN_PLAN.md section 11 is the contract; this file is the canonical Python
implementation of it and is copied **byte-identical** into every Python app in
the ecosystem (`backend/test_buzzcaf_ports.py` fails when a copy drifts).

The rules it implements, in one paragraph: a port is a *wish*, never a fact.
Each listener asks for `<APP>_PORT`, and if something already holds it, steps
forward to the next free one rather than killing whoever is there. Once its own
health route answers, it writes where it landed into the shared ledger
`%LOCALAPPDATA%\\Buzzcaf\\ports.json` (override `BUZZCAF_PORTS_FILE`), and it
removes that entry on a clean exit. Everyone else finds it with `discover()`:
an explicit env URL, then the ledger entry whose pid is still alive, then a scan
of `preferred … preferred+span` — and at every step the health route has to
answer `{"app": "<name>", …}` or the answer does not count. That last check is
the one that matters: loopback ports are crowded, and a scan that accepted any
listener is exactly how CaliberAI got mistaken for the Studio.

    pick_port(preferred, span=20, host="127.0.0.1") -> int
    publish(app, port, health, extra=None, ledger=None) -> None
    withdraw(app, ledger=None) -> None
    discover(app, preferred, health_path="/health", span=20, env_url=None, ttl=3) -> str | None
    ledger_path() -> Path

plus two readers the consumers need — `entry(app)` and `entries()` — for the
cases where the answer is not a base URL: a sub-service port under `extra`, or a
third-party server (buzzcode's `llama-server`) that cannot be taught to name
itself in its health body.

Stdlib only, on purpose: this file is copied into apps that share no
dependencies with each other, and a port helper that cannot be imported during
startup is worse than a hardcoded port.

An entry looks like:

    {"dexter": {"port": 8098, "pid": 33716,
                "started_at": "2026-09-10T18:21:07+00:00",
                "health": "http://127.0.0.1:8098/health",
                "extra": {"vite": 3006}}}

Writes are read-merge-write under `ports.json.lock` (retried, considered stale
after 5 s so one crashed writer cannot wedge the ecosystem) and land through a
temp file plus `os.replace`, so a reader never sees half a file and one app's
crash never loses another app's entry.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "pick_port",
    "publish",
    "withdraw",
    "discover",
    "ledger_path",
    "entry",
    "entries",
    "SPAN",
]

#: How far forward a listener may step, and how far a scan looks. One place, so
#: BuzzBrain's extension and buzzcode's `ports.rs` quote the same number.
SPAN = 20
#: A lock older than this belonged to something that died holding it.
LOCK_STALE_SECONDS = 5.0
LOCK_RETRY_SECONDS = 0.02
LOCK_TIMEOUT_SECONDS = 5.0
#: Health probes are on startup paths and on `/api/dexter/apps`; keep them short.
HEALTH_TIMEOUT = 1.5
CONNECT_TIMEOUT = 0.25
#: Seconds a discovery result (including "offline") is reused.
DEFAULT_TTL = 3.0
#: Windows: GetExitCodeProcess reports this while the process is still running.
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5

_cache: Dict[str, Tuple[float, Optional[str]]] = {}
_cache_lock = threading.Lock()


# ───────────────────────── the ledger file ─────────────────────────


def ledger_path() -> Path:
    """Where the shared ledger lives. `BUZZCAF_PORTS_FILE` overrides it."""
    override = (os.environ.get("BUZZCAF_PORTS_FILE") or "").strip()
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        # Not Windows (CI, a container): the same file under XDG's data dir.
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
    return Path(base) / "Buzzcaf" / "ports.json"


def _resolve(ledger: Optional[Any]) -> Path:
    return Path(ledger) if ledger else ledger_path()


class _FileLock:
    """`ports.json.lock`, created O_EXCL, retried, broken when stale.

    Deliberately not `msvcrt`/`fcntl`: those are per-platform, and the ledger is
    written by Python, Node and Rust. A lock file with a staleness rule is the
    one mechanism all three can implement the same way.
    """

    def __init__(self, target: Path) -> None:
        self.path = Path(str(target) + ".lock")
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        deadline = time.time() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(self._fd, str(os.getpid()).encode("ascii"))
                except OSError:
                    pass
                return self
            except FileExistsError:
                if self._stale() or time.time() >= deadline:
                    # Either the holder died, or it is so slow that waiting
                    # longer would cost more than the write is worth. Losing an
                    # entry is recoverable; a startup that never finishes is not.
                    self._break()
                    continue
                time.sleep(LOCK_RETRY_SECONDS)
            except OSError:
                # An unwritable ledger directory: proceed unlocked rather than
                # refuse to start. The write itself still fails loudly if it must.
                return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._break()

    def _stale(self) -> bool:
        try:
            return (time.time() - self.path.stat().st_mtime) > LOCK_STALE_SECONDS
        except OSError:
            return False  # it vanished between the O_EXCL and here; just retry

    def _break(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def _read(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception:
        # A truncated or hand-edited ledger is not a reason to fail a startup.
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(f"{path}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


# ───────────────────────── picking ─────────────────────────


def _bindable(port: int, host: str) -> bool:
    """A real exclusive bind, not a probe.

    A connect probe answers "is anyone listening", which is a different
    question: a socket in TIME_WAIT, or one bound to `0.0.0.0` by another app,
    refuses our bind while answering nothing. SO_REUSEADDR is *not* set — on
    POSIX it would let us bind on top of a TIME_WAIT the OS is still holding,
    and asking for SO_EXCLUSIVEADDRUSE on Windows is the whole point.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pick_port(preferred: int, span: int = SPAN, host: str = "127.0.0.1") -> int:
    """The preferred port if it is free, else the next free one, else port 0.

    Never evicts. Whatever holds the preferred port keeps it; we move.

    There is an unavoidable gap between this bind and the server's — the socket
    has to be closed for uvicorn (or Node, or Rust) to take it. In practice the
    two are milliseconds apart in the same process; the alternative, handing the
    listening socket to the server, is not something all three languages can do.
    """
    try:
        preferred = int(preferred)
    except (TypeError, ValueError):
        preferred = 0
    try:
        span = max(0, int(span))
    except (TypeError, ValueError):
        span = SPAN

    if 0 < preferred <= 65535:
        for port in range(preferred, min(preferred + span, 65535) + 1):
            if _bindable(port, host):
                return port

    # Everything in the window is taken: let the OS name one. An odd port beats
    # a startup that gives up, and `publish()` tells everyone where it went.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


# ───────────────────────── publishing ─────────────────────────


def publish(
    app: str,
    port: int,
    health: str,
    extra: Optional[Dict[str, Any]] = None,
    ledger: Optional[Any] = None,
) -> None:
    """Record where this app landed. Call it once the health route answers.

    Publishing before the server is up is what turns a ledger into a liar: the
    next app to look sees an entry, trusts the pid, and waits on a socket that
    will never open.
    """
    path = _resolve(ledger)
    record = {
        "port": int(port),
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": str(health),
        "extra": dict(extra or {}),
    }
    with _FileLock(path):
        data = _read(path)
        data[str(app)] = record
        _write(path, data)
    _forget(str(app))


def withdraw(app: str, ledger: Optional[Any] = None) -> None:
    """Remove this app's entry, and only this app's entry."""
    path = _resolve(ledger)
    key = str(app)
    with _FileLock(path):
        data = _read(path)
        if key in data:
            del data[key]
            _write(path, data)
    _forget(key)


def entries(ledger: Optional[Any] = None) -> Dict[str, Any]:
    """The whole ledger, as read. No liveness filtering — that is `discover`."""
    return _read(_resolve(ledger))


def entry(app: str, ledger: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """One app's entry, or None.

    For the two cases a base URL cannot express: a sub-service recorded under
    `extra`, and a third-party server whose health route will never say `app`.
    """
    found = _read(_resolve(ledger)).get(str(app))
    return dict(found) if isinstance(found, dict) else None


# ───────────────────────── discovery ─────────────────────────


def _pid_alive(pid: Any) -> bool:
    """Is this pid a live process?

    **Never `os.kill(pid, 0)` on Windows.** CPython maps every signal except
    CTRL_C_EVENT/CTRL_BREAK_EVENT onto `TerminateProcess`, so the POSIX idiom
    for "does this process exist" *kills the process* here — it would have made
    a liveness check into a way for any app to shoot the Studio.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means it exists and belongs to someone else.
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _base_of(url: str) -> str:
    """'http://127.0.0.1:8098/health' -> 'http://127.0.0.1:8098'."""
    from urllib.parse import urlparse

    parsed = urlparse(str(url))
    if not parsed.scheme or not parsed.netloc:
        return str(url).rstrip("/")
    host = parsed.hostname or "127.0.0.1"
    # localhost resolves to ::1 first on Windows and costs ~2 s per dead call.
    if host.lower() == "localhost":
        host = "127.0.0.1"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


def _health_url(base: str, health_path: str) -> str:
    return f"{str(base).rstrip('/')}/{str(health_path).lstrip('/')}"


def _probe(url: str, timeout: float = HEALTH_TIMEOUT) -> Optional[Dict[str, Any]]:
    """The health body as a dict, `{}` for a 200 that is not JSON, None for no answer."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "buzzcaf-ports/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            body = response.read(65536)
    except Exception:
        return None
    try:
        parsed = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _listening(host: str, port: int) -> bool:
    """A cheap TCP connect, so a scan of 21 dead ports costs milliseconds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(CONNECT_TIMEOUT)
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _identifies(base: str, health_path: str, app: str) -> bool:
    """True only when the listener at `base` says it is `app`.

    This is rule 4, and it is the whole reason discovery is trustworthy.
    """
    body = _probe(_health_url(base, health_path))
    return bool(body) and body.get("app") == app


def _cached(app: str) -> Tuple[bool, Optional[str]]:
    with _cache_lock:
        found = _cache.get(app)
    if not found:
        return False, None
    expires, url = found
    if time.time() >= expires:
        return False, None
    return True, url


def _remember(app: str, url: Optional[str], ttl: float) -> None:
    if ttl <= 0:
        return
    with _cache_lock:
        _cache[app] = (time.time() + ttl, url)


def _forget(app: Optional[str] = None) -> None:
    with _cache_lock:
        if app is None:
            _cache.clear()
        else:
            _cache.pop(app, None)


def discover(
    app: str,
    preferred: int,
    health_path: str = "/health",
    span: int = SPAN,
    env_url: Optional[str] = None,
    ttl: float = DEFAULT_TTL,
    ledger: Optional[Any] = None,
    host: str = "127.0.0.1",
) -> Optional[str]:
    """Where `app` is listening, as a base URL — or None when it is not.

    Order (rule 5): an explicit env URL → the ledger entry whose pid is alive
    *and* whose health names the app → a scan of `preferred … preferred+span`
    with the same identity check → None. The result, offline included, is cached
    for `ttl` seconds: this runs on `/api/dexter/apps`, on every status poll and
    on several startup paths.
    """
    app = str(app)
    hit, url = _cached(app)
    if hit:
        return url
    found = _discover_now(app, preferred, health_path, span, env_url, ledger, host)
    _remember(app, found, ttl)
    return found


def _discover_now(
    app: str,
    preferred: int,
    health_path: str,
    span: int,
    env_url: Optional[str],
    ledger: Optional[Any],
    host: str,
) -> Optional[str]:
    # 1. The owner said where it is. `<APP>_URL` is the convention when the
    #    caller does not pass one.
    explicit = (env_url or os.environ.get(f"{app.upper()}_URL") or "").strip()
    if explicit:
        base = _base_of(explicit)
        body = _probe(_health_url(base, health_path))
        # An address the owner named by hand is allowed to hold a server that
        # cannot name itself — a llama.cpp or a ComfyUI. A *scan* is not.
        if body is not None and (body.get("app") == app or "app" not in body):
            return base

    # 2. The ledger. A live pid is not proof on its own (pids are reused), so
    #    the health route still has to agree.
    record = entry(app, ledger)
    if record and _pid_alive(record.get("pid")):
        health = record.get("health")
        base = _base_of(health) if health else None
        if not base and record.get("port"):
            base = f"http://{host}:{int(record['port'])}"
        if base and _identifies(base, health_path, app):
            return base

    # 3. Step-forward scan, identity enforced.
    try:
        preferred = int(preferred)
        span = max(0, int(span))
    except (TypeError, ValueError):
        return None
    if 0 < preferred <= 65535:
        for port in range(preferred, min(preferred + span, 65535) + 1):
            if not _listening(host, port):
                continue
            base = f"http://{host}:{port}"
            if _identifies(base, health_path, app):
                return base
    return None
