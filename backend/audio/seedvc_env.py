"""Where the seed-vc voice engine lives, and how the backend runs it (VOICE_CHANGER_PLAN.md, brief S0).

seed-vc is GPLv3 and pins numpy 1.26 / transformers 4.46, so it never loads in this process:
`engines/seedvc/cli.py` runs under seed-vc's own Python as a subprocess and speaks JSON --
progress lines on stderr, one result object on stdout, exit 0 ok / 1 error / 2 engine unavailable.

`resolve()` picks that Python and the seed-vc folder: the `seedvc_python` / `seedvc_dir` settings
when set (empty = auto-detect), else BuzzEdit's own `engines/seedvc/.venv` + `engines/seedvc/seed-vc`
(built by scripts/setup_seedvc.ps1), else SkinWalker's existing `.venv` + `vendor/seed-vc` (nothing
is moved until S4, so SkinWalker keeps working), else unavailable.

`run_cli()` follows the GPU policy for the commands that load the models (warm, convert,
apply-edit): it refuses while a render job is running, hands the card over first
(`gpu_handover.prepare_for_phase("voice_conversion", ...)`) and offloads everything after. It blocks
until the CLI exits; call it from a worker thread (`asyncio.to_thread`) inside the server.

`Worker` (brief S1) keeps one `cli.py serve` process alive so the models load once (16-17 s per
process in S0) instead of once per conversion: `ensure()` hands the card over and starts it,
`convert()` / `apply_edit()` send one JSON job at a time and wait for its result line, `stop()`
sends quit and offloads; it stops itself after IDLE_STOP_S without jobs and refuses to start (or
take a job) while a render runs. `WORKER` is the backend's one instance.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import concurrent.futures
import itertools
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from config import BASE_DIR, DATA_DIR
from runtime import gpu_handover
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)

ENGINE_DIR = BASE_DIR / "engines" / "seedvc"          # BuzzEdit's own engine home (.venv, seed-vc)
CLI_PATH = BASE_DIR / "engines" / "seedvc" / "cli.py"
SKINWALKER_DIR = Path(r"B:\youtubeProjects\Buzzcaf_Media\skinWalker")

GPU_COMMANDS = ("warm", "convert", "apply-edit")
GPU_PHASE = "voice_conversion"
NEED_VRAM_MB = 6000
NEED_RAM_MB = 8000
EXIT_UNAVAILABLE = 2
TAIL_LINES = 25


class SeedVCError(RuntimeError):
    """The CLI failed; the message carries its error and the last stderr lines."""

    def __init__(self, message: str, exit_code: Optional[int] = None,
                 stderr_tail: Optional[List[str]] = None,
                 result: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_tail = list(stderr_tail or [])
        self.result = result


class SeedVCUnavailable(SeedVCError):
    """No usable seed-vc: nothing resolved, or the CLI exited 2."""


class SeedVCBusy(SeedVCError):
    """A render is running; voice conversion waits for it."""


# --- where it lives ---------------------------------------------------------------------------

def _configured() -> Dict[str, str]:
    """The `seedvc_dir` / `seedvc_python` settings, "" when unset (= auto-detect)."""
    try:
        from store.app_settings import get_seedvc_settings
        return get_seedvc_settings()
    except Exception as e:
        logger.debug("seed-vc settings unreadable: %s", e)
        return {"seedvc_dir": "", "seedvc_python": ""}


def _venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


def _is_seedvc(folder: Optional[Path]) -> bool:
    return folder is not None and (folder / "seed_vc_wrapper.py").is_file()


def _candidates() -> List[tuple]:
    """(source, python, seed-vc folder) in the order they are tried after the settings."""
    return [
        ("engines", _venv_python(ENGINE_DIR / ".venv"), ENGINE_DIR / "seed-vc"),
        ("skinwalker", _venv_python(SKINWALKER_DIR / ".venv"), SKINWALKER_DIR / "vendor" / "seed-vc"),
    ]


def resolve() -> Dict[str, Any]:
    """{python, seedvc_dir, cli, source, reason}: source is "settings", "engines", "skinwalker" or
    "unavailable" (python and seedvc_dir None, `reason` says why).

    A setting that is set wins and is never silently replaced: when it points nowhere the engine
    is unavailable. With only one of the two set, the other half comes from the first
    auto-detected place that has it."""
    cli = Path(CLI_PATH)
    unavailable: Dict[str, Any] = {"python": None, "seedvc_dir": None, "cli": str(cli),
                                   "source": "unavailable", "reason": ""}
    if not cli.is_file():
        unavailable["reason"] = f"{cli} is missing"
        return unavailable
    configured = _configured()
    want_dir = (configured.get("seedvc_dir") or "").strip()
    want_python = (configured.get("seedvc_python") or "").strip()
    candidates = _candidates()

    if want_dir or want_python:
        python = Path(want_python) if want_python else next(
            (py for _, py, _ in candidates if py.is_file()), None)
        folder = Path(want_dir) if want_dir else next(
            (d for _, _, d in candidates if _is_seedvc(d)), None)
        problems = []
        if python is None or not python.is_file():
            problems.append(f"seedvc_python {python or '(auto-detect found none)'} is not a file")
        if not _is_seedvc(folder):
            problems.append(f"seedvc_dir {folder or '(auto-detect found none)'} has no seed_vc_wrapper.py")
        if problems:
            unavailable["reason"] = "; ".join(problems)
            return unavailable
        return {"python": str(python), "seedvc_dir": str(folder), "cli": str(cli),
                "source": "settings", "reason": ""}

    for source, python, folder in candidates:
        if python.is_file() and _is_seedvc(folder):
            return {"python": str(python), "seedvc_dir": str(folder), "cli": str(cli),
                    "source": source, "reason": ""}
    unavailable["reason"] = ("no seed-vc found: run scripts/setup_seedvc.ps1, or set the seedvc_dir "
                             "and seedvc_python settings")
    return unavailable


def available() -> bool:
    return resolve()["source"] != "unavailable"


def voices_dir() -> Path:
    """BuzzEdit's voices library, data/voices (created on first use)."""
    folder = DATA_DIR / "voices"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --- running it -------------------------------------------------------------------------------

def _refuse_while_rendering() -> None:
    """As routes/audio_process.py: no conversion while `routes/rendering.py` has a render running."""
    try:
        from routes import rendering
    except Exception:
        return
    for job in list(rendering.jobs.values()):
        if isinstance(job, dict) and job.get("job_type") == "render" and job.get("status") == "running":
            raise SeedVCBusy("A render is running; voice conversion waits for it")


def _run_async(coro: Any) -> Any:
    """Run a gpu_handover coroutine from this synchronous code: right here when this thread has no
    event loop, else on a helper thread with its own."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _event(line: str) -> Optional[Dict[str, Any]]:
    """A protocol line -- also when a download bar left text in front of it."""
    start = line.find('{"event"')
    if start < 0:
        return None
    try:
        obj = json.loads(line[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _result(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _tail_text(tail: Sequence[str]) -> str:
    return ("\n--- last stderr lines ---\n" + "\n".join(tail)) if tail else ""


def _spawn(env: Dict[str, Any], argv: List[str], command: str,
           progress: Optional[Callable[[float, str], None]],
           timeout_s: Optional[float]) -> Dict[str, Any]:
    cmd = [str(env["python"]), "-u", str(env["cli"]), *argv]
    if "--seedvc-dir" not in argv and env.get("seedvc_dir"):
        cmd += ["--seedvc-dir", str(env["seedvc_dir"])]
    if "--voices-dir" not in argv:
        cmd += ["--voices-dir", str(voices_dir())]
    child_env = {k: v for k, v in os.environ.items()
                 if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    child_env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    logger.info("seed-vc %s: %s", command, " ".join(cmd))

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            env=child_env, creationflags=NO_WINDOW)
    stdout_parts: List[str] = []
    reader = threading.Thread(target=lambda: stdout_parts.append(proc.stdout.read()), daemon=True)
    reader.start()
    timed_out = threading.Event()
    timer: Optional[threading.Timer] = None
    if timeout_s:
        def _expire() -> None:
            timed_out.set()
            try:
                proc.kill()
            except Exception:
                pass
        timer = threading.Timer(float(timeout_s), _expire)
        timer.daemon = True
        timer.start()

    tail: Deque[str] = collections.deque(maxlen=TAIL_LINES)
    error_message = ""
    try:
        for raw in proc.stderr:
            line = raw.rstrip("\r\n")
            event = _event(line)
            if event is None:
                if line.strip():
                    tail.append(line)
                continue
            kind = event.get("event")
            if kind == "progress":
                if progress is not None:
                    try:
                        progress(float(event.get("fraction") or 0.0), str(event.get("stage") or ""))
                    except Exception as e:
                        logger.debug("seed-vc progress callback failed: %s", e)
            elif kind == "error":
                error_message = str(event.get("message") or "")
                tail.append(line)
        code = proc.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join(timeout=10)
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except Exception:
                pass

    result = _result("".join(stdout_parts))
    if timed_out.is_set():
        raise SeedVCError(f"seed-vc {command} timed out after {float(timeout_s):g} s" + _tail_text(tail),
                          exit_code=code, stderr_tail=list(tail), result=result)
    if code == 0:
        if result is None:
            raise SeedVCError(f"seed-vc {command} exited 0 without a result" + _tail_text(tail),
                              exit_code=code, stderr_tail=list(tail))
        return result
    message = error_message or str((result or {}).get("error") or "") or "no error message"
    error_class = SeedVCUnavailable if code == EXIT_UNAVAILABLE else SeedVCError
    raise error_class(f"seed-vc {command} failed (exit {code}): {message}" + _tail_text(tail),
                      exit_code=code, stderr_tail=list(tail), result=result)


def run_cli(args: Sequence[Any], progress: Optional[Callable[[float, str], None]] = None,
            timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """Run `cli.py <args>` with the resolved seed-vc Python and return its stdout JSON.

    `args` starts with the command (`["convert", "--source", ...]`); `--seedvc-dir` and
    `--voices-dir` are added unless given. Every stderr progress line reaches
    `progress(fraction, stage)`. A non-zero exit raises SeedVCError with the CLI's error and the
    last stderr lines -- SeedVCUnavailable for exit 2 or when nothing resolves; SeedVCBusy while
    a render runs. warm / convert / apply-edit (not with --dry-run) hand the GPU over before and
    offload everything after, also when they fail."""
    argv = [str(a) for a in args]
    command = next((a for a in argv if not a.startswith("-")), "")
    if not command:
        raise ValueError("run_cli needs a command: warm, convert, apply-edit or voices list")
    env = resolve()
    if env["source"] == "unavailable":
        raise SeedVCUnavailable(f"seed-vc engine unavailable: {env.get('reason') or 'not found'}")

    gpu = command in GPU_COMMANDS and "--dry-run" not in argv
    if gpu:
        _refuse_while_rendering()
        handover = _run_async(gpu_handover.prepare_for_phase(
            GPU_PHASE, need_vram_mb=NEED_VRAM_MB, need_ram_mb=NEED_RAM_MB))
        if isinstance(handover, dict) and not handover.get("ready", True):
            logger.warning("Voice conversion starting without a full GPU hand-over: %s",
                           handover.get("reason"))
    try:
        return _spawn(env, argv, command, progress, timeout_s)
    finally:
        if gpu:
            try:
                _run_async(gpu_handover.offload_all(f"voice conversion ({command})"))
            except Exception as e:
                logger.warning("GPU offload after voice conversion failed: %s", e)


# --- the long-lived worker (brief S1) ---------------------------------------------------------

IDLE_STOP_S = 600.0          # a worker without jobs for 10 minutes stops and offloads
START_TIMEOUT_S = 900.0      # the first start downloads ~3 GB of checkpoints (149 s in S0)
QUIT_WAIT_S = 20.0


class Worker:
    """One `cli.py serve` process: the models load once, then jobs run one at a time.

    Every method blocks; call them from a thread inside the server. `progress(fraction, stage)`
    receives the job's own stderr progress (0..1 of that job)."""

    def __init__(self, idle_stop_s: float = IDLE_STOP_S, start_timeout_s: float = START_TIMEOUT_S) -> None:
        self.idle_stop_s = float(idle_stop_s)
        self.start_timeout_s = float(start_timeout_s)
        self._state_lock = threading.RLock()      # the process and its bookkeeping (held briefly)
        self._start_lock = threading.Lock()       # one start at a time
        self._job_lock = threading.Lock()         # one job at a time
        self._starting = False
        self._proc: Optional[subprocess.Popen] = None
        self._results: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._tail: Deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._progress: Optional[Callable[[float, str], None]] = None
        self._ids = itertools.count(1)
        self._timer: Optional[threading.Timer] = None
        self._busy = False
        self.device: Optional[str] = None
        self.load_s: Optional[float] = None
        self.started_s: Optional[float] = None    # wall seconds from spawn to the first ping answer
        self.jobs = 0
        self._last_used = time.monotonic()

    # -- state ------------------------------------------------------------------------------

    def running(self) -> bool:
        with self._state_lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            running = self.running()
            if not running:
                idle = None
            elif self._busy or self._starting:
                idle = 0.0
            else:
                idle = round(time.monotonic() - self._last_used, 1)
            return {"running": running, "device": self.device if running else None, "idle_s": idle,
                    "busy": running and (self._busy or self._starting), "load_s": self.load_s if running else None,
                    "pid": self._proc.pid if running and self._proc else None, "jobs": self.jobs}

    # -- the process ------------------------------------------------------------------------

    def ensure(self) -> Dict[str, Any]:
        """Start the serve process unless it runs: refuse during a render, hand the card over,
        spawn, wait for the models (a ping is answered after the load). Returns `status()`."""
        with self._start_lock:
            with self._state_lock:
                if self.running() and not self._starting:
                    return self.status()
                if self._proc is not None:               # it died since the last job
                    self._finish_stopped("voice conversion worker exited")
            _refuse_while_rendering()
            env = resolve()
            if env["source"] == "unavailable":
                raise SeedVCUnavailable("seed-vc engine unavailable: " + (env.get("reason") or "not found"))
            handover = _run_async(gpu_handover.prepare_for_phase(
                GPU_PHASE, need_vram_mb=NEED_VRAM_MB, need_ram_mb=NEED_RAM_MB))
            if isinstance(handover, dict) and not handover.get("ready", True):
                logger.warning("Voice worker starting without a full GPU hand-over: %s", handover.get("reason"))
            cmd = [str(env["python"]), "-u", str(env["cli"]), "serve",
                   "--seedvc-dir", str(env["seedvc_dir"]), "--voices-dir", str(voices_dir())]
            child_env = {k: v for k, v in os.environ.items()
                         if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
            child_env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            logger.info("seed-vc worker: %s", " ".join(cmd))
            started = time.perf_counter()
            with self._state_lock:
                self._reset()
                self._starting = True
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                              errors="replace", bufsize=1, env=child_env,
                                              creationflags=NO_WINDOW)
                proc = self._proc
            threading.Thread(target=self._read_stdout, args=(proc, self._results),
                             name="seedvc-worker-out", daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(proc,),
                             name="seedvc-worker-err", daemon=True).start()
            try:
                answer = self._send({"op": "ping"}, self.start_timeout_s)
            except BaseException:
                with self._state_lock:
                    self._starting = False
                    if self._proc is proc:
                        self._kill()
                        self._finish_stopped("voice conversion worker failed to start")
                raise
            with self._state_lock:
                self._starting = False
                self.started_s = round(time.perf_counter() - started, 2)
                self.device = answer.get("device") or self.device
                self.load_s = answer.get("load_s", self.load_s)
                self._last_used = time.monotonic()
                self._arm_timer(self.idle_stop_s)
            logger.info("seed-vc worker ready on %s: models loaded in %s s (%s s to the first answer)",
                        self.device, self.load_s, self.started_s)
            return self.status()

    def _reset(self) -> None:
        self._results = queue.Queue()
        self._tail.clear()
        self.device = None
        self.load_s = None
        self.started_s = None

    def _read_stdout(self, proc: subprocess.Popen, results: "queue.Queue[Optional[Dict[str, Any]]]") -> None:
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if not line.startswith("{"):
                    if line:
                        self._tail.append(line)
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    self._tail.append(line)
                    continue
                if isinstance(obj, dict):
                    results.put(obj)
        except Exception as e:
            logger.debug("seed-vc worker stdout closed: %s", e)
        finally:
            results.put(None)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for raw in proc.stderr:
                line = raw.rstrip("\r\n")
                event = _event(line)
                if event is None:
                    if line.strip():
                        self._tail.append(line)
                    continue
                kind = event.get("event")
                if kind == "progress":
                    callback = self._progress
                    if callback is not None:
                        try:
                            callback(float(event.get("fraction") or 0.0), str(event.get("stage") or ""))
                        except Exception as e:
                            logger.debug("seed-vc progress callback failed: %s", e)
                elif kind == "ready":
                    self.device = event.get("device") or self.device
                    self.load_s = event.get("load_s", self.load_s)
                elif kind == "error":
                    self._tail.append(line)
        except Exception as e:
            logger.debug("seed-vc worker stderr closed: %s", e)

    def _send(self, job: Dict[str, Any], timeout_s: Optional[float]) -> Dict[str, Any]:
        """Write one job line and wait for the result line with its id."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise SeedVCError("seed-vc worker is not running")
        job = {**job, "id": f"j{next(self._ids)}"}
        try:
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise SeedVCError(f"seed-vc worker did not take the job: {exc}" + _tail_text(self._tail)) from exc
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                answer = self._results.get(timeout=wait)
            except queue.Empty:
                raise SeedVCError(f"seed-vc worker gave no answer to {job['op']} within {float(timeout_s):g} s"
                                  + _tail_text(self._tail), stderr_tail=list(self._tail))
            if answer is None:
                self._results.put(None)              # every later wait sees the end too
                code = None
                try:
                    code = proc.wait(timeout=10)
                except Exception:
                    pass
                tail = list(self._tail)
                error_class = SeedVCUnavailable if code == EXIT_UNAVAILABLE else SeedVCError
                raise error_class(f"seed-vc worker exited (code {code}) during {job['op']}" + _tail_text(tail),
                                  exit_code=code, stderr_tail=tail)
            if answer.get("id") == job["id"]:
                return answer
            if "exit_code" in answer and answer.get("ok") is False:   # the process failed as a whole
                code = answer.get("exit_code")
                error_class = SeedVCUnavailable if code == EXIT_UNAVAILABLE else SeedVCError
                raise error_class(f"seed-vc worker failed (exit {code}): {answer.get('error')}"
                                  + _tail_text(self._tail), exit_code=code, stderr_tail=list(self._tail),
                                  result=answer)
            logger.debug("seed-vc worker: dropped a stale answer %s", answer.get("id"))

    # -- jobs -------------------------------------------------------------------------------

    def _job(self, job: Dict[str, Any], progress: Optional[Callable[[float, str], None]],
             timeout_s: Optional[float]) -> Dict[str, Any]:
        with self._job_lock:
            _refuse_while_rendering()
            self.ensure()
            with self._state_lock:
                self._busy = True
                self._progress = progress
            try:
                answer = self._send(job, timeout_s)
            except SeedVCError:
                if not self.running():           # the process died with the job
                    self._finish_stopped("voice conversion worker exited")
                raise
            finally:
                with self._state_lock:
                    self._busy = False
                    self._progress = None
                    self._last_used = time.monotonic()
                    self.jobs += 1
            if not answer.get("ok"):
                raise SeedVCError(f"seed-vc {job['op']} failed: {answer.get('error') or 'no error message'}",
                                  result=answer)
            if answer.get("device"):
                self.device = answer["device"]
            return answer

    def convert(self, source: Any, ref: Any, out: Any, params: Optional[Dict[str, Any]] = None,
                progress: Optional[Callable[[float, str], None]] = None,
                timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """`source` in the voice of `ref` into `out` (WAV at the engine's rate, the source's
        length times length_adjust). Returns the job's result line."""
        return self._job({"op": "convert", "source": str(source), "ref": str(ref), "out": str(out),
                          "params": dict(params or {})}, progress, timeout_s)

    def apply_edit(self, source: Any, manifest: Any, out: Any,
                   progress: Optional[Callable[[float, str], None]] = None,
                   timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """An editor manifest (dict, or a JSON file's path) over `source` into `out`."""
        return self._job({"op": "apply-edit", "source": str(source), "out": str(out),
                          "manifest": manifest if isinstance(manifest, dict) else str(manifest)},
                         progress, timeout_s)

    # -- stopping ---------------------------------------------------------------------------

    def stop(self, wait_s: float = 30.0, reason: str = "stopped") -> Dict[str, Any]:
        """Quit the process (after the running job, waiting up to `wait_s`; a job still running
        then is killed with it) and offload everything. A worker that is not running is left
        alone."""
        got = self._job_lock.acquire(timeout=max(0.0, float(wait_s)))
        try:
            return self._stop(graceful=got, reason=reason)
        finally:
            if got:
                self._job_lock.release()

    def _stop(self, graceful: bool, reason: str) -> Dict[str, Any]:
        with self._state_lock:
            proc = self._proc
            if proc is None:
                return self.status()
            self._cancel_timer()
        if proc.poll() is None and graceful:
            try:
                self._send({"op": "quit"}, QUIT_WAIT_S)
            except SeedVCError as e:
                logger.debug("seed-vc worker quit: %s", e)
            try:
                proc.wait(timeout=QUIT_WAIT_S)
            except subprocess.TimeoutExpired:
                pass
        with self._state_lock:
            if self._proc is proc:
                self._kill()
                self._finish_stopped("voice conversion worker " + reason)
            return self.status()

    def _finish_stopped(self, reason: str) -> None:
        with self._state_lock:
            self._cancel_timer()
            proc, self._proc = self._proc, None
            for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None),
                           getattr(proc, "stderr", None)):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        self._offload(reason)

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception as e:
                logger.debug("seed-vc worker kill: %s", e)

    @staticmethod
    def _offload(reason: str) -> None:
        try:
            _run_async(gpu_handover.offload_all(reason))
        except Exception as e:
            logger.warning("GPU offload after the voice worker failed: %s", e)

    # -- idle -------------------------------------------------------------------------------

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _arm_timer(self, delay_s: float) -> None:
        self._cancel_timer()
        self._timer = threading.Timer(max(0.05, delay_s), self._idle_check)
        self._timer.daemon = True
        self._timer.start()

    def _idle_check(self) -> None:
        if not self._job_lock.acquire(blocking=False):      # a job runs: look again later
            with self._state_lock:
                if self.running():
                    self._arm_timer(self.idle_stop_s)
            return
        try:
            with self._state_lock:
                if not self.running() or self._starting:
                    return
                idle = time.monotonic() - self._last_used
                if idle < self.idle_stop_s:
                    self._arm_timer(self.idle_stop_s - idle)
                    return
            logger.info("seed-vc worker idle for %.0f s: stopping", idle)
            self._stop(graceful=True, reason="idle")
        finally:
            self._job_lock.release()

    def kill_at_exit(self) -> None:
        """Backend exit: no quit handshake, no offload; the process must not outlive us."""
        self._cancel_timer()
        self._kill()


WORKER = Worker()
atexit.register(WORKER.kill_at_exit)
