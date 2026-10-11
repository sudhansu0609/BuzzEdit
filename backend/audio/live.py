"""Live audio preview: play and hear every knob as it moves (AUDIO_SUITE_PLAN.md §2.3, brief L).

One `LiveSession` per WebSocket (`routes/audio_live.py`) = one ffmpeg process that runs the
same filters as the render:

    ffmpeg -ss <t> [-t <len>] -i <source> -filter_complex
        "[0:a]asetnsamples=n=256:p=0,[afade,]<voice chain>,<rack chain>,azmq=...[out]"
        -map "[out]" -f s16le -ar 48000 -ac 2 -thread_queue_size 1 pipe:1

Always -filter_complex, never -af: a rack chain may hold ";"-separated segments with internal
labels and extra sources (amovie for a reverb IR); its head input and tail output are unlabeled
by contract, so it composes into the chain with "," and the [0:a] ... [out] ends stay valid.

The pipe is read only while the player holds credits, so ffmpeg blocks when it runs ahead.
`asetnsamples` and the output `-thread_queue_size 1` exist only here, never in the render:
ffmpeg 8 queues frames between the graph and the pipe, and with the source's own frame size
(2048 samples for a mono WAV) a live command was heard 1237 ms of audio late; 256-sample
frames and a one-packet muxer queue bring that to ~85-107 ms (measured 2026-10-09).

Live parameters reach the running graph through `azmq` (pyzmq REQ socket) as
"<filter>@<effect id>_<k> <option> <value>"; anything else restarts ffmpeg at the playhead
(debounced 120 ms, 20 ms fade-in). Commands sent before the socket answers (~500 ms after
spawn) are queued, newest value per (target, option), never dropped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from config import FFMPEG_BIN
from timeline.program_time import Segment, a1_segments, pieces_of
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_FRAME = 4            # s16le stereo
READ_CHUNK = 4096              # never read more than this from the pipe at once
STATE_INTERVAL_S = 0.25
RESTART_DEBOUNCE_S = 0.12
RESTART_FADE = "afade=t=in:d=0.02"
LIVE_HEAD = "asetnsamples=n=256:p=0"   # live preview only (see module docstring)
READY_PROBE = "_live_ready_ ping 0"    # matches no filter: azmq answers "40 ..." once it is up
MAX_SEGMENT_INPUTS = 8   # programme pieces per ffmpeg: start cost ~21 ms + ~6.6 ms per input (measured)
PAD_LABEL = "[live_pad]"

# Two literal backslashes: one level for the filtergraph parser, one for the option parser,
# so azmq receives tcp://127.0.0.1:<port> (one backslash fails: "No option name near '//...'").
_ESC = chr(92) * 2

_INSTANCE = re.compile(r"(?:^|(?<=[\],;\s]))([A-Za-z0-9_]+)@([A-Za-z0-9_.\-]+)")

SESSIONS: Set["LiveSession"] = set()   # open sessions (tests look at their processes)


class LiveError(Exception):
    """A scope that cannot be played (no source, unknown item, unsupported scope)."""


# ---------------------------------------------------------------------------------------
# Rack compilation: audio.compile (brief A1b) when present, else a private minimal compile
# that knows only "amplify" (gain_db -> volume@<id>_0).
# ---------------------------------------------------------------------------------------

@dataclass
class _MiniEffect:
    effect_id: str
    filters: List[str]
    bypass: str = "enable"
    commands: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class _MiniRack:
    filters: List[str]
    chain: str
    effects: List[_MiniEffect]


def _db(value: Any) -> str:
    return f"{float(value):.2f}dB"


def _mini_compile_rack(effects: List[Dict[str, Any]], *, scope: str = "") -> _MiniRack:
    compiled: List[_MiniEffect] = []
    for eff in effects:
        if eff.get("effect") != "amplify":
            logger.warning("live: effect %r needs audio.compile; skipped", eff.get("effect"))
            continue
        gain = (eff.get("params") or {}).get("gain_db", 0.0)
        compiled.append(_MiniEffect(eff["id"], [f"volume@{eff['id']}_0=volume={_db(gain)}"]))
    filters = [f for c in compiled for f in c.filters]
    return _MiniRack(filters=filters, chain=",".join(filters), effects=compiled)


def _mini_commands_for(rack: _MiniRack, effect_id: str, param: str, value: Any,
                       params: Dict[str, Any]) -> Optional[List[Tuple[str, str, str]]]:
    if param != "gain_db" or not any(c.effect_id == effect_id for c in rack.effects):
        return None
    return [(f"volume@{effect_id}_0", "volume", _db(value))]


def _load_compile_api():
    """`audio.compile` once brief A1b lands, else None (imported lazily, every build)."""
    try:
        from audio import compile as api  # type: ignore
    except Exception:
        return None
    if hasattr(api, "compile_rack") and hasattr(api, "commands_for"):
        return api
    return None


def _as_effect_models(effects: List[Dict[str, Any]]) -> List[Any]:
    try:
        from timeline.schema import AudioEffect  # type: ignore  (brief A1a)
    except Exception:
        return effects
    return [AudioEffect.model_validate(e) for e in effects]


class RackBuild:
    """One compiled rack and what the session needs from it: the chain, each effect's
    instance targets (with filter names) and bypass strategy, and the live command map."""

    def __init__(self, effects: List[Dict[str, Any]], scope: str):
        api = _load_compile_api()
        self._api = api
        if api is not None:
            self.rack = api.compile_rack(_as_effect_models(effects), scope=scope)
        else:
            self.rack = _mini_compile_rack(effects, scope=scope)
        chain = getattr(self.rack, "chain", "") or ",".join(getattr(self.rack, "filters", []) or [])
        self.chain: str = chain
        self.bypass: Dict[str, str] = {}
        self.targets: Dict[str, List[Tuple[str, str]]] = {}   # effect id -> [(target, filter)]
        for ce in getattr(self.rack, "effects", []) or []:
            self.bypass[ce.effect_id] = getattr(ce, "bypass", "restart") or "restart"
            found: List[Tuple[str, str]] = []
            for f in ce.filters:
                for m in _INSTANCE.finditer(f):   # labels may precede the instance
                    found.append((f"{m.group(1)}@{m.group(2)}", m.group(1)))
            self.targets[ce.effect_id] = found

    def commands_for(self, effect_id: str, param: str, value: Any,
                     params: Dict[str, Any]) -> Optional[List[Tuple[str, str, str]]]:
        if self._api is not None:
            out = self._api.commands_for(self.rack, effect_id, param, value, params)
        else:
            out = _mini_commands_for(self.rack, effect_id, param, value, params)
        return [tuple(map(str, c)) for c in out] if out else None   # type: ignore[misc]


# ---------------------------------------------------------------------------------------
# What a scope plays
# ---------------------------------------------------------------------------------------

@dataclass
class LiveSource:
    scope: str
    path: str                                   # audio input (a source's audio_override wins)
    voice_filters: List[str]                    # build_voice_chain(...) for A1 / MASTER
    effects: List[Dict[str, Any]]               # the scope's rack, as dicts
    offset_s: float = 0.0                       # item scope: the clip's in-point in its source
    length_s: Optional[float] = None            # item scope: clip length (play stops there)
    segments: List[Segment] = field(default_factory=list)   # A1/MASTER: the cut programme

    @property
    def program_end_s(self) -> Optional[float]:
        return self.segments[-1].prog_end_s if self.segments else None


def _dump(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _find_item(items: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    for it in items or []:
        if it.get("id") == item_id:
            return it
        found = _find_item(it.get("children") or [], item_id)
        if found:
            return found
    return None


def resolve_scope(p_data: Dict[str, Any], scope: str) -> LiveSource:
    """Source, voice chain and rack for "A1", "MASTER" or "item:<id>" of a stored project."""
    from render.audio import build_voice_chain
    from timeline import Timeline

    raw = p_data.get("timeline")
    if not raw:
        raise LiveError("project has no timeline")
    tl = Timeline.model_validate(raw)
    raw_sources: Dict[str, Any] = raw.get("sources") or {}

    def override(source_id: Optional[str]) -> Optional[str]:
        if not source_id:
            return None
        model = tl.sources.get(source_id)
        value = getattr(model, "audio_override", None) if model is not None else None
        return value or (raw_sources.get(source_id) or {}).get("audio_override") or None

    if scope in ("A1", "MASTER"):
        # The cut programme, in programme time, when the timeline has A1 trims; otherwise
        # the uncut source_video as before.
        segments = a1_segments(tl)
        source_video = p_data.get("source_video")
        if not source_video and not segments:
            raise LiveError("project has no source_video and no A1 clips")
        path = source_video or segments[0].path
        for sid, src in raw_sources.items():
            if src.get("path") and _norm(src["path"]) == _norm(source_video):
                path = override(sid) or source_video
                break
        racks = getattr(tl, "audio_racks", None)
        if racks is None:
            racks = raw.get("audio_racks") or {}
        effects = [_dump(e) for e in (racks.get(scope) or [])]
        return LiveSource(scope, path, build_voice_chain(tl.audio_master), effects, segments=segments)

    if scope.startswith("item:"):
        item_id = scope[len("item:"):]
        item = _find_item(raw.get("items") or [], item_id)
        if item is None:
            raise LiveError(f"item {item_id} not found")
        source_id = item.get("source_id")
        src = raw_sources.get(source_id or "")
        if not src:
            raise LiveError(f"item {item_id} has no source")
        fps = float(tl.fps_num) / float(tl.fps_den or 1)
        start_f = int(item.get("source_start_frame") or 0)
        end_f = int(item.get("source_end_frame") or 0)
        length = (end_f - start_f) / fps if end_f > start_f else None
        effects = [_dump(e) for e in (item.get("audio_effects") or [])]
        return LiveSource(scope, override(source_id) or src["path"], [], effects,
                          offset_s=start_f / fps, length_s=length)

    raise LiveError(f"unsupported scope {scope!r} (use A1, MASTER or item:<id>)")


# ---------------------------------------------------------------------------------------
# ffmpeg command
# ---------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def azmq_filter(port: int) -> str:
    return f"azmq=bind_address=tcp{_ESC}://127.0.0.1{_ESC}:{port}"


def build_graph(voice_filters: List[str], rack_chain: str, port: int, fade: bool,
                head: str = "[0:a]", t0: Optional[float] = None) -> str:
    """`head` feeds the chain (an input pad, or a concat ending in a comma); with `t0` the
    timestamps are moved to start at t0, so `t` in the voice chain and the rack is programme
    time, as in the render."""
    parts = [LIVE_HEAD]
    if fade:
        parts.append(RESTART_FADE)
    if t0 is not None:
        parts.append(f"asetpts=PTS-STARTPTS+{t0:.6f}/TB")
    parts.extend(voice_filters)
    if rack_chain:
        parts.append(rack_chain)
    parts.append(azmq_filter(port))
    return head + ",".join(parts) + "[out]"


def program_head(n_inputs: int, pad_s: float) -> str:
    """Silence for the leading programme offset, then the inputs, joined by concat."""
    prefix, members = "", [f"[{k}:a]" for k in range(n_inputs)]
    if pad_s > 0.0005:
        prefix = f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={pad_s:.6f}{PAD_LABEL};"
        members.insert(0, PAD_LABEL)
    if len(members) == 1:
        return prefix + members[0]
    return prefix + "".join(members) + f"concat=n={len(members)}:v=0:a=1,"


def build_program_command(pieces: List[Tuple[str, float, float]], graph: str) -> List[str]:
    """One -ss/-t input per programme piece (path, source start, seconds)."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error"]
    for path, ss, duration in pieces:
        cmd += ["-ss", f"{max(0.0, ss):.6f}", "-t", f"{duration:.6f}", "-i", path]
    cmd += ["-filter_complex", graph, "-map", "[out]",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-thread_queue_size", "1", "pipe:1"]
    return cmd


def build_command(path: str, ss: float, duration: Optional[float], graph: str) -> List[str]:
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, ss):.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-filter_complex", graph, "-map", "[out]",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-thread_queue_size", "1", "pipe:1"]
    return cmd


# ---------------------------------------------------------------------------------------
# zmq link: one REQ socket to one ffmpeg's azmq, owned by its own thread (a zmq socket must
# stay on one thread). azmq answers only when a frame passes it, so a reply can wait while
# playback is paused; nothing here blocks the event loop.
# ---------------------------------------------------------------------------------------

class ZmqLink:
    def __init__(self, port: int, on_failure: Optional[Callable[[str, str, str, str], None]] = None):
        self.port = port
        self._on_failure = on_failure
        self._pending: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        self._cv = threading.Condition()
        self._stop = False
        self.ready = threading.Event()
        self.started_at = time.perf_counter()
        self.ready_at: Optional[float] = None
        self.sent: List[Tuple[float, str]] = []      # (perf_counter, message), for diagnostics
        self._thread = threading.Thread(target=self._run, name=f"azmq-{port}", daemon=True)
        self._thread.start()

    def submit(self, target: str, option: str, value: str) -> None:
        """Queue a command; a newer value for the same (target, option) replaces the old one."""
        with self._cv:
            key = (target, option)
            self._pending.pop(key, None)
            self._pending[key] = value
            self._cv.notify()

    def stop(self) -> None:
        """Non-blocking: the thread closes the socket within ~50 ms."""
        with self._cv:
            self._stop = True
            self._cv.notify()

    def close(self) -> None:
        self.stop()
        self._thread.join(timeout=2.0)

    def _await_reply(self, sock) -> Optional[str]:
        while not self._stop:
            if sock.poll(50):
                return sock.recv_string()
        return None

    def _run(self) -> None:
        import zmq

        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(f"tcp://127.0.0.1:{self.port}")
            sock.send_string(READY_PROBE)
            if self._await_reply(sock) is None:
                return
            self.ready_at = time.perf_counter()
            self.ready.set()
            while True:
                with self._cv:
                    while not self._pending and not self._stop:
                        self._cv.wait(0.1)
                    if self._stop:
                        return
                    (target, option), value = self._pending.popitem(last=False)
                message = f"{target} {option} {value}"
                sock.send_string(message)
                self.sent.append((time.perf_counter(), message))
                reply = self._await_reply(sock)
                if reply is None:
                    return
                if not reply.startswith("0") and self._on_failure is not None:
                    self._on_failure(target, option, value, reply)
        except Exception as exc:   # a dead ffmpeg must never take the session down
            logger.debug("azmq link %s: %s", self.port, exc)
        finally:
            sock.close(0)


# ---------------------------------------------------------------------------------------
# process helpers (run in threads)
# ---------------------------------------------------------------------------------------

def _popen(cmd: List[str]) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0, creationflags=NO_WINDOW)
    proc.last_stderr = ""   # type: ignore[attr-defined]

    def drain() -> None:
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode("utf-8", "replace").strip()
                if text:
                    proc.last_stderr = text   # type: ignore[attr-defined]
        except Exception:
            pass
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass

    threading.Thread(target=drain, name="ffmpeg-live-stderr", daemon=True).start()
    return proc


def _read_pipe(proc: subprocess.Popen, n: int) -> bytes:
    try:
        return proc.stdout.read(n) or b""
    except (OSError, ValueError):
        return b""


def _terminate(proc: subprocess.Popen) -> None:
    proc.live_killed = True   # type: ignore[attr-defined]
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    except Exception as exc:
        logger.warning("live: could not stop ffmpeg %s: %s", proc.pid, exc)
    try:
        proc.stdout.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------------------

SendJson = Callable[[Dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


class LiveSession:
    """One open panel: one ffmpeg at a time, PCM out only against the client's credits."""

    def __init__(self, source: LiveSource, send_json: SendJson, send_bytes: SendBytes):
        self.source = source
        self._send_json = send_json
        self._send_bytes = send_bytes
        self.effects: List[Dict[str, Any]] = [self._normal(e) for e in source.effects]
        self.bypassed: Set[str] = set()
        self.playing = False
        self.credits = 0                       # frames the client may still receive
        self.start_t = 0.0                     # session time where the running ffmpeg started
        self.frames_sent = 0                   # frames sent since that start
        self.loop: Optional[Tuple[float, float]] = None
        self.spawned: List[subprocess.Popen] = []
        self._proc: Optional[subprocess.Popen] = None
        self._link: Optional[ZmqLink] = None
        self._build: Optional[RackBuild] = None
        self._dirty: Dict[str, Set[str]] = {}  # params changed while their effect was bypassed
        self._gen = 0                          # bumps on every kill/spawn: stale reads are dropped
        self._eof_gen = -1
        self._ended = False
        self._chunk_end: Optional[float] = None   # programme time where the running ffmpeg stops
        self._carry = b""
        self._wake = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()
        self._debounce: Optional[asyncio.Future] = None
        self._tasks: List[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._socket_gone = False

    # -- lifecycle ----------------------------------------------------------------------

    @staticmethod
    def _normal(effect: Any) -> Dict[str, Any]:
        d = dict(_dump(effect) or {})
        if not d.get("id") or not d.get("effect"):
            raise LiveError("every rack effect needs an id and an effect")
        d["params"] = dict(d.get("params") or {})
        d.setdefault("enabled", True)
        return d

    def position(self) -> float:
        return self.start_t + self.frames_sent / SAMPLE_RATE

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        SESSIONS.add(self)
        self._tasks = [asyncio.create_task(self._pump()), asyncio.create_task(self._state_loop())]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.playing = False
        SESSIONS.discard(self)
        # Kill first, synchronously: a close cut short by cancellation leaves no ffmpeg behind.
        proc, link = self._detach()
        if self._debounce is not None:
            self._debounce.cancel()
        for task in self._tasks:
            task.cancel()
        self._wake.set()
        if link is not None:
            await asyncio.to_thread(link.close)
        if proc is not None:
            await asyncio.to_thread(_terminate, proc)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- client messages ----------------------------------------------------------------

    async def handle(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("type")
        try:
            if kind == "credits":
                self.credits += max(0, int(msg.get("frames") or 0))
                self._wake.set()
            elif kind == "play":
                await self._play()
            elif kind == "pause":
                self.playing = False
            elif kind == "seek":
                await self._seek(float(msg.get("t") or 0.0))
            elif kind == "loop":
                await self._set_loop(msg)
            elif kind == "set":
                await self._set(str(msg["effect_id"]), str(msg["param"]), msg.get("value"))
            elif kind == "bypass":
                await self._bypass(str(msg["effect_id"]), bool(msg.get("on")))
            elif kind == "rack":
                await self._replace_rack(list(msg.get("effects") or []))
            else:
                await self._error(f"unknown message type {kind!r}")
        except (KeyError, TypeError, ValueError, LiveError) as exc:
            await self._error(f"bad {kind} message: {exc}")

    async def _play(self) -> None:
        if self._proc is None:
            t = self.position()
            if self._ended:
                t = self.loop[0] if self.loop else 0.0
            if not await self._spawn_safe(t, fade=False):
                return
        self.playing = self._proc is not None
        self._wake.set()

    async def _seek(self, t: float) -> None:
        t = max(0.0, t)
        if self._proc is not None:
            await self._restart(t)
        else:
            self.start_t, self.frames_sent, self._ended = t, 0, False
            await self._send({"type": "restarted", "t": round(t, 4)})

    async def _set_loop(self, msg: Dict[str, Any]) -> None:
        if msg.get("off"):
            if self.loop is not None:
                self.loop = None
                if self._proc is not None:
                    await self._restart(None)
            return
        a, b = float(msg["a"]), float(msg["b"])
        if not 0.0 <= a < b:
            raise ValueError("loop needs 0 <= a < b")
        self.loop = (a, b)
        await self._seek(a)

    def _effect(self, effect_id: str) -> Dict[str, Any]:
        for eff in self.effects:
            if eff["id"] == effect_id:
                return eff
        raise LiveError(f"no effect {effect_id} in the rack")

    async def _set(self, effect_id: str, param: str, value: Any) -> None:
        eff = self._effect(effect_id)
        eff["params"][param] = value
        if self._proc is None or not eff.get("enabled", True):
            return                                  # compiled into the next start
        if effect_id in self.bypassed:
            self._dirty.setdefault(effect_id, set()).add(param)
            return                                  # applied on un-bypass
        build, link = self._build, self._link
        cmds = None
        if build is not None and effect_id in build.bypass:
            cmds = build.commands_for(effect_id, param, value, eff["params"])
        if cmds and link is not None:
            for target, option, val in cmds:
                link.submit(target, option, val)
        else:
            self._schedule_restart()

    def _enable_expr(self, eff: Dict[str, Any]) -> str:
        a, b = eff.get("start_s"), eff.get("end_s")
        if a is not None and b is not None:
            return f"between(t,{float(a):.3f},{float(b):.3f})"
        if a is not None:
            return f"gte(t,{float(a):.3f})"
        if b is not None:
            return f"lte(t,{float(b):.3f})"
        return "1"

    def _restore_params(self, eff: Dict[str, Any], require_any: bool) -> bool:
        """Re-send an un-bypassed effect's params live; False when a restart is needed."""
        build, link = self._build, self._link
        dirty = self._dirty.pop(eff["id"], set())
        if build is None or link is None:
            return False
        sent = False
        for param, value in eff["params"].items():
            cmds = build.commands_for(eff["id"], param, value, eff["params"])
            if cmds:
                for target, option, val in cmds:
                    link.submit(target, option, val)
                sent = True
            elif param in dirty:
                return False
        return sent or not require_any

    async def _bypass(self, effect_id: str, on: bool) -> None:
        eff = self._effect(effect_id)
        if on == (effect_id in self.bypassed):
            return
        (self.bypassed.add if on else self.bypassed.discard)(effect_id)
        if self._proc is None or not eff.get("enabled", True):
            return
        build, link = self._build, self._link
        strategy = build.bypass.get(effect_id) if build is not None else None
        targets = build.targets.get(effect_id, []) if build is not None else []
        if strategy is None or link is None:
            if not on:                              # not in the running chain: bring it in
                self._schedule_restart()
            return
        ok = True
        if strategy == "enable" and targets:
            for target, _ in targets:
                link.submit(target, "enable", "0" if on else self._enable_expr(eff))
            if not on:
                ok = self._restore_params(eff, require_any=False)
        elif strategy == "mix" and targets:
            mixed = [t for t, f in targets if f in ("acompressor", "arnndn")] or [t for t, _ in targets]
            for target in mixed:
                link.submit(target, "mix", "0" if on else "1")
            if not on:
                ok = self._restore_params(eff, require_any=False)
        elif strategy == "flat" and targets:
            if on:
                for target in [t for t, f in targets if f == "firequalizer"] or [t for t, _ in targets]:
                    link.submit(target, "gain_entry", "entry(0,0)")
            else:
                ok = self._restore_params(eff, require_any=True)
        else:                                       # "restart"
            ok = False
        if not ok:
            self._schedule_restart()

    async def _replace_rack(self, effects: List[Any]) -> None:
        self.effects = [self._normal(e) for e in effects]
        self.bypassed &= {e["id"] for e in self.effects}
        self._dirty.clear()
        if self._proc is not None:
            await self._restart(None)

    # -- process control ----------------------------------------------------------------

    def _window(self, t: float) -> Tuple[float, Optional[float]]:
        """(start, duration) for a start at session time t; duration None = to the end."""
        duration: Optional[float] = None
        if self.loop is not None:
            a, b = self.loop
            if not a <= t < b - 0.005:
                t = a
            duration = b - t
        length = self.source.program_end_s if self.source.segments else self.source.length_s
        if length is not None:
            rest = length - t
            duration = rest if duration is None else min(duration, rest)
        return t, duration

    def _program_plan(self, t: float, duration: float) -> Tuple[List[Tuple[str, float, float]], float, float]:
        """Inputs (path, source start, seconds) for programme [t, t + duration), at most
        MAX_SEGMENT_INPUTS of them, the leading silence, and where this chunk ends."""
        end = t + duration
        first = self.source.segments[0].prog_start_s
        pad = max(0.0, min(end, first) - t)
        inputs = [(path, a, b - a) for _sid, path, a, b in pieces_of(self.source.segments, t, end)]
        inputs = inputs[:MAX_SEGMENT_INPUTS]
        chunk_end = end
        if inputs:
            covered = sum(d for _p, _a, d in inputs)
            chunk_end = min(end, max(t, first) + covered)
        return inputs, pad, chunk_end

    async def _spawn(self, t: float, fade: bool) -> None:
        t, duration = self._window(max(0.0, t))
        if duration is not None and duration <= 0.005:
            self.start_t, self.frames_sent = t, 0
            await self._finish_ended()
            return
        live = [e for e in self.effects if e.get("enabled", True) and e["id"] not in self.bypassed]
        build = RackBuild(live, self.source.scope)
        port = free_port()
        if self.source.segments:
            # The cut programme: concat of the pieces from the playhead, timestamps in
            # programme time; the voice chain and the rack come after the concat.
            inputs, pad, self._chunk_end = self._program_plan(t, duration)
            graph = build_graph(self.source.voice_filters, build.chain, port, fade,
                                head=program_head(len(inputs), pad), t0=t)
            cmd = build_program_command(inputs, graph)
        else:
            self._chunk_end = None
            graph = build_graph(self.source.voice_filters, build.chain, port, fade)
            cmd = build_command(self.source.path, self.source.offset_s + t, duration, graph)
        proc = await asyncio.to_thread(_popen, cmd)
        self._gen += 1
        self._proc, self._build = proc, build
        self._link = ZmqLink(port, self._command_failed)
        self._dirty.clear()
        self.spawned.append(proc)
        self.start_t, self.frames_sent, self._carry, self._ended = t, 0, b"", False
        asyncio.create_task(self._watch(proc, self._gen))
        self._wake.set()

    async def _spawn_safe(self, t: float, fade: bool) -> bool:
        try:
            await self._spawn(t, fade)
            return True
        except Exception as exc:
            self.playing = False
            await self._error(f"could not start ffmpeg: {exc}")
            return False

    def _detach(self) -> Tuple[Optional[subprocess.Popen], Optional[ZmqLink]]:
        """Forget the running ffmpeg and its link, kill it and stop the link without waiting."""
        proc, link = self._proc, self._link
        self._proc, self._link, self._build = None, None, None
        self._gen += 1
        self._carry = b""
        if proc is not None:
            proc.live_killed = True   # type: ignore[attr-defined]
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        if link is not None:
            link.stop()
        return proc, link

    async def _kill(self) -> None:
        proc, link = self._detach()
        if link is not None:
            await asyncio.to_thread(link.close)
        if proc is not None:
            await asyncio.to_thread(_terminate, proc)

    async def _restart(self, t: Optional[float], fade: bool = True, notify: bool = True) -> None:
        async with self._restart_lock:
            if self._closed:
                return
            if self._debounce is not None and self._debounce is not asyncio.current_task():
                self._debounce.cancel()
                self._debounce = None
            at = self.position() if t is None else t
            await self._kill()
            if await self._spawn_safe(at, fade) and self._proc is not None and notify:
                await self._send({"type": "restarted", "t": round(self.start_t, 4)})

    def _schedule_restart(self) -> None:
        if self._closed:
            return
        if self._debounce is not None:
            self._debounce.cancel()
        self._debounce = asyncio.ensure_future(self._debounced_restart())

    async def _debounced_restart(self) -> None:
        try:
            await asyncio.sleep(RESTART_DEBOUNCE_S)
        except asyncio.CancelledError:
            return
        self._debounce = None
        if self._proc is not None:
            await self._restart(None)

    def _command_failed(self, target: str, option: str, value: str, reply: str) -> None:
        # Called on the link's thread: a refused live command falls back to a restart.
        logger.info("live: '%s %s %s' refused (%s); restarting", target, option, value, reply)
        if self._loop is not None and not self._closed:
            self._loop.call_soon_threadsafe(self._schedule_restart)

    async def _watch(self, proc: subprocess.Popen, gen: int) -> None:
        rc = await asyncio.to_thread(proc.wait)
        if rc != 0 and gen == self._gen and not self._closed:
            await self._report_failure(proc, rc)

    async def _report_failure(self, proc: subprocess.Popen, rc: int) -> None:
        if getattr(proc, "live_killed", False) or getattr(proc, "live_reported", False):
            return
        proc.live_reported = True   # type: ignore[attr-defined]
        for _ in range(20):         # let the stderr thread take the last line
            if proc.stderr is None or proc.stderr.closed:
                break
            await asyncio.sleep(0.01)
        last = getattr(proc, "last_stderr", "")
        await self._kill()
        self.playing = False
        await self._error(f"ffmpeg stopped (exit code {rc})" + (f": {last}" if last else ""))

    async def _on_eof(self, proc: subprocess.Popen, gen: int) -> None:
        rc = await asyncio.to_thread(proc.wait)
        if gen != self._gen or self._closed:
            return
        limit = self.loop[1] if self.loop is not None else self.source.program_end_s
        if rc != 0:
            await self._report_failure(proc, rc)
        elif (self._chunk_end is not None and limit is not None
              and self._chunk_end < limit - 0.002):
            # More programme than one ffmpeg takes: carry on where this one stopped.
            await self._restart(self._chunk_end, fade=False, notify=False)
        elif self.loop is not None:
            await self._restart(self.loop[0])
        else:
            await self._kill()
            await self._finish_ended()

    async def _finish_ended(self) -> None:
        self._ended = True
        self.playing = False
        await self._send({"type": "ended"})

    # -- streaming ----------------------------------------------------------------------

    async def _pump(self) -> None:
        while not self._closed:
            proc = self._proc
            if proc is None or not self.playing or self.credits <= 0 or self._eof_gen == self._gen:
                self._wake.clear()
                await self._wake.wait()
                continue
            gen = self._gen
            want = min(READ_CHUNK, self.credits * BYTES_PER_FRAME - len(self._carry))
            data = await asyncio.to_thread(_read_pipe, proc, want)
            if gen != self._gen or self._closed:
                continue                            # restarted meanwhile: old audio is dropped
            if not data:
                self._eof_gen = gen
                await self._on_eof(proc, gen)
                continue
            buf = self._carry + data
            frames = min(len(buf) // BYTES_PER_FRAME, self.credits)
            cut = frames * BYTES_PER_FRAME
            self._carry = buf[cut:]
            if frames:
                self.credits -= frames
                self.frames_sent += frames
                await self._send(buf[:cut])

    async def _state_loop(self) -> None:
        while not self._closed:
            await self._send({"type": "state", "playing": self.playing, "t": round(self.position(), 3)})
            await asyncio.sleep(STATE_INTERVAL_S)

    async def _error(self, message: str) -> None:
        logger.info("live %s: %s", self.source.scope, message)
        await self._send({"type": "error", "message": message})

    async def _send(self, payload: Any) -> None:
        if self._socket_gone:
            return
        async with self._send_lock:
            try:
                if isinstance(payload, (bytes, bytearray)):
                    await self._send_bytes(bytes(payload))
                else:
                    await self._send_json(payload)
            except Exception:
                self._socket_gone = True            # the route's receive loop closes the session
