"""Narration: text -> a WAV in a voice, with a local TTS engine (AUTO_STORY_PLAN.md brief N1, §3.1).

The engine runs under its own Python (`engines/tts/.venv`) as a subprocess, `engines/tts/cli.py`
(JSON progress on stderr, one JSON result on stdout, exit 2 = engine unavailable); nothing of it
loads in this process. Two voice paths:

* `chatterbox` -- the voice is a reference clip in `data/voices` (`sudhansu`): a direct clone,
  intonation by `style.exaggeration`. Chosen by default for a reference voice (or `"engine"`,
  Chatterbox's own voice).
* `kokoro` -- a Kokoro preset (`hm_omega`, `hf_alpha`, ...); with `convert_to` the result then
  goes through seed-vc into that reference voice (`seedvc_env.run_cli("convert", ...)`, which
  does its own GPU hand-over and offload).

A Chatterbox reference clip whose sidecar `<voice>.seedvc.json` carries `"convert_after": true`
(VOICE_DELIVERY_PLAN.md F8 -- `sudhansu_horror`, cloned for its grave intonation, not his real
timbre) always goes through seed-vc afterwards too, the fine-tuned way: Chatterbox speaks it at
pace 1.0 (the requested pace is realised by seed-vc's own `length_adjust = 1/pace` instead -- no
time-stretch), `run_cli(["convert", ...])` uses the sidecar's `checkpoint`/`config`/
`diffusion_steps`/`inference_cfg_rate` and `convert_ref` (kept out of `data/voices` itself, so it
never shows up as a selectable voice), and `silence_gate.gate_files()` re-imposes the Chatterbox
take's own silences on the result (seed-vc fills true silence with a soft floor of its own). A
failed conversion raises -- there is no silent fallback to the unconverted clone.

GPU rule (as the seed-vc worker): refused while a render runs, a resident seed-vc worker is
stopped (one model at a time), the card is handed over before the subprocess and everything is
offloaded after it, also when it fails. The CLI retries a failed batch once; the result says how
many were retried. Jobs (`start_job` / `get_job`) run on a thread, one synthesis at a time.

`style` (mood, exaggeration, cfg_weight, temperature, pace, word_gap_ms, phrase_pause_s,
sentence_pause_s; VOICE_DELIVERY_PLAN.md) is range-checked here and resolved once by
`cli.resolve_style()` (engine defaults <- mood preset <- the caller's explicit keys), then sent
to the CLI as one `--style` JSON argument; `--exaggeration`/`--pace` are still passed too for
backward compatibility. `list_moods()` is `GET /api/narration/moods`.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from audio import seedvc_env, silence_gate
from config import BASE_DIR, DATA_DIR
from runtime import gpu_handover, job_store
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)

TTS_DIR = BASE_DIR / "engines" / "tts"
CLI_PATH = TTS_DIR / "cli.py"
GPU_PHASE = "narration"
NEED_VRAM_MB = 6000
NEED_RAM_MB = 8000
EXIT_UNAVAILABLE = 2
TAIL_LINES = 25
LANGUAGES = ("hi", "en")
ENGINES = ("chatterbox", "kokoro")
# style.<key> ranges (VOICE_DELIVERY_PLAN.md's "Contract: style" table); mood and the layering
# of engine defaults <- mood preset <- these keys is `cli.resolve_style()`.
STYLE_RANGES: Dict[str, "tuple[float, float]"] = {
    "exaggeration": (0.0, 2.0), "cfg_weight": (0.0, 1.0), "temperature": (0.1, 1.5),
    "pace": (0.5, 2.0), "word_gap_ms": (0.0, 300.0), "phrase_pause_s": (0.0, 2.0),
    "sentence_pause_s": (0.0, 3.0),
}
# A convert_after voice (VOICE_DELIVERY_PLAN.md F8): length_adjust = 1/pace, clamped here --
# seed-vc's own range is wider (engines/seedvc/cli.py's merge_params), this is just how far a
# narration pace is allowed to push it.
CONVERT_AFTER_LENGTH_ADJUST_RANGE = (0.5, 2.0)

Progress = Optional[Callable[[float, str], None]]


class NarrationError(RuntimeError):
    """The CLI failed; the message carries its error and the last stderr lines."""


class NarrationUnavailable(NarrationError):
    """No TTS Python / CLI, or the CLI exited 2 (its engine import failed)."""


class NarrationBusy(NarrationError):
    """A render is running; narration waits for it."""


# --------------------------------------------------------------------------- the engine

_cli_module_cache: Dict[str, Any] = {}


def cli_module() -> Any:
    """`engines/tts/cli.py` as a module: its text handling and preset list are plain Python."""
    if "cli" not in _cli_module_cache:
        spec = importlib.util.spec_from_file_location("buzzedit_tts_cli", str(CLI_PATH))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _cli_module_cache["cli"] = module
    return _cli_module_cache["cli"]


def tts_python() -> Path:
    return seedvc_env._venv_python(TTS_DIR / ".venv")


def available() -> Dict[str, Any]:
    python = tts_python()
    if not CLI_PATH.is_file():
        return {"available": False, "reason": f"{CLI_PATH} is missing"}
    if not python.is_file():
        return {"available": False, "reason": f"no TTS Python at {python}"}
    return {"available": True, "reason": ""}


def voices_dir() -> Path:
    return seedvc_env.voices_dir()


def list_voices() -> List[Dict[str, Any]]:
    """The reference clips in data/voices and the Kokoro presets."""
    return cli_module().list_voices(voices_dir())


def list_moods() -> List[Dict[str, Any]]:
    """`GET /api/narration/moods`: the mood presets (`cli.MOODS` is the single source)."""
    return cli_module().list_moods()


def _reference_stem(voice: str) -> Optional[str]:
    path = cli_module().reference_path(voice, voices_dir()) if voice else None
    return path.stem if path is not None else None


def _voice_convert_sidecar(voice: str) -> Optional[Dict[str, Any]]:
    """`<voice>.seedvc.json` beside a Chatterbox reference clip (VOICE_DELIVERY_PLAN.md F8):
    `{"convert_after": true, "checkpoint", "config", "convert_ref", "diffusion_steps"?,
    "inference_cfg_rate"?}` when this voice is cloned for its intonation only and always goes
    through seed-vc into the creator's real timbre afterwards. `None` when there is no
    reference clip, no sidecar, or the sidecar does not say `convert_after`."""
    path = cli_module().reference_path(voice, voices_dir()) if voice else None
    if path is None:
        return None
    sidecar = path.with_suffix(".seedvc.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except Exception as e:
        logger.warning("voice sidecar %s unreadable: %s", sidecar, e)
        return None
    return data if isinstance(data, dict) and data.get("convert_after") else None


def resolve_request(text: Any, language: Any = "hi", voice: Any = "sudhansu", engine: Any = None,
                    style: Optional[Dict[str, Any]] = None, convert_to: Any = None) -> Dict[str, Any]:
    """Check a request and settle its engine. ValueError says what is wrong."""
    cli = cli_module()
    text = str(text or "").strip()
    if not text:
        raise ValueError("Give the text to narrate")
    if not any("text" in item for item in cli.plan_batches(text)):
        raise ValueError("Nothing to say once the cues are removed")
    language = str(language or "hi").strip().lower()
    if language not in LANGUAGES:
        raise ValueError(f"language is one of {', '.join(LANGUAGES)}, not {language!r}")
    voice = str(voice or "sudhansu").strip()
    is_engine_voice = voice == cli.ENGINE_VOICE
    reference = None if is_engine_voice else _reference_stem(voice)
    preset = cli.kokoro_preset(voice)
    if engine is None or str(engine).strip() == "":
        if reference or is_engine_voice:
            engine = "chatterbox"
        elif preset:
            engine = "kokoro"
        else:
            raise ValueError(f"Unknown voice {voice!r}: not in data/voices and not a Kokoro preset")
    engine = str(engine).strip().lower()
    if engine not in ENGINES:
        raise ValueError(f"engine is one of {', '.join(ENGINES)}, not {engine!r}")
    if engine == "chatterbox" and not (reference or is_engine_voice):
        raise ValueError(f"Chatterbox clones a voice from data/voices; {voice!r} is not there")
    if engine == "kokoro" and not (preset or is_engine_voice):
        raise ValueError(f"Kokoro speaks its presets; {voice!r} is not one "
                         "(use convert_to for a data/voices voice)")
    style = dict(style or {})
    mood = style.get("mood")
    if mood is not None and str(mood).strip().lower() not in cli.MOODS:
        raise ValueError(f"style.mood is one of {', '.join(cli.MOODS)}, not {mood!r}")
    for key, (low, high) in STYLE_RANGES.items():
        if key in style:
            value = float(style[key])
            if not low <= value <= high:
                raise ValueError(f"style.{key} is {low:g}-{high:g}")
    resolved_style = cli.resolve_style(style)      # engine defaults <- mood preset <- these keys
    exaggeration = resolved_style["exaggeration"]
    pace = resolved_style["pace"]
    target = None
    if convert_to:
        target = _reference_stem(str(convert_to).strip())
        if target is None:
            raise ValueError(f"convert_to {convert_to!r} is not a voice in data/voices")
    return {"text": text, "language": language, "voice": reference or voice, "engine": engine,
            "exaggeration": exaggeration, "pace": pace, "style": resolved_style, "convert_to": target}


def refuse_while_rendering() -> None:
    try:
        seedvc_env._refuse_while_rendering()
    except seedvc_env.SeedVCBusy as exc:
        raise NarrationBusy("A render is running; narration waits for it") from exc


def _run_tts(argv: List[str], progress: Progress = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """Run `cli.py <argv>` under the TTS Python; return its stdout JSON."""
    state = available()
    if not state["available"]:
        raise NarrationUnavailable(f"TTS engine unavailable: {state['reason']}")
    cmd = [str(tts_python()), "-u", str(CLI_PATH), *argv]
    child_env = {k: v for k, v in os.environ.items()
                 if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    child_env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    logger.info("tts: %s", " ".join(cmd))
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
    try:
        for raw in proc.stderr:
            line = raw.rstrip("\r\n")
            start = line.find('{"progress"')
            event = None
            if start >= 0:
                try:
                    event = json.loads(line[start:])
                except ValueError:
                    event = None
            if isinstance(event, dict):
                if progress is not None:
                    try:
                        progress(float(event.get("progress") or 0.0), str(event.get("message") or ""))
                    except Exception as e:
                        logger.debug("tts progress callback failed: %s", e)
            elif line.strip():
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
    result = seedvc_env._result("".join(stdout_parts))
    tail_text = seedvc_env._tail_text(tail)
    if timed_out.is_set():
        raise NarrationError(f"TTS timed out after {float(timeout_s):g} s" + tail_text)
    if code == 0 and result is not None and not result.get("error"):
        return result
    message = str((result or {}).get("error") or "") or "no error message"
    if code == EXIT_UNAVAILABLE:
        raise NarrationUnavailable(f"TTS engine unavailable: {message}" + tail_text)
    raise NarrationError(f"TTS failed (exit {code}): {message}" + tail_text)


def _timeout_for(text: str) -> float:
    """Model load (~100 s) plus ~3.3 s of compute per second of speech, with room to spare."""
    return 900.0 + 0.6 * len(text)


def synthesize(text: str, *, language: str = "hi", voice: str = "sudhansu", engine: Optional[str] = None,
               style: Optional[Dict[str, Any]] = None, convert_to: Optional[str] = None,
               out_dir: Optional[str] = None, progress: Progress = None) -> Dict[str, Any]:
    """Text -> WAV (24 kHz mono from the engine; seed-vc's rate when converted).
    Returns {wav, seconds, duration_s, engine, voice, converted, convert_to, batches, retries,
    transliterated_text, tts_wav}; a convert_after voice (F8) adds {length_adjust, convert_ref,
    checkpoint}."""
    req = resolve_request(text, language, voice, engine, style, convert_to)
    t0 = time.time()

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), message)

    # convert_after (F8): Chatterbox speaks it at pace 1.0 -- the requested pace is realised by
    # seed-vc's own length_adjust afterwards instead, never a time-stretch on top of a time-stretch.
    convert_after = _voice_convert_sidecar(req["voice"]) if req["engine"] == "chatterbox" else None
    tts_style = dict(req["style"])
    tts_pace = req["pace"]
    if convert_after:
        tts_style["pace"] = tts_pace = 1.0

    folder = Path(out_dir) if out_dir else DATA_DIR / "narration"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"narration_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    text_file = folder / f"{stem}.txt"
    tts_wav = folder / f"{stem}.wav"
    text_file.write_text(req["text"], encoding="utf-8")
    share = 0.75 if (req["convert_to"] or convert_after) else 1.0

    refuse_while_rendering()
    if seedvc_env.WORKER.running():
        report(0.0, "stopping the voice-conversion worker (one model at a time)")
        seedvc_env.WORKER.stop(reason="narration needs the GPU")
    report(0.0, "freeing the GPU")
    handover = seedvc_env._run_async(gpu_handover.prepare_for_phase(
        GPU_PHASE, need_vram_mb=NEED_VRAM_MB, need_ram_mb=NEED_RAM_MB))
    if isinstance(handover, dict) and not handover.get("ready", True):
        logger.warning("Narration starting without a full GPU hand-over: %s", handover.get("reason"))
    try:
        answer = _run_tts(
            ["synthesize", "--text-file", str(text_file), "--engine", req["engine"],
             "--voice", req["voice"], "--language", req["language"],
             "--exaggeration", f"{req['exaggeration']:g}", "--pace", f"{tts_pace:g}",
             "--style", json.dumps(tts_style, ensure_ascii=False),
             "--out", str(tts_wav), "--voices-dir", str(voices_dir())],
            progress=lambda f, m: report(f * share, m), timeout_s=_timeout_for(req["text"]))
    finally:
        try:
            seedvc_env._run_async(gpu_handover.offload_all("narration"))
        except Exception as e:
            logger.warning("GPU offload after narration failed: %s", e)

    result: Dict[str, Any] = {
        "wav": answer.get("wav") or str(tts_wav), "tts_wav": answer.get("wav") or str(tts_wav),
        "duration_s": answer.get("duration_s"), "engine": req["engine"], "voice": req["voice"],
        "language": req["language"], "converted": False, "convert_to": req["convert_to"],
        "batches": answer.get("batches"), "retries": answer.get("retries", 0),
        "transliterated_text": answer.get("transliterated_text"),
        "style": answer.get("style", req["style"]), "word_gaps": answer.get("word_gaps", 0),
        "phrase_gaps": answer.get("phrase_gaps", 0), "sentence_gaps": answer.get("sentence_gaps", 0),
        "verify": answer.get("verify") or {},
    }
    if req["convert_to"]:
        ref = cli_module().reference_path(req["convert_to"], voices_dir())
        converted = folder / f"{stem}_as_{req['convert_to']}.wav"
        report(share, f"converting to {req['convert_to']}")
        seedvc_env.run_cli(["convert", "--source", str(result["tts_wav"]), "--ref", str(ref),
                            "--out", str(converted)],
                           progress=lambda f, m: report(share + (1 - share) * f, f"voice conversion: {m}"),
                           timeout_s=_timeout_for(req["text"]))
        result["wav"] = str(converted)
        result["converted"] = True
        try:
            import soundfile as sf
            info = sf.info(str(converted))
            result["duration_s"] = round(info.frames / float(info.samplerate), 3)
        except Exception as e:
            logger.debug("converted narration unreadable for its length: %s", e)
    elif convert_after:
        # F8: the voice's own sidecar always goes through the fine-tuned seed-vc model into the
        # creator's real timbre -- length_adjust stands in for the pace Chatterbox did not apply,
        # clamped so a mood's own pace can never push it to an unusable extreme.
        lo, hi = CONVERT_AFTER_LENGTH_ADJUST_RANGE
        length_adjust = max(lo, min(hi, 1.0 / req["pace"])) if req["pace"] > 0 else 1.0
        ref = Path(str(convert_after.get("convert_ref") or ""))
        if not ref.is_file():
            raise NarrationError(f"convert_ref {ref} for voice {req['voice']!r} is missing")
        params: Dict[str, Any] = {"length_adjust": length_adjust}
        for key in ("checkpoint", "config", "diffusion_steps", "inference_cfg_rate"):
            if convert_after.get(key) is not None:
                params[key] = convert_after[key]
        converted = folder / f"{stem}_converted.wav"
        report(share, "converting to the creator's own voice")
        seedvc_env.run_cli(["convert", "--source", str(result["tts_wav"]), "--ref", str(ref),
                            "--out", str(converted), "--params", json.dumps(params)],
                           progress=lambda f, m: report(share + (1 - share) * f, f"voice conversion: {m}"),
                           timeout_s=_timeout_for(req["text"]))
        # seed-vc fills a true silence with a soft floor of its own target voice (~-70 dB, not
        # real quiet): the Chatterbox take's own silences are re-imposed, scaled by length_adjust.
        silence_gate.gate_files(str(result["tts_wav"]), str(converted), length_adjust, str(converted))
        result["wav"] = str(converted)
        result["converted"] = True
        result["length_adjust"] = length_adjust
        result["convert_ref"] = str(ref)
        result["checkpoint"] = convert_after.get("checkpoint")
        try:
            import soundfile as sf
            info = sf.info(str(converted))
            result["duration_s"] = round(info.frames / float(info.samplerate), 3)
        except Exception as e:
            logger.debug("gated narration unreadable for its length: %s", e)
    result["seconds"] = round(time.time() - t0, 1)
    report(1.0, "done")
    return result


# --------------------------------------------------------------------------- jobs

JOB_KIND = "narration"

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()      # one narration on the GPU at a time

# A restart's own startup sweep (STUDIO_DIRECTOR_PLAN.md enhancement 2, Fable
# review row 7): a narration job still "running" on disk belonged to the
# process before this one -- nothing here is synthesising it, so it is
# marked failed rather than left for a caller to poll forever.
job_store.sweep_stale(JOB_KIND)


def _persist(job: Dict[str, Any]) -> None:
    job["updated_at"] = time.time()
    job_store.save(JOB_KIND, job["job_id"], job)


def _update(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)
            _persist(job)


def running_jobs() -> int:
    """How many narrations are still running: the code reloader must wait for them
    (2026-10-23: a reload mid-preview lost the job -- "Unknown narration job")."""
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.get("state") == "running")


def busy() -> bool:
    """`GET /api/narration/busy`: whether a narration is running right now --
    the same shape as `routes.reels.busy`, for the Studio's idle-offload and
    code-reload holds (STUDIO_DIRECTOR_PLAN.md enhancement 2)."""
    return running_jobs() > 0


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
    # The in-memory registry is empty after a restart; the job's own file
    # still answers for it, already marked failed ("BuzzEdit restarted") by
    # the startup sweep if it was still "running" when this process started.
    return job_store.load(JOB_KIND, job_id)


def start_job(text: str, *, language: str = "hi", voice: str = "sudhansu", engine: Optional[str] = None,
              style: Optional[Dict[str, Any]] = None, convert_to: Optional[str] = None,
              out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Check the request (ValueError / NarrationBusy / NarrationUnavailable at once), then run
    `synthesize` on a thread. Returns the job: {job_id, state, message, progress, result, error}."""
    state = available()
    if not state["available"]:
        raise NarrationUnavailable(f"TTS engine unavailable: {state['reason']}")
    resolve_request(text, language, voice, engine, style, convert_to)
    refuse_while_rendering()
    job_id = uuid.uuid4().hex[:12]
    job = {"job_id": job_id, "state": "running", "message": "queued", "progress": 0.0,
           "result": None, "error": None, "started_at": time.time(), "ended_at": None,
           "out_dir": out_dir, "request": {"text": text, "language": language, "voice": voice,
                                           "engine": engine, "style": style, "convert_to": convert_to}}
    with _jobs_lock:
        _jobs[job_id] = job
    _persist(job)
    kwargs = dict(language=language, voice=voice, engine=engine, style=style,
                  convert_to=convert_to, out_dir=out_dir)
    threading.Thread(target=_run_job, args=(job_id, text, kwargs), name=f"narration_{job_id}",
                     daemon=True).start()
    return get_job(job_id) or job


def _run_job(job_id: str, text: str, kwargs: Dict[str, Any]) -> None:
    def progress(fraction: float, message: str) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress"] = round(max(job["progress"], fraction), 3)
                job["message"] = message
                _persist(job)

    # Master pause (BuzzcafStudio docs/STUDIO_DIRECTOR_PLAN.md Part 1): held
    # here, before this job's synthesis starts (the queued -> running
    # boundary is this job's own safe point; its internal TTS batches are
    # not individually checkpointed).
    from runtime import master_control
    master_control.wait_if_paused(f"narration job {job_id}", lambda m: progress(0.0, m))
    _update(job_id, message="waiting for the GPU")
    try:
        with _gpu_lock:
            result = synthesize(text, progress=progress, **kwargs)
    except Exception as exc:
        logger.warning("Narration job %s failed: %s", job_id, exc, exc_info=True)
        _update(job_id, state="failed", message="failed", error=str(exc) or exc.__class__.__name__,
                ended_at=time.time())
        return
    _update(job_id, state="done", message="done", progress=1.0, result=result, ended_at=time.time())
