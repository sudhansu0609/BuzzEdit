"""BuzzEdit's command line to the seed-vc voice-conversion engine (VOICE_CHANGER_PLAN.md, brief S0).

Runs under seed-vc's own Python -- engines/seedvc/.venv, or SkinWalker's .venv until S4 moves it --
never BuzzEdit's: seed-vc is GPLv3 and pins numpy 1.26 / transformers 4.46, so BuzzEdit's backend
only ever starts this file as a separate process (backend/audio/seedvc_env.py, `run_cli`).

seed-vc is driven the way SkinWalker's app/engine.py drives it: the seed-vc folder goes on sys.path,
the models load once with the working directory inside that folder (seed-vc resolves
./checkpoints from there, so its checkpoints download into <seed-vc>/checkpoints), and
`SeedVCWrapper.convert_voice(source, target, stream_output=True, ...)` yields
`(mp3_bytes, (sr, audio))` per chunk, the full audio with the last one.

Commands
  warm        load the models once -- the first run downloads a few GB of checkpoints from
              HuggingFace -- and report checkpoint sizes, VRAM and load seconds
  convert     --source <wav|mp3> --ref <wav> --out <wav> [--params <json>]
              one whole file into the reference's voice. The WAV is written at the engine's rate
              (22050 Hz, 44100 Hz with f0_condition) and exactly as long as the source times
              length_adjust: a drift is padded with silence or trimmed at the end.
  apply-edit  --source <wav> --manifest <json> --out <wav>
              SkinWalker's editor manifest {duration, master_sr, regions[]}: each convert region in
              its own voice (a reference path, or a stem in --voices-dir) and engine params, cut
              regions removed, unmarked audio untouched, 15 ms crossfades (segments.py, SkinWalker's
              stitcher). A region that fails keeps the original audio and is flagged in the result.
              Without master_sr the output stays at the source's own rate.
  voices list the reference clips (<stem>.wav) in --voices-dir
  serve       load the models once, then run jobs read from stdin, one JSON object per line:
              {"op": "convert", "source", "ref", "out", "params"?, "wav_format"?, "id"?}
              {"op": "apply-edit", "source", "manifest", "out", "wav_format"?, "id"?}
              {"op": "ping"} (answered once the models are loaded), {"op": "quit"}.
              One JSON result line per job on stdout: {"id", "ok": true, "op", "out", "seconds",
              "device", ...the command's own fields} or {"id", "ok": false, "op", "error", "seconds"};
              `device` is where torch really holds the weights (cuda or cpu). A failed job leaves
              the process serving; a failed model load ends it like any command (exit 2/1, one
              {"ok": false, ...} line). End of stdin ends it as "quit" does.

Every command takes --seedvc-dir (default: seed-vc beside this file), --voices-dir,
--device auto|cuda|cpu (cuda = required) and --dry-run (check the arguments and the seed-vc
folder and speak the protocol, but load no model and write nothing). --params and --manifest
take inline JSON or the path of a JSON file.

Protocol
  stderr  JSON lines: {"event": "progress", "fraction": 0.42, "stage": "convert"} (fraction of the
          whole command; under serve also "job": <id>), then {"event": "done", ...} or
          {"event": "error", "message": ...}.
          Anything else on stderr -- library logs, download bars -- is plain text for the log tail.
  stdout  exactly one JSON object: the result, or {"ok": false, "error": ..., "exit_code": n}
          (serve: one per job, see above)
  exit    0 ok, 1 error, 2 engine unavailable (no seed-vc or its packages, checkpoints not
          downloaded while HuggingFace is unreachable, no CUDA with --device cuda)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PRESETS_FILE = HERE / "presets.json"
DEFAULT_SEEDVC_DIR = HERE / "seed-vc"

EXIT_OK, EXIT_ERROR, EXIT_UNAVAILABLE = 0, 1, 2

# The arguments SeedVCWrapper.convert_voice takes besides source/target/stream_output
# (SkinWalker app/engine.py _ENGINE_PARAM_KEYS).
ENGINE_PARAM_KEYS = ("diffusion_steps", "length_adjust", "inference_cfg_rate",
                     "f0_condition", "auto_f0_adjust", "pitch_shift")
# SkinWalker app/config.py ENGINE_DEFAULTS; presets.json is the source, this only its fallback.
FALLBACK_ENGINE_DEFAULTS: Dict[str, Any] = {
    "diffusion_steps": 25, "length_adjust": 1.0, "inference_cfg_rate": 0.7,
    "f0_condition": False, "auto_f0_adjust": True, "pitch_shift": 0,
}
XFADE_MS = 15.0
HEARTBEAT_S = 5.0
LOAD_END, CONVERT_END, WRITE_AT = 0.10, 0.95, 0.97   # overall fractions of a converting command

# Repos seed-vc's wrapper pulls into the user's HuggingFace cache (transformers' Whisper and
# BigVGAN's from_pretrained); everything else lands in <seed-vc>/checkpoints (hf_utils).
HF_CACHE_REPOS = ("openai/whisper-small", "nvidia/bigvgan_v2_22khz_80band_256x",
                  "nvidia/bigvgan_v2_44khz_128band_512x")

_OFFLINE_TYPES = {"LocalEntryNotFoundError", "OfflineModeIsEnabled", "ConnectionError",
                  "ConnectTimeout", "ReadTimeout", "MaxRetryError", "NewConnectionError",
                  "NameResolutionError", "ProxyError"}
_OFFLINE_TEXT = ("couldn't connect", "could not connect", "internet connection", "offline mode",
                 "failed to resolve", "getaddrinfo failed", "connection error")


# --- the protocol -----------------------------------------------------------------------------

class CliError(Exception):
    """A failure the caller sees as the error message (exit 1)."""
    exit_code = EXIT_ERROR


class EngineUnavailable(CliError):
    """seed-vc cannot run here at all (exit 2)."""
    exit_code = EXIT_UNAVAILABLE


_EMIT_LOCK = threading.Lock()
_RESULT_FD: Optional[int] = None
_CURRENT_JOB: Any = None            # the serve job whose progress is being reported


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):        # numpy scalars and arrays
        return value.tolist()
    return str(value)


def _protect_stdout() -> None:
    """Keep the real stdout for the one result object; everything else this process prints --
    seed-vc's and transformers' messages, native libraries -- goes to stderr."""
    global _RESULT_FD
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        _RESULT_FD = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr


def _write_result(obj: Dict[str, Any]) -> None:
    data = (json.dumps(obj, default=_json_default) + "\n").encode("utf-8")
    if _RESULT_FD is None:
        sys.__stdout__.buffer.write(data)
        sys.__stdout__.flush()
        return
    view = memoryview(data)
    while view:
        written = os.write(_RESULT_FD, view)
        view = view[written:]


def emit(event: str, **fields: Any) -> None:
    """One protocol line on stderr."""
    line = json.dumps({"event": event, **fields}, default=_json_default)
    with _EMIT_LOCK:
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def progress(fraction: float, stage: str, **extra: Any) -> None:
    if _CURRENT_JOB is not None and "job" not in extra:
        extra["job"] = _CURRENT_JOB
    emit("progress", fraction=round(min(1.0, max(0.0, float(fraction))), 4), stage=stage, **extra)


def _heartbeat(stop: threading.Event, fraction: float, stage: str, started: float) -> None:
    """A model load (or the first run's download) gives no fractions: say it is alive."""
    while not stop.wait(HEARTBEAT_S):
        progress(fraction, stage, elapsed_s=round(time.perf_counter() - started, 1))


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, and 2 means "engine unavailable" here: a usage error
    raises instead and leaves as exit 1 with an error event."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CliError(f"{self.prog}: {message}")


def build_parser() -> argparse.ArgumentParser:
    common = _Parser(add_help=False)
    common.add_argument("--seedvc-dir", default=str(DEFAULT_SEEDVC_DIR),
                        help="the seed-vc checkout (holds seed_vc_wrapper.py)")
    common.add_argument("--voices-dir", default="", help="reference clips, <stem>.wav")
    common.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto = CUDA when torch sees it; cuda = required (exit 2 without)")
    common.add_argument("--dry-run", action="store_true",
                        help="check the arguments and the seed-vc folder; load no model, write nothing")

    parser = _Parser(prog="cli.py", description="BuzzEdit's seed-vc voice-conversion engine")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    commands.add_parser("warm", parents=[common], help="load the models once (downloads them the first time)")

    convert = commands.add_parser("convert", parents=[common], help="convert one file into a reference voice")
    convert.add_argument("--source", required=True, help="the speech to convert (wav or mp3)")
    convert.add_argument("--ref", required=True, help="the target voice's reference clip (wav)")
    convert.add_argument("--out", required=True, help="the WAV to write")
    convert.add_argument("--params", default="", help="JSON object or file: " + ", ".join(ENGINE_PARAM_KEYS))
    convert.add_argument("--wav-format", choices=("float32", "int16"), default="float32")

    edit = commands.add_parser("apply-edit", parents=[common], help="convert and cut the regions of an edit manifest")
    edit.add_argument("--source", required=True, help="the recording (wav)")
    edit.add_argument("--manifest", required=True, help="JSON object or file: {duration, master_sr, regions[]}")
    edit.add_argument("--out", required=True, help="the WAV to write")
    edit.add_argument("--wav-format", choices=("float32", "int16"), default="float32")

    voices = commands.add_parser("voices", help="the voices folder")
    voice_commands = voices.add_subparsers(dest="voices_command", required=True, parser_class=_Parser)
    voice_commands.add_parser("list", parents=[common], help="list the reference clips")

    commands.add_parser("serve", parents=[common],
                        help="load the models once, then run JSON jobs from stdin, one result line each")
    return parser


# --- arguments --------------------------------------------------------------------------------

def _json_arg(value: str, what: str) -> Any:
    """Inline JSON, or the path of a JSON file; empty = {}."""
    text = (value or "").strip()
    if not text:
        return {}
    if not text.startswith(("{", "[")):
        path = Path(text)
        if not path.is_file():
            raise CliError(f"{what}: {text!r} is neither JSON nor a JSON file")
        text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise CliError(f"{what} is not valid JSON: {exc}") from exc


def engine_defaults() -> Dict[str, Any]:
    """presets.json's engine_defaults (SkinWalker's ENGINE_DEFAULTS)."""
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        defaults = data.get("engine_defaults")
        if isinstance(defaults, dict) and defaults:
            return {**FALLBACK_ENGINE_DEFAULTS, **defaults}
    except Exception:
        pass
    return dict(FALLBACK_ENGINE_DEFAULTS)


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
        return value.strip().lower() in ("true", "1", "yes")
    raise CliError(f"{key} must be true or false, not {value!r}")


def merge_params(overrides: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The engine defaults with `overrides` on top (unknown keys ignored), checked and typed."""
    if overrides in (None, ""):
        overrides = {}
    if not isinstance(overrides, dict):
        raise CliError("engine params must be a JSON object")
    merged = dict(base if base is not None else engine_defaults())
    merged.update({key: value for key, value in overrides.items() if key in ENGINE_PARAM_KEYS})
    try:
        steps = int(merged["diffusion_steps"])
        length = float(merged["length_adjust"])
        cfg = float(merged["inference_cfg_rate"])
        shift = float(merged["pitch_shift"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CliError(f"engine params: {exc}") from exc
    if not 1 <= steps <= 200:
        raise CliError(f"diffusion_steps must be 1..200, not {steps}")
    if not 0.25 <= length <= 4.0:
        raise CliError(f"length_adjust must be 0.25..4, not {length}")
    if not 0.0 <= cfg <= 1.0:
        raise CliError(f"inference_cfg_rate must be 0..1, not {cfg}")
    if not -24.0 <= shift <= 24.0:
        raise CliError(f"pitch_shift must be -24..24 semitones, not {shift}")
    return {
        "diffusion_steps": steps,
        "length_adjust": length,
        "inference_cfg_rate": cfg,
        "f0_condition": _as_bool(merged.get("f0_condition", False), "f0_condition"),
        "auto_f0_adjust": _as_bool(merged.get("auto_f0_adjust", True), "auto_f0_adjust"),
        "pitch_shift": int(shift) if shift.is_integer() else shift,
    }


def _existing(path: str, flag: str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise CliError(f"{flag}: {path} does not exist")
    return str(candidate.resolve())


def _seedvc_dir(args: argparse.Namespace) -> Path:
    folder = Path(args.seedvc_dir or DEFAULT_SEEDVC_DIR).resolve()
    if not (folder / "seed_vc_wrapper.py").is_file():
        raise EngineUnavailable(f"seed-vc not found: {folder} has no seed_vc_wrapper.py "
                                "(run scripts/setup_seedvc.ps1, or pass --seedvc-dir)")
    return folder


def resolve_voice(voice: str, voices_dir: str) -> str:
    """A region's voice: the stem of <voices_dir>/<stem>.wav, or a reference clip's path."""
    name = str(voice).strip()
    if voices_dir and name and Path(name).name == name:
        for candidate in (Path(voices_dir) / f"{name}.wav", Path(voices_dir) / name):
            if candidate.is_file():
                return str(candidate.resolve())
    if name and Path(name).is_file():
        return str(Path(name).resolve())
    raise CliError(f"voice {name!r} is neither a file nor <stem>.wav in "
                   f"{voices_dir or 'a voices folder (no --voices-dir)'}")


# --- audio ------------------------------------------------------------------------------------

def _duration_s(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except Exception as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc


def _audio_info(path: Path) -> Dict[str, Any]:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {"duration_s": round(info.frames / float(info.samplerate), 3),
                "sr": int(info.samplerate), "channels": int(info.channels)}
    except Exception:
        pass
    try:
        import wave
        with wave.open(str(path), "rb") as handle:
            return {"duration_s": round(handle.getnframes() / float(handle.getframerate()), 3),
                    "sr": handle.getframerate(), "channels": handle.getnchannels()}
    except Exception:
        return {}


def _read_mono(path: str) -> Tuple[Any, int]:
    """float32 mono at the file's own rate, exactly as decoded (one channel is not touched)."""
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
    except Exception as first:
        try:
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=True)
            return np.asarray(audio, dtype=np.float32), int(sr)
        except Exception:
            raise CliError(f"cannot read {path}: {first}") from first


def fit_length(audio: Any, target: int) -> Tuple[Any, int]:
    """Pad with silence or trim at the end to exactly `target` samples. Returns the audio and the
    drift in samples before the fix (+ = seed-vc's output was longer)."""
    import numpy as np
    drift = int(audio.shape[0]) - int(target)
    if drift > 0:
        audio = audio[:target]
    elif drift < 0:
        audio = np.concatenate([audio, np.zeros(-drift, dtype=audio.dtype)])
    return audio, drift


def write_wav(path: str, audio: Any, sr: int, wav_format: str) -> str:
    import numpy as np
    import soundfile as sf
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32)
    part = out.with_name(out.name + ".part")
    if wav_format == "int16":
        sf.write(str(part), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    else:
        sf.write(str(part), data, sr, subtype="FLOAT", format="WAV")
    os.replace(part, out)
    return str(out)


# --- the engine -------------------------------------------------------------------------------

def _offline_error(exc: Optional[BaseException]) -> bool:
    """Whether a load failure is "no checkpoints and no HuggingFace", anywhere in its chain."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if {cls.__name__ for cls in type(exc).__mro__} & _OFFLINE_TYPES:
            return True
        text = str(exc).lower()
        if any(marker in text for marker in _OFFLINE_TEXT):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def expected_chunks(source_s: float, ref_s: float, f0_condition: bool) -> int:
    """How many pieces convert_voice yields: a 30 s context window holds the reference (at most
    25 s) and a piece of the source; pieces overlap by 16 frames."""
    sr, hop = (44100, 512) if f0_condition else (22050, 256)
    fps = sr / hop
    window = max(17, (sr // hop) * 30 - int(min(ref_s, 25.0) * fps))
    frames = max(1, int(source_s * fps))
    return 1 + max(0, math.ceil((frames - window) / (window - 16)))


class Engine:
    """seed-vc loaded once, as SkinWalker's app/engine.py VoiceEngine loads it."""

    def __init__(self, seedvc_dir: Path, device: str = "auto") -> None:
        self.seedvc_dir = seedvc_dir
        self.device_pref = device
        self.device = "cpu"
        self.wrapper: Any = None
        self.torch: Any = None
        self.load_s = 0.0

    def load(self, fraction: float = 0.0) -> None:
        if self.wrapper is not None:
            return
        started = time.perf_counter()
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise EngineUnavailable(f"torch cannot be imported with {sys.executable}: {exc}") from exc
        self.torch = torch
        cuda = bool(torch.cuda.is_available())
        if self.device_pref == "cuda" and not cuda:
            raise EngineUnavailable("CUDA is required (--device cuda) but torch sees no CUDA device")
        self.device = "cuda" if cuda and self.device_pref != "cpu" else "cpu"
        vendor = str(self.seedvc_dir)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        progress(fraction, "load")
        stop = threading.Event()
        threading.Thread(target=_heartbeat, args=(stop, fraction, "load", started), daemon=True).start()
        cwd = os.getcwd()
        try:
            os.chdir(vendor)     # seed-vc resolves ./checkpoints from its own folder
            try:
                from seed_vc_wrapper import SeedVCWrapper
            except ImportError as exc:
                raise EngineUnavailable(f"seed-vc cannot be imported with {sys.executable}: {exc}") from exc
            try:
                self.wrapper = SeedVCWrapper(device=torch.device(self.device))
            except Exception as exc:
                if _offline_error(exc):
                    raise EngineUnavailable("seed-vc's checkpoints are not downloaded and HuggingFace "
                                            f"cannot be reached: {type(exc).__name__}: {exc}") from exc
                raise
        finally:
            os.chdir(cwd)
            stop.set()
        self.load_s = time.perf_counter() - started

    def convert(self, source: str, ref: str, params: Dict[str, Any], source_s: float, ref_s: float,
                on_fraction: Optional[Callable[[float], None]] = None) -> Tuple[Any, int]:
        """`source` in the voice of `ref`: float32 mono at the engine's rate."""
        import numpy as np
        self.load()
        pieces = expected_chunks(source_s, ref_s, bool(params.get("f0_condition")))
        generator = self.wrapper.convert_voice(
            os.path.abspath(source), os.path.abspath(ref), stream_output=True,
            **{key: params[key] for key in ENGINE_PARAM_KEYS if key in params})
        last = None
        done = 0
        for _mp3_bytes, full_audio in generator:
            done += 1
            if full_audio is not None:
                last = full_audio
            if on_fraction is not None:
                on_fraction(min(1.0, done / float(pieces)))
        if last is None:
            raise RuntimeError("seed-vc produced no audio")
        sr, audio = last
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)

    def actual_device(self) -> Optional[str]:
        """Where torch really holds the weights ("cuda" or "cpu"): the Whisper encoder's
        parameters, else a DiT module's; the chosen device when neither can be read."""
        if self.wrapper is None:
            return None
        modules: List[Any] = [getattr(self.wrapper, "whisper_model", None)]
        model = getattr(self.wrapper, "model", None)
        if model is not None and hasattr(model, "values"):
            modules.extend(list(model.values()))
        for module in modules:
            try:
                return str(next(module.parameters()).device.type)
            except Exception:
                continue
        return self.device


_SHARED_ENGINE: Optional[Engine] = None     # serve's engine, loaded once for every job


def _engine(seedvc_dir: Path, device: str) -> Engine:
    """The loaded engine: serve's shared one, else a fresh load for this one command."""
    if _SHARED_ENGINE is not None:
        return _SHARED_ENGINE
    engine = Engine(seedvc_dir, device)
    engine.load(0.0)
    return engine


def _vram(torch: Any) -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {}
    mb = 1024.0 * 1024.0
    out: Dict[str, Any] = {"allocated_mb": round(torch.cuda.memory_allocated() / mb),
                           "reserved_mb": round(torch.cuda.memory_reserved() / mb),
                           "peak_mb": round(torch.cuda.max_memory_allocated() / mb)}
    try:
        free, total = torch.cuda.mem_get_info()
        out.update(device_free_mb=round(free / mb), device_total_mb=round(total / mb))
    except Exception:
        pass
    return out


def _hf_cache_dir() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def _tree_bytes(root: Path) -> int:
    total = 0
    for folder, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            try:
                if not os.path.islink(path):
                    total += os.lstat(path).st_size
            except OSError:
                pass
    return total


def checkpoints(seedvc_dir: Path) -> List[Dict[str, Any]]:
    """The checkpoint repos on disk and their sizes."""
    found: List[Dict[str, Any]] = []
    local = seedvc_dir / "checkpoints"
    if local.is_dir():
        for repo_dir in sorted(local.glob("models--*")):
            size = _tree_bytes(repo_dir)
            found.append({"repo": repo_dir.name[len("models--"):].replace("--", "/"),
                          "where": "seed-vc/checkpoints", "bytes": size, "mb": round(size / 1048576.0, 1)})
    hub = _hf_cache_dir()
    for repo in HF_CACHE_REPOS:
        repo_dir = hub / ("models--" + repo.replace("/", "--"))
        if repo_dir.is_dir():
            size = _tree_bytes(repo_dir)
            found.append({"repo": repo, "where": "huggingface cache", "bytes": size,
                          "mb": round(size / 1048576.0, 1)})
    return found


# --- commands ---------------------------------------------------------------------------------

def cmd_warm(args: argparse.Namespace) -> Dict[str, Any]:
    seedvc = _seedvc_dir(args)
    before = checkpoints(seedvc)
    before_bytes = sum(item["bytes"] for item in before)
    if args.dry_run:
        progress(1.0, "dry-run")
        return {"seedvc_dir": str(seedvc), "loaded": False, "checkpoints": before,
                "checkpoint_bytes": before_bytes}
    engine = Engine(seedvc, args.device)
    engine.load(0.0)
    progress(1.0, "load")
    after = checkpoints(seedvc)
    after_bytes = sum(item["bytes"] for item in after)
    torch = engine.torch
    return {
        "seedvc_dir": str(seedvc), "loaded": True, "device": engine.device,
        "gpu": torch.cuda.get_device_name(0) if engine.device == "cuda" else None,
        "torch": str(torch.__version__), "python": sys.version.split()[0],
        "load_s": round(engine.load_s, 2), "vram": _vram(torch),
        "checkpoints": after, "checkpoint_bytes": after_bytes,
        "checkpoint_mb": round(after_bytes / 1048576.0, 1),
        "downloaded_bytes": max(0, after_bytes - before_bytes),
    }


def cmd_convert(args: argparse.Namespace) -> Dict[str, Any]:
    source = _existing(args.source, "--source")
    ref = _existing(args.ref, "--ref")
    params = merge_params(_json_arg(args.params, "--params"))
    seedvc = _seedvc_dir(args)
    source_s, ref_s = _duration_s(source), _duration_s(ref)
    plan = {"source": source, "ref": ref, "out": str(Path(args.out).resolve()), "params": params,
            "source_s": round(source_s, 3), "ref_s": round(ref_s, 3), "seedvc_dir": str(seedvc)}
    if args.dry_run:
        progress(1.0, "dry-run")
        return plan
    engine = _engine(seedvc, args.device)
    progress(LOAD_END, "load")
    started = time.perf_counter()
    audio, sr = engine.convert(
        source, ref, params, source_s, ref_s,
        lambda f: progress(LOAD_END + (CONVERT_END - LOAD_END) * f, "convert"))
    convert_s = time.perf_counter() - started
    audio, drift = fit_length(audio, int(round(source_s * params["length_adjust"] * sr)))
    progress(WRITE_AT, "write")
    out = write_wav(args.out, audio, sr, args.wav_format)
    return {**plan, "out": out, "sr": sr, "wav_format": args.wav_format,
            "output_s": round(audio.shape[0] / float(sr), 3), "drift_ms": round(drift * 1000.0 / sr, 1),
            "load_s": round(engine.load_s, 2), "convert_s": round(convert_s, 2),
            "seconds_per_minute": round(convert_s * 60.0 / source_s, 1) if source_s > 0 else None,
            "device": engine.device, "vram": _vram(engine.torch)}


def _span_view(span: Dict[str, Any]) -> Dict[str, Any]:
    region = span.get("region") or {}
    return {"kind": span["kind"], "start": round(span["start"], 4), "end": round(span["end"], 4),
            "id": region.get("id")}


def cmd_apply_edit(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import segments

    source = _existing(args.source, "--source")
    raw = _json_arg(args.manifest, "--manifest")
    if not isinstance(raw, dict):
        raise CliError("--manifest must be a JSON object {duration, master_sr, regions[]}")
    audio, native_sr = _read_mono(source)
    manifest = dict(raw)
    if not manifest.get("duration"):
        manifest["duration"] = audio.shape[0] / float(native_sr)
    if not manifest.get("master_sr"):
        manifest["master_sr"] = native_sr          # unmarked audio stays as it is, sample for sample
    parsed = segments.parse_manifest(manifest)
    master_sr = int(parsed["master_sr"])
    spans = segments.build_spans(parsed["duration"], parsed["regions"])
    convert_spans = [span for span in spans if span["kind"] == "convert"]
    plan = {"source": source, "out": str(Path(args.out).resolve()), "sr": master_sr,
            "source_s": round(audio.shape[0] / float(native_sr), 3), "xfade_ms": XFADE_MS,
            "spans": [_span_view(span) for span in spans]}

    if args.dry_run:
        regions = []
        for span in convert_spans:
            region = span["region"]
            entry = {"id": region.get("id"), "start": region["start"], "end": region["end"],
                     "voice": region.get("voice")}
            try:
                if not region.get("voice"):
                    raise CliError("A Convert region has no target voice set.")
                entry["ref"] = resolve_voice(str(region["voice"]), args.voices_dir)
                entry["params"] = merge_params(region.get("engine") or {})
            except CliError as exc:
                entry["error"] = str(exc)
            regions.append(entry)
        progress(1.0, "dry-run")
        return {**plan, "regions": regions}

    engine: Optional[Engine] = None
    if convert_spans:
        engine = _engine(_seedvc_dir(args), args.device)
    progress(LOAD_END, "load")
    defaults = engine_defaults()
    total_s = sum(span["end"] - span["start"] for span in convert_spans) or 1.0
    state = {"done_s": 0.0}
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="seedvc_edit_") as tmp:
        def convert_fn(region: Dict[str, Any], seg: Any, seg_sr: int) -> Tuple[Any, int]:
            import soundfile as sf
            seg_s = seg.shape[0] / float(seg_sr)
            record = {"id": region.get("id"), "start": region["start"], "end": region["end"],
                      "voice": region.get("voice"), "ok": False}
            records.append(record)
            region_started = time.perf_counter()

            def on_fraction(f: float) -> None:
                done = (state["done_s"] + f * seg_s) / total_s
                progress(LOAD_END + (CONVERT_END - LOAD_END) * min(1.0, done), "convert", region=record["id"])

            try:
                if not region.get("voice"):
                    raise CliError("A Convert region has no target voice set.")
                ref = resolve_voice(str(region["voice"]), args.voices_dir)
                params = merge_params(region.get("engine") or {}, defaults)
                params["length_adjust"] = 1.0       # as SkinWalker: a region keeps its length
                seg_path = os.path.join(tmp, f"region_{len(records):03d}.wav")
                sf.write(seg_path, np.asarray(seg, dtype=np.float32), seg_sr, subtype="FLOAT", format="WAV")
                out, out_sr = engine.convert(seg_path, ref, params, seg_s, _duration_s(ref), on_fraction)
                out, drift = fit_length(out, int(round(seg_s * out_sr)))
                record.update(ok=True, ref=ref, params=params, drift_ms=round(drift * 1000.0 / out_sr, 1))
                return out, out_sr
            except Exception as exc:
                record["error"] = str(exc) if isinstance(exc, CliError) else f"{type(exc).__name__}: {exc}"
                print(f"region {record['id']} keeps the original audio: {record['error']}", file=sys.stderr)
                raise
            finally:
                record["seconds"] = round(time.perf_counter() - region_started, 2)
                state["done_s"] += seg_s

        result = segments.apply_edit(audio, native_sr, manifest, convert_fn,
                                     master_sr=master_sr, xfade_ms=XFADE_MS)
    convert_s = time.perf_counter() - started
    progress(WRITE_AT, "write")
    out = write_wav(args.out, result, master_sr, args.wav_format)
    return {**plan, "out": out, "wav_format": args.wav_format,
            "output_s": round(result.shape[0] / float(master_sr), 3),
            "regions": records, "converted": sum(1 for r in records if r["ok"]),
            "failed": sum(1 for r in records if not r["ok"]),
            "cut_s": round(sum(s["end"] - s["start"] for s in spans if s["kind"] == "cut"), 3),
            "load_s": round(engine.load_s, 2) if engine else 0.0, "convert_s": round(convert_s, 2),
            "device": engine.device if engine else None, "vram": _vram(engine.torch) if engine else {}}


def cmd_voices(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.voices_dir:
        raise CliError("voices list needs --voices-dir")
    folder = Path(args.voices_dir).resolve()
    voices = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.wav"), key=lambda p: p.stem.lower()):
            entry: Dict[str, Any] = {"name": path.stem, "path": str(path), "bytes": path.stat().st_size}
            entry.update(_audio_info(path))
            voices.append(entry)
    progress(1.0, "list")
    return {"voices_dir": str(folder), "voices": voices}


SERVE_OPS = ("convert", "apply-edit", "ping", "quit")


def _job_args(args: argparse.Namespace, job: Dict[str, Any], **fields: Any) -> argparse.Namespace:
    """A serve job as the arguments of the one-shot command it runs."""
    wav_format = str(job.get("wav_format") or "float32")
    if wav_format not in ("float32", "int16"):
        raise CliError(f"wav_format must be float32 or int16, not {wav_format!r}")
    for key, value in fields.items():
        if value in (None, ""):
            raise CliError(f"{job.get('op')} needs {key!r}")
    return argparse.Namespace(seedvc_dir=args.seedvc_dir, voices_dir=args.voices_dir, device=args.device,
                              dry_run=args.dry_run, wav_format=wav_format, **fields)


def _as_json_arg(value: Any) -> Any:
    """params / manifest: an object travels as JSON text, a string (a file path or JSON) as it is."""
    if value is None or isinstance(value, str):
        return value or ""
    return json.dumps(value)


def _run_job(args: argparse.Namespace, job: Dict[str, Any], engine: Optional[Engine]) -> Dict[str, Any]:
    op = job.get("op")
    if op == "ping":
        return {"out": None, "loaded": engine is not None and engine.wrapper is not None,
                "load_s": round(engine.load_s, 2) if engine else 0.0,
                "vram": _vram(engine.torch) if engine else {}}
    if op == "convert":
        return cmd_convert(_job_args(args, job, source=job.get("source"), ref=job.get("ref"),
                                     out=job.get("out"), params=_as_json_arg(job.get("params"))))
    if op == "apply-edit":
        return cmd_apply_edit(_job_args(args, job, source=job.get("source"), out=job.get("out"),
                                        manifest=_as_json_arg(job.get("manifest"))))
    raise CliError(f"unknown op {op!r}: one of {', '.join(SERVE_OPS)}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Load once, then one result line per job line until "quit" or the end of stdin."""
    global _SHARED_ENGINE, _CURRENT_JOB
    seedvc = _seedvc_dir(args)
    engine: Optional[Engine] = None
    if not args.dry_run:
        engine = Engine(seedvc, args.device)
        engine.load(0.0)
        _SHARED_ENGINE = engine
    device = engine.actual_device() if engine else None
    emit("ready", device=device, load_s=round(engine.load_s, 2) if engine else 0.0,
         dry_run=bool(args.dry_run))
    count = 0
    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            break                                   # the parent went away: as "quit"
        text = raw.decode("utf-8-sig", errors="replace").strip()
        if not text:
            continue
        count += 1
        started = time.perf_counter()
        job_id: Any = count
        op: Any = None
        try:
            try:
                job = json.loads(text)
            except ValueError as exc:
                raise CliError(f"a job line is not valid JSON: {exc}") from exc
            if not isinstance(job, dict):
                raise CliError("a job must be a JSON object")
            job_id = job.get("id", count)
            op = job.get("op")
            _CURRENT_JOB = job_id
            if op == "quit":
                _write_result({"id": job_id, "ok": True, "op": "quit", "out": None,
                               "seconds": round(time.perf_counter() - started, 3), "device": device,
                               "jobs": count - 1})
                break
            result = _run_job(args, job, engine)
            device = engine.actual_device() if engine else None
            _write_result({**result, "id": job_id, "ok": True, "op": op, "out": result.get("out"),
                           "seconds": round(time.perf_counter() - started, 3), "device": device})
        except Exception as exc:  # noqa: BLE001 -- a failed job answers and the process serves on
            if not isinstance(exc, CliError):
                traceback.print_exc(file=sys.stderr)
            message = str(exc) if isinstance(exc, CliError) else f"{type(exc).__name__}: {exc}"
            emit("error", command="serve", job=job_id, message=message)
            _write_result({"id": job_id, "ok": False, "op": op, "error": message,
                           "seconds": round(time.perf_counter() - started, 3)})
        finally:
            _CURRENT_JOB = None
    return None


COMMANDS: Dict[str, Callable[[argparse.Namespace], Optional[Dict[str, Any]]]] = {
    "warm": cmd_warm, "convert": cmd_convert, "apply-edit": cmd_apply_edit, "voices": cmd_voices,
    "serve": cmd_serve,
}


def main(argv: Optional[List[str]] = None) -> int:
    _protect_stdout()
    started = time.perf_counter()
    command = ""
    try:
        args = build_parser().parse_args(argv)
        command = args.command + (f" {args.voices_command}" if args.command == "voices" else "")
        result = COMMANDS[args.command](args)
        seconds = round(time.perf_counter() - started, 2)
        emit("done", command=command, ok=True, seconds=seconds)
        if result is None:                      # serve answered every job on its own line
            return EXIT_OK
        _write_result({"ok": True, "command": command, "dry_run": bool(getattr(args, "dry_run", False)),
                       **result, "seconds": seconds})
        return EXIT_OK
    except CliError as exc:
        if exc.__cause__ is not None:
            traceback.print_exception(type(exc.__cause__), exc.__cause__, exc.__cause__.__traceback__,
                                      file=sys.stderr)
        code, message = exc.exit_code, str(exc)
    except Exception as exc:  # noqa: BLE001 -- every failure leaves as JSON
        traceback.print_exc(file=sys.stderr)
        code, message = EXIT_ERROR, f"{type(exc).__name__}: {exc}"
    emit("error", command=command, message=message, exit_code=code)
    _write_result({"ok": False, "command": command, "error": message, "exit_code": code})
    return code


if __name__ == "__main__":
    sys.exit(main())
