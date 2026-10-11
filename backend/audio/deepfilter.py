"""DeepFilterNet 0.5.6 as a standalone binary (AUDIO_SUITE_PLAN.md §2.5, brief A2).

The `df` Python package cannot be installed next to this torch (see
`render/audio.py::deepfilter_available`), so the speech denoiser runs as the
project's own release binary instead: `deep-filter.exe` (CPU, WAV in → WAV
out), downloaded once into `data/models/deepfilter/`. The release's LADSPA
plugin is fetched beside it and probed through ffmpeg's `ladspa` filter; the
outcome is recorded in `status.json` (`ladspa_ok`) for the live engine.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf

from audio.analysis import AudioToolError, decode_wav, run_ffmpeg
from config import DATA_DIR, FFMPEG_BIN, TEMP_DIR
from utils.proc import NO_WINDOW

logger = logging.getLogger("audio.deepfilter")

RELEASE_URL = "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/"
EXE_ASSET = "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe"
LADSPA_ASSET = "deep_filter_ladspa-0.5.6-x86_64-pc-windows-msvc.dll"
DEEPFILTER_DIR = Path(DATA_DIR) / "models" / "deepfilter"
MODEL_RATE = 48000
RUN_TIMEOUT_S = 3600

_lock = threading.Lock()


class DeepFilterUnavailable(RuntimeError):
    """The binary is not on disk and could not be downloaded."""


def exe_path() -> Path:
    return DEEPFILTER_DIR / "deep-filter.exe"


def ladspa_path() -> Path:
    return DEEPFILTER_DIR / LADSPA_ASSET


def status_path() -> Path:
    return DEEPFILTER_DIR / "status.json"


def available() -> bool:
    """Whether the binary is on disk (no download is attempted)."""
    exe = exe_path()
    return exe.exists() and exe.stat().st_size > 1_000_000


def read_status() -> Dict[str, Any]:
    try:
        with open(status_path(), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """`url` to `dest` through a `.part` file, checked against Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(part, "wb") as fh:
        expected = int(response.headers.get("Content-Length") or 0)
        shutil.copyfileobj(response, fh, 1 << 20)
    size = part.stat().st_size
    if expected and size != expected:
        part.unlink(missing_ok=True)
        raise OSError(f"download of {url} stopped at {size} of {expected} bytes")
    os.replace(part, dest)
    return dest


def _probe_ladspa() -> Dict[str, Any]:
    """Fetch the LADSPA plugin and run it through ffmpeg on 1 s of silence.
    Never raises: the outcome is the status record.

    ffmpeg's `ladspa` loader takes a `file` that starts with '.' or '/' as a
    path and anything else as a name to look up as `<dir>/<name>.so`, so a
    drive-letter path is never loaded on Windows. The probe (and anything
    that runs the plugin) names it `./<dll>` with ffmpeg's working directory
    set to this folder. The plugin logs its real-time factor when it falls
    behind; the worst one seen is kept (`ladspa_rtf`), since above 1.0 it
    cannot stream live.
    """
    record: Dict[str, Any] = {"ladspa": None, "ladspa_ok": False, "ladspa_file_arg": None,
                              "ladspa_cwd": None, "ladspa_rtf": None, "ladspa_error": None}
    try:
        plugin = ladspa_path()
        if not plugin.exists():
            _download(RELEASE_URL + LADSPA_ASSET, plugin)
        record["ladspa"] = str(plugin)
        run_ffmpeg(["-h", "filter=ladspa"], timeout=60)
        file_arg = f"./{plugin.name}"
        probe = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-nostats", "-loglevel", "error", "-f", "lavfi",
             "-i", "anullsrc=r=48000:cl=mono", "-t", "1",
             "-af", f"ladspa=file={file_arg}:plugin=deep_filter_mono", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
            cwd=str(plugin.parent), creationflags=NO_WINDOW)
        stderr = (probe.stderr or b"").decode("utf-8", "replace")
        rtfs = [float(v) for v in re.findall(r"RTF:\s*([0-9.]+)", stderr)]
        record["ladspa_rtf"] = max(rtfs) if rtfs else None
        record["ladspa_ok"] = probe.returncode == 0
        if probe.returncode == 0:
            record["ladspa_file_arg"] = file_arg
            record["ladspa_cwd"] = str(plugin.parent)
        else:
            record["ladspa_error"] = stderr[-400:]
    except Exception as exc:
        record["ladspa_error"] = str(exc)[-400:]
    return record


def _write_status(record: Dict[str, Any]) -> None:
    record = {"exe": str(exe_path()), **record, "checked": datetime.now().isoformat(timespec="seconds")}
    tmp = status_path().with_name("status.tmp.json")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, status_path())


def ensure_binary() -> Path:
    """The binary's path, downloading it (and probing the LADSPA plugin) once."""
    with _lock:
        if available():
            if not status_path().exists():
                _write_status(_probe_ladspa())
            return exe_path()
        try:
            _download(RELEASE_URL + EXE_ASSET, exe_path())
        except Exception as exc:
            raise DeepFilterUnavailable(
                f"DeepFilterNet is not installed and could not be downloaded from "
                f"{RELEASE_URL + EXE_ASSET}: {exc}") from exc
        _write_status(_probe_ladspa())
        return exe_path()


def run(in_wav: str, out_wav: str, attenuation_limit_db: float = 100.0,
        postfilter_beta: float = 0.0,
        progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Noise Reduction (AI): DeepFilterNet over a WAV, written as float32 WAV.

    The input is resampled to the model's 48 kHz mono first and the result is
    brought back to the input's rate, same length. `-D` compensates the
    model's delay (it then trims that much off the end, so the staged input
    carries a short silent tail). `attenuation_limit_db` 100 = no limit;
    `postfilter_beta` > 0 enables the post-filter with that beta.
    """
    exe = ensure_binary()
    info = sf.info(str(in_wav))
    rate, frames = int(info.samplerate), int(info.frames)
    attenuation = max(0.0, min(100.0, float(attenuation_limit_db)))
    beta = max(0.0, float(postfilter_beta))
    if progress:
        progress(0.05, "DeepFilterNet: staging 48 kHz mono")
    with tempfile.TemporaryDirectory(dir=str(TEMP_DIR)) as tmp:
        tmp_dir = Path(tmp)
        staged = decode_wav(str(in_wav), tmp_dir / "in" / "audio.wav",
                            sample_rate=MODEL_RATE, mono=True)
        samples, _ = sf.read(str(staged), dtype="float32")
        length = len(samples)
        tail = np.zeros(MODEL_RATE // 10, dtype=np.float32)
        sf.write(str(staged), np.concatenate([samples, tail]), MODEL_RATE, subtype="FLOAT")
        del samples
        out_dir = tmp_dir / "out"
        command = [str(exe), "-D", "-a", f"{attenuation:g}", "-o", str(out_dir)]
        if beta > 0.0:
            command += ["--pf", "--pf-beta", f"{beta:g}"]
        command.append(str(staged))
        if progress:
            progress(0.1, "DeepFilterNet: enhancing")
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=RUN_TIMEOUT_S, cwd=str(tmp_dir), creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired as exc:
            raise AudioToolError(f"DeepFilterNet timed out after {RUN_TIMEOUT_S} s") from exc
        produced = out_dir / staged.name
        if result.returncode != 0 or not produced.exists():
            err = (result.stderr or b"").decode("utf-8", "replace")[-600:]
            raise AudioToolError(f"DeepFilterNet failed ({result.returncode}): {err}")
        enhanced, _ = sf.read(str(produced), dtype="float32")
        if enhanced.ndim > 1:
            enhanced = enhanced.mean(axis=1)
        enhanced = enhanced[:length]
        if enhanced.size < length:
            enhanced = np.pad(enhanced, (0, length - enhanced.size))
        if progress:
            progress(0.9, "DeepFilterNet: writing")
        out_path = Path(out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if rate == MODEL_RATE:
            sf.write(str(out_path), enhanced, MODEL_RATE, subtype="FLOAT")
        else:
            back = tmp_dir / "enhanced48.wav"
            sf.write(str(back), enhanced, MODEL_RATE, subtype="FLOAT")
            restored = decode_wav(str(back), tmp_dir / "restored.wav", sample_rate=rate, mono=True)
            data, _ = sf.read(str(restored), dtype="float32")
            data = data[:frames] if data.size >= frames else np.pad(data, (0, frames - data.size))
            sf.write(str(out_path), data, rate, subtype="FLOAT")
    if progress:
        progress(1.0, "DeepFilterNet: done")
    return out_path
