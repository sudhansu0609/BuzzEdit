"""Enhance Speech (AI): availability check and runner (AUDIO_SUITE_PLAN.md
§2.5, §3 "Vocal Enhancer", brief A3).

`check_install()` tries, within a time budget (15 minutes), to make one
speech-restoration engine importable in this venv:

  1. `uv pip install voicefixer`, then `import voicefixer`;
  2. else `uv pip install --no-deps resemble-enhance` plus its light
     dependencies, then `import resemble_enhance.enhancer.inference`.

Every install is dry-run first and refused when it would change torch,
torchaudio or numpy; if the versions change anyway, the packages it added are
uninstalled. The outcome goes to `data/models/enhance/status.json`
({engine: "voicefixer" | "resemble_enhance" | "none", error, ...});
`available()` reads it, and the process catalogue hides
`enhance_speech_ai` while it says "none".

`enhance(in_wav, out_wav, progress)` runs the recorded engine (CUDA when
torch sees a GPU), then brings the result back to the input's sample rate,
channel count and length. The model is loaded for the call and released
after it (GPU policy: nothing stays resident); ejecting other models first
is the job runner's part.
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import DATA_DIR, FFMPEG_BIN

ENHANCE_DIR = Path(DATA_DIR) / "models" / "enhance"
STATUS_PATH = ENHANCE_DIR / "status.json"

ENGINES = ("voicefixer", "resemble_enhance")
CHECK_BUDGET_S = 15 * 60
PROTECTED = ("torch", "torchaudio", "numpy")
RESEMBLE_LIGHT_DEPS = ["omegaconf", "einops", "rich", "tabulate", "resampy", "celluloid",
                       "ptflops", "tqdm"]
VOICEFIXER_RATE = 44100
# voicefixer downloads its weights (vocoder 0.14 GB, analysis model 0.49 GB, from
# Zenodo) the first time it is imported, into this folder
VOICEFIXER_CACHE = Path.home() / ".cache" / "voicefixer"

Progress = Optional[Callable[[float, str], None]]


# --- status --------------------------------------------------------------------

def status() -> Dict[str, Any]:
    """The recorded outcome of the last check ({"engine": "none"} if never run)."""
    try:
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    engine = data.get("engine") if data.get("engine") in ENGINES else "none"
    return {**data, "engine": engine, "error": data.get("error")}


def available() -> bool:
    """True when status.json names an engine that imported cleanly."""
    return status()["engine"] != "none"


def _write_status(data: Dict[str, Any]) -> None:
    ENHANCE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- install check ---------------------------------------------------------------

def _run(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    """subprocess.run that reports a timeout as a failed run (returncode None)
    instead of raising; the child has been killed by then."""
    timeout = max(5.0, timeout)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, None, "", f"timed out after {timeout:.0f} s")


def _drop_partial_download(cache: Path, since: float) -> str:
    """After an import was killed mid-download: remove the newest file written
    since `since` (the one being downloaded), so the next import fetches it
    again instead of loading a truncated checkpoint."""
    try:
        files = [p for p in cache.rglob("*") if p.is_file() and p.stat().st_mtime >= since]
    except OSError:
        return ""
    if not files:
        return ""
    newest = max(files, key=lambda p: p.stat().st_mtime)
    size = newest.stat().st_size
    try:
        newest.unlink()
    except OSError as exc:
        return f"; could not remove the partial {newest.name} ({exc})"
    return f"; removed the partial download {newest.name} ({size} bytes)"


def _versions(timeout: float = 60.0) -> Dict[str, Optional[str]]:
    code = ("import json\nfrom importlib import metadata\nout = {}\n"
            f"for n in {list(PROTECTED)!r}:\n"
            "    try:\n        out[n] = metadata.version(n)\n"
            "    except metadata.PackageNotFoundError:\n        out[n] = None\n"
            "print(json.dumps(out))")
    r = _run([sys.executable, "-c", code], timeout)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {}


_CHANGE = re.compile(r"^\s*([-+])\s+([A-Za-z0-9_.\-]+)==(\S+)", re.M)


def _uv_install(uv: str, packages: List[str], extra: List[str],
                deadline: float) -> Tuple[bool, str]:
    """Dry-run, refuse a torch/numpy change, install, verify the versions."""
    base = [uv, "pip", "install", "--python", sys.executable, *extra]
    left = deadline - time.monotonic()
    if left < 30:
        return False, "time budget used up"
    dry = _run(base + ["--dry-run", *packages], left)
    if dry.returncode != 0:
        return False, f"dry run failed: {(dry.stderr or dry.stdout)[-600:]}"
    changes = _CHANGE.findall(dry.stdout + dry.stderr)
    touched = sorted({name.lower().replace("_", "-") for _, name, _ in changes}
                     & {p.replace("_", "-") for p in PROTECTED})
    if touched:
        return False, f"would change {', '.join(touched)}; not installed"
    added = sorted({name for sign, name, _ in changes if sign == "+"})
    before = _versions()
    real = _run(base + packages, deadline - time.monotonic())
    if real.returncode != 0:
        return False, f"install failed: {(real.stderr or real.stdout)[-600:]}"
    after = _versions()
    if after != before:
        if added:
            _run([uv, "pip", "uninstall", "--python", sys.executable, *added], 300)
        return False, f"changed {before} -> {after}; uninstalled {added}"
    return True, f"installed {', '.join(added) or 'nothing new'}"


def _imports(module: str, timeout: float) -> Tuple[bool, str]:
    r = _run([sys.executable, "-c", f"import {module}"], timeout)
    if r.returncode == 0:
        return True, ""
    return False, (r.stderr or r.stdout).strip()[-800:]


def check_install(budget_s: float = CHECK_BUDGET_S) -> Dict[str, Any]:
    """Try voicefixer, then resemble-enhance; record and return the outcome."""
    started = time.monotonic()
    started_wall = time.time()
    deadline = started + budget_s
    attempts: List[Dict[str, Any]] = []
    engine, error = "none", None
    uv = shutil.which("uv")
    if not uv:
        error = "uv not found on PATH"
    else:
        ok, log = _uv_install(uv, ["voicefixer"], [], deadline)
        if ok:
            ok, err = _imports("voicefixer", deadline - time.monotonic())
            if not ok and err.startswith("timed out"):
                err = ("import (which downloads the weights from Zenodo on first use) "
                       + err + _drop_partial_download(VOICEFIXER_CACHE, started_wall))
            log = log if ok else f"{log}; import failed: {err}"
        attempts.append({"engine": "voicefixer", "ok": ok, "log": log})
        if ok:
            engine = "voicefixer"
        else:
            ok, log = _uv_install(uv, ["resemble-enhance"], ["--no-deps"], deadline)
            if ok:
                ok, log2 = _uv_install(uv, RESEMBLE_LIGHT_DEPS, [], deadline)
                log = f"{log}; {log2}"
            if ok:
                ok, err = _imports("resemble_enhance.enhancer.inference",
                                   deadline - time.monotonic())
                log = log if ok else f"{log}; import failed: {err}"
            attempts.append({"engine": "resemble_enhance", "ok": ok, "log": log})
            if ok:
                engine = "resemble_enhance"
        if engine == "none":
            error = " | ".join(f"{a['engine']}: {a['log']}" for a in attempts)
    result = {"engine": engine, "error": error,
              "checked": datetime.now().isoformat(timespec="seconds"),
              "seconds": round(time.monotonic() - started, 1),
              "attempts": attempts, "versions": _versions()}
    _write_status(result)
    return result


# --- running ------------------------------------------------------------------------

def _ffmpeg(args: List[str]) -> None:
    r = subprocess.run([FFMPEG_BIN, "-hide_banner", "-nostats", "-v", "error", "-y", *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def enhance(in_wav: Path, out_wav: Path, progress: Progress = None) -> Path:
    """Restore speech in `in_wav` with the recorded engine into `out_wav`
    (same rate, channels and length as the input)."""
    import numpy as np
    import soundfile as sf
    import torch

    st = status()
    engine = st["engine"]
    if engine == "none":
        raise RuntimeError(f"Enhance Speech (AI) is not available: {st.get('error')}")
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    info = sf.info(str(in_wav))
    rate, channels, frames = int(info.samplerate), int(info.channels), int(info.frames)
    cuda = torch.cuda.is_available()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="enhance_ai_") as tmp:
        staged, raw = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        if progress:
            progress(0.05, f"{engine}: staging mono audio")
        if engine == "voicefixer":
            _ffmpeg(["-i", str(in_wav), "-ac", "1", "-ar", str(VOICEFIXER_RATE),
                     "-c:a", "pcm_f32le", str(staged)])
            if progress:
                progress(0.15, "voicefixer: loading the model" + (" on the GPU" if cuda else ""))
            from voicefixer import VoiceFixer

            model = VoiceFixer()
            try:
                if progress:
                    progress(0.3, "voicefixer: restoring")
                model.restore(input=str(staged), output=str(raw), cuda=cuda, mode=0)
            finally:
                del model
                _release()
        else:
            import torchaudio
            from resemble_enhance.enhancer.inference import enhance as re_enhance

            _ffmpeg(["-i", str(in_wav), "-ac", "1", "-c:a", "pcm_f32le", str(staged)])
            wav, sr = torchaudio.load(str(staged))
            if progress:
                progress(0.3, "resemble-enhance: restoring" + (" on the GPU" if cuda else ""))
            try:
                hwav, new_sr = re_enhance(wav.mean(0), sr, "cuda" if cuda else "cpu",
                                          nfe=32, solver="midpoint", lambd=0.5, tau=0.5)
                torchaudio.save(str(raw), hwav.unsqueeze(0).cpu(), new_sr)
            finally:
                _release()
        if progress:
            progress(0.85, f"{engine}: back to {rate} Hz")
        back = Path(tmp) / "back.wav"
        _ffmpeg(["-i", str(raw), "-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_f32le",
                 str(back)])
        data, _ = sf.read(str(back), dtype="float32", always_2d=True)
        if len(data) < frames:
            data = np.concatenate([data, np.zeros((frames - len(data), data.shape[1]),
                                                  dtype=np.float32)])
        sf.write(str(out_wav), data[:frames], rate, subtype="FLOAT")
    if progress:
        progress(1.0, f"{engine}: done")
    return out_wav


if __name__ == "__main__":  # python -m audio.enhance_ai [budget_seconds] -> run the check
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else CHECK_BUDGET_S
    print(json.dumps(check_install(budget), indent=2))
