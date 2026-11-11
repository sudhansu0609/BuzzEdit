"""BuzzEdit's environment self-test (2026-10-26).

Checks that the environment WORKS, not only that its files exist: after the
2026-10-24 SmartCleaner wipe the rebuilt .venv had every file the launcher
looked for, but CTranslate2 could not find cuBLAS, so Whisper fell back to the
CPU and hung a production for 9 hours.

Run with the backend venv's python from the BuzzEdit folder:
    .venv\\Scripts\\python.exe installer\\selftest.py
Prints one line per check, writes installer\\selftest.json, exits 0 when all
required checks pass. Used by installer\\setup_env.ps1 (a setup that does not
pass is not finished), START_APP.ps1 (before every launch, unless a recent pass
still holds) and the Studio's Health page (app_envs.py).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
RESULT = ROOT / "installer" / "selftest.json"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

checks = []


def check(name, required=True):
    def wrap(fn):
        t0 = time.time()
        try:
            detail = fn() or "ok"
            ok = True
        except Exception as exc:  # noqa: BLE001 - every failure is a result
            detail, ok = f"{type(exc).__name__}: {exc}"[:400], False
        checks.append({"name": name, "ok": ok, "required": required, "detail": str(detail),
                       "s": round(time.time() - t0, 1)})
        print(f"[{'ok' if ok else 'FAIL' if required else 'warn'}] {name}: {detail}", flush=True)
        return fn
    return wrap


sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)


# First, before anything imports torch: torch puts its own CUDA folder on the
# DLL path when imported, which is exactly what hid the 2026-10-26 problem.
@check("Whisper's CUDA libraries load (cuBLAS, cuDNN)")
def _cuda_dlls():
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before the check; the result would mean nothing")
    import asr.faster_whisper_engine  # noqa: F401  registers the DLL folders, as at run time
    if sys.platform != "win32":
        return "not Windows"
    import ctypes
    for dll in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
        ctypes.WinDLL(dll)
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    if n < 1:
        raise RuntimeError("CTranslate2 sees no CUDA device")
    return f"{n} CUDA device(s) for Whisper"


@check("backend packages import")
def _imports():
    import fastapi, faster_whisper, ctranslate2, librosa, torchvision  # noqa: F401,E401
    return f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}"


@check("PyTorch on the GPU")
def _torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} has no CUDA (CPU-only build or driver problem)")
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


@check("backend app loads")
def _app():
    import main
    return f"{len(main.app.routes)} routes"


def _venv_check(rel, code, what):
    py = ROOT / rel
    if not py.exists():
        raise RuntimeError(f"{rel} is missing")
    out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=180, creationflags=NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1:] or [f"{what} failed"])
    return out.stdout.strip()


@check("narration engine on the GPU")
def _tts():
    return _venv_check(r"engines\tts\.venv\Scripts\python.exe",
                       "import torch, chatterbox.mtl_tts, sys; sys.exit(0 if torch.cuda.is_available() else 1); ",
                       "narration") or "torch CUDA ok"


@check("voice conversion engine")
def _seedvc():
    return _venv_check(r"engines\seedvc\.venv\Scripts\python.exe", "import torch; print('torch', torch.__version__)",
                       "seed-vc")


@check("desktop shell", required=True)
def _electron():
    exe = ROOT / "node_modules" / "electron" / "dist" / "electron.exe"
    if not exe.exists():
        raise RuntimeError("node_modules\\electron\\dist\\electron.exe is missing")
    return "electron present"


@check("ffmpeg on PATH", required=False)
def _ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("not found (winget install Gyan.FFmpeg.Shared)")
    return path


@check("motion graphics engine (Remotion)", required=False)
def _motion():
    # A real render, not a file check (MOTION_GRAPHICS_PLAN.md M1): --selftest
    # renders 1s of every registered composition at 320x180. Not required --
    # a video still renders without it, every motion beat just falls back to
    # its static PIL card (presentation/motion.py).
    node = shutil.which("node") or shutil.which("node.exe")
    if not node:
        raise RuntimeError("node not on PATH")
    motion_dir = ROOT / "motion"
    render_script = motion_dir / "render.mjs"
    if not render_script.exists():
        raise RuntimeError("motion/render.mjs is missing")
    if not (motion_dir / "node_modules").is_dir():
        raise RuntimeError("motion/node_modules is missing (npm install in motion\\)")
    out = subprocess.run([node, str(render_script), "--selftest"], cwd=str(motion_dir),
                         capture_output=True, text=True, timeout=180, creationflags=NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError((out.stdout + out.stderr).strip().splitlines()[-1:] or ["selftest failed"])
    lines = [l for l in out.stdout.strip().splitlines() if l.startswith("[selftest]")]
    return "; ".join(lines) or "ok"


ok = all(c["ok"] for c in checks if c["required"])
RESULT.write_text(json.dumps({"ok": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": checks}, indent=1),
                  encoding="utf-8")
print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
