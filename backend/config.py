import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
WORKFLOWS_DIR = BASE_DIR / "workflows"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"

# ───────────────────────────── ports ─────────────────────────────
#
# GUARDIAN_PLAN.md section 11. These are the *preferred* numbers and nothing
# more: if something already holds one, the app steps forward and publishes
# where it actually landed. Nothing downstream may assume them — read
# `bound_port()` and `COMFYUI_URL`, which are the truth after startup.
APP_NAME = "buzzedit"
APP_VERSION = "0.2.0"
BIND_HOST = os.environ.get("BUZZEDIT_HOST", "127.0.0.1")
PREFERRED_PORT = int(os.environ.get("BUZZEDIT_PORT") or 8099)
COMFYUI_PREFERRED_PORT = int(os.environ.get("COMFYUI_PORT") or 8188)
PORT_SPAN = 20

# Which ComfyUI serves generation. Electron picks its port too and passes it
# down as COMFYUI_URL, so the default below only applies to a backend started
# on its own. Override to point elsewhere (e.g. the Desktop app on 8001).
COMFYUI_URL = os.environ.get(
    "COMFYUI_URL", f"http://127.0.0.1:{COMFYUI_PREFERRED_PORT}"
)

# The port the server actually bound. Set once by `main._serve()`; the health
# route reports it, so a scan that finds us learns where we really are.
_BOUND_PORT = PREFERRED_PORT


def set_bound_port(port: int) -> None:
    """Record the port uvicorn was handed. Called once, before it serves."""
    global _BOUND_PORT
    _BOUND_PORT = int(port)
    # Anything this process spawns inherits the real number rather than the wish.
    os.environ["BUZZEDIT_PORT"] = str(_BOUND_PORT)


def bound_port() -> int:
    return _BOUND_PORT


def comfyui_port() -> int:
    """The port of the ComfyUI we are actually pointed at."""
    from urllib.parse import urlparse

    return urlparse(COMFYUI_URL).port or COMFYUI_PREFERRED_PORT
# ComfyUI Desktop keeps its user data (models, input, output) apart from the
# source tree. Overridable so a different machine does not need a code change.
COMFYUI_BASE_DIR = os.environ.get("COMFYUI_BASE_DIR", r"C:\Users\singh\Documents\ComfyUI")
COMFYUI_INPUT_DIR = os.path.join(COMFYUI_BASE_DIR, "input")
COMFYUI_OUTPUT_DIR = os.path.join(COMFYUI_BASE_DIR, "output")

import shutil
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

