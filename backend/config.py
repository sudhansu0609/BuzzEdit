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
# Which ComfyUI serves generation. START_APP.bat launches ComfyUI from
# C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI on the default port 8188, and
# that install now loads the full B:\ model library via its extra_model_paths.yaml.
# Override with the COMFYUI_URL env var to point elsewhere (e.g. the Desktop app
# on 8001).
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
# ComfyUI Desktop keeps its user data (models, input, output) apart from the
# source tree. Overridable so a different machine does not need a code change.
COMFYUI_BASE_DIR = os.environ.get("COMFYUI_BASE_DIR", r"C:\Users\singh\Documents\ComfyUI")
COMFYUI_INPUT_DIR = os.path.join(COMFYUI_BASE_DIR, "input")
COMFYUI_OUTPUT_DIR = os.path.join(COMFYUI_BASE_DIR, "output")

import shutil
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

