"""Health probe.

This endpoint is on the app's startup path: Electron polls it and gives each
call **two seconds** before it retries, twenty times, and then refuses to start
with "Backend failed to start". So everything here has to be cheap and, above
all, must not touch anything that loads a native library. Probing whisper by
importing `faster_whisper` cost exactly that — the import pulls in CTranslate2
and the CUDA DLLs, which takes far longer than the two-second budget on a cold
start and can pop a Windows "procedure entry point could not be located" dialog
when the DLL search path has both a cu118 and a cu12 toolchain on it.

Availability is therefore checked by *finding* the module, never importing it.
"""

import importlib.util

import requests
from fastapi import APIRouter

from config import COMFYUI_URL
from models import HealthResponse
from utils.ffmpeg_utils import run_ffprobe

router = APIRouter()

# ComfyUI is optional and may be starting up or absent; a long wait here would
# eat the startup budget on its own.
_COMFYUI_TIMEOUT_SECONDS = 0.5


def _module_installed(name: str) -> bool:
    """Whether a module could be imported — without importing it.

    `find_spec` reads packaging metadata only. It loads no DLLs, initialises no
    CUDA context, and cannot hang.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    response = HealthResponse()

    try:
        run_ffprobe("NUL")
    except FileNotFoundError:
        response.ffmpeg_available = False
    except Exception:
        response.ffmpeg_available = True

    # faster-whisper is the engine this app transcribes with; the old probe
    # looked for `whisper` (openai-whisper), which is not a dependency, so a
    # working install always reported transcription unavailable.
    response.whisper_available = _module_installed("faster_whisper")

    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats",
                            timeout=_COMFYUI_TIMEOUT_SECONDS)
        response.comfyui_connected = resp.status_code == 200
    except Exception:
        response.comfyui_connected = False

    return response
