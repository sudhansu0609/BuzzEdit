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

import asyncio
import importlib.util
import os

import requests
from fastapi import APIRouter

from config import APP_NAME, APP_VERSION, COMFYUI_URL, bound_port
from models import HealthResponse
from runtime import code_reload
from runtime.comfy_status import comfy_state_and_detail
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
    # The identity block first, and unconditionally: the launcher polls this
    # route to decide whether the thing that answered on the port it picked is
    # really us, so it has to be right even when every probe below fails.
    response = HealthResponse(
        app=APP_NAME,
        status="ok",
        port=bound_port(),
        pid=os.getpid(),
        version=APP_VERSION,
    )

    # The ffprobe spawn and the ComfyUI request are blocking calls; run on the
    # event loop they held every other request (the Studio's transcription
    # status polls included) for as long as they took. Off the loop, and the
    # ffmpeg answer is remembered: the binary does not come and go.
    global _ffmpeg_available
    if _ffmpeg_available is None:
        _ffmpeg_available = await asyncio.to_thread(_probe_ffmpeg)
    response.ffmpeg_available = _ffmpeg_available

    # faster-whisper is the engine this app transcribes with; the old probe
    # looked for `whisper` (openai-whisper), which is not a dependency, so a
    # working install always reported transcription unavailable.
    response.whisper_available = _module_installed("faster_whisper")

    response.comfyui_connected = await asyncio.to_thread(_probe_comfyui)

    # The launcher's own view (starting/running/repairing/down/repair_failed).
    # A file that still says "running" when our own probe just failed is stale
    # -- the probe above is the truth, so that disagreement is reported as down.
    comfy_state, comfy_detail = comfy_state_and_detail()
    if comfy_state == "running" and not response.comfyui_connected:
        comfy_state = "down"
    response.comfyui_state = comfy_state
    response.comfyui_detail = comfy_detail
    response.reload_pending = code_reload.reload_pending()

    return response


@router.get("/reload/busy")
async def reload_busy():
    """{busy, reason}: why this process must not be restarted/quit right now
    (STUDIO_DIRECTOR_PLAN.md enhancement 4) -- `code_reload.busy_reason`'s own
    check (reels, narration, the scheduler, ComfyUI, a master-pause hold),
    the one the Electron watchdog polls instead of quitting a working
    BuzzEdit the moment its Studio parent disappears."""
    route_tasks = code_reload._route_tasks()
    reason = await asyncio.to_thread(code_reload.busy_reason, route_tasks)
    return {"busy": bool(reason), "reason": reason}


_ffmpeg_available = None


def _probe_ffmpeg() -> bool:
    try:
        run_ffprobe("NUL")
    except FileNotFoundError:
        return False
    except Exception:
        return True
    return True


def _probe_comfyui() -> bool:
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=_COMFYUI_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except Exception:
        return False
