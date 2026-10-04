from fastapi import APIRouter
from pathlib import Path
from utils.gpu_utils import get_gpu_info, clear_vram_cache
from utils.ffmpeg_utils import run_ffprobe
from config import FFMPEG_BIN, COMFYUI_URL, COMFYUI_INPUT_DIR, COMFYUI_OUTPUT_DIR
from comfyui_bridge.client import ComfyUIClient
import shutil

router = APIRouter()
comfy_client = ComfyUIClient()


@router.get("/gpu")
async def gpu_status():
    return get_gpu_info()


@router.get("/paths")
async def system_paths_status():
    ffmpeg_ok = False
    try:
        run_ffprobe("NUL")
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = shutil.which("ffmpeg") is not None

    comfy_conn = comfy_client.is_connected()
    input_dir_exists = Path(COMFYUI_INPUT_DIR).exists()
    output_dir_exists = Path(COMFYUI_OUTPUT_DIR).exists()

    return {
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_bin": FFMPEG_BIN,
        "comfyui_connected": comfy_conn,
        "comfyui_url": COMFYUI_URL,
        "comfyui_input_dir": str(COMFYUI_INPUT_DIR),
        "comfyui_input_dir_exists": input_dir_exists,
        "comfyui_output_dir": str(COMFYUI_OUTPUT_DIR),
        "comfyui_output_dir_exists": output_dir_exists,
    }


@router.post("/clear_vram")
async def trigger_vram_clear():
    """Give the card back: drop this process's Whisper and aligner weights
    (they stay resident after a transcription) and ask an idle ComfyUI to
    unload its models. The Studio calls this before it plans visuals on a
    local LLM, which could not fit beside them on a 16 GB card."""
    from runtime import gpu_handover

    gpu_handover.release_in_process_models()
    comfyui_freed = await gpu_handover.free_comfyui()
    clear_vram_cache()
    info = get_gpu_info()
    return {"status": "cleared", "comfyui_freed": comfyui_freed, "gpu": info}


@router.post("/test_comfyui")
async def test_comfyui_connection():
    connected = comfy_client.is_connected()
    return {"connected": connected, "url": COMFYUI_URL}
