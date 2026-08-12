from fastapi import APIRouter, HTTPException
from models import HealthResponse
from utils.ffmpeg_utils import run_ffprobe, FFmpegError
from config import COMFYUI_URL
import subprocess
import requests

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    response = HealthResponse()

    try:
        run_ffprobe("NUL")
    except FileNotFoundError:
        response.ffmpeg_available = False
    except Exception:
        response.ffmpeg_available = True

    try:
        import whisper
        response.whisper_available = True
    except ImportError:
        response.whisper_available = False

    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
        if resp.status_code == 200:
            response.comfyui_connected = True
    except Exception:
        response.comfyui_connected = False

    return response
