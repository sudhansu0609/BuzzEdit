import gc
import logging
from runtime.gpu_broker import gpu_broker, NVML_AVAILABLE

logger = logging.getLogger(__name__)

try:
    import pynvml
except Exception:
    pynvml = None

def get_gpu_info() -> dict:
    if not NVML_AVAILABLE or pynvml is None:
        return {
            "available": False,
            "device_name": "CPU / Fallback",
            "total_vram_mb": 16384,
            "allocated_vram_mb": 0,
            "free_vram_mb": 16384,
        }

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8")
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        total_mb = round(info.total / (1024 * 1024), 2)
        free_mb = round(info.free / (1024 * 1024), 2)
        allocated_mb = round(info.used / (1024 * 1024), 2)

        return {
            "available": True,
            "device_name": str(device_name),
            "total_vram_mb": total_mb,
            "allocated_vram_mb": allocated_mb,
            "free_vram_mb": free_mb,
        }
    except Exception as e:
        logger.error(f"Error querying GPU info via NVML: {e}")
        return {
            "available": False,
            "error": str(e),
            "device_name": "NVIDIA GPU",
            "total_vram_mb": 0,
            "allocated_vram_mb": 0,
            "free_vram_mb": 0,
        }

def clear_vram_cache():
    logger.info("Running garbage collection...")
    gc.collect()

def is_oom_error(exception: Exception) -> bool:
    err_str = str(exception).lower()
    return "out of memory" in err_str or "cuda error: out of memory" in err_str or "oom" in err_str
