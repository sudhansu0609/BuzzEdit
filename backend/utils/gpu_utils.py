import gc
import logging
import torch

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_name": "CPU Only",
            "total_vram_mb": 0,
            "allocated_vram_mb": 0,
            "free_vram_mb": 0,
        }

    try:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)

        free_mb = round(free_bytes / (1024 * 1024), 2)
        total_mb = round(total_bytes / (1024 * 1024), 2)
        allocated_mb = round((total_bytes - free_bytes) / (1024 * 1024), 2)

        return {
            "available": True,
            "device_name": device_name,
            "total_vram_mb": total_mb,
            "allocated_vram_mb": allocated_mb,
            "free_vram_mb": free_mb,
        }
    except Exception as e:
        logger.error(f"Error querying GPU info: {e}")
        return {
            "available": False,
            "error": str(e),
            "device_name": "NVIDIA GPU",
            "total_vram_mb": 0,
            "allocated_vram_mb": 0,
            "free_vram_mb": 0,
        }


def clear_vram_cache():
    logger.info("Clearing PyTorch CUDA memory cache and running GC...")
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"Error flushing CUDA cache: {e}")


def is_oom_error(exception: Exception) -> bool:
    err_str = str(exception).lower()
    return "out of memory" in err_str or "cuda error: out of memory" in err_str or "oom" in err_str
