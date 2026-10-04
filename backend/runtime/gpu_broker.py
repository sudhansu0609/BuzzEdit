import asyncio
import logging
import requests
from typing import Optional

logger = logging.getLogger("gpu_broker")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception as e:
    logger.warning(f"pynvml initialization failed: {e}. GPU VRAM monitoring fallback active.")
    NVML_AVAILABLE = False

class GPUBroker:
    """
    VRAM lease broker managing GPU memory across ComfyUI, faster-whisper, and LM Studio tenants.
    """
    _instance: Optional["GPUBroker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_tenant = None
            cls._instance.lock = asyncio.Lock()
        return cls._instance

    def get_free_vram_mb(self) -> float:
        """Get available VRAM in MB using NVML."""
        if not NVML_AVAILABLE:
            return 16384.0  # Fallback estimate
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / (1024.0 * 1024.0)
        except Exception as e:
            logger.warning(f"NVML get memory info error: {e}")
            return 8192.0

    def get_total_vram_mb(self) -> float:
        """Total VRAM on the card in MB — the ceiling a tenant can ever fit in.

        Free memory answers "can this load right now"; total answers "can this
        machine run this at all". A language model already resident but larger
        than the card is not usable just because it loaded: llama.cpp offloaded
        part of it and it dies on a real prompt.
        """
        if not NVML_AVAILABLE:
            return 16384.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.total) / (1024.0 * 1024.0)
        except Exception as e:
            logger.warning(f"NVML get memory info error: {e}")
            return 16384.0

    async def release_comfyui_vram(self, comfyui_url: Optional[str] = None) -> bool:
        """Request ComfyUI to unload models and free VRAM if queue is idle."""
        if comfyui_url is None:
            from config import COMFYUI_URL
            comfyui_url = COMFYUI_URL
        try:
            # Check if queue is empty first
            res = requests.get(f"{comfyui_url}/queue", timeout=3)
            if res.status_code == 200:
                data = res.json()
                exec_info = data.get("queue_running", []) + data.get("queue_pending", [])
                if len(exec_info) > 0:
                    logger.info("ComfyUI has active jobs in queue, skipping VRAM unload.")
                    return False

            # Free VRAM
            post_res = requests.post(f"{comfyui_url}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
            if post_res.status_code == 200:
                logger.info("ComfyUI VRAM successfully released.")
                return True
        except Exception as e:
            logger.debug(f"ComfyUI /free call failed or unreachable: {e}")
        return False

    async def acquire_lease(self, tenant_name: str, required_vram_mb: float = 4000.0) -> None:
        """Acquire GPU lease for tenant_name."""
        async with self.lock:
            logger.info(f"Tenant '{tenant_name}' requesting lease (needs ~{required_vram_mb:.0f} MB).")
            free_vram = self.get_free_vram_mb()
            logger.info(f"Current free VRAM: {free_vram:.0f} MB.")

            # ComfyUI asking for the card must NOT make ComfyUI unload: the VRAM
            # "in use" is its own models from the previous job, which the next
            # job reuses. Freeing them here made every image and every clip
            # reload ~6-20GB of weights from disk -- a 5s turbo image took 35s
            # and each Wan clip spent minutes loading. Only another tenant
            # (Whisper, the LLM, the matte) needs ComfyUI to let go.
            if free_vram < required_vram_mb and tenant_name != "comfyui":
                await self.release_comfyui_vram()
                await asyncio.sleep(0.5)

            self.current_tenant = tenant_name
            logger.info(f"Lease granted to tenant '{tenant_name}'.")

    async def release_lease(self, tenant_name: str) -> None:
        """Release GPU lease for tenant_name."""
        async with self.lock:
            if self.current_tenant == tenant_name:
                self.current_tenant = None
                logger.info(f"Lease released by tenant '{tenant_name}'.")

gpu_broker = GPUBroker()
