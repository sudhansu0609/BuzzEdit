import asyncio
import logging
from typing import Any, Dict, List, Optional

from comfyui_bridge.client import ComfyUIClient
from runtime.gpu_broker import gpu_broker

logger = logging.getLogger(__name__)


class ComfyUIQueueManager:
    """Serialises ComfyUI submissions and claims the GPU while one is running.

    ComfyUI was the one tenant the broker knew how to evict but that never asked
    for anything itself — Whisper and LM Studio both take leases, so an overnight
    run could start a diffusion job while the language model still held 7GB. The
    lease here is what makes the hand-off between stages orderly.
    """

    def __init__(self, client: Optional[ComfyUIClient] = None):
        self.client = client or ComfyUIClient()
        self._lock = asyncio.Lock()

    async def submit_and_wait(self, workflow_dict: Dict[str, Any], timeout: int = 300,
                              required_vram_mb: float = 8000.0) -> List[str]:
        async with self._lock:
            # ComfyUI blocks its HTTP server while loading a large model, so a
            # single failed health check does not mean it is down. Give it up to
            # ~90s to answer before treating it as truly offline.
            connected = False
            for attempt in range(9):
                if await asyncio.to_thread(self.client.is_connected):
                    connected = True
                    break
                await asyncio.sleep(10)
            if not connected:
                raise RuntimeError(f"ComfyUI server is offline or unreachable at {self.client.server_url}")

            await gpu_broker.acquire_lease("comfyui", required_vram_mb=required_vram_mb)
            try:
                prompt_id = await asyncio.to_thread(self.client.queue_prompt, workflow_dict)
                return await asyncio.to_thread(self.client.wait_for_prompt, prompt_id, timeout)
            finally:
                await gpu_broker.release_lease("comfyui")
