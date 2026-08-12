import asyncio
import logging
from typing import Dict, Any, List, Optional
from comfyui_bridge.client import ComfyUIClient

logger = logging.getLogger(__name__)


class ComfyUIQueueManager:
    def __init__(self, client: Optional[ComfyUIClient] = None):
        self.client = client or ComfyUIClient()
        self._lock = asyncio.Lock()

    async def submit_and_wait(self, workflow_dict: Dict[str, Any], timeout: int = 300) -> List[str]:
        async with self._lock:
            if not self.client.is_connected():
                raise RuntimeError("ComfyUI server is offline or unreachable at localhost:8188")

            prompt_id = await asyncio.to_thread(self.client.queue_prompt, workflow_dict)
            output_files = await asyncio.to_thread(self.client.wait_for_prompt, prompt_id, timeout)
            return output_files
