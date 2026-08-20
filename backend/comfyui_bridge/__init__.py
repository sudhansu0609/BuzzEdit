"""ComfyUI bridge.

`queue_manager` is a **module-level singleton on purpose**. Its `asyncio.Lock`
serialises submissions so two jobs never fight over the GPU — but a lock only
serialises the instance that holds it, and the agent routes and the nightly
worker each used to construct their own manager. A manual B-roll request and an
overnight job could therefore submit at the same moment and both stall. Import
this one; never construct another.
"""

from comfyui_bridge.client import ComfyUIClient
from comfyui_bridge.queue_manager import ComfyUIQueueManager

client = ComfyUIClient()
queue_manager = ComfyUIQueueManager(client)

__all__ = ["ComfyUIClient", "ComfyUIQueueManager", "client", "queue_manager"]
