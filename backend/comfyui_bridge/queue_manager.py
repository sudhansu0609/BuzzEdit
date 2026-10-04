import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from comfyui_bridge.client import ComfyUIClient
from runtime.gpu_broker import gpu_broker

logger = logging.getLogger(__name__)

# How long a submission waits for a ComfyUI that does not answer: covers a
# model load blocking its HTTP server and the launcher restarting it after a crash.
OFFLINE_WAIT_S = 150


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
        # Set once a full wait found ComfyUI gone. Later submissions then probe
        # once instead of waiting again: on 2026-10-01 every asset, music cue
        # and the thumbnail sat out its own ~8-minute wait on a dead ComfyUI.
        self._offline = False

    async def _wait_until_up(self) -> bool:
        """True once ComfyUI answers.

        ComfyUI blocks its HTTP server while loading a large model, and after a
        crash BuzzEdit's launcher restarts it (a few seconds' pause plus ~40 s
        of startup) -- so a first failed check gets up to OFFLINE_WAIT_S. After
        one such wait has failed, it is one quick look per submission until
        ComfyUI is seen again.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (0 if self._offline else OFFLINE_WAIT_S)
        while True:
            if await asyncio.to_thread(self.client.is_connected, 5.0):
                if self._offline:
                    logger.info("ComfyUI is back at %s", self.client.server_url)
                self._offline = False
                return True
            if loop.time() >= deadline:
                if not self._offline:
                    logger.warning("ComfyUI did not come back within %ds; failing fast until it does",
                                   OFFLINE_WAIT_S)
                self._offline = True
                return False
            await asyncio.sleep(5)

    async def submit_and_wait(self, workflow_dict: Dict[str, Any], timeout: int = 300,
                              required_vram_mb: float = 8000.0,
                              on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
                              ) -> List[str]:
        """Run one workflow; `on_event(type, data)`, when given, receives
        ComfyUI's live websocket events (see ComfyUIClient.watch_events),
        called from a watcher thread."""
        async with self._lock:
            if not await self._wait_until_up():
                raise RuntimeError(f"ComfyUI server is offline or unreachable at {self.client.server_url}")

            await gpu_broker.acquire_lease("comfyui", required_vram_mb=required_vram_mb)
            prompt_id: Optional[str] = None
            stop = threading.Event()
            if on_event is not None:
                ready = threading.Event()
                threading.Thread(target=self.client.watch_events, args=(on_event, stop, ready),
                                 name="comfyui-events", daemon=True).start()
                await asyncio.to_thread(ready.wait, 6)
            try:
                prompt_id = await asyncio.to_thread(self.client.queue_prompt, workflow_dict)
                outputs = await asyncio.to_thread(self.client.wait_for_prompt, prompt_id, timeout)
                if not outputs:
                    await asyncio.to_thread(self.client.recover, prompt_id)
                return outputs
            except Exception:
                # Still inside the lock: the next submission must not start
                # until the stuck job is gone and the card is free again. Never
                # interrupt when nothing of ours was queued -- that would kill
                # whatever the user is running in ComfyUI by hand.
                if prompt_id:
                    await asyncio.to_thread(self.client.recover, prompt_id)
                raise
            finally:
                stop.set()
                await gpu_broker.release_lease("comfyui")
