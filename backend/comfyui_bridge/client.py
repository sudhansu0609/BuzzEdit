import json
import logging
import urllib.request
import urllib.parse
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from config import COMFYUI_URL, COMFYUI_OUTPUT_DIR

logger = logging.getLogger(__name__)


class ComfyUIClient:
    def __init__(self, server_url: str = COMFYUI_URL):
        self.server_url = server_url.rstrip('/')
        self.client_id = "buzzcaf_editor_client"

    def is_connected(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/system_stats")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def queue_prompt(self, prompt_dict: Dict[str, Any]) -> str:
        payload = {
            "prompt": prompt_dict,
            "client_id": self.client_id
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.server_url}/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                prompt_id = result.get('prompt_id')
                logger.info(f"ComfyUI prompt queued successfully: {prompt_id}")
                return prompt_id
        except Exception as e:
            logger.error(f"Failed to queue prompt in ComfyUI: {e}")
            raise RuntimeError(f"ComfyUI queue prompt failed: {e}")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(f"{self.server_url}/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to get history for prompt {prompt_id}: {e}")
            return {}

    def wait_for_prompt(self, prompt_id: str, timeout: int = 300, poll_interval: float = 2.0) -> List[str]:
        start_time = time.time()
        output_files = []

        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                prompt_output = history[prompt_id]
                outputs = prompt_output.get('outputs', {})
                for node_id, node_output in outputs.items():
                    if 'images' in node_output:
                        for img in node_output['images']:
                            filename = img.get('filename')
                            subfolder = img.get('subfolder', '')
                            folder_type = img.get('type', 'output')
                            
                            # Check portable ComfyUI output directory or local folder
                            full_path = Path(COMFYUI_OUTPUT_DIR) / subfolder / filename
                            if not full_path.exists():
                                full_path = Path(COMFYUI_OUTPUT_DIR) / filename

                            output_files.append(str(full_path))
                
                logger.info(f"ComfyUI job {prompt_id} finished with {len(output_files)} output files")
                return output_files

            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout} seconds")
