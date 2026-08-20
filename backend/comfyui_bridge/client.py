"""Talking to a local ComfyUI instance.

Three things here were wrong for a long time and each produced the same symptom —
a generation that "worked" and returned nothing:

  * Only ``images`` outputs were read. Video workflows emit ``gifs``, ``videos``
    or ``animated`` depending on which save node they end in, so every video
    workflow silently returned an empty list and the caller fell back to a grey
    placeholder card.
  * Output files were located by guessing a path under ``COMFYUI_OUTPUT_DIR``.
    ComfyUI can be pointed at any output directory, and on a remote or
    containerised instance there is no shared filesystem at all. The ``/view``
    endpoint is the only answer that always works, so it is the fallback now.
  * ``get_history`` swallowed every exception into ``{}``, which is
    indistinguishable from "the job is still queued". A crashed or restarted
    ComfyUI therefore burned the entire timeout — twenty minutes, for video —
    before reporting anything. Connection errors now abort immediately; a 404 on
    a job that has not started yet is still normal and still ignored.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import COMFYUI_OUTPUT_DIR, COMFYUI_URL, TEMP_DIR

logger = logging.getLogger(__name__)

# Every key a ComfyUI save node may publish its results under. `images` is
# SaveImage/PreviewImage; the rest come from the video nodes (VHS_VideoCombine,
# SaveAnimatedWEBP, SaveAnimatedPNG).
OUTPUT_KEYS = ("images", "gifs", "videos", "animated")


class ComfyUIClient:
    def __init__(self, server_url: str = COMFYUI_URL):
        self.server_url = server_url.rstrip('/')
        self.client_id = "buzzedit_client"

    def is_connected(self, timeout: float = 10.0) -> bool:
        return self.connection_status(timeout)[0]

    def connection_status(self, timeout: float = 10.0) -> tuple[bool, str]:
        """(reachable, reason). The reason is empty when up, otherwise a short
        human explanation so "B-roll didn't work" can become "ComfyUI is not
        running at 127.0.0.1:8188 (connection refused)". `is_connected` collapsed
        connection-refused, a slow model load and a crash into a bare False, which
        is exactly the ambiguity that makes an offline ComfyUI hard to diagnose.
        """
        try:
            req = urllib.request.Request(f"{self.server_url}/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, ""
                return False, f"ComfyUI answered {resp.status} at {self.server_url}"
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            # ConnectionRefused = nothing is listening (not started / wrong port);
            # a timeout usually means it is up but still loading a model.
            msg = (f"ComfyUI is not reachable at {self.server_url} ({reason}). "
                   "Start ComfyUI (START_APP.bat launches it) and confirm the port.")
            logger.warning(msg)
            return False, msg
        except Exception as e:
            logger.warning("ComfyUI status check failed: %s", e)
            return False, f"ComfyUI status check failed: {e}"

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
            # Generous: ComfyUI can block its HTTP server while loading a large
            # model, so /prompt may not answer for a minute or two on a cold model.
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                prompt_id = result.get('prompt_id')
                logger.info(f"ComfyUI prompt queued successfully: {prompt_id}")
                return prompt_id
        except urllib.error.HTTPError as e:
            # ComfyUI answers a rejected workflow with 400 and a JSON body naming
            # the node that failed validation. Passing that through turns "it
            # didn't work" into "node 14 is missing an input".
            body = ""
            try:
                body = e.read().decode('utf-8', 'replace')[:600]
            except Exception:
                pass
            logger.error("ComfyUI rejected the workflow (%s): %s", e.code, body)
            raise RuntimeError(f"ComfyUI rejected the workflow: {body or e}")
        except Exception as e:
            logger.error(f"Failed to queue prompt in ComfyUI: {e}")
            raise RuntimeError(f"ComfyUI queue prompt failed: {e}")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """History for one prompt, or {} if it has not appeared yet.

        Raises ConnectionError when the server itself is unreachable — see the
        module docstring for why that distinction matters.
        """
        try:
            req = urllib.request.Request(f"{self.server_url}/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            # The job is queued but has no history entry yet. Normal; keep waiting.
            if e.code in (404, 400):
                return {}
            logger.warning("ComfyUI history for %s returned %s", prompt_id, e.code)
            return {}
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise ConnectionError(f"ComfyUI is unreachable: {e}") from e
        except Exception as e:
            logger.error(f"Failed to get history for prompt {prompt_id}: {e}")
            return {}

    def _resolve_output(self, entry: Dict[str, Any]) -> Optional[str]:
        """A readable local path for one output entry, downloading it if need be."""
        filename = entry.get("filename")
        if not filename:
            return None
        subfolder = entry.get("subfolder", "") or ""
        folder_type = entry.get("type", "output") or "output"

        for candidate in (
            Path(COMFYUI_OUTPUT_DIR) / subfolder / filename,
            Path(COMFYUI_OUTPUT_DIR) / filename,
        ):
            if candidate.exists():
                return str(candidate)

        # Not on this filesystem. Ask ComfyUI for the bytes.
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type})
        url = f"{self.server_url}/view?{query}"
        destination = Path(TEMP_DIR) / "comfyui" / filename
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                destination.write_bytes(resp.read())
            logger.info("Fetched %s from ComfyUI over HTTP", filename)
            return str(destination)
        except Exception as e:
            logger.error("Could not retrieve %s from ComfyUI: %s", filename, e)
            return None

    def wait_for_prompt(self, prompt_id: str, timeout: int = 300,
                        poll_interval: float = 2.0) -> List[str]:
        """Block until the prompt finishes; return the files it produced."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
            except ConnectionError as e:
                # Almost always ComfyUI blocked loading the model for THIS job.
                # Keep waiting — the overall `timeout` still bounds the whole thing.
                logger.info("ComfyUI busy/unreachable while job %s runs (%s); still waiting",
                            prompt_id, e)
                time.sleep(poll_interval)
                continue
            if prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                files: List[str] = []
                for node_output in outputs.values():
                    for key in OUTPUT_KEYS:
                        for entry in node_output.get(key, []) or []:
                            path = self._resolve_output(entry)
                            if path:
                                files.append(path)
                logger.info("ComfyUI job %s finished with %d output files",
                            prompt_id, len(files))
                if not files:
                    logger.warning(
                        "ComfyUI job %s produced no readable outputs. Its save node "
                        "may write a kind this client does not know: saw keys %s",
                        prompt_id,
                        sorted({k for o in outputs.values() for k in o}) or "none")
                return files

            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout} seconds")
