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
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from config import (
    COMFYUI_OUTPUT_DIR,
    COMFYUI_PREFERRED_PORT,
    COMFYUI_URL,
    PORT_SPAN,
    TEMP_DIR,
)

logger = logging.getLogger(__name__)

# Every key a ComfyUI save node may publish its results under. `images` is
# SaveImage/PreviewImage; the video nodes (VHS_VideoCombine, SaveVideo) publish
# `gifs`/`videos`; `audio` is SaveAudio/SaveAudioMP3/SaveAudioOpus (the text-
# to-audio "audio" role) — same {filename, subfolder, type} entry shape as
# `images`, just under its own key, so leaving it out meant a finished Stable
# Audio Open job reported "produced no readable outputs" every time even
# though ComfyUI's own history had the file right there under "audio".
# `animated` is deliberately NOT here: in history it is a list of booleans
# (`"animated": [true]`) marking sibling `images` entries as animated, not a
# list of files — reading it as one crashed every Wan video generation with
# "'bool' object has no attribute 'get'".
OUTPUT_KEYS = ("images", "gifs", "videos", "audio")
# How long a running job may see "connection refused" before it is given up:
# nothing is listening, so the job died with the process.
REFUSED_GIVE_UP_S = 20


class ComfyUIClient:
    """Talks to whichever ComfyUI is actually up.

    The machine typically has more than one install — the portable one and the
    Comfy Desktop app — and since GUARDIAN_PLAN.md section 11 either of them may
    have stepped forward off its preferred port. Both see the same model folders
    (the Desktop maps the portable's models via extra_models_config.yaml), so
    whichever one the user actually opened is the right one to talk to; the
    configured URL is tried first and a short scan finds the other.
    """

    def __init__(self, server_url: str = COMFYUI_URL):
        self.server_url = server_url.rstrip('/')
        self.client_id = "buzzedit_client"

    def is_connected(self, timeout: float = 10.0) -> bool:
        return self.connection_status(timeout)[0]

    def watch_events(self, on_event, stop, ready=None) -> None:
        """Relay ComfyUI's websocket events for our client_id -- `executing`
        (which node runs), `progress` (sampler step value/max) -- to
        `on_event(type, data)` until `stop` (a threading.Event) is set.

        Progress is best-effort: no websocket means no events, never an error;
        the HTTP history poll still decides when the job is done. `ready` is
        set once the socket is open (or has failed), so the caller can queue
        the prompt without missing its first events.
        """
        url = self.server_url.replace("https://", "wss://").replace("http://", "ws://")
        try:
            from websockets.sync.client import connect
            with connect(f"{url}/ws?clientId={urllib.parse.quote(self.client_id)}",
                         open_timeout=5, max_size=None) as ws:
                if ready is not None:
                    ready.set()
                while not stop.is_set():
                    try:
                        message = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    if not isinstance(message, str):
                        continue   # binary preview frames
                    try:
                        event = json.loads(message)
                        on_event(event.get("type"), event.get("data") or {})
                    except Exception as e:
                        logger.debug("Ignoring ComfyUI event: %s", e)
        except Exception as e:
            logger.info("No ComfyUI progress events (%s); the job still runs", e)
        finally:
            if ready is not None:
                ready.set()

    def _probe(self, url: str, timeout: float) -> tuple[bool, str]:
        try:
            req = urllib.request.Request(f"{url}/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, ""
                return False, f"ComfyUI answered {resp.status} at {url}"
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            # ConnectionRefused = nothing is listening (not started / wrong port);
            # a timeout usually means it is up but still loading a model.
            return False, (f"ComfyUI is not reachable at {url} ({reason}). "
                           "Start ComfyUI (START_APP.bat launches it) and confirm the port.")
        except Exception as e:
            return False, f"ComfyUI status check failed: {e}"

    def connection_status(self, timeout: float = 10.0) -> tuple[bool, str]:
        """(reachable, reason). The reason is empty when up, otherwise a short
        human explanation so "B-roll didn't work" can become "ComfyUI is not
        running at 127.0.0.1:8188 (connection refused)". `is_connected` collapsed
        connection-refused, a slow model load and a crash into a bare False, which
        is exactly the ambiguity that makes an offline ComfyUI hard to diagnose.

        When the configured server is down, the preferred range is scanned and
        the first ComfyUI that answers is adopted — so opening the Desktop app
        instead of the portable install, or a ComfyUI that stepped forward off
        8188 because something else held it, both just work without touching
        COMFYUI_URL.
        """
        ok, reason = self._probe(self.server_url, timeout)
        if ok:
            return True, ""
        # Short per-port timeout: a refused local port costs ~2 s on Windows
        # (it retries the SYN), and the scan covers 20 of them -- at the old 5 s
        # cap one "is it up?" took ~43 s and a submission's retries ~8 minutes.
        alternate = self._find_elsewhere(min(timeout, 0.5))
        if alternate:
            logger.info("ComfyUI found at %s (configured %s is down); switching",
                        alternate, self.server_url)
            self.server_url = alternate
            return True, ""
        logger.warning(reason)
        return False, reason

    def _candidates(self) -> list[str]:
        """Every address worth trying, in order, minus the one that just failed.

        No literal port: the preferred number and the span come from config,
        which reads them from the environment (rule 1), and COMFYUI_ALT_URL is
        there for an install that is nowhere near the usual range.
        """
        seen = {self.server_url}
        found: list[str] = []
        for url in (os.environ.get("COMFYUI_ALT_URL", "").strip(),):
            url = url.rstrip("/")
            if url and url not in seen:
                seen.add(url)
                found.append(url)
        host = urlparse(self.server_url).hostname or "127.0.0.1"
        for port in range(COMFYUI_PREFERRED_PORT, COMFYUI_PREFERRED_PORT + PORT_SPAN + 1):
            url = f"http://{host}:{port}"
            if url not in seen:
                seen.add(url)
                found.append(url)
        return found

    def _find_elsewhere(self, timeout: float) -> Optional[str]:
        for url in self._candidates():
            ok, _ = self._probe(url, timeout)
            if ok:
                return url
        return None

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

    def _post_quiet(self, path: str, body: Dict[str, Any], timeout: float = 30) -> bool:
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception as e:
            logger.warning("ComfyUI %s failed: %s", path, e)
            return False

    def upload_image(self, path: str, timeout: float = 60) -> str:
        """`POST /upload/image` -- put a local picture into ComfyUI's input
        folder and return the name a `LoadImage` node takes. Raises on failure.

        Used to start an image-to-video clip from the still the image phase
        already made, instead of re-generating a first frame inside the video
        graph (which also kept the image model resident beside Wan)."""
        source = Path(path)
        boundary = f"----buzzedit{int(time.time() * 1000)}"
        head = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{source.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
        tail = (f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
                f"\r\n--{boundary}--\r\n").encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/upload/image", data=head + source.read_bytes() + tail,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            answer = json.loads(resp.read().decode("utf-8"))
        name = answer.get("name") or source.name
        subfolder = answer.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def recover(self, prompt_id: Optional[str] = None) -> bool:
        """Clear the way after a failed or timed-out job.

        A job that timed out on OUR side keeps running on ComfyUI's, and every
        later prompt queues behind it -- which turned one slow Wan clip into a
        chain of 30-minute timeouts. So: drop it from the queue, interrupt
        whatever is executing, and ask ComfyUI to unload models and free VRAM
        so the next job starts from a clean card.
        """
        if prompt_id:
            self._post_quiet("/queue", {"delete": [prompt_id]})
        # Recent ComfyUI interrupts only the named prompt; older builds ignore
        # the body and interrupt whatever is running.
        self._post_quiet("/interrupt", {"prompt_id": prompt_id} if prompt_id else {})
        return self._post_quiet("/free", {"unload_models": True, "free_memory": True})

    def has_node(self, class_type: str, timeout: float = 15) -> bool:
        """Whether this ComfyUI has a node class installed (a custom node that
        failed to load is simply absent from /object_info)."""
        try:
            url = f"{self.server_url}/object_info/{urllib.parse.quote(class_type)}"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return class_type in json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            logger.info("ComfyUI node check for %s failed: %s", class_type, e)
            return False

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
        refused_since: Optional[float] = None

        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
            except ConnectionError as e:
                # A timeout is almost always ComfyUI blocked loading the model for
                # THIS job: keep waiting, the overall `timeout` bounds it. A
                # refused connection means nothing is listening any more -- the
                # process died and this job died with it, so do not sit out the
                # full timeout for it.
                text = str(e)
                if "10061" in text or "refused" in text.lower():
                    refused_since = refused_since or time.time()
                    if time.time() - refused_since >= REFUSED_GIVE_UP_S:
                        raise ConnectionError(
                            f"ComfyUI stopped while job {prompt_id} ran (connection refused)") from e
                else:
                    refused_since = None
                logger.info("ComfyUI busy/unreachable while job %s runs (%s); still waiting",
                            prompt_id, e)
                time.sleep(poll_interval)
                continue
            refused_since = None
            if prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                files: List[str] = []
                for node_output in outputs.values():
                    for key in OUTPUT_KEYS:
                        for entry in node_output.get(key, []) or []:
                            # A node may publish flags (booleans) alongside its
                            # file entries; only dicts describe files.
                            if not isinstance(entry, dict):
                                continue
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
