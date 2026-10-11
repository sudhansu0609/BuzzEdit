"""WS /api/audio/{project_id}/live — the live audio preview socket (AUDIO_SUITE_PLAN.md §2.6).

Registered by main.py under the /api/audio prefix. Query: `scope` = "A1" (default), "MASTER"
or "item:<id>". Client -> server: JSON control messages (play, pause, seek, loop, set, bypass,
rack, credits). Server -> client: binary s16le stereo 48 kHz PCM, never more than credited,
and JSON state / restarted / ended / error. The session itself lives in audio/live.py.
"""

import json
import logging

import anyio
from fastapi import APIRouter, WebSocket

from audio import live
from config import PROJECTS_DIR
from store.project_store import ProjectStore

logger = logging.getLogger(__name__)

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


@router.websocket("/{project_id}/live")
async def audio_live(websocket: WebSocket, project_id: str, scope: str = "A1"):
    await websocket.accept()
    p_data = project_store.get_project_readonly(project_id)
    if not p_data:
        await websocket.send_json({"type": "error", "message": f"project {project_id} not found"})
        await websocket.close(code=4404)
        return
    try:
        source = live.resolve_scope(p_data, scope)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=4400)
        return

    session = live.LiveSession(source, websocket.send_json, websocket.send_bytes)
    await session.start()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            if text is None and message.get("bytes") is not None:
                text = message["bytes"].decode("utf-8", "replace")
            try:
                data = json.loads(text or "")
                if not isinstance(data, dict):
                    raise ValueError("not an object")
            except ValueError as exc:
                await session._error(f"bad message: {exc}")
                continue
            await session.handle(data)
    except Exception as exc:   # the socket went away mid-receive
        logger.debug("audio live socket %s/%s: %s", project_id, scope, exc)
    finally:
        # Shielded: when the server cancels this task, the session must still stop ffmpeg.
        with anyio.CancelScope(shield=True):
            await session.close()
