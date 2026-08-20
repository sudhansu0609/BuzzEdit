from fastapi import APIRouter, HTTPException
from comfyui_bridge.client import ComfyUIClient
from comfyui_bridge.workflow_loader import load_workflow
from config import WORKFLOWS_DIR
import os

router = APIRouter()
client = ComfyUIClient()


@router.get("/status")
async def comfyui_status():
    connected, reason = client.connection_status()
    return {
        "status": "online" if connected else "offline",
        "connected": connected,
        "reason": reason,
        "url": client.server_url
    }


@router.get("/workflows")
async def list_workflows():
    """Every workflow file on disk with its auto-detected bindings, which roles
    it can serve, and which file each role currently resolves to.

    Drives the Generation settings dropdowns (image / video / thumbnail). The
    `settings_key` map tells the UI which app-setting to PUT when the user picks
    a file for a role.
    """
    from presentation import workflows as wf
    return {
        "workflows": wf.list_workflows(),
        "roles": list(wf.ROLES),
        "selection": wf.current_selection(),
        "settings_key": wf.SETTINGS_KEY,
    }


@router.get("/models")
async def comfyui_models():
    """The model files ComfyUI can load, per loader type, for the settings UI.

    Read live from ComfyUI's `/object_info` so it reflects whatever the running
    server actually sees (including the extra paths from extra_model_paths.yaml).
    Returns empty lists if ComfyUI is offline.
    """
    import httpx
    wanted = {
        "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
        "diffusion_models": ("UNETLoader", "unet_name"),
        "loras": ("LoraLoader", "lora_name"),
        "vae": ("VAELoader", "vae_name"),
    }
    result = {k: [] for k in wanted}
    try:
        async with httpx.AsyncClient(timeout=6.0) as http:
            for dest, (cls, key) in wanted.items():
                try:
                    r = await http.get(f"{client.server_url}/object_info/{cls}")
                    if r.status_code != 200:
                        continue
                    spec = (r.json().get(cls, {}).get("input", {})
                            .get("required", {}).get(key))
                    if isinstance(spec, list) and spec and isinstance(spec[0], list):
                        result[dest] = spec[0]
                except Exception:
                    continue
    except Exception:
        pass
    return result
