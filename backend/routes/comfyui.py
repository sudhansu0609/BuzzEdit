from fastapi import APIRouter, HTTPException
from comfyui_bridge.client import ComfyUIClient
from comfyui_bridge.workflow_loader import load_workflow
from config import WORKFLOWS_DIR
import os

router = APIRouter()
client = ComfyUIClient()


@router.get("/status")
async def comfyui_status():
    connected = client.is_connected()
    return {
        "status": "online" if connected else "offline",
        "connected": connected,
        "url": client.server_url
    }


@router.get("/workflows")
async def list_workflows():
    workflows = []
    if WORKFLOWS_DIR.exists():
        for f in os.listdir(WORKFLOWS_DIR):
            if f.endswith(".json"):
                workflows.append(f.replace(".json", ""))
    return {"workflows": workflows}
