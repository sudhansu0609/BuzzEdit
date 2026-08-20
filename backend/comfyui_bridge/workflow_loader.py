import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from config import WORKFLOWS_DIR

logger = logging.getLogger(__name__)


def load_workflow(workflow_name: str) -> Dict[str, Any]:
    file_path = WORKFLOWS_DIR / f"{workflow_name}.json"
    if not file_path.exists():
        file_path = WORKFLOWS_DIR / workflow_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Workflow {workflow_name} not found at {file_path}")

    return json.loads(file_path.read_text())


def prepare_thumbnail_workflow(
    prompt: str,
    negative_prompt: str = "boring, dull, low quality, dark, blurry",
    width: int = 1280,
    height: int = 720,
    seed: Optional[int] = None,
    output_prefix: str = "buzzedit_thumbnail",
) -> Dict[str, Any]:
    workflow = load_workflow("thumbnail.json")

    import random
    if seed is None:
        seed = random.randint(1, 1000000000)

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncode":
            if node_id == "6" or "positive" in node_id.lower():
                inputs["text"] = prompt
            elif node_id == "7" or "negative" in node_id.lower():
                inputs["text"] = negative_prompt

        elif class_type == "EmptyLatentImage":
            inputs["width"] = width
            inputs["height"] = height

        elif class_type == "KSampler":
            inputs["seed"] = seed

        elif class_type == "SaveImage":
            inputs["filename_prefix"] = output_prefix

    return workflow
