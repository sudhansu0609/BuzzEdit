"""Stage C1 — which ComfyUI workflow runs, and how to fill it in.

The old loader matched nodes by `class_type` and hardcoded node ids "6" and "7"
for the positive and negative prompts. That works for exactly the two workflows
that shipped with the app and breaks on everything else — which meant the user's
own workflows, the entire point of running ComfyUI locally, could not be used.

This module addresses nodes by **path** — `["6", "inputs", "text"]` — and works
out those paths itself by reading the graph. A user drops a workflow file into
`workflows/`, picks it in the settings, and it runs. Where the guess is wrong,
`workflows/manifest.json` overrides it one binding at a time, so a broken guess
never means writing out the whole mapping by hand.

Roles, not filenames, are what the rest of the pass asks for: `broll_image`,
`broll_video`, `thumbnail`, `graphic`. Resolution order is
settings → manifest → built-in default.
"""

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import WORKFLOWS_DIR

logger = logging.getLogger("presentation.workflows")

MANIFEST_FILE = "manifest.json"

ROLES = ("broll_image", "broll_video", "thumbnail", "graphic")

# Which settings key names the workflow for each role.
SETTINGS_KEY = {
    "broll_image": "workflow_broll_image",
    "broll_video": "workflow_broll_video",
    "thumbnail": "workflow_thumbnail",
    "graphic": "workflow_graphic",
}

# Used only when neither settings nor manifest say otherwise. `broll_video` and
# `graphic` have no default on purpose: there is no bundled workflow that can do
# either, and silently substituting an image generator for a video one would
# produce a still where the plan asked for motion.
DEFAULT_FILE = {
    "broll_image": "broll_generation.json",
    "thumbnail": "thumbnail.json",
}

# Node classes that carry a prompt.
_TEXT_CLASSES = ("CLIPTextEncode", "CLIPTextEncodeSDXL", "T5TextEncode",
                 "CLIPTextEncodeFlux", "TextEncodeQwenImageEdit")
# Node classes that define the output geometry — and, when they carry a `length`,
# tell us the workflow makes video rather than a still.
_LATENT_HINTS = ("EmptyLatent", "EmptySD3Latent", "EmptyHunyuanLatentVideo",
                 "EmptyMochiLatentVideo", "WanImageToVideo", "EmptyCosmosLatentVideo")
# Node classes that write the result to disk.
_SAVE_IMAGE = ("SaveImage", "Image Save", "SaveImageWebsocket")
_SAVE_VIDEO = ("VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG",
               "SaveWEBM", "SaveVideo")


class ResolvedWorkflow:
    """A workflow file plus the paths needed to fill it in."""

    def __init__(self, role: str, file: str, graph: Dict[str, Any],
                 bindings: Dict[str, List[str]], output_kind: str):
        self.role = role
        self.file = file
        self.graph = graph
        self.bindings = bindings
        self.output_kind = output_kind      # "image" | "video"

    def build(self, positive: str, negative: str = "", width: int = 1920,
              height: int = 1080, seed: Optional[int] = None,
              length: Optional[int] = None, fps: Optional[int] = None,
              prefix: str = "buzzedit", steps: Optional[int] = None,
              cfg: Optional[float] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """A copy of the graph with this generation's values written into it.

        `steps`, `cfg` and `model` are optional overrides from the app settings:
        when None the workflow keeps its own value, so a user who leaves them
        blank gets exactly the workflow they configured.
        """
        values = {
            "positive": positive,
            "negative": negative,
            "width": width,
            "height": height,
            "seed": seed if seed is not None else random.randint(1, 1_000_000_000),
            "prefix": prefix,
        }
        if length is not None:
            values["length"] = length
        if fps is not None:
            values["fps"] = fps
        if steps is not None:
            values["steps"] = steps
        if cfg is not None:
            values["cfg"] = cfg
        if model:
            values["model"] = model
        return apply_bindings(self.graph, self.bindings, values)


# --- reading the graph -----------------------------------------------------

def load_graph(file_name: str) -> Dict[str, Any]:
    path = WORKFLOWS_DIR / file_name
    if not path.exists() and not file_name.endswith(".json"):
        path = WORKFLOWS_DIR / f"{file_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow {file_name} not found in {WORKFLOWS_DIR}")
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError(f"Workflow {file_name} is not an API-format ComfyUI graph")
    # A UI export has a "nodes" list; the API format is a flat id → node map.
    if "nodes" in graph and isinstance(graph.get("nodes"), list):
        raise ValueError(
            f"Workflow {file_name} looks like a UI export. Save it with "
            f"'Export (API)' in ComfyUI — the UI format has no node ids to bind to.")
    return graph


def _linked_node(graph: Dict[str, Any], node: Dict[str, Any], key: str) -> Optional[str]:
    """The node id feeding `key` of this node, if it is a link."""
    value = (node.get("inputs") or {}).get(key)
    if isinstance(value, list) and value and isinstance(value[0], (str, int)):
        candidate = str(value[0])
        return candidate if candidate in graph else None
    return None


def guess_bindings(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Work out where to write prompts, size and seed by reading the graph.

    This is what lets an unknown workflow just work. Each rule below is chosen to
    be right on the common case and to fail *silently* rather than wrongly — an
    absent binding means that value is left at the workflow's own default, which
    is always a legal generation.
    """
    bindings: Dict[str, List[str]] = {}
    text_nodes: List[str] = []
    sampler_id: Optional[str] = None
    output_kind = "image"

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs") or {}

        if class_type in _TEXT_CLASSES or ("TextEncode" in class_type and "text" in inputs):
            text_nodes.append(node_id)

        if any(hint in class_type for hint in _LATENT_HINTS):
            if "width" in inputs:
                bindings.setdefault("width", [node_id, "inputs", "width"])
            if "height" in inputs:
                bindings.setdefault("height", [node_id, "inputs", "height"])
            if "length" in inputs:
                bindings.setdefault("length", [node_id, "inputs", "length"])
                output_kind = "video"
            elif "num_frames" in inputs:
                bindings.setdefault("length", [node_id, "inputs", "num_frames"])
                output_kind = "video"

        if "seed" in inputs:
            bindings.setdefault("seed", [node_id, "inputs", "seed"])
        elif "noise_seed" in inputs:
            bindings.setdefault("seed", [node_id, "inputs", "noise_seed"])

        # The model loader, so the app can override which checkpoint/diffusion
        # model a workflow runs with. Checkpoints carry their own CLIP+VAE;
        # diffusion_models (UNET) do not, so an override only works within the
        # same family the workflow was built for.
        if "ckpt_name" in inputs:
            bindings.setdefault("model", [node_id, "inputs", "ckpt_name"])
        elif "unet_name" in inputs:
            bindings.setdefault("model", [node_id, "inputs", "unet_name"])
        if "steps" in inputs:
            bindings.setdefault("steps", [node_id, "inputs", "steps"])
        if "cfg" in inputs:
            bindings.setdefault("cfg", [node_id, "inputs", "cfg"])
        if "positive" in inputs and "negative" in inputs:
            sampler_id = node_id

        if class_type in _SAVE_VIDEO:
            output_kind = "video"
            if "filename_prefix" in inputs:
                bindings["prefix"] = [node_id, "inputs", "filename_prefix"]
            if "frame_rate" in inputs:
                bindings.setdefault("fps", [node_id, "inputs", "frame_rate"])
            elif "fps" in inputs:
                bindings.setdefault("fps", [node_id, "inputs", "fps"])
        elif class_type in _SAVE_IMAGE and "filename_prefix" in inputs:
            bindings.setdefault("prefix", [node_id, "inputs", "filename_prefix"])

    # Positive vs negative. Following the sampler's own links is the only reliable
    # way; the fallbacks exist because plenty of graphs route through a
    # ConditioningCombine or a LoRA node in between.
    positive_id = negative_id = None
    if sampler_id:
        positive_id = _trace_text(graph, sampler_id, "positive", text_nodes)
        negative_id = _trace_text(graph, sampler_id, "negative", text_nodes)

    if positive_id is None or negative_id is None:
        if len(text_nodes) == 1:
            positive_id = positive_id or text_nodes[0]
        elif len(text_nodes) >= 2:
            # A negative prompt is a short keyword list; a positive one is a
            # sentence. On a stock workflow that heuristic is right every time.
            ordered = sorted(
                text_nodes,
                key=lambda n: len(str((graph[n].get("inputs") or {}).get("text", ""))),
                reverse=True)
            positive_id = positive_id or ordered[0]
            negative_id = negative_id or ordered[-1]

    if positive_id:
        bindings["positive"] = [positive_id, "inputs", "text"]
    if negative_id and negative_id != positive_id:
        bindings["negative"] = [negative_id, "inputs", "text"]

    bindings["output_kind"] = output_kind
    return bindings


def _trace_text(graph: Dict[str, Any], node_id: str, key: str,
                text_nodes: List[str], depth: int = 0) -> Optional[str]:
    """Follow a conditioning input back to the text node that feeds it."""
    if depth > 6:
        return None
    linked = _linked_node(graph, graph.get(node_id, {}), key)
    if linked is None:
        return None
    if linked in text_nodes:
        return linked
    # Keep walking: conditioning often passes through combine/set-area nodes.
    for candidate_key in ("conditioning", "conditioning_1", "positive", "negative", "text"):
        found = _trace_text(graph, linked, candidate_key, text_nodes, depth + 1)
        if found:
            return found
    return None


def apply_bindings(graph: Dict[str, Any], bindings: Dict[str, Any],
                   values: Dict[str, Any]) -> Dict[str, Any]:
    """A deep copy of the graph with `values` written at the bound paths.

    A value whose binding is missing or does not resolve is skipped rather than
    raising: a workflow with no `steps` input simply keeps its own step count,
    which is a working generation, not an error.

    A binding may be one path (`["6", "inputs", "text"]`) or a LIST of paths —
    a two-stage workflow (image model feeding a video model) needs the same
    prompt, size and seed written into both stages. An empty list is a valid
    manifest entry meaning "never write this value here": it protects a
    multi-stage graph from global overrides (steps/cfg/model) that only make
    sense for a single-model workflow.
    """
    result = copy.deepcopy(graph)
    for name, value in values.items():
        binding = bindings.get(name)
        if not isinstance(binding, list):
            continue
        paths = binding if (binding and isinstance(binding[0], list)) else [binding]
        for path in paths:
            if not isinstance(path, list) or len(path) < 2:
                continue
            node = result.get(str(path[0]))
            if not isinstance(node, dict):
                continue
            target = node
            for key in path[1:-1]:
                target = target.get(key) if isinstance(target, dict) else None
                if target is None:
                    break
            if isinstance(target, dict) and path[-1] in target:
                target[path[-1]] = value
    return result


# --- manifest and settings -------------------------------------------------

def load_manifest() -> Dict[str, Any]:
    path = WORKFLOWS_DIR / MANIFEST_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}


def _settings_value(key: str) -> Optional[str]:
    try:
        from store.app_settings import AppSettings
        value = AppSettings().get(key)
        return str(value) if value else None
    except Exception:
        return None


def resolve(role: str) -> Optional[ResolvedWorkflow]:
    """The workflow to use for a role, fully bound, or None if unconfigured."""
    if role not in ROLES:
        return None
    manifest = load_manifest().get("roles", {}).get(role, {})
    file_name = _settings_value(SETTINGS_KEY[role]) or manifest.get("file") \
        or DEFAULT_FILE.get(role)
    if not file_name:
        return None

    try:
        graph = load_graph(file_name)
    except Exception as e:
        logger.warning("Workflow for %s (%s) could not be loaded: %s", role, file_name, e)
        return None

    bindings = guess_bindings(graph)
    # Manifest bindings win, key by key — so one wrong guess can be corrected
    # without writing out the whole mapping.
    for key, path in (manifest.get("bindings") or {}).items():
        bindings[key] = path

    output_kind = str(bindings.pop("output_kind", "image"))
    if role == "broll_video" and output_kind != "video":
        logger.warning(
            "Workflow %s is configured for video but has no length or video save "
            "node; video beats will fall back to stills.", file_name)
    return ResolvedWorkflow(role, file_name, graph, bindings, output_kind)


def describe(file_name: str) -> Dict[str, Any]:
    """What the settings UI shows about one workflow file."""
    try:
        graph = load_graph(file_name)
    except Exception as e:
        return {"file": file_name, "valid": False, "error": str(e)}
    bindings = guess_bindings(graph)
    output_kind = bindings.get("output_kind", "image")
    return {
        "file": file_name,
        "valid": True,
        "nodes": len(graph),
        "output_kind": output_kind,
        "bindings": {k: v for k, v in bindings.items() if k != "output_kind"},
        "can_prompt": "positive" in bindings,
        "roles_ok": {
            "broll_image": "positive" in bindings and output_kind == "image",
            "broll_video": "positive" in bindings and output_kind == "video",
            "thumbnail": "positive" in bindings and output_kind == "image",
            "graphic": "positive" in bindings and output_kind == "image",
        },
    }


def list_workflows() -> List[Dict[str, Any]]:
    """Every workflow file on disk, with what it can do."""
    if not WORKFLOWS_DIR.exists():
        return []
    files = sorted(p.name for p in WORKFLOWS_DIR.glob("*.json")
                   if p.name != MANIFEST_FILE)
    return [describe(name) for name in files]


def current_selection() -> Dict[str, Optional[str]]:
    """Which file each role resolves to right now."""
    selection: Dict[str, Optional[str]] = {}
    manifest_roles = load_manifest().get("roles", {})
    for role in ROLES:
        selection[role] = (_settings_value(SETTINGS_KEY[role])
                           or manifest_roles.get(role, {}).get("file")
                           or DEFAULT_FILE.get(role))
    return selection
