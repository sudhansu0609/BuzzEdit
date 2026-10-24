"""Read ComfyUI's environment status, published by electron/main.cjs.

The launcher tracks whether ComfyUI is starting, running, being repaired, down
or failed to repair (see electron/main.cjs's `setComfyStatus`) and writes it
atomically to `data/comfy_status.json` so the backend's health routes can
report it without reaching into Electron. A missing, unreadable or malformed
file is not an error here -- it just means "no opinion" ({} / (None, None)),
the same as before this file existed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from config import DATA_DIR

STATUS_PATH = DATA_DIR / "comfy_status.json"


def read_status() -> Dict[str, Any]:
    """The parsed status file, or `{}` if it is missing, unreadable or not a
    JSON object. Never raises."""
    try:
        raw = STATUS_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def comfy_state_and_detail() -> Tuple[Optional[str], Optional[str]]:
    """(state, detail) from the status file, or (None, None) if there is
    nothing usable to report."""
    data = read_status()
    state = data.get("state")
    detail = data.get("detail")
    return (state if isinstance(state, str) else None,
            detail if isinstance(detail, str) else None)
