"""A tolerant read of ComfyUI's own self-repair state.

The self-repair work (parallel to this one) publishes `{"state", "detail",
"since", "repair", "updated_at"}` to `data/comfy_status.json` and, once it
lands, to a `runtime/comfy_status.read_status()` helper. Prefer that helper
when it exists; otherwise read the file directly. Either way this never
raises -- a missing or malformed status just means "nothing to show", not a
reason to fail whatever is waiting on ComfyUI.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def read_status() -> Dict[str, Any]:
    """`{}` when there is nothing to report."""
    try:
        from runtime.comfy_status import read_status as _read_status
        return dict(_read_status() or {})
    except Exception:
        pass
    try:
        from config import DATA_DIR
        path = Path(DATA_DIR) / "comfy_status.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
