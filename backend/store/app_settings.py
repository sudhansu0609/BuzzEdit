import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

class AppSettings:
    """
    Persistent app-level settings store for crash recovery and session persistence.
    Stores: last_project_id, auto_save_interval, last_save_timestamp, etc.
    """
    _instance: Optional["AppSettings"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.settings_dir = Path(__file__).parent.parent / "data"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.settings_dir / "app_settings.json"
        self._cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "last_project_id": None,
            "auto_save_enabled": True,
            "auto_save_interval_seconds": 30,
            "last_save_timestamp": None,
        }

    def _save(self) -> None:
        temp_fd, temp_path = tempfile.mkstemp(dir=str(self.settings_dir), prefix="settings_tmp_")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.settings_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._save()

    def get_all(self) -> Dict[str, Any]:
        return dict(self._cache)

    def update(self, updates: Dict[str, Any]) -> None:
        self._cache.update(updates)
        self._save()

    def set_last_project(self, project_id: Optional[str]) -> None:
        self.set("last_project_id", project_id)

    def get_last_project(self) -> Optional[str]:
        return self.get("last_project_id")

    def mark_auto_save(self) -> None:
        self.set("last_save_timestamp", datetime.utcnow().isoformat())


app_settings = AppSettings()


# --- BuzzEdit-side presentation overrides (beat Studio) ---------------------
#
# BuzzcafStudio enqueues presentation jobs with its own settings dict; these
# let the BuzzEdit user pin fields of their own that win over whatever Studio
# sent, without touching the Studio repo. `presentation_overrides` is a flat
# `{field: value}` dict of PresentationSettings field names; `override_mode`
# decides whether it applies at all.
PRESENTATION_OVERRIDES_KEY = "presentation_overrides"
OVERRIDE_MODE_KEY = "override_mode"
OVERRIDE_MODES = ("off", "fields", "all")


def get_presentation_overrides() -> Dict[str, Any]:
    return dict(app_settings.get(PRESENTATION_OVERRIDES_KEY, {}) or {})


def get_override_mode() -> str:
    mode = app_settings.get(OVERRIDE_MODE_KEY, "off")
    return mode if mode in OVERRIDE_MODES else "off"


def set_presentation_overrides(fields: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """`PUT /api/settings/presentation_overrides`: persist the override fields
    and mode, and return the state as `get_presentation_overrides_state` would."""
    mode = mode if mode in OVERRIDE_MODES else "off"
    app_settings.update({
        PRESENTATION_OVERRIDES_KEY: dict(fields or {}),
        OVERRIDE_MODE_KEY: mode,
    })
    return get_presentation_overrides_state()


def get_presentation_overrides_state() -> Dict[str, Any]:
    return {"presentation_overrides": get_presentation_overrides(),
            "override_mode": get_override_mode()}


# --- free stock API keys (Pexels / Pixabay) ---------------------------------
#
# Only used as a fallback when ComfyUI cannot produce a visual and the
# project's `allow_free_stock` is on — see presentation.stock. A stored key
# wins over the matching environment variable. Never returned or logged in
# full: routes.stock masks all but the last 4 characters.
STOCK_KEYS_KEY = "stock_api_keys"
_STOCK_KEY_NAMES = ("pexels_api_key", "pixabay_api_key")
_STOCK_KEY_ENV_VARS = {"pexels_api_key": "PEXELS_API_KEY", "pixabay_api_key": "PIXABAY_API_KEY"}


def _mask_key(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def get_stock_api_key(name: str) -> Optional[str]:
    """The real key for `name` ('pexels_api_key' | 'pixabay_api_key'): the
    stored value if set, else the matching environment variable, else None.
    For internal use by presentation.stock only — never log or return this."""
    if name not in _STOCK_KEY_NAMES:
        return None
    stored = (app_settings.get(STOCK_KEYS_KEY, {}) or {}).get(name)
    if stored:
        return stored
    return os.environ.get(_STOCK_KEY_ENV_VARS[name]) or None


def set_stock_api_keys(pexels_api_key: Optional[str] = None,
                       pixabay_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Persist whichever of the two keys were actually passed (None = leave
    unchanged; "" clears it). Returns the masked state."""
    current = dict(app_settings.get(STOCK_KEYS_KEY, {}) or {})
    if pexels_api_key is not None:
        current["pexels_api_key"] = pexels_api_key
    if pixabay_api_key is not None:
        current["pixabay_api_key"] = pixabay_api_key
    app_settings.set(STOCK_KEYS_KEY, current)
    return get_stock_api_keys_masked()


def get_stock_api_keys_masked() -> Dict[str, Any]:
    """GET-safe view: whether each key is configured (stored or via
    environment variable), and a masked-all-but-last-4 preview of it."""
    result: Dict[str, Any] = {}
    for name in _STOCK_KEY_NAMES:
        key = get_stock_api_key(name)
        result[name] = _mask_key(key)
        result[f"{name}_set"] = bool(key)
    return result


def apply_presentation_overrides(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`incoming settings, then overrides on top` — the merge every path that
    builds a presentation job's PresentationSettings from a raw dict must run.

    Returns `(merged, override_fields)`: `merged` is `settings` with whichever
    fields the override applies laid on top; `override_fields` is exactly the
    fields that were actually applied (empty when `override_mode` is "off"),
    for `presentation.models.settings_sources_for`.
    """
    mode = get_override_mode()
    if mode == "off":
        return dict(settings), {}
    overrides = get_presentation_overrides()
    if not overrides:
        return dict(settings), {}
    merged = dict(settings)
    merged.update(overrides)
    return merged, dict(overrides)
