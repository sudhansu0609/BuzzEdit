import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
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
