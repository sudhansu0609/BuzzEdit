import tempfile
import os
from pathlib import Path


def _make_settings(tmp_path):
    """Build an AppSettings instance bound to an isolated temp file.

    AppSettings is a process-wide singleton; instantiating it normally (or via a
    subclass) returns the shared real instance and — worse — lets a test rebind
    that instance's settings_file, clobbering the real app_settings.json. We
    bypass __new__/__init__ entirely so the test never touches the singleton."""
    from backend.store.app_settings import AppSettings
    inst = object.__new__(AppSettings)
    inst._initialized = True
    inst.settings_dir = tmp_path
    inst.settings_file = tmp_path / "app_settings.json"
    inst._cache = inst._load()
    return inst


def test_app_settings_persistence(tmp_path):
    """Test that app settings persist and recover correctly."""
    settings = _make_settings(tmp_path)

    settings.set("last_project_id", "proj_abc123")
    settings.set("auto_save_enabled", True)
    assert settings.get("last_project_id") == "proj_abc123"
    assert settings.get("auto_save_enabled") is True

    settings2 = _make_settings(tmp_path)
    assert settings2.get("last_project_id") == "proj_abc123"

    settings.set_last_project("proj_xyz789")
    assert settings.get_last_project() == "proj_xyz789"

    settings.mark_auto_save()
    assert settings.get("last_save_timestamp") is not None


def test_app_settings_update_batch(tmp_path):
    """Test batch update of settings."""
    settings = _make_settings(tmp_path)

    settings.update({
        "last_project_id": "proj_batch",
        "auto_save_interval_seconds": 60,
        "custom_setting": "value"
    })

    assert settings.get("last_project_id") == "proj_batch"
    assert settings.get("auto_save_interval_seconds") == 60
    assert settings.get("custom_setting") == "value"
