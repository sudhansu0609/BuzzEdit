"""store/app_settings.py: a setting written on disk by another process is seen
by the running backend and survives its next save (2026-10-21: the still
workflow setting written by a tool was erased by the old cache)."""
import json
import os
import time

from store.app_settings import AppSettings


def test_external_edits_are_seen_and_kept(tmp_path, monkeypatch):
    settings = AppSettings.__new__(AppSettings)
    settings._initialized = True
    settings.settings_dir = tmp_path
    settings.settings_file = tmp_path / "app_settings.json"
    settings._cache = {"a": 1}
    settings._save()
    assert settings.get("a") == 1

    # Someone else writes the file (a newer mtime).
    time.sleep(0.02)
    data = json.loads(settings.settings_file.read_text(encoding="utf-8"))
    data["workflow_broll_image"] = "zimage1_fast.json"
    settings.settings_file.write_text(json.dumps(data), encoding="utf-8")
    future = time.time() + 5
    os.utime(settings.settings_file, (future, future))
    assert settings.get("workflow_broll_image") == "zimage1_fast.json"

    # Our own save keeps it instead of writing the stale cache back.
    settings.set("b", 2)
    on_disk = json.loads(settings.settings_file.read_text(encoding="utf-8"))
    assert on_disk == {"a": 1, "workflow_broll_image": "zimage1_fast.json", "b": 2}
