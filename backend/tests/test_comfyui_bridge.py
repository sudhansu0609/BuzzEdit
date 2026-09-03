"""Tests for reading ComfyUI job history.

The history's `outputs` mixes file entries with flags: a SaveVideo node
publishes its clip under `images` and marks it with `"animated": [true]` — a
list of BOOLEANS. Reading that list as file entries crashed every Wan video
generation with "'bool' object has no attribute 'get'", which tripped the
consecutive-failure breaker and cost the night's remaining assets.
"""

from comfyui_bridge.client import ComfyUIClient


def test_video_history_flags_are_not_read_as_files(monkeypatch):
    client = ComfyUIClient()
    history = {"p1": {"outputs": {"v_save": {
        "images": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}],
        "animated": [True],
    }}}}
    monkeypatch.setattr(client, "get_history", lambda pid: history)
    monkeypatch.setattr(client, "_resolve_output",
                        lambda entry: "/resolved/" + entry["filename"])

    files = client.wait_for_prompt("p1", timeout=5)

    assert files == ["/resolved/clip.mp4"]
