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


def test_saveaudio_history_key_is_read(monkeypatch):
    """SaveAudio (the text-to-audio "audio" role) publishes its result under
    an `audio` key in history — the same {filename, subfolder, type} shape as
    `images`, just its own key. A finished Stable Audio Open job reported
    "produced no readable outputs" until this key was added to OUTPUT_KEYS,
    even though the file was right there in ComfyUI's own history."""
    client = ComfyUIClient()
    history = {"p1": {"outputs": {"8": {
        "audio": [{"filename": "buzzedit_00001.flac", "subfolder": "", "type": "output"}],
    }}}}
    monkeypatch.setattr(client, "get_history", lambda pid: history)
    monkeypatch.setattr(client, "_resolve_output",
                        lambda entry: "/resolved/" + entry["filename"])

    files = client.wait_for_prompt("p1", timeout=5)

    assert files == ["/resolved/buzzedit_00001.flac"]


def test_queue_manager_fails_fast_once_comfyui_is_known_offline(monkeypatch):
    """After one full wait finds ComfyUI gone, later submissions probe once
    instead of each sitting out their own wait; it recovers when it is back."""
    import asyncio
    import importlib
    qm = importlib.import_module("comfyui_bridge.queue_manager")

    class Fake:
        server_url = "http://127.0.0.1:8188"
        up = False
        probes = 0

        def is_connected(self, timeout=10.0):
            Fake.probes += 1
            return Fake.up

    monkeypatch.setattr(qm, "OFFLINE_WAIT_S", 0)
    m = qm.ComfyUIQueueManager(client=Fake())

    async def check():
        return await m._wait_until_up()

    assert asyncio.run(check()) is False and m._offline
    Fake.probes = 0
    assert asyncio.run(check()) is False
    assert Fake.probes == 1
    Fake.up = True
    assert asyncio.run(check()) is True and not m._offline
