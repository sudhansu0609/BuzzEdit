"""Route-level tests for the Voice panel's upload and download routes
(VOICE_CHANGER_PLAN.md, brief S2c): a MediaRecorder blob into the project's
media pool, and any audio file back out as WAV or MP3.
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from store import media_pool
from store.project_store import ProjectStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    from main import app
    from routes import media as media_route
    from routes import projects as projects_route
    from store.app_settings import AppSettings

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))

    monkeypatch.setattr(projects_route, "project_store", store)
    monkeypatch.setattr(projects_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(media_pool, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(media_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(media_route, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(media_route, "_ALLOWED_DOWNLOAD_ROOTS", (projects_dir, voices_dir))

    # Creating a project stamps last_project_id; bind that to a throwaway file
    # so the real app_settings.json is never touched (see test_media_routes.py).
    isolated = object.__new__(AppSettings)
    isolated._initialized = True
    isolated.settings_dir = tmp_path
    isolated.settings_file = tmp_path / "app_settings.json"
    isolated._cache = isolated._load()
    monkeypatch.setattr(projects_route, "app_settings", isolated)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """Real (tiny) media files -- ffprobe/ffmpeg have to be able to read them."""
    from config import FFMPEG_BIN
    root = tmp_path_factory.mktemp("voice_media")

    def run(args, dest):
        result = subprocess.run([FFMPEG_BIN, "-y", "-v", "error", *args, str(dest)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            pytest.skip("ffmpeg unavailable")

    video = root / "seed.mp4"
    run(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1", "-pix_fmt", "yuv420p"], video)

    recording = root / "recording.webm"
    run(["-f", "lavfi", "-i", "sine=frequency=440", "-t", "1", "-c:a", "libopus"], recording)

    empty = root / "empty.webm"
    empty.write_bytes(b"")

    outside_root = tmp_path_factory.mktemp("outside")
    outside = outside_root / "outside.wav"
    run(["-f", "lavfi", "-i", "sine=frequency=330", "-t", "1", "-c:a", "pcm_s16le"], outside)

    return {"video": video, "recording": recording, "empty": empty, "outside": outside}


def _new_project(client, path) -> str:
    res = client.post("/api/projects/import_path", json={"path": str(path)})
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _probe(path):
    from utils.ffmpeg_utils import get_video_info
    return get_video_info(str(path))


# --- upload ------------------------------------------------------------

def test_uploading_a_recorder_blob_transcodes_to_48k_wav(client, samples):
    pid = _new_project(client, samples["video"])
    with open(samples["recording"], "rb") as fh:
        res = client.post(f"/api/projects/{pid}/media/upload",
                          files={"file": ("recording.webm", fh, "audio/webm")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["kind"] == "audio"
    assert 0.8 < body["seconds"] < 1.3

    saved = Path(body["path"])
    assert saved.suffix == ".wav"
    info = _probe(saved)
    assert info["sample_rate"] == 48000
    assert info["video_codec"] == "none"

    library = client.get(f"/api/projects/{pid}/media").json()["media"]
    assert any(m["id"] == body["media_id"] for m in library)


def test_uploading_an_empty_file_is_refused(client, samples):
    pid = _new_project(client, samples["video"])
    with open(samples["empty"], "rb") as fh:
        res = client.post(f"/api/projects/{pid}/media/upload",
                          files={"file": ("empty.webm", fh, "audio/webm")})
    assert res.status_code == 422


# --- download ------------------------------------------------------------

def test_download_as_wav_and_mp3(client, samples):
    pid = _new_project(client, samples["video"])
    with open(samples["recording"], "rb") as fh:
        res = client.post(f"/api/projects/{pid}/media/upload",
                          files={"file": ("recording.webm", fh, "audio/webm")})
    assert res.status_code == 200, res.text
    saved_path = res.json()["path"]

    wav_res = client.get("/api/media/download", params={"path": saved_path, "format": "wav", "name": "take"})
    assert wav_res.status_code == 200, wav_res.text
    assert wav_res.headers["content-type"] == "audio/wav"
    assert wav_res.headers["content-disposition"] == 'attachment; filename="take.wav"'

    mp3_res = client.get("/api/media/download", params={"path": saved_path, "format": "mp3", "name": "take"})
    assert mp3_res.status_code == 200, mp3_res.text
    assert mp3_res.headers["content-type"] == "audio/mpeg"
    assert mp3_res.headers["content-disposition"] == 'attachment; filename="take.mp3"'


def test_download_outside_allowed_trees_is_refused(client, samples):
    res = client.get("/api/media/download", params={"path": str(samples["outside"]), "format": "wav"})
    assert res.status_code == 403
