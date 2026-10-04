"""Route-level tests for `GET /api/projects/{id}/kept_words`."""

import pytest
from fastapi.testclient import TestClient

from store.project_store import ProjectStore
from timeline import build_timeline_from_transcript


@pytest.fixture
def client(tmp_path, monkeypatch):
    from main import app
    from routes import kept_words as kept_words_route

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))
    monkeypatch.setattr(kept_words_route, "project_store", store)

    with TestClient(app) as test_client:
        yield test_client, store


def _four_words():
    return [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "um", "start": 1.0, "end": 1.5, "disfluency": True},
        {"word": "welcome", "start": 1.5, "end": 2.0},
    ]


def test_kept_words_returns_only_enabled_in_programme_order(client):
    test_client, store = client
    tl = build_timeline_from_transcript(
        source_path="video.mp4",
        duration_seconds=5.0,
        transcript_words=_four_words(),
        fps_num=30,
        fps_den=1,
        pause_padding_seconds=0.0,
    )
    store.save_project("proj1", {"id": "proj1", "name": "T", "timeline": tl.model_dump()})

    resp = test_client.get("/api/projects/proj1/kept_words")
    assert resp.status_code == 200
    words = resp.json()["words"]

    assert [w["word"] for w in words] == ["Hello", "world", "welcome"]
    # Source time preserved as-is.
    assert words[0]["start"] == pytest.approx(0.0)
    assert words[2]["start"] == pytest.approx(1.5)
    # Programme time ripples past the cut "um": welcome starts at 1.0s on the
    # programme timeline even though it was spoken at 1.5s in the source.
    assert words[2]["tl_start"] == pytest.approx(1.0)


def test_kept_words_no_timeline_returns_empty(client):
    test_client, store = client
    store.save_project("proj2", {"id": "proj2", "name": "T"})

    resp = test_client.get("/api/projects/proj2/kept_words")
    assert resp.status_code == 200
    assert resp.json() == {"words": []}


def test_kept_words_missing_project_404(client):
    test_client, store = client
    resp = test_client.get("/api/projects/nope/kept_words")
    assert resp.status_code == 404
