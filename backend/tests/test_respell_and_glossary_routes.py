"""Route-level tests for `POST /api/transcription/{id}/respell` and
`GET/PUT /api/glossary/{key}`.

Respell re-applies the romanizer (and a glossary) to an already-transcribed
project's `word`/`hinglish` fields from the untouched `word_native` Whisper
produced, without re-running Whisper.
"""

import pytest
from fastapi.testclient import TestClient

from store.project_store import ProjectStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    from main import app
    from routes import transcription as transcription_route
    from routes import glossary as glossary_route
    from store import glossary_store

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))
    monkeypatch.setattr(transcription_route, "project_store", store)

    glossary_dir = tmp_path / "glossary"
    monkeypatch.setattr(glossary_store, "GLOSSARY_DIR", glossary_dir)

    with TestClient(app) as test_client:
        yield test_client, store


def _project_with_stale_transcript():
    return {
        "id": "proj1",
        "name": "Test",
        "source_video": "x.mp4",
        "settings": {"language": "hi"},
        "transcript": {
            "language": "hi",
            "duration": 1.0,
            "segments": [{
                "id": 0, "start": 0.0, "end": 1.0,
                "text": "hogaa naheen",
                "text_native": "होगा नहीं",
                "words": [
                    {"word": "hogaa", "word_native": "होगा", "hinglish": "hogaa",
                     "start": 0.0, "end": 0.4},
                    {"word": "naheen", "word_native": "नहीं", "hinglish": "naheen",
                     "start": 0.4, "end": 0.8},
                ],
            }],
        },
    }


def test_respell_updates_word_and_hinglish_but_never_word_native(client):
    test_client, store = client
    store.save_project("proj1", _project_with_stale_transcript())

    resp = test_client.post("/api/transcription/proj1/respell")
    assert resp.status_code == 200
    body = resp.json()
    assert body["words_changed"] == 2

    saved = store.get_project("proj1")
    words = saved["transcript"]["segments"][0]["words"]
    assert words[0]["word"] == "hoga"
    assert words[0]["hinglish"] == "hoga"
    assert words[0]["word_native"] == "होगा"          # untouched
    assert words[1]["word"] == "nahi"
    assert words[1]["word_native"] == "नहीं"           # untouched
    assert saved["transcript"]["segments"][0]["text"] == "hoga nahi"


def test_respell_applies_a_glossary_override(client):
    test_client, store = client
    store.save_project("proj1", _project_with_stale_transcript())

    put_resp = test_client.put("/api/glossary/life3baje",
                                json={"entries": {"होगा": "hoga-custom"}})
    assert put_resp.status_code == 200

    resp = test_client.post("/api/transcription/proj1/respell",
                             json={"glossary": "life3baje"})
    assert resp.status_code == 200

    saved = store.get_project("proj1")
    words = saved["transcript"]["segments"][0]["words"]
    assert words[0]["word"] == "hoga-custom"
    assert words[1]["word"] == "nahi"                  # unaffected key


def test_respell_missing_project_is_404(client):
    test_client, _ = client
    resp = test_client.post("/api/transcription/does-not-exist/respell")
    assert resp.status_code == 404


def test_glossary_get_put_round_trip(client):
    test_client, _ = client
    assert test_client.get("/api/glossary/life3baje").json() == {
        "key": "life3baje", "entries": {}}

    put_resp = test_client.put("/api/glossary/life3baje",
                                json={"entries": {"laaiph": "life"}})
    assert put_resp.json()["entries"] == {"laaiph": "life"}

    merge_resp = test_client.put("/api/glossary/life3baje",
                                  json={"entries": {"staart": "start"}})
    assert merge_resp.json()["entries"] == {"laaiph": "life", "staart": "start"}

    get_resp = test_client.get("/api/glossary/life3baje")
    assert get_resp.json()["entries"] == {"laaiph": "life", "staart": "start"}
