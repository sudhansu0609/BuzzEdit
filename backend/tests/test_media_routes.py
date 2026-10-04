"""Route-level tests for the media library and timeline drops.

The interesting behaviour here lives in the handlers rather than in a pure
function — which track a dropped file lands on, whether a timeline gets created,
whether audio is paired with picture — so these drive the real ASGI app with the
project store pointed at a temp directory.
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# NOTE: import the *unprefixed* modules. conftest puts backend/ on sys.path and
# main.py does `from routes import projects`, so `routes.projects` and
# `backend.routes.projects` are two separate module objects holding two separate
# ProjectStore instances. Patching the `backend.`-prefixed copy leaves the running
# app pointed at the real data directory — which is how an earlier version of this
# fixture wrote test projects into data/projects.
from store import media_pool
from store.project_store import ProjectStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    from main import app
    from routes import projects as projects_route
    from routes import timeline as timeline_route
    from store.app_settings import AppSettings

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))

    monkeypatch.setattr(projects_route, "project_store", store)
    monkeypatch.setattr(projects_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(timeline_route, "project_store", store)
    monkeypatch.setattr(timeline_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(media_pool, "PROJECTS_DIR", projects_dir)

    # Creating a project stamps last_project_id; bind that to a throwaway file so
    # the real app_settings.json is never touched. AppSettings is a singleton, so
    # build the stand-in without going through __new__/__init__.
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
    """Real (tiny) media files — ffprobe has to be able to read them."""
    from config import FFMPEG_BIN
    root = tmp_path_factory.mktemp("media")
    (root / "music").mkdir()

    def run(args):
        result = subprocess.run([FFMPEG_BIN, "-y", "-v", "error", *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            pytest.skip("ffmpeg unavailable")

    talking = root / "talking.mp4"
    run(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(talking)])
    silent = root / "silent.mp4"
    run(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
         "-pix_fmt", "yuv420p", str(silent)])
    song = root / "music" / "song.mp3"
    run(["-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-c:a", "libmp3lame", str(song)])
    still = root / "still.png"
    run(["-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1", "-frames:v", "1", str(still)])
    (root / "notes.txt").write_text("ignore me")

    return {"root": root, "talking": talking, "silent": silent, "song": song, "still": still}


def _new_project(client, path) -> str:
    res = client.post("/api/projects/import_path", json={"path": str(path)})
    assert res.status_code == 200, res.text
    return res.json()["id"]


# --- library ---------------------------------------------------------------

def test_new_project_starts_with_its_footage_in_the_library(client, samples):
    pid = _new_project(client, samples["talking"])
    body = client.get(f"/api/projects/{pid}/media").json()
    assert body["total"] == 1
    assert body["media"][0]["name"] == "talking.mp4"
    assert body["media"][0]["kind"] == "video"


def test_folder_import_adds_every_supported_file(client, samples):
    pid = _new_project(client, samples["talking"])
    res = client.post(f"/api/projects/{pid}/media/import",
                      json={"paths": [str(samples["root"])]})
    assert res.status_code == 200, res.text

    body = client.get(f"/api/projects/{pid}/media").json()
    # talking(seeded) + silent + song + still; the .txt is not media, and the
    # re-encountered talking.mp4 is a duplicate.
    assert body["counts"] == {"video": 2, "audio": 1, "image": 1}
    assert not any(m["name"].endswith(".txt") for m in body["media"])


def test_import_rejects_a_path_with_no_media(client, samples, tmp_path):
    pid = _new_project(client, samples["talking"])
    empty = tmp_path / "empty"
    empty.mkdir()
    res = client.post(f"/api/projects/{pid}/media/import", json={"paths": [str(empty)]})
    assert res.status_code == 400


def test_library_filters_by_kind_and_name(client, samples):
    pid = _new_project(client, samples["talking"])
    client.post(f"/api/projects/{pid}/media/import", json={"paths": [str(samples["root"])]})

    assert len(client.get(f"/api/projects/{pid}/media?kind=audio").json()["media"]) == 1
    assert len(client.get(f"/api/projects/{pid}/media?q=silent").json()["media"]) == 1
    assert len(client.get(f"/api/projects/{pid}/media?q=nothing").json()["media"]) == 0


def test_thumbnail_is_served_even_when_the_entry_never_recorded_one(client, samples):
    pid = _new_project(client, samples["talking"])
    media_id = client.get(f"/api/projects/{pid}/media").json()["media"][0]["id"]

    res = client.get(f"/api/projects/{pid}/media/{media_id}/thumb")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 500


def test_audio_entry_exposes_a_waveform(client, samples):
    pid = _new_project(client, samples["talking"])
    client.post(f"/api/projects/{pid}/media/import", json={"paths": [str(samples["song"])]})
    song = next(m for m in client.get(f"/api/projects/{pid}/media").json()["media"]
                if m["kind"] == "audio")

    peaks = client.get(f"/api/projects/{pid}/media/{song['id']}/waveform").json()["peaks"]
    assert len(peaks) > 10


def test_rename_only_changes_the_label(client, samples):
    pid = _new_project(client, samples["talking"])
    entry = client.get(f"/api/projects/{pid}/media").json()["media"][0]
    res = client.patch(f"/api/projects/{pid}/media/{entry['id']}", json={"name": "Interview"})
    assert res.status_code == 200
    updated = res.json()["media"][0]
    assert updated["name"] == "Interview"
    assert updated["path"] == entry["path"]
    assert Path(entry["path"]).exists()


def test_removing_a_linked_entry_never_deletes_the_users_file(client, samples):
    pid = _new_project(client, samples["talking"])
    client.post(f"/api/projects/{pid}/media/import", json={"paths": [str(samples["song"])]})
    song = next(m for m in client.get(f"/api/projects/{pid}/media").json()["media"]
                if m["kind"] == "audio")
    assert song["linked"] is True

    client.delete(f"/api/projects/{pid}/media/{song['id']}?delete_file=true")
    assert samples["song"].exists(), "a linked source file was deleted"
    assert not any(m["kind"] == "audio"
                   for m in client.get(f"/api/projects/{pid}/media").json()["media"])


def test_missing_file_is_flagged_and_relinkable(client, samples, tmp_path):
    pid = _new_project(client, samples["talking"])
    moving = tmp_path / "temp_clip.mp4"
    moving.write_bytes(samples["silent"].read_bytes())
    client.post(f"/api/projects/{pid}/media/import", json={"paths": [str(moving)]})

    entry = next(m for m in client.get(f"/api/projects/{pid}/media").json()["media"]
                 if m["name"] == "temp_clip.mp4")
    moved_to = tmp_path / "renamed_clip.mp4"
    moving.rename(moved_to)

    entry = next(m for m in client.get(f"/api/projects/{pid}/media").json()["media"]
                 if m["id"] == entry["id"])
    assert entry["missing"] is True

    res = client.post(f"/api/projects/{pid}/media/{entry['id']}/relink",
                      json={"path": str(moved_to)})
    restored = next(m for m in res.json()["media"] if m["id"] == entry["id"])
    assert restored["missing"] is False


# --- dragging onto the timeline -------------------------------------------

def test_first_drop_creates_a_timeline_and_becomes_the_program(client, samples):
    pid = _new_project(client, samples["silent"])
    assert client.get(f"/api/timeline/{pid}").status_code == 404   # nothing yet

    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["silent"]), "timeline_start_frame": 0})
    assert res.status_code == 200, res.text
    assert res.json()["item"]["track"] == "V1"


def test_dropping_a_video_with_sound_pairs_it_onto_a1(client, samples):
    pid = _new_project(client, samples["talking"])
    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["talking"]), "timeline_start_frame": 0})
    items = res.json()["timeline"]["items"]
    assert {i["track"] for i in items} == {"V1", "A1"}
    video = next(i for i in items if i["track"] == "V1")
    audio = next(i for i in items if i["track"] == "A1")
    assert audio["source_id"] == video["source_id"]
    assert audio["timeline_start_frame"] == video["timeline_start_frame"]


def test_the_first_drop_starts_the_timeline_with_sound_linked(client, samples):
    pid = _new_project(client, samples["talking"])
    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["talking"]), "timeline_start_frame": 90})
    items = res.json()["timeline"]["items"]
    video = next(i for i in items if i["track"] == "V1")
    audio = next(i for i in items if i["track"] == "A1")
    assert video["timeline_start_frame"] == audio["timeline_start_frame"] == 0
    assert video["link_id"] and video["link_id"] == audio["link_id"]
    assert set(res.json()["item_ids"]) == {video["id"], audio["id"]}


def test_an_overlay_video_brings_its_sound_linked_on_an_audio_lane(client, samples):
    pid = _new_project(client, samples["talking"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["talking"])})
    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["talking"]), "track": "V2",
                            "timeline_start_frame": 15})
    items = res.json()["timeline"]["items"]
    video = next(i for i in items if i["track"] == "V2")
    audio = next(i for i in items if i["track"] == "A2")
    assert video["timeline_start_frame"] == audio["timeline_start_frame"] == 15
    assert video["link_id"] == audio["link_id"]

    moved = client.post(f"/api/timeline/{pid}/clip/move",
                        json={"item_id": video["id"], "timeline_start_frame": 30}).json()
    audio_after = next(i for i in moved["timeline"]["items"] if i["id"] == audio["id"])
    assert audio_after["timeline_start_frame"] == 30


def test_replace_restores_a_snapshot_with_a_newer_revision(client, samples):
    pid = _new_project(client, samples["silent"])
    first = client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    snapshot = first.json()["timeline"]
    second = client.post(f"/api/timeline/{pid}/add_media",
                         json={"path": str(samples["still"]), "timeline_start_frame": 30})
    later_revision = second.json()["timeline"]["revision"]

    res = client.post(f"/api/timeline/{pid}/replace", json={"timeline": snapshot})
    assert res.status_code == 200, res.text
    restored = res.json()["timeline"]
    assert len(restored["items"]) == len(snapshot["items"])
    assert restored["revision"] > later_revision


def test_a_silent_video_gets_no_audio_clip(client, samples):
    pid = _new_project(client, samples["silent"])
    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["silent"]), "timeline_start_frame": 0})
    assert {i["track"] for i in res.json()["timeline"]["items"]} == {"V1"}


def test_later_drops_stack_onto_overlay_tracks(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    res = client.post(f"/api/timeline/{pid}/add_media",
                      json={"path": str(samples["still"]), "timeline_start_frame": 30})
    assert res.json()["item"]["track"] == "V2"


def test_music_becomes_the_program_audio_when_a1_is_free(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    res = client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["song"])})
    assert res.json()["item"]["track"] == "A1"


def test_dropping_the_same_file_twice_reuses_its_source(client, samples):
    pid = _new_project(client, samples["silent"])
    first = client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    before = len(first.json()["timeline"]["sources"])
    second = client.post(f"/api/timeline/{pid}/add_media",
                         json={"path": str(samples["silent"]), "timeline_start_frame": 90})
    after = second.json()["timeline"]["sources"]
    assert len(after) == before
    assert second.json()["item"]["source_id"] == first.json()["item"]["source_id"]


def test_timeline_links_rather_than_copying_the_file(client, samples, tmp_path):
    pid = _new_project(client, samples["silent"])
    res = client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    sources = res.json()["timeline"]["sources"].values()
    assert any(Path(s["path"]) == samples["silent"] for s in sources)


def test_text_can_be_added_before_any_transcription(client, samples):
    pid = _new_project(client, samples["silent"])
    res = client.post(f"/api/timeline/{pid}/text/add",
                      json={"content": "Hello", "duration_seconds": 2})
    assert res.status_code == 200, res.text
    assert res.json()["timeline"]["items"][0]["kind"] == "text"


def test_dropping_a_file_that_is_not_there_is_rejected(client, samples, tmp_path):
    pid = _new_project(client, samples["silent"])
    res = client.post(f"/api/timeline/{pid}/add_media", json={"path": str(tmp_path / "gone.mp4")})
    assert res.status_code == 400


# --- tracks (layers) -------------------------------------------------------

def test_track_flags_persist_on_the_saved_timeline(client, samples):
    pid = _new_project(client, samples["talking"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["talking"])})

    res = client.post(f"/api/timeline/{pid}/track/flags",
                      json={"track": "A1", "muted": True, "locked": True})
    assert res.status_code == 200, res.text
    assert res.json()["timeline"]["tracks"]["A1"] == {"hidden": False, "locked": True, "muted": True}

    # re-read from disk: the flags are part of the project, not just the response
    reloaded = client.get(f"/api/timeline/{pid}").json()["tracks"]
    assert reloaded["A1"]["muted"] is True


def test_a_locked_track_rejects_clip_edits_through_the_api(client, samples):
    pid = _new_project(client, samples["silent"])
    drop = client.post(f"/api/timeline/{pid}/add_media",
                       json={"path": str(samples["still"]), "track": "V2"})
    item_id = drop.json()["item"]["id"]

    client.post(f"/api/timeline/{pid}/track/flags", json={"track": "V2", "locked": True})
    res = client.post(f"/api/timeline/{pid}/clip/delete", json={"item_id": item_id})
    assert res.status_code == 400
    assert "locked" in res.json()["detail"].lower()

    client.post(f"/api/timeline/{pid}/track/flags", json={"track": "V2", "locked": False})
    assert client.post(f"/api/timeline/{pid}/clip/delete", json={"item_id": item_id}).status_code == 200


def test_deleting_a_track_takes_its_clips_with_it(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    client.post(f"/api/timeline/{pid}/add_media",
                json={"path": str(samples["still"]), "timeline_start_frame": 30})

    res = client.post(f"/api/timeline/{pid}/track/delete", json={"track": "V2"})
    assert res.status_code == 200, res.text
    assert {i["track"] for i in res.json()["timeline"]["items"]} == {"V1"}


def test_a_reserved_empty_track_survives_a_reload(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})

    name = client.post(f"/api/timeline/{pid}/add_track", json={"kind": "A"}).json()["track"]
    assert name == "A1"
    assert name in client.get(f"/api/timeline/{pid}").json()["extra_tracks"]


# --- adjustment layers -----------------------------------------------------

def test_an_adjustment_layer_can_be_added_and_graded_through_the_api(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})

    res = client.post(f"/api/timeline/{pid}/adjustment/add",
                      json={"duration_seconds": 2.0, "preset": "moody"})
    assert res.status_code == 200, res.text
    adjustment = next(i for i in res.json()["timeline"]["items"] if i["kind"] == "adjustment")
    assert adjustment["track"] == "V2"
    assert adjustment["source_id"] is None
    assert adjustment["color"]["preset"] == "moody"

    res = client.post(f"/api/timeline/{pid}/effects/add",
                      json={"preset": "rain", "item_id": adjustment["id"]})
    assert res.status_code == 200, res.text
    timeline = res.json()["timeline"]
    assert timeline["effects"] == []            # the programme is untouched
    updated = next(i for i in timeline["items"] if i["id"] == adjustment["id"])
    assert len(updated["atmosphere"]) == 1


def test_an_adjustment_layer_needs_something_underneath_it(client, samples):
    pid = _new_project(client, samples["silent"])
    client.post(f"/api/timeline/{pid}/add_media", json={"path": str(samples["silent"])})
    res = client.post(f"/api/timeline/{pid}/adjustment/add", json={"track": "V1"})
    assert res.status_code == 400
