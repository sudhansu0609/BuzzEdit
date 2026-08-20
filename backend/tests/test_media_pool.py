import shutil
from pathlib import Path

import pytest

from backend.store import media_pool


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A media_pool bound to a throwaway projects directory."""
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(media_pool, "PROJECTS_DIR", projects)
    return projects


def _make_files(root: Path, names) -> list:
    made = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really media, but it has an extension")
        made.append(path)
    return made


# --- classification & folder walking --------------------------------------

def test_media_kind_by_extension(tmp_path):
    assert media_pool.media_kind(Path("a.MP4")) == "video"
    assert media_pool.media_kind(Path("song.Mp3")) == "audio"
    assert media_pool.media_kind(Path("shot.PNG")) == "image"
    assert media_pool.media_kind(Path("clip.mts")) == "video"


def test_non_media_files_are_ignored(tmp_path):
    _make_files(tmp_path, ["notes.txt", "project.json", "clip.mp4"])
    found = media_pool.expand_paths([str(tmp_path)])
    assert [p.name for p in found] == ["clip.mp4"]


def test_folder_import_walks_subdirectories(tmp_path):
    _make_files(tmp_path, ["a.mp4", "music/b.mp3", "stills/deep/c.png"])
    found = media_pool.expand_paths([str(tmp_path)])
    assert {p.name for p in found} == {"a.mp4", "b.mp3", "c.png"}


def test_folder_walk_stops_at_the_depth_limit(tmp_path):
    deep = "/".join(["lvl"] * (media_pool.MAX_FOLDER_DEPTH + 2)) + "/buried.mp4"
    _make_files(tmp_path, ["top.mp4", deep])
    found = media_pool.expand_paths([str(tmp_path)])
    assert {p.name for p in found} == {"top.mp4"}


def test_expand_paths_dedupes_a_file_listed_twice(tmp_path):
    files = _make_files(tmp_path, ["a.mp4"])
    found = media_pool.expand_paths([str(files[0]), str(files[0]), str(tmp_path)])
    assert len(found) == 1


def test_mixed_files_and_folders_import_together(tmp_path):
    _make_files(tmp_path, ["folder/a.mp4", "loose.mp3"])
    found = media_pool.expand_paths([str(tmp_path / "folder"), str(tmp_path / "loose.mp3")])
    assert {p.name for p in found} == {"a.mp4", "loose.mp3"}


# --- importing ------------------------------------------------------------

def _fake_probe(monkeypatch, duration=12.0):
    monkeypatch.setattr(media_pool, "probe", lambda path, kind: {
        "duration": 5.0 if kind == "image" else duration,
        "width": 1920, "height": 1080, "has_audio": kind != "image", "fps": 30.0,
    })


def test_import_links_by_default(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    pool = []
    added, skipped = media_pool.import_paths("proj", pool, [str(source)])

    assert len(added) == 1 and not skipped
    entry = added[0]
    assert entry["linked"] is True
    assert entry["path"] == str(source)          # points at the original
    assert not list(library.iterdir())           # nothing copied into the project
    assert pool == added


def test_import_with_copy_stores_a_project_owned_duplicate(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    added, _ = media_pool.import_paths("proj", [], [str(source)], copy=True)

    entry = added[0]
    assert entry["linked"] is False
    assert Path(entry["path"]).parent == library
    assert Path(entry["path"]).exists()
    assert entry["source_path"] == str(source)


def test_reimporting_the_same_file_is_skipped(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    pool = []
    media_pool.import_paths("proj", pool, [str(source)])
    added, skipped = media_pool.import_paths("proj", pool, [str(source)])

    assert added == []
    assert len(pool) == 1
    assert "already in library" in skipped[0]


def test_one_unreadable_file_does_not_sink_the_batch(tmp_path, library, monkeypatch):
    good, bad = _make_files(tmp_path, ["good.mp4", "bad.mp4"])

    def probe(path, kind):
        duration = 0.0 if Path(path).name == "bad.mp4" else 9.0
        return {"duration": duration, "width": 1920, "height": 1080,
                "has_audio": True, "fps": 30.0}
    monkeypatch.setattr(media_pool, "probe", probe)

    added, skipped = media_pool.import_paths("proj", [], [str(good), str(bad)])
    assert [e["name"] for e in added] == ["good.mp4"]
    assert any("unreadable" in s for s in skipped)


def test_images_import_even_though_they_have_no_duration(tmp_path, library, monkeypatch):
    monkeypatch.setattr(media_pool, "probe", lambda path, kind: {
        "duration": 0.0, "width": 800, "height": 600, "has_audio": False, "fps": 0.0})
    still = _make_files(tmp_path, ["photo.png"])[0]
    added, skipped = media_pool.import_paths("proj", [], [str(still)])
    assert len(added) == 1 and not skipped
    assert added[0]["kind"] == "image"


def test_import_records_size_and_geometry(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch, duration=7.5)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    entry = media_pool.import_paths("proj", [], [str(source)])[0][0]
    assert entry["duration"] == 7.5
    assert entry["width"] == 1920 and entry["height"] == 1080
    assert entry["size_bytes"] == source.stat().st_size


# --- listing state --------------------------------------------------------

def test_decorate_flags_a_file_that_disappeared(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    pool = []
    media_pool.import_paths("proj", pool, [str(source)])

    assert media_pool.decorate("proj", pool)[0]["missing"] is False
    source.unlink()
    view = media_pool.decorate("proj", pool)[0]
    assert view["missing"] is True
    assert view["has_thumb"] is False
    assert pool[0].get("missing") is None      # the stored entry is left alone


def test_relink_points_an_entry_at_a_new_file(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    original, replacement = _make_files(tmp_path, ["clip.mp4", "moved/clip.mp4"])
    pool = []
    media_pool.import_paths("proj", pool, [str(original)])
    original.unlink()
    assert media_pool.decorate("proj", pool)[0]["missing"] is True

    updated = media_pool.relink(pool, pool[0]["id"], str(replacement))
    assert updated is not None
    assert media_pool.decorate("proj", pool)[0]["missing"] is False


def test_relink_rejects_a_path_that_does_not_exist(tmp_path, library, monkeypatch):
    _fake_probe(monkeypatch)
    source = _make_files(tmp_path, ["clip.mp4"])[0]
    pool = []
    media_pool.import_paths("proj", pool, [str(source)])
    assert media_pool.relink(pool, pool[0]["id"], str(tmp_path / "nope.mp4")) is None


# --- thumbnails -----------------------------------------------------------

def test_thumbnail_is_addressed_by_convention_not_a_stored_field(library):
    # An entry saved before thumbnails existed carries no `thumb` key; the poster
    # must still be found from the ids alone.
    entry = {"id": "media_abc", "kind": "video", "path": "missing.mp4"}
    poster = library / "proj_media_abc_thumb.jpg"
    poster.write_bytes(b"jpegdata")
    assert media_pool.ensure_thumbnail("proj", entry) == poster


def test_thumbnail_is_skipped_for_audio_and_missing_files(library):
    assert media_pool.ensure_thumbnail("proj", {"id": "m1", "kind": "audio", "path": "x.mp3"}) is None
    assert media_pool.ensure_thumbnail("proj", {"id": "m2", "kind": "video", "path": "gone.mp4"}) is None


def test_real_thumbnail_generation(tmp_path, library):
    """Exercises ffmpeg for real — this is the path that was silently failing."""
    import subprocess
    from config import FFMPEG_BIN
    clip = tmp_path / "clip.mp4"
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=30:duration=3",
         "-pix_fmt", "yuv420p", str(clip)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not clip.exists():
        pytest.skip("ffmpeg unavailable")

    entry = {"id": "media_real", "kind": "video", "path": str(clip)}
    poster = media_pool.ensure_thumbnail("proj", entry)
    assert poster is not None and poster.exists() and poster.stat().st_size > 0


def test_thumbnail_falls_back_for_a_clip_shorter_than_the_seek_point(tmp_path, library):
    import subprocess
    from config import FFMPEG_BIN
    clip = tmp_path / "blink.mp4"
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=160x120:rate=30:duration=0.2",
         "-pix_fmt", "yuv420p", str(clip)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not clip.exists():
        pytest.skip("ffmpeg unavailable")

    poster = media_pool.ensure_thumbnail("proj", {"id": "m_short", "kind": "video", "path": str(clip)})
    assert poster is not None and poster.stat().st_size > 0
