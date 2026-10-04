"""Importing a recording: a hard link on the same drive (no second multi-GB
copy), a copy otherwise, and a clear refusal -- with nothing half-written left
behind -- when the drive cannot hold the copy."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.routes import projects as routes


def test_same_drive_recording_is_hard_linked_not_copied(tmp_path):
    src = tmp_path / "take.mp4"
    src.write_bytes(b"video" * 100)
    dest = tmp_path / "proj_source.mp4"
    assert routes.place_source(src, dest) == "linked"
    assert os.path.samefile(src, dest)
    # Removing the project's name for it leaves the original recording.
    dest.unlink()
    assert src.read_bytes() == b"video" * 100


def test_without_hard_links_it_copies(tmp_path, monkeypatch):
    src = tmp_path / "take.mp4"
    src.write_bytes(b"video" * 100)
    dest = tmp_path / "proj_source.mp4"

    def no_links(self, target):
        raise OSError(17, "cross-device link")

    monkeypatch.setattr(Path, "hardlink_to", no_links)
    assert routes.place_source(src, dest) == "copied"
    assert dest.read_bytes() == src.read_bytes() and not os.path.samefile(src, dest)


def test_a_full_drive_is_refused_up_front_and_leaves_nothing(tmp_path, monkeypatch):
    src = tmp_path / "take.mp4"
    src.write_bytes(b"video" * 100)
    dest = tmp_path / "proj_source.mp4"
    monkeypatch.setattr(Path, "hardlink_to", lambda self, target: (_ for _ in ()).throw(OSError(18, "x")))
    monkeypatch.setattr(routes.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024))
    with pytest.raises(OSError) as info:
        routes.place_source(src, dest)
    assert routes._disk_full(info.value) and "Not enough space" in str(info.value)
    assert not dest.exists()


def test_a_copy_that_dies_midway_removes_the_partial_file(tmp_path, monkeypatch):
    src = tmp_path / "take.mp4"
    src.write_bytes(b"video" * 100)
    dest = tmp_path / "proj_source.mp4"
    monkeypatch.setattr(Path, "hardlink_to", lambda self, target: (_ for _ in ()).throw(OSError(18, "x")))

    def dies(a, b):
        Path(b).write_bytes(b"half")
        err = OSError(28, "There is not enough space on the disk")
        err.winerror = 112
        raise err

    monkeypatch.setattr(routes.shutil, "copy", dies)
    with pytest.raises(OSError) as info:
        routes.place_source(src, dest)
    assert routes._disk_full(info.value)
    assert not dest.exists()
