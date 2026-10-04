"""Tests for `store/glossary_store.py`: per-channel Hinglish spelling
overrides, `_global.json` merged under a channel-specific file."""

import pytest

from backend.store import glossary_store


@pytest.fixture(autouse=True)
def _isolated_glossary_dir(tmp_path, monkeypatch):
    """Point the store at a temp directory so tests never touch the real
    data/glossary/ files."""
    monkeypatch.setattr(glossary_store, "GLOSSARY_DIR", tmp_path)
    yield tmp_path


def test_merge_entries_creates_and_persists_the_channel_file(tmp_path):
    result = glossary_store.merge_entries("life3baje", {"laaiph": "life"})
    assert result == {"laaiph": "life"}
    assert (tmp_path / "life3baje.json").exists()
    assert glossary_store.load_channel("life3baje") == {"laaiph": "life"}


def test_merge_entries_adds_new_keys_and_overwrites_existing_ones():
    glossary_store.merge_entries("life3baje", {"a": "1", "b": "2"})
    result = glossary_store.merge_entries("life3baje", {"b": "2-new", "c": "3"})
    assert result == {"a": "1", "b": "2-new", "c": "3"}


def test_load_merged_combines_global_and_channel_with_channel_winning():
    glossary_store.merge_entries("_global", {"nahi": "nahi", "shared": "global"})
    glossary_store.merge_entries("life3baje", {"laaiph": "life", "shared": "channel"})

    merged = glossary_store.load_merged("life3baje")
    assert merged["nahi"] == "nahi"          # from _global
    assert merged["laaiph"] == "life"        # from the channel file
    assert merged["shared"] == "channel"     # channel wins over global


def test_load_merged_with_no_key_is_global_only():
    glossary_store.merge_entries("_global", {"nahi": "nahi"})
    glossary_store.merge_entries("some_channel", {"laaiph": "life"})
    assert glossary_store.load_merged(None) == {"nahi": "nahi"}


def test_missing_files_return_empty_dicts():
    assert glossary_store.load_channel("never_created") == {}
    assert glossary_store.load_merged("never_created") == {}
