"""store.app_settings free-stock key storage (masked), and routes/stock.py."""

import importlib

import pytest
from fastapi.testclient import TestClient

# store/__init__.py does `from .app_settings import ..., app_settings`, which
# rebinds the `store.app_settings` attribute to the *instance* -- so both
# `from store import app_settings` and `import store.app_settings as x` hand
# back the singleton, not the module. importlib sidesteps that shadowing.
app_settings_mod = importlib.import_module("store.app_settings")


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    store = app_settings_mod.AppSettings()
    monkeypatch.setattr(store, "settings_dir", tmp_path)
    monkeypatch.setattr(store, "settings_file", tmp_path / "app_settings.json")
    monkeypatch.setattr(store, "_cache", {})
    monkeypatch.setattr(app_settings_mod, "app_settings", store)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    return store


def test_masks_all_but_last_4(isolated_settings):
    app_settings_mod.set_stock_api_keys(pexels_api_key="abcdefgh1234")
    masked = app_settings_mod.get_stock_api_keys_masked()
    assert masked["pexels_api_key"] == "********1234"
    assert masked["pexels_api_key_set"] is True
    assert masked["pixabay_api_key"] == ""
    assert masked["pixabay_api_key_set"] is False


def test_real_key_is_never_returned_by_the_masked_view(isolated_settings):
    app_settings_mod.set_stock_api_keys(pexels_api_key="super-secret-key")
    masked = app_settings_mod.get_stock_api_keys_masked()
    assert "super-secret-key" not in masked["pexels_api_key"]
    assert app_settings_mod.get_stock_api_key("pexels_api_key") == "super-secret-key"


def test_env_var_fallback_when_nothing_stored(isolated_settings, monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "env-key-5678")
    assert app_settings_mod.get_stock_api_key("pixabay_api_key") == "env-key-5678"
    masked = app_settings_mod.get_stock_api_keys_masked()
    assert masked["pixabay_api_key_set"] is True
    assert masked["pixabay_api_key"].endswith("5678")


def test_a_stored_key_wins_over_the_env_var(isolated_settings, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "env-key")
    app_settings_mod.set_stock_api_keys(pexels_api_key="stored-key-9999")
    assert app_settings_mod.get_stock_api_key("pexels_api_key") == "stored-key-9999"


def test_setting_one_key_leaves_the_other_untouched(isolated_settings):
    app_settings_mod.set_stock_api_keys(pexels_api_key="pexels-key-1111")
    app_settings_mod.set_stock_api_keys(pixabay_api_key="pixabay-key-2222")
    assert app_settings_mod.get_stock_api_key("pexels_api_key") == "pexels-key-1111"
    assert app_settings_mod.get_stock_api_key("pixabay_api_key") == "pixabay-key-2222"


def test_empty_string_clears_a_key(isolated_settings):
    app_settings_mod.set_stock_api_keys(pexels_api_key="pexels-key-1111")
    app_settings_mod.set_stock_api_keys(pexels_api_key="")
    assert app_settings_mod.get_stock_api_key("pexels_api_key") is None


# --- routes ------------------------------------------------------------------

@pytest.fixture
def client(isolated_settings):
    from main import app
    with TestClient(app) as test_client:
        yield test_client


def test_get_stock_keys_route_never_returns_the_full_key(client):
    app_settings_mod.set_stock_api_keys(pexels_api_key="abcdefgh1234")
    resp = client.get("/api/settings/stock_keys")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pexels_api_key"] == "********1234"
    assert "abcdefgh1234" not in resp.text


def test_put_stock_keys_route_persists_and_masks(client):
    resp = client.put("/api/settings/stock_keys", json={"pixabay_api_key": "zzzzzz4321"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pixabay_api_key"] == "******4321"
    assert app_settings_mod.get_stock_api_key("pixabay_api_key") == "zzzzzz4321"
