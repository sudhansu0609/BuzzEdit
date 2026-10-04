"""Free stock (Pexels/Pixabay) fallback: presentation.stock, and its wiring
into presentation.assets.generate_assets. All HTTP is mocked -- these never
touch the network.
"""

import json

import pytest

from presentation import assets as assets_stage
from presentation import stock
from presentation import workflows
from presentation.models import Beat, PresentationSettings


def _beat(**kwargs) -> Beat:
    base = dict(id="b1", start_s=5.0, end_s=11.0, topic="a lighthouse at dusk",
               kind="broll_image", priority=0.8, image_prompt="a lighthouse at dusk")
    base.update(kwargs)
    return Beat(**base)


def _photo_hit(source="pexels", width=1920, height=1080, id_=1):
    return {"source": source, "id": id_, "page_url": f"https://{source}.example/p/{id_}",
           "download_url": f"https://{source}.example/download/{id_}.jpg",
           "author": "Jane Doe", "width": width, "height": height,
           "duration": 0.0, "licence": "test licence", "ext": ".jpg"}


def _video_hit(source="pexels", duration=6.0, height=1080, id_=2):
    return {"source": source, "id": id_, "page_url": f"https://{source}.example/v/{id_}",
           "download_url": f"https://{source}.example/download/{id_}.mp4",
           "author": "John Roe", "width": int(height * 16 / 9), "height": height,
           "duration": duration, "licence": "test licence", "ext": ".mp4"}


# --- availability guard ------------------------------------------------------

def test_stock_unavailable_when_not_allowed(monkeypatch):
    monkeypatch.setattr(stock, "_api_key", lambda name: "a-key")
    settings = PresentationSettings(allow_free_stock=False)
    assert stock.stock_available(settings) is False


def test_stock_unavailable_without_a_key(monkeypatch):
    monkeypatch.setattr(stock, "_api_key", lambda name: None)
    settings = PresentationSettings(allow_free_stock=True)
    assert stock.stock_available(settings) is False


def test_stock_available_when_allowed_and_keyed(monkeypatch):
    monkeypatch.setattr(stock, "_api_key", lambda name: "a-key" if name == "pexels_api_key" else None)
    settings = PresentationSettings(allow_free_stock=True)
    assert stock.stock_available(settings) is True


# --- picking -------------------------------------------------------------

def test_pick_best_prefers_the_3_to_10s_clip_and_matching_aspect():
    hits = [
        _video_hit(id_=1, duration=25.0, height=1080),   # way too long
        _video_hit(id_=2, duration=6.0, height=1080),    # in range, 16:9
        _video_hit(id_=3, duration=6.0, height=2000),    # in range but wrong aspect
    ]
    best = stock.pick_best(hits, canvas_aspect=16 / 9, is_video=True)
    assert best["id"] == 2


def test_pick_best_returns_none_for_no_hits():
    assert stock.pick_best([], canvas_aspect=16 / 9, is_video=True) is None


# --- fill_missing: mocked HTTP, no network ------------------------------------

@pytest.mark.asyncio
async def test_fill_missing_downloads_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(stock, "_api_key",
                        lambda name: "pexels-key" if name == "pexels_api_key" else None)
    monkeypatch.setattr(stock, "search_pexels_photos",
                        lambda q, o, k: [_photo_hit(id_=7)])
    monkeypatch.setattr(stock, "_http_download", lambda url: b"fake-image-bytes")

    beats = [_beat(id="b1")]
    failures = [{"beat_id": "b1", "reason": "no output produced"}]
    settings = PresentationSettings(allow_free_stock=True)

    new_assets, still_failed, used = await stock.fill_missing(
        beats, tmp_path, settings, failures, canvas_size=(1920, 1080))

    assert still_failed == []
    assert len(new_assets) == 1
    asset = new_assets[0]
    assert asset.beat_id == "b1"
    assert asset.kind == "image"
    assert asset.source == "pexels"
    assert asset.stock_url == "https://pexels.example/p/7"
    assert asset.stock_author == "Jane Doe"
    assert used == [{"beat_id": "b1", "source": "pexels",
                     "url": "https://pexels.example/p/7", "author": "Jane Doe"}]

    sidecar = list((tmp_path / "assets" / "generated").glob("*.json"))
    assert len(sidecar) == 1
    payload = json.loads(sidecar[0].read_text(encoding="utf-8"))
    assert payload["source"] == "pexels"
    assert payload["author"] == "Jane Doe"


@pytest.mark.asyncio
async def test_fill_missing_falls_back_to_pixabay_when_pexels_has_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(stock, "_api_key", lambda name: "a-key")
    monkeypatch.setattr(stock, "search_pexels_photos", lambda q, o, k: [])
    monkeypatch.setattr(stock, "search_pixabay_images", lambda q, o, k: [_photo_hit(source="pixabay", id_=9)])
    monkeypatch.setattr(stock, "_http_download", lambda url: b"fake-image-bytes")

    beats = [_beat(id="b1")]
    failures = [{"beat_id": "b1", "reason": "no output produced"}]
    settings = PresentationSettings(allow_free_stock=True)

    new_assets, still_failed, used = await stock.fill_missing(beats, tmp_path, settings, failures)

    assert len(new_assets) == 1
    assert new_assets[0].source == "pixabay"
    assert still_failed == []


@pytest.mark.asyncio
async def test_fill_missing_is_a_noop_without_allow_free_stock(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("HTTP must never be reached when disabled")
    monkeypatch.setattr(stock, "_http_get_json", _boom)
    monkeypatch.setattr(stock, "_http_download", _boom)
    monkeypatch.setattr(stock, "_api_key", lambda name: "a-key")

    beats = [_beat(id="b1")]
    failures = [{"beat_id": "b1", "reason": "no output produced"}]
    settings = PresentationSettings(allow_free_stock=False)

    new_assets, still_failed, used = await stock.fill_missing(beats, tmp_path, settings, failures)

    assert new_assets == []
    assert still_failed == failures
    assert used == []


@pytest.mark.asyncio
async def test_fill_missing_is_a_noop_without_a_key(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("HTTP must never be reached without a key")
    monkeypatch.setattr(stock, "_http_get_json", _boom)
    monkeypatch.setattr(stock, "_http_download", _boom)
    monkeypatch.setattr(stock, "_api_key", lambda name: None)

    beats = [_beat(id="b1")]
    failures = [{"beat_id": "b1", "reason": "no output produced"}]
    settings = PresentationSettings(allow_free_stock=True)

    new_assets, still_failed, used = await stock.fill_missing(beats, tmp_path, settings, failures)

    assert new_assets == []
    assert still_failed == failures
    assert used == []


@pytest.mark.asyncio
async def test_fill_missing_failure_never_breaks_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(stock, "_api_key", lambda name: "a-key")

    def _search_boom(q, o, k):
        raise ConnectionError("network is down")
    monkeypatch.setattr(stock, "search_pexels_photos", _search_boom)
    monkeypatch.setattr(stock, "search_pixabay_images", _search_boom)

    beats = [_beat(id="b1")]
    failures = [{"beat_id": "b1", "reason": "no output produced"}]
    settings = PresentationSettings(allow_free_stock=True)

    new_assets, still_failed, used = await stock.fill_missing(beats, tmp_path, settings, failures)

    assert new_assets == []
    assert still_failed == failures
    assert used == []


# --- integration with generate_assets: never called when ComfyUI succeeded ---

class _FakeComfyQueue:
    def __init__(self):
        self.client = self

    def is_connected(self, timeout: float = 10.0) -> bool:
        return True

    async def submit_and_wait(self, graph, timeout=0):
        return [str(self._output)]


def _fake_image_workflow():
    return workflows.ResolvedWorkflow("broll_image", "fake.json", {}, {}, "image")


@pytest.mark.asyncio
async def test_stock_fallback_never_runs_when_comfyui_succeeded(tmp_path, monkeypatch):
    import comfyui_bridge

    out = tmp_path / "out.png"
    out.write_bytes(b"not really a png")
    queue = _FakeComfyQueue()
    queue._output = out
    monkeypatch.setattr(comfyui_bridge, "queue_manager", queue)
    monkeypatch.setattr(assets_stage.workflows, "resolve",
                        lambda role: _fake_image_workflow() if role == "broll_image" else None)
    monkeypatch.setattr(assets_stage, "_setting", lambda key, default: default)

    def _boom(*a, **k):
        raise AssertionError("stock fallback must not run when ComfyUI succeeded")
    monkeypatch.setattr(stock, "fill_missing", _boom)

    beats = [_beat(id="b1")]
    settings = PresentationSettings(allow_free_stock=True)

    generated, failures = await assets_stage.generate_assets(beats, tmp_path, settings)

    assert failures == []
    assert len(generated) == 1
    assert generated[0].source == "generated"
