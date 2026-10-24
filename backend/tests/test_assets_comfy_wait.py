"""`generate_assets` waits for ComfyUI instead of giving up on it the moment
it is offline, and fails the job outright (ComfyUIUnavailableError) rather
than render with next to no B-roll -- the 2026-10-20 incident: ComfyUI was
down for 8 days and the pass "degraded" once at the start and never looked
again, rendering 263 planned beats with zero B-roll."""
import asyncio
from types import SimpleNamespace

import pytest

from comfyui_bridge.errors import ComfyUIUnavailableError
from presentation import assets as assets_mod
from presentation.models import Asset, Beat, PresentationSettings


def _beat(beat_id: str, **kwargs) -> Beat:
    base = dict(id=beat_id, start_s=0.0, end_s=4.0, topic="a topic", kind="broll_image",
                priority=0.8, image_prompt="a photograph of something")
    base.update(kwargs)
    return Beat(**base)


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    """A poll every few milliseconds instead of every 15s, so a test that
    waits out the whole budget still finishes almost instantly."""
    monkeypatch.setattr(assets_mod, "COMFY_OFFLINE_POLL_S", 0.01)


@pytest.fixture(autouse=True)
def _resolvable_image_workflow(monkeypatch):
    """A resolvable image workflow, so generate_assets gets past its own
    "no workflow configured" short-circuit and actually checks ComfyUI."""
    def fake_resolve(role):
        return SimpleNamespace(output_kind="image", file="fake.json", fallback=None) if role == "broll_image" else None
    monkeypatch.setattr(assets_mod.workflows, "resolve", fake_resolve)


@pytest.fixture
def set_is_connected(monkeypatch):
    """Install a fake `comfyui_bridge.queue_manager.client.is_connected`."""
    import comfyui_bridge

    def _set(fn):
        monkeypatch.setattr(comfyui_bridge.queue_manager.client, "is_connected", fn)
    return _set


def _short_wait_budget(monkeypatch, minutes: float):
    monkeypatch.setattr(
        assets_mod, "_setting",
        lambda key, default: minutes if key == "comfy_offline_wait_minutes" else default)


def test_comfyui_comes_back_mid_wait_and_generation_proceeds(monkeypatch, tmp_path, set_is_connected):
    calls = {"n": 0}

    def is_connected(timeout=10.0):
        calls["n"] += 1
        return calls["n"] > 2  # offline for the first two checks, then back

    set_is_connected(is_connected)
    _short_wait_budget(monkeypatch, 60.0)  # plenty -- it comes back well within it

    async def fake_generate_one(beat, resolved, directory, queue_manager, seed_base,
                                canvas_size=None, start_image=None, cache_only=False,
                                face=None, boilerplate=None, used_paths=None,
                                size_override=None):
        return Asset(beat_id=beat.id, kind="image", path=f"{beat.id}.png", width=100, height=100,
                     cache_hit=False)

    monkeypatch.setattr(assets_mod, "_generate_one", fake_generate_one)

    beats = [_beat("b1"), _beat("b2"), _beat("b3")]
    assets, failures = asyncio.run(assets_mod.generate_assets(beats, tmp_path, PresentationSettings()))

    assert failures == []
    assert {a.beat_id for a in assets} == {"b1", "b2", "b3"}
    assert calls["n"] > 2  # it actually waited and rechecked, not just one probe


def test_never_comes_back_and_no_cache_raises(monkeypatch, tmp_path, set_is_connected):
    set_is_connected(lambda timeout=10.0: False)
    _short_wait_budget(monkeypatch, 0.0005)  # ~30ms budget: it runs out fast

    async def fake_generate_one(beat, resolved, directory, queue_manager, seed_base,
                                canvas_size=None, start_image=None, cache_only=False,
                                face=None, boilerplate=None, used_paths=None,
                                size_override=None):
        return None  # nothing on disk either

    monkeypatch.setattr(assets_mod, "_generate_one", fake_generate_one)

    beats = [_beat("b1"), _beat("b2")]
    with pytest.raises(ComfyUIUnavailableError):
        asyncio.run(assets_mod.generate_assets(beats, tmp_path, PresentationSettings()))


def test_never_comes_back_with_enough_cache_returns_cached(monkeypatch, tmp_path, set_is_connected):
    set_is_connected(lambda timeout=10.0: False)
    _short_wait_budget(monkeypatch, 0.0005)

    # 9 of 10 wanted beats already have a cached picture -- exactly the 90%
    # floor -- the tenth has none.
    cached_ids = {f"b{i}" for i in range(1, 10)}

    async def fake_generate_one(beat, resolved, directory, queue_manager, seed_base,
                                canvas_size=None, start_image=None, cache_only=False,
                                face=None, boilerplate=None, used_paths=None,
                                size_override=None):
        if cache_only and beat.id in cached_ids:
            return Asset(beat_id=beat.id, kind="image", path=f"{beat.id}.png", width=100, height=100,
                         cache_hit=True)
        return None

    monkeypatch.setattr(assets_mod, "_generate_one", fake_generate_one)

    beats = [_beat(f"b{i}") for i in range(1, 11)]
    assets, failures = asyncio.run(assets_mod.generate_assets(beats, tmp_path, PresentationSettings()))

    assert {a.beat_id for a in assets} == cached_ids
    assert all(a.cache_hit for a in assets)
    assert [f["beat_id"] for f in failures] == ["b10"]
