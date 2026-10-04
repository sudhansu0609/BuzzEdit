"""A retry after a failed render reuses what the first run made: the shot plan
(so the picture prompts, and with them the image cache keys, stay the same),
and the cached pictures even while ComfyUI is down."""
import asyncio
import types

from presentation import assets as assets_stage
from presentation import director
from presentation.models import Asset, Beat, PresentationSettings, ShotPlan, Topic
from presentation.structure import Hook


def _plan():
    return ShotPlan(beats=[Beat(id="b1", start_s=1, end_s=4, image_prompt="a dark corridor")],
                    topics=[Topic(start_s=0, end_s=10, title="The Corridor")]
                    if "title" in Topic.model_fields else [],
                    genre="horror")


def test_the_shot_plan_round_trips_for_the_same_inputs(tmp_path):
    key = director._shotplan_key(None, PresentationSettings(), "horror", None, [])
    hook = Hook(start_s=0.0, end_s=3.0, text="Rule one", source="llm")
    director._save_shotplan(tmp_path, key, _plan(), hook)
    plan, got_hook = director._load_shotplan(tmp_path, key)
    assert [b.image_prompt for b in plan.beats] == ["a dark corridor"]
    assert got_hook == hook
    assert director._load_shotplan(tmp_path, "different") is None
    assert director._load_shotplan(tmp_path / "nowhere", key) is None


def test_the_plan_key_ignores_the_model_but_not_the_inputs():
    base = director._shotplan_key(None, PresentationSettings(), "horror", None, [])
    other_model = PresentationSettings(llm_model="something-else", seed=99)
    assert director._shotplan_key(None, other_model, "horror", None, []) == base
    assert director._shotplan_key(None, PresentationSettings(), "comedy", None, []) != base


def test_an_offline_comfyui_still_uses_cached_pictures(tmp_path, monkeypatch):
    from comfyui_bridge import queue_manager

    monkeypatch.setattr(queue_manager.client, "is_connected", lambda *a, **k: False)
    monkeypatch.setattr(assets_stage.workflows, "resolve",
                        lambda role: types.SimpleNamespace(file="zimage1.json", output_kind="image"))

    async def fake_generate(beat, resolved, directory, qm, seed_base, canvas_size=None,
                            start_image=None, cache_only=False, face=None):
        assert cache_only, "must not try to generate while ComfyUI is down"
        if beat.id == "b1":
            return Asset(beat_id="", kind="image", path=str(tmp_path / "b1.png"), cache_hit=True)
        return None

    monkeypatch.setattr(assets_stage, "_generate_one", fake_generate)
    beats = [Beat(id="b1", start_s=1, end_s=4, image_prompt="x"),
             Beat(id="b2", start_s=5, end_s=8, image_prompt="y")]
    made, failures = asyncio.run(assets_stage.generate_assets(
        beats, tmp_path, PresentationSettings(allow_free_stock=False)
        if "allow_free_stock" in PresentationSettings.model_fields else PresentationSettings()))
    assert [a.beat_id for a in made] == ["b1"]
    assert [f["beat_id"] for f in failures] == ["b2"]
