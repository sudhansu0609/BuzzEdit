"""Clips that differ and run long enough (CLIPS_AND_SOUND_PLAN.md, E1-E5).

Raat3Baje ep2 (2026-10-09) showed a tiger in most clips, the same scene more than
once and near-identical clips back to back, with clips cut to 2-3 s. These guard
the fixes: near matches on the beat's own words and one asset per beat, a
prompt-drift report, variations that are different subjects, the adjacent-twin
drop, a 5 s floor for clips, the classic/hq generation toggle and the
frozen-frame render check.
"""

import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from presentation import assets, shotplan, workflows
from presentation.models import Asset, Beat, PresentationSettings, Program

WORLD = ("A dense remote protected forest in Central India during the monsoon, heavy rain, "
         "dripping leaves, wet red earth, low grey clouds, muted green and ochre palette, "
         "soft overcast daylight, cinematic photograph, shallow depth of field, film grain")
TRACKS = "A ranger checks fresh tyre marks on a muddy jeep track. " + WORLD
TRACKS_REWORDED = "A ranger checking the fresh tyre marks on the muddy jeep track. " + WORLD
TOWER = "An old wooden watchtower rises above the canopy. " + WORLD
OWL = "A lone owl sits on a dead branch at dusk. " + WORLD
VILLAGE = "Village women carry brass water pots along a narrow ridge. " + WORLD


@pytest.fixture
def isolated_app_settings(tmp_path, monkeypatch):
    from store.app_settings import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", tmp_path)
    monkeypatch.setattr(app_settings, "settings_file", tmp_path / "app_settings.json")
    monkeypatch.setattr(app_settings, "_cache", {})
    yield app_settings


def _sidecar(directory, key, positive, beat_id, workflow="zimage1.json", width=1584, height=896):
    (directory / f"{key}.png").write_bytes(b"png")
    (directory / f"{key}.json").write_text(json.dumps({
        "beat_id": beat_id, "kind": "broll_image", "workflow": workflow, "positive": positive,
        "negative": "", "seed": 1, "width": width, "height": height}), encoding="utf-8")


# --- E1: near match on the beat's own words, one asset per beat, drift ----------------------

def test_the_world_line_alone_is_not_a_near_match_but_the_beats_own_words_are(tmp_path):
    d = tmp_path / "generated"
    d.mkdir()
    _sidecar(d, "aaaa", TRACKS, "b213")
    assets._sidecar_cache.clear()
    boilerplate = assets.boilerplate_tokens([TOWER, TRACKS_REWORDED, OWL, VILLAGE])
    assert assets._tokens(WORLD) <= boilerplate

    # Sharing only the world line used to clear the 70% bar on its own.
    assert assets._near_cached(d, False, "zimage1.json", TOWER, 1584, 896, 0.7) is not None
    assert assets._near_cached(d, False, "zimage1.json", TOWER, 1584, 896, 0.7,
                               boilerplate=boilerplate) is None
    # A reworded prompt about the same thing still finds the picture.
    hit = assets._near_cached(d, False, "zimage1.json", TRACKS_REWORDED, 1584, 896, 0.7,
                              boilerplate=boilerplate)
    assert hit is not None and hit[0].name == "aaaa.png" and hit[2] == "b213"
    # Too few words of its own left: no near match at all.
    stub = "A ranger checks tyre marks. " + WORLD
    assert assets._near_cached(d, False, "zimage1.json", stub, 1584, 896, 0.7,
                               boilerplate=boilerplate) is None


def test_boilerplate_needs_four_prompts():
    assert assets.boilerplate_tokens([TOWER, OWL, VILLAGE]) == set()


def test_a_near_match_already_serving_a_beat_is_passed_over(tmp_path):
    d = tmp_path / "generated"
    d.mkdir()
    _sidecar(d, "aaaa", "A ranger checking fresh tyre marks on a muddy jeep track. " + WORLD, "b001")
    _sidecar(d, "cccc", "A ranger checking the fresh tyre marks on the muddy jeep road. " + WORLD, "b002")
    assets._sidecar_cache.clear()
    boilerplate = assets._tokens(WORLD)
    best = assets._near_cached(d, False, "zimage1.json", TRACKS_REWORDED, 1584, 896, 0.7,
                               boilerplate=boilerplate)
    assert best is not None and best[0].name == "aaaa.png"
    used = {str(best[0])}
    second = assets._near_cached(d, False, "zimage1.json", TRACKS_REWORDED, 1584, 896, 0.7,
                                 boilerplate=boilerplate, exclude=used)
    assert second is not None and second[0].name == "cccc.png"
    used.add(str(second[0]))
    assert assets._near_cached(d, False, "zimage1.json", TRACKS_REWORDED, 1584, 896, 0.7,
                               boilerplate=boilerplate, exclude=used) is None


def test_prompt_drift_flags_a_word_in_most_prompts_only():
    places = ["river", "temple", "village", "market", "bridge",
              "school", "station", "garden", "harbour", "valley"]
    prompts = [("a tiger near " if i < 9 else "")
               + ("a jeep by " if i < 3 else "")
               + f"the {place}, misty night light" for i, place in enumerate(places)]
    drift = assets.prompt_drift(prompts)
    assert drift["tiger"] == 0.9
    assert "jeep" not in drift, "30% is not drift"
    assert "misty" not in drift and "night" not in drift, "look words are not content"
    assert assets.prompt_drift([]) == {}


# --- E2: variations are different subjects; adjacent twins go --------------------------------

def _cut(beat_id, start, prompt):
    return Beat(id=beat_id, start_s=start, end_s=start + 6.0, topic=beat_id, kind="broll_image",
                priority=0.8, image_prompt=prompt)


def test_the_later_of_two_adjacent_twins_is_dropped_and_different_neighbours_stay():
    beats = [_cut("b00", 0.0, "A ranger kneels beside fresh tyre marks on a muddy jeep track. " + WORLD),
             _cut("b01", 10.0, "Ranger kneeling beside the fresh tyre marks on a muddy jeep track. " + WORLD),
             _cut("b02", 20.0, OWL),
             _cut("b03", 30.0, VILLAGE),
             _cut("b04", 40.0, TOWER),
             _cut("b05", 50.0, "Torchlight on a rusted iron gate of the forest rest house. " + WORLD)]
    kept, dropped = shotplan.adjacent_similarity(beats)
    assert [b.id for b in kept] == ["b00", "b02", "b03", "b04", "b05"]
    assert dropped == [{"topic": "b01", "reason": "twin of b00"}]


def test_variations_ask_for_different_subjects_and_have_no_angle_suffixes():
    assert "DIFFERENT subject" in shotplan.VARIATION_SYSTEM
    assert "another angle" in shotplan.VARIATION_SYSTEM
    assert not hasattr(shotplan, "SHOT_VARIATIONS")


async def test_fewer_variation_prompts_mean_fewer_longer_slices():
    beat = Beat(id="t", start_s=0.0, end_s=60.0, topic="the night patrol", kind="broll_image",
                priority=0.8, image_prompt="a ranger walks a muddy forest trail")
    given = ["headlights of a jeep cutting through rain on a forest road",
             "a wooden watchtower above the canopy"]

    async def two(system, user, schema=None):
        return json.dumps({"prompts": given})

    out = await shotplan.multiply_beats([beat], PresentationSettings(), two)
    assert len(out) == 2
    assert [(b.start_s, b.end_s) for b in out] == [(0.0, 30.0), (30.0, 60.0)]
    assert [b.image_prompt.startswith(p) for b, p in zip(out, given)] == [True, True]
    assert not any("close-up on a telling detail" in b.image_prompt for b in out)

    async def none(system, user, schema=None):
        return json.dumps({"prompts": []})

    out = await shotplan.multiply_beats([beat], PresentationSettings(), none)
    assert out == [beat], "no usable variation: the topic stays one shot"


# --- E3: a clip runs at least 5 s ----------------------------------------------------------------

def test_every_generated_clip_has_at_least_five_seconds_of_frames():
    assert assets.MIN_VIDEO_FRAMES == 81
    frames = assets._video_length_for(2.0, 16)
    assert frames >= 81 and (frames - 1) % 4 == 0


def test_a_video_beat_is_planned_five_to_eight_seconds_or_becomes_a_still():
    program = Program(duration_s=120.0, fps=30.0)
    settings = PresentationSettings(broll_seconds_min=3.0, broll_seconds_max=3.0, target_coverage=0.75)
    short = Beat(start_s=0.0, end_s=4.0, topic="short", kind="broll_video", priority=0.9,
                 image_prompt="a", video_prompt="a")
    long = Beat(start_s=30.0, end_s=45.0, topic="long", kind="broll_video", priority=0.8,
                image_prompt="b", video_prompt="b")
    kept, _ = shotplan.budget_beats([short, long], program, settings)
    by_topic = {b.topic: b for b in kept}
    assert by_topic["short"].kind == "broll_image"
    assert by_topic["long"].kind == "broll_video" and by_topic["long"].planned_duration_s == 5.0

    wide = PresentationSettings(broll_seconds_min=3.0, broll_seconds_max=8.0, target_coverage=0.75)
    longer = long.model_copy(update={"end_s": 60.0})
    kept, _ = shotplan.budget_beats([longer], program, wide)
    assert 5.0 <= kept[0].planned_duration_s <= 8.0


def test_only_a_span_that_holds_five_seconds_is_promoted_to_a_clip():
    beats = [Beat(id="b00", start_s=0.0, end_s=4.0, topic="short", kind="broll_image", priority=1.0,
                  planned_duration_s=3.0, image_prompt="a"),
             Beat(id="b01", start_s=10.0, end_s=17.0, topic="long", kind="broll_image", priority=0.9,
                  planned_duration_s=3.0, image_prompt="b")]
    out = shotplan.balance_video_share(beats, PresentationSettings(video_broll_share=0.5))
    planned = {b.topic: (b.kind, b.planned_duration_s) for b in out}
    assert planned["short"] == ("broll_image", 3.0)
    assert planned["long"] == ("broll_video", 5.0)


async def test_the_video_cache_key_carries_the_clip_length(tmp_path, monkeypatch):
    """An old clip under 5 s is never an exact hit for a beat that now plans 5 s or
    more: the clip length is part of the key (near matches check the duration)."""
    made = tmp_path / "made.mp4"
    made.write_bytes(b"not really an mp4")

    class _Queue:
        def __init__(self):
            self.graphs = []

        async def submit_and_wait(self, graph, timeout=0):
            self.graphs.append(graph)
            return [str(made)]

    monkeypatch.setattr(assets, "_setting", lambda key, default: default)
    monkeypatch.setattr(assets, "_probe", lambda path: (1920, 1080, 5.06))
    resolved = {"broll_video": workflows.ResolvedWorkflow("broll_video", "fakev.json", {}, {}, "video")}
    beat = Beat(id="b01", start_s=0.0, end_s=10.0, kind="broll_video", priority=0.8,
                image_prompt="a jeep", video_prompt="a jeep crossing a flooded forest road at night",
                planned_duration_s=5.0)
    directory = tmp_path / "generated"
    directory.mkdir()
    queue = _Queue()
    five = await assets._generate_one(beat, resolved, directory, queue, 0)
    again = await assets._generate_one(beat, resolved, directory, queue, 0)
    eight = await assets._generate_one(beat.model_copy(update={"planned_duration_s": 8.0}),
                                       resolved, directory, queue, 0)
    assert again.cache_hit and again.prompt_sha == five.prompt_sha, "same length: the same file"
    assert eight.prompt_sha != five.prompt_sha
    assert len(queue.graphs) == 2, "a longer clip is generated, not taken from the shorter one"


def test_placement_never_cuts_a_clip_under_five_seconds():
    from presentation.placement import _window_for

    class _P:
        duration_s = 600.0
        words = []

    settings = PresentationSettings()
    clip = Asset(beat_id="b", kind="video", path="x.mp4", duration_s=5.06)
    beat = Beat(id="b", kind="broll_video", start_s=20.0, end_s=24.0, priority=0.8,
                planned_duration_s=5.0, image_prompt="x", video_prompt="x")
    assert _window_for(beat, clip, _P(), [], settings) == (20.0, 0.0, "a clip needs 5 s; no room")
    roomy = beat.model_copy(update={"end_s": 30.0})
    assert _window_for(roomy, clip, _P(), [], settings) == (20.0, 5.0, "")
    # A shorter clip made before the rule keeps the old fitting.
    older = Asset(beat_id="b", kind="video", path="x.mp4", duration_s=3.0)
    _, seconds, _ = _window_for(beat, older, _P(), [], settings)
    assert 0.0 < seconds <= 3.0


# --- E4: the classic / hq toggle -----------------------------------------------------------------

def test_the_hq16_graph_is_the_hq_graph_without_rife_at_16_fps():
    graph = workflows.load_graph("zimage_wan22_i2v_hq16.json")
    hq = workflows.load_graph("zimage_wan22_i2v_hq.json")
    assert "r_run" not in graph and "r_model" not in graph
    assert not any(isinstance(value, list) and value and value[0] in ("r_run", "r_model")
                   for node in graph.values() for value in (node.get("inputs") or {}).values())
    assert workflows.frame_multiplier(graph) == 1
    assert graph["v_create"]["inputs"]["images"] == ["u_scale", 0]
    assert graph["v_create"]["inputs"]["fps"] == 16.0
    assert ({k: v for k, v in graph.items() if k != "v_create"}
            == {k: v for k, v in hq.items() if k not in ("v_create", "r_run", "r_model")})
    detail = workflows.describe("zimage_wan22_i2v_hq16.json")
    assert detail["valid"] and detail["roles_ok"]["broll_video"]


def test_resolve_follows_the_clip_generation_mode(isolated_app_settings):
    classic = workflows.resolve("broll_video")
    assert classic.file == "zimage_wan22_i2v.json" and classic.fallback is None

    isolated_app_settings._cache["clip_generation_mode"] = "hq"
    hq = workflows.resolve("broll_video")
    assert hq.file == "zimage_wan22_i2v_hq16.json"
    assert hq.fallback is not None and hq.fallback.file == "zimage_wan22_i2v.json"
    built = hq.build(positive="mist toward a door", width=1024, height=576, seed=11, length=81,
                     fps=16, prefix="beat_b02", out_width=1920, out_height=1080)
    assert built["v_create"]["inputs"]["fps"] == 16
    assert (built["u_scale"]["inputs"]["width"], built["u_scale"]["inputs"]["height"]) == (1920, 1080)
    assert built["w_i2v"]["inputs"]["length"] == 81

    isolated_app_settings._cache["clip_generation_mode"] = "turbo"
    assert workflows.resolve("broll_video").file == "zimage_wan22_i2v.json", "unknown reads as classic"

    # A graph named in the settings wins over the mode, as before.
    isolated_app_settings._cache["clip_generation_mode"] = "hq"
    isolated_app_settings._cache["workflow_broll_video"] = "zimage_wan22_i2v_hq.json"
    explicit = workflows.resolve("broll_video")
    assert explicit.file == "zimage_wan22_i2v_hq.json" and explicit.fallback is None


def test_the_selection_shown_follows_the_clip_generation_mode(isolated_app_settings):
    assert workflows.current_selection()["broll_video"] == "zimage_wan22_i2v.json"
    isolated_app_settings._cache["clip_generation_mode"] = "hq"
    assert workflows.current_selection()["broll_video"] == "zimage_wan22_i2v_hq16.json"
    isolated_app_settings._cache["workflow_broll_video"] = "zimage_wan22_i2v_hq.json"
    assert workflows.current_selection()["broll_video"] == "zimage_wan22_i2v_hq.json", \
        "a graph named in the settings still wins"


def test_video_generation_size_follows_the_mode(isolated_app_settings):
    assert assets.max_video_gen_pixels() == 832 * 480
    w, h = assets._video_canvas_default((1920, 1080))
    assert w * h <= 832 * 480
    isolated_app_settings._cache["clip_generation_mode"] = "hq"
    assert assets.max_video_gen_pixels() == 1024 * 576
    w, h = assets._video_canvas_default((1920, 1080))
    assert 832 * 480 < w * h <= 1024 * 576 and abs(w / h - 16 / 9) < 0.05


@pytest.fixture
def client(isolated_app_settings):
    from main import app
    with TestClient(app) as test_client:
        yield test_client


def test_the_clip_generation_route_reads_stores_and_validates_the_mode(client, isolated_app_settings):
    got = client.get("/api/settings/clip_generation")
    assert got.status_code == 200
    body = got.json()
    assert body["mode"] == "classic"
    assert body["modes"]["classic"]["file"] == "zimage_wan22_i2v.json"
    assert body["modes"]["hq"]["file"] == "zimage_wan22_i2v_hq16.json"
    assert body["modes"]["hq"]["size"] == [1024, 576]
    assert body["modes"]["hq"]["fps"] == body["modes"]["classic"]["fps"] == 16

    put = client.put("/api/settings/clip_generation", json={"mode": "hq"})
    assert put.status_code == 200 and put.json()["mode"] == "hq"
    assert isolated_app_settings.get("clip_generation_mode") == "hq"

    bad = client.put("/api/settings/clip_generation", json={"mode": "turbo"})
    assert bad.status_code == 422
    assert client.get("/api/settings/clip_generation").json()["mode"] == "hq"


# --- E5: no still inside a clip ------------------------------------------------------------------

def test_frozen_frames_inside_a_clip_are_reported(tmp_path):
    from presentation import verify
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    render = tmp_path / "render.mp4"
    # Two seconds of motion, then the last frame held for two more.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=320x180:rate=16:duration=2",
                    "-vf", "tpad=stop_mode=clone:stop_duration=2", "-pix_fmt", "yuv420p",
                    str(render)], check=True)
    frozen = verify.frozen_clip_frames(str(render), [{"start_s": 0.0, "end_s": 4.0, "beat_id": "b07"}])
    assert len(frozen) == 1
    assert frozen[0]["beat_id"] == "b07"
    assert 1.5 <= frozen[0]["start_s"] <= 2.25
    assert frozen[0]["seconds"] >= 1.5
    moving = verify.frozen_clip_frames(str(render), [{"start_s": 0.0, "end_s": 2.0, "beat_id": "b08"}])
    assert moving == []
