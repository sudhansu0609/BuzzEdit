"""Tests for handheld-clip stabilization.

No GPU and no ffmpeg run here (the real check is a manual run of stabilize_video on a
shaky clip, see docs/STABILIZE_PLAN.md); what is tested is everything that decides what
ffmpeg is asked to do, how the .trf is read, and the project swap/revert.
"""
import os
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import stabilize
from stabilize import Settings, detect_filter, frame_motion, parse_trf, shake_stats, shakiness, transform_filter


# ---------------------------------------------------------------- settings -> filters

def test_strength_maps_to_shakiness_1_to_10():
    assert shakiness(0.0) == 1 and shakiness(1.0) == 10 and shakiness(0.6) == 6
    assert shakiness(-1) == 1 and shakiness(5) == 10


def test_detect_filter():
    assert detect_filter(Settings(), "a.trf") == \
        "vidstabdetect=shakiness=6:accuracy=15:fileformat=ascii:result=a.trf"
    assert detect_filter(Settings(strength=1.0, tripod=True), "a.trf").endswith(":result=a.trf:tripod=1")
    assert "shakiness=10" in detect_filter(Settings(strength=1.0), "a.trf")


def test_transform_filter_crop_auto():
    vf = transform_filter(Settings(), "a.trf", 25.0, 2.0)
    assert vf == "vidstabtransform=input=a.trf:smoothing=25:optzoom=1:crop=keep:interpol=bicubic"


def test_transform_filter_crop_keep_and_black():
    assert transform_filter(Settings(crop="keep"), "a.trf", 30.0) == \
        "vidstabtransform=input=a.trf:smoothing=30:optzoom=0:crop=keep:interpol=bicubic"
    assert transform_filter(Settings(crop="black"), "a.trf", 30.0) == \
        "vidstabtransform=input=a.trf:smoothing=30:optzoom=0:crop=black:interpol=bicubic"


def test_transform_filter_fixed_zoom():
    vf = transform_filter(Settings(zoom_pct=5.0), "a.trf", 30.0, 5.0)
    assert "optzoom=0:zoom=5:crop=keep" in vf and vf.endswith(",unsharp=5:5:0.4:3:3:0.2")
    vf = transform_filter(Settings(zoom_pct=2.5, crop="black"), "a.trf", 30.0, 2.5)
    assert "optzoom=0:zoom=2.5:crop=black" in vf and "unsharp" not in vf


def test_transform_filter_tripod_and_smoothing():
    vf = transform_filter(Settings(tripod=True, smoothing_s=1.0), "a.trf", 29.97)
    assert ":smoothing=30:" in vf and ":tripod=1:" in vf
    assert ":smoothing=0:" in transform_filter(Settings(smoothing_s=0.0), "a.trf", 30.0)
    assert ":smoothing=1000:" in transform_filter(Settings(smoothing_s=10.0), "a.trf", 240.0)  # clamped


def test_unsharp_only_past_3_percent():
    assert "unsharp" not in transform_filter(Settings(), "a.trf", 30.0, 3.0)
    assert transform_filter(Settings(), "a.trf", 30.0, 3.01).endswith(",unsharp=5:5:0.4:3:3:0.2")


@pytest.mark.parametrize("bad", [dict(strength=1.5), dict(crop="zoom"), dict(smoothing_s=-1),
                                 dict(zoom_pct=80.0), dict(quality="ultra")])
def test_settings_reject_out_of_range(bad):
    with pytest.raises(ValueError):
        Settings(**bad)


# ---------------------------------------------------------------- .trf parsing

TRF = """VID.STAB 1
#      accuracy = 15
#     shakiness = 6
#      stepsize = 6
#   mincontrast = 0.250000
Frame 1 (List 0 [])
Frame 2 (List 3 [(LM 10 2 1456 216 112 0.47 0.66),(LM 11 -2 1084 702 112 0.70 0.30),(LM 40 43 712 216 112 0.63 0.58)])
Frame 3 (List 2 [(LM -4 0 10 10 112 0.5 0.5),(LM -6 2 20 20 112 0.5 0.5)])
Frame 4 (List 0 [])
"""


def test_parse_trf(tmp_path):
    p = tmp_path / "t.trf"
    p.write_text(TRF)
    m = parse_trf(p)
    assert m.shape == (4, 2)
    assert np.isnan(m[0]).all() and np.isnan(m[3]).all()
    assert m[1].tolist() == [11.0, 2.0]          # median: the outlier field (40, 43) does not count
    assert m[2].tolist() == [-5.0, 1.0]


def test_tripod_motion_is_differenced(tmp_path):
    p = tmp_path / "t.trf"
    p.write_text("VID.STAB 1\nFrame 1 (List 1 [(LM 0 0 1 1 112 0.5 0.5)])\n"
                 "Frame 2 (List 1 [(LM 3 4 1 1 112 0.5 0.5)])\nFrame 3 (List 1 [(LM 3 4 1 1 112 0.5 0.5)])\n")
    assert frame_motion(p, Settings()).tolist() == [[0, 0], [3, 4], [3, 4]]
    assert frame_motion(p, Settings(tripod=True)).tolist() == [[0, 0], [3, 4], [0, 0]]


def test_shake_stats():
    fps = 25
    t = np.arange(250) / fps
    pan = np.stack([np.full_like(t, 5.0), np.zeros_like(t)], axis=1)     # steady pan: not shake
    mean, peak = shake_stats(pan, fps, 1080)
    assert mean < 0.5
    wobble = np.stack([10 * np.sin(2 * np.pi * 3 * t), np.zeros_like(t)], axis=1)
    mean, peak = shake_stats(wobble, fps, 1080)
    assert 5 < mean < 8 and 9 < peak <= 10.5
    assert shake_stats(wobble, fps, 2160)[0] == pytest.approx(mean / 2, abs=0.02)   # 1080p-equivalent
    assert shake_stats(np.full((3, 2), np.nan), fps, 1080) == (0.0, 0.0)


def test_cache_key_is_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(stabilize, "CACHE_DIR", tmp_path / "cache")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x" * 100)
    a = stabilize._cache_path(str(src), Settings())
    assert a == stabilize._cache_path(str(src), Settings(crop="black", smoothing_s=2.0, quality="max"))
    assert a.parent == tmp_path / "cache" and a.suffix == ".trf"
    assert a != stabilize._cache_path(str(src), Settings(strength=1.0))     # shakiness changes the trf
    assert a != stabilize._cache_path(str(src), Settings(tripod=True))
    src.write_bytes(b"x" * 101)
    assert a != stabilize._cache_path(str(src), Settings())                  # the file changed


def test_default_output_never_overwrites(monkeypatch):
    assert stabilize.default_output(r"B:\F\take1.MOV").endswith("take1_stable.mp4")
    monkeypatch.setattr(stabilize, "available", lambda: (True, ""))
    with pytest.raises(ValueError):
        stabilize.stabilize_video(r"B:\F\x.mp4", r"b:\f\X.mp4")


# ---------------------------------------------------------------- project swap / revert

def test_swap_and_revert_on_a_project():
    from routes.stabilize import swap_source
    old, new = r"B:\Footage\walk.MOV", r"B:\Footage\walk_stable.mp4"
    p = {"source_video": old, "timeline": {"sources": {"s1": {"path": old.upper()}, "s2": {"path": r"B:\b.mp4"}}}}
    assert swap_source(p, old, new) == 2
    assert p["source_video"] == new and p["timeline"]["sources"]["s1"]["path"] == new
    assert p["timeline"]["sources"]["s2"]["path"] == r"B:\b.mp4"
    assert swap_source(p, new, old) == 2
    assert p["source_video"] == old and p["timeline"]["sources"]["s1"]["path"] == old


# ---------------------------------------------------------------- routes

@pytest.fixture
def client(tmp_path, monkeypatch):
    from routes import stabilize as route
    from store import media_pool
    from store.project_store import ProjectStore

    store = ProjectStore(base_dir=str(tmp_path / "projects"))
    monkeypatch.setattr(route, "project_store", store)
    monkeypatch.setattr(stabilize, "available", lambda: (True, ""))
    monkeypatch.setattr(media_pool, "probe", lambda path, kind: {"duration": 20.0, "width": 1920, "height": 1080,
                                                                 "has_audio": True, "fps": 25.0})
    saved = {}
    monkeypatch.setattr(route.app_settings, "get", lambda key, default=None: saved.get(key, default))
    monkeypatch.setattr(route.app_settings, "set", lambda key, value: saved.__setitem__(key, value))

    async def _no_lease(*a, **k):
        return None
    monkeypatch.setattr(route.gpu_broker, "acquire_lease", _no_lease)
    monkeypatch.setattr(route.gpu_broker, "release_lease", _no_lease)

    calls = []

    def fake_stabilize(src, dst=None, settings=Settings(), progress=None, cancelled=None, codec="hevc"):
        calls.append((src, dst, settings))
        if progress:
            progress(0.5)
        with open(dst, "wb") as f:
            f.write(b"stable")
        return {"output_path": dst, "frames": 500, "seconds": 1.0, "zoom_pct": 2.0,
                "mean_shift_px": 8.0, "max_shift_px": 13.0, "settings": {}}
    monkeypatch.setattr(stabilize, "stabilize_video", fake_stabilize)

    src = tmp_path / "walk.mp4"
    src.write_bytes(b"original")
    store.save_project("p1", {"id": "p1", "source_video": str(src), "timeline": {"sources": {"s1": {"path": str(src)}}},
                              "media_pool": [{"id": "m1", "name": "My walk", "path": str(src), "kind": "video"}]})
    app = FastAPI()
    app.include_router(route.router, prefix="/api/stabilize")
    route.jobs.clear()
    with TestClient(app) as c:
        yield c, store, src, calls, saved


def _wait(c, job_id):
    for _ in range(200):
        job = c.get(f"/api/stabilize/status/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_start_status_revert_round_trip(client):
    c, store, src, calls, _ = client
    r = c.post("/api/stabilize/projects/p1/media/m1", json={"strength": 0.8, "crop": "black"})
    assert r.status_code == 200
    job = _wait(c, r.json()["job_id"])
    assert job["status"] == "completed" and job["result"]["zoom_pct"] == 2.0
    dst = str(src.with_name("walk_stable.mp4"))
    assert calls[0][0] == str(src) and calls[0][1] == dst
    assert calls[0][2] == Settings(strength=0.8, crop="black")
    p = store.get_project("p1")
    entry = p["media_pool"][0]
    assert entry["path"] == dst and entry["name"] == "My walk"
    assert entry["stabilize"]["original_path"] == str(src) and entry["stabilize"]["report"]["frames"] == 500
    assert p["source_video"] == dst and p["timeline"]["sources"]["s1"]["path"] == dst
    assert src.read_bytes() == b"original"                     # never overwritten

    # a re-run starts from the original again, not from the stabilized copy
    job = _wait(c, c.post("/api/stabilize/projects/p1/media/m1", json={}).json()["job_id"])
    assert job["status"] == "completed" and calls[1][0] == str(src)

    r = c.post("/api/stabilize/projects/p1/media/m1/revert")
    assert r.status_code == 200
    p = store.get_project("p1")
    entry = p["media_pool"][0]
    assert entry["path"] == str(src) and "stabilize" not in entry and entry["name"] == "My walk"
    assert p["source_video"] == str(src) and p["timeline"]["sources"]["s1"]["path"] == str(src)
    assert os.path.exists(dst)                                  # the stabilized file stays on disk
    assert c.post("/api/stabilize/projects/p1/media/m1/revert").status_code == 400


def test_unknown_media_and_project_404(client):
    c = client[0]
    assert c.post("/api/stabilize/projects/p1/media/nope", json={}).status_code == 404
    assert c.post("/api/stabilize/projects/nope/media/m1", json={}).status_code == 404
    assert c.post("/api/stabilize/projects/p1/media/nope/revert").status_code == 404
    assert c.get("/api/stabilize/status/nope").status_code == 404
    assert c.post("/api/stabilize/cancel/nope").status_code == 404


@pytest.mark.parametrize("body", [{"strength": 2}, {"crop": "zoom"}, {"smoothing_s": -1},
                                  {"zoom_pct": 90}, {"quality": "ultra"}])
def test_bad_settings_422(client, body):
    c = client[0]
    assert c.post("/api/stabilize/projects/p1/media/m1", json=body).status_code == 422
    assert c.put("/api/stabilize/settings", json=body).status_code == 422


def test_settings_saved_and_merged(client):
    c, _store, _src, calls, saved = client
    assert c.get("/api/stabilize/settings").json() == {"strength": 0.6, "crop": "auto", "smoothing_s": 1.0,
                                                       "zoom_pct": None, "tripod": False, "quality": "high"}
    r = c.put("/api/stabilize/settings", json={"smoothing_s": 1.0, "zoom_pct": 4.0})
    assert r.json()["smoothing_s"] == 1.0 and saved["stabilize"]["zoom_pct"] == 4.0
    _wait(c, c.post("/api/stabilize/projects/p1/media/m1", json={"tripod": True}).json()["job_id"])
    assert calls[-1][2] == Settings(smoothing_s=1.0, zoom_pct=4.0, tripod=True)
    assert c.put("/api/stabilize/settings", json={"zoom_pct": None}).json()["zoom_pct"] is None


def test_available_route(client):
    assert client[0].get("/api/stabilize/available").json() == {"available": True, "reason": ""}
