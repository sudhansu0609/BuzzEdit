"""Lip-sync check (docs/LIPSYNC_CHECK_PLAN.md): the scoring, the per-span rules and the route's
503. No GPU: the landmark pass and the audio decode are monkeypatched."""

import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import lipsync_check

N = 200            # 8 s at 25 fps


def _mouth(seed: int = 0) -> np.ndarray:
    """A syllable-rate (~4 Hz) mouth-open series, smooth and positive."""
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.random(N + 8), np.hanning(7) / np.hanning(7).sum(), mode="same")[4:N + 4]
    return 0.05 + 0.3 * (x - x.min()) / (x.max() - x.min())


def _shift(x: np.ndarray, frames: int) -> np.ndarray:
    """The same series `frames` later (audio after the mouth)."""
    return np.concatenate([np.full(frames, x[0]), x[:-frames]])


@pytest.mark.parametrize("case", ["matched", "shifted", "noise"])
def test_sync_score(case):
    mouth = _mouth()
    env = {"matched": np.sqrt(mouth), "shifted": _shift(np.sqrt(mouth), 4),
           "noise": np.random.default_rng(9).random(N)}[case]
    score = lipsync_check.sync_score(mouth, env)
    if case == "noise":
        assert score["corr"] < 0.3
    else:
        assert score["corr"] > 0.9 and score["lag_frames"] == (0 if case == "matched" else 4)


def test_check_spans_rules(monkeypatch):
    mouth = _mouth()
    no_face = mouth.copy()
    no_face[: N // 2] = np.nan                                   # face in 50 % of frames
    mouths = {0.0: mouth, 10.0: mouth, 20.0: mouth, 40.0: no_face}
    envs = {"conv": {0.0: np.sqrt(mouth), 10.0: _shift(np.sqrt(mouth), 4),
                     20.0: np.random.default_rng(3).random(N), 40.0: np.sqrt(mouth)}}
    monkeypatch.setattr(lipsync_check, "mouth_openness", lambda src, s, e, fps: mouths[s])
    monkeypatch.setattr(lipsync_check, "audio_envelope",
                        lambda path, s, e, fps: np.sqrt(mouth) if path == "orig.wav" else envs["conv"][s])
    rows = lipsync_check.check_spans("v.mp4", "orig.wav", "conv.wav",
                                     [[0, 8], [10, 18], [20, 28], [30, 30.5], [40, 48]], fps=25.0)
    by = {row["start_s"]: row for row in rows}
    assert by[0.0]["flag"] is False and by[0.0]["reason"] is None and by[0.0]["corr_converted"] > 0.9
    assert by[10.0]["flag"] is True and by[10.0]["reason"] == "lag" and by[10.0]["lag_frames"] == 4
    assert by[20.0]["flag"] is True and by[20.0]["reason"] == "weak_match"
    assert by[30.0]["flag"] is False and by[30.0]["reason"] == "short" and math.isnan(by[30.0]["corr_original"])
    assert by[40.0]["flag"] is False and by[40.0]["reason"] == "no_face"
    summary = lipsync_check.summary(rows)
    assert summary["flag"] is True and len(summary["spans"]) == 5


def test_route_503_without_landmarks(monkeypatch):
    import eyecontact
    from routes import audio_voice
    monkeypatch.setattr(eyecontact, "available", lambda: (False, "No CUDA GPU is available to torch"))
    app = FastAPI()
    app.include_router(audio_voice.router, prefix="/api/audio")
    resp = TestClient(app).post("/api/audio/p1/voice/lipsync_check", json={"spans": [[1.0, 3.0]]})
    assert resp.status_code == 503 and "CUDA" in resp.json()["detail"]
