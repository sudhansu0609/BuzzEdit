"""Live audio engine (AUDIO_SUITE_PLAN.md brief L): PCM against credits, a live azmq `set`
heard within 250 ms of audio, a restart that resumes in place within 400 ms, no zombies.

Synthetic input only (a 1 kHz sine made once with ffmpeg); the project is a stub written the
way ProjectStore stores it. Set BUZZEDIT_LIVE_TEST_LOADED=1 to skip only the wall-clock
assertion of the restart test on a loaded machine.
"""

import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import live
from config import FFMPEG_BIN
from routes import audio_live
from store.project_store import ProjectStore
from timeline.schema import SourceFile, Timeline, TimelineItem

SR = 48000
PROJECT = "live_test"
URL = f"/api/audio/{PROJECT}/live?scope=A1"
LOADED = os.environ.get("BUZZEDIT_LIVE_TEST_LOADED") == "1"


# -- fixtures -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("live") / "sine.wav"
    subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=1000:sample_rate=48000", "-t", "4", str(path)], check=True)
    return path


@pytest.fixture
def client(tmp_path, sine_wav, monkeypatch):
    store = ProjectStore(base_dir=str(tmp_path / "projects"))
    tl = Timeline(
        sources={"src1": SourceFile(id="src1", path=str(sine_wav), duration_seconds=4.0, kind="audio")},
        items=[TimelineItem(id="a1", track="A1", source_id="src1", source_end_frame=120, timeline_end_frame=120)],
        duration_frames=120,
    ).model_dump()
    tl["audio_racks"] = {"A1": [{"id": "fx_amp", "effect": "amplify", "params": {"gain_db": 0.0}, "enabled": True}]}
    store.save_project(PROJECT, {"id": PROJECT, "name": "live test", "source_video": str(sine_wav), "timeline": tl})
    monkeypatch.setattr(audio_live, "project_store", store)
    # The brief's minimal amplify compile, whatever audio.compile (brief A1b) ships meanwhile.
    monkeypatch.setattr(live, "_load_compile_api", lambda: None)
    app = FastAPI()
    app.include_router(audio_live.router, prefix="/api/audio")
    with TestClient(app) as c:
        yield c


# -- helpers ------------------------------------------------------------------------------

def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float64).reshape(-1, 2) / 32768.0


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


class Player:
    """Acts as the app: grants credits, collects PCM and JSON events."""

    def __init__(self, ws):
        self.ws = ws
        self.pcm = bytearray()
        self.events = []          # (perf_counter, event)
        self.pcm_times = []       # (perf_counter, frames after this message)
        self.credited = 0
        self.max_ahead = 0        # most frames ever received beyond the credits granted

    @property
    def frames(self) -> int:
        return len(self.pcm) // 4

    def send(self, **msg):
        self.ws.send_text(json.dumps(msg))

    def grant(self, frames: int):
        self.send(type="credits", frames=int(frames))
        self.credited += int(frames)

    def receive_one(self):
        msg = self.ws.receive()
        if msg.get("bytes") is not None:
            assert len(msg["bytes"]) % 4 == 0, "PCM message is not whole stereo frames"
            self.pcm += msg["bytes"]
            self.pcm_times.append((time.perf_counter(), self.frames))
            self.max_ahead = max(self.max_ahead, self.frames - self.credited)
            return "pcm", None
        if msg.get("text") is not None:
            event = json.loads(msg["text"])
            self.events.append((time.perf_counter(), event))
            return "json", event
        raise AssertionError(f"socket closed: {msg}")

    def collect(self, seconds: float):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.receive_one()

    def play_realtime(self, seconds: float, lead: float = 0.15, until=None):
        """Consume in real time; credits keep the server at most `lead` s ahead of the clock."""
        t0 = time.perf_counter()
        base = self.credited
        while time.perf_counter() - t0 < seconds:
            want = base + int((time.perf_counter() - t0 + lead) * SR)
            if want - self.credited >= 256:
                self.grant(want - self.credited)
            kind, event = self.receive_one()
            if until is not None and until(kind, event):
                return True
        return False

    def errors(self):
        return [e for _, e in self.events if e.get("type") == "error"]


# -- tests --------------------------------------------------------------------------------

def test_play_with_credits_streams_the_sine(client, sine_wav):
    ref = to_float(subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-i", str(sine_wav),
         "-f", "s16le", "-ar", "48000", "-ac", "2", "pipe:1"], capture_output=True, check=True).stdout)
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.grant(SR)
        end = time.perf_counter() + 10
        while p.frames < SR and time.perf_counter() < end:
            p.receive_one()
    assert not p.errors()
    assert p.frames == SR
    x = to_float(p.pcm)[SR // 10:]
    assert abs(rms_db(x) - rms_db(ref[SR // 10:SR])) < 0.3
    crossings = np.count_nonzero(np.diff(np.signbit(x[:, 0])))
    assert abs(crossings / 2 / (len(x) / SR) - 1000) < 10   # it is the 1 kHz tone


def test_live_set_is_heard_within_250ms_of_audio(client):
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.play_realtime(1.0)                     # the azmq link is up ~0.5 s after spawn
        mark = p.frames
        p.send(type="set", effect_id="fx_amp", param="gain_db", value=-20.0)
        p.play_realtime(0.8)
    assert not p.errors()
    x = to_float(p.pcm)
    before = rms_db(x[mark - SR // 4:mark])
    block = 240                                  # 5 ms
    lag_frames = None
    for start in range(mark, len(x) - block, block):
        if rms_db(x[start:start + block]) <= before - 15.0:
            lag_frames = start - mark
            break
    assert lag_frames is not None, "the -20 dB set was never heard"
    lag_ms = lag_frames / SR * 1000
    print(f"\nknob->ear (live azmq set): {lag_ms:.0f} ms of audio")
    assert lag_ms <= 250
    after = rms_db(x[-SR // 5:])
    assert abs((after - before) + 20.0) < 0.5


def test_rack_replace_restarts_in_place_within_400ms(client):
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.play_realtime(1.0)
        position = p.frames / SR
        before = rms_db(to_float(p.pcm)[-SR // 4:])
        sent_at = time.perf_counter()
        p.send(type="rack", effects=[{"id": "fx_amp", "effect": "amplify", "params": {"gain_db": -6.0}}])
        seen = {}

        def watch(kind, event):
            if kind == "json" and event.get("type") == "restarted" and "restarted" not in seen:
                seen["restarted"] = event["t"]
                seen["frames"] = p.frames
            elif kind == "pcm" and "restarted" in seen and p.frames > seen["frames"]:
                seen["resumed_at"] = time.perf_counter()
                return True
            return False

        assert p.play_realtime(3.0, until=watch), "no PCM after the restart"
        p.play_realtime(0.4)
    assert not p.errors()
    assert abs(seen["restarted"] - position) <= 0.1
    resume_ms = (seen["resumed_at"] - sent_at) * 1000
    print(f"\nrack replace (restart) -> first PCM: {resume_ms:.0f} ms wall")
    if not LOADED:
        assert resume_ms <= 400
    after = rms_db(to_float(p.pcm)[-SR // 5:])
    assert abs((after - before) + 6.0) < 0.5      # the new rack is what plays now


def test_server_never_sends_more_than_credited(client):
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.grant(4800)
        p.collect(0.8)
        assert p.frames == 4800
        p.grant(2400)
        p.collect(0.6)
        assert p.frames == 7200
    assert p.max_ahead <= 0
    assert not p.errors()


def test_disconnect_kills_ffmpeg(client):
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.grant(9600)
        p.collect(0.5)
        sessions = list(live.SESSIONS)
        assert len(sessions) == 1
        procs = list(sessions[0].spawned)
        assert procs and procs[-1].poll() is None   # alive, blocked on the pipe
    end = time.perf_counter() + 5
    while any(proc.poll() is None for proc in procs) and time.perf_counter() < end:
        time.sleep(0.05)
    assert all(proc.poll() is not None for proc in procs)
    assert not live.SESSIONS


class _SegmentedCompile:
    """Stands in for audio.compile with the shape a reverb's in-graph IR gives: ";"-separated
    segments, internal labels, an extra source; head input and tail output unlabeled."""

    @staticmethod
    def compile_rack(effects, *, scope):
        eid = getattr(effects[0], "id", None) or effects[0]["id"]
        filters = [f"volume@{eid}_0=volume=1[{eid}_main]",
                   f"sine=frequency=500:sample_rate=48000,volume@{eid}_1=volume=0[{eid}_side]",
                   f"[{eid}_main][{eid}_side]amix@{eid}_2=inputs=2:duration=first:normalize=0"]
        effect = SimpleNamespace(effect_id=eid, filters=filters, bypass="restart", commands={}, warnings=[])
        return SimpleNamespace(filters=filters, chain=";".join(filters), effects=[effect])

    @staticmethod
    def commands_for(rack, effect_id, param, value, params):
        if param == "side":
            return [(f"volume@{effect_id}_1", "volume", str(float(value)))]
        return None


def _tone_ratio(x: np.ndarray) -> float:
    """|500 Hz| / |1 kHz| over the last 0.25 s of the left channel (4 Hz bins)."""
    spectrum = np.abs(np.fft.rfft(x[-12000:, 0]))
    return float(spectrum[125] / spectrum[250])


def test_segmented_rack_chain_plays_and_takes_live_commands(client, monkeypatch):
    monkeypatch.setattr(live, "_load_compile_api", lambda: _SegmentedCompile)
    with client.websocket_connect(URL) as ws:
        p = Player(ws)
        p.send(type="play")
        p.play_realtime(1.0)
        before = to_float(p.pcm)
        p.send(type="set", effect_id="fx_amp", param="side", value=1.0)
        p.play_realtime(0.6)
    assert not p.errors()
    assert abs(rms_db(before[SR // 2:]) - rms_db(before[SR // 4:SR // 2])) < 0.3
    assert _tone_ratio(before) < 0.01                      # the extra source is mixed in silent
    assert 0.8 < _tone_ratio(to_float(p.pcm)) < 1.25       # a live command reached a labeled segment


# -- programme time (row L2): a cut timeline, 440 Hz kept first, 880 Hz kept second ----------

_REAL_COMPILE_API = live._load_compile_api
CUT_PROJECT = "live_cut"
CUT_URL = f"/api/audio/{CUT_PROJECT}/live?scope=A1"


@pytest.fixture(scope="module")
def two_tones(tmp_path_factory):
    """6 s stereo 48 kHz: 440 Hz for 0-2 s, silence 2-4 s, 880 Hz 4-6 s."""
    path = tmp_path_factory.mktemp("live_cut") / "two_tones.wav"
    subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                    "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=2",
                    "-filter_complex", "[1:a]atrim=duration=2[gap];[0:a][gap][2:a]concat=n=3:v=0:a=1",
                    "-ar", "48000", "-ac", "2", str(path)], check=True)
    return path


@pytest.fixture
def cut_client(tmp_path, two_tones, monkeypatch):
    """Kept: source 0.5-1.5 s (440 Hz, programme 0-1) and 4.5-5.5 s (880 Hz, programme 1-2)."""
    store = ProjectStore(base_dir=str(tmp_path / "projects"))
    items = [TimelineItem(id=f"a1_{k}", track="A1", source_id="s", source_start_frame=a,
                          source_end_frame=b, timeline_start_frame=30 * k, timeline_end_frame=30 * k + 30)
             for k, (a, b) in enumerate([(15, 45), (135, 165)])]
    tl = Timeline(sources={"s": SourceFile(id="s", path=str(two_tones), duration_seconds=6.0, kind="audio")},
                  items=items, duration_frames=60).model_dump()
    store.save_project(CUT_PROJECT, {"id": CUT_PROJECT, "name": "cut", "source_video": str(two_tones),
                                     "timeline": tl})
    monkeypatch.setattr(audio_live, "project_store", store)
    monkeypatch.setattr(live, "_load_compile_api", lambda: None)
    app = FastAPI()
    app.include_router(audio_live.router, prefix="/api/audio")
    with TestClient(app) as c:
        yield c


def dominant_hz(x: np.ndarray) -> float:
    mono = x[:, 0] * np.hanning(len(x))
    return float(np.argmax(np.abs(np.fft.rfft(mono))) * SR / len(x))


def test_programme_seek_into_the_second_segment_plays_its_tone(cut_client):
    with cut_client.websocket_connect(CUT_URL) as ws:
        p = Player(ws)
        p.send(type="seek", t=1.3)
        p.send(type="play")
        p.grant(SR // 2)
        p.collect(0.6)
    assert not p.errors()
    x = to_float(p.pcm)
    assert len(x) == SR // 2
    assert abs(dominant_hz(x[:SR // 20]) - 880) <= 20          # the first 50 ms are the 880 Hz piece
    playing = [e["t"] for _, e in p.events if e.get("type") == "state" and e.get("playing")]
    assert playing and all(1.3 - 1e-6 <= t <= 1.8 + 1e-6 for t in playing)   # programme time


def test_loop_across_the_cut_plays_both_tones_and_wraps(cut_client):
    with cut_client.websocket_connect(CUT_URL) as ws:
        p = Player(ws)
        p.send(type="loop", a=0.6, b=1.4)
        p.send(type="play")
        p.play_realtime(2.0)
    assert not p.errors()
    x = to_float(p.pcm)
    assert len(x) >= int(1.6 * SR)

    def tone(a, b):
        return dominant_hz(x[int(a * SR):int(b * SR)])

    # programme 0.6-1.0 is 440 Hz, 1.0-1.4 is 880 Hz, then the loop wraps to 0.6
    assert abs(tone(0.05, 0.35) - 440) <= 10
    assert abs(tone(0.45, 0.75) - 880) <= 10
    assert abs(tone(0.85, 1.15) - 440) <= 10
    assert abs(tone(1.25, 1.55) - 880) <= 10


def test_window_in_programme_time_attenuates_exactly_that_interval(cut_client, monkeypatch):
    monkeypatch.setattr(live, "_load_compile_api", _REAL_COMPILE_API)   # windows come from audio.compile
    rack = [{"id": "fx_win", "effect": "amplify", "params": {"gain": -20.0}, "start_s": 0.8, "end_s": 1.2}]
    with cut_client.websocket_connect(CUT_URL) as ws:
        p = Player(ws)
        p.send(type="rack", effects=rack)
        p.send(type="seek", t=0.5)
        p.send(type="play")
        p.play_realtime(1.1)
    assert not p.errors()
    x = to_float(p.pcm)                                        # received audio starts at programme 0.5

    def level(a, b):                                           # programme seconds
        return rms_db(x[int((a - 0.5) * SR):int((b - 0.5) * SR)])

    before, inside, after = level(0.52, 0.78), level(0.82, 1.18), level(1.22, 1.48)
    assert abs((inside - before) + 20.0) < 0.5                 # the window spans the cut
    assert abs(after - before) < 0.5
    block = 240
    env = [rms_db(x[k:k + block]) for k in range(0, int(0.95 * SR), block)]
    low = [k * block / SR + 0.5 for k, v in enumerate(env) if v < before - 10.0]
    assert abs(low[0] - 0.8) <= 0.01 and abs(low[-1] + block / SR - 1.2) <= 0.01
