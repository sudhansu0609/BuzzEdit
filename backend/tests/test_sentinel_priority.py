"""BuzzEdit's work registered with Sentinel as `interactive` while it runs,
so a render is no longer the first thing frozen or killed under memory
pressure (ffmpeg was batch priority 10; the backend and ComfyUI matched no
rule). See runtime/sentinel_priority.py."""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import sentinel_priority as sp  # noqa: E402


@pytest.fixture()
def wire(monkeypatch):
    sent = []
    done = threading.Event()

    def fake_send(command):
        sent.append(command)
        done.set()
        return "Ok"

    monkeypatch.setattr(sp, "_send", fake_send)
    monkeypatch.setattr(sp, "_held", {})
    monkeypatch.setattr(sp, "_active_jobs", 0)
    monkeypatch.setattr(sp, "listening_pid", lambda port: 4321)
    return sent, done


def _registered(sent):
    return {c["Register"]["client"]: c["Register"] for c in sent if "Register" in c}


def _wait_for(predicate, timeout=3.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        sp._wake.set()
        time.sleep(0.02)
    return False


def test_a_job_registers_the_backend_and_comfyui_as_interactive(wire):
    sent, _ = wire
    with sp.job():
        assert _wait_for(lambda: {sp.BACKEND, sp.COMFYUI} <= set(_registered(sent)))
        regs = _registered(sent)
        assert regs[sp.COMFYUI]["pid"] == 4321
        assert {r["class"] for r in regs.values()} == {"interactive"}
    assert sp._held == {}
    assert _wait_for(lambda: {"Unregister": {"client": sp.BACKEND}} in sent)


def test_overlapping_jobs_keep_the_registration_until_the_last_ends(wire):
    with sp.job():
        with sp.job():
            pass
        assert sp.BACKEND in sp._held
    assert sp.BACKEND not in sp._held


def test_an_ffmpeg_pass_is_held_while_it_runs(wire):
    sent, _ = wire
    with sp.ffmpeg(777):
        assert _wait_for(lambda: "buzzedit-ffmpeg-777" in _registered(sent))
    assert "buzzedit-ffmpeg-777" not in sp._held


def test_the_payload_stays_v1_compatible(wire):
    # A v1 service rejects a Register carrying keys it does not know.
    sent, _ = wire
    sp.hold("x", 1)
    assert _wait_for(lambda: "x" in _registered(sent))
    assert set(_registered(sent)["x"]) == {"client", "pid", "class", "budget_mib"}
    sp.drop("x")


def test_no_sentinel_changes_nothing(monkeypatch):
    monkeypatch.setattr(sp, "PIPE", r"\\.\pipe\no-such-sentinel-pipe")
    assert sp._send({"Register": {}}) is None
