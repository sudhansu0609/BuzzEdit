"""The BuzzEdit side of the master pause / resume
(BuzzcafStudio docs/STUDIO_DIRECTOR_PLAN.md Part 1):
`runtime/master_control.py` -- durable state, the async hold point, the
scheduler's own pause flag, "now" mode's ComfyUI interrupt, and never
quitting BuzzEdit."""
import asyncio
import threading
import time

import pytest

from runtime import master_control


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Never touch the real `data/master_control.json`."""
    monkeypatch.setattr(master_control, "STATE_PATH", tmp_path / "master_control.json")
    master_control._state_cache = None
    master_control._held.clear()
    yield
    if master_control.is_paused():
        master_control.resume()
    master_control._state_cache = None
    master_control._held.clear()


@pytest.fixture(autouse=True)
def _fake_scheduler(monkeypatch):
    """A tiny stand-in for `routes.scheduler.scheduler_instance` -- the real
    one starts an asyncio worker loop this suite does not want running."""
    class _Sched:
        is_paused = False

    import routes.scheduler as scheduler_module
    sched = _Sched()
    monkeypatch.setattr(scheduler_module, "scheduler_instance", sched)
    return sched


def test_resume_without_a_pause_is_a_noop_and_never_blocks():
    """Regression: `resume()` on an already-unpaused state once called
    `state()` (which re-acquires `_lock`) while still holding it -- a plain
    `threading.Lock` is not reentrant, so that call never returned."""
    st = master_control.resume()
    assert st["paused"] is False


def test_pause_resume_state_durable_across_reload():
    master_control.pause(mode="soft", reason="testing", by="tester")
    assert master_control.is_paused() is True

    master_control._state_cache = None  # simulate a restart: only the file remains
    st = master_control.state()
    assert st["paused"] is True
    assert st["reason"] == "testing"
    assert st["by"] == "tester"
    assert st["mode"] == "soft"

    master_control.resume()
    master_control._state_cache = None
    assert master_control.is_paused() is False


def test_pause_never_quits_buzzedit(monkeypatch):
    """Pausing must never call anything that stops/exits the process."""
    killed = []
    monkeypatch.setattr("os._exit", lambda code: killed.append(code), raising=False)
    master_control.pause(mode="now", reason="safety check")
    master_control.resume()
    assert killed == []


def test_pause_sets_scheduler_flag_and_resume_clears_it(_fake_scheduler):
    assert _fake_scheduler.is_paused is False
    master_control.pause(mode="soft")
    assert _fake_scheduler.is_paused is True
    master_control.resume()
    assert _fake_scheduler.is_paused is False


def test_pause_does_not_clear_a_pre_existing_scheduler_pause(_fake_scheduler):
    _fake_scheduler.is_paused = True  # the creator paused the scheduler by hand first
    master_control.pause(mode="soft")
    master_control.resume()
    assert _fake_scheduler.is_paused is True


def test_now_mode_interrupts_comfyui(monkeypatch):
    calls = []
    monkeypatch.setattr(master_control, "_interrupt_comfyui_now", lambda: calls.append("interrupt"))
    master_control.pause(mode="soft")
    assert calls == []  # soft never interrupts
    master_control.pause(mode="now")
    assert calls == ["interrupt"]


def test_release_memory_is_called_on_pause(monkeypatch):
    calls = []

    async def fake_offload_all(reason):
        calls.append(reason)

    from runtime import gpu_handover
    monkeypatch.setattr(gpu_handover, "offload_all", fake_offload_all)
    master_control.pause(mode="soft", reason="release test")
    # _release_memory fires the coroutine as a task (there is no running
    # loop in a plain sync test), so give it a moment to run.
    deadline = time.time() + 2
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert calls and "master pause" in calls[0]


def test_wait_if_paused_blocks_and_returns_on_resume(monkeypatch):
    monkeypatch.setattr(master_control, "POLL_S", 0.02)
    master_control.pause(mode="soft")
    finished = threading.Event()

    def worker():
        asyncio.run(master_control.wait_if_paused_async("unit-test-shot"))
        finished.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + 2
    while "unit-test-shot" not in master_control.held() and time.time() < deadline:
        time.sleep(0.01)
    assert "unit-test-shot" in master_control.held()
    assert not finished.is_set()

    master_control.resume()
    assert finished.wait(timeout=2)
    assert "unit-test-shot" not in master_control.held()


def test_sync_wait_if_paused_blocks_and_returns_on_resume(monkeypatch):
    """The thread-run (narration) variant."""
    monkeypatch.setattr(master_control, "POLL_S", 0.02)
    master_control.pause(mode="soft")
    finished = threading.Event()

    def worker():
        master_control.wait_if_paused("unit-test-narration-job")
        finished.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + 2
    while "unit-test-narration-job" not in master_control.held() and time.time() < deadline:
        time.sleep(0.01)
    assert "unit-test-narration-job" in master_control.held()
    assert not finished.is_set()

    master_control.resume()
    assert finished.wait(timeout=2)
