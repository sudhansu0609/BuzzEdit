"""runtime/code_reload.py: a changed backend moves onto its new code only when
it is idle, and only under the supervisor that will start it again."""
import asyncio
import os
import time

import pytest

from runtime import code_reload


def test_code_mtime_sees_a_changed_source_and_skips_tests(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    src = tmp_path / "pkg" / "a.py"
    src.write_text("x = 1\n")
    old = time.time() - 100
    os.utime(src, (old, old))
    before = code_reload.code_mtime(tmp_path)
    test = tmp_path / "tests" / "test_a.py"
    test.write_text("")
    assert code_reload.code_mtime(tmp_path) == before
    src.write_text("x = 2\n")
    assert code_reload.code_mtime(tmp_path) > before


def test_supervised_only_under_electron(monkeypatch):
    monkeypatch.delenv("BUZZEDIT_SUPERVISED", raising=False)
    monkeypatch.delenv("BUZZEDIT_PORT", raising=False)
    assert not code_reload.supervised()
    monkeypatch.setenv("BUZZEDIT_SUPERVISED", "1")
    assert code_reload.supervised()


def test_inflight_counts_requests_but_not_media_streams():
    seen = []

    async def app(scope, receive, send):
        seen.append(code_reload._inflight)

    mw = code_reload.InflightMiddleware(app)
    asyncio.run(mw({"type": "http", "path": "/api/presentation/x"}, None, None))
    asyncio.run(mw({"type": "http", "path": "/api/media/x/stream"}, None, None))
    assert seen == [1, 0] and code_reload._inflight == 0


def _fake_world(monkeypatch, busy):
    monkeypatch.setattr(code_reload, "CHECK_S", 0.01)
    monkeypatch.setattr(code_reload, "QUIET_S", 0.03)
    monkeypatch.setattr(code_reload, "_exit_backstop", lambda: None)
    stamps = iter([1.0] + [2.0] * 10_000)
    monkeypatch.setattr(code_reload, "code_mtime", lambda root=None: next(stamps))
    monkeypatch.setattr(code_reload, "busy_reason", busy)
    monkeypatch.setenv("BUZZEDIT_SUPERVISED", "1")


def test_changed_code_exits_once_idle(monkeypatch):
    calls = {"n": 0}

    def busy(route_tasks=0):
        calls["n"] += 1
        return "scheduler job job_1" if calls["n"] < 3 else None

    _fake_world(monkeypatch, busy)
    exited = []
    asyncio.run(asyncio.wait_for(code_reload.watch(lambda: exited.append(True)), 5))
    assert exited == [True] and calls["n"] >= 3


def test_never_exits_while_busy(monkeypatch):
    _fake_world(monkeypatch, lambda route_tasks=0: "2 request(s) open")
    exited = []
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(code_reload.watch(lambda: exited.append(True)), 0.5))
    assert exited == []


def test_unsupervised_backend_only_logs(monkeypatch):
    _fake_world(monkeypatch, lambda route_tasks=0: None)
    monkeypatch.delenv("BUZZEDIT_SUPERVISED", raising=False)
    monkeypatch.delenv("BUZZEDIT_PORT", raising=False)
    exited = []
    asyncio.run(asyncio.wait_for(code_reload.watch(lambda: exited.append(True)), 5))
    assert exited == []


def test_a_pending_reload_is_reported_until_the_restart(monkeypatch):
    # The Studio's queue reads this off /api/health and waits for the restart
    # before its next item: the queue itself never leaves BuzzEdit idle.
    monkeypatch.setattr(code_reload, "_pending", False)
    _fake_world(monkeypatch, lambda route_tasks=0: "2 request(s) open")
    assert not code_reload.reload_pending()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(code_reload.watch(lambda: None), 0.5))
    assert code_reload.reload_pending()
    assert "/api/health" in code_reload._IGNORED_PATHS
