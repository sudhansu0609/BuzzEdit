"""Leaving the shared port ledger on the way down (GUARDIAN_PLAN.md section 11).

Publishing is well covered by `scripts/buzzcaf-ports.test.mjs`; *withdrawing* was
not, and the P4 integration run found the hole. The backend stopped gracefully -
uvicorn logged "Application shutdown complete" - and `buzzedit -> 8100` was still
in `%LOCALAPPDATA%\\Buzzcaf\\ports.json` afterwards, naming a pid that no longer
existed. Every consumer then had to find that entry, probe it and throw it away.

The cause is a Windows one. A console control event (Ctrl+Break, closing the
console window, the machine shutting down) ends the process from the CRT's
default handler as soon as uvicorn's own handler returns, so `atexit` - which is
where the withdrawal lived - never runs, and neither does the `finally` around
`server.run()`. FastAPI's lifespan shutdown runs *before* that, so that is where
the withdrawal belongs, with `atexit` kept only as a backstop.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import buzzcaf_ports  # noqa: E402
import main as backend_main  # noqa: E402
from config import APP_NAME  # noqa: E402


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ports.json"
    monkeypatch.setenv("BUZZCAF_PORTS_FILE", str(path))
    buzzcaf_ports._forget()
    yield path
    buzzcaf_ports._forget()


def test_the_lifespan_shutdown_withdraws_the_entry(ledger, monkeypatch):
    buzzcaf_ports.publish(APP_NAME, 8100, "http://127.0.0.1:8100/api/health")
    assert buzzcaf_ports.entry(APP_NAME)

    # As `_serve()` leaves it when it published for itself.
    monkeypatch.setattr(backend_main, "_owns_ledger_entry", True)
    with TestClient(backend_main.app):
        pass  # entering runs lifespan startup, leaving runs the shutdown

    assert buzzcaf_ports.entry(APP_NAME) is None, (
        "a graceful shutdown left buzzedit in the port ledger"
    )


def test_a_backend_started_by_electron_leaves_the_entry_alone(ledger, monkeypatch):
    """Under Electron the launcher owns the entry, because only it knows which
    ComfyUI belongs to this backend. Withdrawing here would delete a record we
    did not write and hide a running app from everybody."""
    buzzcaf_ports.publish(APP_NAME, 8100, "http://127.0.0.1:8100/api/health",
                          {"comfyui": 8188})

    monkeypatch.setattr(backend_main, "_owns_ledger_entry", False)
    with TestClient(backend_main.app):
        pass

    record = buzzcaf_ports.entry(APP_NAME)
    assert record and record["extra"] == {"comfyui": 8188}, (
        "the backend withdrew an entry that belongs to the launcher"
    )


def test_withdrawing_twice_is_harmless(ledger, monkeypatch):
    """`atexit` is still registered as a backstop, so both paths can fire."""
    buzzcaf_ports.publish(APP_NAME, 8100, "http://127.0.0.1:8100/api/health")
    monkeypatch.setattr(backend_main, "_owns_ledger_entry", True)
    backend_main._withdraw_ledger_entry()
    backend_main._withdraw_ledger_entry()
    assert buzzcaf_ports.entry(APP_NAME) is None
