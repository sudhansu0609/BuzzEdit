"""BuzzEdit -> BuzzAnimate (BuzzAnimate docs/ECOSYSTEM_BRIDGE_PLAN.md step E4).

The routes under `/api/buzzanimate` drive a *fake* BuzzAnimate: a tiny threaded
`http.server` on an ephemeral port that speaks BuzzAnimate's API shape
(`crates/buzz-app/src/api.rs`) and is found through a temp port ledger
(`BUZZCAF_PORTS_FILE`), never the real one. No real BuzzAnimate is started.
"""

import json
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import buzzcaf_ports
from integrations import buzzanimate_client as animate
from store import media_pool
from store.project_store import ProjectStore


class FakeBuzzAnimate:
    """BuzzAnimate's local API, as far as these routes use it."""

    def __init__(self, export_source=None, app_name="buzzanimate"):
        self.export_source = export_source
        self.app_name = app_name
        self.requests = []
        self.open_path = None
        self.jobs = {}
        self.next_id = 1
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _record(self, body=None):
                fake.requests.append({"method": self.command, "path": self.path, "body": body,
                                      "origin": self.headers.get("Origin")})

            def do_GET(self):
                self._record()
                if self.path == "/health":
                    return self._answer(200, {"app": fake.app_name, "status": "ok", "port": fake.port,
                                              "pid": 1, "version": "test"})
                if self.path == "/api/state":
                    return self._answer(200, {"path": fake.open_path, "title": "t", "dirty": False})
                if self.path.startswith("/api/jobs/"):
                    job = fake.jobs.get(int(self.path.rsplit("/", 1)[1]))
                    if not job:
                        return self._answer(404, {"ok": False, "error": "no job"})
                    job["polls"] += 1
                    if job["polls"] >= 2:
                        job["state"] = "done"
                        job["progress"] = {"fraction": 1.0}
                    return self._answer(200, {k: v for k, v in job.items() if k != "polls"})
                return self._answer(404, {"ok": False, "error": f"no route {self.path}"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._record(body)
                if self.path == "/api/call":
                    if body.get("tool") == "quit":
                        return self._answer(409, {"ok": False, "value": None,
                                                  "error": "refused: this quit call would ask the user to confirm"})
                    return self._answer(200, {"ok": True, "value": {"echo": body}, "error": None})
                if self.path == "/api/open":
                    fake.open_path = body["path"]
                    return self._answer(200, {"ok": True, "opening": body["path"]})
                if self.path == "/api/export":
                    if fake.export_source:
                        shutil.copy(fake.export_source, body["path"])
                    return self._answer(200, {"ok": True, "job_id": fake._job("export", body["path"])})
                return self._answer(404, {"ok": False, "error": f"no route {self.path}"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _job(self, kind, output, result=None):
        job_id = self.next_id
        self.next_id += 1
        self.jobs[job_id] = {"id": job_id, "kind": kind, "state": "running",
                             "progress": {"fraction": 0.5}, "output": output, "error": None,
                             "result": result, "polls": 0}
        return job_id

    def publish(self):
        buzzcaf_ports.publish("buzzanimate", self.port, f"{self.url}/health")

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A temp ledger, and a scan that can only find what the test starts."""
    monkeypatch.setenv("BUZZCAF_PORTS_FILE", str(tmp_path / "ports.json"))
    monkeypatch.delenv("BUZZANIMATE_URL", raising=False)
    monkeypatch.setattr(animate, "PREFERRED_PORT", _dead_port())
    buzzcaf_ports._forget()
    yield
    buzzcaf_ports._forget()


@pytest.fixture
def client(tmp_path, monkeypatch, ledger):
    from main import app
    from routes import projects as projects_route
    from routes import timeline as timeline_route
    from store.app_settings import AppSettings

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))
    monkeypatch.setattr(projects_route, "project_store", store)
    monkeypatch.setattr(projects_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(timeline_route, "project_store", store)
    monkeypatch.setattr(timeline_route, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(media_pool, "PROJECTS_DIR", projects_dir)
    isolated = object.__new__(AppSettings)
    isolated._initialized = True
    isolated.settings_dir = tmp_path
    isolated.settings_file = tmp_path / "app_settings.json"
    isolated._cache = isolated._load()
    monkeypatch.setattr(projects_route, "app_settings", isolated)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake():
    server = FakeBuzzAnimate()
    yield server
    server.close()


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    from config import FFMPEG_BIN
    root = tmp_path_factory.mktemp("clips")

    def make(name, extra):
        out = root / name
        result = subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
                                 "testsrc2=size=160x120:rate=24:duration=1", *extra, str(out)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            pytest.skip("ffmpeg unavailable")
        return out

    return {"footage": make("footage.mp4", ["-pix_fmt", "yuv420p"]),
            "render": make("render.mp4", ["-pix_fmt", "yuv420p", "-vf", "hflip"])}


def _wait(client, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/buzzanimate/render/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError(f"render job {job_id} still running")


def test_not_running_is_a_clear_503_that_offers_start(client):
    status = client.get("/api/buzzanimate/status").json()
    assert status["running"] is False
    assert "start: true" in status["message"]

    res = client.post("/api/buzzanimate/call", json={"tool": "new_layer"})
    assert res.status_code == 503
    assert "not running" in res.json()["detail"] and "start: true" in res.json()["detail"]


def test_status_call_and_open_reach_buzzanimate_through_the_ledger(client, fake, tmp_path):
    fake.publish()
    status = client.get("/api/buzzanimate/status").json()
    assert status["running"] is True
    assert status["url"] == fake.url
    assert status["health"]["app"] == "buzzanimate"

    res = client.post("/api/buzzanimate/call", json={"tool": "new_layer", "args": {"name": "Sky"}})
    assert res.status_code == 200, res.text
    assert res.json() == {"ok": True, "value": {"echo": {"tool": "new_layer", "args": {"name": "Sky"}}},
                          "error": None}

    buzz = tmp_path / "scene.buzz"
    buzz.write_text("{}")
    res = client.post("/api/buzzanimate/open", json={"buzz_path": str(buzz)})
    assert res.status_code == 200, res.text
    assert fake.open_path == str(buzz)

    # BuzzAnimate refuses any request carrying a web page's Origin.
    assert all(r["origin"] is None for r in fake.requests)


def test_a_refused_call_keeps_buzzanimates_status_and_reason(client, fake):
    fake.publish()
    res = client.post("/api/buzzanimate/call", json={"tool": "quit"})
    assert res.status_code == 409
    assert "confirm" in res.json()["detail"]


def test_an_impostor_on_the_port_is_not_buzzanimate(client, monkeypatch):
    impostor = FakeBuzzAnimate(app_name="buzzedit")
    try:
        monkeypatch.setattr(animate, "PREFERRED_PORT", impostor.port)
        buzzcaf_ports._forget()
        assert client.get("/api/buzzanimate/status").json()["running"] is False
    finally:
        impostor.close()


def test_start_true_launches_buzzanimate_and_waits_for_its_health(client, tmp_path, monkeypatch):
    launcher = tmp_path / "BuzzAnimate.vbs"
    launcher.write_text("' fake")
    monkeypatch.setattr(animate, "LAUNCHER", launcher)
    started = []

    def fake_launch(path):
        server = FakeBuzzAnimate()
        started.append((path, server))
        server.publish()

    monkeypatch.setattr(animate, "_launch", fake_launch)
    try:
        res = client.post("/api/buzzanimate/call", json={"tool": "new_layer", "start": True, "start_timeout": 20})
        assert res.status_code == 200, res.text
        assert [p for p, _ in started] == [launcher]
    finally:
        for _, server in started:
            server.close()


def test_start_true_without_a_launcher_says_where_it_looked(client, tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "BuzzAnimate.vbs"
    monkeypatch.setattr(animate, "LAUNCHER", missing)
    res = client.post("/api/buzzanimate/open", json={"buzz_path": "x.buzz", "start": True})
    assert res.status_code == 503
    assert str(missing) in res.json()["detail"]


def test_render_exports_an_mp4_and_adds_it_to_the_project(client, tmp_path, clips):
    fake = FakeBuzzAnimate(export_source=clips["render"])
    try:
        fake.publish()
        pid = client.post("/api/projects/import_path", json={"path": str(clips["footage"])}).json()["id"]
        buzz = tmp_path / "walk.buzz"
        buzz.write_text("{}")

        res = client.post("/api/buzzanimate/render", json={"buzz_path": str(buzz), "project_id": pid})
        assert res.status_code == 200, res.text
        job = _wait(client, res.json()["job_id"])
        assert job["status"] == "completed", job
        out = tmp_path / "walk.mp4"
        assert Path(job["result"]["output_path"]) == out and out.is_file()

        # Open first, then the export, as MP4, beside the .buzz.
        posts = [r for r in fake.requests if r["method"] == "POST"]
        assert [r["path"] for r in posts] == ["/api/open", "/api/export"]
        assert posts[1]["body"] == {"path": str(out), "format": "mp4"}

        media = client.get(f"/api/projects/{pid}/media").json()["media"]
        assert "walk.mp4" in [m["name"] for m in media]

        # A second render of the same file updates it in place.
        again = _wait(client, client.post("/api/buzzanimate/render",
                                          json={"buzz_path": str(buzz), "project_id": pid}).json()["job_id"])
        assert again["status"] == "completed", again
    finally:
        fake.close()


def test_render_checks_its_inputs_before_starting_a_job(client, fake, tmp_path):
    fake.publish()
    assert client.post("/api/buzzanimate/render", json={"buzz_path": str(tmp_path / "none.buzz")}).status_code == 404
    buzz = tmp_path / "a.buzz"
    buzz.write_text("{}")
    assert client.post("/api/buzzanimate/render", json={"buzz_path": str(buzz), "format": "avi"}).status_code == 400
    assert client.post("/api/buzzanimate/render",
                       json={"buzz_path": str(buzz), "project_id": "nope"}).status_code == 404
    assert client.get("/api/buzzanimate/render/nope").status_code == 404


def test_only_this_machine_may_reach_buzzanimate_through_buzzedit(client, fake):
    """BuzzEdit's CORS lets any page call it, so the BuzzAnimate routes check
    the Origin themselves: a website, or a sandboxed iframe's `null`, is
    refused before anything reaches BuzzAnimate; a backend (no Origin) and a
    page on this machine are served."""
    fake.publish()
    for origin in ["https://evil.example", "null", "http://localhost.evil.example", "http://127.0.0.1.evil.example:8099"]:
        res = client.post("/api/buzzanimate/call", json={"tool": "new_layer"}, headers={"Origin": origin})
        assert res.status_code == 403, (origin, res.text)
    assert fake.requests == [] or all(r.get("path") != "/api/call" for r in fake.requests), "nothing reached BuzzAnimate"
    for origin in [None, "http://localhost:5173", "http://127.0.0.1:8099"]:
        headers = {"Origin": origin} if origin else {}
        assert client.get("/api/buzzanimate/status", headers=headers).status_code == 200, origin
