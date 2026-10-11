"""Voice conversion inside BuzzEdit (VOICE_CHANGER_PLAN.md, brief S1).

`engines/seedvc/cli.py serve` speaks its protocol under this -- BuzzEdit's -- venv in --dry-run
(no model); the `seedvc_env.Worker` runs a stub serve script that answers every convert with a
330 Hz sine at 22.05 kHz (seed-vc's rate) as long as the input, so a converted stretch is easy to
tell from the white-noise "speech" around it. Projects, the voices library and the job registry
live in tmp_path. No GPU, no model.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from asr import script_align
from audio import noise, processes, seedvc_env, voice_conversion
from runtime import gpu_handover
from store.project_store import ProjectStore
from timeline.schema import SourceFile, Timeline, TimelineItem, WordItem

CLI = Path(__file__).resolve().parents[2] / "engines" / "seedvc" / "cli.py"
RATE = 48000
FPS = 30
HALF = int(round(processes.FADE_S * RATE / 2.0))      # the lane's crossfade half-width

STUB_SERVE = textwrap.dedent('''
    import json, sys, time
    import numpy as np
    import soundfile as sf

    args = sys.argv[1:]
    assert args[0] == "serve", args
    folder = args[args.index("--seedvc-dir") + 1]
    log = open(folder + "/jobs.jsonl", "a", encoding="utf-8")
    sys.stderr.write(json.dumps({"event": "ready", "device": "cuda", "load_s": 0.25}) + "\\n")
    sys.stderr.flush()
    for line in sys.stdin:
        job = json.loads(line)
        log.write(json.dumps(job) + "\\n")
        log.flush()
        op, jid = job.get("op"), job.get("id")
        if op == "quit":
            print(json.dumps({"id": jid, "ok": True, "op": "quit", "out": None}), flush=True)
            break
        if op == "ping":
            print(json.dumps({"id": jid, "ok": True, "op": "ping", "out": None, "device": "cuda",
                              "load_s": 0.25}), flush=True)
            continue
        if op == "convert":
            sys.stderr.write(json.dumps({"event": "progress", "fraction": 0.5, "stage": "convert",
                                         "job": jid}) + "\\n")
            sys.stderr.flush()
            time.sleep(0.2)          # stdout and stderr are separate pipes: let the progress land first
            info = sf.info(job["source"])
            n = int(round(info.frames * 22050 / info.samplerate))
            t = np.arange(n) / 22050.0
            sf.write(job["out"], (0.3 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32), 22050,
                     subtype="FLOAT")
            print(json.dumps({"id": jid, "ok": True, "op": "convert", "out": job["out"], "seconds": 0.01,
                              "device": "cuda", "sr": 22050}), flush=True)
            continue
        print(json.dumps({"id": jid, "ok": False, "op": op, "error": "stub: unknown op"}), flush=True)
''')


def _noise(seconds: float, sr: int = RATE, seed: int = 0, level: float = 0.1) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(-1.0, 1.0, int(round(seconds * sr))) * level).astype(np.float32)


def _peak_hz(samples: np.ndarray, sr: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / sr)[int(np.argmax(spectrum))])


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture
def library(tmp_path, monkeypatch):
    """data/ in tmp_path: the voices library starts empty."""
    monkeypatch.setattr(seedvc_env, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / "voices"


@pytest.fixture
def stub_worker(tmp_path, monkeypatch, library):
    """A Worker on the stub serve script, installed as seedvc_env.WORKER; GPU calls recorded."""
    engine = tmp_path / "engine"
    engine.mkdir()
    stub = engine / "stub_serve.py"
    stub.write_text(STUB_SERVE, encoding="utf-8")
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": sys.executable, "seedvc_dir": str(engine), "cli": str(stub), "source": "test", "reason": ""})
    calls = []

    async def prepare(name, need_vram_mb, need_ram_mb, settle_s=3.0):
        calls.append(("prepare", name, need_vram_mb, need_ram_mb))
        return {"phase": name, "ready": True}

    async def offload(reason="task finished"):
        calls.append(("offload", reason))

    monkeypatch.setattr(gpu_handover, "prepare_for_phase", prepare)
    monkeypatch.setattr(gpu_handover, "offload_all", offload)
    worker = seedvc_env.Worker(idle_stop_s=600.0, start_timeout_s=60.0)
    monkeypatch.setattr(seedvc_env, "WORKER", worker)

    def jobs():
        path = engine / "jobs.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    yield SimpleNamespace(worker=worker, calls=calls, engine=engine, jobs=jobs)
    worker.stop(wait_s=5.0)


def _voice_clip(path: Path, seconds: float, sr: int = 16000, seed: int = 1) -> Path:
    sf.write(str(path), _noise(seconds, sr, seed, 0.2), sr, subtype="PCM_16")
    return path


SCRIPT = textwrap.dedent('''\
    NARRATOR: ek do teen char

    MAYA: paanch chhe saat

    NARRATOR: aath nau das

    RAVI (V.O.)
    gyarah barah terah

    **V.O. (Narrator):** chaudah pandrah
''')
# (speaker, words, start_s, end_s): every word sits on whole frames at 30 fps
PARAGRAPHS = [
    (None, ["ek", "do", "teen", "char"], 0.5, 2.5),
    ("MAYA", ["paanch", "chhe", "saat"], 3.0, 5.0),
    (None, ["aath", "nau", "das"], 5.5, 7.5),
    ("RAVI", ["gyarah", "barah", "terah"], 8.0, 10.0),
    (None, ["chaudah", "pandrah"], 10.4, 11.4),
]
DURATION_S = 12.0


def _words():
    items = []
    for _speaker, words, start, end in PARAGRAPHS:
        step = (end - start) / len(words)
        for k, text in enumerate(words):
            a, b = start + k * step, start + (k + 1) * step
            items.append(WordItem(id=f"w{len(items)}", text=text, start_frame=int(round(a * FPS)),
                                  end_frame=int(round(b * FPS))))
    return items


@pytest.fixture
def project(tmp_path, monkeypatch, library):
    """A 12 s white-noise "recording", the script above, its words, two voices in the library."""
    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    recording = tmp_path / "recording.wav"
    sf.write(str(recording), _noise(DURATION_S, seed=7), RATE, subtype="FLOAT")
    frames = int(DURATION_S * FPS)
    tl = Timeline(
        fps_num=FPS, fps_den=1,
        sources={"main": SourceFile(id="main", path=str(recording), duration_seconds=DURATION_S,
                                    kind="audio", width=0, height=0)},
        words=_words(),
        items=[TimelineItem(id="a1_0", track="A1", source_id="main", source_start_frame=0,
                            source_end_frame=frames, timeline_start_frame=0, timeline_end_frame=frames)],
        duration_frames=frames, revision=3)
    store.save_project("p1", {"id": "p1", "name": "Synthetic story", "timeline": tl.model_dump(),
                              "script": {"text": SCRIPT}, "settings": {}})
    voice_conversion.add_voice("maya_voice", _voice_clip(tmp_path / "maya.wav", 4.0, seed=2), "woman", "test")
    voice_conversion.add_voice("ravi_voice", _voice_clip(tmp_path / "ravi.wav", 5.0, seed=3), "demon", "test")
    original, _ = sf.read(str(recording), dtype="float32")
    return SimpleNamespace(store=store, recording=recording, original=original, dir=projects / "p1")


def _source(store: ProjectStore) -> SourceFile:
    return Timeline.model_validate(store.get_project("p1")["timeline"]).sources["main"]


def _samples(seconds: float) -> int:
    return int(round(seconds * RATE))


def _assert_converted_only(version: np.ndarray, original: np.ndarray, spans):
    """Outside every span (and its crossfade) sample for sample the original; inside, the stub's
    330 Hz sine in place of the noise."""
    assert version.shape == original.shape
    untouched = np.ones(original.size, dtype=bool)
    for start_s, end_s in spans:
        a, b = _samples(start_s), _samples(end_s)
        untouched[a - HALF:b + HALF] = False
        inside = version[a + HALF:b - HALF]
        assert not np.array_equal(inside, original[a + HALF:b - HALF])
        assert abs(_peak_hz(inside) - 330.0) < 3.0
        assert np.sqrt(np.mean(inside ** 2)) == pytest.approx(0.3 / np.sqrt(2.0), rel=0.05)
    assert np.array_equal(version[untouched], original[untouched])


# --- cli.py serve, dry run under BuzzEdit's venv ---------------------------------------------

def test_cli_serve_dry_run_answers_one_line_per_job(tmp_path):
    seedvc = tmp_path / "seed-vc"
    seedvc.mkdir()
    (seedvc / "seed_vc_wrapper.py").write_text("raise SystemExit('a dry run must not import seed-vc')\n")
    source, ref = tmp_path / "line.wav", tmp_path / "ref.wav"
    sf.write(str(source), _noise(2.0, 16000), 16000)
    sf.write(str(ref), _noise(4.0, 16000, seed=1), 16000)
    jobs = [
        {"op": "ping", "id": "p"},
        {"op": "convert", "id": 7, "source": str(source), "ref": str(ref), "out": str(tmp_path / "o.wav"),
         "params": {"diffusion_steps": 30, "f0_condition": True, "pitch_shift": -5}},
        {"op": "apply-edit", "id": "e", "source": str(source), "out": str(tmp_path / "e.wav"),
         "manifest": {"duration": 2.0, "regions": [{"id": "r1", "start": 0.5, "end": 1.0, "kind": "convert",
                                                     "voice": str(ref)}]}},
        "not json",
        {"op": "convert", "id": "bad", "source": str(source)},
        {"op": "dance", "id": "d"},
        {"op": "quit", "id": "q"},
    ]
    stdin = "".join((job if isinstance(job, str) else json.dumps(job)) + "\n" for job in jobs)
    proc = subprocess.run([sys.executable, str(CLI), "serve", "--dry-run", "--seedvc-dir", str(seedvc)],
                          input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=120,
                          env={**os.environ, "HF_HUB_CACHE": str(tmp_path / "hf")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    answers = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [a["id"] for a in answers] == ["p", 7, "e", 4, "bad", "d", "q"]
    assert [a["ok"] for a in answers] == [True, True, True, False, False, False, True]
    ping, convert, edit, bad_json, missing, unknown, quit_ = answers
    assert ping["op"] == "ping" and ping["loaded"] is False and ping["device"] is None
    assert convert["op"] == "convert" and convert["params"] == {
        "diffusion_steps": 30, "length_adjust": 1.0, "inference_cfg_rate": 0.7, "f0_condition": True,
        "auto_f0_adjust": True, "pitch_shift": -5}
    assert convert["out"] == str((tmp_path / "o.wav").resolve()) and "seconds" in convert and "device" in convert
    assert edit["op"] == "apply-edit" and [s["kind"] for s in edit["spans"]] == ["keep", "convert", "keep"]
    assert "not valid JSON" in bad_json["error"]
    assert "needs 'ref'" in missing["error"]
    assert "unknown op" in unknown["error"]
    assert quit_["op"] == "quit" and quit_["jobs"] == 6
    assert not (tmp_path / "o.wav").exists() and not (tmp_path / "e.wav").exists()
    events = [json.loads(line[line.find('{"event"'):]) for line in proc.stderr.splitlines() if '{"event"' in line]
    assert events[0]["event"] == "ready" and events[0]["dry_run"] is True
    assert any(e["event"] == "progress" and e.get("job") == 7 for e in events)
    assert events[-1]["event"] == "done"


def test_cli_serve_without_seed_vc_exits_2_with_one_line(tmp_path):
    proc = subprocess.run([sys.executable, str(CLI), "serve", "--seedvc-dir", str(tmp_path / "nowhere")],
                          input='{"op": "ping"}\n', capture_output=True, text=True, encoding="utf-8",
                          timeout=120)
    assert proc.returncode == 2
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0]["ok"] is False and lines[0]["exit_code"] == 2


# --- the worker -------------------------------------------------------------------------------

def test_worker_starts_once_converts_and_stops_with_the_gpu_policy(stub_worker, tmp_path):
    worker = stub_worker.worker
    assert worker.status() == {"running": False, "device": None, "idle_s": None, "busy": False,
                               "load_s": None, "pid": None, "jobs": 0}
    source = tmp_path / "src.wav"
    sf.write(str(source), _noise(1.0), RATE, subtype="FLOAT")
    seen = []
    first = worker.convert(source, source, tmp_path / "a.wav", {"diffusion_steps": 10},
                           progress=lambda f, s: seen.append((f, s)))
    pid = worker.status()["pid"]
    second = worker.convert(source, source, tmp_path / "b.wav")
    status = worker.status()
    assert first["ok"] and second["ok"] and first["device"] == "cuda"
    assert status["running"] and status["device"] == "cuda" and status["pid"] == pid and status["jobs"] == 2
    assert status["load_s"] == 0.25 and status["idle_s"] >= 0.0
    assert seen == [(0.5, "convert")]
    assert sf.info(str(tmp_path / "a.wav")).samplerate == 22050
    assert [job["op"] for job in stub_worker.jobs()] == ["ping", "convert", "convert"]
    assert stub_worker.jobs()[1]["params"] == {"diffusion_steps": 10}
    assert stub_worker.calls == [("prepare", "voice_conversion", 6000, 8000)]      # one start, one hand-over

    stopped = worker.stop()
    assert stopped["running"] is False and stub_worker.jobs()[-1]["op"] == "quit"
    assert stub_worker.calls[-1] == ("offload", "voice conversion worker stopped")


def test_worker_stops_itself_when_idle(stub_worker, tmp_path):
    worker = stub_worker.worker
    worker.idle_stop_s = 0.5
    source = tmp_path / "src.wav"
    sf.write(str(source), _noise(0.5), RATE, subtype="FLOAT")
    worker.convert(source, source, tmp_path / "a.wav")
    deadline = time.monotonic() + 10.0
    while worker.running() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not worker.running()
    assert stub_worker.calls[-1] == ("offload", "voice conversion worker idle")
    worker.convert(source, source, tmp_path / "b.wav")          # the next job starts it again
    assert worker.running()
    assert [c[0] for c in stub_worker.calls] == ["prepare", "offload", "prepare"]


def test_worker_refuses_to_start_while_a_render_runs(stub_worker, monkeypatch, tmp_path):
    from routes import rendering
    monkeypatch.setitem(rendering.jobs, "render_x", {"id": "render_x", "job_type": "render", "status": "running"})
    with pytest.raises(seedvc_env.SeedVCBusy):
        stub_worker.worker.ensure()
    with pytest.raises(seedvc_env.SeedVCBusy):
        stub_worker.worker.convert(tmp_path / "x.wav", tmp_path / "y.wav", tmp_path / "z.wav")
    assert stub_worker.calls == [] and not stub_worker.worker.running()


def test_a_failed_job_keeps_the_worker_and_a_dead_one_is_restarted(stub_worker, tmp_path):
    worker = stub_worker.worker
    with pytest.raises(seedvc_env.SeedVCError, match="stub: unknown op"):
        worker._job({"op": "dance"}, None, 30.0)
    assert worker.running()
    worker._proc.kill()
    worker._proc.wait(timeout=10)
    source = tmp_path / "src.wav"
    sf.write(str(source), _noise(0.5), RATE, subtype="FLOAT")
    assert worker.convert(source, source, tmp_path / "a.wav")["ok"]
    assert [c[0] for c in stub_worker.calls] == ["prepare", "offload", "prepare"]


# --- speaker labels ---------------------------------------------------------------------------

def test_the_four_label_forms_give_speakers_and_are_not_spoken():
    parsed = script_align.parse_script(textwrap.dedent('''\
        MAYA (V.O.)
        Kaun hai wahan?

        **V.O. (Narrator):** Koi jawab nahi aaya.
        HOST: Aur phir...
        RAVI: Main hoon.
        Socho: agar woh tum hote?
        VISUAL: a dark corridor

        MAYA:

        Bhaago!
        Ek aur line.
    '''))
    got = [(p["speaker"], p["text"]) for p in parsed["paragraphs"]]
    assert got == [
        ("MAYA", "Kaun hai wahan?"),
        (None, "Koi jawab nahi aaya."),
        (None, "Aur phir..."),
        ("RAVI", "Main hoon. Socho: agar woh tum hote? VISUAL: a dark corridor"),
        ("MAYA", "Bhaago! Ek aur line."),                       # a label alone speaks the next block
    ]
    assert script_align.speaker_label("**Maya:** Ruko") == ("Maya", "Ruko")       # emphasised: any case
    assert script_align.speaker_label("NARRATOR (V.O.)") == (None, "")
    assert script_align.speaker_label("TEXT OVERLAY: 40%") is None
    assert script_align.speaker_label("Socho: agar") is None
    plain = script_align.parse_script("ek do teen\n\nchar paanch")
    assert [p["speaker"] for p in plain["paragraphs"]] == [None, None]


def test_speaker_spans_merge_the_same_speaker_and_skip_unheard_paragraphs():
    parsed = script_align.parse_script("MAYA: ek do\n\nMAYA: teen char\n\nNARRATOR: paanch chhe\n\n"
                                       "RAVI: kabhi nahi bola\n\nsaat aath")
    words = [{"text": t, "start": s, "end": s + 0.4} for t, s in
             [("ek", 0.0), ("do", 0.5), ("teen", 1.2), ("char", 1.7), ("paanch", 3.0), ("chhe", 3.5),
              ("saat", 5.0), ("aath", 5.5)]]
    spans = script_align.speaker_spans(parsed, words)
    # MAYA's two paragraphs merge; RAVI's line was never heard: left out, and the narrator's
    # spans either side of it stay apart
    assert [(s["speaker"], round(s["start_s"], 3), round(s["end_s"], 3)) for s in spans] == [
        ("MAYA", 0.0, 2.1), (None, 3.0, 3.9), (None, 5.0, 5.9)]
    assert spans[0]["text"] == "ek do teen char"
    assert set(spans[0]) == {"speaker", "start_s", "end_s", "text"}


# --- voices library and presets ---------------------------------------------------------------

def test_presets_expose_fx_racks_of_existing_catalogue_ids_only(monkeypatch, library):
    from audio import catalogue
    got = voice_conversion.presets()["presets"]
    assert set(got) == {"Woman", "Child", "Demon", "Monster", "Ghost"}
    for name, preset in got.items():
        assert preset["engine"]["length_adjust"] == 1.0 and preset["voice"]
        for effect in preset["fx_rack"]:
            assert effect["effect"] in catalogue.EFFECTS
            assert set(effect) == {"id", "effect", "params", "enabled", "start_s", "end_s", "preset"}
            keys = {p["key"] for p in catalogue.EFFECTS[effect["effect"]]["params"]}
            assert set(effect["params"]) <= keys
    assert [e["effect"] for e in got["Demon"]["fx_rack"]] == ["pitch_shifter", "distortion", "reverb", "sub_octave"]
    demon = {e["effect"]: e["params"] for e in got["Demon"]["fx_rack"]}
    assert demon == {"pitch_shifter": {"semitones": -4.0}, "distortion": {"drive": 10.5},
                     "reverb": {"room_size": 40.0, "wet": 20.0}, "sub_octave": {"mix": 50.0}}
    assert [e["effect"] for e in got["Woman"]["fx_rack"]] == ["reverb"]
    assert got["Demon"]["params"] == {"preset": "Demon", "match_pitch": True, "pitch_shift": -5,
                                      "steps": 30, "cfg_rate": 0.7}

    trimmed = {k: v for k, v in catalogue.EFFECTS.items() if k != "sub_octave"}
    monkeypatch.setattr(catalogue, "EFFECTS", trimmed)
    assert "sub_octave" not in [e["effect"] for e in voice_conversion.presets()["presets"]["Demon"]["fx_rack"]]


def test_voice_conversion_is_a_process_with_library_choices(library, tmp_path):
    definition = processes.PROCESSES["voice_conversion"]
    assert definition["engine"] == "process" and definition["windowable"] is True
    assert [p["key"] for p in definition["params"]] == ["voice", "preset", "match_pitch", "pitch_shift",
                                                        "steps", "cfg_rate"]
    with pytest.raises(ValueError):
        processes.resolve_params("voice_conversion", {"voice": "maya_voice"})
    voice_conversion.add_voice("maya_voice", _voice_clip(tmp_path / "m.wav", 3.5), "woman", "mine")
    values = processes.resolve_params("voice_conversion", {"voice": "maya_voice", "steps": 99, "pitch_shift": -20})
    assert values["steps"] == 50.0 and values["pitch_shift"] == -12.0 and values["preset"] == "custom"
    voice, params = voice_conversion.engine_params({**values, "preset": "Demon", "voice": ""})
    assert voice == "demon" and params["length_adjust"] == 1.0 and params["auto_f0_adjust"] is True
    assert "maya_voice" in processes.listed_processes()["voice_conversion"]["params"][0]["choices"]


@pytest.fixture
def voice_client(library):
    from routes import audio_voice
    app = FastAPI()
    app.include_router(audio_voice.router, prefix="/api/audio")
    return TestClient(app)


def _upload(client, name, path, kind="woman", consent="recorded with consent"):
    with open(path, "rb") as handle:
        return client.post("/api/audio/voices", data={"name": name, "kind": kind, "consent": consent},
                           files={"file": (path.name, handle, "audio/wav")})


def test_voices_api_round_trip_and_length_refusals(voice_client, tmp_path, library):
    client = voice_client
    assert client.get("/api/audio/voices").json() == {"voices": []}
    resp = _upload(client, "Woman 1", _voice_clip(tmp_path / "w.wav", 5.0))
    assert resp.status_code == 200, resp.text
    voice = resp.json()["voice"]
    assert voice["name"] == "Woman 1" and voice["kind"] == "woman" and voice["seconds"] == pytest.approx(5.0, abs=0.01)
    info = sf.info(str(library / "Woman 1.wav"))
    assert (info.samplerate, info.channels) == (48000, 1)
    index = json.loads((library / "voices.json").read_text(encoding="utf-8"))
    assert index["Woman 1"]["consent"] == "recorded with consent" and index["Woman 1"]["source"] == "w.wav"

    assert _upload(client, "short", _voice_clip(tmp_path / "s.wav", 2.0)).status_code == 422
    assert _upload(client, "long", _voice_clip(tmp_path / "l.wav", 61.0)).status_code == 422
    assert _upload(client, "woman 1", _voice_clip(tmp_path / "d.wav", 4.0)).status_code == 409
    assert _upload(client, "../evil", _voice_clip(tmp_path / "e.wav", 4.0)).status_code == 422
    bad = tmp_path / "not_audio.wav"
    bad.write_text("hello")
    assert _upload(client, "broken", bad).status_code == 422
    assert [v["name"] for v in client.get("/api/audio/voices").json()["voices"]] == ["Woman 1"]
    assert not any(p.name.endswith(".part") for p in library.iterdir())

    resp = client.put("/api/audio/voices/Woman 1", json={"name": "maya"})
    assert resp.status_code == 200 and resp.json()["voice"]["name"] == "maya"
    assert (library / "maya.wav").exists() and not (library / "Woman 1.wav").exists()
    assert client.put("/api/audio/voices/ghost", json={"name": "x"}).status_code == 404
    assert client.delete("/api/audio/voices/maya").json() == {"deleted": "maya"}
    assert client.get("/api/audio/voices").json() == {"voices": []}
    assert client.delete("/api/audio/voices/maya").status_code == 404

    presets = client.get("/api/audio/voice/presets").json()
    assert set(presets["presets"]) == {"Woman", "Child", "Demon", "Monster", "Ghost"}


def test_voice_routes_refuse_while_a_render_runs(voice_client, monkeypatch):
    from routes import rendering
    monkeypatch.setitem(rendering.jobs, "render_r", {"id": "render_r", "job_type": "render", "status": "running"})
    for method, path in [("get", "/api/audio/voices"), ("get", "/api/audio/voice/presets"),
                         ("get", "/api/audio/voice/worker"), ("post", "/api/audio/voice/worker/stop"),
                         ("get", "/api/audio/p1/speaker_spans"), ("post", "/api/audio/p1/voice/apply_map")]:
        assert getattr(voice_client, method)(path).status_code == 409, path


# --- the voice map on a synthetic project -----------------------------------------------------

def test_speaker_spans_of_the_project(project):
    got = voice_conversion.project_speaker_spans("p1")
    assert got["has_script"] is True and got["speakers"] == ["MAYA", "RAVI"]
    assert [(s["speaker"], round(s["start_s"], 3), round(s["end_s"], 3)) for s in got["spans"]] == [
        (sp, a, b) for sp, _w, a, b in PARAGRAPHS]


def test_apply_voice_map_replaces_exactly_the_character_spans(project, stub_worker):
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice", "steps": 10},
                                          "ravi": {"voice": "ravi_voice", "preset": "Demon"}})
    progress = []
    report = voice_conversion.apply_voice_map("p1", lambda f, m: progress.append(f))

    assert report["voice_conversions"] == [
        {"speaker": "MAYA", "voice": "maya_voice", "spans": [[3.0, 5.0]], "seconds": 2.0},
        {"speaker": "RAVI", "voice": "ravi_voice", "spans": [[8.0, 10.0]], "seconds": 2.0}]
    assert report["failed_spans"] == [] and report["history_len"] == 1
    source = _source(project.store)
    assert len(source.audio_history) == 1 and source.audio_override == report["version"]["path"]
    assert len(list((project.dir / "audio" / "versions").glob("*.wav"))) == 1
    version, rate = sf.read(source.audio_override, dtype="float32")
    assert rate == RATE
    _assert_converted_only(version, project.original, [(3.0, 5.0), (8.0, 10.0)])
    stored = project.store.get_project("p1")
    assert stored["metadata"]["voice_conversions"] == report["voice_conversions"]
    assert progress[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in progress) and len(progress) > 4

    converts = [job for job in stub_worker.jobs() if job["op"] == "convert"]
    assert len(converts) == 2                                        # the narrator was never sent
    by_ref = {Path(job["ref"]).stem: job["params"] for job in converts}
    assert by_ref["maya_voice"]["diffusion_steps"] == 10 and by_ref["maya_voice"]["length_adjust"] == 1.0
    assert by_ref["ravi_voice"]["pitch_shift"] == -5 and by_ref["ravi_voice"]["f0_condition"] is True
    assert by_ref["ravi_voice"]["diffusion_steps"] == 30

    undone = processes.undo("p1")
    assert undone["audio_override"] is None and undone["history_len"] == 0
    assert _source(project.store).audio_override is None


def test_voice_map_validation(project):
    with pytest.raises(ValueError, match="not in the voices library"):
        voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "nobody"}})
    with pytest.raises(ValueError, match="narrator"):
        voice_conversion.set_voice_map("p1", {"Narrator": {"voice": "maya_voice"}})
    with pytest.raises(ValueError, match="Pick a voice"):
        voice_conversion.set_voice_map("p1", {"MAYA": {"steps": 20}})
    saved = voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice", "match_pitch": True,
                                                           "pitch_shift": 2, "steps": 20, "extra": 1}})
    assert saved == {"voice_map": {"MAYA": {"voice": "maya_voice", "match_pitch": True, "pitch_shift": 2,
                                            "steps": 20}}}
    assert voice_conversion.get_voice_map("p1") == saved


def test_a_rack_window_range_converts_and_undo_restores(project, stub_worker):
    result = voice_conversion.convert_range("p1", None, 3.0, 5.0, {"voice": "maya_voice"})
    assert result["effect"] == "voice_conversion" and result["history_len"] == 1
    version, _rate = sf.read(result["audio_override"], dtype="float32")
    # the lane splices the range back with its crossfades: inside the sine, outside the original
    _assert_converted_only(version, project.original, [(3.0, 5.0)])
    assert processes.undo("p1")["audio_override"] is None


def test_voice_routes_on_a_project(project, stub_worker):
    from routes import audio_process, audio_voice
    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    app.include_router(audio_voice.router, prefix="/api/audio")
    client = TestClient(app)

    def wait(job_id):
        for _ in range(600):
            job = client.get(f"/api/audio/p1/process/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    assert client.post("/api/audio/p1/voice/apply_map").status_code == 409          # no voice map yet
    spans = client.get("/api/audio/p1/speaker_spans").json()
    assert spans["speakers"] == ["MAYA", "RAVI"]
    resp = client.put("/api/audio/p1/voice_map", json={"voice_map": {"RAVI": {"voice": "ravi_voice"}}})
    assert resp.status_code == 200, resp.text
    assert client.get("/api/audio/p1/voice_map").json() == {"voice_map": {"RAVI": {"voice": "ravi_voice"}},
                                                            "applied": None}
    assert client.put("/api/audio/p1/voice_map", json={"voice_map": {"RAVI": {"voice": "x"}}}).status_code == 422

    job = wait(client.post("/api/audio/p1/voice/apply_map").json()["job_id"])
    assert job["status"] == "completed", job
    assert job["result"]["voice_conversions"] == [
        {"speaker": "RAVI", "voice": "ravi_voice", "spans": [[8.0, 10.0]], "seconds": 2.0}]
    worker = client.get("/api/audio/voice/worker").json()
    assert worker["running"] is True and worker["device"] == "cuda" and set(worker) == {"running", "device", "idle_s"}

    resp = client.post("/api/audio/p1/voice/convert",
                       json={"start_s": 3.0, "end_s": 5.0, "params": {"voice": "maya_voice"}})
    assert resp.status_code == 200, resp.text
    job = wait(resp.json()["job_id"])
    assert job["status"] == "completed", job
    assert job["result"]["history_len"] == 2
    version, _rate = sf.read(job["result"]["audio_override"], dtype="float32")
    _assert_converted_only(version, project.original, [(3.0, 5.0), (8.0, 10.0)])
    assert client.post("/api/audio/p1/voice/convert",
                       json={"start_s": 3.0, "end_s": 5.0, "params": {"voice": "nobody"}}).status_code == 422
    assert client.post("/api/audio/p1/voice/convert",
                       json={"start_s": 5.0, "end_s": 3.0, "params": {"voice": "maya_voice"}}).status_code == 422

    assert client.post("/api/audio/p1/undo", json={}).json()["history_len"] == 1
    assert client.post("/api/audio/p1/undo", json={}).json()["audio_override"] is None
    stopped = client.post("/api/audio/voice/worker/stop").json()
    assert stopped == {"running": False, "device": None, "idle_s": None}


def test_routes_answer_503_without_seed_vc(project, monkeypatch, voice_client):
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}})
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": None, "seedvc_dir": None, "cli": "cli.py", "source": "unavailable", "reason": "no seed-vc found"})
    assert voice_client.post("/api/audio/p1/voice/apply_map").status_code == 503
    assert voice_client.post("/api/audio/p1/voice/convert", json={
        "start_s": 3.0, "end_s": 5.0, "params": {"voice": "maya_voice"}}).status_code == 503


# --- an idempotent voice-map apply: replace, never stack ----------------------------------------

TONES = {"maya_voice": 330.0, "ravi_voice": 550.0}
SPANS = [(3.0, 5.0), (8.0, 10.0)]              # MAYA's and RAVI's spans, source seconds
PAD = int(round(processes.PAD_S * RATE))       # the context sent with each span


class ToneWorker:
    """`seedvc_env.Worker.convert` in process: a sine at 22.05 kHz as long as the input, 330 Hz in
    maya_voice and 550 Hz in ravi_voice. Every input is kept, so a test can tell whether a span
    was fed the recording or an earlier conversion."""

    def __init__(self):
        self.jobs = []

    def convert(self, source, ref, out, params=None, progress=None, timeout_s=None):
        data, sr = sf.read(str(source), dtype="float32")
        stem = Path(str(ref)).stem
        self.jobs.append({"ref": stem, "input": data})
        t = np.arange(int(round(data.size * 22050 / sr))) / 22050.0
        sf.write(str(out), (0.3 * np.sin(2 * np.pi * TONES[stem] * t)).astype(np.float32), 22050,
                 subtype="FLOAT")
        return {"ok": True, "op": "convert", "out": str(out), "seconds": 0.01, "device": "cuda", "sr": 22050}

    def status(self):
        return {"running": True, "device": "cuda", "idle_s": 0.0}

    def stop(self, wait_s=30.0, reason="stopped"):
        return {"running": False, "device": None, "idle_s": None}


@pytest.fixture
def tones(monkeypatch, library):
    stub = ToneWorker()
    monkeypatch.setattr(seedvc_env, "WORKER", stub)
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": "python", "seedvc_dir": "stub", "cli": "cli.py", "source": "test", "reason": ""})
    return stub


def _read(path) -> np.ndarray:
    return sf.read(str(path), dtype="float32")[0]


def _versions(project):
    return sorted(p.name for p in (project.dir / "audio" / "versions").glob("*.wav"))


def _record(project):
    return (project.store.get_project("p1").get("settings") or {}).get("voice_map_applied")


def _fed_from(jobs, audio: np.ndarray):
    """One convert per span, in time order, each fed `audio` itself around its span."""
    assert len(jobs) == len(SPANS)
    for job, (start_s, end_s) in zip(jobs, SPANS):
        assert np.array_equal(job["input"], audio[_samples(start_s) - PAD:_samples(end_s) + PAD])


def _assert_tones(version: np.ndarray, before: np.ndarray, hz):
    """Inside each span its tone; outside every span and seam, sample for sample `before`."""
    assert version.shape == before.shape
    untouched = np.ones(before.size, dtype=bool)
    for (start_s, end_s), tone in zip(SPANS, hz):
        a, b = _samples(start_s), _samples(end_s)
        untouched[a - HALF:b + HALF] = False
        assert abs(_peak_hz(version[a + HALF:b - HALF]) - tone) < 3.0
    assert np.array_equal(version[untouched], before[untouched])


def test_the_same_voice_map_applied_twice_converts_once(project, tones):
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}, "RAVI": {"voice": "maya_voice"}})
    first = voice_conversion.apply_voice_map("p1")
    assert not {"skipped", "replaced", "removed", "warning"} & set(first)
    _fed_from(tones.jobs, project.original)
    converted = _read(first["audio_override"])
    _assert_tones(converted, project.original, [330.0, 330.0])
    record = _record(project)
    assert set(record) == {"map_hash", "version_path", "spans_signature", "applied_at", "voice_conversions"}
    assert record["version_path"] == first["version"]["path"] == first["audio_override"]
    assert record["spans_signature"] == [["maya", 3.0, 5.0], ["ravi", 8.0, 10.0]]
    assert record["voice_conversions"] == first["voice_conversions"]
    assert record["applied_at"] == first["version"]["created"]

    tones.jobs.clear()
    second = voice_conversion.apply_voice_map("p1")
    assert second["skipped"] is True and second["reason"] == "already applied"
    assert tones.jobs == []                                              # nothing converted again
    assert second["voice_conversions"] == first["voice_conversions"] and second["failed_spans"] == []
    assert second["version"] == first["version"] and second["revision"] == first["revision"]
    source = _source(project.store)
    assert [entry["path"] for entry in source.audio_history] == [first["version"]["path"]]
    assert source.audio_override == first["version"]["path"]
    assert _versions(project) == ["1_voice_conversion.wav"]
    assert np.array_equal(_read(source.audio_override), converted)
    assert _record(project) == record


def test_a_changed_voice_map_replaces_the_applied_version(project, tones):
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}, "RAVI": {"voice": "maya_voice"}})
    first = voice_conversion.apply_voice_map("p1")
    first_hash = _record(project)["map_hash"]
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}, "RAVI": {"voice": "ravi_voice"}})
    tones.jobs.clear()
    second = voice_conversion.apply_voice_map("p1")

    assert second["replaced"] is True and second["undone"] == first["version"]["path"]
    assert "skipped" not in second and "warning" not in second
    # the first version undone (its file stays), one new version in the history chain
    assert _versions(project) == ["1_voice_conversion.wav", "2_voice_conversion.wav"]
    source = _source(project.store)
    assert [entry["path"] for entry in source.audio_history] == [second["version"]["path"]]
    assert source.audio_override == second["version"]["path"] != first["version"]["path"]
    assert second["history_len"] == 1
    # each span converted once: fed the recording itself, not the first version's tone
    assert [job["ref"] for job in tones.jobs] == ["maya_voice", "ravi_voice"]
    _fed_from(tones.jobs, project.original)
    _assert_tones(_read(source.audio_override), project.original, [330.0, 550.0])
    assert second["voice_conversions"] == [
        {"speaker": "MAYA", "voice": "maya_voice", "spans": [[3.0, 5.0]], "seconds": 2.0},
        {"speaker": "RAVI", "voice": "ravi_voice", "spans": [[8.0, 10.0]], "seconds": 2.0}]
    record = _record(project)
    assert record["version_path"] == second["version"]["path"] and record["map_hash"] != first_hash
    assert processes.undo("p1")["audio_override"] is None                # one Undo takes it back


def test_an_apply_after_later_processing_goes_on_top_with_a_warning(project, tones):
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}, "RAVI": {"voice": "maya_voice"}})
    first = voice_conversion.apply_voice_map("p1")
    normalized = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0})
    tones.jobs.clear()
    again = voice_conversion.apply_voice_map("p1")

    assert again["warning"] == "applied on top of later processing"
    assert "skipped" not in again and "replaced" not in again
    source = _source(project.store)
    assert [entry["path"] for entry in source.audio_history] == [          # nothing undone
        first["version"]["path"], normalized["audio_override"], again["version"]["path"]]
    assert source.audio_override == again["version"]["path"]
    _fed_from(tones.jobs, _read(normalized["audio_override"]))            # built on the processed audio
    assert _record(project)["version_path"] == again["version"]["path"]


def test_an_emptied_voice_map_takes_the_applied_version_back(project, tones):
    earlier = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0})
    voice_conversion.set_voice_map("p1", {"MAYA": {"voice": "maya_voice"}})
    applied = voice_conversion.apply_voice_map("p1")
    processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -3.0})
    voice_conversion.set_voice_map("p1", {})
    voice_conversion.preflight_voice_map("p1")              # an empty map passes while an apply is recorded
    with pytest.raises(noise.Conflict, match="later processing"):
        voice_conversion.apply_voice_map("p1")              # what was made on top of it is never undone
    assert len(_source(project.store).audio_history) == 3 and _record(project) is not None

    processes.undo("p1")                                    # the later normalize goes
    removed = voice_conversion.apply_voice_map("p1")
    assert removed["removed"] is True and removed["undone"] == applied["version"]["path"]
    assert removed["version"] is None and removed["voice_conversions"] == []
    source = _source(project.store)
    assert source.audio_override == earlier["audio_override"]               # the override restored
    assert [entry["path"] for entry in source.audio_history] == [earlier["audio_override"]]
    assert Path(applied["version"]["path"]).exists()                         # the file stays
    assert _record(project) is None
    with pytest.raises(noise.Conflict, match="no voice map"):
        voice_conversion.apply_voice_map("p1")              # nothing recorded any more
    with pytest.raises(noise.Conflict, match="no voice map"):
        voice_conversion.preflight_voice_map("p1")


def test_voice_map_route_shows_applied_and_the_apply_job_flags(project, tones):
    from routes import audio_process, audio_voice
    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    app.include_router(audio_voice.router, prefix="/api/audio")
    client = TestClient(app)

    def apply():
        resp = client.post("/api/audio/p1/voice/apply_map")
        assert resp.status_code == 200, resp.text
        for _ in range(600):
            job = client.get(f"/api/audio/p1/process/{resp.json()['job_id']}").json()
            if job["status"] in ("completed", "failed"):
                assert job["status"] == "completed", job
                return job["result"]
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    assert client.get("/api/audio/p1/voice_map").json() == {"voice_map": {}, "applied": None}
    resp = client.put("/api/audio/p1/voice_map", json={"voice_map": {"MAYA": {"voice": "maya_voice"}}})
    assert resp.status_code == 200, resp.text
    first = apply()
    assert not {"skipped", "replaced", "removed", "warning"} & set(first)
    got = client.get("/api/audio/p1/voice_map").json()
    assert got["voice_map"] == {"MAYA": {"voice": "maya_voice"}}
    assert set(got["applied"]) == {"map_hash", "version_path", "applied_at"}       # no span lists
    assert got["applied"]["version_path"] == first["version"]["path"]

    second = apply()
    assert second["skipped"] is True and second["reason"] == "already applied" and len(tones.jobs) == 1

    client.put("/api/audio/p1/voice_map", json={"voice_map": {"MAYA": {"voice": "ravi_voice"}}})
    third = apply()
    assert third["replaced"] is True and third["undone"] == first["version"]["path"]
    applied = client.get("/api/audio/p1/voice_map").json()["applied"]
    assert applied["version_path"] == third["version"]["path"]
    assert applied["map_hash"] != got["applied"]["map_hash"]

    client.put("/api/audio/p1/voice_map", json={"voice_map": {}})
    removed = apply()
    assert removed["removed"] is True and removed["audio_override"] is None
    assert client.get("/api/audio/p1/voice_map").json() == {"voice_map": {}, "applied": None}
    assert client.post("/api/audio/p1/voice/apply_map").status_code == 409        # nothing left to take back
