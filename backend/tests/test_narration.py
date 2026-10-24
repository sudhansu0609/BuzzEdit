"""Narration (AUTO_STORY_PLAN.md brief N1): the TTS CLI's text handling, and the job registry and
routes with the subprocess replaced."""
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import narration, seedvc_env
from runtime import job_store

cli = narration.cli_module()


# --------------------------------------------------------------------------- the CLI's text handling

def test_cues_become_silences_and_are_never_spoken():
    plan = cli.plan_batches("Pehli line. (pause) Doosri line. (pause 2s) Teesri. (beat) Chauthi.")
    assert [p.get("silence") for p in plan if "silence" in p] == [0.6, 2.0, 0.3]
    spoken = " ".join(p["text"] for p in plan if "text" in p)
    assert "(" not in spoken and "pause" not in spoken and "beat" not in spoken
    assert [("text" in p) for p in plan] == [True, False, True, False, True, False, True]


def test_slow_and_normal_switch_the_pace_of_what_follows():
    plan = cli.plan_batches("Ek. (slow) Do. Teen. (normal) Chaar.")
    texts = [p for p in plan if "text" in p]
    assert [p["slow"] for p in texts] == [False, True, False]
    assert all("silence" not in p for p in plan)


def test_adjacent_silences_merge():
    plan = cli.plan_batches("Ek. (pause) (beat) Do.")
    assert plan[1] == {"silence": pytest.approx(0.9)}


def test_batches_are_whole_sentences_up_to_220_chars():
    sentence = "Yeh ek lambi kahani ka hissa hai jo chalta rehta hai."     # 53 chars
    plan = cli.plan_batches(" ".join([sentence] * 10))
    texts = [p["text"] for p in plan]
    assert all(len(t) <= cli.MAX_BATCH_CHARS for t in texts)
    assert len(texts) == 3                                   # 4 + 4 + 2 sentences
    assert all(t.endswith(".") for t in texts)
    assert " ".join(texts) == " ".join([sentence] * 10)


def test_sentence_punctuation_includes_danda_and_ellipsis():
    assert cli.split_sentences("Raat hui। Koi aaya… Kaun? Pata nahi!") == \
        ["Raat hui।", "Koi aaya…", "Kaun?", "Pata nahi!"]


def test_an_overlong_sentence_is_cut_at_commas_then_spaces():
    long = ", ".join(["shabd " * 15] * 4) + "."
    pieces = cli.plan_batches(long)
    assert all(len(p["text"]) <= cli.MAX_BATCH_CHARS for p in pieces)
    assert len(pieces) >= 2
    no_commas = "a" * 10 + " " + ("lafz " * 80) + "."
    assert all(len(p["text"]) <= cli.MAX_BATCH_CHARS for p in cli.plan_batches(no_commas))


@pytest.mark.parametrize("word,start,hindi", [
    ("raat", False, True), ("darwaza", False, True), ("kholna", False, True),
    ("minute", False, False), ("unlock", False, False), ("flat", False, False),
    ("NASA", False, False), ("Sudhansu", False, False), ("Raat", True, True),
    ("check", False, False), ("phone", False, False), ("don't", False, False),
    ("hai", False, True), ("nahi", False, True), ("", False, False), ("रात", False, False),
])
def test_transliteration_decisions(word, start, hindi):
    assert cli.is_hindi_word(word, sentence_start=start) is hindi


def test_hinglish_spelling_rules():
    assert cli.to_itrans("darwaza") == "darvazA"
    assert cli.to_itrans("apni") == "apnI"
    assert cli.to_itrans("koi") == "koI"
    assert cli.to_itrans("hain") == "haiM"


def test_transliterate_text_keeps_english_and_names_latin():
    pytest.importorskip("indic_transliteration")
    out = cli.transliterate_text("Raat ko Sudhansu ne door unlock kiya aur bajkar dekha.")
    assert "Sudhansu" in out and "door" in out and "unlock" in out
    assert "Raat" not in out and "bajkar" not in out
    assert "रात" in out                 # रात
    assert "बजकर" in out           # बजकर: no halant inside
    assert not any(w.endswith("्") for w in out.split())


def test_plan_text_shows_spoken_text_and_pauses():
    plan = [{"text": "a", "spoken": "A1", "slow": False}, {"silence": 0.6}, {"text": "b", "slow": True}]
    assert cli.plan_text(plan) == "A1 (pause 0.6s) (slow) b"


def test_join_crossfades_60ms_and_silences_keep_their_length():
    sr = cli.SAMPLE_RATE
    a = np.ones(sr, dtype=np.float32)
    b = np.ones(sr // 2, dtype=np.float32)
    joined = cli.join_audio([a, b])
    n = int(round(cli.XFADE_S * sr))
    assert joined.size == a.size + b.size - n
    gap = cli.silence(0.6)
    assert gap.size == int(round((0.6 + 2 * cli.XFADE_S) * sr))
    with_gap = cli.join_audio([a, gap, b])
    assert with_gap.size == a.size + gap.size + b.size - 2 * n
    quiet = np.flatnonzero(with_gap == 0.0)
    assert quiet.size == pytest.approx(int(round(0.6 * sr)), abs=2)


def test_batch_ok_rejects_empty_and_nan():
    assert not cli.batch_ok(np.zeros(10, dtype=np.float32))
    assert not cli.batch_ok(np.full(cli.SAMPLE_RATE, np.nan, dtype=np.float32))
    assert cli.batch_ok(np.zeros(cli.SAMPLE_RATE, dtype=np.float32))


def test_cli_voices_lists_references_and_presets(tmp_path):
    (tmp_path / "sudhansu.wav").write_bytes(b"RIFF")
    voices = cli.list_voices(tmp_path)
    assert voices[0] == {"id": "sudhansu", "kind": "reference", "path": str(tmp_path / "sudhansu.wav")}
    assert {"id": "hm_omega", "kind": "kokoro", "language": "hi"} in voices


# --------------------------------------------------------------------------- service, jobs, routes

@pytest.fixture
def env(monkeypatch, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "sudhansu.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(narration, "voices_dir", lambda: voices)
    monkeypatch.setattr(narration, "available", lambda: {"available": True, "reason": ""})
    monkeypatch.setattr(seedvc_env.WORKER, "running", lambda: False)
    # Durable jobs (STUDIO_DIRECTOR_PLAN.md enhancement 2): never touch the
    # real `data/jobs/narration`.
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path / "jobs")
    narration._jobs.clear()
    calls = {"tts": [], "convert": []}

    def fake_tts(argv, progress=None, timeout_s=None):
        calls["tts"].append(argv)
        out = argv[argv.index("--out") + 1]
        if progress:
            progress(0.5, "batch 1/2")
        return {"wav": out, "duration_s": 12.5, "engine": argv[argv.index("--engine") + 1],
                "voice": argv[argv.index("--voice") + 1], "batches": 2, "retries": 1,
                "transliterated_text": "रात (pause 0.6s) ..."}

    def fake_convert(args, progress=None, timeout_s=None):
        calls["convert"].append([str(a) for a in args])
        return {"out": args[args.index("--out") + 1]}

    monkeypatch.setattr(narration, "_run_tts", fake_tts)
    monkeypatch.setattr(seedvc_env, "run_cli", fake_convert)
    return calls, tmp_path


def test_engine_follows_the_voice(env):
    assert narration.resolve_request("Raat.", voice="sudhansu")["engine"] == "chatterbox"
    assert narration.resolve_request("Raat.", voice="hm_omega")["engine"] == "kokoro"
    assert narration.resolve_request("Raat.", voice="engine")["engine"] == "chatterbox"
    with pytest.raises(ValueError):
        narration.resolve_request("Raat.", voice="nobody")
    with pytest.raises(ValueError):
        narration.resolve_request("Raat.", voice="hm_omega", engine="chatterbox")
    with pytest.raises(ValueError):
        narration.resolve_request("(pause)", voice="sudhansu")
    with pytest.raises(ValueError):
        narration.resolve_request("Raat.", voice="hm_omega", convert_to="nobody")


@pytest.mark.parametrize("bad_style", [
    {"mood": "furious"}, {"exaggeration": 2.5}, {"cfg_weight": -0.1}, {"temperature": 0.0},
    {"temperature": 1.6}, {"pace": 0.4}, {"word_gap_ms": 301}, {"phrase_pause_s": 2.1},
    {"sentence_pause_s": 3.5},
])
def test_style_validates_the_mood_and_every_range(env, bad_style):
    with pytest.raises(ValueError):
        narration.resolve_request("Raat.", voice="sudhansu", style=bad_style)


def test_resolve_request_resolves_mood_into_a_full_style(env):
    req = narration.resolve_request("Raat.", voice="sudhansu", style={"mood": "dramatic"})
    assert req["style"]["mood"] == "dramatic"
    assert req["exaggeration"] == req["style"]["exaggeration"] == 0.8
    assert req["style"]["sentence_pause_s"] == 0.3   # halved (VOICE_DELIVERY_PLAN.md E3)
    # Unknown keys are ignored, not rejected.
    req = narration.resolve_request("Raat.", voice="sudhansu", style={"mood": "steady", "foo": 1})
    assert "foo" not in req["style"]


def test_synthesize_chatterbox_passes_the_style(env):
    calls, tmp = env
    result = narration.synthesize("Raat ke do baje.", style={"exaggeration": 0.7, "pace": 0.9},
                                  out_dir=str(tmp / "out"))
    argv = calls["tts"][0]
    assert argv[0] == "synthesize"
    assert argv[argv.index("--engine") + 1] == "chatterbox"
    assert argv[argv.index("--voice") + 1] == "sudhansu"
    assert argv[argv.index("--exaggeration") + 1] == "0.7"
    assert argv[argv.index("--pace") + 1] == "0.9"
    style = json.loads(argv[argv.index("--style") + 1])
    assert style["exaggeration"] == 0.7 and style["pace"] == 0.9 and style["mood"] == "steady"
    assert result["converted"] is False and result["engine"] == "chatterbox"
    assert result["duration_s"] == 12.5 and result["retries"] == 1
    assert "seconds" in result and result["wav"].endswith(".wav")
    assert calls["convert"] == []


def test_kokoro_with_convert_to_runs_seedvc(env):
    calls, tmp = env
    result = narration.synthesize("Raat.", voice="hm_omega", convert_to="sudhansu", out_dir=str(tmp / "o"))
    assert calls["tts"][0][calls["tts"][0].index("--engine") + 1] == "kokoro"
    conv = calls["convert"][0]
    assert conv[0] == "convert"
    assert conv[conv.index("--ref") + 1].endswith("sudhansu.wav")
    assert result["converted"] is True and result["wav"].endswith("_as_sudhansu.wav")


def test_refused_while_a_render_runs(env, monkeypatch):
    def busy():
        raise seedvc_env.SeedVCBusy("render")
    monkeypatch.setattr(seedvc_env, "_refuse_while_rendering", busy)
    with pytest.raises(narration.NarrationBusy):
        narration.synthesize("Raat.")


def test_offloads_after_a_failed_run(env, monkeypatch):
    from runtime import gpu_handover
    offloaded = []

    async def offload(reason="task finished"):
        offloaded.append(reason)
    monkeypatch.setattr(gpu_handover, "offload_all", offload)

    def fail(argv, progress=None, timeout_s=None):
        raise narration.NarrationError("boom")
    monkeypatch.setattr(narration, "_run_tts", fail)
    with pytest.raises(narration.NarrationError):
        narration.synthesize("Raat.")
    assert offloaded == ["narration"]


def _client():
    from routes import narration as routes
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/narration")
    return TestClient(app)


def _wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/narration/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_routes_run_a_job_to_done(env):
    _, tmp = env
    client = _client()
    started = client.post("/api/narration/synthesize",
                          json={"text": "Raat ke do baje. (pause) Darwaza.", "out_dir": str(tmp / "r")})
    assert started.status_code == 200
    body = started.json()
    assert body["state"] == "running" and body["job_id"]
    job = _wait(client, body["job_id"])
    assert job["state"] == "done", job
    assert set(("wav", "seconds", "duration_s", "engine", "converted")) <= set(job["result"])


def test_routes_report_a_failed_job_and_bad_requests(env, monkeypatch):
    client = _client()
    assert client.post("/api/narration/synthesize", json={"text": ""}).status_code == 400
    assert client.post("/api/narration/synthesize", json={"text": "a.", "voice": "x"}).status_code == 400
    bad_mood = client.post("/api/narration/synthesize",
                           json={"text": "a.", "style": {"mood": "furious"}})
    assert bad_mood.status_code == 400
    assert client.get("/api/narration/jobs/nope").status_code == 404

    def fail(argv, progress=None, timeout_s=None):
        raise narration.NarrationError("batch 1 failed twice")
    monkeypatch.setattr(narration, "_run_tts", fail)
    body = client.post("/api/narration/synthesize", json={"text": "Raat."}).json()
    job = _wait(client, body["job_id"])
    assert job["state"] == "failed" and "failed twice" in job["error"]


def test_routes_409_while_rendering_and_503_without_engine(env, monkeypatch):
    client = _client()
    monkeypatch.setattr(narration, "available", lambda: {"available": False, "reason": "no venv"})
    assert client.post("/api/narration/synthesize", json={"text": "Raat."}).status_code == 503
    monkeypatch.setattr(narration, "available", lambda: {"available": True, "reason": ""})

    def busy():
        raise seedvc_env.SeedVCBusy("render")
    monkeypatch.setattr(seedvc_env, "_refuse_while_rendering", busy)
    assert client.post("/api/narration/synthesize", json={"text": "Raat."}).status_code == 409


# --- durable jobs (STUDIO_DIRECTOR_PLAN.md enhancement 2, Fable review row 7) --------

def test_job_is_persisted_and_answers_from_disk_after_a_restart(env):
    client = _client()
    started = client.post("/api/narration/synthesize", json={"text": "Raat ke do baje."})
    job_id = started.json()["job_id"]
    job = _wait(client, job_id)
    assert job["state"] == "done"

    on_disk = job_store.load(narration.JOB_KIND, job_id)
    assert on_disk is not None and on_disk["state"] == "done"

    # Simulate a restart: the in-memory registry is gone, the route still answers.
    narration._jobs.clear()
    resp = client.get(f"/api/narration/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["state"] == "done"


def test_startup_sweep_marks_a_stale_narration_job_failed(env):
    job_store.save(narration.JOB_KIND, "stale1", {"job_id": "stale1", "state": "running", "progress": 0.2})
    marked = job_store.sweep_stale(narration.JOB_KIND)
    assert marked == 1
    on_disk = job_store.load(narration.JOB_KIND, "stale1")
    assert on_disk["state"] == "failed" and on_disk["error"] == job_store.RESTARTED_ERROR


def test_busy_route_reports_a_running_job(env, monkeypatch):
    client = _client()
    assert client.get("/api/narration/busy").json() == {"busy": False, "job_id": None}

    started_sync = threading.Event()
    release = threading.Event()

    def slow_tts(argv, progress=None, timeout_s=None):
        started_sync.set()
        release.wait(5)
        out = argv[argv.index("--out") + 1]
        return {"wav": out, "duration_s": 1.0, "engine": argv[argv.index("--engine") + 1],
                "voice": argv[argv.index("--voice") + 1], "batches": 1, "retries": 0}

    monkeypatch.setattr(narration, "_run_tts", slow_tts)
    started = client.post("/api/narration/synthesize", json={"text": "Raat."})
    job_id = started.json()["job_id"]
    assert started_sync.wait(5)
    try:
        resp = client.get("/api/narration/busy").json()
        assert resp["busy"] is True and resp["job_id"] == job_id
    finally:
        release.set()
    _wait(client, job_id)
    assert client.get("/api/narration/busy").json()["busy"] is False


def test_voices_route(env):
    voices = _client().get("/api/narration/voices").json()["voices"]
    assert voices[0]["id"] == "sudhansu" and voices[0]["kind"] == "reference"
    assert any(v["id"] == "hm_omega" and v["kind"] == "kokoro" for v in voices)


def test_moods_route(env):
    moods = _client().get("/api/narration/moods").json()["moods"]
    assert [m["id"] for m in moods] == \
        ["steady", "calm", "warm", "eerie", "dramatic", "intense", "horror_narrator"]
    for mood in moods:
        assert set(("id", "label", "description", "style")) <= set(mood)


# --------------------------------------------------------------------------- convert_after voices (F8)

def _horror_voice(tmp):
    """`sudhansu_horror.wav` + its convert_after sidecar in the fixture's voices dir, and the
    seed-vc target it points at."""
    voices = tmp / "voices"
    sf.write(str(voices / "sudhansu_horror.wav"), np.zeros(240, dtype=np.float32), 24000)
    target = tmp / "target.wav"
    sf.write(str(target), np.zeros(240, dtype=np.float32), 24000)
    (voices / "sudhansu_horror.seedvc.json").write_text(json.dumps({
        "convert_after": True, "checkpoint": "ckpt.pth", "config": "cfg.yml",
        "convert_ref": str(target), "diffusion_steps": 40, "inference_cfg_rate": 0.7,
    }), encoding="utf-8")
    return target


def test_voice_convert_sidecar_reads_convert_after_only(env):
    _, tmp = env
    _horror_voice(tmp)
    sidecar = narration._voice_convert_sidecar("sudhansu_horror")
    assert sidecar["checkpoint"] == "ckpt.pth" and sidecar["diffusion_steps"] == 40
    assert narration._voice_convert_sidecar("sudhansu") is None            # no sidecar at all
    (tmp / "voices" / "sudhansu.seedvc.json").write_text(
        json.dumps({"checkpoint": "other.pth"}), encoding="utf-8")        # no convert_after key
    assert narration._voice_convert_sidecar("sudhansu") is None


def _fake_tts_writing_speech(calls):
    def fake_tts(argv, progress=None, timeout_s=None):
        calls["tts"].append(argv)
        out = argv[argv.index("--out") + 1]
        sf.write(out, np.full(24000, 0.2, dtype=np.float32), 24000)        # 1 s of "speech"
        return {"wav": out, "duration_s": 1.0, "engine": argv[argv.index("--engine") + 1],
                "voice": argv[argv.index("--voice") + 1], "batches": 1, "retries": 0}
    return fake_tts


def test_convert_after_voice_speaks_at_pace_1_and_converts_with_length_adjust(env, monkeypatch):
    calls, tmp = env
    target = _horror_voice(tmp)

    def fake_convert(args, progress=None, timeout_s=None):
        calls["convert"].append([str(a) for a in args])
        out = args[args.index("--out") + 1]
        sf.write(out, np.full(24000, 0.2, dtype=np.float32), 24000)
        return {"out": out}

    monkeypatch.setattr(narration, "_run_tts", _fake_tts_writing_speech(calls))
    monkeypatch.setattr(seedvc_env, "run_cli", fake_convert)

    result = narration.synthesize("Raat.", voice="sudhansu_horror", style={"mood": "horror_narrator"},
                                  out_dir=str(tmp / "o"))

    tts_argv = calls["tts"][0]
    assert tts_argv[tts_argv.index("--pace") + 1] == "1"                   # Chatterbox always at 1.0
    assert json.loads(tts_argv[tts_argv.index("--style") + 1])["pace"] == 1.0

    conv = calls["convert"][0]
    assert conv[0] == "convert" and conv[conv.index("--ref") + 1] == str(target)
    params = json.loads(conv[conv.index("--params") + 1])
    assert params["length_adjust"] == pytest.approx(1.0 / 0.90, rel=1e-3)  # 1/the mood's own pace
    assert params["checkpoint"] == "ckpt.pth" and params["diffusion_steps"] == 40

    assert result["converted"] is True
    assert result["length_adjust"] == pytest.approx(1.0 / 0.90, rel=1e-3)
    assert result["convert_ref"] == str(target) and result["checkpoint"] == "ckpt.pth"


def test_convert_after_length_adjust_is_clamped(env, monkeypatch):
    calls, tmp = env
    _horror_voice(tmp)

    def fake_convert(args, progress=None, timeout_s=None):
        calls["convert"].append([str(a) for a in args])
        out = args[args.index("--out") + 1]
        sf.write(out, np.zeros(2400, dtype=np.float32), 24000)
        return {"out": out}

    monkeypatch.setattr(narration, "_run_tts", _fake_tts_writing_speech(calls))
    monkeypatch.setattr(seedvc_env, "run_cli", fake_convert)

    result = narration.synthesize("Raat.", voice="sudhansu_horror", style={"pace": 0.5},
                                  out_dir=str(tmp / "o2"))
    assert result["length_adjust"] == 2.0                                 # 1/0.5 clamped to the 0.5-2 range


def test_convert_after_voice_seedvc_failure_raises(env, monkeypatch):
    calls, tmp = env
    _horror_voice(tmp)

    def fail_convert(args, progress=None, timeout_s=None):
        raise seedvc_env.SeedVCError("boom")

    monkeypatch.setattr(narration, "_run_tts", _fake_tts_writing_speech(calls))
    monkeypatch.setattr(seedvc_env, "run_cli", fail_convert)
    with pytest.raises(seedvc_env.SeedVCError):
        narration.synthesize("Raat.", voice="sudhansu_horror", out_dir=str(tmp / "o3"))
