"""The seed-vc voice engine as a separate process (VOICE_CHANGER_PLAN.md, brief S0).

`engines/seedvc/segments.py` (SkinWalker's stitcher) is loaded by path; `engines/seedvc/cli.py` runs
under this -- BuzzEdit's -- venv in --dry-run mode, which loads no model; `audio/seedvc_env.py` is
tested with monkeypatched paths and a stub script standing in for the CLI. No GPU, no model.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio import seedvc_env
from runtime import gpu_handover

ENGINE = Path(__file__).resolve().parents[2] / "engines" / "seedvc"
CLI = ENGINE / "cli.py"
SR = 16000


def _load_segments():
    spec = importlib.util.spec_from_file_location("seedvc_segments", ENGINE / "segments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


segments = _load_segments()


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * SR)).astype(np.float32)


def _overlap(xfade_ms: float) -> int:
    return int(round(SR * xfade_ms / 1000.0))


def _constant(region, seg, sr):
    return np.full(seg.shape[0], 0.25, dtype=np.float32), sr


# --- the stitcher -----------------------------------------------------------------------------

def test_convert_region_is_replaced_and_unmarked_audio_is_bit_identical():
    source = _noise(5.0)
    seen = []

    def fake_convert(region, seg, sr):
        seen.append((region["id"], seg.copy(), sr))
        return _constant(region, seg, sr)

    manifest = {"duration": 5.0, "master_sr": SR,
                "regions": [{"id": "r1", "start": 1.0, "end": 2.0, "kind": "convert", "voice": "demon"}]}
    out = segments.apply_edit(source, SR, manifest, fake_convert, master_sr=SR, xfade_ms=15.0)

    ov = _overlap(15.0)
    assert [s[0] for s in seen] == ["r1"]
    assert np.array_equal(seen[0][1], source[SR:2 * SR]) and seen[0][2] == SR
    assert out.shape[0] == 5 * SR - 2 * ov                            # two seams, overlap-added
    assert np.array_equal(out[:SR - ov], source[:SR - ov])            # before the region: untouched
    assert np.all(out[SR:2 * SR - 2 * ov] == np.float32(0.25))        # the region: the conversion
    assert np.array_equal(out[2 * SR - ov:], source[2 * SR + ov:])    # after it: untouched


def test_cut_region_is_removed():
    source = _noise(5.0, seed=1)
    called = []
    manifest = {"duration": 5.0, "master_sr": SR,
                "regions": [{"id": "c1", "start": 2.0, "end": 3.0, "kind": "cut"}]}
    out = segments.apply_edit(source, SR, manifest, lambda *a: called.append(a), master_sr=SR,
                              xfade_ms=15.0)

    ov = _overlap(15.0)
    assert called == []
    assert out.shape[0] == 4 * SR - ov
    assert np.array_equal(out[:2 * SR - ov], source[:2 * SR - ov])
    assert np.array_equal(out[2 * SR:], source[3 * SR + ov:])         # the second removed


@pytest.mark.parametrize("xfade_ms", [5.0, 15.0, 40.0])
def test_crossfade_is_as_long_as_asked(xfade_ms):
    source = _noise(3.0, seed=2)
    manifest = {"duration": 3.0, "master_sr": SR,
                "regions": [{"id": "r1", "start": 1.0, "end": 2.0, "kind": "convert", "voice": "v"}]}
    out = segments.apply_edit(source, SR, manifest, _constant, master_sr=SR, xfade_ms=xfade_ms)

    ov = _overlap(xfade_ms)
    assert out.shape[0] == 3 * SR - 2 * ov
    t = np.linspace(0.0, np.pi / 2.0, num=ov, dtype=np.float32)
    expected = source[SR - ov:SR] * np.cos(t) + np.float32(0.25) * np.sin(t)   # equal power
    np.testing.assert_allclose(out[SR - ov:SR], expected, rtol=0, atol=1e-6)
    assert out[SR - ov - 1] == source[SR - ov - 1]                    # untouched right before it
    assert out[SR] == np.float32(0.25)                                # pure conversion right after


def test_failed_conversion_keeps_the_original_segment():
    source = _noise(3.0, seed=3)
    manifest = {"duration": 3.0, "master_sr": SR,
                "regions": [{"id": "r1", "start": 1.0, "end": 2.0, "kind": "convert", "voice": "v"}]}

    def broken(region, seg, sr):
        raise RuntimeError("CUDA out of memory")

    out = segments.apply_edit(source, SR, manifest, broken, master_sr=SR, xfade_ms=15.0)
    ov = _overlap(15.0)
    assert out.shape[0] == 3 * SR - 2 * ov
    assert np.array_equal(out[SR:2 * SR - 2 * ov], source[SR + ov:2 * SR - ov])


# --- cli.py, dry run under BuzzEdit's venv ---------------------------------------------------

def _cli(tmp_path: Path, *args):
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hf-cache")}
    proc = subprocess.run([sys.executable, str(CLI), *[str(a) for a in args]], capture_output=True,
                          text=True, encoding="utf-8", timeout=120, env=env)
    events = []
    for line in proc.stderr.splitlines():
        start = line.find('{"event"')
        if start >= 0:
            events.append(json.loads(line[start:]))
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"stdout must hold exactly one JSON object: {proc.stdout!r} / {proc.stderr!r}"
    return proc.returncode, events, json.loads(lines[0])


@pytest.fixture
def engine_files(tmp_path):
    """A seed-vc folder whose wrapper must never be imported, a voices folder, two WAVs."""
    seedvc = tmp_path / "seed-vc"
    seedvc.mkdir()
    (seedvc / "seed_vc_wrapper.py").write_text("raise SystemExit('a dry run must not import seed-vc')\n")
    voices = tmp_path / "voices"
    voices.mkdir()
    sf.write(str(voices / "demon.wav"), _noise(1.5, seed=4) * 0.2, SR)
    source = tmp_path / "line.wav"
    sf.write(str(source), _noise(2.0, seed=5) * 0.2, SR)
    return SimpleNamespace(root=tmp_path, seedvc=seedvc, voices=voices, source=source,
                           ref=voices / "demon.wav")


def test_cli_convert_dry_run_speaks_the_protocol(engine_files):
    out = engine_files.root / "out.wav"
    params = {"diffusion_steps": 30, "f0_condition": True, "pitch_shift": -5, "not_a_param": 1}
    code, events, result = _cli(engine_files.root, "convert", "--dry-run", "--source", engine_files.source,
                                "--ref", engine_files.ref, "--out", out, "--params", json.dumps(params),
                                "--seedvc-dir", engine_files.seedvc, "--voices-dir", engine_files.voices)
    assert code == 0
    progress = [e for e in events if e["event"] == "progress"]
    assert progress and all(0.0 <= e["fraction"] <= 1.0 and isinstance(e["stage"], str) for e in progress)
    assert events[-1]["event"] == "done" and events[-1]["command"] == "convert"
    assert result["ok"] is True and result["dry_run"] is True and result["command"] == "convert"
    assert result["params"] == {"diffusion_steps": 30, "length_adjust": 1.0, "inference_cfg_rate": 0.7,
                                "f0_condition": True, "auto_f0_adjust": True, "pitch_shift": -5}
    assert result["source_s"] == pytest.approx(2.0, abs=0.001)
    assert not out.exists()                                           # a dry run writes nothing


@pytest.mark.parametrize("args, message", [
    (["convert", "--source", "<source>"], "required"),                # a usage error is exit 1, not 2
    (["convert", "--source", "<root>/missing.wav", "--ref", "<ref>", "--out", "<root>/o.wav"], "does not exist"),
    (["convert", "--dry-run", "--source", "<source>", "--ref", "<ref>", "--out", "<root>/o.wav",
      "--params", "{not json"], "not valid JSON"),
    (["convert", "--dry-run", "--source", "<source>", "--ref", "<ref>", "--out", "<root>/o.wav",
      "--params", '{"diffusion_steps": 0}'], "diffusion_steps"),
])
def test_cli_errors_exit_1_with_an_error_event(engine_files, args, message):
    names = {"<source>": engine_files.source, "<ref>": engine_files.ref, "<root>": engine_files.root}
    resolved = []
    for arg in args:
        for token, value in names.items():
            arg = arg.replace(token, str(value))
        resolved.append(arg)
    code, events, result = _cli(engine_files.root, *resolved, "--seedvc-dir", engine_files.seedvc)
    assert code == 1
    assert events[-1]["event"] == "error" and message in events[-1]["message"]
    assert result["ok"] is False and result["exit_code"] == 1 and message in result["error"]


def test_cli_exits_2_when_the_engine_is_missing(engine_files):
    code, events, result = _cli(engine_files.root, "warm", "--dry-run",
                                "--seedvc-dir", engine_files.root / "nowhere")
    assert code == 2
    assert events[-1]["event"] == "error" and "seed_vc_wrapper.py" in events[-1]["message"]
    assert result == {"ok": False, "command": "warm", "error": events[-1]["message"], "exit_code": 2}


def test_cli_warm_dry_run_loads_nothing(engine_files):
    code, events, result = _cli(engine_files.root, "warm", "--dry-run", "--seedvc-dir", engine_files.seedvc)
    assert code == 0 and events[-1]["event"] == "done"
    assert result["loaded"] is False and result["seedvc_dir"] == str(engine_files.seedvc.resolve())
    assert result["checkpoints"] == [] and result["checkpoint_bytes"] == 0


def test_cli_apply_edit_dry_run_plans_spans_and_voices(engine_files):
    manifest = engine_files.root / "edit.json"
    manifest.write_text(json.dumps({"duration": 2.0, "regions": [
        {"id": "r1", "start": 0.5, "end": 1.0, "kind": "convert", "voice": "demon",
         "engine": {"pitch_shift": -5}},
        {"id": "c1", "start": 1.2, "end": 1.4, "kind": "cut"},
        {"id": "r2", "start": 1.5, "end": 1.8, "kind": "convert", "voice": "nobody"},
    ]}))
    out = engine_files.root / "edit.wav"
    code, events, result = _cli(engine_files.root, "apply-edit", "--dry-run", "--source", engine_files.source,
                                "--manifest", manifest, "--out", out, "--seedvc-dir", engine_files.seedvc,
                                "--voices-dir", engine_files.voices)
    assert code == 0
    assert [s["kind"] for s in result["spans"]] == ["keep", "convert", "keep", "cut", "keep", "convert", "keep"]
    assert result["sr"] == SR                                         # no master_sr: the source's rate
    assert result["xfade_ms"] == 15.0
    first, second = result["regions"]
    assert first["ref"] == str(engine_files.ref.resolve()) and first["params"]["pitch_shift"] == -5
    assert "nobody" in second["error"]
    assert not out.exists()


def test_cli_voices_list(engine_files):
    code, events, result = _cli(engine_files.root, "voices", "list", "--voices-dir", engine_files.voices)
    assert code == 0 and result["command"] == "voices list"
    assert [(v["name"], v["sr"]) for v in result["voices"]] == [("demon", SR)]
    assert result["voices"][0]["duration_s"] == pytest.approx(1.5, abs=0.001)


# --- seedvc_env.resolve -----------------------------------------------------------------------

def _make_python(venv: Path) -> Path:
    python = seedvc_env._venv_python(venv)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    return python


def _make_seedvc(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "seed_vc_wrapper.py").write_text("")
    return folder


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Empty stand-ins for engines/seedvc and skinWalker, the CLI present, no settings."""
    cli = tmp_path / "cli.py"
    cli.write_text("")
    settings = {"seedvc_dir": "", "seedvc_python": ""}
    monkeypatch.setattr(seedvc_env, "ENGINE_DIR", tmp_path / "engines" / "seedvc")
    monkeypatch.setattr(seedvc_env, "SKINWALKER_DIR", tmp_path / "skinWalker")
    monkeypatch.setattr(seedvc_env, "CLI_PATH", cli)
    monkeypatch.setattr(seedvc_env, "_configured", lambda: dict(settings))
    return SimpleNamespace(root=tmp_path, settings=settings, cli=cli)


def test_resolve_precedence_settings_engines_skinwalker_unavailable(places):
    root = places.root
    got = seedvc_env.resolve()
    assert got["source"] == "unavailable" and got["python"] is None and got["reason"]
    assert seedvc_env.available() is False

    skin_python = _make_python(root / "skinWalker" / ".venv")
    skin_dir = _make_seedvc(root / "skinWalker" / "vendor" / "seed-vc")
    got = seedvc_env.resolve()
    assert (got["source"], got["python"], got["seedvc_dir"], got["cli"]) == (
        "skinwalker", str(skin_python), str(skin_dir), str(places.cli))

    own_python = _make_python(root / "engines" / "seedvc" / ".venv")
    own_dir = _make_seedvc(root / "engines" / "seedvc" / "seed-vc")
    got = seedvc_env.resolve()
    assert (got["source"], got["python"], got["seedvc_dir"]) == ("engines", str(own_python), str(own_dir))

    set_python = _make_python(root / "custom" / ".venv")
    set_dir = _make_seedvc(root / "custom" / "seed-vc")
    places.settings.update(seedvc_dir=str(set_dir), seedvc_python=str(set_python))
    got = seedvc_env.resolve()
    assert (got["source"], got["python"], got["seedvc_dir"]) == ("settings", str(set_python), str(set_dir))
    assert seedvc_env.available() is True


def test_a_setting_that_points_nowhere_is_not_replaced(places):
    _make_python(places.root / "engines" / "seedvc" / ".venv")
    _make_seedvc(places.root / "engines" / "seedvc" / "seed-vc")
    places.settings["seedvc_python"] = str(places.root / "gone" / "python.exe")
    got = seedvc_env.resolve()
    assert got["source"] == "unavailable" and "seedvc_python" in got["reason"]


def test_one_setting_takes_the_other_half_from_auto_detect(places):
    own_python = _make_python(places.root / "engines" / "seedvc" / ".venv")
    set_dir = _make_seedvc(places.root / "custom" / "seed-vc")
    places.settings["seedvc_dir"] = str(set_dir)
    got = seedvc_env.resolve()
    assert (got["source"], got["python"], got["seedvc_dir"]) == ("settings", str(own_python), str(set_dir))


def test_resolve_without_the_cli_is_unavailable(places):
    _make_python(places.root / "engines" / "seedvc" / ".venv")
    _make_seedvc(places.root / "engines" / "seedvc" / "seed-vc")
    places.cli.unlink()
    got = seedvc_env.resolve()
    assert got["source"] == "unavailable" and "cli.py" in got["reason"]


def test_voices_dir_is_data_voices_and_created(tmp_path, monkeypatch):
    monkeypatch.setattr(seedvc_env, "DATA_DIR", tmp_path / "data")
    folder = seedvc_env.voices_dir()
    assert folder == tmp_path / "data" / "voices" and folder.is_dir()


# --- seedvc_env.run_cli with a stub script as the CLI ----------------------------------------

STUB = textwrap.dedent('''
    import json, sys, time
    args = sys.argv[1:]
    command = args[0]

    def event(**fields):
        sys.stderr.write(json.dumps(fields) + "\\n")
        sys.stderr.flush()

    if command == "sleep":
        time.sleep(30)
    sys.stderr.write("Loading seed-vc models...\\n")
    if command in ("convert", "voices"):
        event(event="progress", fraction=0.25, stage="load")
        sys.stderr.write("Downloading: 50%|#####     |"
                         + json.dumps({"event": "progress", "fraction": 0.5, "stage": "convert"}) + "\\n")
        event(event="done", command=command)
        print(json.dumps({"ok": True, "argv": args}))
        sys.exit(0)
    code = 2 if command == "fail2" else 1
    sys.stderr.write("OSError: no checkpoints in seed-vc/checkpoints and HuggingFace is offline\\n")
    event(event="error", message="engine unavailable: offline" if code == 2 else "boom", exit_code=code)
    print(json.dumps({"ok": False, "error": "x", "exit_code": code}))
    sys.exit(code)
''')


@pytest.fixture
def stub_cli(tmp_path, monkeypatch):
    stub = tmp_path / "stub_cli.py"
    stub.write_text(STUB, encoding="utf-8")
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": sys.executable, "seedvc_dir": str(tmp_path / "seed-vc"), "cli": str(stub),
        "source": "test", "reason": ""})
    monkeypatch.setattr(seedvc_env, "voices_dir", lambda: tmp_path / "voices")
    calls = []

    async def prepare(name, need_vram_mb, need_ram_mb, settle_s=3.0):
        calls.append(("prepare", name, need_vram_mb, need_ram_mb))
        return {"phase": name, "ready": True}

    async def offload(reason="task finished"):
        calls.append(("offload", reason))

    monkeypatch.setattr(gpu_handover, "prepare_for_phase", prepare)
    monkeypatch.setattr(gpu_handover, "offload_all", offload)
    return SimpleNamespace(root=tmp_path, calls=calls)


def test_run_cli_streams_progress_and_returns_the_result(stub_cli):
    seen = []
    result = seedvc_env.run_cli(["convert", "--source", "a.wav"], progress=lambda f, s: seen.append((f, s)))
    assert seen == [(0.25, "load"), (0.5, "convert")]
    assert result["ok"] is True
    assert result["argv"] == ["convert", "--source", "a.wav", "--seedvc-dir", str(stub_cli.root / "seed-vc"),
                              "--voices-dir", str(stub_cli.root / "voices")]
    assert stub_cli.calls == [("prepare", "voice_conversion", 6000, 8000),
                              ("offload", "voice conversion (convert)")]


def test_run_cli_without_gpu_work_skips_the_hand_over(stub_cli):
    seedvc_env.run_cli(["voices", "list"])
    seedvc_env.run_cli(["convert", "--dry-run"])
    assert stub_cli.calls == []


def test_run_cli_raises_on_exit_2_with_the_stderr_tail(stub_cli):
    with pytest.raises(seedvc_env.SeedVCUnavailable) as info:
        seedvc_env.run_cli(["fail2"])
    err = info.value
    assert err.exit_code == 2
    assert "engine unavailable: offline" in str(err)
    assert any("no checkpoints" in line for line in err.stderr_tail)
    assert "no checkpoints" in str(err)


def test_run_cli_raises_on_exit_1(stub_cli):
    with pytest.raises(seedvc_env.SeedVCError) as info:
        seedvc_env.run_cli(["fail1"])
    assert not isinstance(info.value, seedvc_env.SeedVCUnavailable)
    assert info.value.exit_code == 1 and "boom" in str(info.value)


def test_a_failed_conversion_still_offloads(stub_cli):
    with pytest.raises(seedvc_env.SeedVCError):
        seedvc_env.run_cli(["apply-edit", "--source", "a.wav"])
    assert [call[0] for call in stub_cli.calls] == ["prepare", "offload"]


def test_run_cli_times_out(stub_cli):
    with pytest.raises(seedvc_env.SeedVCError, match="timed out"):
        seedvc_env.run_cli(["sleep"], timeout_s=1)


def test_run_cli_refuses_while_a_render_runs(stub_cli, monkeypatch):
    from routes import rendering
    monkeypatch.setitem(rendering.jobs, "render_x", {"id": "render_x", "job_type": "render",
                                                     "status": "running"})
    with pytest.raises(seedvc_env.SeedVCBusy):
        seedvc_env.run_cli(["convert", "--source", "a.wav"])
    assert stub_cli.calls == []


def test_run_cli_when_nothing_resolves(monkeypatch):
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": None, "seedvc_dir": None, "cli": "cli.py", "source": "unavailable",
        "reason": "no seed-vc found"})
    with pytest.raises(seedvc_env.SeedVCUnavailable, match="no seed-vc found"):
        seedvc_env.run_cli(["warm"])


# --- the settings -----------------------------------------------------------------------------

def test_seedvc_settings_default_to_auto_detect_and_persist(monkeypatch):
    store = importlib.import_module("store.app_settings")   # the package re-exports the instance
    monkeypatch.setattr(store.app_settings, "_cache", {})
    monkeypatch.setattr(store.app_settings, "_save", lambda: None)
    assert store.get_seedvc_settings() == {"seedvc_dir": "", "seedvc_python": ""}
    got = store.set_seedvc_settings(seedvc_dir="  D:/engines/seed-vc ", seedvc_python=None)
    assert got == {"seedvc_dir": "D:/engines/seed-vc", "seedvc_python": ""}
    assert store.app_settings.get("seedvc_dir") == "D:/engines/seed-vc"


def test_settings_route_reports_and_sets_the_engine(places, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes import settings as settings_routes
    store = importlib.import_module("store.app_settings")   # the package re-exports the instance
    monkeypatch.setattr(store.app_settings, "_cache", {})
    monkeypatch.setattr(store.app_settings, "_save", lambda: None)
    monkeypatch.setattr(seedvc_env, "_configured", store.get_seedvc_settings)
    app = FastAPI()
    app.include_router(settings_routes.router, prefix="/api/settings")
    client = TestClient(app)

    body = client.get("/api/settings/seedvc").json()
    assert (body["seedvc_dir"], body["seedvc_python"]) == ("", "")
    assert body["resolved"]["source"] == "unavailable"

    own_dir = _make_seedvc(places.root / "custom" / "seed-vc")
    own_python = _make_python(places.root / "custom" / ".venv")
    body = client.put("/api/settings/seedvc",
                      json={"seedvc_dir": str(own_dir), "seedvc_python": str(own_python)}).json()
    assert body["seedvc_dir"] == str(own_dir) and body["resolved"]["source"] == "settings"
    assert body["resolved"]["python"] == str(own_python)
