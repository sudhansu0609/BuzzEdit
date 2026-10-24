"""asr/editor.py: a long recording is read in parts that fit the model's context."""
from types import SimpleNamespace

import pytest

from asr import editor

#: The real loader, kept before conftest stubs it for every test.
_REAL_ENSURE_LOCAL_LOADED = editor.ensure_local_loaded
from asr.utterances import table


def _utts(n, chars=120):
    return [SimpleNamespace(uid=f"U{i}", start=float(i), end=i + 0.9, pause_before=0.1,
                            text="x" * chars, notes=[], similar=[], same_opening=[], script=None,
                            word_indices=[i]) for i in range(n)]


def test_everything_in_one_part_when_it_fits():
    assert editor.parts_for(_utts(50), budget=100_000) == [(0, 0, 50)]


def test_a_long_table_is_split_with_overlap_and_nothing_skipped():
    utts = _utts(400)
    parts = editor.parts_for(utts, budget=8_000)
    assert len(parts) > 1
    assert parts[0] == (0, 0, parts[0][2]) and parts[-1][2] == 400
    for (_a, _s, e), (a2, s2, _e2) in zip(parts, parts[1:]):
        assert s2 == e and a2 == max(0, s2 - editor.PART_OVERLAP)    # contiguous, overlapping context
    room = 8_000 - editor.estimate_tokens(editor.SYSTEM) - editor.estimate_tokens(editor.request(""))
    for a, _s, e in parts:
        assert editor.estimate_tokens(table(utts[a:e])) <= room + (e - a)


def test_part_decisions_are_merged():
    a = editor.Decisions(removed_words={1: "retake"}, trim_ranges=[("U1", 1, 1)], groups=1)
    b = editor.Decisions(removed_words={9: "retake"}, trim_ranges=[("U1", 1, 1), ("U9", 9, 9)], groups=2)
    out = editor.merge_decisions(a, b)
    assert out.removed_words == {1: "retake", 9: "retake"} and out.groups == 3
    assert out.trim_ranges == [("U1", 1, 1), ("U9", 9, 9)]


def test_edit_reads_in_parts_for_a_small_context():
    utts = _utts(120)
    calls = []

    class Small:
        model = "local-test"

        def input_budget(self):
            return 4_000

        def ask(self, system, user):
            calls.append(user)
            return '{"groups": []}', {"model": "local-test", "usage": {"total_tokens": 10}, "seconds": 1.0}

    _decisions, meta = editor.edit(utts, [{} for _ in utts], Small())
    assert len(calls) > 1 and meta["parts"] == len(calls)
    assert meta["usage"]["total_tokens"] == 10 * len(calls)


def test_a_part_that_overflows_is_read_in_two():
    utts = _utts(60)
    seen = []

    class Overflowing:
        model = "local-test"

        def input_budget(self):
            return 100_000

        def ask(self, system, user):
            n = user.count("\nU")
            seen.append(n)
            if n > 45:
                raise RuntimeError("HTTP 400: the request exceeds the available context size")
            return '{"groups": []}', {"model": "local-test", "usage": {}, "seconds": 0.5}

    editor.edit(utts, [{} for _ in utts], Overflowing())
    assert max(seen) > 45 and len(seen) == 3        # the whole table once, then two halves


def test_a_copy_loaded_with_too_little_context_is_reloaded_not_reused(monkeypatch):
    # LM Studio's own just-in-time load is 16k; the editor needs 32k (2026-10-21).
    monkeypatch.setattr(editor, "ollama_loaded", lambda *a, **k: [])
    monkeypatch.setattr(editor, "lm_studio_loaded",
                        lambda *a, **k: [{"id": "google/gemma-4-12b", "context_length": 16128}])
    cfg = editor.local_config({"editor_local_model": "google/gemma-4-12b"})
    assert cfg["load"] is True and cfg["context"] == editor.LOCAL_CONTEXT

    monkeypatch.setattr(editor, "lm_studio_loaded",
                        lambda *a, **k: [{"id": "google/gemma-4-12b", "context_length": 32768}])
    cfg = editor.local_config({"editor_local_model": "google/gemma-4-12b"})
    assert cfg["load"] is False and cfg["context"] == 32768
    client = editor.EditorClient.local({"editor_local_model": "google/gemma-4-12b"})
    assert client.input_budget() == 32768 - editor.LOCAL_ANSWER_TOKENS - 512

    monkeypatch.setattr(editor, "ollama_loaded",
                        lambda *a, **k: [{"name": "gemma-4-12b:latest", "context_length": 20000}])
    cfg = editor.local_config({"editor_local_model": "google/gemma-4-12b"})
    assert cfg["backend"] == "ollama" and cfg["context"] == 20000


def test_halving_stops_when_the_error_is_not_the_parts_size():
    # A context too small or shared (LM Studio's 16k JIT load, three slots): every
    # part fails whatever its size, and halving forever wasted 36 minutes.
    utts = _utts(200)
    seen = []

    class AlwaysTooSmall:
        model = "local-test"

        def input_budget(self):
            return 100_000

        def ask(self, system, user):
            seen.append(user.count("\nU"))
            raise RuntimeError("HTTP 400: Context size has been exceeded.")

    import pytest
    with pytest.raises(RuntimeError, match="Context size"):
        editor.edit(utts, [{} for _ in utts], AlwaysTooSmall())
    assert len(seen) <= 1 + 2 + 4           # the whole table, two halves, four quarters, then stop


def test_prepare_probes_the_model_before_any_read(monkeypatch):
    # The gate editor_smoke.py was run for by hand, now made on every job: a copy
    # that answers empty (thinking on), without JSON or with too little context is
    # refused by prepare(), so the planner moves on to the next editor.
    answers = {"text": ""}
    loaded = {"context": 32768}
    monkeypatch.setattr(editor, "ensure_local_loaded", lambda cfg: {
        "base_url": "http://127.0.0.1:1234/v1", "model": "google/gemma-4-12b",
        "backend": "lm_studio", "context": loaded["context"], "load": False})
    asked = []
    monkeypatch.setattr(editor.EditorClient, "ask",
                        lambda self, system, user, retries=1: (asked.append(user), (answers["text"], {"seconds": 1}))[1])
    client = editor.EditorClient.local({"editor_local_model": "google/gemma-4-12b"})
    with pytest.raises(RuntimeError, match="without JSON"):
        client.prepare()
    assert len(asked) == 1 and "The recording:" in asked[0] and not client._prepared

    answers["text"] = 'Here you go: {"groups": [], "trims": [], "drops": []}'
    client.prepare()
    assert client._prepared and len(asked) == 2
    client.prepare()
    assert len(asked) == 2            # prepared once per client

    loaded["context"] = 16128
    small = editor.EditorClient.local({"editor_local_model": "google/gemma-4-12b"})
    with pytest.raises(RuntimeError, match="16128 tokens of context"):
        small.prepare()
    assert len(asked) == 2            # not even asked


def test_loading_the_editor_model_first_clears_the_card(monkeypatch):
    # The cut follows the transcription (Whisper resident) and may follow the
    # previous video's render (ComfyUI's Wan experts resident): both go first.
    order = []
    monkeypatch.setattr(editor, "_free_card_for_the_editor", lambda: order.append("free"))
    monkeypatch.setattr(editor, "_unload_ollama", lambda *a, **k: order.append("ollama"))
    import llm.lm_launcher as launcher
    monkeypatch.setattr(launcher, "find_lms_cli", lambda: "lms")

    def run(cmd, **kw):
        order.append(" ".join(cmd[1:3]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import subprocess
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(editor, "lm_studio_loaded",
                        lambda *a, **k: [{"id": "google/gemma-4-12b", "context_length": 32768}])
    # Under the lock the loader looks again: nothing usable is up, so it loads.
    monkeypatch.setattr(editor, "local_config", lambda settings=None: {"load": True, "model": "google/gemma-4-12b",
                                                                        "backend": "lm_studio", "context": 32768})
    cfg = _REAL_ENSURE_LOCAL_LOADED({"load": True, "model": "google/gemma-4-12b", "backend": "lm_studio",
                                      "base_url": "http://127.0.0.1:1234/v1"})
    assert order == ["free", "ollama", "unload --all", "load google/gemma-4-12b"]
    assert cfg["context"] == 32768 and cfg["load"] is False

    # A copy another loader put up while we waited for the lock is reused, not ejected.
    order.clear()
    monkeypatch.setattr(editor, "local_config", lambda settings=None: {
        "load": False, "model": "google/gemma-4-12b", "backend": "lm_studio", "context": 32768,
        "base_url": "http://127.0.0.1:1234/v1"})
    cfg = _REAL_ENSURE_LOCAL_LOADED({"load": True, "model": "google/gemma-4-12b", "backend": "lm_studio",
                                      "base_url": "http://127.0.0.1:1234/v1"})
    assert order == [] and cfg["load"] is False and cfg["context"] == 32768


def test_a_read_whose_model_was_unloaded_reloads_it_and_reads_again(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    client = editor.EditorClient(base_url="http://127.0.0.1:1234/v1", model="google/gemma-4-12b",
                                 effort="", backend="lm_studio")
    client._prepared = True
    calls = {"n": 0, "prepared": 0}

    def fake_prepare():
        calls["prepared"] += 1
        client._prepared = True

    monkeypatch.setattr(client, "prepare", fake_prepare)

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                         io.BytesIO(b'{"error":"Model unloaded by user or API request."}'))
        return Resp(b'{"model":"google/gemma-4-12b","choices":[{"message":{"content":"[]"}}],"usage":{}}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    answer, meta = client.ask("sys", "user", retries=0)
    assert calls["prepared"] == 1 and calls["n"] == 2
    assert answer == "[]" and meta["model"] == "google/gemma-4-12b"

