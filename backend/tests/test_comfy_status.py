"""electron/main.cjs writes data/comfy_status.json; the backend only reads it,
and must never fail on a missing or garbage file."""

from runtime import comfy_status


def test_missing_file_reports_no_opinion(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_status, "STATUS_PATH", tmp_path / "comfy_status.json")

    assert comfy_status.read_status() == {}
    assert comfy_status.comfy_state_and_detail() == (None, None)


def test_valid_file_reports_state_and_detail(tmp_path, monkeypatch):
    status_path = tmp_path / "comfy_status.json"
    status_path.write_text(
        '{"state": "repairing", "since": "2026-10-20T03:00:00", '
        '"detail": "python not found", "repair": {"state": "repairing"}, '
        '"updated_at": "2026-10-20T03:05:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(comfy_status, "STATUS_PATH", status_path)

    data = comfy_status.read_status()
    assert data["state"] == "repairing"
    assert data["repair"] == {"state": "repairing"}
    assert comfy_status.comfy_state_and_detail() == ("repairing", "python not found")


def test_garbage_file_reports_no_opinion(tmp_path, monkeypatch):
    status_path = tmp_path / "comfy_status.json"
    status_path.write_text("not json at all {{{", encoding="utf-8")
    monkeypatch.setattr(comfy_status, "STATUS_PATH", status_path)

    assert comfy_status.read_status() == {}
    assert comfy_status.comfy_state_and_detail() == (None, None)


def test_non_object_json_reports_no_opinion(tmp_path, monkeypatch):
    status_path = tmp_path / "comfy_status.json"
    status_path.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(comfy_status, "STATUS_PATH", status_path)

    assert comfy_status.read_status() == {}
    assert comfy_status.comfy_state_and_detail() == (None, None)
