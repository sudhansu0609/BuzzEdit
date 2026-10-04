"""tools/autocut_mark.py + tools/autocut_eval.py's --no-rerun path, round
tripped through a real (temp) project on disk. No transcription, no model --
both just read `timeline.words` that a project already has saved.
"""

import asyncio
import json

import pytest

import config
from store.project_store import ProjectStore
from tools import autocut_eval, autocut_mark


def _word(start_s, end_s, enabled=True, reason=None, fps=30):
    return {"id": f"w{start_s}", "text": "x", "start_frame": round(start_s * fps),
           "end_frame": round(end_s * fps), "enabled": enabled, "reason": reason}


@pytest.fixture
def project(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    eval_dir = tmp_path / "eval"
    monkeypatch.setattr(config, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    store = ProjectStore(base_dir=str(projects_dir))
    words = [
        _word(0.0, 1.0, enabled=True),
        _word(1.0, 2.0, enabled=False, reason="retake"),
        _word(2.0, 3.0, enabled=True),
    ]
    store.save_project("proj1", {
        "id": "proj1", "name": "proj1", "source_video": "irrelevant.mp4",
        "status": "transcribed",
        "timeline": {"fps_num": 30, "fps_den": 1, "words": words},
    })
    return "proj1", eval_dir


def test_autocut_mark_exports_the_stored_cut_list(project):
    project_id, eval_dir = project
    out_path = autocut_mark.export_ground_truth(project_id, "roundtrip")
    assert out_path == eval_dir / "roundtrip.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == project_id
    assert payload["spans"] == [{"start": 1.0, "end": 2.0, "kind": "retake"}]


def test_no_rerun_scores_the_same_stored_plan_it_was_marked_from(project):
    project_id, eval_dir = project
    autocut_mark.export_ground_truth(project_id, "roundtrip")

    result = asyncio.run(autocut_eval.run_and_score(project_id, "roundtrip", no_rerun=True))

    assert result["no_rerun"] is True
    assert len(result["runs"]) == 1
    # Ground truth == the predicted plan it was exported from -> a perfect score.
    assert result["runs"][0]["span"]["recall"] == 1.0
    assert result["runs"][0]["span"]["precision"] == 1.0
    assert result["aggregate"]["span"]["recall"]["mean"] == 1.0
