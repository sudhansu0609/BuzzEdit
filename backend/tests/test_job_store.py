"""Durable job state (STUDIO_DIRECTOR_PLAN.md enhancement 2, Fable review row 7):
`runtime/job_store.py` persists reel/narration jobs to disk and marks a stale
"queued"/"running" job failed at startup."""
from runtime import job_store


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path)
    job_store.save("reels", "abc123", {"job_id": "abc123", "state": "running", "progress": 0.4})
    loaded = job_store.load("reels", "abc123")
    assert loaded == {"job_id": "abc123", "state": "running", "progress": 0.4}


def test_load_of_unknown_job_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path)
    assert job_store.load("reels", "nope") is None


def test_all_jobs_skips_unreadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path)
    job_store.save("narration", "ok1", {"job_id": "ok1", "state": "done"})
    bad_dir = tmp_path / "narration"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "broken.json").write_text("{not json", encoding="utf-8")
    jobs = job_store.all_jobs("narration")
    assert set(jobs) == {"ok1"}


def test_sweep_stale_marks_queued_and_running_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path)
    job_store.save("reels", "queued1", {"job_id": "queued1", "state": "queued"})
    job_store.save("reels", "running1", {"job_id": "running1", "state": "running", "progress": 0.6})
    job_store.save("reels", "done1", {"job_id": "done1", "state": "done", "result": {"video": "x.mp4"}})
    job_store.save("reels", "failed1", {"job_id": "failed1", "state": "failed", "error": "boom"})

    marked = job_store.sweep_stale("reels")

    assert marked == 2
    assert job_store.load("reels", "queued1")["state"] == "failed"
    assert job_store.load("reels", "queued1")["error"] == job_store.RESTARTED_ERROR
    assert job_store.load("reels", "running1")["state"] == "failed"
    assert job_store.load("reels", "running1")["error"] == job_store.RESTARTED_ERROR
    # Already-terminal jobs are left exactly as they were.
    assert job_store.load("reels", "done1")["state"] == "done"
    assert job_store.load("reels", "done1")["result"] == {"video": "x.mp4"}
    assert job_store.load("reels", "failed1")["error"] == "boom"


def test_sweep_stale_is_a_noop_with_nothing_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path)
    assert job_store.sweep_stale("reels") == 0
