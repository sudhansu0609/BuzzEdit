"""agents/scheduler.py: cancelling a running job stops it (2026-10-21: the
Studio cancelled a production and BuzzEdit kept generating for it)."""
import asyncio

import pytest

from agents import scheduler as sched_mod


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    monkeypatch.setattr(sched_mod, "JOBS_FILE", tmp_path / "jobs.json")
    monkeypatch.setattr(sched_mod.PowerManager, "prevent_sleep", staticmethod(lambda **k: None))
    monkeypatch.setattr(sched_mod.PowerManager, "restore_sleep", staticmethod(lambda: None))
    s = sched_mod.JobScheduler.__new__(sched_mod.JobScheduler)
    s.jobs, s.is_running, s.is_paused, s.current_job_id = [], False, False, None
    s._worker_task, s._last_progress_save, s._current_task, s._cancelling = None, 0.0, None, None
    s.edit_agent = None
    return s


def test_cancelling_the_running_job_stops_it_and_the_loop_goes_on(scheduler, monkeypatch):
    seen = []

    async def slow(job):
        seen.append(("start", job["id"]))
        await asyncio.sleep(10)
        seen.append(("done", job["id"]))

    async def no_comfy():
        seen.append(("interrupt",))

    monkeypatch.setattr(scheduler, "_process_job", slow)
    monkeypatch.setattr(scheduler, "_interrupt_comfyui", no_comfy)
    monkeypatch.setattr(sched_mod, "_is_due", lambda j: True)
    scheduler.jobs = [{"id": "job_a", "project_id": "p", "status": "pending", "progress": 0.0},
                      {"id": "job_b", "project_id": "p", "status": "pending", "progress": 0.0}]

    async def run():
        scheduler.is_running = True
        worker = asyncio.create_task(scheduler._worker_loop())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if scheduler.current_job_id == "job_a":
                break
        assert scheduler.cancel_job("job_a")
        for _ in range(100):
            await asyncio.sleep(0.01)
            if scheduler.current_job_id == "job_b":
                break
        assert scheduler.jobs[0]["status"] == "cancelled"
        assert scheduler.current_job_id == "job_b", "the loop went on to the next job"
        scheduler.is_running = False
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert ("start", "job_a") in seen and ("done", "job_a") not in seen
    assert ("interrupt",) in seen
