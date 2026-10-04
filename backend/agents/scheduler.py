import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import DATA_DIR, OUTPUT_DIR, PROJECTS_DIR
from utils.power import PowerManager
from runtime import sentinel_priority
from agents.edit_agent import EditAgent

logger = logging.getLogger(__name__)

JOBS_FILE = DATA_DIR / "jobs.json"

# How often a running job's progress is written to disk.
PROGRESS_SAVE_INTERVAL = 2.0


def resolve_start_at(start_at: Optional[str]) -> Optional[str]:
    """Turn "23:00" or an ISO timestamp into an absolute ISO instant.

    A bare time means the next occurrence of it: later today if it is still
    ahead, otherwise tomorrow. Returns None for "run as soon as you can", which
    is what an empty field means.
    """
    if not start_at:
        return None
    text = str(start_at).strip()
    if not text:
        return None
    try:
        if ":" in text and len(text) <= 5:
            hour, minute = (int(part) for part in text.split(":", 1))
            now = datetime.now()
            when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if when <= now:
                when += timedelta(days=1)
            return when.isoformat()
        return datetime.fromisoformat(text).isoformat()
    except Exception:
        logger.warning("Scheduler: could not read start time %r; running immediately", start_at)
        return None


def _is_due(job: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Whether a pending job is allowed to start yet."""
    start_at = job.get("start_at")
    if not start_at:
        return True
    try:
        return datetime.fromisoformat(start_at) <= (now or datetime.now())
    except Exception:
        return True


class JobScheduler:
    def __init__(self):
        self.jobs: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.is_paused: bool = False
        self.current_job_id: Optional[str] = None
        self.edit_agent = EditAgent()
        self._worker_task: Optional[asyncio.Task] = None
        self._last_progress_save: float = 0.0
        self._load_jobs()

    def _load_jobs(self):
        if JOBS_FILE.exists():
            try:
                data = json.loads(JOBS_FILE.read_text())
                self.jobs = data.get("jobs", [])
                # Reset stuck 'running' jobs to 'pending' on startup
                for j in self.jobs:
                    if j.get("status") == "running":
                        j["status"] = "pending"
            except Exception as e:
                logger.error(f"Failed to load jobs.json: {e}")
                self.jobs = []
        else:
            self.jobs = []

    def _save_jobs(self):
        try:
            JOBS_FILE.write_text(json.dumps({"jobs": self.jobs}, indent=2))
        except Exception as e:
            logger.error(f"Failed to save jobs.json: {e}")

    def enqueue_job(
        self,
        project_id: str,
        job_type: str = "full_edit",
        priority: str = "normal",
        settings: Optional[Dict[str, Any]] = None,
        start_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        import uuid
        job_id = f"job_{str(uuid.uuid4())[:8]}"

        priority_order = {"urgent": 0, "normal": 1, "low": 2}
        p_val = priority_order.get(priority, 1)

        job = {
            "id": job_id,
            "project_id": project_id,
            "job_type": job_type,
            "priority": priority,
            "priority_val": p_val,
            "status": "pending",
            "progress": 0.0,
            "message": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            # Resolved to an absolute instant at enqueue time, deliberately: a job
            # queued at 23:30 asking for "23:00" must run tomorrow, not straight
            # away, and must still mean the same moment after a restart.
            "start_at": resolve_start_at(start_at),
            "error": None,
            "result": None,
            "settings": settings or {},
        }

        self.jobs.append(job)
        self.sort_queue()
        self._save_jobs()
        logger.info("Scheduler: Enqueued %s job %s for project %s (priority %s%s)",
                    job_type, job_id, project_id, priority,
                    f", starting {job['start_at']}" if job["start_at"] else "")
        return job

    def _progress(self, job: Dict[str, Any], fraction: float, message: str = "") -> None:
        """Record how far a job has got, throttled.

        An overnight run reports progress hundreds of times; rewriting the whole
        jobs file on each one is pointless disk churn. The in-memory value always
        updates so the API is current — only the write is rate limited.
        """
        job["progress"] = max(0.0, min(1.0, float(fraction)))
        if message:
            job["message"] = message
        now = time.monotonic()
        if now - self._last_progress_save >= PROGRESS_SAVE_INTERVAL:
            self._last_progress_save = now
            job["updated_at"] = datetime.now().isoformat()
            self._save_jobs()

    def sort_queue(self):
        # Sort pending jobs by priority_val then created_at
        pending = [j for j in self.jobs if j["status"] == "pending"]
        other = [j for j in self.jobs if j["status"] != "pending"]

        pending.sort(key=lambda j: (j.get("priority_val", 1), j.get("created_at", "")))
        self.jobs = pending + other

    def cancel_job(self, job_id: str) -> bool:
        for j in self.jobs:
            if j["id"] == job_id:
                if j["status"] == "running":
                    j["status"] = "cancelled"
                else:
                    j["status"] = "cancelled"
                j["updated_at"] = datetime.now().isoformat()
                self._save_jobs()
                logger.info(f"Scheduler: Cancelled job {job_id}")
                return True
        return False

    def clear_completed(self):
        self.jobs = [j for j in self.jobs if j["status"] in ("pending", "running")]
        self._save_jobs()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Scheduler: Background worker loop started")

    def stop(self):
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
        PowerManager.restore_sleep()

    async def _worker_loop(self):
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(2)
                continue

            # A job waiting for its start time is not work yet. Taking only jobs
            # that are *due* is what lets the machine idle until 11pm instead of
            # being held awake from the moment something is queued.
            pending_jobs = [j for j in self.jobs
                            if j["status"] == "pending" and _is_due(j)]

            if pending_jobs:
                PowerManager.prevent_sleep(keep_display_on=False)
                job = pending_jobs[0]
                self.current_job_id = job["id"]
                job["status"] = "running"
                job["updated_at"] = datetime.now().isoformat()
                self._save_jobs()

                logger.info(f"Scheduler: Processing job {job['id']} for project {job['project_id']}")

                try:
                    # While a job runs, the backend and ComfyUI outrank
                    # background work in Sentinel (runtime/sentinel_priority.py).
                    with sentinel_priority.job():
                        await self._process_job(job)
                    job["status"] = "completed"
                    job["progress"] = 1.0
                except Exception as e:
                    logger.error(f"Scheduler: Job {job['id']} failed: {e}")
                    job["status"] = "failed"
                    job["error"] = str(e)
                finally:
                    job["updated_at"] = datetime.now().isoformat()
                    self.current_job_id = None
                    self._save_jobs()
            else:
                PowerManager.restore_sleep()
                await asyncio.sleep(2)

    async def _process_job(self, job: Dict[str, Any]):
        from store.project_store import ProjectStore
        from models import Project

        project_id = job["project_id"]
        store = ProjectStore(base_dir=str(PROJECTS_DIR))
        p_data = store.get_project(project_id)

        if not p_data:
            raise FileNotFoundError(f"Project data not found for {project_id}")

        project = Project.model_validate(p_data)

        # Date-organized output folder: output/YYYY-MM-DD/project_name/
        today_str = datetime.now().strftime("%Y-%m-%d")
        safe_proj_name = "".join([c for c in project.name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
        job_output_dir = OUTPUT_DIR / today_str / safe_proj_name
        job_output_dir.mkdir(parents=True, exist_ok=True)

        job_settings = job.get("settings", {})
        report = lambda fraction, message="": self._progress(job, fraction, message)

        if job.get("job_type") == "presentation_render":
            # Only the render of a pass that already saved its timeline -- the
            # retry after a render that died. See director.rerender_presented.
            from presentation.director import rerender_presented
            from runtime import gpu_handover
            try:
                result = await rerender_presented(project_id, progress_cb=report,
                                                  output_dir=job_output_dir)
            finally:
                await gpu_handover.offload_all(f"re-render {project_id} finished")
            job["result"] = {
                "output_directory": str(job_output_dir),
                "report": result.model_dump(),
                "output_path": result.output_path,
            }
            return

        if job.get("job_type") == "presentation":
            # The full overnight pass: reads the transcript, generates and places
            # B-roll, zooms, pop-ups, captions and a thumbnail, then renders.
            from presentation import PresentationSettings, run_presentation_pass
            from presentation.models import settings_sources_for
            from store.app_settings import apply_presentation_overrides

            # BuzzEdit-side overrides beat whatever BuzzcafStudio sent, applied
            # fresh here (not just at enqueue time) so a delayed overnight job
            # picks up an override changed after it was queued.
            merged_settings, override_fields = apply_presentation_overrides(job_settings)
            settings = PresentationSettings(**merged_settings).resolve_density()
            settings_sources = settings_sources_for(job_settings, override_fields, settings)
            result = await run_presentation_pass(
                project_id, settings, progress_cb=report, output_dir=job_output_dir,
                settings_sources=settings_sources)
            job["result"] = {
                "output_directory": str(job_output_dir),
                "report": result.model_dump(),
                "output_path": result.output_path,
            }
            return

        results = await self.edit_agent.execute_full_auto_edit(
            project=project,
            generate_thumbnail=job_settings.get("generate_thumbnail", True),
            burn_captions=job_settings.get("burn_captions", True),
            progress_cb=report,
            output_dir=job_output_dir,
        )

        job["result"] = {
            "output_directory": str(job_output_dir),
            "results": results,
        }
