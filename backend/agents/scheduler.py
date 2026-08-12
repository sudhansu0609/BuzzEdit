import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import DATA_DIR, OUTPUT_DIR, PROJECTS_DIR
from utils.power import PowerManager
from agents.edit_agent import EditAgent

logger = logging.getLogger(__name__)

JOBS_FILE = DATA_DIR / "jobs.json"


class JobScheduler:
    def __init__(self):
        self.jobs: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.is_paused: bool = False
        self.current_job_id: Optional[str] = None
        self.edit_agent = EditAgent()
        self._worker_task: Optional[asyncio.Task] = None
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
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "error": None,
            "result": None,
            "settings": settings or {},
        }

        self.jobs.append(job)
        self.sort_queue()
        self._save_jobs()
        logger.info(f"Scheduler: Enqueued job {job_id} for project {project_id} (priority: {priority})")
        return job

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

            pending_jobs = [j for j in self.jobs if j["status"] == "pending"]

            if pending_jobs:
                PowerManager.prevent_sleep(keep_display_on=False)
                job = pending_jobs[0]
                self.current_job_id = job["id"]
                job["status"] = "running"
                job["updated_at"] = datetime.now().isoformat()
                self._save_jobs()

                logger.info(f"Scheduler: Processing job {job['id']} for project {job['project_id']}")

                try:
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

        results = await self.edit_agent.execute_full_auto_edit(
            project=project,
            generate_broll=job_settings.get("generate_broll", True),
            generate_thumbnail=job_settings.get("generate_thumbnail", True),
            burn_captions=job_settings.get("burn_captions", True),
        )

        job["result"] = {
            "output_directory": str(job_output_dir),
            "results": results,
        }
