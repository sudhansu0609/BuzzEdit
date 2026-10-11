import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

class ProjectStore:
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Default to backend/data/projects
            base_dir = str(Path(__file__).parent.parent / "data" / "projects")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._readonly_cache: Dict[str, Any] = {}

    def _get_project_file(self, project_id: str) -> Path:
        return self.base_dir / f"{project_id}.json"

    def save_project(self, project_id: str, data: Dict[str, Any]) -> None:
        """Atomically save project data to disk."""
        target_path = self._get_project_file(project_id)
        temp_fd, temp_path = tempfile.mkstemp(dir=str(self.base_dir), prefix="proj_tmp_")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # Replace target file atomically
            os.replace(temp_path, target_path)
            self._readonly_cache.pop(project_id, None)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project data from disk."""
        target_path = self._get_project_file(project_id)
        if not target_path.exists():
            return None
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_project_readonly(self, project_id: str) -> Optional[Dict[str, Any]]:
        """The project as `get_project` returns it, parsed once per saved version.

        For readers that only look (thumbnails, waveforms, filmstrips): the dict
        is shared, so a caller must not mutate it. Opening a presented project
        asks for a thumbnail and two filmstrips per library entry -- 275 entries
        on a Raat3Baje cut -- and parsing the 10 MB file for each kept the server
        busy for 20 s while the preview showed "Loading video stream" (2026-10-08).
        Keyed on the file's mtime and size, so a save is seen at once.
        """
        target_path = self._get_project_file(project_id)
        try:
            st = target_path.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        cached = self._readonly_cache.get(project_id)
        if cached and cached[0] == key:
            return cached[1]
        data = self.get_project(project_id)
        if data is not None:
            self._readonly_cache[project_id] = (key, data)
        return data

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in base directory."""
        projects: List[Dict[str, Any]] = []
        for p_file in self.base_dir.glob("*.json"):
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    projects.append(data)
            except Exception:
                continue
        # Sort by updated_at if present
        projects.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete project from disk."""
        target_path = self._get_project_file(project_id)
        if target_path.exists():
            target_path.unlink()
            return True
        return False
