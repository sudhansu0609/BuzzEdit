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
