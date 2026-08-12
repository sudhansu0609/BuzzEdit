import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def tmp_job_store_path(tmp_path):
    return tmp_path / "test_jobs.json"

@pytest.fixture
def tmp_project_dir(tmp_path):
    p_dir = tmp_path / "projects"
    p_dir.mkdir(parents=True, exist_ok=True)
    return p_dir
