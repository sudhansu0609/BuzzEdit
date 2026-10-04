import sys
import pytest
from pathlib import Path

backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

@pytest.fixture
def tmp_job_store_path(tmp_path):
    return tmp_path / "test_jobs.json"

@pytest.fixture
def tmp_project_dir(tmp_path):
    p_dir = tmp_path / "projects"
    p_dir.mkdir(parents=True, exist_ok=True)
    return p_dir


@pytest.fixture(autouse=True)
def _no_real_gpu_handover(monkeypatch):
    """Never run `lms unload --all` or wait on the real card from a test: the
    hand-over (runtime/gpu_handover.py) reports a clear card instantly. A test
    that wants a starved card patches `prepare_for_phase` itself."""
    from runtime import gpu_handover

    async def _clear(name, need_vram_mb, need_ram_mb, settle_s=3.0):
        return {"phase": name, "ready": True, "free_vram_mb": 16000, "available_ram_mb": 32000,
                "need_vram_mb": need_vram_mb, "need_ram_mb": need_ram_mb, "reason": ""}

    async def _no_eject():
        return False

    async def _no_offload(reason="task finished"):
        return None

    monkeypatch.setattr(gpu_handover, "prepare_for_phase", _clear)
    monkeypatch.setattr(gpu_handover, "eject_lm_studio", _no_eject)
    monkeypatch.setattr(gpu_handover, "eject_ollama", _no_eject)
    monkeypatch.setattr(gpu_handover, "offload_all", _no_offload)


@pytest.fixture(autouse=True)
def _a_clear_card(monkeypatch):
    from runtime import gpu_handover
    monkeypatch.setattr(gpu_handover, "free_vram_mb", lambda: 16000.0)
    monkeypatch.setattr(gpu_handover, "lm_studio_has_models", lambda base_url="": False)
