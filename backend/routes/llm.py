from fastapi import APIRouter
from llm import lm_studio_client, get_status, ensure_ready
from llm.lm_launcher import catalogue
from llm.client import AUTO_MODEL
from store.app_settings import app_settings

router = APIRouter()


@router.get("/status")
async def llm_status():
    """Report whether LM Studio is up and the model is loaded (no side effects)."""
    return await get_status(lm_studio_client.base_url, lm_studio_client.model_name)


@router.get("/models")
async def llm_models():
    """Every text model LM Studio knows about, so the UI can offer a picker.

    `selected` is the saved preference, or `"auto"` (the default) which means
    "use whichever local model LM Studio has loaded". If LM Studio is not running
    the catalogue is empty but we still return `selected`/`default` so the
    dropdown shows the saved choice rather than going blank.
    """
    entries = await catalogue(lm_studio_client.base_url)
    models = [m for m in entries if m.get("type") != "embeddings"]
    return {
        "models": models,
        "selected": app_settings.get("llm_model") or AUTO_MODEL,
        "default": AUTO_MODEL,
    }


@router.post("/ensure")
async def llm_ensure():
    """Auto-start LM Studio and load the model if needed, then report status."""
    ready = await ensure_ready(lm_studio_client.base_url, lm_studio_client.model_name)
    status = await get_status(lm_studio_client.base_url, lm_studio_client.model_name)
    status["ready"] = ready
    return status
