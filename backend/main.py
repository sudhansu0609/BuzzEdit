import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BuzzEdit backend...")
    # Start the nightly job scheduler here rather than via a router startup
    # event (deprecated in FastAPI, and only worked under the lifespan shim).
    # Imported locally: the routes package is imported further down this module.
    from routes.scheduler import scheduler_instance
    scheduler_instance.start()
    try:
        yield
    finally:
        scheduler_instance.stop()

app = FastAPI(
    title="BuzzEdit API",
    description="AI-powered video editing backend with single-pass EDL compiler",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

from routes import (
    projects,
    transcription,
    analysis,
    rendering,
    health,
    comfyui,
    agents,
    scheduler,
    system,
    advanced,
    media,
    timeline,
    settings,
    llm,
    presets,
    style,
    presentation,
    script,
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(media.router, tags=["media"])
app.include_router(timeline.router, prefix="/api/timeline", tags=["timeline"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(transcription.router, prefix="/api/transcription", tags=["transcription"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(rendering.router, prefix="/api/rendering", tags=["rendering"])
app.include_router(comfyui.router, prefix="/api/comfyui", tags=["comfyui"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(scheduler.router, prefix="/api/scheduler", tags=["scheduler"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(advanced.router, prefix="/api/advanced", tags=["advanced"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])
app.include_router(presets.router, prefix="/api/presets", tags=["presets"])
app.include_router(style.router, prefix="/api/style", tags=["style"])
app.include_router(presentation.router, prefix="/api/presentation", tags=["presentation"])
app.include_router(script.router, prefix="/api/projects", tags=["script"])

@app.get("/")
async def root():
    return {"message": "BuzzEdit API", "version": "0.2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8099, reload=False)
