import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
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
    logger.info("Starting Buzzcaf Media Editor backend...")
    yield

app = FastAPI(
    title="Buzzcaf Media Editor API",
    description="AI-powered video editing backend",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

from routes import projects, transcription, analysis, rendering, health, comfyui, agents, scheduler, system, advanced

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(transcription.router, prefix="/api/transcription", tags=["transcription"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(rendering.router, prefix="/api/rendering", tags=["rendering"])
app.include_router(comfyui.router, prefix="/api/comfyui", tags=["comfyui"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(scheduler.router, prefix="/api/scheduler", tags=["scheduler"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(advanced.router, prefix="/api/advanced", tags=["advanced"])






import uvicorn

@app.get("/")
async def root():
    return {"message": "Buzzcaf Media Editor API", "version": "0.1.0"}


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8099, reload=False)

