import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import (
    APP_NAME,
    APP_VERSION,
    BIND_HOST,
    OUTPUT_DIR,
    PORT_SPAN,
    PREFERRED_PORT,
    comfyui_port,
    set_bound_port,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

#: True when *this* process owns the ledger entry, i.e. it stepped forward and
#: published for itself. Under Electron the launcher owns the entry (it is the
#: only one that knows which ComfyUI belongs to this backend), and we must never
#: delete somebody else's record on the way out. `_serve()` sets this.
_owns_ledger_entry = False


def _withdraw_ledger_entry() -> None:
    """Take `buzzedit` back out of the shared ledger, at most once."""
    global _owns_ledger_entry
    if not _owns_ledger_entry:
        return
    _owns_ledger_entry = False
    try:
        import buzzcaf_ports

        buzzcaf_ports.withdraw(APP_NAME)
        logger.info("Withdrew %s from the port ledger.", APP_NAME)
    except Exception as exc:  # a stale entry is bad; a crash on the way out is worse
        logger.warning("Could not withdraw %s from the port ledger: %s", APP_NAME, exc)


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
        # Out of the ledger *here*, not only at `atexit`. On Windows a console
        # control event - Ctrl+Break, closing the console window, the machine
        # shutting down - ends the process from the CRT's default handler as
        # soon as uvicorn's own handler returns, so `atexit` and the `finally`
        # around `server.run()` never run and the entry outlives the app.
        # This block is part of uvicorn's *application shutdown*, which happens
        # before that, and is the last point still reliably ours. Found in the
        # P4 integration run: a graceful stop left `buzzedit -> 8100` in the
        # ledger pointing at a pid that no longer existed.
        _withdraw_ledger_entry()
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
    return {"message": "BuzzEdit API", "version": APP_VERSION}


def _serve() -> None:
    """Run the API, on a port nobody else holds.

    GUARDIAN_PLAN.md section 11. Two ways in, and they differ only in who owns
    the ledger entry:

    * **Under Electron.** `electron/main.cjs` has already picked a free port,
      passes it as ``--port``, waits for `/api/health` to answer with
      ``app: "buzzedit"`` and publishes the entry itself — because only it
      knows which ComfyUI went with this backend, and that belongs in `extra`.
      So we bind exactly what we were told and publish nothing.

    * **On its own** (`python backend/main.py`, `npm run dev:backend`, a test).
      Nobody else is going to do it, so we step forward from the preferred port
      ourselves and publish/withdraw our own entry.

    Either way the port is never assumed: `BUZZEDIT_PORT` is a wish.
    """
    import argparse
    import atexit
    import json
    import threading
    import time
    import urllib.request

    import uvicorn

    import buzzcaf_ports

    def answers_as_buzzedit(url: str) -> bool:
        """Rule 4, applied to ourselves before we tell the family where we are."""
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status != 200:
                    return False
                body = json.loads(response.read(65536).decode("utf-8", "replace"))
        except Exception:
            return False
        return isinstance(body, dict) and body.get("app") == APP_NAME

    parser = argparse.ArgumentParser(description="BuzzEdit API server")
    parser.add_argument(
        "--port", type=int, default=None,
        help="bind exactly this port (the launcher has already checked it is free)",
    )
    parser.add_argument("--host", default=BIND_HOST)
    args = parser.parse_args()

    launcher_chose = args.port is not None
    if launcher_chose:
        port = int(args.port)
    else:
        port = buzzcaf_ports.pick_port(PREFERRED_PORT, PORT_SPAN, args.host)
        if port != PREFERRED_PORT:
            logger.info("Port %s was busy; taking %s instead.", PREFERRED_PORT, port)

    set_bound_port(port)
    health = f"http://{args.host}:{port}/api/health"

    # `main:app` (the import string) would re-import this module in a fresh
    # namespace and lose the port we just recorded, so the app object goes over
    # directly. `uvicorn.Server` rather than `uvicorn.run` because publishing
    # has to wait for `server.started`.
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=port, reload=False))

    if not launcher_chose:
        def _publish_when_serving() -> None:
            # Not the lifespan hook: uvicorn runs lifespan startup *before* it
            # binds the socket, and an entry written before the socket opens is
            # a promise nobody can keep.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if server.started and answers_as_buzzedit(health):
                    buzzcaf_ports.publish(
                        APP_NAME, port, health, {"comfyui": comfyui_port()},
                    )
                    logger.info("Published %s -> %s in the port ledger.", APP_NAME, port)
                    return
                time.sleep(0.25)
            logger.warning("Gave up publishing %s to the port ledger.", APP_NAME)

        threading.Thread(target=_publish_when_serving, daemon=True).start()
        # The lifespan shutdown above is the primary path; this is the backstop
        # for the ways a process ends without one (a SystemExit raised before
        # the server ever serves, an early crash).
        global _owns_ledger_entry
        _owns_ledger_entry = True
        atexit.register(_withdraw_ledger_entry)

    logger.info("BuzzEdit backend listening on http://%s:%s", args.host, port)
    server.run()


if __name__ == "__main__":
    _serve()
