"""Builds the FastAPI app: middleware, routers, and what happens at startup and shutdown."""
import os

from fastapi import FastAPI

from src.api import auth as authmod
from src.api.bootstrap import bootstrap_users
from src.api.middleware import install_middleware
from src.api.routes import accounts, admin, auth, health, patients, reports, reviews, uploads
from src.core.config import APP_VERSION
from src.core.llm import required_key_name
from src.db.session import create_tables
from src.monitoring import observability as obs
from src.pipeline import workers


def create_app() -> FastAPI:
    obs.configure_logging()   # one log format everywhere, carrying the request id and job id

    app = FastAPI(
        title="Agentic Medical Assistant API",
        description="Multi-Agent Clinical Intelligence Platform",
        version=APP_VERSION,
    )
    install_middleware(app)
    for module in (health, auth, accounts, uploads, reviews, reports, patients, admin):
        app.include_router(module.router)

    @app.on_event("startup")
    async def startup():
        # Fail fast and loud on missing secrets rather than 500ing on first
        # request — much easier to diagnose in a demo / interview setting.
        required = [required_key_name()]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                f"See backend/.env.example."
            )
        authmod.validate_auth_config()
        create_tables()
        bootstrap_users()
        await workers.start_job_worker()
        await workers.start_outbox_relay()

    @app.on_event("shutdown")
    async def shutdown():
        # Running jobs are handed back to the queue (not lost, not counted as an attempt).
        if workers.job_worker is not None:
            await workers.job_worker.stop()
        if workers.outbox_relay is not None:
            await workers.outbox_relay.stop()

    return app
