"""Browser access (CORS), response security headers and per-request correlation ids."""
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.monitoring import observability as obs


def install_middleware(app: FastAPI) -> None:
    """Order matters: the last one added is the outermost, so a request meets request_context first,
    then security_headers, then CORS."""
    # Allowed browser origins. Entries are trimmed (a stray space after a comma in .env used to
    # silently break matching). localhost / 127.0.0.1 on ANY port are also accepted so local
    # development works whichever way the dev server is opened; set
    # ALLOW_LOCALHOST_ORIGINS=false in production so only CORS_ORIGINS is trusted.
    cors_origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]
    localhost_regex = (
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
        if os.getenv("ALLOW_LOCALHOST_ORIGINS", "true").lower() != "false" else None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=localhost_regex,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Defence-in-depth response headers. Responses carry patient data, so they must
        not be cached by browsers or shared proxies, sniffed, or framed."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Give every request a correlation id (logged on every line, returned as X-Request-ID) and
        count it. The metric's route label is the route TEMPLATE (/report/{report_id}), never the
        concrete path, so it cannot grow with the number of reports or leak an id."""
        rid = obs.accept_request_id(request.headers.get("x-request-id"), lambda: uuid.uuid4().hex[:16])
        token = obs.request_id_var.set(rid)
        started = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            obs.request_id_var.reset(token)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        obs.inc("http_requests_total", method=request.method, route=route, status=str(response.status_code))
        obs.observe("http_request_seconds", time.monotonic() - started, route=route)
        response.headers["X-Request-ID"] = rid
        return response
