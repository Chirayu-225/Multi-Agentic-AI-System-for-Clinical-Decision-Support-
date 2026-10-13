"""
src/api/auth.py

Authentication and authorization for the Agentic Medical Assistant.

Two ways to authenticate, resolved into one `Principal`:

  1. Named user account  -> `Authorization: Bearer <JWT>` (issued by POST /auth/login)
       role "clinician": full access, the ONLY role that may approve/reject a report,
                         read the audit log, delete a patient
       role "patient"  : read-only, bound to ONE patient_id, sees APPROVED reports only

  2. Shared service key  -> `X-API-Key: <API_KEY>`   (kept so scripts / the original
       frontend keep working; disable with ALLOW_API_KEY_AUTH=false)
       role "service"  : may upload and read, but can NEVER approve a report or read
                         the audit log - a release decision must be attributable to a
                         named human, and a shared secret is not a person.

Why a JWT and not server sessions: the API is stateless and the Angular client is a
separate origin. Tradeoff accepted and documented in SECURITY.md: tokens cannot be
revoked before they expire (keep ACCESS_TOKEN_MINUTES short), and the SPA keeps the
token in sessionStorage, so an XSS bug would expose it - which is why the frontend
never renders model output with innerHTML (OWASP LLM10).

WebSocket: browsers cannot set headers on a WebSocket, so the credential travels as a
`token` (JWT) or `api_key` query parameter. Use wss:// outside local development.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import jwt
from fastapi import Depends, Header, HTTPException, Request, WebSocket, status

logger = logging.getLogger("medsight")

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "agentic-medical-assistant"
MIN_SECRET_LENGTH = 32

ROLE_CLINICIAN = "clinician"
ROLE_PATIENT = "patient"
ROLE_SERVICE = "service"


# ── Principal ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Principal:
    subject: str            # username, or "service:<fingerprint>"
    role: str               # clinician | patient | service
    patient_id: str | None  # set for role == patient
    via: str                # "jwt" | "api_key"

    @property
    def is_staff(self) -> bool:
        return self.role in (ROLE_CLINICIAN, ROLE_SERVICE)


# ── Secrets ────────────────────────────────────────────────────────────────────

_ephemeral_secret: str | None = None


def _jwt_secret() -> str:
    """JWT signing secret. If JWT_SECRET is unset we fall back to a random per-process
    secret (tokens die on restart) and warn - a missing secret must never mean 'sign
    with a guessable default'. A too-short configured secret is refused outright."""
    global _ephemeral_secret
    configured = os.getenv("JWT_SECRET", "")
    if configured:
        if len(configured) < MIN_SECRET_LENGTH:
            raise RuntimeError(
                f"JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters. Generate one with: "
                f'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return configured
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET is not set - using a random per-process secret. Logins will be "
            "invalidated on every restart. Set JWT_SECRET in backend/.env."
        )
    return _ephemeral_secret


def validate_auth_config() -> None:
    """Called at startup so a bad JWT_SECRET fails fast instead of on first login."""
    _jwt_secret()


def _configured_api_key() -> str:
    key = os.getenv("API_KEY", "")
    if not key:
        # Fail CLOSED: an unconfigured secret must never mean "auth disabled".
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Server misconfiguration: API_KEY is not set.",
        )
    return key


def _api_key_auth_allowed() -> bool:
    return os.getenv("ALLOW_API_KEY_AUTH", "true").lower() != "false"


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def fingerprint(api_key: str | None) -> str:
    """Short, non-reversible identifier for a credential - for audit correlation
    only, never store the raw key."""
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


# ── Tokens ─────────────────────────────────────────────────────────────────────

def access_token_minutes() -> int:
    try:
        return max(1, int(os.getenv("ACCESS_TOKEN_MINUTES", "60")))
    except ValueError:
        return 60


def create_access_token(*, username: str, role: str, patient_id: str | None = None) -> tuple[str, int]:
    now = int(time.time())
    ttl = access_token_minutes() * 60
    claims = {
        "iss": JWT_ISSUER, "sub": username, "role": role, "pid": patient_id,
        "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, _jwt_secret(), algorithm=JWT_ALGORITHM), ttl


def decode_access_token(token: str) -> Principal:
    """Raises jwt.PyJWTError on any problem (bad signature, expired, wrong issuer,
    missing claims, algorithm confusion). `algorithms` is pinned so a token with
    alg=none or a different algorithm is rejected."""
    claims = jwt.decode(
        token, _jwt_secret(), algorithms=[JWT_ALGORITHM], issuer=JWT_ISSUER,
        options={"require": ["exp", "iat", "sub", "iss"]},
    )
    role = claims.get("role")
    if role not in (ROLE_CLINICIAN, ROLE_PATIENT):
        raise jwt.InvalidTokenError("unknown role")
    if role == ROLE_PATIENT and not claims.get("pid"):
        raise jwt.InvalidTokenError("patient token without patient binding")
    return Principal(subject=claims["sub"], role=role, patient_id=claims.get("pid"), via="jwt")


# ── Request dependencies ───────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def get_principal(
    request: Request,
    authorization: str = Header(default=""),
    x_api_key: str = Header(default=""),
) -> Principal:
    """Resolve the caller. Bearer JWT wins; the shared key is the fallback."""
    from src.security import audit  # local: audit imports the DB layer

    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        try:
            principal = decode_access_token(token)
        except jwt.PyJWTError:
            audit.record(actor="anonymous", action="auth.token_rejected", outcome="denied",
                         ip=_client_ip(request))
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token.",
                                headers={"WWW-Authenticate": "Bearer"})
        request.state.principal = principal
        return principal

    if x_api_key and _api_key_auth_allowed():
        expected = _configured_api_key()
        if _constant_time_eq(x_api_key, expected):
            principal = Principal(subject=f"service:{fingerprint(x_api_key)}", role=ROLE_SERVICE,
                                  patient_id=None, via="api_key")
            request.state.principal = principal
            return principal

    audit.record(actor="anonymous", action="auth.missing_or_invalid_credentials",
                 outcome="denied", ip=_client_ip(request),
                 detail={"path": request.url.path})
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required.",
                        headers={"WWW-Authenticate": "Bearer"})


def require_roles(*roles: str) -> Callable:
    """Dependency factory: `Depends(require_roles("clinician"))`. A denied attempt
    is written to the audit log (an authenticated user probing endpoints they
    shouldn't reach is itself worth recording)."""
    allowed = set(roles)

    async def _dep(request: Request, principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in allowed:
            from src.security import audit
            audit.record(actor=principal.subject, role=principal.role, action="authz.denied",
                         resource=request.url.path, outcome="denied", ip=_client_ip(request),
                         detail={"required": sorted(allowed)})
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have access to this resource.")
        return principal

    return _dep


# ── Backwards-compatible helpers (original shared-key API) ────────────────────

async def require_api_key(x_api_key: str = Header(default="")) -> str:
    """Original dependency: raise 401 unless X-API-Key matches API_KEY. Kept for
    scripts; endpoints use get_principal/require_roles now."""
    expected = _configured_api_key()
    if not x_api_key or not _constant_time_eq(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")
    return x_api_key


async def verify_ws_auth(websocket: WebSocket) -> Principal | None:
    """Authenticate a WebSocket via `token` (JWT) or `api_key` query param. Returns
    None (caller closes the socket) rather than raising: a dependency-raised
    HTTPException doesn't cleanly close an already-accepted socket."""
    token = websocket.query_params.get("token", "")
    if token:
        try:
            return decode_access_token(token)
        except Exception:
            return None
    provided = websocket.query_params.get("api_key", "")
    if provided and _api_key_auth_allowed():
        try:
            expected = _configured_api_key()
        except HTTPException:
            return None
        if _constant_time_eq(provided, expected):
            return Principal(subject=f"service:{fingerprint(provided)}", role=ROLE_SERVICE,
                             patient_id=None, via="api_key")
    return None


async def verify_ws_api_key(websocket: WebSocket) -> bool:
    """Original WebSocket check (shared key only). Kept for compatibility."""
    return (await verify_ws_auth(websocket)) is not None
