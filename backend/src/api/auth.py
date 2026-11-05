"""
src/api/auth.py

Authentication and authorization for the Agentic Medical Assistant.

Every caller is a named user account. POST /auth/login exchanges a username and password
for a short-lived JWT (`Authorization: Bearer <JWT>`), which `get_principal` resolves into
one `Principal`:

  role "clinician": full access, the ONLY role that may upload, approve/reject a report,
                    read the audit log, delete a patient, or create/reset patient accounts
  role "patient"  : read-only, bound to ONE patient_id, sees APPROVED reports only

There is deliberately no shared secret or "service" role: a release decision (and every
other action) must be attributable to a named human.

Forced password change: accounts created or reset by a clinician carry a temporary
password. Their token has the `pwc` claim, `Principal.must_change_password` is True, and
`require_roles` refuses every endpoint except the ones that use `get_principal` directly
(/auth/me, /auth/change-password). Changing the password issues a fresh, unrestricted token.

Why a JWT and not server sessions: the API is stateless and the Angular client is a
separate origin. Tradeoff accepted and documented in SECURITY.md: tokens cannot be
revoked before they expire (keep ACCESS_TOKEN_MINUTES short), and the SPA keeps the
token in sessionStorage, so an XSS bug would expose it - which is why the frontend
never renders model output with innerHTML (OWASP LLM10).

WebSocket: browsers cannot set headers on a WebSocket, so the JWT travels as a `token`
query parameter. Use wss:// outside local development.
"""

import hashlib
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


# ── Principal ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Principal:
    subject: str            # username
    role: str               # clinician | patient
    patient_id: str | None  # set for role == patient
    via: str = "jwt"
    must_change_password: bool = False   # temporary password: only /auth/change-password works

    @property
    def is_staff(self) -> bool:
        return self.role == ROLE_CLINICIAN


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


def fingerprint(value: str | None) -> str:
    """Short, non-reversible identifier - for audit correlation only."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


# ── Tokens ─────────────────────────────────────────────────────────────────────

def access_token_minutes() -> int:
    try:
        return max(1, int(os.getenv("ACCESS_TOKEN_MINUTES", "60")))
    except ValueError:
        return 60


def create_access_token(*, username: str, role: str, patient_id: str | None = None,
                        must_change_password: bool = False) -> tuple[str, int]:
    now = int(time.time())
    ttl = access_token_minutes() * 60
    claims = {
        "iss": JWT_ISSUER, "sub": username, "role": role, "pid": patient_id,
        "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex,
    }
    if must_change_password:
        claims["pwc"] = True
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
    return Principal(subject=claims["sub"], role=role, patient_id=claims.get("pid"), via="jwt",
                     must_change_password=bool(claims.get("pwc")))


# ── Request dependencies ───────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def get_principal(
    request: Request,
    authorization: str = Header(default=""),
) -> Principal:
    """Resolve the caller from the Bearer JWT. Does NOT refuse a must-change-password
    token: /auth/me and /auth/change-password need it. Everything else goes through
    require_roles, which does."""
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
        if principal.must_change_password:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "You must change your temporary password before using the system.")
        if principal.role not in allowed:
            from src.security import audit
            audit.record(actor=principal.subject, role=principal.role, action="authz.denied",
                         resource=request.url.path, outcome="denied", ip=_client_ip(request),
                         detail={"required": sorted(allowed)})
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have access to this resource.")
        return principal

    return _dep


# ── WebSocket ──────────────────────────────────────────────────────────────────

async def verify_ws_auth(websocket: WebSocket) -> Principal | None:
    """Authenticate a WebSocket via the `token` (JWT) query param. Returns None (caller
    closes the socket) rather than raising: a dependency-raised HTTPException doesn't
    cleanly close an already-accepted socket. A must-change-password token is refused."""
    token = websocket.query_params.get("token", "")
    if not token:
        return None
    try:
        principal = decode_access_token(token)
    except Exception:
        return None
    return None if principal.must_change_password else principal
