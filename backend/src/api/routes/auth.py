"""Login, current user and change-password (token and role machinery lives in src/api/auth.py)."""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api import auth as authmod
from src.api.auth import Principal, get_principal
from src.api.deps import client_ip
from src.api.schemas import ChangePasswordIn, LoginIn
from src.db.models import User
from src.db.session import get_db
from src.security import audit
from src.security.passwords import DUMMY_HASH, WeakPassword, hash_password, validate_password_strength, verify_password
from src.security.rate_limit import allow as rate_limit_allow

router = APIRouter()


LOGIN_MAX_FAILURES = 5


LOGIN_LOCK_MINUTES = 15


@router.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    ip = client_ip(request)
    username = body.username.strip().lower()

    # Throttle by IP and by username so neither a spray nor a single-account brute force
    # gets unlimited guesses (in-process limiter - see rate_limit.py caveat).
    for key, limit in ((f"login-ip:{ip}", 20), (f"login-user:{username}", 8)):
        ok, retry = rate_limit_allow(key, limit, 300)
        if not ok:
            audit.record(actor=username, action="auth.login_throttled", outcome="denied", ip=ip)
            raise HTTPException(429, f"Too many login attempts. Retry in {retry}s.",
                                headers={"Retry-After": str(retry)})

    user = db.query(User).filter(User.username == username).first()
    now = datetime.utcnow()

    if user and user.locked_until and user.locked_until > now:
        audit.record(actor=username, role=user.role, action="auth.login_locked", outcome="denied", ip=ip)
        raise HTTPException(423, "Account temporarily locked after repeated failed logins.")

    # Always run one scrypt verification (against a dummy hash for unknown users) so
    # response time does not reveal whether the username exists.
    valid = verify_password(body.password, user.password_hash if user else DUMMY_HASH)
    if not user or not user.is_active or not valid:
        if user:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= LOGIN_MAX_FAILURES:
                user.locked_until = now + timedelta(minutes=LOGIN_LOCK_MINUTES)
                user.failed_logins = 0
            db.commit()
        audit.record(actor=username, action="auth.login_failed", outcome="denied", ip=ip)
        raise HTTPException(401, "Invalid username or password.")

    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = now
    db.commit()
    audit.record(actor=user.username, role=user.role, action="auth.login", ip=ip)
    return _session_payload(user)


def _session_payload(user: User) -> dict:
    """Token + profile returned by login and by a password change. A user whose password is
    temporary gets a token that works for nothing except /auth/change-password."""
    must_change = bool(user.must_change_password)
    token, ttl = authmod.create_access_token(username=user.username, role=user.role,
                                             patient_id=user.patient_id,
                                             must_change_password=must_change,
                                             token_version=user.token_version or 0)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "role": user.role, "username": user.username,
            "display_name": user.display_name or user.username, "patient_id": user.patient_id,
            "must_change_password": must_change}


@router.get("/auth/me")
def whoami(principal: Principal = Depends(get_principal)):
    return {"username": principal.subject, "role": principal.role,
            "patient_id": principal.patient_id, "via": principal.via,
            "must_change_password": principal.must_change_password}


@router.post("/auth/change-password")
def change_password(body: ChangePasswordIn, request: Request, db: Session = Depends(get_db),
                    principal: Principal = Depends(get_principal)):
    """Change your own password. Also the only call a temporary-password token may make;
    on success the temporary flag is cleared and a fresh, unrestricted token is returned.
    Uses get_principal (not require_roles) on purpose, see api/auth.py."""
    ip = client_ip(request)
    # A stolen token must not become an unlimited oracle for guessing the current password.
    ok, retry = rate_limit_allow(f"chpw:{principal.subject}", 5, 900)
    if not ok:
        audit.record(actor=principal.subject, role=principal.role, action="auth.password_change_throttled",
                     outcome="denied", ip=ip)
        raise HTTPException(429, f"Too many attempts. Retry in {retry}s.", headers={"Retry-After": str(retry)})

    user = db.query(User).filter(User.username == principal.subject).first()
    if not user or not user.is_active:
        raise HTTPException(401, "Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"})
    # 400 (not 401/403): the frontend treats any 401 as "session over" and signs the user out.
    if not verify_password(body.current_password, user.password_hash):
        audit.record(actor=user.username, role=user.role, action="auth.password_change_failed",
                     outcome="denied", ip=ip)
        raise HTTPException(400, "Current password is incorrect.")
    if body.new_password == body.current_password:
        raise HTTPException(400, "Choose a password different from the current one.")
    try:
        validate_password_strength(body.new_password)
    except WeakPassword as e:
        raise HTTPException(400, str(e))

    user.password_hash = hash_password(body.new_password)
    user.must_change_password = False
    user.failed_logins = 0
    user.locked_until = None
    # Every token issued so far (including the one making this call) stops working; the fresh
    # token returned below carries the new version, so this device stays signed in.
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    audit.record(actor=user.username, role=user.role, action="auth.password_changed", ip=ip)
    return _session_payload(user)
