"""Patient accounts created and managed by clinicians."""
import re
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api.auth import Principal, ROLE_CLINICIAN, ROLE_PATIENT, require_roles
from src.api.deps import client_ip, valid_patient_id
from src.api.schemas import CreatePatientAccountIn
from src.api.views import iso
from src.db.models import Patient, User
from src.db.session import get_db
from src.security import audit
from src.security.passwords import hash_password

router = APIRouter()


# ── Patient accounts (created and reset by clinicians) ─────────────────────────

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


# No 0/O, 1/l/I: a temporary password is read aloud or copied by hand.
_TEMP_PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


def _temporary_password() -> str:
    return "".join(secrets.choice(_TEMP_PASSWORD_ALPHABET) for _ in range(14))


def _account_view(u: User) -> dict:
    return {"username": u.username, "display_name": u.display_name or u.username,
            "patient_id": u.patient_id, "is_active": bool(u.is_active),
            "must_change_password": bool(u.must_change_password),
            "last_login_at": iso(u.last_login_at)}


@router.get("/admin/users")
def list_patient_accounts(db: Session = Depends(get_db),
                          principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    rows = db.query(User).filter(User.role == ROLE_PATIENT).order_by(User.username).all()
    return {"users": [_account_view(u) for u in rows]}


@router.post("/admin/users", status_code=201)
def create_patient_account(body: CreatePatientAccountIn, request: Request, db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """A clinician creates a login for a patient, bound to ONE existing patient_id. The
    temporary password is returned once, here, and is never stored in clear. The patient
    must change it at first sign-in. Clinician accounts are not created through the API."""
    username = body.username.strip().lower()
    if not USERNAME_RE.match(username):
        raise HTTPException(422, "Username must be 3-64 characters: lowercase letters, digits, '.', '_' or '-'.")
    patient_id = valid_patient_id(body.patient_id.strip())
    # Must already exist: a typo here would give someone access to a different person's reports.
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(404, "No patient with this ID. Upload one of their reports first.")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(409, "That username is already taken.")

    temp = _temporary_password()
    audit.record(actor=principal.subject, role=principal.role, action="user.create",
                 resource=f"user:{username}", ip=client_ip(request), critical=True,
                 detail={"role": ROLE_PATIENT, "patient_id": patient_id})
    user = User(username=username, display_name=(body.display_name or "").strip() or username,
                password_hash=hash_password(temp), role=ROLE_PATIENT, patient_id=patient_id,
                must_change_password=True)
    db.add(user)
    db.commit()
    return {**_account_view(user), "temporary_password": temp,
            "note": "Shown once. Give it to the patient securely; they must change it at first sign-in."}


def _patient_account_or_404(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username.strip().lower()).first()
    if not user or user.role != ROLE_PATIENT:
        raise HTTPException(404, "No patient account with that username.")
    return user


@router.post("/admin/users/{username}/deactivate")
def deactivate_patient_account(username: str, request: Request, db: Session = Depends(get_db),
                               principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Disable a patient login and end any session it has open. Their reports are untouched."""
    user = _patient_account_or_404(db, username)
    audit.record(actor=principal.subject, role=principal.role, action="user.deactivate",
                 resource=f"user:{user.username}", ip=client_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.is_active = False
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    return _account_view(user)


@router.post("/admin/users/{username}/reactivate")
def reactivate_patient_account(username: str, request: Request, db: Session = Depends(get_db),
                               principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Re-enable a disabled patient login. The password is unchanged; sessions from before the
    deactivation stay dead (the token version was already bumped)."""
    user = _patient_account_or_404(db, username)
    audit.record(actor=principal.subject, role=principal.role, action="user.reactivate",
                 resource=f"user:{user.username}", ip=client_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.is_active = True
    user.failed_logins = 0
    user.locked_until = None
    db.commit()
    return _account_view(user)


@router.post("/admin/users/{username}/reset-password")
def reset_patient_password(username: str, request: Request, db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Forgotten password: issue a new temporary one, unlock the account, force a change.
    Patient accounts only (there is no admin role yet, so one clinician cannot reset another)."""
    user = _patient_account_or_404(db, username)
    temp = _temporary_password()
    audit.record(actor=principal.subject, role=principal.role, action="user.password_reset",
                 resource=f"user:{user.username}", ip=client_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.password_hash = hash_password(temp)
    user.must_change_password = True
    user.failed_logins = 0
    user.locked_until = None
    user.token_version = (user.token_version or 0) + 1      # signs out any session already open
    db.commit()
    return {**_account_view(user), "temporary_password": temp,
            "note": "Shown once. Give it to the patient securely; they must change it at first sign-in."}
