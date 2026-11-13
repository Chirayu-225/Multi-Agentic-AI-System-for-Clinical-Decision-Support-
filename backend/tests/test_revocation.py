"""Token revocation: password change/reset and deactivation end sessions that are already open."""
import time

import jwt
import pytest
from starlette.websockets import WebSocketDisconnect

from src.api import auth as authmod
from src.db.models import AuditEvent, User
from src.db.session import SessionLocal
from tests.test_api_flow import (CLIN, PAT, client, fresh_state, login, upload)  # noqa: F401


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _login_token(client, creds):
    r = client.post("/auth/login", json={"username": creds[0], "password": creds[1]})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _set_version(username, value):
    with SessionLocal() as db:
        db.query(User).filter_by(username=username).one().token_version = value
        db.commit()


# ── password change ───────────────────────────────────────────────────────────

def test_changing_password_signs_out_other_sessions_but_not_this_one(client):
    phone = _bearer(_login_token(client, PAT))
    laptop = _bearer(_login_token(client, PAT))
    assert client.get("/me/reports", headers=phone).status_code == 200

    r = client.post("/auth/change-password", headers=laptop,
                    json={"current_password": PAT[1], "new_password": "Another-Str0ng-Pw!"})
    assert r.status_code == 200
    fresh = _bearer(r.json()["access_token"])

    assert client.get("/me/reports", headers=phone).status_code == 401      # other device: out
    assert client.get("/me/reports", headers=laptop).status_code == 401     # the old token: out
    assert client.get("/me/reports", headers=fresh).status_code == 200      # new token: in


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_ends_a_session_the_patient_already_has_open(client):
    ch = login(client, CLIN)
    upload(client, ch)
    open_session = _bearer(_login_token(client, PAT))
    assert client.get("/me/reports", headers=open_session).status_code == 200

    assert client.post("/admin/users/priya/reset-password", headers=ch).status_code == 200
    assert client.get("/me/reports", headers=open_session).status_code == 401
    assert client.get("/auth/me", headers=open_session).status_code == 401


# ── deactivate / reactivate ───────────────────────────────────────────────────

def test_deactivate_kills_sessions_and_blocks_login_until_reactivated(client):
    ch = login(client, CLIN)
    open_session = _bearer(_login_token(client, PAT))

    r = client.post("/admin/users/priya/deactivate", headers=ch)
    assert r.status_code == 200 and r.json()["is_active"] is False
    assert client.get("/me/reports", headers=open_session).status_code == 401
    assert client.post("/auth/login", json={"username": PAT[0], "password": PAT[1]}).status_code == 401

    r = client.post("/admin/users/priya/reactivate", headers=ch)
    assert r.status_code == 200 and r.json()["is_active"] is True
    assert client.get("/me/reports", headers=open_session).status_code == 401   # old session stays dead
    again = _bearer(_login_token(client, PAT))
    assert client.get("/me/reports", headers=again).status_code == 200          # a new login works


def test_only_clinicians_can_deactivate_and_only_patient_accounts(client):
    ch = login(client, CLIN)
    ph = login(client, PAT)
    assert client.post("/admin/users/priya/deactivate", headers=ph).status_code == 403
    assert client.post("/admin/users/priya/reactivate", headers=ph).status_code == 403
    assert client.post("/admin/users/priya/deactivate").status_code == 401
    assert client.post("/admin/users/dr.rao/deactivate", headers=ch).status_code == 404     # clinician
    assert client.post("/admin/users/nobody/deactivate", headers=ch).status_code == 404
    assert client.get("/auth/me", headers=ch).status_code == 200                            # unaffected


def test_deactivation_is_audited(client):
    ch = login(client, CLIN)
    client.post("/admin/users/priya/deactivate", headers=ch)
    client.post("/admin/users/priya/reactivate", headers=ch)
    with SessionLocal() as db:
        actions = {e.action for e in db.query(AuditEvent).all()}
    assert {"user.deactivate", "user.reactivate"} <= actions


def test_a_rejected_revoked_token_is_audited(client):
    old = _bearer(_login_token(client, PAT))
    _set_version("priya", 5)
    assert client.get("/auth/me", headers=old).status_code == 401
    with SessionLocal() as db:
        assert db.query(AuditEvent).filter_by(action="auth.token_revoked", actor="priya").count() == 1


# ── websocket ─────────────────────────────────────────────────────────────────

def test_websocket_refuses_a_revoked_token(client):
    ch_token = _login_token(client, CLIN)
    with client.websocket_connect(f"/ws/abc?token={ch_token}"):
        pass                                          # valid at first
    with SessionLocal() as db:
        db.query(User).filter_by(username="dr.rao").one().token_version = 3
        db.commit()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/abc?token={ch_token}"):
            pass


# ── compatibility ─────────────────────────────────────────────────────────────

def test_tokens_issued_before_this_feature_still_work_until_the_account_changes(client):
    """A token without the `tv` claim counts as version 0, and an account whose token_version is
    NULL (created before the column existed) counts as 0, so nobody is signed out by the upgrade."""
    with SessionLocal() as db:
        db.query(User).filter_by(username="priya").one().token_version = None
        db.commit()
    now = int(time.time())
    legacy = jwt.encode({"iss": authmod.JWT_ISSUER, "sub": "priya", "role": "patient", "pid": "P001",
                         "iat": now, "exp": now + 600, "jti": "legacy"},
                        authmod._jwt_secret(), algorithm=authmod.JWT_ALGORITHM)
    assert client.get("/me/reports", headers=_bearer(legacy)).status_code == 200
    _set_version("priya", 1)
    assert client.get("/me/reports", headers=_bearer(legacy)).status_code == 401


def test_deleted_account_token_is_rejected(client):
    token = _login_token(client, PAT)
    with SessionLocal() as db:
        db.query(User).filter_by(username="priya").delete()
        db.commit()
    assert client.get("/me/reports", headers=_bearer(token)).status_code == 401
