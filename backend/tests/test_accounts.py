"""Clinician-created patient accounts, forced password change, forced reset."""
import jwt
import pytest

from src.db.models import AuditEvent, User
from src.db.session import SessionLocal
from tests.test_api_flow import (CLIN, PAT, client, fresh_state, login, upload)  # noqa: F401


def _create(client, ch, username="rahul", patient_id="P001", **extra):
    return client.post("/admin/users", headers=ch,
                       json={"username": username, "patient_id": patient_id, **extra})


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _login_raw(client, username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


def _setup(client):
    ch = login(client, CLIN)
    upload(client, ch)                       # creates patient P001
    return ch


# ── creation ──────────────────────────────────────────────────────────────────

def test_clinician_creates_patient_account_with_one_time_temp_password(client):
    ch = _setup(client)
    r = _create(client, ch, display_name="Rahul K")
    assert r.status_code == 201
    body = r.json()
    assert len(body["temporary_password"]) >= 10
    assert body["patient_id"] == "P001" and body["must_change_password"] is True
    assert "password_hash" not in body
    # never stored or listed in clear
    listing = client.get("/admin/users", headers=ch).json()["users"]
    assert any(u["username"] == "rahul" for u in listing)
    assert "temporary_password" not in str(listing)
    with SessionLocal() as db:
        assert body["temporary_password"] not in db.query(User).filter_by(username="rahul").one().password_hash


def test_creation_rules(client):
    ch = _setup(client)
    assert _create(client, ch, patient_id="NOPE").status_code == 404        # patient must exist
    assert _create(client, ch, username="Bad Name!").status_code == 422
    assert _create(client, ch).status_code == 201
    assert _create(client, ch).status_code == 409                           # username taken
    assert _create(client, ch, username="priya").status_code == 409         # bootstrap user too


def test_only_clinicians_manage_accounts(client):
    ch = _setup(client)
    ph = login(client, PAT)
    assert _create(client, ph, username="x1y").status_code == 403
    assert client.get("/admin/users", headers=ph).status_code == 403
    assert client.post("/admin/users/priya/reset-password", headers=ph).status_code == 403
    assert client.post("/admin/users", json={"username": "abc", "patient_id": "P001"}).status_code == 401
    assert ch  # clinician unaffected


def test_creation_is_audited_without_the_password(client):
    ch = _setup(client)
    temp = _create(client, ch).json()["temporary_password"]
    with SessionLocal() as db:
        ev = db.query(AuditEvent).filter_by(action="user.create").one()
        assert ev.resource == "user:rahul" and ev.actor == "dr.rao"
        assert temp not in str(ev.detail)


# ── forced change ─────────────────────────────────────────────────────────────

def test_temporary_password_token_can_only_change_password(client):
    ch = _setup(client)
    temp = _create(client, ch).json()["temporary_password"]
    r = _login_raw(client, "rahul", temp)
    assert r.status_code == 200 and r.json()["must_change_password"] is True
    th = _bearer(r.json()["access_token"])

    assert client.get("/auth/me", headers=th).json()["must_change_password"] is True
    assert client.get("/me/reports", headers=th).status_code == 403
    assert client.get("/report/anything", headers=th).status_code == 403
    assert client.get("/patient/P001/history", headers=th).status_code == 403
    token = r.json()["access_token"]
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/abc?token={token}"):
            pass


def test_change_password_unlocks_and_old_password_stops_working(client):
    ch = _setup(client)
    temp = _create(client, ch).json()["temporary_password"]
    th = _bearer(_login_raw(client, "rahul", temp).json()["access_token"])

    r = client.post("/auth/change-password", headers=th,
                    json={"current_password": temp, "new_password": "My-New-Passw0rd!"})
    assert r.status_code == 200 and r.json()["must_change_password"] is False
    new_h = _bearer(r.json()["access_token"])
    assert client.get("/me/reports", headers=new_h).status_code == 200      # now unrestricted
    assert client.get("/auth/me", headers=new_h).json()["must_change_password"] is False

    assert _login_raw(client, "rahul", temp).status_code == 401
    again = _login_raw(client, "rahul", "My-New-Passw0rd!")
    assert again.status_code == 200 and again.json()["must_change_password"] is False


@pytest.mark.parametrize("new, why", [("short", "at least"), ("password123", "common"), ("TempPass-12345", "different")])
def test_change_password_rejects_bad_new_passwords(client, new, why):
    ch = _setup(client)
    temp = "TempPass-12345"
    _create(client, ch)
    with SessionLocal() as db:
        from src.security.passwords import hash_password
        db.query(User).filter_by(username="rahul").one().password_hash = hash_password(temp)
        db.commit()
    th = _bearer(_login_raw(client, "rahul", temp).json()["access_token"])
    r = client.post("/auth/change-password", headers=th, json={"current_password": temp, "new_password": new})
    assert r.status_code == 400 and why in r.json()["detail"].lower()


def test_wrong_current_password_is_400_not_401_and_is_throttled(client):
    ch = _setup(client)
    temp = _create(client, ch).json()["temporary_password"]
    th = _bearer(_login_raw(client, "rahul", temp).json()["access_token"])
    for _ in range(5):
        r = client.post("/auth/change-password", headers=th,
                        json={"current_password": "wrong-guess-1", "new_password": "My-New-Passw0rd!"})
        assert r.status_code == 400       # 401 would sign the user out in the frontend
    r = client.post("/auth/change-password", headers=th,
                    json={"current_password": temp, "new_password": "My-New-Passw0rd!"})
    assert r.status_code == 429


def test_any_user_can_change_own_password(client):
    ph = login(client, PAT)
    r = client.post("/auth/change-password", headers=ph,
                    json={"current_password": PAT[1], "new_password": "Another-Str0ng-Pw!"})
    assert r.status_code == 200
    assert _login_raw(client, "priya", "Another-Str0ng-Pw!").status_code == 200


def test_change_password_requires_login(client):
    assert client.post("/auth/change-password",
                       json={"current_password": "x", "new_password": "y"}).status_code == 401


# ── forced reset ──────────────────────────────────────────────────────────────

def test_reset_issues_new_temp_password_unlocks_and_forces_change(client):
    ch = _setup(client)
    first = _create(client, ch).json()["temporary_password"]
    th = _bearer(_login_raw(client, "rahul", first).json()["access_token"])
    client.post("/auth/change-password", headers=th,
                json={"current_password": first, "new_password": "My-New-Passw0rd!"})
    with SessionLocal() as db:                           # simulate a lockout
        from datetime import datetime, timedelta
        u = db.query(User).filter_by(username="rahul").one()
        u.locked_until = datetime.utcnow() + timedelta(minutes=10)
        db.commit()
    assert _login_raw(client, "rahul", "My-New-Passw0rd!").status_code == 423

    r = client.post("/admin/users/rahul/reset-password", headers=ch)
    assert r.status_code == 200
    temp = r.json()["temporary_password"]
    assert temp != first and r.json()["must_change_password"] is True
    assert _login_raw(client, "rahul", "My-New-Passw0rd!").status_code == 401    # old password gone
    again = _login_raw(client, "rahul", temp)
    assert again.status_code == 200 and again.json()["must_change_password"] is True
    with SessionLocal() as db:
        assert db.query(AuditEvent).filter_by(action="user.password_reset").count() == 1


def test_reset_only_applies_to_existing_patient_accounts(client):
    ch = _setup(client)
    assert client.post("/admin/users/nobody/reset-password", headers=ch).status_code == 404
    assert client.post("/admin/users/dr.rao/reset-password", headers=ch).status_code == 404   # clinician
