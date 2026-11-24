"""First-run account bootstrap from environment variables."""
import logging
import os

from src.api.auth import ROLE_CLINICIAN, ROLE_PATIENT
from src.db.models import User
from src.db.session import SessionLocal
from src.security.passwords import WeakPassword, hash_password, validate_password_strength

logger = logging.getLogger("medsight")


def bootstrap_users():
    """Create a first clinician (and optionally a demo patient) from env vars so a fresh
    install is usable. Passwords come from the environment, never from source code."""
    specs = [
        ("BOOTSTRAP_CLINICIAN", ROLE_CLINICIAN),
        ("BOOTSTRAP_PATIENT", ROLE_PATIENT),
    ]
    db = SessionLocal()
    try:
        for prefix, role in specs:
            username = os.getenv(f"{prefix}_USERNAME", "").strip()
            password = os.getenv(f"{prefix}_PASSWORD", "")
            if not username or not password:
                continue
            if db.query(User).filter(User.username == username).first():
                continue
            patient_id = os.getenv(f"{prefix}_PATIENT_ID", "").strip() or None
            if role == ROLE_PATIENT and not patient_id:
                logger.error("%s_PATIENT_ID is required for a patient account - skipped.", prefix)
                continue
            try:
                validate_password_strength(password)
            except WeakPassword as e:
                logger.error("Bootstrap user '%s' skipped: %s", username, e)
                continue
            db.add(User(username=username, display_name=os.getenv(f"{prefix}_DISPLAY_NAME", username),
                        password_hash=hash_password(password), role=role, patient_id=patient_id))
            db.commit()
            logger.warning("Bootstrap %s account created: %s", role, username)
    finally:
        db.close()
