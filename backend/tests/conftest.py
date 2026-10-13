"""
Shared pytest setup.

The production dependency set is heavy (torch, GLiNER, EasyOCR, ChromaDB,
google-generativeai). None of it is needed to test the security, auth, audit,
review-workflow, FHIR and eval code, and requiring it would make CI slow and
flaky. So: any of those modules that is NOT installed is replaced with an inert
stub BEFORE the app is imported. If the real module is installed it is used.
Tests that exercise the pipeline monkeypatch the agent entry points directly.
"""
import os
import sys
import types
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Environment the app requires at startup. Set before any app import.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("API_KEY", "test-service-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-not-for-production-0123456789")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("AUDIT_ENABLED", "true")
os.environ.setdefault("UPLOAD_RATE_LIMIT_MAX", "1000")
os.environ.setdefault("BOOTSTRAP_CLINICIAN_USERNAME", "dr.rao")
os.environ.setdefault("BOOTSTRAP_CLINICIAN_PASSWORD", "Str0ng-Clinician-Pw!")
os.environ.setdefault("BOOTSTRAP_PATIENT_USERNAME", "priya")
os.environ.setdefault("BOOTSTRAP_PATIENT_PASSWORD", "Str0ng-Patient-Pw!")
os.environ.setdefault("BOOTSTRAP_PATIENT_PATIENT_ID", "P001")


from tests.stubs import install_stubs  # noqa: E402

install_stubs()
