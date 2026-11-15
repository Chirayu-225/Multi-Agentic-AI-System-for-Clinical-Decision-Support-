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
# Never read the developer's real backend/.env during tests (see main.py).
os.environ["MEDSIGHT_SKIP_DOTENV"] = "1"
# Hard-set: a developer's shell may already export GROQ_API_KEY or LLM_PROVIDER, and the suite's
# existing tests fake the Gemini SDK. Provider tests switch it themselves (tests/test_groq_provider.py).
os.environ["LLM_PROVIDER"] = "gemini"
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-not-for-production-0123456789")
# Hard-set, never setdefault: the API tests DROP ALL TABLES between tests. A DATABASE_URL
# inherited from the shell or .env once pointed them at the real medsight.db and wiped it.
# A throw-away FILE database in a fresh temp directory (not :memory:). The job queue is used from
# several threads at once, and an in-memory SQLite database is one shared connection that cannot
# be: a file gives every thread its own connection, as PostgreSQL does in production.
import shutil  # noqa: E402
import tempfile  # noqa: E402

_TEST_DB_DIR = tempfile.mkdtemp(prefix="medsight-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_DB_DIR, "test.db").replace("\\", "/")
# Optional: run the suite against PostgreSQL instead (CI does; scripts/run_tests_on_postgres.py
# does locally). The database name must end in "_test": the tests DROP ALL TABLES between cases.
_PG_TEST_URL = os.environ.get("MEDSIGHT_TEST_DATABASE_URL", "")
if _PG_TEST_URL:
    os.environ["DATABASE_URL"] = _PG_TEST_URL
os.environ["UPLOAD_SPOOL_DIR"] = os.path.join(_TEST_DB_DIR, "spool")
os.environ.setdefault("AUDIT_ENABLED", "true")
os.environ.setdefault("UPLOAD_RATE_LIMIT_MAX", "1000")
# Job queue: retry immediately and poll quickly so tests do not wait on backoff timers.
os.environ.setdefault("JOB_RETRY_BASE_SECONDS", "0")
os.environ.setdefault("JOB_POLL_SECONDS", "0.02")
os.environ.setdefault("OUTBOX_POLL_SECONDS", "0.05")
os.environ.setdefault("BOOTSTRAP_CLINICIAN_USERNAME", "dr.rao")
os.environ.setdefault("BOOTSTRAP_CLINICIAN_PASSWORD", "Str0ng-Clinician-Pw!")
os.environ.setdefault("BOOTSTRAP_PATIENT_USERNAME", "priya")
os.environ.setdefault("BOOTSTRAP_PATIENT_PASSWORD", "Str0ng-Patient-Pw!")
os.environ.setdefault("BOOTSTRAP_PATIENT_PATIENT_ID", "P001")


from tests.stubs import install_stubs  # noqa: E402

install_stubs()


def pytest_sessionstart(session):
    """Refuse to run unless the app's engine is the throw-away database created above."""
    from src.db.session import engine
    if _PG_TEST_URL:
        if not (engine.url.database or "").endswith("_test"):
            raise RuntimeError("MEDSIGHT_TEST_DATABASE_URL must point at a database whose name ends in "
                               f"'_test' (tables are dropped between tests), got {engine.url.database!r}.")
        return
    db_path = os.path.abspath(engine.url.database or "")
    if not db_path.startswith(os.path.abspath(_TEST_DB_DIR) + os.sep):
        raise RuntimeError(f"Tests must use the temporary test database, got {engine.url!s} - aborting "
                           "before any table is dropped.")


def pytest_sessionfinish(session, exitstatus):
    from src.db.session import engine
    engine.dispose()
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_llm_protection():
    """Circuit breakers are process-wide; a test that trips one must not affect the next."""
    from src import llm
    llm._breakers.clear()
    yield
    llm._breakers.clear()
