"""Small shared settings and helpers: the AI disclaimer, app version, env-var readers, spool folder."""
import os
import tempfile


# Every AI-generated summary carries this disclaimer end-to-end (API + UI) —
# this tool assists a clinician's review, it does not replace it.
AI_DISCLAIMER = (
    "AI-generated analysis for clinician review only. Not a diagnosis and not "
    "a substitute for professional medical judgment. Verify all findings "
    "against the source report before acting on them."
)


APP_VERSION = "3.0.0"


def require_review() -> bool:
    return os.getenv("REQUIRE_CLINICIAN_REVIEW", "true").lower() != "false"


def env_num(name: str, default, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def spool_dir() -> str:
    """Where uploaded files wait until their job finishes. Must survive a restart (so a queued
    job can still run afterwards) and be readable by every worker. Mode 0700: it holds PHI."""
    path = os.getenv("UPLOAD_SPOOL_DIR") or os.path.join(tempfile.gettempdir(), "medsight-spool")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path
