"""
src/db/models.py
PostgreSQL models via SQLAlchemy.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id         = Column(String, primary_key=True)  # user-defined patient ID
    created_at = Column(DateTime, default=datetime.utcnow)
    reports    = relationship("Report", back_populates="patient",
                              cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    id           = Column(String, primary_key=True)
    patient_id   = Column(String, ForeignKey("patients.id"), nullable=False)
    filename     = Column(String)
    report_date  = Column(String)  # stored as YYYY-MM-DD string
    source_type  = Column(String)  # "digital" or "scanned"
    has_hindi    = Column(String, default="false")
    pages        = Column(Integer, default=1)
    created_at   = Column(DateTime, default=datetime.utcnow)

    # Pipeline outputs stored as JSON
    entities          = Column(JSON)
    rule_flags        = Column(JSON)
    trend_flags       = Column(JSON)
    risk_summary      = Column(JSON)
    lab_values        = Column(JSON)
    ingestion_warnings = Column(JSON)  # e.g. text/entity-extraction truncation
    security_warnings = Column(JSON)  # e.g. suspected prompt-injection patterns

    # Agent outputs
    clinical_summary      = Column(Text)
    patient_summary       = Column(Text)
    explained_flags       = Column(JSON)
    followup_actions      = Column(JSON)
    verification          = Column(JSON)
    longitudinal_narrative = Column(Text)   # Tier 1 longitudinal intelligence
    baseline_flags         = Column(JSON)   # Tier 2 — patient-specific baseline deviations
    forecast_flags         = Column(JSON)   # Tier 3 — slope-based threshold projections
    similar_history         = Column(JSON)  # Tier 4 — semantic recall of past episodes

    # ── Human-in-the-loop release ──────────────────────────────────────────
    # A report is "pending_review" until a named clinician approves it. Patient
    # accounts can only ever see approved reports. NULL (rows created before
    # this column existed) is treated as pending_review - fail closed.
    review_status = Column(String, default="pending_review")  # pending_review|approved|rejected
    reviewed_by   = Column(String)
    reviewed_at   = Column(DateTime)
    review_note   = Column(Text)
    uploaded_by   = Column(String)

    # ── Grounding / interactions / translation ─────────────────────────────
    evidence             = Column(JSON)   # claim -> source-text spans
    grounding_warnings   = Column(JSON)   # numbers in a summary absent from ground truth
    drug_interactions    = Column(JSON)   # drug-drug / drug-lab findings
    drug_safety          = Column(JSON)   # per-drug OpenFDA label warnings
    pubmed_references    = Column(JSON)
    patient_summary_translated = Column(Text)
    translation_language = Column(String)
    translation_check    = Column(JSON)

    patient = relationship("Patient", back_populates="reports")


class AgentRun(Base):
    """Audit trail — every agent invocation logged here."""
    __tablename__ = "agent_runs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    report_id    = Column(String, ForeignKey("reports.id"))
    agent_name   = Column(String)   # "orchestrator" | "verifier"
    tool_calls   = Column(JSON)     # list of tools invoked
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    was_corrected = Column(String, default="false")
    # Non-reversible fingerprint of the API key that authenticated this
    # request — enough to correlate runs to a caller without storing the
    # shared secret itself. See src/api/auth.py:fingerprint().
    caller_fingerprint = Column(String)
    duration_ms  = Column(Integer)   # wall-clock for the whole pipeline
    llm_calls    = Column(Integer)   # Gemini round trips made for this report
    created_at   = Column(DateTime, default=datetime.utcnow)


class User(Base):
    """Named accounts. Roles: 'clinician' (full access, can review/approve) and
    'patient' (read-only, own approved reports only, bound to one patient_id)."""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String, unique=True, nullable=False, index=True)
    display_name  = Column(String)
    password_hash = Column(String, nullable=False)
    role          = Column(String, nullable=False)          # clinician | patient
    patient_id    = Column(String)                          # required when role == patient
    is_active     = Column(Boolean, default=True)
    failed_logins = Column(Integer, default=0)
    locked_until  = Column(DateTime)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime)


class AuditEvent(Base):
    """Append-only, hash-chained security audit log. Each row's `hash` covers its
    own content plus the previous row's hash, so editing or deleting any row
    breaks every later link (verified by security/audit.py:verify_chain).
    Contains identifiers and counts only - never report content or PHI."""
    __tablename__ = "audit_events"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    ts        = Column(String, nullable=False)     # ISO-8601 UTC, stored as text so the hash is byte-stable
    actor     = Column(String, nullable=False)     # username, "service:<fp>", or "anonymous"
    role      = Column(String)
    action    = Column(String, nullable=False, index=True)
    resource  = Column(String)
    outcome   = Column(String, nullable=False)     # success | denied | failure
    ip        = Column(String)
    detail    = Column(JSON)
    prev_hash = Column(String, nullable=False)
    hash      = Column(String, nullable=False)
