"""
src/db/models.py
PostgreSQL models via SQLAlchemy.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, ForeignKey, Boolean, Index, Float
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

    # ── Analysis lifecycle ─────────────────────────────────────────────────
    # "preliminary": the deterministic results (lab flags, critical values, interactions) are
    # saved the moment they exist, before any model call, so a critical value is visible to a
    # clinician in seconds - and still visible if the model provider is down for an hour.
    # "complete": summaries and verification are done. NULL (older rows) means complete.
    # A preliminary report cannot be approved (the summaries it would release do not exist yet).
    analysis_status = Column(String, default="complete")
    has_critical    = Column(Boolean)   # any CRITICAL lab flag; lets the queue surface it first

    # ── Human-in-the-loop release ──────────────────────────────────────────
    # A report is "pending_review" until a named clinician approves it. Patient
    # accounts can only ever see approved reports. NULL (rows created before
    # this column existed) is treated as pending_review - fail closed.
    review_status = Column(String, default="pending_review")  # pending_review|approved|rejected
    reviewed_by   = Column(String)
    reviewed_at   = Column(DateTime)
    review_note   = Column(Text)
    # Review detail (all nullable so existing databases auto-migrate, see db/session.py).
    # When a clinician edits a summary before approving, the model's original text is kept
    # here and the edited text becomes clinical_summary/patient_summary.
    ai_clinical_summary   = Column(Text)
    ai_patient_summary    = Column(Text)
    review_edited         = Column(Boolean)   # True if the clinician changed either summary
    edit_ratio            = Column(Float)     # 0..1 word-level change vs the AI text (src/ai_quality.py)
    edit_words_changed    = Column(Integer)   # words inserted + deleted by the clinician
    warnings_acknowledged = Column(Boolean)   # True if approved with warnings explicitly accepted
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

    # Hot paths: a patient's history (filter by patient, order by time) and the clinician's
    # review queue (filter by status). Existing databases get the same indexes from
    # db/session.py:_ensure_indexes() because create_all() never alters an existing table.
    __table_args__ = (
        Index("ix_reports_patient_created", "patient_id", "created_at"),
        Index("ix_reports_review_status", "review_status"),
    )


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
    model        = Column(String)    # exact model id that produced this run (see src/llm.py)
    prompt_versions = Column(JSON)   # {"summarizer": "summarizer@4e8f94dc1f", ...} see src/prompts.py
    created_at   = Column(DateTime, default=datetime.utcnow)


class LlmCall(Base):
    """One model call attempt (a retry is its own row). Metadata only: no prompt or response text,
    so it holds no PHI and is safe to aggregate. No foreign key to reports: calls made by a job
    that was deferred or failed are recorded too, and erasure deletes them by report id."""
    __tablename__ = "llm_calls"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    report_id     = Column(String, index=True)
    created_at    = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    purpose       = Column(String, nullable=False)       # orchestrator|summarizer|verifier|translator
    model         = Column(String)
    prompt_version = Column(String)                      # e.g. "summarizer@4e8f94dc1f"
    attempt       = Column(Integer)                      # attempt number inside with_retry
    status        = Column(String, nullable=False)       # ok|rate_limited|unavailable|error|circuit_open|budget
    error_class   = Column(String)
    latency_ms    = Column(Integer)
    input_tokens  = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd      = Column(Float, default=0.0)           # NOTIONAL: what this would cost on a paid tier


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
    # True after a clinician creates or resets the account: the password is temporary and the
    # only thing the token allows is changing it. NULL (older rows) means False.
    must_change_password = Column(Boolean, default=False)
    # Bumped on password change/reset and on deactivation. Every token carries the value it was
    # issued with and is rejected once it no longer matches (see api/auth.py). NULL means 0.
    token_version = Column(Integer, default=0)
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

    # UNIQUE prev_hash is what makes the chain impossible to fork: two writers that read the same
    # head both try to append after it, and the database rejects the second one (which then
    # re-reads the head and retries, see security/audit.py). Works on SQLite and Postgres alike.
    __table_args__ = (Index("ux_audit_prev_hash", "prev_hash", unique=True),)


class UploadJob(Base):
    """Durable queue entry for one report upload (see src/jobs/queue.py for the protocol).

    The row is the single source of truth for a job: it survives a restart, it is what the
    polling endpoint reads, and the claim/lease columns are what let several workers (or a
    restarted process) share the queue without two of them running the same job.
    The id IS the report id, so a second upload with the same id is rejected by the primary key.
    Holds identifiers, stage names and counters only - the report content lives on disk in the
    spool until the job finishes, and in `reports` once it has been analysed.
    """
    __tablename__ = "upload_jobs"

    id           = Column(String, primary_key=True)             # == report id
    owner        = Column(String, nullable=False, index=True)   # uploader's username
    ip           = Column(String)
    patient_id   = Column(String, nullable=False)
    filename     = Column(String)
    ext          = Column(String)
    report_date  = Column(String)
    language     = Column(String, default="en")
    spool_path   = Column(String, nullable=False)               # the uploaded file, until the job ends

    status       = Column(String, nullable=False, default="queued")   # queued|running|complete|failed
    stage        = Column(String, default="queued")
    percent      = Column(Integer, default=0)
    error        = Column(Text)                                  # user-safe message, never a traceback

    attempts     = Column(Integer, nullable=False, default=0)    # incremented when a worker claims it
    max_attempts = Column(Integer, nullable=False, default=3)
    next_attempt_at  = Column(DateTime, nullable=False)          # not claimable before this (backoff)
    locked_by        = Column(String)                            # worker id holding the lease
    lease_expires_at = Column(DateTime)                          # a running job past this is presumed dead

    created_at   = Column(DateTime, nullable=False)
    started_at   = Column(DateTime)
    finished_at  = Column(DateTime)

    __table_args__ = (
        # The claim query filters on status and orders by age; this keeps it an index scan.
        Index("ix_upload_jobs_claim", "status", "next_attempt_at"),
    )


class OutboxEvent(Base):
    """A side effect that must happen if (and only if) its database transaction committed.

    The report row and its history-store (ChromaDB) write live in two systems with no shared
    transaction. Writing the event in the SAME transaction as the report means a crash can never
    leave "report saved, history update forgotten": the relay applies pending events with retries,
    and the handler is idempotent (upsert by report id), so at-least-once delivery is safe.
    The payload is the already-redacted text, the same content the vector store holds; it is
    deleted with the patient (erasure) and cancelled when a report is rejected.
    """
    __tablename__ = "outbox_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    kind         = Column(String, nullable=False)                  # e.g. "history.index"
    report_id    = Column(String, nullable=False, index=True)
    patient_id   = Column(String, nullable=False, index=True)
    payload      = Column(JSON)
    status       = Column(String, nullable=False, default="pending")   # pending|done|dead|cancelled
    attempts     = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=False)             # also the lease while being applied
    last_error   = Column(Text)
    created_at   = Column(DateTime, nullable=False)
    done_at      = Column(DateTime)

    __table_args__ = (Index("ix_outbox_due", "status", "next_attempt_at"),)

