"""
src/db/models.py
PostgreSQL models via SQLAlchemy.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, ForeignKey
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
    entities     = Column(JSON)
    rule_flags   = Column(JSON)
    trend_flags  = Column(JSON)
    risk_summary = Column(JSON)
    lab_values   = Column(JSON)

    # Agent outputs
    clinical_summary = Column(Text)
    patient_summary  = Column(Text)
    explained_flags  = Column(JSON)
    followup_actions = Column(JSON)
    verification     = Column(JSON)

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
    created_at   = Column(DateTime, default=datetime.utcnow)
