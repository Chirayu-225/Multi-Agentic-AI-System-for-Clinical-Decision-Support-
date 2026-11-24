"""Request bodies (Pydantic models) for the API."""
from typing import Literal

from pydantic import BaseModel, Field


# ── Auth ───────────────────────────────────────────────────────────────────────

class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class CreatePatientAccountIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    patient_id: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=100)


# ── Review (human-in-the-loop release) ─────────────────────────────────────────

class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=1000)
    # Approve-with-edits: the clinician's corrected text replaces the model's (which is kept).
    edited_clinical_summary: str | None = Field(default=None, max_length=5000)
    edited_patient_summary: str | None = Field(default=None, max_length=5000)
    # Required when the report carries any warning (see review_warnings).
    acknowledge_warnings: bool = False
