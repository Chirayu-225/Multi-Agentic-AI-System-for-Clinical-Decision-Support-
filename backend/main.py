"""
main.py: the entry point. Run it with `uvicorn main:app --reload --port 8001`.

This file only loads the environment and builds the app. The code lives in src/:
  src/api/         the HTTP layer (routes/, schemas, response views, auth, middleware)
  src/pipeline/    the background job that analyses a report, plus its workers
  src/agents/      the LLM agents (orchestrator, summarizer, verifier, translator)
  src/core/        the single LLM client, resilience, settings and feature flags
See the README at the repo root for the full map.
"""
import os
from pathlib import Path

# Load .env by an explicit path next to this file, before any of the src.* imports below:
# src/core/llm.py reads GEMINI_MODEL at import time, so .env must be loaded first.
# Tests set MEDSIGHT_SKIP_DOTENV=1 (tests/conftest.py): with override=True, a developer's real
# backend/.env would otherwise replace the test values (JWT_SECRET...) and fail the
# auth tests on any machine that has one.
from dotenv import load_dotenv
if os.getenv("MEDSIGHT_SKIP_DOTENV") != "1":
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from src.api.app import create_app  # noqa: E402  (must come after load_dotenv)

app = create_app()
