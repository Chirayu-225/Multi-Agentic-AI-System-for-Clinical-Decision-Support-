# Agentic Medical Assistant backend

Backend-specific quick reference. **See the [repository root README](../README.md) for
full setup, architecture, the longitudinal-intelligence tiers, and the security model** —
this file used to duplicate that content with some details drifted out of sync; it now
just points there plus a few backend-only dev notes.

## Quick start (backend only, no Docker)

```bash
cp .env.example .env
# edit .env: set GEMINI_API_KEY, JWT_SECRET, API_KEY and BOOTSTRAP_CLINICIAN_* (see root README)
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs at `http://localhost:8000` — Swagger UI at `/docs`, health check at `/health`
(the only unauthenticated endpoint besides `/auth/login`; everything else needs a Bearer token or `X-API-Key`).

## Project layout

```
src/
  agents/       orchestrator (ReAct + tool calling), summarizer, verifier (fail-closed)
  api/          FastAPI app, routes, auth
  db/           SQLAlchemy models + session
  flagging/     ICMR rule engine + longitudinal intelligence (trend/baseline/forecast)
  ingestion/    PDF text extraction (digital + OCR), Hindi transliteration
  ner/          GLiNER-based medical entity extraction
  security/     injection detection, de-identification, passwords, hash-chained audit log, rate limiting
  evidence/     numeric grounding + evidence linking
  interop/      FHIR R4 export
  eval/         synthetic data + evaluation harness (outside src/)
  tools/        agent tool registry + ChromaDB history store (incl. semantic search)
```

## Testing a change locally

```bash
pip install -r requirements-dev.txt
python -m py_compile $(find . -name "*.py" -not -path "*/venv/*")
python -m pytest -q                 # 195 tests; Gemini and the ML libraries are stubbed
python -m eval.run_eval --check     # regenerates ../EVAL_RESULTS.md, fails on regression
```

Add `--live` to `run_eval` (with `GEMINI_API_KEY` set) to measure the real Verifier.
Test accounts used by the suite live in `tests/conftest.py`.
