# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- Use Windows Command Prompt (cmd) syntax, never PowerShell.
- Never create README files.
- Never read or edit backend\.env. Never commit .db or .pdf files.
- Tests: cd backend && python -m pytest -q
- Run backend: uvicorn main:app --reload --port 8001
- Run frontend: cd frontend && npm start
- Run the tests after every change and show me the result.

## What this is

Agentic Medical Assistant: a clinician-in-the-loop system for Indian GP clinics. It ingests lab reports (PDF/photo), flags values against ICMR ranges (age/sex-specific), tracks patients longitudinally, and produces AI summaries that a clinician must review and approve before a patient can see them. FastAPI + SQLAlchemy backend (`backend/`), Angular 18 + Tailwind frontend (`frontend/`), Gemini via `google-genai`. See [README.md](README.md) for setup/env vars and [docs/SECURITY.md](docs/SECURITY.md) for the threat model.

## Commands

Backend (run from `backend/`; Windows venv activate is `venv\Scripts\activate`):

```bash
pip install -r requirements-dev.txt     # tests/CI deps (lighter than requirements.txt)
uvicorn main:app --reload               # API on :8000, Swagger at /docs
python -m pytest -q                     # whole suite; no API key/GPU needed
python -m pytest tests/test_review_flow.py::test_name -q   # single test
python -m eval.run_eval --check         # 200-report synthetic eval; rewrites ../docs/EVAL_RESULTS.md, fails below eval/thresholds.json
python -m eval.run_eval --live          # also scores the real Verifier (needs GEMINI_API_KEY)
python -m eval.run_gold --check         # real de-identified gold set (eval/gold/labels.json, git-ignored)
python scripts/check_gemini.py          # Gemini connectivity diagnostics
bandit -r src main.py -ll               # SAST, as in CI
```

Frontend (from `frontend/`): `npm install`, `npm start` (:4200, needs backend on :8000), `npx ng build --configuration production`.
Full stack: `docker-compose up --build` from the repo root (needs root `.env` with `GEMINI_API_KEY`, `JWT_SECRET`, `POSTGRES_PASSWORD`).

There is no linter/formatter config; CI is `.github/workflows/security.yml` (pytest, eval gate, bandit, pip-audit, npm audit, gitleaks, prod build, and a grep that **fails on `innerHTML`, `bypassSecurityTrust`, `DomSanitizer`, `eval(`, `document.write` anywhere in `frontend/src`** — model output must only be interpolated).

## Architecture

Entry point is `backend/main.py`: it loads `.env` and calls `create_app()` in `src/api/app.py`, which installs the middleware (`src/api/middleware.py`) and includes one router per area in `src/api/routes/` (health, auth, accounts, uploads, reviews, reports, patients, admin). Request bodies are in `src/api/schemas.py`, response shaping in `src/api/views.py`, shared request helpers in `src/api/deps.py`; `src/api/auth.py` holds accounts/JWT/roles. `POST /report/upload` (`routes/uploads.py`) validates the file synchronously, then returns **202** with a job id (= report id) and the job runs the pipeline below in a background worker (`process_upload` in `src/pipeline/upload.py`; `src/pipeline/workers.py` starts the worker and the outbox relay); progress goes out as `progress` events on `/ws/{report_id}` (`src/pipeline/progress.py`) and is pollable at `GET /report/{id}/job`. Tests use `upload()` in `tests/test_api_flow.py`, which waits for the job. Request flow for an upload:

1. `src/ingestion/document_loader.py` — PDF text, local EasyOCR for scans/photos, hidden-text detection.
2. `src/agents/orchestrator.py` — runs a deterministic pipeline first (GLiNER NER in `src/ner`, ICMR flagging in `src/flagging/risk_flagging.py` + `demographics.py` + `icmr_reference_ranges.json`, trend/baseline, drug interactions in `src/tools`), **without an LLM**. Gemini is only used in a ReAct tool loop to pick conditional tools (history recall via ChromaDB, OpenFDA, optional PubMed) registered in `src/tools/tool_registry.py`.
3. `summarizer.py` writes clinical + patient summaries; `verifier.py` audits them and **fails closed**; `src/evidence/grounding.py` deterministically checks every number in the summaries exists in extracted data and links facts to source lines.
4. The report is saved as *pending review*. Clinician approve/reject/edit endpoints gate visibility; warnings must be acknowledged before approval; AI originals are retained on edit; rejected reports are removed from history so they never affect trends.

Cross-cutting pieces:
- `src/core/llm.py` is the single Gemini client and pinned model (`GEMINI_MODEL`, optional `GEMINI_FALLBACK_MODEL`, timeout/one-retry policy for overload). Don't create other clients.
- `src/security/` — prompt-injection detection (`llm_guardrails.py`), de-identification (patient identifiers are redacted before every prompt and restored server-side at the end), hash-chained audit log (no PHI), password hashing, rate limiting.
- `src/db/` — SQLAlchemy models/session; SQLite locally, Postgres in Docker. Token revocation lives in the auth/models layer (see `tests/test_revocation.py`).
- `src/tools/history_store.py` — ChromaDB store; erasure must clear it too and must not report success if that fails. Only redacted text is embedded.
- `src/core/features.py` — flag-gated features off by default (`ENABLE_FORECASTING`, `ENABLE_PUBMED`).
- `src/interop/fhir.py` (FHIR R4 export), `src/export/queue_pdf.py` (review-queue PDF).
- Frontend: Angular components under `frontend/src/app/components`, all HTTP through `services/api.service.ts`; nginx (`frontend/nginx.conf`) proxies API paths and the WebSocket to the backend, so a new backend route prefix must be added there too.

## Testing notes

- `tests/conftest.py` installs inert stubs (`tests/stubs.py`) for torch/GLiNER/EasyOCR/Chroma/Gemini when not installed, sets env vars, and **hard-sets `DATABASE_URL=sqlite:///:memory:`** with `MEDSIGHT_SKIP_DOTENV=1`. API tests drop all tables between tests — this once wiped the real `backend/medsight.db`, so never weaken that guard.
- Tests that exercise the pipeline monkeypatch agent entry points directly, through `patch_used()` in `tests/patching.py` (it patches every module that calls the name, so tests don't depend on where code lives); real routing/auth/DB/flagging/grounding code runs. Fixture accounts (e.g. `dr.rao`) are defined in `conftest.py`.
- `eval/` is the measured-results harness (synthetic data in `eval/synthetic.py`); CI fails if metrics regress, so changes to flagging, de-identification, or grounding should be checked with `run_eval --check`.

## Repo hygiene

`.pre-commit-config.yaml` blocks committing `.db`/`.pdf` files (possible patient data) and runs gitleaks and bandit. The `samples/` PDFs, `medsight.db`, and any root `*.patch` files are local artifacts; don't commit patient-like data.
