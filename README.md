# Agentic Medical Assistant

**For GPs in Indian clinics: see what changed in a patient's labs since the last visit —
checked against the right reference range, verified, and explained in the patient's
language — in under a minute.**

A multi-agent AI system that reads lab reports (PDFs or phone photos), flags values against
ICMR ranges chosen for the patient's age and sex, tracks each patient across visits, and
writes AI-verified summaries that a clinician reviews, edits and approves before release.

> **This tool assists a clinician's review — it is not a diagnosis and not a substitute
> for professional medical judgment.** Every AI-generated summary carries this disclaimer
> end-to-end (API response + UI banner). See [Safety & security](#safety--security) below
> for what this project does and does not protect against.

## Architecture

| Layer          | Tech                                          |
|----------------|------------------------------------------------|
| Frontend       | Angular + TypeScript + TailwindCSS              |
| Backend        | FastAPI + SQLAlchemy                            |
| Database       | PostgreSQL (SQLite supported for local dev)     |
| Vector store   | ChromaDB                                        |
| Model          | Google Gemini 3.8 Flash via the `google-genai` SDK, pinned by `GEMINI_MODEL` and recorded on every run |
| Agents         | Orchestrator (ReAct + tool calling), Verifier (structured audit + correction, fails CLOSED) |
| Local tools    | GLiNER NER, EasyOCR (scans/photos), ICMR rule engine, drug-interaction rules, 4-tier longitudinal intelligence |
| External tools | OpenFDA API; PubMed API (optional, `ENABLE_PUBMED`) |
| Security       | JWT accounts with roles, clinician approval gate, hash-chained audit log, prompt-injection detection, de-identification, numeric grounding, rate limiting |
| Infra          | Docker Compose + Nginx                          |

## How it works

1. A clinician signs in and uploads a PDF report **or a photo/scan** (JPG, PNG, WebP) with a
   pseudonymous patient ID and report date. Scans go through local OCR with a confidence
   check; hidden text in PDFs is detected.
2. The **Orchestrator** runs entity extraction (GLiNER), ICMR risk-flagging,
   trend/baseline detection and a deterministic **drug-interaction check** as a plain
   pipeline, no LLM call. Sex and age are normalised (`F`, `32Y/F`, `54 Years / Male`...)
   so the right ICMR range is used, and the lab's own printed range is read and compared.
   Gemini is only asked to choose among the conditional tools (semantic history recall,
   OpenFDA, and PubMed when enabled). Patient identifiers are redacted before any
   prompt is built and restored once, server-side, at the end.
3. The **Summarizer** writes the clinical and patient summaries, the **Verifier** audits
   both in one call (and fails closed), and a deterministic **numeric grounding** check
   flags any number not found in the extracted data. Every extracted fact is linked back
   to its source line (**evidence panel**).
4. The report is saved as **pending review**. From the review queue a clinician can edit
   either summary (the AI's original is kept), must explicitly acknowledge any warning
   (unverified summary, ungrounded number, injection signal, extraction problem, range
   disagreement) before approving, or rejects it with a reason. Only approved reports are
   visible to the patient. Each decision is written to a tamper-evident audit log, and a
   rejected report is removed from the patient's history so it never shapes a trend.
5. Optional: translate the patient summary (Hindi, Kannada, Tamil, Telugu, Marathi,
   Bengali; numbers are machine-checked against the English text), chart lab trends over
   time, and export a FHIR R4 Bundle.

## Longitudinal intelligence (4 tiers)

Beyond a single report, the system tracks a patient across visits at increasing depth:

1. **Trajectory narrative** — an LLM-written paragraph tracing the patient's trend across
   visits, grounded only in the actual flagged values.
2. **Personal-baseline deviation** — flags a value that has drifted meaningfully from
   *this patient's own* historical mean (z-score + minimum % change gate), even while
   still inside the general population's normal range. Requires ≥3 prior visits.
3. **Slope-based forecasting** *(off by default, `ENABLE_FORECASTING=true`)* — an
   illustrative linear projection of when a monotonically-trending value would cross its
   next threshold. A fit over a handful of visits shown to a clinician reads as a
   prediction it can't support, so it is hidden from the product; when enabled it is
   capped at MODERATE and excluded from the top-level risk score.
4. **Semantic history recall** — searches the patient's past report *text* (not just lab
   values) for episodes similar in meaning to the current findings, via the ChromaDB
   embeddings already computed on every upload.

## Safety & security

Full threat model, OWASP LLM Top 10 (2026) mapping, measured results and known gaps are in
[SECURITY.md](SECURITY.md). In short:

- **Named accounts and roles** (clinician / patient / service), scrypt passwords, lockout,
  JWT with pinned algorithm, object-level authorisation (patients only ever see their own
  approved reports).
- **Human-in-the-loop release gate**: nothing reaches a patient until a clinician approves;
  warnings must be acknowledged, edits keep the AI original, and `/admin/metrics` reports
  approve-without-edit rate and time-to-review so rubber-stamping is visible.
- **Erasure that is real**: deleting a patient clears the vector store too (and refuses to
  report success if it can't); only redacted text, never filenames, is embedded.
- **Hash-chained audit log** (no PHI) with an integrity check and a dashboard.
- **Layered misinformation defence**: fail-closed Verifier, deterministic numeric grounding,
  evidence linking, curated drug-interaction rules.
- **Prompt-injection defence**: normalisation, pattern detection, hidden-PDF-text detection,
  untrusted-text delimiting, output scanning. Detected and surfaced, not silently blocked.
- **De-identification before every model call**, local-only OCR, translation validated
  deterministically, model text rendered without `innerHTML` (enforced in CI).
- **Measured, not claimed**: `cd backend && python -m eval.run_eval` runs a 200-report
  synthetic evaluation and writes [EVAL_RESULTS.md](EVAL_RESULTS.md); CI fails if it
  regresses. The results include the weaknesses (for example a 14% name leak when the NER
  misses, and 0% detection of value swaps by the numeric check). Synthetic data only proves
  the logic; `python -m eval.run_gold` scores the same pipeline on **real, de-identified
  reports** you collect (see the module docstring — the gold set is git-ignored).
- **No PHI in logs or version control.**

## Tests and CI

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest -q                # no API key or GPU needed
python -m eval.run_eval --check    # synthetic evaluation with regression thresholds
python -m eval.run_eval --live     # also measures the real Verifier (needs GEMINI_API_KEY)
python -m eval.run_gold --check    # real-report gold set (needs eval/gold/labels.json)
```

Heavy ML libraries and Gemini are stubbed in tests; the real routing, auth, database,
flagging, interaction and grounding code runs. GitHub Actions
(`.github/workflows/security.yml`) runs tests, the eval gate, bandit, pip-audit, npm audit,
gitleaks and a production frontend build on every push and weekly.

## Prerequisites

- Python 3.11+
- Node.js 20+
- A Gemini API key ([aistudio.google.com](https://aistudio.google.com/) → Get API Key)

## Local setup

### 1. Configure environment

```bash
cp backend/.env.example backend/.env      # macOS/Linux
copy backend\.env.example backend\.env    # Windows
```

Edit `backend/.env` and set:
- `GEMINI_API_KEY` — from [aistudio.google.com](https://aistudio.google.com/) (Get API Key)
- `GEMINI_MODEL` — defaults to `gemini-3.8-flash`. `gemini-2.5-flash` only works for API
  keys that have used it before, so new keys need a current model
- `JWT_SECRET` — 32+ random characters: `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `API_KEY` — a random string for the optional shared service key
- `BOOTSTRAP_CLINICIAN_USERNAME` / `BOOTSTRAP_CLINICIAN_PASSWORD` (10+ characters) — your
  first account, created at startup. Optionally also `BOOTSTRAP_PATIENT_USERNAME`,
  `BOOTSTRAP_PATIENT_PASSWORD` and `BOOTSTRAP_PATIENT_PATIENT_ID` for a patient-portal demo.

Then sign in with that account on the login screen. Leave `apiKey` blank in
`frontend/src/environments/*.ts`; filling it in only enables a limited "service key"
fallback that cannot approve reports (and bakes a secret into the bundle).

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000` (Swagger docs at `/docs`).

### 3. Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:4200`.

**Start the backend first** — the frontend expects it to be up on `:8000`.

## Docker deployment

```bash
docker-compose up --build
```

Run this from the **repository root** (the compose file lives at the root, not inside
`backend/`). It builds and runs the backend, frontend (served via Nginx, which also
proxies API and WebSocket traffic to the backend), and a PostgreSQL database.

`docker-compose.yml` requires `GEMINI_API_KEY`, `API_KEY`, `JWT_SECRET` and
`POSTGRES_PASSWORD` (no insecure defaults) — put them in a `.env` file at the repository root:

```bash
# .env at repo root (used by docker-compose, separate from backend/.env)
GEMINI_API_KEY=your-key-here
API_KEY=your-generated-shared-secret
JWT_SECRET=at-least-32-random-characters
POSTGRES_PASSWORD=some-strong-password
BOOTSTRAP_CLINICIAN_USERNAME=dr.example
BOOTSTRAP_CLINICIAN_PASSWORD=a-strong-passphrase
```

Users sign in with a username and password, so no secret is baked into the frontend
bundle. Nginx proxies `/auth`, `/report`, `/reviews`, `/me`, `/patient(s)`, `/admin`,
`/health` and the WebSocket to the backend. Put TLS in front of it for anything beyond a demo.

- Frontend: http://localhost:80
- Backend API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## Environment variables

See `backend/.env.example` for the full list and inline documentation. Key ones:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini API access |
| `JWT_SECRET` | yes (32+ chars) | Signs login tokens; unset = random per-process secret |
| `API_KEY` | yes | Shared service key (role `service`; set `ALLOW_API_KEY_AUTH=false` to disable) |
| `BOOTSTRAP_CLINICIAN_USERNAME` / `_PASSWORD` | first run | Creates the first clinician account |
| `REQUIRE_CLINICIAN_REVIEW` | no | default `true`; `false` auto-approves (demo only) |
| `ACCESS_TOKEN_MINUTES` | no | default 60 |
| `AUDIT_ENABLED` | no | default `true` |
| `OCR_GPU` | no | `auto` / `true` / `false` |
| `GEMINI_MODEL` | no | defaults to `gemini-3.8-flash`; one model for every agent |
| `ENABLE_FORECASTING` | no | default `false`; slope-based projections |
| `ENABLE_PUBMED` | no | default `false`; literature links for HIGH/CRITICAL findings |
| `DATABASE_URL` | no | defaults to local SQLite; Docker Compose overrides to Postgres |
| `CORS_ORIGINS` | no | comma-separated allowed origins |
| `UPLOAD_RATE_LIMIT_MAX` / `UPLOAD_RATE_LIMIT_WINDOW_SECONDS` | no | defaults 10 / 600 |
| `LOG_LEVEL` | no | defaults `INFO` |

## Notes

- Free-tier Gemini quotas apply to the model-calling agents (and translation); heavy concurrent use may hit rate
  limits (both `summarizer.py` and `verifier.py` retry with backoff; `orchestrator.py`'s
  ReAct-loop calls retry too).
- PubMed and OpenFDA APIs are used without an API key (both support this for
  reasonable request volumes).
- Long reports are truncated (report text at ~20k characters, entity extraction at 20
  chunks) with the truncation surfaced as an `ingestion_warnings` entry in the API
  response and a banner in the UI — not silent.

## Output Snippet

<img width="1919" height="907" alt="image" src="https://github.com/user-attachments/assets/3397caf5-3801-420f-b40e-a55a9f9d9bbe" />
