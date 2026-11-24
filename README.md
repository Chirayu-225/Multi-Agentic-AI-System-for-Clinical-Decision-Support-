# Med2: Agentic Medical Assistant

An AI assistant for GPs in Indian clinics. A doctor uploads a patient's lab report (PDF or phone photo). Within about a minute the system shows which values are abnormal, what changed since the last visit, and a plain-language summary the patient can read in their own language. **A clinician must review and approve everything before the patient sees it.**

> This tool assists a clinician's review. It is not a diagnosis and not a substitute for professional medical judgment. Every AI-generated summary carries this disclaimer in the API response and in the UI.

---

## How it works

Think of a junior doctor (the AI), a strict checker (a second AI plus plain code) and a senior doctor (the human) who signs off.

1. **Upload.** A clinician signs in and uploads a PDF or a photo with a pseudonymous patient ID and report date. The API validates the file, answers `202`, and queues a background job.
2. **Read.** Text is extracted from the PDF. Scans and photos go through local OCR (EasyOCR). Hidden text in PDFs is detected.
3. **Extract and flag (no LLM).** GLiNER pulls out the lab values. Plain code compares each one with the **ICMR reference ranges** for the patient's age and sex, checks drug interactions, and compares against the patient's own history. These are the safety-critical steps, so they are deterministic and testable.
4. **Orchestrate.** An LLM agent decides which optional tools to call: semantic recall of past reports (ChromaDB), OpenFDA, PubMed.
5. **Summarise.** One agent writes a clinical summary and a patient summary.
6. **Verify.** A verifier agent audits both summaries and **fails closed**: if verification breaks, the summary is marked unverified instead of passing. A separate code check flags any number in a summary that is not in the extracted data. Every fact links back to its source line (evidence panel).
7. **Review.** The report waits in a review queue. The clinician can edit (the AI original is kept), must acknowledge every warning, then approves or rejects with a reason. Each decision is written to a tamper-evident audit log.
8. **Release.** Only approved reports are visible to the patient. The patient summary can be translated (Hindi, Kannada, Tamil, Telugu, Marathi, Bengali) with the numbers machine-checked, charted over time, or exported as a FHIR R4 bundle.

**The design rule:** the LLM only does what needs language (summarising, choosing tools). Anything that could hurt a patient if wrong (range checks, interactions, number checks, access control, approval) is plain code, with a human at the end.

### Longitudinal intelligence

Beyond one report, the system tracks a patient across visits: a trajectory narrative, deviation from the patient's *own* baseline (needs 3+ prior visits), semantic recall of similar past episodes, and an optional slope forecast (off by default, `ENABLE_FORECASTING`).

---

## Project layout

```
Med2/
├─ README.md                  this file (the only README)
├─ CLAUDE.md                  guidance for Claude Code working in this repo
├─ docker-compose.yml         backend + frontend (nginx) + PostgreSQL
├─ docs/                      DESIGN, SAFETY (hazard log), SECURITY (threat model), EVAL_RESULTS
├─ samples/                   example lab reports for trying the upload (git-ignored)
├─ .github/workflows/         CI: tests, eval gate, bandit, pip-audit, npm audit, gitleaks
│
├─ backend/
│  ├─ main.py                 entry point: loads .env, builds the app (about 25 lines)
│  ├─ requirements*.txt, Dockerfile, .env.example
│  ├─ eval/                   measured-results harness (synthetic data, regression gate, gold set)
│  ├─ scripts/                provider checks, queue benchmark, run tests on PostgreSQL
│  ├─ tests/                  about 570 tests; heavy ML libraries and LLMs are stubbed
│  └─ src/
│     ├─ api/                 the HTTP layer
│     │  ├─ app.py            builds the FastAPI app, startup and shutdown
│     │  ├─ routes/           one file per area: health, auth, accounts, uploads,
│     │  │                    reviews, reports, patients, admin
│     │  ├─ schemas.py        request bodies          views.py   response shaping
│     │  ├─ deps.py           shared request helpers  middleware.py  CORS, headers, request ids
│     │  └─ auth.py           JWT, roles, token revocation
│     ├─ pipeline/            what happens after an upload
│     │  ├─ upload.py         the background job: the whole analysis, step by step
│     │  ├─ progress.py       live progress over WebSocket
│     │  ├─ history_sync.py   keeps ChromaDB in step with the database (outbox)
│     │  └─ workers.py        starts and stops the job worker and outbox relay
│     ├─ agents/              orchestrator, summarizer, verifier, translator, prompts
│     ├─ ingestion/           PDF text, OCR, hidden-text detection
│     ├─ ner/                 GLiNER entity extraction
│     ├─ flagging/            ICMR rule engine, age/sex range selection, trends
│     ├─ evidence/            numeric grounding and fact-to-source linking
│     ├─ tools/               agent tools: drug interactions, history store, PubMed/OpenFDA
│     ├─ security/            guardrails, de-identification, passwords, audit chain, rate limit
│     ├─ jobs/                database-backed job queue, worker, history outbox
│     ├─ db/                  SQLAlchemy models and session
│     ├─ core/                the single LLM client, resilience (breaker, rate limit), feature flags, config
│     ├─ monitoring/          logging and metrics, AI-quality tracking
│     ├─ interop/             FHIR R4 export
│     └─ export/              review-queue PDF
│
└─ frontend/                  Angular 18 + Tailwind
   └─ src/app/
      ├─ components/          auth, dashboard, report, review, history, portal, accounts, shared
      ├─ services/            API client, auth, interceptor, review state, severity
      ├─ models/              TypeScript types for API data
      └─ utils/               small helpers: error messages, downloads, languages
```

**Where to start reading:** `backend/src/pipeline/upload.py` is the story of one report from upload to review queue. `backend/src/api/routes/` shows what the API offers.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Angular 18, TypeScript, TailwindCSS |
| Backend | FastAPI, SQLAlchemy |
| Database | PostgreSQL (SQLite for local dev) |
| Vector store | ChromaDB |
| LLM | Groq (`openai/gpt-oss-120b`, free tier) or Google Gemini, chosen by `LLM_PROVIDER`; one pinned model recorded on every run |
| Local ML | GLiNER (entities), EasyOCR (scans), deterministic ICMR rules |
| External tools | OpenFDA, PubMed (optional) |
| Infra | Docker Compose, Nginx |

---

## Run it locally (Windows Command Prompt)

You need Python 3.11+, Node.js 20+, and a Groq or Gemini API key.

**1. Configure**

```cmd
copy backend\.env.example backend\.env
```

Edit `backend\.env` and set at least:
- `GROQ_API_KEY` (or `GEMINI_API_KEY` with `LLM_PROVIDER=gemini`)
- `JWT_SECRET`, 32+ random characters
- `BOOTSTRAP_CLINICIAN_USERNAME` and `BOOTSTRAP_CLINICIAN_PASSWORD` (10+ characters), your first account

**2. Backend**

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

API on `http://localhost:8001`, Swagger docs at `/docs`.

**3. Frontend** (in a second window, backend first)

```cmd
cd frontend
npm install
npm start
```

App on `http://localhost:4200`. Sign in with the bootstrap account. To give a patient access, open **Accounts**, pick the patient, and hand over the one-time temporary password; they must change it at first sign-in.

**Docker** (from the repo root): put `GROQ_API_KEY` (or `GEMINI_API_KEY`), `JWT_SECRET`, `POSTGRES_PASSWORD` and the bootstrap clinician variables in a root `.env`, then run `docker-compose up --build`. The app is served at `http://localhost`, with Nginx proxying the API and WebSocket.

### Main settings (full list in `backend/.env.example`)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER`, `GROQ_API_KEY`, `GEMINI_API_KEY` | which model provider and its key |
| `JWT_SECRET` | signs login tokens (required) |
| `BOOTSTRAP_CLINICIAN_*` | creates the first clinician account |
| `REQUIRE_CLINICIAN_REVIEW` | default `true`; `false` skips approval (demo only) |
| `DATABASE_URL` | defaults to local SQLite; Docker uses PostgreSQL |
| `ENABLE_FORECASTING`, `ENABLE_PUBMED` | optional features, off by default |
| `JOB_WORKERS`, `LLM_RPM`, `LLM_MAX_CALLS_PER_REPORT` | background capacity and LLM cost controls |

---

## Tests and evaluation

```cmd
cd backend
pip install -r requirements-dev.txt
python -m pytest -q
python -m eval.run_eval --check
```

The test suite needs no API key and no GPU. The eval harness scores 200 synthetic reports against thresholds and rewrites `docs/EVAL_RESULTS.md`; CI fails if the results regress. `python -m eval.run_gold --check` does the same on real de-identified reports you collect yourself (git-ignored).

---

## Safety and security in short

- **Human approval gate.** Nothing reaches a patient before a clinician approves; warnings must be acknowledged; edits keep the AI original.
- **Fail-closed verification** plus deterministic number grounding and curated drug-interaction rules.
- **De-identification before every model call.** Names and IDs are removed from prompts and restored only on the server. OCR is local only.
- **Prompt-injection defence:** text normalisation, pattern detection, hidden-PDF-text detection, output scanning.
- **Accounts and roles** (clinician, patient, service), scrypt passwords, lockout, revocable JWTs, per-patient access checks.
- **Tamper-evident audit log** (hash chain, no patient data) and **real erasure** that also clears the vector store.
- **Reliability:** a database-backed job queue, circuit breaker and shared rate limit for the LLM, an outbox for the vector store.
- **Measured, including the weak spots.** `docs/SECURITY.md` has the threat model and known gaps; `docs/SAFETY.md` lists each hazard with the test that proves its control.

Deeper reading: `docs/DESIGN.md` (why it is built this way), `docs/SAFETY.md`, `docs/SECURITY.md`, `docs/EVAL_RESULTS.md`.
