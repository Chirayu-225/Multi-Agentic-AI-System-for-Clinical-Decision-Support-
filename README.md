# MedSight — Clinical Decision Support System

A multi-agent AI system for analysing patient lab reports, flagging risks against ICMR
reference ranges, and generating AI-verified clinical summaries for healthcare providers.

## Architecture

| Layer         | Tech                                      |
|---------------|--------------------------------------------|
| Frontend      | Angular + TypeScript + TailwindCSS         |
| Backend       | FastAPI + SQLAlchemy                       |
| Database      | PostgreSQL (SQLite supported for local dev)|
| Vector store  | ChromaDB                                   |
| Model         | Gemini 2.5 Flash (free tier)               |
| Agents        | Orchestrator (ReAct + tool calling), Verifier (structured audit + correction) |
| Local tools   | GLiNER NER, ICMR rule engine, trend detection |
| External tools| PubMed API, OpenFDA API                    |
| Infra         | Docker Compose + Nginx                     |

## How it works

1. A PDF report is uploaded with a patient ID and report date.
2. The **Orchestrator** agent runs a ReAct loop with native Gemini tool calling —
   extracting entities (GLiNER), checking ICMR risk flags, pulling patient history from
   ChromaDB, and conditionally calling PubMed / OpenFDA when relevant.
3. The **Verifier** agent audits the Orchestrator's output for accuracy and issues a
   corrected clinical summary and patient-facing plain-language summary.
4. Results, including risk flags and recommended actions, are shown in the UI with a
   live agent-activity feed streamed over WebSocket during processing.

## Prerequisites

- Python 3.11+
- Node.js 20+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))

## Local setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# add your GEMINI_API_KEY to .env

uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.

### Frontend

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

This builds and runs the backend, frontend (served via Nginx, which also proxies API
and WebSocket traffic to the backend), and a PostgreSQL database.

## Environment variables

See `backend/.env.example` for the full list. Required:

- `GEMINI_API_KEY` — your Gemini API key
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash`
- `DATABASE_URL` — defaults to local SQLite; set to a Postgres URL for production

## Notes

- Free-tier Gemini API quotas apply to both agents; heavy concurrent use may hit rate
  limits.
- PubMed and OpenFDA APIs are used without an API key (both support this for
  reasonable request volumes).
