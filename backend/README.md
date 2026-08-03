# MedSight v2 — Multi-Agent Clinical Intelligence System

> A cloud-native, multi-agent clinical decision support system for Indian healthcare settings.
> Built with Angular, FastAPI, PostgreSQL, and Gemini 2.5 Flash.

---

## Architecture

```
Angular (TypeScript + TailwindCSS)
    ↕ REST + WebSocket
FastAPI backend
    ↕
┌─────────────────────────────────────┐
│  Orchestrator Agent (Gemini 2.5)    │
│  ReAct loop + tool calling          │
│  Tools:                             │
│    extract_entities   [GLiNER]      │
│    check_icmr_flags   [rule engine] │
│    detect_trends      [ChromaDB]    │
│    get_patient_history[ChromaDB]    │
│    search_pubmed      [PubMed API]  │
│    check_drug_safety  [OpenFDA API] │
└──────────────┬──────────────────────┘
               ↓ draft output
┌─────────────────────────────────────┐
│  Verification Agent (Gemini 2.5)    │
│  Audits for hallucinated claims     │
│  Corrects if needed                 │
└──────────────┬──────────────────────┘
               ↓ verified output
PostgreSQL + ChromaDB storage
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Angular 18 + TypeScript + TailwindCSS |
| Backend | FastAPI + SQLAlchemy |
| Database | PostgreSQL |
| Vector store | ChromaDB |
| Agents | Gemini 2.5 Flash (tool calling) |
| NER | GLiNER (zero-shot) |
| OCR | EasyOCR + PyMuPDF |
| External APIs | PubMed NCBI, OpenFDA |
| Deployment | Docker Compose + Nginx |

---

## Setup

### 1. Get your Gemini API key (free)

1. Go to https://aistudio.google.com
2. Sign in with your Google account
3. Click **Get API key → Create API key**
4. Copy it

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your GEMINI_API_KEY
```

### 3. Run with Docker

```bash
docker-compose up --build
```

- Frontend: http://localhost:80
- Backend API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

### 4. Run locally (without Docker)

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm start
# Opens at http://localhost:4200
```

---

## Novelty

- **Multi-agent**: Orchestrator (generates) + Verifier (audits) — same model, distinct roles
- **ICMR reference ranges**: Indian-specific lab thresholds, not Western defaults
- **Longitudinal trends**: Worsening patterns detected across visits via ChromaDB
- **Real external tools**: PubMed for clinical evidence, OpenFDA for drug safety
- **WebSocket streaming**: Live agent activity visible in the UI as it happens
- **Explainable flags**: Every risk traces back to the violated ICMR guideline

---

## Data Sources for Testing

- MTSamples (https://mtsamples.com) — free de-identified medical transcripts
- Synthea (https://github.com/synthetichealth/synthea) — synthetic patient records
