# MedSight — Applied Changes Scaffold

This zip mirrors your project's folder structure. Extract it and copy each
file into the SAME relative path in your actual project (C:\Users\Test\Downloads\Med2\),
overwriting the existing file. Nothing here touches venv/, node_modules/, or
your .env — safe to merge directly.

## New files (didn't exist before)
- backend/src/llm_client.py        — shared OpenRouter client
- frontend/nginx.conf              — was completely missing
- .gitignore                       — add `med/` (your venv folder) to it
- README.md                        — project overview for GitHub

## Modified files
- backend/src/tools/tool_registry.py   — fixed chunk_text import bug
- backend/src/api/main.py              — accepts client-supplied report_id (WebSocket fix)
- backend/src/agents/orchestrator.py   — migrated Gemini → OpenRouter, rewritten ReAct loop
- backend/src/agents/verifier.py       — migrated Gemini → OpenRouter
- backend/src/agents/summarizer.py     — migrated Gemini → OpenRouter, fixed boxed_warning crash
- backend/src/ner/medical_ner.py       — added table-row parser (Method 2)
- backend/requirements.txt             — google-generativeai → openai
- backend/.env.example                 — GEMINI_* → OPENROUTER_* vars
- docker-compose.yml                   — moved to project ROOT (was in backend/)
- frontend/Dockerfile                  — fixed broken nginx.conf copy path
- frontend/src/environments/environment.prod.ts — aligned apiUrl with nginx routes
- frontend/src/app/services/api.service.ts       — sends reportId on upload
- frontend/src/app/components/report/report-upload.component.ts — passes reportId to uploadReport()
- frontend/src/app/app.component.ts    — sidebar badge text (Gemini → OpenRouter)

## After merging
1. `cd backend && pip install openai==1.51.0 --break-system-packages` (or in your venv)
2. Edit `backend/.env` — replace GEMINI_API_KEY/GEMINI_MODEL with your OPENROUTER_API_KEY
   and a tool-calling-capable :free model ID from https://openrouter.ai/models
3. `del backend\medsight.db` (reset local dev DB — safe, SQLite only)
4. Restart `uvicorn main:app --reload`, restart `npm start`, re-test
