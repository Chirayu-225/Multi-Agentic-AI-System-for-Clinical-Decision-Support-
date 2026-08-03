"""
src/tools/tool_registry.py

All tools available to the Orchestrator Agent.
Each tool is a plain async function — the agent decides which to call.

Tools:
  Local (no external API):
    - extract_entities       GLiNER NER + custom parser
    - check_icmr_flags       ICMR rule engine
    - detect_trends          ChromaDB longitudinal lookup
    - get_patient_history    ChromaDB history retrieval

  External (free, no API key):
    - search_pubmed          PubMed E-utilities API
    - check_drug_safety      OpenFDA drug safety API
"""

import json
import httpx
import asyncio
from typing import Any


# ── Local tools ───────────────────────────────────────────────────────────────

async def extract_entities(text: str) -> dict:
    """
    Extract medical entities from report text using GLiNER NER
    and a custom line-by-line lab value parser.
    """
    try:
        from src.ner.medical_ner import extract_entities as _extract
        from src.ner.medical_ner import merge_chunk_entities
        from src.ingestion.document_loader import chunk_text
        chunks = chunk_text(text, chunk_size=400, overlap=50)
        results = [_extract(c) for c in chunks[:8]]
        return merge_chunk_entities(results)
    except Exception as e:
        import traceback
        print(f"[TOOL ERROR] extract_entities failed: {traceback.format_exc()}")
        # Return minimal structure so pipeline continues
        return {
            "patient": {"name": None, "age": None, "gender": None},
            "clinical": {"diagnoses": [], "symptoms": [], "procedures": [], "body_parts": []},
            "medications": [],
            "lab_values": [],
            "dates": [],
            "providers": {"doctor": None, "hospital": None},
            "raw_entities": [],
            "error": str(e),
        }


async def check_icmr_flags(lab_values: list[dict], gender: str = None) -> dict:
    """
    Check lab values against ICMR Indian reference ranges.
    Returns rule_flags with severity levels (NORMAL/MODERATE/HIGH/CRITICAL).
    """
    from src.flagging.risk_flagging import flag_lab_values, compute_risk_summary
    flags = flag_lab_values(lab_values, gender=gender)
    summary = compute_risk_summary(flags, [])
    return {
        "flags": [{**f, "severity": str(f["severity"])} for f in flags],
        "summary": {**summary, "overall_severity": str(summary["overall_severity"])},
    }


async def detect_trends(patient_id: str, current_labs: list[dict]) -> list[dict]:
    """
    Compare current lab values against patient's stored history
    to detect consistently worsening trends across visits.
    """
    from src.flagging.risk_flagging import flag_trends
    from src.tools.history_store import get_historical_lab_values
    historical = get_historical_lab_values(patient_id)
    trends = flag_trends(current_labs, historical)
    return [{**t, "severity": str(t["severity"])} for t in trends]


async def get_patient_history(patient_id: str) -> list[dict]:
    """
    Retrieve all stored reports for a patient from ChromaDB,
    sorted oldest-first for longitudinal context.
    """
    from src.tools.history_store import get_patient_history as _get
    return _get(patient_id)


# ── External tools ────────────────────────────────────────────────────────────

async def search_pubmed(query: str, max_results: int = 3) -> list[dict]:
    """
    Search PubMed for clinical evidence related to a diagnosis or condition.
    Uses the free NCBI E-utilities API — no API key required.

    Returns: list of { title, abstract_snippet, pmid, url }
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Step 1: search for PMIDs
            search_resp = await client.get(f"{base}/esearch.fcgi", params={
                "db": "pubmed",
                "term": f"{query} [Title/Abstract] AND (guideline[pt] OR review[pt])",
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            })
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Step 2: fetch summaries
            summary_resp = await client.get(f"{base}/esummary.fcgi", params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            })
            summary_resp.raise_for_status()
            result_data = summary_resp.json().get("result", {})

            articles = []
            for pmid in pmids:
                if pmid not in result_data:
                    continue
                article = result_data[pmid]
                articles.append({
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "authors": article.get("sortfirstauthor", ""),
                    "journal": article.get("source", ""),
                    "pub_date": article.get("pubdate", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            return articles

    except Exception as e:
        return [{"error": str(e), "query": query}]


async def check_drug_safety(medication_name: str) -> dict:
    """
    Check OpenFDA for active safety recalls or black box warnings
    for a medication. Free API — no key required for basic usage.

    Returns: { medication, has_warnings, warnings, recalls }
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check for adverse events / warnings
            resp = await client.get(
                "https://api.fda.gov/drug/label.json",
                params={
                    "search": f'openfda.brand_name:"{medication_name}" OR openfda.generic_name:"{medication_name}"',
                    "limit": 1,
                },
            )

            if resp.status_code == 404:
                return {
                    "medication": medication_name,
                    "has_warnings": False,
                    "warnings": [],
                    "source": "OpenFDA",
                    "note": "No label data found",
                }

            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                return {"medication": medication_name, "has_warnings": False,
                        "warnings": [], "source": "OpenFDA"}

            label = results[0]
            warnings = label.get("warnings_and_cautions", []) or label.get("warnings", [])
            boxed = label.get("boxed_warning", [])

            return {
                "medication": medication_name,
                "has_warnings": bool(warnings or boxed),
                "boxed_warning": boxed[0][:300] if boxed else None,
                "warnings_snippet": warnings[0][:300] if warnings else None,
                "source": "OpenFDA",
            }

    except Exception as e:
        return {"medication": medication_name, "error": str(e), "source": "OpenFDA"}


# ── Tool schema for Gemini function calling ───────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "extract_entities",
        "description": "Extract patient info, diagnoses, medications, and lab values from medical report text using NER.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw text extracted from the medical report"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "check_icmr_flags",
        "description": "Check extracted lab values against ICMR Indian reference ranges and return severity-graded risk flags.",
        "parameters": {
            "type": "object",
            "properties": {
                "lab_values": {
                    "type": "array",
                    "description": "List of lab value objects with test, value, unit keys",
                    "items": {"type": "object"},
                },
                "gender": {
                    "type": "string",
                    "description": "Patient gender for range selection (Male or Female)",
                },
            },
            "required": ["lab_values"],
        },
    },
    {
        "name": "detect_trends",
        "description": "Detect worsening trends in lab values across multiple patient visits using longitudinal history.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"},
                "current_labs": {
                    "type": "array",
                    "description": "Current visit lab values",
                    "items": {"type": "object"},
                },
            },
            "required": ["patient_id", "current_labs"],
        },
    },
    {
        "name": "get_patient_history",
        "description": "Retrieve stored report history for a patient for longitudinal context.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"}
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "search_pubmed",
        "description": "Search PubMed for clinical guidelines or evidence related to a detected diagnosis. Only call when HIGH or CRITICAL flags are found.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clinical term to search e.g. anemia management guidelines India"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_drug_safety",
        "description": "Check OpenFDA for black box warnings or safety alerts for a detected medication.",
        "parameters": {
            "type": "object",
            "properties": {
                "medication_name": {
                    "type": "string",
                    "description": "Name of the medication to check"
                }
            },
            "required": ["medication_name"],
        },
    },
]

# Map tool names to their async functions
TOOL_FUNCTIONS: dict[str, Any] = {
    "extract_entities":   extract_entities,
    "check_icmr_flags":   check_icmr_flags,
    "detect_trends":      detect_trends,
    "get_patient_history": get_patient_history,
    "search_pubmed":      search_pubmed,
    "check_drug_safety":  check_drug_safety,
}