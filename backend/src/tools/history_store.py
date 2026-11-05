"""
src/tools/history_store.py
ChromaDB patient history store — same logic as the prior single-agent version,
adapted to the new project structure.
"""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"
DB_PATH.mkdir(parents=True, exist_ok=True)

_client = None
_embedder = None
# These run in worker threads (see tool_registry.py); the lock stops two first requests
# from each building their own client / loading the embedding model.
_init_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=str(DB_PATH),
                    settings=Settings(anonymized_telemetry=False),
                )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _init_lock:
            if _embedder is None:
                _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _collection_name(patient_id: str) -> str:
    return "patient-" + "".join(c if c.isalnum() else "-" for c in patient_id)


def _get_collection(patient_id: str):
    client = _get_client()
    return client.get_or_create_collection(name=_collection_name(patient_id),
                                           metadata={"hnsw:space": "cosine"})


def store_report(patient_id: str, report_text: str, lab_values: list,
                 report_date: str = None, filename: str = "", report_id: str | None = None) -> str:
    """Store one report for longitudinal recall. Callers pass ALREADY-REDACTED text (see
    main.py). Semantic-recall results from this store are sent to the model as tool
    results, so nothing identifying may live here — which is also why the uploaded
    filename (often "Report_<Patient_Name>.pdf") is NOT stored; `filename` is accepted
    for backward compatibility and ignored. The vector id is the report id, so a single
    report can be removed later."""
    collection = _get_collection(patient_id)
    embedder = _get_embedder()
    report_id = report_id or str(uuid.uuid4())
    date_str = report_date or datetime.now().strftime("%Y-%m-%d")
    short_text = " ".join(report_text.split()[:512])
    embedding = embedder.encode(short_text).tolist()
    metadata = {
        "patient_id": patient_id,
        "report_date": date_str,
        "lab_values_json": json.dumps(lab_values),
    }
    collection.add(ids=[report_id], embeddings=[embedding],
                   documents=[short_text], metadatas=[metadata])
    return report_id


def get_patient_history(patient_id: str) -> list[dict]:
    try:
        collection = _get_collection(patient_id)
        results = collection.get(include=["metadatas", "documents"])
    except Exception:
        return []
    if not results or not results["ids"]:
        return []
    reports = []
    for i, report_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        reports.append({
            "report_id": report_id,
            "report_date": meta.get("report_date", ""),
            "filename": "",
            "lab_values": json.loads(meta.get("lab_values_json", "[]")),
        })
    reports.sort(key=lambda r: r["report_date"])
    return reports


def get_historical_lab_values(patient_id: str) -> list[list[dict]]:
    history = get_patient_history(patient_id)
    return [r["lab_values"] for r in history]


def semantic_search_history(patient_id: str, query: str, top_k: int = 3) -> list[dict]:
    """
    Tier 4 longitudinal intelligence — semantic recall.

    Query this patient's stored report embeddings (already computed and
    persisted by store_report() on every upload — nothing new to compute
    here) for past reports whose TEXT is semantically similar to `query`.
    This is meaning-based recall ("similar past episodes"), distinct from
    the numeric lab-value comparisons in flag_trends/flag_baseline_deviation
    — it can surface a recurring free-text pattern (symptoms, findings,
    notes) that never shows up as a matching lab test name.

    Returns up to top_k: { report_id, report_date, filename, similarity_pct,
    snippet }, most similar first. similarity_pct is a 0-100 approximation
    derived from Chroma's cosine distance (higher = more similar); treat it
    as a relative ranking signal, not a calibrated probability.
    """
    try:
        collection = _get_collection(patient_id)
        if collection.count() == 0:
            return []
        embedder = _get_embedder()
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "documents", "distances"],
        )
    except Exception:
        return []

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    matches = []
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    docs = results["documents"][0]
    dists = results["distances"][0]
    for i in range(len(ids)):
        # Chroma's cosine "distance" ranges roughly 0 (identical) .. 2
        # (opposite); this is a rough, unnormalized similarity signal for
        # ranking/display, not a calibrated probability.
        similarity_pct = max(0.0, min(100.0, (1 - dists[i] / 2) * 100))
        snippet = docs[i][:280] + ("…" if len(docs[i]) > 280 else "")
        matches.append({
            "report_id": ids[i],
            "report_date": metas[i].get("report_date", ""),
            "filename": "",
            "similarity_pct": round(similarity_pct, 1),
            "snippet": snippet,
        })
    return matches


def get_historical_report_dates(patient_id: str) -> list[str]:
    """
    Report dates (YYYY-MM-DD strings, possibly empty), oldest-first, in the
    same order as get_historical_lab_values() — used for slope-based
    forecasting to estimate elapsed time between visits.
    """
    history = get_patient_history(patient_id)
    return [r["report_date"] for r in history]


# ── Erasure ─────────────────────────────────────────────────────────────────
# Deleting a patient in SQL alone used to leave their report text and lab values here, and
# because trend detection reads this store, re-using the patient id brought the old history
# back. These are called by DELETE /patient/{id} and when a clinician rejects a report.

def delete_patient_history(patient_id: str) -> bool:
    """Drop the patient's whole collection. Returns True if one existed. Raises on store
    errors so the caller can refuse to report an erasure that did not happen."""
    client = _get_client()
    name = _collection_name(patient_id)
    try:
        existing = {getattr(c, "name", c) for c in client.list_collections()}
    except Exception:
        existing = {name}  # can't list: attempt the delete and let it raise if it must
    if name not in existing:
        return False
    client.delete_collection(name=name)
    return True


def delete_report_history(patient_id: str, report_id: str) -> None:
    """Remove one report (e.g. rejected by a clinician) so it stops feeding trends."""
    _get_collection(patient_id).delete(ids=[report_id])
