"""
src/tools/history_store.py
ChromaDB patient history store — same logic as MedSight v1,
adapted to the new project structure.
"""
import json
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


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection(patient_id: str):
    client = _get_client()
    safe_id = "patient-" + "".join(c if c.isalnum() else "-" for c in patient_id)
    return client.get_or_create_collection(name=safe_id, metadata={"hnsw:space": "cosine"})


def store_report(patient_id: str, report_text: str, lab_values: list,
                 report_date: str = None, filename: str = "") -> str:
    collection = _get_collection(patient_id)
    embedder = _get_embedder()
    report_id = str(uuid.uuid4())
    date_str = report_date or datetime.now().strftime("%Y-%m-%d")
    short_text = " ".join(report_text.split()[:512])
    embedding = embedder.encode(short_text).tolist()
    metadata = {
        "patient_id": patient_id,
        "report_date": date_str,
        "filename": filename,
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
            "filename": meta.get("filename", ""),
            "lab_values": json.loads(meta.get("lab_values_json", "[]")),
        })
    reports.sort(key=lambda r: r["report_date"])
    return reports


def get_historical_lab_values(patient_id: str) -> list[list[dict]]:
    history = get_patient_history(patient_id)
    return [r["lab_values"] for r in history]
