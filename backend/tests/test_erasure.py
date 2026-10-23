"""
Erasure and the vector-store copy of a report.

Before: DELETE /patient/{id} removed SQL rows only. ChromaDB kept the first 512 words of the
UNREDACTED report text plus lab values, and because trend detection reads that store, a
re-used patient id resurrected the old history. Rejected reports also kept feeding trends.
"""
import pytest

import main
from src.tools import history_store
from tests.test_api_flow import (CLIN, RAW_TEXT, client, fresh_state, login, upload)  # noqa: F401


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def add(self, ids, embeddings, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.rows[i] = (d, m)

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


class FakeChroma:
    def __init__(self):
        self.cols = {}

    def get_or_create_collection(self, name, metadata=None):
        return self.cols.setdefault(name, FakeCollection())

    def list_collections(self):
        return list(self.cols)

    def delete_collection(self, name):
        del self.cols[name]


class FakeEmbedder:
    def encode(self, text):
        class V:
            def tolist(self_inner): return [0.0]
        return V()


@pytest.fixture()
def store(monkeypatch):
    fake = FakeChroma()
    monkeypatch.setattr(history_store, "_get_client", lambda: fake)
    monkeypatch.setattr(history_store, "_get_embedder", lambda: FakeEmbedder())
    return fake


def test_store_uses_report_id_and_both_deletes_work(store):
    history_store.store_report("P001", "text a", [], "2026-01-01", report_id="r1")
    history_store.store_report("P001", "text b", [], "2026-02-01", report_id="r2")
    col = store.cols["patient-P001"]
    assert set(col.rows) == {"r1", "r2"}
    history_store.delete_report_history("P001", "r1")
    assert set(col.rows) == {"r2"}
    assert history_store.delete_patient_history("P001") is True
    assert "patient-P001" not in store.cols
    assert history_store.delete_patient_history("P001") is False      # idempotent


def test_upload_stores_redacted_text_only(client, store, monkeypatch):
    monkeypatch.setattr(main, "chroma_store", history_store.store_report)
    ch = login(client, CLIN)
    rid = upload(client, ch, name="Report_Priya_Sharma.pdf").json()["report_id"]
    doc, meta = store.cols["patient-P001"].rows[rid]
    assert "Priya" not in doc and "Sharma" not in doc and "9876543210" not in doc
    assert "filename" not in meta              # filenames often carry the patient's name
    assert "Hemoglobin" in doc                                       # clinical content kept


def test_delete_patient_clears_history_store(client, store, monkeypatch):
    monkeypatch.setattr(main, "chroma_store", history_store.store_report)
    ch = login(client, CLIN)
    upload(client, ch)
    assert "patient-P001" in store.cols
    assert client.delete("/patient/P001", headers=ch).status_code == 200
    assert "patient-P001" not in store.cols


def test_delete_refuses_to_claim_success_if_store_fails(client, monkeypatch):
    ch = login(client, CLIN)
    upload(client, ch)

    def boom(_pid):
        raise RuntimeError("store down")
    monkeypatch.setattr(main, "delete_patient_history", boom)
    assert client.delete("/patient/P001", headers=ch).status_code == 500
    assert client.get("/patient/P001/history", headers=ch).json()["report_count"] == 1


def test_rejected_report_is_removed_from_history(client, store, monkeypatch):
    monkeypatch.setattr(main, "chroma_store", history_store.store_report)
    ch = login(client, CLIN)
    rid = upload(client, ch).json()["report_id"]
    r = client.post(f"/report/{rid}/review", headers=ch,
                    json={"decision": "reject", "note": "values misread from scan"})
    assert r.status_code == 200
    assert rid not in store.cols["patient-P001"].rows


def test_history_store_failure_does_not_fail_a_saved_upload(client, monkeypatch):
    def boom(**_k):
        raise OSError("embedding model unavailable")
    monkeypatch.setattr(main, "chroma_store", boom)
    ch = login(client, CLIN)
    r = upload(client, ch)
    assert r.status_code == 200
    assert any("could not be added to the patient's history" in w
               for w in r.json()["ingestion_warnings"])
