"""Replace a collaborator for a test without caring which module calls it.

Python looks a name up in the module that imported it, so patching it means patching every module
that uses it. This table is that knowledge in one place: when code moves between modules, fix the
table and the tests stay as they are."""
import importlib

_USED_IN = {
    "load_document": ["src.pipeline.upload"],
    "run_orchestrator": ["src.pipeline.upload"],
    "generate_summaries": ["src.pipeline.upload"],
    "verify_and_correct": ["src.pipeline.upload"],
    "detect_hidden_pdf_text": ["src.pipeline.upload"],
    "llm_unavailable_for": ["src.pipeline.upload"],
    "translate_patient_summary": ["src.pipeline.upload", "src.api.routes.reports"],
    "chroma_store": ["src.pipeline.history_sync"],
    "delete_report_history": ["src.pipeline.history_sync", "src.api.routes.reviews"],
    "delete_patient_history": ["src.pipeline.history_sync", "src.api.routes.patients"],
}


def patch_used(monkeypatch, name, value):
    """monkeypatch `name` in every module that calls it. Raises if a listed module lacks the name."""
    for module in _USED_IN[name]:
        monkeypatch.setattr(importlib.import_module(module), name, value)
