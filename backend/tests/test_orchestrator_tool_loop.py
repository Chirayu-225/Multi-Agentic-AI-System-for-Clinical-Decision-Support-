"""
The orchestrator's tool-calling loop against a fake google-genai chat.

Guards the SDK contract the migration to google-genai depends on:
  - the model is offered only the three conditional tools, as JSON-schema declarations;
  - the SDK's automatic function calling is OFF (we run tools ourselves);
  - tool results go back on the SAME chat session as function_response parts, so the
    session can resend the model's thought signatures (Gemini 3 rejects the turn otherwise);
  - a repeated identical call is served from cache and ends the loop.
"""
import asyncio

import pytest

genai_types = pytest.importorskip("google.genai.types")


class _Call:
    def __init__(self, name, args):
        self.name, self.args = name, args


class _Resp:
    def __init__(self, calls):
        self.function_calls = calls


class FakeChat:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    async def send_message(self, message):
        self.sent.append(message)
        return _Resp(self.script.pop(0) if self.script else [])


class FakeClient:
    def __init__(self, chat):
        self.chat, self.created = chat, []
        outer = self

        class _Chats:
            def create(self, model, config):
                outer.created.append((model, config))
                return outer.chat

        class _Aio:
            chats = _Chats()

        self.aio = _Aio()


DETERMINISTIC = {
    "extract_entities": {"lab_values": [{"test": "Hemoglobin", "value": 10.2, "unit": "g/dL"}],
                         "medications": [], "patient": {"gender": "Male", "age": "54"}},
    "check_icmr_flags": {"flags": []},
    "detect_trends": {"trend_flags": [], "baseline_flags": [], "forecast_flags": []},
    "get_patient_history": {"result": []},
}


def _run(monkeypatch, script):
    import src.agents.orchestrator as orch

    executed = []

    async def fake_execute(name, args):
        executed.append(name)
        if name in DETERMINISTIC:
            return DETERMINISTIC[name]
        if name == "search_pubmed":
            return {"result": [{"title": "t", "pmid": "1"}]}
        return {"matches": []}

    chat = FakeChat(script)
    client = FakeClient(chat)
    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    monkeypatch.setattr(orch, "get_client", lambda: client)
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01"))
    return out, chat, client, executed


def test_tool_loop_round_trip(monkeypatch):
    monkeypatch.setenv("ENABLE_PUBMED", "true")
    out, chat, client, executed = _run(monkeypatch, [
        [_Call("search_pubmed", {"query": "anaemia"})],
        [],
    ])
    (model, config), = client.created
    import src.agents.orchestrator as orch
    assert model == orch.MODEL
    assert config.automatic_function_calling.disable is True
    names = {d.name for d in config.tools[0].function_declarations}
    assert names == orch.AGENTIC_TOOL_NAMES          # PubMed enabled for this test

    assert len(chat.sent) == 2                       # opening turn + one tool-result turn
    parts = chat.sent[1]
    assert isinstance(parts, list) and parts[0].function_response.name == "search_pubmed"
    assert "search_pubmed" in executed
    assert out["llm_calls"] == 2


def test_repeated_call_is_cached_and_stops(monkeypatch):
    monkeypatch.setenv("ENABLE_PUBMED", "true")
    same = [_Call("search_pubmed", {"query": "anaemia"})]
    out, chat, _client, executed = _run(monkeypatch, [same, same, same])
    assert executed.count("search_pubmed") == 1
    assert len(chat.sent) == 2                       # loop stops instead of resending


def test_pubmed_and_forecasting_are_off_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_PUBMED", raising=False)
    monkeypatch.delenv("ENABLE_FORECASTING", raising=False)
    DETERMINISTIC["detect_trends"] = {"trend_flags": [], "baseline_flags": [],
                                      "forecast_flags": [{"test": "hba1c", "severity": "MODERATE"}]}
    try:
        out, chat, client, _ = _run(monkeypatch, [[]])
    finally:
        DETERMINISTIC["detect_trends"] = {"trend_flags": [], "baseline_flags": [], "forecast_flags": []}
    (_m, config), = client.created
    names = {d.name for d in config.tools[0].function_declarations}
    assert "search_pubmed" not in names and names == {"get_similar_history", "check_drug_safety"}
    assert "search_pubmed" not in config.system_instruction
    assert "check_drug_safety" in config.system_instruction
    assert out["forecast_flags"] == []


def test_forecasting_flag_turns_forecasts_back_on(monkeypatch):
    monkeypatch.setenv("ENABLE_FORECASTING", "true")
    DETERMINISTIC["detect_trends"] = {"trend_flags": [], "baseline_flags": [],
                                      "forecast_flags": [{"test": "hba1c", "severity": "MODERATE"}]}
    try:
        out, *_ = _run(monkeypatch, [[]])
    finally:
        DETERMINISTIC["detect_trends"] = {"trend_flags": [], "baseline_flags": [], "forecast_flags": []}
    assert len(out["forecast_flags"]) == 1


def test_model_failure_keeps_deterministic_results(monkeypatch):
    """A Gemini outage / bad key must not turn a fully-flagged report into a 500."""
    import src.agents.orchestrator as orch

    class DeadChat:
        async def send_message(self, message):
            raise RuntimeError("401 API key not valid")

    class DeadClient(FakeClient):
        def __init__(self):
            super().__init__(DeadChat())

    async def fake_execute(name, args):
        return DETERMINISTIC.get(name, {})

    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    monkeypatch.setattr(orch, "get_client", lambda: DeadClient())
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01"))
    assert out["entities"]["lab_values"]                      # deterministic work survived
    assert "risk_summary" in out
    assert any("could not run" in w for w in out["ingestion_warnings"])


def test_ner_outage_still_extracts_lab_values(monkeypatch):
    """GLiNER failing to load (offline host, first-run download blocked) used to drop EVERY
    lab value, including ones the model-free parser reads perfectly. Found by a real run."""
    from src.ner import medical_ner
    from src.tools import tool_registry

    def no_model():
        raise OSError("could not download urchade/gliner_mediumv2.1")
    monkeypatch.setattr(medical_ner, "_get_model", no_model)
    out = asyncio.run(tool_registry.extract_entities(
        "Hemoglobin        12.5    g/dL   12.0 - 15.0\nSerum Creatinine - 1.9 mg/dL 0.7 to 1.3\n"))
    assert out["ner_unavailable"] is True
    assert {l["test"] for l in out["lab_values"]} == {"hemoglobin", "creatinine"}
