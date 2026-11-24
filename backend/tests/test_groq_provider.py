"""The Groq provider: request shape, JSON mode, error classification, retries, tool-calling chat.

No network: httpx.MockTransport stands in for api.groq.com. What these cannot prove is how the real
service behaves; scripts/check_groq.py does that with a real key.
"""
import asyncio
import json

import httpx
import pytest

import src.core.llm as llm
from pydantic import BaseModel


@pytest.fixture()
def groq(monkeypatch):
    """Switch the provider to groq and record every request the code sends."""
    monkeypatch.setattr(llm, "PROVIDER", "groq")
    monkeypatch.setattr(llm, "MODEL", "model-a")
    monkeypatch.setattr(llm, "FALLBACK_MODEL", None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    state = {"requests": [], "replies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append({"auth": request.headers.get("authorization"),
                                  "url": str(request.url), "body": json.loads(request.content)})
        reply = state["replies"].pop(0) if state["replies"] else {"status": 200, "content": "ok"}
        if reply.get("status", 200) != 200:
            return httpx.Response(reply["status"], json={"error": {"message": "nope"}})
        message = {"role": "assistant", "content": reply.get("content"), "tool_calls": reply.get("tool_calls")}
        return httpx.Response(200, json={"choices": [{"message": message}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 5}})

    monkeypatch.setattr(llm, "_groq_transport", httpx.MockTransport(handler))
    return state


def _call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# ── provider choice ────────────────────────────────────────────────────────────

def test_provider_defaults_follow_the_keys_present():
    assert llm.resolve_provider({}) == "gemini"
    assert llm.resolve_provider({"GROQ_API_KEY": "x"}) == "groq"
    assert llm.resolve_provider({"GROQ_API_KEY": "x", "LLM_PROVIDER": "gemini"}) == "gemini"
    assert llm.resolve_provider({"LLM_PROVIDER": " Groq "}) == "groq"


def test_an_unknown_provider_name_is_refused_not_guessed():
    with pytest.raises(RuntimeError, match="gemini' or 'groq"):
        llm.resolve_provider({"LLM_PROVIDER": "openai"})


def test_the_required_key_follows_the_provider(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "groq")
    assert llm.required_key_name() == "GROQ_API_KEY"
    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    assert llm.required_key_name() == "GEMINI_API_KEY"


# ── single-turn text ───────────────────────────────────────────────────────────

def test_generate_text_sends_model_key_system_and_user(groq):
    groq["replies"] = [{"content": "  hello  "}]
    out = asyncio.run(llm.generate_text("say hi", system_instruction="be brief", label="T"))
    assert out == "hello"
    req = groq["requests"][0]
    assert req["auth"] == "Bearer test-key" and req["url"].endswith("/chat/completions")
    assert req["body"]["model"] == "model-a"
    assert req["body"]["messages"] == [{"role": "system", "content": "be brief"},
                                       {"role": "user", "content": "say hi"}]
    assert "response_format" not in req["body"]


def test_json_schema_turns_on_json_mode_and_states_the_shape(groq):
    class Shape(BaseModel):
        verdict: str

    groq["replies"] = [{"content": '{"verdict": "PASS"}'}]
    out = asyncio.run(llm.generate_text("audit this", json_schema=Shape, label="T"))
    assert json.loads(out) == {"verdict": "PASS"}
    body = groq["requests"][0]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert "verdict" in body["messages"][0]["content"] and "JSON" in body["messages"][0]["content"]


def test_token_usage_is_read_for_call_tracing(groq):
    groq["replies"] = [{"content": "ok"}]
    calls = llm.begin_call_tracking()
    asyncio.run(llm.generate_text("x", label="SUMMARIZER"))
    assert calls[0]["status"] == "ok" and calls[0]["model"] == "model-a"
    resp = llm.GroqResponse({"content": "a"}, {"prompt_tokens": 11, "completion_tokens": 5})
    assert llm._usage(resp) == (11, 5)


def test_reasoning_effort_is_sent_only_to_models_that_accept_it(groq, monkeypatch):
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "Low")
    monkeypatch.setattr(llm, "MODEL", "openai/gpt-oss-120b")
    asyncio.run(llm.generate_text("x", label="T"))
    monkeypatch.setattr(llm, "MODEL", "some/other-model")
    asyncio.run(llm.generate_text("x", label="T"))
    assert groq["requests"][0]["body"]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in groq["requests"][1]["body"]


def test_the_default_model_is_not_one_groq_has_retired():
    retired = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b",
               "meta-llama/llama-4-scout-17b-16e-instruct"}
    assert llm.GROQ_DEFAULT_MODEL not in retired


def test_a_missing_key_fails_before_any_request(groq, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        asyncio.run(llm.generate_text("x", label="T"))
    assert groq["requests"] == []


# ── errors and retries ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,rate,unavail", [(429, True, False), (503, False, True), (500, False, True),
                                                 (400, False, False), (401, False, False), (413, False, False)])
def test_http_statuses_are_classified(status, rate, unavail):
    err = llm.ProviderHTTPError(status, "x")
    assert llm.is_rate_limit(err) is rate
    assert llm.is_unavailable(err) is unavail
    assert llm.is_retryable(err) is (rate or unavail)


def test_transport_failures_are_retryable():
    assert llm.is_retryable(httpx.ConnectTimeout("slow"))
    assert llm.is_retryable(httpx.ConnectError("refused"))


def test_a_rate_limit_is_retried_then_succeeds(groq):
    groq["replies"] = [{"status": 429}, {"content": "recovered"}]
    out = asyncio.run(llm.generate_text("x", label="T", attempts=2, base_wait=0))
    assert out == "recovered" and len(groq["requests"]) == 2


def test_a_bad_key_is_not_retried(groq):
    groq["replies"] = [{"status": 401}]
    with pytest.raises(llm.ProviderHTTPError):
        asyncio.run(llm.generate_text("x", label="T", attempts=3, base_wait=0))
    assert len(groq["requests"]) == 1


def test_the_fallback_model_answers_when_the_main_one_stays_down(groq, monkeypatch):
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "model-b")
    groq["replies"] = [{"status": 429}, {"content": "from b"}]
    used = llm.begin_model_tracking()
    assert asyncio.run(llm.generate_text("x", label="T", attempts=1, base_wait=0)) == "from b"
    assert [r["body"]["model"] for r in groq["requests"]] == ["model-a", "model-b"]
    assert used == {"model-b"}


# ── tool-calling chat ──────────────────────────────────────────────────────────

TOOLS = llm.openai_tools([{"name": "check_drug_safety", "description": "d",
                           "parameters": {"type": "object", "properties": {"drug": {"type": "string"}}}}])


def test_tool_definitions_use_the_chat_completions_shape():
    assert TOOLS == [{"type": "function", "function": {
        "name": "check_drug_safety", "description": "d",
        "parameters": {"type": "object", "properties": {"drug": {"type": "string"}}}}}]


def test_chat_round_trip_answers_each_call_by_its_own_id(groq):
    groq["replies"] = [
        {"content": None, "tool_calls": [_call("check_drug_safety", {"drug": "warfarin"}, "c1"),
                                         _call("check_drug_safety", {"drug": "aspirin"}, "c2")]},
        {"content": "done"},
    ]

    async def go():
        chat = llm.GroqChat("model-a", "system prompt", TOOLS)
        first = await chat.send_message("patient data")
        calls = list(first.function_calls)
        second = await chat.send_message([llm.ToolResult(c.name, {"ok": c.args["drug"]}, c.id) for c in calls])
        return calls, second

    calls, second = asyncio.run(go())
    assert [(c.name, c.args, c.id) for c in calls] == [
        ("check_drug_safety", {"drug": "warfarin"}, "c1"), ("check_drug_safety", {"drug": "aspirin"}, "c2")]
    assert second.text == "done" and second.function_calls == []
    sent = groq["requests"][1]["body"]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool"]
    assert sent[2]["tool_calls"][0]["id"] == "c1"
    assert [m["tool_call_id"] for m in sent[3:]] == ["c1", "c2"]
    assert json.loads(sent[3]["content"]) == {"ok": "warfarin"}
    assert groq["requests"][0]["body"]["tools"] == TOOLS and groq["requests"][0]["body"]["tool_choice"] == "auto"


def test_a_retried_turn_does_not_duplicate_the_history(groq):
    groq["replies"] = [{"status": 429}, {"content": "fine"}]

    async def go():
        chat = llm.GroqChat("model-a", "sys", TOOLS)
        with pytest.raises(llm.ProviderHTTPError):
            await chat.send_message("hello")
        assert len(chat.messages) == 1                    # only the system message: nothing half-added
        await chat.send_message("hello")
        return chat

    chat = asyncio.run(go())
    assert [m["role"] for m in chat.messages] == ["system", "user", "assistant"]


def test_malformed_tool_arguments_become_an_empty_dict_not_a_crash():
    resp = llm.GroqResponse({"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{not json"}}]}, None)
    assert resp.function_calls[0].args == {} and resp.function_calls[0].name == "t"


# ── the orchestrator end to end on the groq provider ────────────────────────────

def test_orchestrator_runs_its_tool_loop_on_groq(groq, monkeypatch):
    import src.agents.orchestrator as orch

    executed = []

    async def fake_execute(name, args):
        executed.append((name, args))
        return {
            "extract_entities": {"lab_values": [{"test": "Hemoglobin", "value": 10.2, "unit": "g/dL"}],
                                 "medications": [], "patient": {"gender": "Male", "age": "54"}},
            "check_icmr_flags": {"flags": []},
            "detect_trends": {"trend_flags": [], "baseline_flags": [], "forecast_flags": []},
            "get_patient_history": {"result": []},
        }.get(name, {"matches": []})

    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    groq["replies"] = [
        {"content": None, "tool_calls": [_call("get_similar_history", {"query": "anaemia"}, "call_9")]},
        {"content": "No further tools needed."},
    ]
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01"))

    assert ("get_similar_history", {"query": "anaemia"}) in executed
    assert out["token_usage"]["model"] == "model-a" and out["llm_calls"] == 2
    assert not [w for w in out["ingestion_warnings"] if "could not run" in w]
    first, second = (r["body"] for r in groq["requests"])
    names = {t["function"]["name"] for t in first["tools"]}
    assert "get_similar_history" in names and "extract_entities" not in names     # only the optional tools
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_9",
                                      "content": json.dumps({"matches": []})}


def test_an_orchestrator_outage_on_groq_keeps_the_deterministic_results(groq, monkeypatch):
    import src.agents.orchestrator as orch

    async def fake_execute(name, args):
        return {"extract_entities": {"lab_values": [], "medications": [], "patient": {}},
                "check_icmr_flags": {"flags": []},
                "detect_trends": {"trend_flags": [], "baseline_flags": [], "forecast_flags": []},
                "get_patient_history": {"result": []}}.get(name, {"matches": []})

    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    groq["replies"] = [{"status": 401}]                      # a bad key is not retried
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01"))
    assert any("could not run" in w for w in out["ingestion_warnings"])
