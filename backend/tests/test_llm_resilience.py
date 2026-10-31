"""src/llm.py: timeouts, the single retry policy, and the fallback model.

Regression: gemini-3.8-flash returned '503 UNAVAILABLE: high demand' and the SDK's default
5 silent attempts with no request timeout made every Gemini call look frozen."""
import asyncio

import pytest

import src.llm as llm


class Err503(Exception):
    def __str__(self):
        return "503 UNAVAILABLE. This model is currently experiencing high demand."


class Err400(Exception):
    def __str__(self):
        return "400 INVALID_ARGUMENT. API key not valid."


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _instant(_s):
        return None
    monkeypatch.setattr(llm.asyncio, "sleep", _instant)


def _calls(script):
    """make_call(model) that replays `script` per model: a list of exceptions/results."""
    seen = []

    async def make_call(model):
        seen.append(model)
        outcome = script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return make_call, seen


def test_overload_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "MODEL", "main")
    call, seen = _calls({"main": [Err503(), Err503(), "ok"]})
    assert asyncio.run(llm.with_fallback(call, label="T")) == ("ok", "main")
    assert seen == ["main"] * 3


def test_non_retryable_error_fails_fast_without_fallback(monkeypatch):
    monkeypatch.setattr(llm, "MODEL", "main")
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "backup")
    call, seen = _calls({"main": [Err400()]})
    with pytest.raises(Err400):
        asyncio.run(llm.with_fallback(call, label="T"))
    assert seen == ["main"]                       # no retries, no fallback for a bad key


def test_fallback_used_and_recorded_when_main_stays_overloaded(monkeypatch):
    monkeypatch.setattr(llm, "MODEL", "main")
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "backup")
    call, seen = _calls({"main": [Err503()] * 3, "backup": ["from backup"]})

    async def run():
        used = llm.begin_model_tracking()
        return await llm.with_fallback(call, label="T"), used
    (result, model), used = asyncio.run(run())
    assert (result, model) == ("from backup", "backup")
    assert seen == ["main", "main", "main", "backup"] and used == {"backup"}


def test_without_fallback_overload_surfaces(monkeypatch):
    monkeypatch.setattr(llm, "MODEL", "main")
    monkeypatch.setattr(llm, "FALLBACK_MODEL", None)
    call, _ = _calls({"main": [Err503()] * 3})
    with pytest.raises(Err503):
        asyncio.run(llm.with_fallback(call, label="T"))


def test_timeouts_count_as_retryable():
    assert llm.is_retryable(asyncio.TimeoutError())
    class ReadTimeout(Exception): pass
    assert llm.is_retryable(ReadTimeout("read timed out"))
    assert not llm.is_retryable(Err400())


def test_client_has_timeout_and_sdk_retries_off(monkeypatch):
    pytest.importorskip("google.genai.types")
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "TIMEOUT_SECONDS", 90.0)
    captured = {}
    import google.genai as genai

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)
    monkeypatch.setattr(genai, "Client", FakeClient)
    llm.get_client()
    opts = captured["http_options"]
    assert opts.timeout == 90_000 and opts.retry_options.attempts == 1
    monkeypatch.setattr(llm, "_client", None)


def test_orchestrator_falls_back_and_reports_model(monkeypatch):
    from tests.test_orchestrator_tool_loop import DETERMINISTIC, FakeChat, _Resp
    import src.agents.orchestrator as orch

    monkeypatch.setattr(llm, "MODEL", "main")
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "backup")
    created = []

    class DeadChat:
        async def send_message(self, m):
            raise Err503()

    class Client:
        class aio:
            class chats:
                @staticmethod
                def create(model, config):
                    created.append(model)
                    return DeadChat() if model == "main" else FakeChat([[]])

    async def fake_execute(name, args):
        return DETERMINISTIC.get(name, {})
    monkeypatch.setattr(orch, "MODEL", "main")
    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    monkeypatch.setattr(orch, "get_client", lambda: Client)
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01"))
    assert created == ["main", "main", "main", "backup"]
    assert out["token_usage"]["model"] == "backup"
    assert not any("could not run" in w for w in out["ingestion_warnings"])
