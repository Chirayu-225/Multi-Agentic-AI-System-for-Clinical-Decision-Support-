"""Model-call tracing, per-report budget, notional cost, and prompt versions."""
import asyncio
from types import SimpleNamespace

import pytest

import main
from src import llm, prompts
from src.db.models import AgentRun, LlmCall
from src.db.session import SessionLocal

from tests.test_api_flow import (  # noqa: F401
    CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


def resp(tin=100, tout=40):
    return SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=tin, candidates_token_count=tout),
                           text="ok")


def run(coro_fn, **kw):
    calls = llm.begin_call_tracking(kw.pop("max_calls", None))
    out = asyncio.run(coro_fn())
    return out, calls


# ── trace records ──────────────────────────────────────────────────────────────

def test_a_successful_call_records_purpose_model_latency_and_tokens():
    async def go():
        return await llm.with_retry(lambda: asyncio.sleep(0, resp(120, 30)), label="SUMMARIZER", model="m-1")

    _, calls = run(go)
    (c,) = calls
    assert c["purpose"] == "summarizer" and c["model"] == "m-1" and c["status"] == "ok"
    assert (c["input_tokens"], c["output_tokens"]) == (120, 30) and c["latency_ms"] >= 0 and c["attempt"] == 1


def test_a_failed_then_successful_call_records_both_attempts(monkeypatch):
    real_sleep = asyncio.sleep

    async def no_wait(_s):
        await real_sleep(0)

    monkeypatch.setattr(llm.asyncio, "sleep", no_wait)
    n = {"i": 0}

    async def flaky():
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("503 UNAVAILABLE")
        return resp()

    async def go():
        return await llm.with_retry(flaky, label="VERIFIER", model="m-1", base_wait=1)

    _, calls = run(go)
    assert [(c["attempt"], c["status"]) for c in calls] == [(1, "unavailable"), (2, "ok")]
    assert calls[0]["error_class"] == "RuntimeError"


def test_trace_holds_metadata_only_never_prompt_or_response_text():
    secret = "Priya Sharma 9876543210"

    async def go():
        return await llm.with_retry(lambda: asyncio.sleep(0, SimpleNamespace(text=secret, usage_metadata=None)),
                                    label="SUMMARIZER", model="m")

    _, calls = run(go)
    assert secret not in repr(calls)
    assert set(calls[0]) == {"purpose", "model", "attempt", "status", "error_class", "latency_ms",
                             "input_tokens", "output_tokens"}


def test_missing_usage_metadata_is_zero_not_an_error():
    async def go():
        return await llm.with_retry(lambda: asyncio.sleep(0, "plain string"), label="X", model="m")

    _, calls = run(go)
    assert calls[0]["input_tokens"] == 0 and calls[0]["status"] == "ok"


def test_open_circuit_is_recorded_without_calling_the_provider():
    from src.resilience import CircuitBreaker, CircuitOpen
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
    b.record_failure()
    called = []

    async def go():
        with pytest.raises(CircuitOpen):
            await llm.with_retry(lambda: called.append(1), label="SUMMARIZER", breaker=b, model="m")

    _, calls = run(go)
    assert called == [] and [c["status"] for c in calls] == ["circuit_open"]


# ── budget ─────────────────────────────────────────────────────────────────────

def test_per_report_budget_stops_a_runaway_loop():
    async def go():
        for _ in range(10):
            await llm.with_retry(lambda: asyncio.sleep(0, resp()), label="ORCHESTRATOR", model="m")

    calls = llm.begin_call_tracking(max_calls=3)
    with pytest.raises(llm.LlmBudgetExceeded):
        asyncio.run(go())
    assert [c["status"] for c in calls] == ["ok", "ok", "ok", "budget"]


def test_no_budget_means_unlimited():
    async def go():
        for _ in range(20):
            await llm.with_retry(lambda: asyncio.sleep(0, resp()), label="X", model="m")

    calls = llm.begin_call_tracking(None)
    asyncio.run(go())
    assert len(calls) == 20


# ── prompt versions ────────────────────────────────────────────────────────────

def test_fingerprint_is_stable_and_changes_with_the_text():
    a = prompts.fingerprint("You are a verifier.")
    assert a == prompts.fingerprint("You are a verifier.") and len(a) == 10
    assert a != prompts.fingerprint("You are a strict verifier.")
    assert prompts.fingerprint("x\r\ny") == prompts.fingerprint("x\ny")       # line endings do not matter


def test_every_agent_has_a_version_and_labels_map_to_them():
    v = prompts.current_versions()
    assert set(v) == {"orchestrator", "summarizer", "verifier", "translator"}
    assert prompts.version_of("SUMMARIZER") == f"summarizer@{v['summarizer']}"
    assert prompts.version_of("ORCHESTRATOR/fallback") == f"orchestrator@{v['orchestrator']}"
    assert prompts.version_of("SOMETHING_ELSE") is None


def test_editing_a_prompt_function_changes_its_version():
    from src.agents import summarizer
    before = prompts.fingerprint(summarizer._build_prompt)

    def _build_prompt(context, flag_lines, has_trends):        # a different prompt text
        return "Write something else"

    assert before != prompts.fingerprint(_build_prompt)


def test_summarizer_prompt_still_carries_its_inputs():
    from src.agents import summarizer
    p = summarizer._build_prompt("CTX-123", "\nFLAG_0: low Hb", True)
    assert "CTX-123" in p and "FLAG_0: low Hb" in p and "SECTION: LONGITUDINAL_NARRATIVE" in p
    assert "Write exactly: N/A" in summarizer._build_prompt("c", "", False)


# ── end to end ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def traced_summaries(monkeypatch):
    real = main.generate_summaries

    async def summaries(*a, **k):
        await llm.with_retry(lambda: asyncio.sleep(0, resp(1000, 200)), label="SUMMARIZER", model="m-x")
        return await real(*a, **k)

    monkeypatch.setattr(main, "generate_summaries", summaries)


def test_calls_are_persisted_with_versions_tokens_and_notional_cost(client, traced_summaries):
    h = login(client, CLIN)
    assert upload(client, h, report_id="t-1").status_code == 200
    with SessionLocal() as db:
        (c,) = db.query(LlmCall).filter(LlmCall.report_id == "t-1").all()
        assert c.purpose == "summarizer" and c.model == "m-x" and c.status == "ok"
        assert c.prompt_version == prompts.version_of("SUMMARIZER")
        assert (c.input_tokens, c.output_tokens) == (1000, 200)
        assert c.cost_usd == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1e6)
        run_ = db.query(AgentRun).filter(AgentRun.report_id == "t-1").first()
        assert (run_.input_tokens, run_.output_tokens) == (1000, 200)
        assert run_.prompt_versions["verifier"].startswith("verifier@")


def test_prices_are_configurable(client, traced_summaries, monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_M", "1")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_M", "2")
    upload(client, login(client, CLIN), report_id="t-2")
    with SessionLocal() as db:
        assert db.query(LlmCall).first().cost_usd == pytest.approx((1000 * 1 + 200 * 2) / 1e6)


def test_calls_made_by_a_failed_attempt_are_still_recorded(client, monkeypatch):
    async def summaries(*a, **k):
        await llm.with_retry(lambda: asyncio.sleep(0, resp()), label="SUMMARIZER", model="m")
        raise jobq_perm()

    def jobq_perm():
        from src.jobs.queue import PermanentJobError
        return PermanentJobError("stop")

    monkeypatch.setattr(main, "generate_summaries", summaries)
    h = login(client, CLIN)
    upload_raw(client, h, report_id="t-3")
    assert wait_for_job(client, h, "t-3")["status"] == "failed"
    with SessionLocal() as db:
        assert db.query(LlmCall).filter(LlmCall.report_id == "t-3").count() == 1


def test_usage_endpoint_aggregates_by_prompt_version(client, traced_summaries):
    h = login(client, CLIN)
    upload(client, h, report_id="t-4")
    upload(client, h, report_id="t-5")
    body = client.get("/admin/llm-usage?days=1", headers=h).json()
    assert body["calls"] == 2 and body["reports"] == 2 and body["calls_per_report"] == 1.0
    assert body["cost_is_notional"] is True
    (row,) = body["by_prompt"]
    assert row["purpose"] == "summarizer" and row["calls"] == 2 and row["input_tokens"] == 2000
    assert row["prompt_version"] == prompts.version_of("SUMMARIZER")
    assert body["notional_cost_per_report_usd"] == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1e6)
    assert "t-4" not in str(body)                                  # aggregates only


def test_usage_endpoint_needs_a_clinician(client):
    assert client.get("/admin/llm-usage").status_code == 401


def test_patient_erasure_removes_traces_and_agent_runs(client, traced_summaries):
    h = login(client, CLIN)
    upload(client, h, report_id="t-6")
    assert client.delete("/patient/P001", headers=h).status_code == 200
    with SessionLocal() as db:
        assert db.query(LlmCall).filter(LlmCall.report_id == "t-6").count() == 0
        assert db.query(AgentRun).filter(AgentRun.report_id == "t-6").count() == 0
