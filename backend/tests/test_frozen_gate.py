"""The prompt/model evaluation gate: targeted, small, frozen, quota-guarded."""
import asyncio
import json
import re

import pytest

import eval.frozen_gate as gate

STATE = {"model": "m-1", "prompt_versions": {"verifier": "v1", "summarizer": "s1", "orchestrator": "o1", "translator": "t1"}}


def _numbers(text):
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def make_generate(catch=True, garbage=False, calls=None, empty_summaries=False, bare=False, orders=False):
    """A fake model. As verifier: FAILs a summary holding a number the ground truth lacks.
    As summarizer: returns grounded text."""
    async def generate(prompt, **kw):
        if calls is not None:
            calls.append(kw.get("label"))
        if garbage:
            return "not json at all"
        if "GROUND TRUTH DATA" in prompt:
            ground = prompt.split("CLINICAL SUMMARY TO AUDIT:")[0]
            clinical = prompt.split("CLINICAL SUMMARY TO AUDIT:")[1].split("PATIENT SUMMARY TO AUDIT:")[0]
            bad = catch and bool(_numbers(clinical) - _numbers(ground))
            return json.dumps({"clinical": {"verdict": "FAIL" if bad else "PASS",
                                            "issues": ["novel value"] if bad else [],
                                            # a real correction keeps the findings; it only fixes the claim
                                            "corrected": ("Corrected: " + clinical.strip()) if bad else None},
                               "patient": {"verdict": "PASS", "issues": [], "corrected": None}})
        if empty_summaries:
            return "I am unable to format that."
        # a realistic summary: it repeats the abnormal results it was given
        findings = " ".join(line.strip() for line in prompt.splitlines() if line.startswith("  ["))
        if bare:
            # the values only, as Groq wrote them: no "critical", no "contact your doctor"
            findings = " ".join(f"{name} is {value}." for name, value in
                                re.findall(r"^  \[\w+\] ([^:]+): ([\d.]+)", prompt, re.MULTILINE))
        patient = "Here are your values." if bare else "Some results need attention; please contact your doctor."
        action = "Admit the patient and start treatment." if orders else "Consider review."
        return (f"SECTION: CLINICAL_SUMMARY\nValues reviewed. {findings}\nSECTION: PATIENT_SUMMARY\n{patient}\n"
                f"SECTION: FLAG_EXPLANATIONS\nSECTION: FOLLOWUP_ACTIONS\n1. {action}\nSECTION: LONGITUDINAL_NARRATIVE\nN/A")
    return generate


@pytest.fixture()
def paths(tmp_path):
    return {"frozen_path": gate.FROZEN_PATH, "baseline_path": tmp_path / "baseline.json",
            "ledger_path": tmp_path / "ledger.json"}


def run(paths, **kw):
    kw.setdefault("generate", make_generate())
    kw.setdefault("state", STATE)
    return asyncio.run(gate.run_gate(**paths, **kw))


# ── the frozen set ─────────────────────────────────────────────────────────────

def test_frozen_set_is_committed_and_reproducible(tmp_path):
    committed = json.loads(gate.FROZEN_PATH.read_text(encoding="utf-8"))
    assert len(committed["cases"]) == gate.N_CASES
    out = tmp_path / "again.json"
    gate.build_frozen_set(out)
    assert json.loads(out.read_text(encoding="utf-8")) == committed       # nobody edited it by hand
    for c in committed["cases"]:
        assert c["summary_clean"] != c["summary_novel_value"]


# ── planning: only what the change touches ─────────────────────────────────────

def test_no_baseline_runs_everything():
    p = gate.plan(None, STATE)
    assert set(p["run"]) == {"verifier", "summarizer"} and p["planned_calls"] == 2 * gate.N_CASES + gate.N_SUMMARY_CASES


def test_unchanged_prompts_and_model_run_nothing():
    base = {"model": "m-1", "prompt_versions": STATE["prompt_versions"]}
    assert gate.plan(base, STATE) == {"run": {}, "planned_calls": 0}


def test_changing_one_prompt_runs_only_that_component():
    base = {"model": "m-1", "prompt_versions": {**STATE["prompt_versions"], "verifier": "OLD"}}
    p = gate.plan(base, STATE)
    assert list(p["run"]) == ["verifier"] and p["planned_calls"] == 2 * gate.N_CASES
    base = {"model": "m-1", "prompt_versions": {**STATE["prompt_versions"], "summarizer": "OLD"}}
    p = gate.plan(base, STATE)
    assert list(p["run"]) == ["summarizer"] and p["planned_calls"] == gate.N_SUMMARY_CASES


def test_changing_the_model_runs_both():
    base = {"model": "m-OLD", "prompt_versions": STATE["prompt_versions"]}
    assert set(gate.plan(base, STATE)["run"]) == {"verifier", "summarizer"}


# ── running ────────────────────────────────────────────────────────────────────

def test_dry_run_makes_no_calls(paths):
    calls = []
    r = run(paths, generate=make_generate(calls=calls), dry_run=True)
    assert calls == [] and r["status"] == "dry-run" and r["plan"]["planned_calls"] == 20


def test_first_run_records_a_baseline_and_a_second_run_has_nothing_to_do(paths):
    calls = []
    r = run(paths, generate=make_generate(calls=calls), update_baseline=True)
    assert r["status"] == "pass" and r["baseline_updated"] is True and len(calls) == r["calls_used"] == 20
    assert r["metrics"]["verifier.catch_rate"] == 1.0 and r["metrics"]["verifier.false_alarm_rate"] == 0.0
    assert r["metrics"]["summarizer.ungrounded_rate"] == 0.0
    again = run(paths)
    assert again["status"] == "nothing-to-run"


def test_a_worse_verifier_is_reported_as_a_regression_and_does_not_update_the_baseline(paths):
    run(paths, update_baseline=True)
    changed = {**STATE, "prompt_versions": {**STATE["prompt_versions"], "verifier": "v2"}}
    r = run(paths, state=changed, generate=make_generate(catch=False), update_baseline=True)
    assert r["status"] == "regression" and any("catch_rate" in p for p in r["regressions"])
    assert r["baseline_updated"] is False
    saved = json.loads(paths["baseline_path"].read_text())
    assert saved["prompt_versions"]["verifier"] == "v1"                    # the old baseline stays


def test_an_unchanged_result_after_a_prompt_change_updates_the_baseline(paths):
    run(paths, update_baseline=True)
    changed = {**STATE, "prompt_versions": {**STATE["prompt_versions"], "verifier": "v2"}}
    r = run(paths, state=changed, update_baseline=True)
    assert r["status"] == "pass" and r["baseline_updated"] is True and r["calls_used"] == 16
    saved = json.loads(paths["baseline_path"].read_text())
    assert saved["prompt_versions"]["verifier"] == "v2" and "summarizer.ungrounded_rate" in saved["metrics"]


def test_tolerance_is_one_case():
    base = {"verifier.catch_rate": 1.0}
    assert gate.compare(base, {"verifier.catch_rate": 0.875}, 1) == []     # 7/8: one case worse is allowed
    assert gate.compare(base, {"verifier.catch_rate": 0.75}, 1) != []      # 6/8: two cases worse is not
    assert gate.compare({"verifier.false_alarm_rate": 0.0}, {"verifier.false_alarm_rate": 0.25}, 1) != []


# ── quota guard ────────────────────────────────────────────────────────────────

def test_plan_over_the_per_run_cap_is_refused_before_any_call(paths, monkeypatch):
    monkeypatch.setenv("GATE_MAX_CALLS", "5")
    calls = []
    with pytest.raises(gate.QuotaGuard, match="GATE_MAX_CALLS"):
        run(paths, generate=make_generate(calls=calls))
    assert calls == []


def test_daily_ledger_blocks_a_run_that_would_pass_the_limit(paths, monkeypatch):
    monkeypatch.setenv("GATE_DAILY_CALLS", "30")
    run(paths, update_baseline=True)                                       # uses 20
    changed = {**STATE, "model": "m-2"}
    calls = []
    with pytest.raises(gate.QuotaGuard, match="already used today"):
        run(paths, state=changed, generate=make_generate(calls=calls))     # would need 20 more
    assert calls == []


def test_repair_calls_count_and_the_cap_stops_a_run_mid_way(paths, monkeypatch):
    """A garbage-producing model makes the verifier repair every answer (2 calls each), so the
    real count outruns the plan; the guard must stop it rather than burn the quota."""
    monkeypatch.setenv("GATE_MAX_CALLS", "20")
    calls = []
    with pytest.raises(gate.QuotaGuard, match="mid-run"):
        run(paths, generate=make_generate(garbage=True, calls=calls))
    assert len(calls) == 20
    assert gate._ledger_used_today(paths["ledger_path"]) == 20             # the spend was still recorded


def test_cli_refusal_exit_code(monkeypatch, capsys):
    async def refuse(**k):
        raise gate.QuotaGuard("nope")

    monkeypatch.setattr(gate, "run_gate", refuse)
    monkeypatch.setattr("src.core.llm.generate_text", lambda *a, **k: None, raising=False)
    assert gate.main([]) == 2 and "REFUSED" in capsys.readouterr().out


def test_failed_calls_make_the_run_invalid_not_a_pass(paths):
    """The agents swallow provider errors and fail closed, so without this a dead provider looked like a pass."""
    good = make_generate()
    state = {"n": 0}

    async def flaky(prompt, **kw):
        state["n"] += 1
        if state["n"] % 2:
            raise AttributeError("module 'aiohttp' has no attribute 'ClientConnectorDNSError'")
        return await good(prompt, **kw)

    report = run(paths, generate=flaky, update_baseline=True)
    assert report["status"] == "invalid"
    assert report["call_errors"] > 0 and "aiohttp" in report["first_error"]
    assert report["baseline_updated"] is False
    assert not paths["baseline_path"].exists()


def test_cli_exit_code_for_an_invalid_run(monkeypatch, capsys):
    async def fake_run_gate(**kw):
        return {"status": "invalid", "call_errors": 19, "calls_used": 20, "first_error": "AttributeError: x"}
    monkeypatch.setattr(gate, "run_gate", fake_run_gate)
    import src.core.llm
    assert gate.main([]) == 3
    assert "INVALID RUN" in capsys.readouterr().out


def test_a_dead_provider_stops_the_run_after_a_few_failures_and_saves_quota(paths):
    seen = {"n": 0}

    async def down(prompt, **kw):
        seen["n"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    report = run(paths, generate=down, update_baseline=True)
    assert report["status"] == "invalid"
    assert seen["n"] == 3                      # GATE_MAX_CONSECUTIVE_ERRORS, not the full 20
    assert report["calls_used"] == 3
    assert json.loads(paths["ledger_path"].read_text())  # the 3 real attempts were still recorded
    assert not paths["baseline_path"].exists()


def _serving(model, inner=None):
    """A fake that reports which model answered, the way src.core.llm.with_fallback does."""
    inner = inner or make_generate()

    async def generate(prompt, **kw):
        import src.core.llm as llm
        llm.note_model(model)
        return await inner(prompt, **kw)
    return generate


def test_a_run_answered_by_the_fallback_model_is_off_model_and_writes_no_baseline(paths):
    report = run(paths, generate=_serving("fallback-m"), update_baseline=True)
    assert report["status"] == "off-model"
    assert report["served_models"] == {"fallback-m": report["calls_used"]}
    assert report["off_primary_calls"] == report["calls_used"]
    assert report["baseline_updated"] is False
    assert not paths["baseline_path"].exists()


def test_a_run_on_the_primary_model_records_who_answered_and_saves_the_baseline(paths):
    report = run(paths, generate=_serving(STATE["model"]), update_baseline=True)
    assert report["status"] == "pass" and report["off_primary_calls"] == 0
    assert report["served_models"] == {STATE["model"]: report["calls_used"]}
    assert paths["baseline_path"].exists()


def test_a_couple_of_fallback_calls_are_tolerated(paths):
    n = {"i": 0}
    primary, fallback = _serving(STATE["model"]), _serving("fallback-m")

    async def mostly_primary(prompt, **kw):
        n["i"] += 1
        return await (fallback if n["i"] in (3, 9) else primary)(prompt, **kw)

    report = run(paths, generate=mostly_primary, update_baseline=True)
    assert report["status"] == "pass" and report["off_primary_calls"] == 2
    assert paths["baseline_path"].exists()


def test_cli_exit_code_for_an_off_model_run(monkeypatch, capsys):
    async def fake_run_gate(**kw):
        return {"status": "off-model", "off_primary_calls": 20, "calls_used": 20, "allowed_off_primary_calls": 2,
                "served_models": {"b": 20}, "state": {"model": "a"}}
    monkeypatch.setattr(gate, "run_gate", fake_run_gate)
    assert gate.main([]) == 4
    assert "OFF-MODEL RUN" in capsys.readouterr().out


def test_summaries_that_cannot_be_read_fail_the_gate_and_never_become_the_baseline(paths):
    """Before this, an empty summary scored 0.0 ungrounded - a perfect result for producing nothing."""
    report = run(paths, generate=make_generate(empty_summaries=True), update_baseline=True)
    assert report["metrics"]["summarizer.unusable_rate"] == 1.0
    assert report["metrics"]["summarizer.ungrounded_rate"] == 0.0        # the old, blind measure
    assert report["status"] == "regression" and "unusable_rate" in report["regressions"][0]
    assert report["baseline_updated"] is False and not paths["baseline_path"].exists()


def test_complete_summaries_score_zero_unusable(paths):
    report = run(paths)
    assert report["metrics"]["summarizer.unusable_rate"] == 0.0 and report["status"] == "pass"


def test_an_old_baseline_without_the_new_measure_still_gets_the_absolute_check(paths):
    paths["baseline_path"].write_text(json.dumps({
        "model": "OLD", "prompt_versions": STATE["prompt_versions"],
        "metrics": {"verifier.catch_rate": 1.0, "verifier.false_alarm_rate": 0.0,
                    "verifier.unverified_rate": 0.0, "summarizer.ungrounded_rate": 0.0}}), encoding="utf-8")
    report = run(paths, generate=make_generate(empty_summaries=True), force_all=True)
    assert report["status"] == "regression" and "unusable_rate" in report["regressions"][0]


def test_summaries_that_list_critical_values_without_saying_so_fail_the_gate(paths):
    """What the real model wrote on 2026-11-17: every value present, nothing said about urgency."""
    report = run(paths, generate=make_generate(bare=True), update_baseline=True)
    assert report["metrics"]["summarizer.unusable_rate"] == 0.0          # nothing is missing...
    assert report["metrics"]["summarizer.thin_rate"] == 0.5               # ...the two critical cases are silent
    assert report["status"] == "regression" and "thin_rate" in report["regressions"][0]
    assert report["baseline_updated"] is False


def test_actions_written_as_treatment_orders_fail_the_gate(paths):
    report = run(paths, generate=make_generate(orders=True), update_baseline=True)
    assert report["metrics"]["actions.directive_rate"] == 1.0
    assert report["status"] == "regression" and "directive_rate" in report["regressions"][0]
    assert report["baseline_updated"] is False


def test_good_summaries_score_zero_on_every_content_measure(paths):
    m = run(paths)["metrics"]
    assert m["summarizer.thin_rate"] == 0.0 and m["actions.directive_rate"] == 0.0
