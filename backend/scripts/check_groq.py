r"""
scripts/check_groq.py - check that this machine can make every kind of Groq call the app uses.

    python scripts\check_groq.py

Needs GROQ_API_KEY in backend\.env (the script loads it itself and never prints it). Makes 7 small
calls: plain text, JSON mode with the verifier's schema, a tool call and its answer, the real
Verifier agent on one summary, and the real Summarizer then Verifier on a synthetic critical report. Prints PASS or FAIL per check, then the model(s) that answered.
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "groq"                      # this script is about Groq whatever the default is

import src.llm as llm  # noqa: E402


def say(text=""):
    """Print without crashing on a console that cannot show a character.

    Models use characters such as the non-breaking hyphen, which the default
    Windows console (cp1252) cannot encode.
    """
    enc = sys.stdout.encoding or "utf-8"
    print(str(text).encode(enc, errors="replace").decode(enc, errors="replace"))



def report(name: str, ok: bool, detail: str = "") -> bool:
    say(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))
    return ok


async def list_models() -> None:
    """Which models this key can use right now (a free listing call, not an inference call)."""
    import httpx
    try:
        async with httpx.AsyncClient(base_url=llm.GROQ_BASE_URL, timeout=30) as c:
            r = await c.get("/models", headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
        ids = sorted(m["id"] for m in r.json().get("data", [])) if r.status_code == 200 else []
    except Exception as e:  # noqa: BLE001
        say(f"models      could not list ({type(e).__name__})")
        return
    chat = [i for i in ids if not any(w in i for w in ("whisper", "orpheus", "guard", "tts"))]
    say(f"models      {', '.join(chat) if chat else 'none listed (HTTP ' + str(r.status_code) + ')'}")
    if llm.MODEL not in ids and ids:
        say(f"WARNING     {llm.MODEL} is not in that list; set GROQ_MODEL to one that is")


async def check_text() -> bool:
    out = await llm.generate_text("Reply with exactly the word: pong", label="CHECK")
    return report("plain text", "pong" in out.lower(), repr(out[:60]))


async def check_json_mode() -> bool:
    from src.agents.verifier import AuditResult, parse_audit
    prompt = ('Audit these two summaries against the data. Data: Hemoglobin 9.2 g/dL. '
              'CLINICAL SUMMARY: Hemoglobin is 9.2 g/dL, low. PATIENT SUMMARY: Your blood count is low. '
              'Return a verdict (PASS or FAIL) for each, with issues and a corrected text only on FAIL.')
    out = await llm.generate_text(prompt, json_schema=AuditResult, label="CHECK")
    try:
        parsed = parse_audit(out)
    except Exception as e:  # noqa: BLE001
        return report("JSON mode matches the verifier schema", False, f"{type(e).__name__}: {str(e)[:120]}")
    return report("JSON mode matches the verifier schema", parsed is not None, "parsed")


async def check_tool_call() -> bool:
    tools = llm.openai_tools([{"name": "lookup_range", "description": "Look up the reference range for a lab test.",
                               "parameters": {"type": "object", "properties": {"test": {"type": "string"}},
                                              "required": ["test"]}}])

    async def open_chat(model):
        chat = llm.GroqChat(model, "You are a lab assistant. Use the tool when asked about ranges.", tools)
        return chat, await chat.send_message("What is the normal range for hemoglobin? Use the tool.")

    (chat, first), _model = await llm.with_fallback(open_chat, label="CHECK", attempts=2, base_wait=10)
    calls = list(first.function_calls)
    if not calls:
        return report("tool call", False, "the model did not request the tool")
    second = await chat.send_message([llm.ToolResult(c.name, {"range": "12-15 g/dL"}, c.id) for c in calls])
    return report("tool call and answer", bool(second.text), repr(second.text[:70]))


async def check_verifier_agent() -> bool:
    from src.agents.verifier import verify_and_correct
    entities = {"lab_values": [{"test": "Hemoglobin", "value": 9.2, "unit": "g/dL"}], "medications": [],
                "patient": {"gender": "Female", "age": "34"}}
    out = await verify_and_correct("Hemoglobin is 9.2 g/dL, which is low.", "Your blood count is a little low.",
                                   entities, [], [])
    return report("verifier agent", not out["clinical_unverified"], f"unverified={out['clinical_unverified']}")


async def check_summarizer() -> bool:
    """The real Summarizer on a synthetic report with two critical values. This is the check that
    was missing: the model can pass every other one and still format its summary in a way the
    reader cannot use."""
    import src.agents.summarizer as summarizer
    from src.tools.tool_registry import check_icmr_flags
    labs = [{"test": "Hemoglobin", "value": 6.2, "unit": "g/dL"},
            {"test": "Glucose (fasting)", "value": 540, "unit": "mg/dL"},
            {"test": "Total WBC", "value": 7200, "unit": "cells/mcL"}]
    entities = {"patient": {"gender": "Male", "age": "54"}, "clinical": {}, "medications": [], "lab_values": labs}
    flags = (await check_icmr_flags(labs, "Male", 54))["flags"]
    raw = {}
    original = summarizer.generate_text

    async def capture(prompt, **kw):
        raw["text"] = await original(prompt, **kw)
        return raw["text"]

    summarizer.generate_text = capture
    try:
        out = await summarizer.generate_summaries(entities, flags, [], [], [], [])
    finally:
        summarizer.generate_text = original
    text = raw.get("text") or ""
    sections = sorted(summarizer.split_sections(text))
    say(f"            sections read: {', '.join(sections) or 'NONE'}")
    say(f"            summarizer clinical: {out['clinical_summary'][:400]!r}")
    say(f"            summarizer patient:  {out['patient_summary'][:300]!r}")
    for i, action in enumerate(out["followup_actions"][:3], 1):
        say(f"            action {i}: {action[:200]!r}")

    # Then the verifier, as in the app: what the clinician sees is the text after its audit and rewrite.
    from src.agents.verifier import verify_and_correct
    ver = await verify_and_correct(out["clinical_summary"], out["patient_summary"], entities, flags, [])
    if ver["clinical_corrected"] or ver["patient_corrected"]:
        say(f"            verifier rewrote it. clinical: {ver['clinical_summary'][:400]!r}")
        say(f"                                 patient:  {ver['patient_summary'][:300]!r}")
    orders = summarizer.directive_actions(out["followup_actions"])
    problems = list(ver["content_issues"])
    if ver["clinical_audit_unverified"]:
        problems.append(f"verifier audit did not run: {ver['clinical_issues']}")
    if orders:
        problems.append(f"{len(orders)} action(s) read as treatment orders")
    ok = not problems
    report("summarizer + verifier (critical report)", ok, "; ".join(problems) or "complete, urgent, advisory")
    if not ok:      # synthetic data, so the raw answer is safe to show and is what explains a failure
        say(f"            raw summarizer answer, {len(text)} chars, first 900:\n{text[:900]!r}")
    return ok


async def main() -> int:
    say(f"provider {llm.PROVIDER}   model {llm.MODEL}   fallback {llm.FALLBACK_MODEL or 'none'}")
    if not os.getenv("GROQ_API_KEY"):
        say("FAIL  GROQ_API_KEY is not set in backend\\.env")
        return 2
    await list_models()
    used = llm.begin_model_tracking()
    ok = True
    for check in (check_text, check_json_mode, check_tool_call, check_verifier_agent, check_summarizer):
        try:
            ok = await check() and ok
        except Exception as e:  # noqa: BLE001
            ok = report(check.__name__, False, f"{type(e).__name__}: {str(e)[:400]}") and ok
    say(f"answered by: {', '.join(sorted(used)) or 'unknown'}")
    say("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
